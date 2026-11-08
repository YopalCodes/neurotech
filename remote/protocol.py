"""Encrypted message channel between controller.py and agent.py.

Handshake (plaintext): the agent sends MAGIC + a 16-byte nonce, the controller
replies with its own nonce. Both derive an AES-256-GCM key from the shared
password and the two nonces (PBKDF2, so guessing passwords offline is slow).
The controller's first encrypted message proves it knows the password; a
wrong password fails decryption and the agent drops the connection.

Every message after that is: 4-byte length + AES-GCM(kind byte + payload).
Nonces are a direction byte + a counter, so nothing is ever reused or replayed.
"""

import hashlib
import json
import os
import socket
import struct
import threading

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEFAULT_PORT = 7801
MAGIC = b"NTREMOTE1"
MAX_MESSAGE = 32 * 1024 * 1024
KDF_ITERATIONS = 200_000

FRAME = 1  # agent -> controller: JPEG bytes
JSON = 2   # either direction: UTF-8 JSON


class AuthError(Exception):
    pass


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def derive_key(password, agent_nonce, controller_nonce):
    salt = MAGIC + agent_nonce + controller_nonce
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, KDF_ITERATIONS, 32)


class Channel:
    def __init__(self, sock, key, is_agent):
        self.sock = sock
        self.aead = AESGCM(key)
        self.send_dir, self.recv_dir = (b"A", b"C") if is_agent else (b"C", b"A")
        self.send_count = self.recv_count = 0
        self.send_lock = threading.Lock()

    def send(self, kind, payload):
        with self.send_lock:
            nonce = self.send_dir + self.send_count.to_bytes(11, "big")
            self.send_count += 1
            ct = self.aead.encrypt(nonce, bytes([kind]) + payload, None)
            self.sock.sendall(struct.pack(">I", len(ct)) + ct)

    def send_json(self, obj):
        self.send(JSON, json.dumps(obj).encode())

    def recv(self):
        (n,) = struct.unpack(">I", recv_exact(self.sock, 4))
        if n > MAX_MESSAGE:
            raise ConnectionError("message too large")
        ct = recv_exact(self.sock, n)
        nonce = self.recv_dir + self.recv_count.to_bytes(11, "big")
        self.recv_count += 1
        pt = self.aead.decrypt(nonce, ct, None)  # raises InvalidTag if tampered
        return pt[0], pt[1:]

    def recv_json(self):
        kind, payload = self.recv()
        if kind != JSON:
            raise ConnectionError("expected JSON message")
        return json.loads(payload)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def agent_handshake(sock, password, info):
    """Run on the agent. Returns a Channel, or raises AuthError."""
    agent_nonce = os.urandom(16)
    sock.sendall(MAGIC + agent_nonce)
    controller_nonce = recv_exact(sock, 16)
    channel = Channel(sock, derive_key(password, agent_nonce, controller_nonce), is_agent=True)
    try:
        hello = channel.recv_json()
    except InvalidTag:
        raise AuthError("wrong password") from None
    if hello.get("type") != "hello":
        raise AuthError("bad hello")
    channel.send_json({"type": "welcome", **info})
    return channel


def controller_handshake(sock, password):
    """Run on the controller. Returns (Channel, agent info dict)."""
    header = recv_exact(sock, len(MAGIC) + 16)
    if not header.startswith(MAGIC):
        raise AuthError("not a neurotech remote agent")
    controller_nonce = os.urandom(16)
    sock.sendall(controller_nonce)
    channel = Channel(sock, derive_key(password, header[len(MAGIC):], controller_nonce), is_agent=False)
    channel.send_json({"type": "hello"})
    try:
        welcome = channel.recv_json()
    except (InvalidTag, ConnectionError):
        raise AuthError("wrong password") from None
    return channel, welcome
