"""Sends commands to the Pi over Wi-Fi (TCP, one JSON object per line). Reconnects on its own."""

import json
import logging
import socket
import threading
import time

log = logging.getLogger("pi_link")


class PiLink:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._sock = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()  # keeps messages from different threads from interleaving
        self._running = True
        threading.Thread(target=self._connect_loop, name="pi_link", daemon=True).start()

    @property
    def connected(self):
        return self._sock is not None

    def send(self, command):
        """Sends a command such as {"cmd": "left"}. Returns False if the Pi isn't connected."""
        sock = self._sock
        if sock is None:
            return False
        try:
            with self._send_lock:
                sock.sendall((json.dumps(command) + "\n").encode())
            return True
        except OSError as e:
            log.warning("Lost connection to the Pi: %s", e)
            self._drop(sock)
            return False

    def close(self):
        self._running = False
        sock = self._sock
        if sock is not None:
            self._drop(sock)

    def _drop(self, sock):
        with self._lock:
            if self._sock is sock:
                self._sock = None
        sock.close()

    def _connect_loop(self):
        warned = False
        while self._running:
            if self._sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=3)
                except OSError as e:
                    if not warned:
                        log.warning("Can't reach the Pi's command server at %s:%d (%s). "
                                    "Retrying in the background.", self.host, self.port, e)
                        warned = True
                else:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.settimeout(0.5)  # never let a stuck send stall the AI loop for long
                    with self._lock:
                        if not self._running:
                            sock.close()
                            return
                        self._sock = sock
                    warned = False
                    log.info("Connected to the Pi at %s:%d", self.host, self.port)
            time.sleep(1)
