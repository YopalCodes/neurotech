"""Run this on a computer you want to control from controller.py.

    python agent.py --password <shared password>
    python agent.py --password <pw> --monitor 2 --fps 20

It streams the screen and applies mouse/keyboard input from one controller at
a time. Connections are encrypted and require the password. The console shows
when someone is connected; press Ctrl+C to stop sharing.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import socket
import sys
import threading
import time

import mss
from PIL import Image
from pynput import keyboard, mouse

import protocol

BUTTONS = {"left": mouse.Button.left, "right": mouse.Button.right, "middle": mouse.Button.middle}


def make_dpi_aware():
    """On Windows with display scaling, make capture and cursor use the same (physical) pixels."""
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def to_key(name):
    """Common key name ('enter', 'ctrl', 'a', ...) -> pynput key."""
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return getattr(keyboard.Key, name, None)


class Session:
    def __init__(self, channel, monitor, fps, quality, max_width):
        self.channel = channel
        self.monitor = monitor
        self.fps = fps
        self.quality = quality
        self.max_width = max_width
        self.mouse = mouse.Controller()
        self.keyboard = keyboard.Controller()
        self.pressed_keys = set()
        self.pressed_buttons = set()
        self.running = True

    def stream_screen(self):
        last_digest = None
        last_sent = 0.0
        with mss.MSS() as sct:
            while self.running:
                start = time.monotonic()
                shot = sct.grab(self.monitor)
                digest = hashlib.blake2b(shot.raw, digest_size=16).digest()
                # Skip unchanged frames, but resend once a second so a new view gets a picture.
                if digest != last_digest or start - last_sent > 1.0:
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    if img.width > self.max_width:
                        img = img.resize((self.max_width, round(img.height * self.max_width / img.width)),
                                         Image.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=self.quality)
                    try:
                        self.channel.send(protocol.FRAME, buf.getvalue())
                    except OSError:
                        self.running = False
                        return
                    last_digest, last_sent = digest, start
                time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - start)))

    def to_screen(self, x, y):
        m = self.monitor
        return (m["left"] + round(x * (m["width"] - 1)), m["top"] + round(y * (m["height"] - 1)))

    def handle(self, cmd):
        t = cmd.get("type")
        if t == "move":
            self.mouse.position = self.to_screen(cmd["x"], cmd["y"])
        elif t in ("down", "up"):
            if "x" in cmd:
                self.mouse.position = self.to_screen(cmd["x"], cmd["y"])
            button = BUTTONS.get(cmd.get("button"))
            if button is None:
                return
            if t == "down":
                self.mouse.press(button)
                self.pressed_buttons.add(button)
            else:
                self.mouse.release(button)
                self.pressed_buttons.discard(button)
        elif t == "scroll":
            self.mouse.scroll(cmd.get("dx", 0), cmd.get("dy", 0))
        elif t in ("key_down", "key_up"):
            key = to_key(cmd.get("key", ""))
            if key is None:
                return
            if t == "key_down":
                self.keyboard.press(key)
                self.pressed_keys.add(key)
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
        elif t == "text":
            self.keyboard.type(cmd.get("text", ""))

    def release_all(self):
        """Never leave a key or button stuck down when the controller goes away."""
        for key in list(self.pressed_keys):
            self.keyboard.release(key)
        for button in list(self.pressed_buttons):
            self.mouse.release(button)
        self.pressed_keys.clear()
        self.pressed_buttons.clear()

    def run(self):
        threading.Thread(target=self.stream_screen, daemon=True).start()
        try:
            while self.running:
                kind, payload = self.channel.recv()
                if kind == protocol.JSON:
                    self.handle(json.loads(payload))
        except (ConnectionError, OSError):
            pass
        except Exception as e:  # noqa: BLE001 - tampered or malformed input ends this session only
            print(f"{time.strftime('%H:%M:%S')}  Dropping connection: {type(e).__name__} {e}")
        finally:
            self.running = False
            self.release_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--password", default=os.environ.get("NEUROTECH_REMOTE_PASSWORD"),
                        help="shared password (or set NEUROTECH_REMOTE_PASSWORD)")
    parser.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    parser.add_argument("--bind", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--monitor", type=int, default=1, help="1 = primary monitor")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--quality", type=int, default=60, help="JPEG quality 1-95")
    parser.add_argument("--max-width", type=int, default=1600, help="downscale wider screens to this")
    args = parser.parse_args()
    if not args.password or len(args.password) < 8:
        parser.error("a --password of at least 8 characters is required")

    make_dpi_aware()
    with mss.MSS() as sct:
        if not 1 <= args.monitor < len(sct.monitors):
            parser.error(f"--monitor must be 1..{len(sct.monitors) - 1}")
        monitor = dict(sct.monitors[args.monitor])
    info = {"name": platform.node(), "os": platform.system(), "width": monitor["width"], "height": monitor["height"]}

    server = socket.create_server((args.bind, args.port))
    print(f"Sharing monitor {args.monitor} ({monitor['width']}x{monitor['height']}) on port {args.port}.")
    print("Addresses for the controller:", ", ".join(local_addresses()) or "(none found)")
    print("Ctrl+C to stop.\n")
    try:
        while True:
            sock, addr = server.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(10)
            try:
                channel = protocol.agent_handshake(sock, args.password, info)
            except (protocol.AuthError, ConnectionError, OSError) as e:
                print(f"{time.strftime('%H:%M:%S')}  Rejected {addr[0]}: {e}")
                sock.close()
                continue
            sock.settimeout(None)
            print(f"{time.strftime('%H:%M:%S')}  >>> {addr[0]} is now controlling this computer <<<")
            Session(channel, monitor, args.fps, args.quality, args.max_width).run()
            channel.close()
            print(f"{time.strftime('%H:%M:%S')}  {addr[0]} disconnected")
    except KeyboardInterrupt:
        print("Stopped sharing.")
    finally:
        server.close()


def local_addresses():
    try:
        return sorted({ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                       if not ai[4][0].startswith("127.")})
    except OSError:
        return []


if __name__ == "__main__":
    main()
