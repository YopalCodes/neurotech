"""Devices the controller can show and drive, behind one interface.

Coordinates are normalized: (0, 0) is the top-left of the device screen and
(1, 1) the bottom-right, so callers never need to know the real resolution.

    manager = DeviceManager()
    manager.refresh_android()
    device = manager.select("Pixel 7", password=None)
    device.pointer_down(0.5, 0.5); device.pointer_up(0.5, 0.5)   # tap the center
    device.type_text("hello")
    device.open_app("com.android.chrome")

A later camera-based selector only needs DeviceManager.select().
"""

import glob
import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image

import protocol

DEVICES_FILE = Path(__file__).parent / "devices.json"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class Device:
    kind = "device"
    actions = ()  # extra toolbar buttons, e.g. ("back", "home")

    def __init__(self, name):
        self.name = name
        self.frame = None      # latest PIL image
        self.frame_id = 0      # increments with each new frame
        self.error = None
        self.connected = False

    def _set_frame(self, img):
        self.frame = img
        self.frame_id += 1

    # Subclasses implement these.
    def connect(self, password=None): ...
    def close(self): ...
    def pointer_move(self, x, y): ...
    def pointer_down(self, x, y, button="left"): ...
    def pointer_up(self, x, y, button="left"): ...
    def scroll(self, dy, x=0.5, y=0.5): ...
    def key_down(self, key): ...
    def key_up(self, key): ...
    def type_text(self, text): ...
    def list_apps(self): return []
    def open_app(self, app): ...
    def action(self, name): ...


# ---------------------------------------------------------------- computers --

class ComputerDevice(Device):
    """A computer running agent.py."""
    kind = "computer"

    def __init__(self, name, host, port=protocol.DEFAULT_PORT):
        super().__init__(name)
        self.host, self.port = host, port
        self.channel = None
        self.info = {}

    def connect(self, password=None):
        sock = socket.create_connection((self.host, self.port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.channel, self.info = protocol.controller_handshake(sock, password or "")
        except Exception:
            sock.close()
            raise
        sock.settimeout(None)
        self.connected = True
        threading.Thread(target=self._receive, daemon=True).start()

    def _receive(self):
        try:
            while self.connected:
                kind, payload = self.channel.recv()
                if kind == protocol.FRAME:
                    img = Image.open(io.BytesIO(payload))
                    img.load()
                    self._set_frame(img)
        except Exception as e:  # noqa: BLE001 - any failure means the link is gone
            if self.connected:
                self.error = f"disconnected: {e}"
        self.connected = False

    def close(self):
        self.connected = False
        if self.channel:
            self.channel.close()
            self.channel = None

    def _send(self, **cmd):
        if self.connected:
            try:
                self.channel.send_json(cmd)
            except OSError as e:
                self.error = f"send failed: {e}"
                self.connected = False

    def pointer_move(self, x, y):
        self._send(type="move", x=x, y=y)

    def pointer_down(self, x, y, button="left"):
        self._send(type="down", x=x, y=y, button=button)

    def pointer_up(self, x, y, button="left"):
        self._send(type="up", x=x, y=y, button=button)

    def scroll(self, dy, x=0.5, y=0.5):
        self._send(type="scroll", dy=dy)

    def key_down(self, key):
        self._send(type="key_down", key=key)

    def key_up(self, key):
        self._send(type="key_up", key=key)

    def type_text(self, text):
        self._send(type="text", text=text)

    def open_app(self, app):
        """Open an app by name through the OS launcher (Start menu / Spotlight)."""
        def run():
            if self.info.get("os") == "Darwin":
                self.key_down("cmd"); self.key_down("space"); self.key_up("space"); self.key_up("cmd")
            else:
                self.key_down("cmd"); self.key_up("cmd")
            time.sleep(0.8)
            self.type_text(app)
            time.sleep(0.8)
            self.key_down("enter"); self.key_up("enter")
        threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------ android --

ANDROID_KEYS = {
    "enter": 66, "backspace": 67, "delete": 112, "tab": 61, "esc": 111, "space": 62,
    "up": 19, "down": 20, "left": 21, "right": 22, "home": 122, "end": 123,
    "page_up": 92, "page_down": 93,
}
ANDROID_ACTIONS = {"back": 4, "home": 3, "recents": 187, "power": 26, "volume_up": 24, "volume_down": 25}


def find_adb():
    path = shutil.which("adb")
    if path:
        return path
    pattern = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Google.PlatformTools*\platform-tools\adb.exe")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def adb(*args, serial=None, timeout=10):
    exe = find_adb()
    if exe is None:
        raise RuntimeError("adb not found. Install Android platform-tools.")
    cmd = [exe] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW).stdout


def shell_quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


class AndroidDevice(Device):
    """An Android phone/tablet over ADB (USB or Wi-Fi). Needs USB debugging enabled."""
    kind = "android"
    actions = ("back", "home", "recents", "power", "volume_up", "volume_down")

    def __init__(self, name, serial, use_screencap=False):
        super().__init__(name)
        self.serial = serial
        self.use_screencap = use_screencap
        self.size = (1080, 1920)   # physical portrait size, read on connect
        self.shell = None
        self.stream = None
        self.lock = threading.Lock()
        self.text_buffer = ""
        self.ctrl = False
        self.down = None           # (x, y, time) of the current press

    # -- connection --

    def connect(self, password=None):
        out = adb("shell", "wm", "size", serial=self.serial)
        sizes = dict(line.split(": ") for line in out.strip().splitlines() if ": " in line)
        size = sizes.get("Override size") or sizes.get("Physical size")
        if not size:
            raise RuntimeError(f"Can't read screen size from {self.serial}. Is USB debugging authorized?")
        w, h = map(int, size.strip().split("x"))
        self.size = (min(w, h), max(w, h))
        self.shell = subprocess.Popen([find_adb(), "-s", self.serial, "shell"], stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
                                      creationflags=NO_WINDOW)
        self.connected = True
        threading.Thread(target=self._capture, daemon=True).start()
        threading.Thread(target=self._flush_text_loop, daemon=True).start()

    def close(self):
        self.connected = False
        for proc in (self.stream, self.shell):
            if proc and proc.poll() is None:
                proc.kill()

    # -- screen --

    def _capture(self):
        while self.connected:
            got_frames = False if self.use_screencap else self._capture_h264()
            if not got_frames and self.connected:
                self.use_screencap = True  # screenrecord unsupported: fall back to stills
                self._capture_screencap()

    def _capture_h264(self):
        """Stream H.264 from screenrecord (smooth, ~30 fps). Restarts when it hits its time limit."""
        import av

        self.stream = subprocess.Popen(
            [find_adb(), "-s", self.serial, "exec-out", "screenrecord", "--output-format=h264",
             "--bit-rate=6000000", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        codec = av.CodecContext.create("h264", "r")
        got = False
        try:
            while self.connected:
                chunk = self.stream.stdout.read1(65536)
                if not chunk:
                    break
                for packet in codec.parse(chunk):
                    for frame in codec.decode(packet):
                        self._set_frame(frame.to_image())
                        got = True
        except Exception as e:  # noqa: BLE001
            self.error = f"video stream error: {e}"
        finally:
            if self.stream.poll() is None:
                self.stream.kill()
        return got

    def _capture_screencap(self):
        exe = find_adb()
        while self.connected:
            png = subprocess.run([exe, "-s", self.serial, "exec-out", "screencap", "-p"],
                                 capture_output=True, timeout=10, creationflags=NO_WINDOW).stdout
            try:
                self._set_frame(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception:  # noqa: BLE001
                time.sleep(0.5)

    # -- input --

    def _run(self, command):
        if self.shell and self.shell.poll() is None:
            with self.lock:
                self.shell.stdin.write(command + "\n")
                self.shell.stdin.flush()
        else:
            self.error = "adb shell closed"
            self.connected = False

    def _pixels(self, x, y):
        w, h = self.size
        if self.frame is not None and self.frame.width > self.frame.height:
            w, h = h, w  # landscape
        return round(x * (w - 1)), round(y * (h - 1))

    def pointer_down(self, x, y, button="left"):
        if button == "left":
            self.down = (x, y, time.monotonic())

    def pointer_up(self, x, y, button="left"):
        self._flush_text()
        if button == "right":
            return self.action("back")
        if button == "middle":
            return self.action("home")
        if self.down is None:
            return
        x0, y0, t0 = self.down
        self.down = None
        held_ms = int((time.monotonic() - t0) * 1000)
        (px0, py0), (px1, py1) = self._pixels(x0, y0), self._pixels(x, y)
        moved = abs(x - x0) > 0.02 or abs(y - y0) > 0.02
        if not moved and held_ms < 500:
            self._run(f"input tap {px0} {py0}")
        else:  # drag or long-press, replayed with the same duration
            self._run(f"input swipe {px0} {py0} {px1} {py1} {max(held_ms, 100)}")

    def scroll(self, dy, x=0.5, y=0.5):
        self._flush_text()
        px, py = self._pixels(x, y)
        _, h = self._pixels(1, 1)
        dist = int(0.25 * h) * (1 if dy > 0 else -1)  # wheel up = content moves down
        self._run(f"input swipe {px} {py} {px} {max(0, min(h, py + dist))} 150")

    def key_down(self, key):
        if key in ("ctrl", "ctrl_r"):
            self.ctrl = True
        elif len(key) == 1 and self.ctrl and key.isalpha():
            self._flush_text()  # Ctrl+letter (Android 13+): CTRL_LEFT + KEYCODE_A..Z
            self._run(f"input keycombination 113 {29 + ord(key.lower()) - ord('a')}")
        elif len(key) == 1:
            self.type_text(key)
        elif key in ANDROID_KEYS:
            self._flush_text()
            self._run(f"input keyevent {ANDROID_KEYS[key]}")

    def key_up(self, key):
        if key in ("ctrl", "ctrl_r"):
            self.ctrl = False

    def type_text(self, text):
        # `input text` starts a JVM each call, so batch keystrokes typed close together.
        with self.lock:
            self.text_buffer += text

    def _flush_text(self):
        with self.lock:
            text, self.text_buffer = self.text_buffer, ""
        if text:
            # `input text` treats %s as a space; spaces themselves must be encoded.
            self._run("input text " + shell_quote(text.replace("%", "\\%").replace(" ", "%s")))

    def _flush_text_loop(self):
        while self.connected:
            time.sleep(0.15)
            self._flush_text()

    def action(self, name):
        if name in ANDROID_ACTIONS:
            self._flush_text()
            self._run(f"input keyevent {ANDROID_ACTIONS[name]}")

    def list_apps(self):
        out = adb("shell", "cmd", "package", "query-activities", "--brief", "-a", "android.intent.action.MAIN",
                  "-c", "android.intent.category.LAUNCHER", serial=self.serial)
        apps = sorted({line.strip().split("/")[0] for line in out.splitlines() if "/" in line})
        if not apps:
            out = adb("shell", "pm", "list", "packages", serial=self.serial)
            apps = sorted(line[len("package:"):].strip() for line in out.splitlines() if line.startswith("package:"))
        return apps

    def open_app(self, app):
        self._run(f"monkey -p {shell_quote(app)} -c android.intent.category.LAUNCHER 1")


# ------------------------------------------------------------------ manager --

class DeviceManager:
    def __init__(self):
        self.devices = {}
        self.current = None
        self._load()

    def _load(self):
        if DEVICES_FILE.exists():
            for d in json.loads(DEVICES_FILE.read_text()):
                self.devices[d["name"]] = ComputerDevice(d["name"], d["host"], d.get("port", protocol.DEFAULT_PORT))

    def _save(self):
        computers = [{"name": d.name, "host": d.host, "port": d.port}
                     for d in self.devices.values() if isinstance(d, ComputerDevice)]
        DEVICES_FILE.write_text(json.dumps(computers, indent=2))

    def refresh_android(self):
        """Add phones listed by `adb devices`; returns their names."""
        found = []
        for line in adb("devices", "-l").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                serial = parts[0]
                model = next((p.split(":", 1)[1] for p in parts if p.startswith("model:")), serial)
                name = f"{model.replace('_', ' ')} ({serial})"
                if name not in self.devices:
                    self.devices[name] = AndroidDevice(name, serial)
                found.append(name)
        return found

    def connect_android_wifi(self, address):
        """`adb connect host:port` for a phone with wireless debugging on."""
        return adb("connect", address, timeout=15).strip()

    def add_computer(self, name, host, port=protocol.DEFAULT_PORT):
        self.devices[name] = ComputerDevice(name, host, port)
        self._save()

    def remove(self, name):
        device = self.devices.pop(name, None)
        if device is self.current:
            self.deselect()
        self._save()

    def select(self, name, password=None):
        """Connect to a device and make it the one being controlled."""
        self.deselect()
        device = self.devices[name]
        device.error = None
        device.connect(password)
        self.current = device
        return device

    def deselect(self):
        if self.current:
            self.current.close()
            self.current = None
