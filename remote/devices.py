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
import hashlib
import io
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.request
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


# scrcpy's on-device server (Apache-2.0, github.com/Genymobile/scrcpy) streams the
# screen as H.264 and injects input instantly. It works on phones whose Android
# build lacks `screenrecord`. Downloaded once and checked against the hash that
# scrcpy's own install_release.sh pins for this version.
SCRCPY_VERSION = "3.1"
SCRCPY_URL = f"https://github.com/Genymobile/scrcpy/releases/download/v{SCRCPY_VERSION}/scrcpy-server-v{SCRCPY_VERSION}"
SCRCPY_SHA256 = "958f0944a62f23b1f33a16e9eb14844c1a04b882ca175a738c16d23cb22b86c0"
SCRCPY_REMOTE = "/data/local/tmp/neurotech-scrcpy-server.jar"
VENDOR_DIR = Path(__file__).parent / "vendor"

# scrcpy control messages (server/.../control/ControlMessageReader.java, v3.1)
MSG_KEYCODE, MSG_TEXT, MSG_TOUCH = 0, 1, 2
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
FINGER = -2              # POINTER_ID_GENERIC_FINGER
META_CTRL = 0x1000 | 0x2000  # META_CTRL_ON | META_CTRL_LEFT_ON
KEYCODE_A = 29


def scrcpy_server():
    path = VENDOR_DIR / f"scrcpy-server-v{SCRCPY_VERSION}"
    if not path.exists():
        data = urllib.request.urlopen(SCRCPY_URL, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != SCRCPY_SHA256:
            raise RuntimeError("scrcpy-server download failed its checksum; not using it")
        VENDOR_DIR.mkdir(exist_ok=True)
        path.write_bytes(data)
    return path


def to_image(frame):
    """Decoded video frame -> PIL image. Some phone encoders tag frames with a
    colorspace FFmpeg's converter rejects; treat those as standard HD (BT.709)."""
    try:
        return frame.to_image()
    except OSError:
        return frame.reformat(format="rgb24", src_colorspace="ITU709", dst_colorspace="ITU709").to_image()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AndroidDevice(Device):
    """An Android phone/tablet over ADB (USB or Wi-Fi). Needs USB debugging enabled.

    Normally runs scrcpy's server on the phone for video and input. If that
    fails, falls back to `screenrecord` or `screencap` for video and
    `adb shell input` for control (slower, and drags replay on release).
    """
    kind = "android"
    actions = ("back", "home", "recents", "power", "volume_up", "volume_down")

    def __init__(self, name, serial):
        super().__init__(name)
        self.serial = serial
        # Wi-Fi links (serial "ip:port") get a lighter stream than USB.
        wireless = ":" in serial
        self.max_size = 1024 if wireless else 1600
        self.bit_rate = 2_000_000 if wireless else 8_000_000
        self.mode = None           # "scrcpy", "screenrecord" or "screencap"
        self.size = (1080, 1920)   # physical portrait size, read on connect
        self.shell = None
        self.server = None         # scrcpy server process (adb shell)
        self.stream = None         # screenrecord process
        self.control = None        # scrcpy control socket
        self.forward_port = None
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.text_buffer = ""
        self.ctrl = False
        self.down = None           # (x, y, time) of the current press
        self.ready = threading.Event()

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
        self.ready.clear()
        threading.Thread(target=self._capture, daemon=True).start()
        threading.Thread(target=self._flush_text_loop, daemon=True).start()
        self.ready.wait(10)  # until the video mode is decided, so input goes the right way

    def close(self):
        self.connected = False
        for s in (self.control,):
            if s:
                s.close()
        self.control = None
        for proc in (self.server, self.stream, self.shell):
            if proc and proc.poll() is None:
                proc.kill()
        if self.forward_port:
            try:
                adb("forward", "--remove", f"tcp:{self.forward_port}", serial=self.serial)
            except Exception:  # noqa: BLE001
                pass
            self.forward_port = None

    # -- screen --

    def _capture(self):
        try:
            self._capture_scrcpy()
        except Exception as e:  # noqa: BLE001
            self.error = f"scrcpy unavailable ({e}); using slower fallback"
        self.control = None
        while self.connected:
            self.mode = "screenrecord"
            self.ready.set()
            if not self._capture_h264() and self.connected:
                self.mode = "screencap"
                self._capture_screencap()

    def _capture_scrcpy(self):
        adb("push", str(scrcpy_server()), SCRCPY_REMOTE, serial=self.serial, timeout=30)
        scid = f"{random.randrange(1 << 31):08x}"
        self.forward_port = free_port()
        adb("forward", f"tcp:{self.forward_port}", f"localabstract:scrcpy_{scid}", serial=self.serial)
        self.server = subprocess.Popen(
            [find_adb(), "-s", self.serial, "shell",
             f"CLASSPATH={SCRCPY_REMOTE} app_process / com.genymobile.scrcpy.Server {SCRCPY_VERSION} "
             f"scid={scid} log_level=warn tunnel_forward=true audio=false control=true cleanup=true "
             f"raw_stream=true clipboard_autosync=false max_size={self.max_size} max_fps=30 "
             f"video_bit_rate={self.bit_rate}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)

        # The server accepts the video socket, then the control socket, then streams.
        # Until it's listening, adb accepts and immediately closes our connection.
        deadline = time.monotonic() + 10
        video = first = None
        while time.monotonic() < deadline and self.connected:
            video = control = None
            try:
                video = socket.create_connection(("127.0.0.1", self.forward_port))
                control = socket.create_connection(("127.0.0.1", self.forward_port))
                video.settimeout(3)
                first = video.recv(65536)
            except OSError:  # not listening yet: refused, reset or timed out
                first = b""
            if first:
                control.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.control = control
                break
            for s in (video, control):
                if s:
                    s.close()
            time.sleep(0.3)
        if not first:
            raise RuntimeError("server didn't start")

        import av

        self.mode = "scrcpy"
        self.ready.set()
        threading.Thread(target=self._drain_control, args=(self.control,), daemon=True).start()
        codec = av.CodecContext.create("h264", "r")
        video.settimeout(None)
        chunk = first
        try:
            while self.connected and chunk:
                for packet in codec.parse(chunk):
                    for frame in codec.decode(packet):
                        self._set_frame(to_image(frame))
                chunk = video.recv(65536)
        finally:
            video.close()
        if self.connected:
            raise RuntimeError("video stream ended")

    def _drain_control(self, sock):
        """The server can send device messages; read and ignore them so its buffer never fills."""
        try:
            while sock.recv(4096):
                pass
        except OSError:
            pass

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
                        self._set_frame(to_image(frame))
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

    # -- scrcpy input --

    def _send_control(self, data):
        try:
            with self.control_lock:
                self.control.sendall(data)
        except (OSError, AttributeError):
            self.error = "control connection lost"

    def _touch(self, action, x, y):
        if self.frame is None:
            return
        w, h = self.frame.size  # the server only accepts positions in the current video size
        px, py = round(x * (w - 1)), round(y * (h - 1))
        pressure = 0 if action == ACTION_UP else 0xFFFF
        self._send_control(struct.pack(">BBqiiHHHii", MSG_TOUCH, action, FINGER, px, py, w, h, pressure, 0, 0))

    def _keycode(self, action, keycode, meta=0):
        self._send_control(struct.pack(">BBiii", MSG_KEYCODE, action, keycode, 0, meta))

    def _press(self, keycode, meta=0):
        self._keycode(ACTION_DOWN, keycode, meta)
        self._keycode(ACTION_UP, keycode, meta)

    def _inject_text(self, text):
        data = text.encode()
        for i in range(0, len(data), 300):  # server limit per message
            part = data[i:i + 300]
            self._send_control(struct.pack(">BI", MSG_TEXT, len(part)) + part)

    # -- adb fallback input --

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

    # -- input API --

    def pointer_move(self, x, y):
        if self.control and self.down:
            self._touch(ACTION_MOVE, x, y)

    def pointer_down(self, x, y, button="left"):
        if button != "left":
            return
        self.down = (x, y, time.monotonic())
        if self.control:
            self._touch(ACTION_DOWN, x, y)

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
        if self.control:
            self._touch(ACTION_UP, x, y)
            return
        held_ms = int((time.monotonic() - t0) * 1000)
        (px0, py0), (px1, py1) = self._pixels(x0, y0), self._pixels(x, y)
        moved = abs(x - x0) > 0.02 or abs(y - y0) > 0.02
        if not moved and held_ms < 500:
            self._run(f"input tap {px0} {py0}")
        else:  # drag or long-press, replayed with the same duration
            self._run(f"input swipe {px0} {py0} {px1} {py1} {max(held_ms, 100)}")

    def scroll(self, dy, x=0.5, y=0.5):
        self._flush_text()
        if self.control:
            def drag():  # a short finger drag; wheel up = content moves down
                y1 = max(0.0, min(1.0, y + (0.25 if dy > 0 else -0.25)))
                self._touch(ACTION_DOWN, x, y)
                for i in range(1, 9):
                    self._touch(ACTION_MOVE, x, y + (y1 - y) * i / 8)
                    time.sleep(0.015)
                time.sleep(0.05)  # pause before lifting so it scrolls without flinging far
                self._touch(ACTION_UP, x, y1)
            threading.Thread(target=drag, daemon=True).start()
            return
        px, py = self._pixels(x, y)
        _, h = self._pixels(1, 1)
        dist = int(0.25 * h) * (1 if dy > 0 else -1)
        self._run(f"input swipe {px} {py} {px} {max(0, min(h, py + dist))} 150")

    def key_down(self, key):
        if key in ("ctrl", "ctrl_r"):
            self.ctrl = True
        elif len(key) == 1 and self.ctrl and key.isalpha():
            code = KEYCODE_A + ord(key.lower()) - ord("a")
            if self.control:
                self._press(code, META_CTRL)
            else:  # Android 13+ only
                self._flush_text()
                self._run(f"input keycombination 113 {code}")
        elif len(key) == 1:
            self.type_text(key)
        elif key in ANDROID_KEYS:
            if self.control:
                self._keycode(ACTION_DOWN, ANDROID_KEYS[key])
            else:
                self._flush_text()
                self._run(f"input keyevent {ANDROID_KEYS[key]}")

    def key_up(self, key):
        if key in ("ctrl", "ctrl_r"):
            self.ctrl = False
        elif self.control and key in ANDROID_KEYS:
            self._keycode(ACTION_UP, ANDROID_KEYS[key])

    def type_text(self, text):
        if self.control and text.isascii():
            self._inject_text(text)
            return
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
            if self.control:
                self._press(ANDROID_ACTIONS[name])
            else:
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

    def pair_android_wifi(self, pair_address, code, connect_address):
        """Android 11+ Wireless debugging: pair once with the 6-digit code, then connect."""
        out = adb("pair", pair_address, code, timeout=30).strip()
        if "Successfully paired" not in out:
            raise RuntimeError(out or "pairing failed")
        return self.connect_android_wifi(connect_address)

    def android_to_wifi(self, name, port=5555):
        """Switch a USB-connected phone to ADB over Wi-Fi so it can be unplugged.

        Lasts until the phone reboots. The phone still only accepts computers
        it has already authorized.
        """
        serial = self.devices[name].serial
        out = adb("shell", "ip", "-f", "inet", "addr", "show", "wlan0", serial=serial)
        ip = next((w.split("/")[0] for line in out.splitlines() for i, w in enumerate(line.split())
                   if line.split()[0:1] == ["inet"] and i == 1), None)
        if not ip:
            raise RuntimeError("The phone isn't on Wi-Fi.")
        adb("tcpip", str(port), serial=serial)
        time.sleep(3)
        address = f"{ip}:{port}"
        for _ in range(3):  # the first attempt can race adbd restarting
            out = self.connect_android_wifi(address)
            if "connected" in out and "fail" not in out:
                break
            time.sleep(2)
        self.refresh_android()
        return address

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
