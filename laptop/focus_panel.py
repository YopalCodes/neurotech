"""The Focus tab: pick the phone or the laptop by looking at its flickering box (SSVEP), then
control it through the Devices tab.

How a selection goes:
  1. Double blink (or click Start selecting). A full-screen window opens with two boxes,
     Phone (left) and Laptop (right), each flickering at its own rate (eog/focus.py, after
     NeuroPawn's SSVEP tutorial). A box only appears while the camera sees that kind of device.
  2. Keep looking at the box of the device you want. When the same box wins CONFIRM looks in a
     row (about 4 s), it's selected: the window closes, and the device chosen for that box in
     this tab is connected and its screen opens (Devices tab, remote/).

Each selection is also announced to other programs as one JSON line on 127.0.0.1:5010:

    {"event": "focus_select", "box": 0, "device": "Phone", "connects_to": "DE2118 (3f93e4ec)",
     "frequency": 6.67, "time": "2026-11-08T09:31:02"}

WARNING: flicker at these rates can trigger seizures in people with photosensitive epilepsy.
"""

import json
import logging
import math
import random
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget,
)

sys.path.append(str(Path(__file__).resolve().parent.parent / "eog"))
import focus  # noqa: E402

log = logging.getLogger("focus")

# One entry per box, in focus.FREQUENCIES order (left to right on screen).
NAMES = ["Phone", "Laptop"]
CAMERA_WORDS = ["phone", "laptop"]               # a detection whose name contains this word counts
DEVICE_KINDS = ["AndroidDevice", "ComputerDevice"]   # which Devices-tab entries each box can connect to
assert len(NAMES) == len(CAMERA_WORDS) == len(DEVICE_KINDS) == focus.N_TARGETS

EVENT_PORT = 5010
CHOICES_FILE = Path(__file__).with_name("focus_devices.json")
CONFIRM = 2              # the same box must win this many looks in a row to be selected
SEEN_FOR = 2.0           # a device counts as in view for this long after the camera last saw it
SELECT_TIMEOUT = 20.0    # a selection window closes by itself if nothing is selected in this time
SELECTED_SHOW_S = 1.0    # how long the selected box is shown before the window closes
GREEN, AMBER, RED, GREY = "#2e9e4f", "#d49a00", "#d64545", "#888888"
WARNING = ("Flickering boxes can trigger seizures in people with photosensitive epilepsy. "
           "Don't use Focus if that applies to you, and stop if you feel unwell.")


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


def short(name):
    """"DE2118 (3f93e4ec)" -> "DE2118" (as the Devices tab shows it)."""
    return name.split(" (")[0] if name else name


def load_choices():
    try:
        choices = json.loads(CHOICES_FILE.read_text(encoding="utf-8"))
        return choices if isinstance(choices, dict) else {}
    except (OSError, ValueError):
        return {}


class FocusEvents:
    """Announces selections to other programs: one JSON line per selection on 127.0.0.1:port."""

    def __init__(self, port=EVENT_PORT):
        self.port, self.error = port, None
        self._clients, self._lock = [], threading.Lock()
        try:
            self._server = socket.create_server(("127.0.0.1", port))
            threading.Thread(target=self._accept, name="focus-events", daemon=True).start()
        except OSError as e:
            self._server = None
            self.error = f"Can't announce selections on port {port}: {e}"
            log.warning(self.error)

    def _accept(self):
        while self._server is not None:
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            with self._lock:
                self._clients.append(client)

    @property
    def listeners(self):
        return len(self._clients)

    def publish(self, event):
        line = (json.dumps(event) + "\n").encode()
        with self._lock:
            for client in list(self._clients):
                try:
                    client.sendall(line)
                except OSError:
                    self._clients.remove(client)
                    client.close()

    def close(self):
        server, self._server = self._server, None
        if server is not None:
            server.close()
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()


class FlickerView(QOpenGLWidget):
    """Draws the boxes, redrawn on every screen refresh (vsync) so each box's flicker follows
    its sine wave frame by frame, as in NeuroPawn's stimulus."""

    def __init__(self, names, subtitles=None):
        super().__init__()
        fmt = QSurfaceFormat()
        fmt.setSwapInterval(1)          # one frame per screen refresh
        self.setFormat(fmt)
        self.names = names
        self.subtitles = subtitles or [""] * len(names)
        self.shown = set(range(len(names)))   # boxes on screen; the others are greyed out
        self.phase = "message"          # message | cue | flicker | rest
        self.message = ""
        self.cue = None                 # box outlined in red (training)
        self.selected = None            # box outlined in green after a selection
        self.onset_perf = None          # first flicker frame (perf_counter)
        self.onset_wall = None          # same moment, time.time() (matches EEG timestamps)
        self.flicker_requested = 0.0
        self.frame_times = []
        self.frameSwapped.connect(self.update)

    def start_flicker(self):
        self.phase, self.onset_perf, self.onset_wall, self.frame_times = "flicker", None, None, []
        self.flicker_requested = time.perf_counter()

    def frame_stats(self):
        """(measured refresh Hz, % of frames late) over the last flicker."""
        times = self.frame_times
        if len(times) < 10:
            return None
        gaps = [b - a for a, b in zip(times, times[1:])]
        typical = sorted(gaps)[len(gaps) // 2]
        late = sum(1 for g in gaps if g > 1.5 * typical)
        return 1 / typical, 100 * late / len(gaps)

    def boxes(self, w, h):
        n = len(self.names)
        side = min(w / n * 0.6, h * 0.5)
        return [QRectF(w * (i + 0.5) / n - side / 2, h * 0.45 - side / 2, side, side) for i in range(n)]

    def paintGL(self):
        p = QPainter(self)
        self.draw(p, time.perf_counter(), self.width(), self.height())
        p.end()

    def draw(self, p, now, w, h):
        """Draws one frame of size w x h at time `now` (perf_counter)."""
        p.fillRect(QRectF(0, 0, w, h), QColor("black"))
        if self.phase == "flicker":
            if self.onset_perf is None:
                self.onset_perf, self.onset_wall = now, time.time()
            self.frame_times.append(now)
        name_font, small_font = QFont(), QFont()
        name_font.setPointSize(20)
        name_font.setBold(True)
        small_font.setPointSize(12)
        for i, box in enumerate(self.boxes(w, h)):
            shown = i in self.shown
            if not shown:
                p.fillRect(box, QColor("#0c0c0c"))
            elif self.phase == "flicker":
                on = math.sin(2 * math.pi * focus.FREQUENCIES[i] * (now - self.onset_perf)) >= 0
                p.fillRect(box, QColor("white") if on else QColor("black"))
            else:
                p.fillRect(box, QColor("#202020"))
            outline = RED if self.cue == i else GREEN if self.selected == i else None
            if outline:
                p.setPen(QPen(QColor(outline), 10))
                p.drawRect(box.adjusted(-12, -12, 12, 12))
            p.setPen(QColor(GREEN if self.selected == i else "#cfcfcf" if shown else "#555555"))
            p.setFont(name_font)
            label = QRectF(box.left() - 60, box.bottom() + 18, box.width() + 120, 44)
            p.drawText(label, Qt.AlignHCenter | Qt.AlignTop, ("✔ " if self.selected == i else "") + self.names[i])
            p.setFont(small_font)
            p.setPen(QColor("#8a8a8a" if shown else "#555555"))
            sub = self.subtitles[i] if shown else "not in camera view"
            p.drawText(label.translated(0, 44), Qt.AlignHCenter | Qt.AlignTop, sub)
        if self.message:
            msg_font = QFont()
            msg_font.setPointSize(16)
            p.setFont(msg_font)
            p.setPen(QColor("#e8e8e8"))
            p.drawText(QRectF(w * 0.1, 0, w * 0.8, h * 0.2), Qt.AlignCenter | Qt.TextWordWrap, self.message)


class FocusWindow(QWidget):
    """Runs a training session (mode "train") or one selection (mode "select").
    visible() -> the boxes whose device the camera sees (select mode)."""

    def __init__(self, eyes, names, mode, events, confirm=CONFIRM, blocks=focus.TRAINING_BLOCKS,
                 on_selected=None, on_closed=None, visible=None, subtitles=None):
        super().__init__()
        self.eyes, self.names, self.mode, self.events = eyes, names, mode, events
        self.confirm, self.on_selected, self.on_closed = confirm, on_selected, on_closed
        self.visible = visible or (lambda: set(range(len(names))))
        self.setWindowTitle("Watchdog Focus: " + ("training" if mode == "train" else "look at a device"))
        self.view = FlickerView(names, subtitles)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.resize(1280, 800)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self.toggle_fullscreen)

        self.session_dir = focus.new_session_dir() if mode == "train" else None
        if mode == "train":
            self.plan = self.training_plan(blocks)
        else:
            self.plan = [("message", "Look at the device you want, and keep looking.", 1.0)]
            self.view.shown = self.visible()
        self.trial_id = int(time.time() * 1000) % 10 ** 9
        self.collected, self.pending = [], []
        self.streak_box, self.streak = None, 0
        self.phase_end, self.current = 0.0, None
        self.deadline = time.perf_counter() + SELECT_TIMEOUT
        self.close_at = None
        self.result = None              # select mode: the selection event, or None
        self.last_stats = None
        self.failure = None
        self.done = False
        self.timer = QTimer(self, interval=5, timeout=self.step)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.start()

    def training_plan(self, blocks):
        plan = [("message", "Training. Look at the box outlined in red, and keep looking at it while "
                            "the boxes flicker. Blink between trials, not during the flicker.", 4.0)]
        for block in range(1, blocks + 1):
            plan.append(("message", f"Block {block} of {blocks}: get ready", 1.5))
            for target in random.sample(range(focus.N_TARGETS), focus.N_TARGETS):
                plan += [("cue", target, focus.CUE_S), ("flicker", ("collect", block, target), None),
                         ("rest", None, focus.REST_S)]
        return plan

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ---- sequencing ----

    def next_phase(self, now):
        v = self.view
        if self.mode == "select" and not self.plan:
            v.shown = self.visible()
            if v.shown:
                self.plan = [("flicker", ("predict",), None), ("rest", None, focus.REST_S)]
            else:
                self.plan = [("message", "Point the camera at the phone or the laptop.", 0.5)]
        if not self.plan:
            return self.finish_training()
        kind, arg, seconds = self.plan.pop(0)
        self.current = (kind, arg)
        v.cue = arg if kind == "cue" else None
        if kind == "message":
            v.phase, v.message = "message", arg
        elif kind == "cue":
            v.phase, v.message = "cue", ""
        elif kind == "rest":
            v.phase = "rest"
        elif kind == "flicker":
            v.message = ""
            v.start_flicker()
            self.phase_end = float("inf")   # set once the first flicker frame is on screen
            return
        self.phase_end = now + seconds

    def end_flicker(self):
        self.trial_id += 1
        what = self.current[1]
        self.last_stats = self.view.frame_stats()
        if what[0] == "collect":
            _, block, target = what
            self.eyes.focus_capture(self.trial_id, self.view.onset_wall, "collect", self.session_dir, block, target)
            self.collected.append(self.trial_id)
        else:
            self.eyes.focus_capture(self.trial_id, self.view.onset_wall, "predict")
            self.pending.append(self.trial_id)

    def step(self):
        now = time.perf_counter()
        v = self.view
        if self.close_at is not None:
            if now >= self.close_at:
                self.close()
            return
        if v.phase == "flicker":
            v.update()   # normally redrawn on each vsync already; this restarts it if a frame was skipped
        elif self.mode == "select":
            v.shown = self.visible()   # between looks, show exactly what the camera sees now
        if self.current and self.current[0] == "flicker" and self.phase_end == float("inf"):
            if v.onset_perf is not None:
                self.phase_end = v.onset_perf + focus.FLICKER_S
            elif now - v.flicker_requested > 1.0:
                return self.fail("The flicker window isn't being drawn, so Focus can't run. "
                                 "Update the graphics driver, and keep the window on screen "
                                 "(not minimised or covered).")
        if self.mode == "select" and now > self.deadline:
            log.info("Focus: nothing selected in %.0f s", SELECT_TIMEOUT)
            return self.close()
        if now >= self.phase_end and not self.done:
            if self.current and self.current[0] == "flicker":
                self.end_flicker()
            self.next_phase(now)
        for trial_id in list(self.pending):
            if trial_id in self.eyes.focus_predictions:
                self.pending.remove(trial_id)
                self.on_prediction(self.eyes.focus_predictions.pop(trial_id), now)
            elif trial_id in self.eyes.focus_missed:
                self.pending.remove(trial_id)
                self.eyes.focus_missed.pop(trial_id)
                self.streak_box, self.streak = None, 0   # a gap breaks the run of matching looks
        if self.done and self.mode == "train":
            self.show_training_result()

    def on_prediction(self, box, now):
        if self.close_at is not None:
            return
        if box not in self.view.shown:     # a device the camera doesn't see can't be picked
            self.streak_box, self.streak = None, 0
            return
        self.streak = self.streak + 1 if box == self.streak_box else 1
        self.streak_box = box
        if self.streak < self.confirm:
            return
        self.result = {"event": "focus_select", "box": box, "device": self.names[box],
                       "frequency": focus.FREQUENCIES[box],
                       "time": datetime.now().isoformat(timespec="seconds")}
        self.view.selected, self.view.phase, self.view.message = box, "rest", ""
        self.close_at = now + SELECTED_SHOW_S
        log.info("Focus: selected %s", self.names[box])
        if self.on_selected:
            self.on_selected(self.result)
        if self.events is not None:
            self.events.publish(self.result)

    # ---- end of training ----

    def finish_training(self):
        self.done = True
        self.view.phase, self.view.cue = "message", None
        self.view.message = "Saving the last trials..."
        self.finish_deadline = time.perf_counter() + 6
        self.fit_requested = False

    def show_training_result(self):
        saved = sum(1 for t in self.collected if t in self.eyes.focus_saved)
        missed = sum(1 for t in self.collected if t in self.eyes.focus_missed)
        if not self.fit_requested:
            if saved + missed < len(self.collected) and time.perf_counter() < self.finish_deadline:
                return
            self.fit_requested = True
            self.eyes.focus_fit(self.session_dir)
            self.view.message = f"Saved {saved} of {len(self.collected)} trials. Training the model..."
            return
        info = self.eyes.focus_model
        if info is None:
            return
        if "error" in info:
            self.view.message = f"Training failed: {info['error']}\n\nEsc to close."
        else:
            acc = info["accuracy"]
            acc_text = f"{acc:.0%} accurate on blocks it didn't train on" if acc is not None else "trained"
            self.view.message = (f"Done: {info['blocks']} blocks, {acc_text}.\n\n"
                                 "Esc to close. Then double blink to pick a device.")
        self.timer.stop()

    def fail(self, message):
        log.error("Focus: %s", message)
        self.failure = message
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.on_closed:
            self.on_closed(self)
        event.accept()


class FocusPanel(QWidget):
    """The Focus tab. `engine` supplies the camera's detections; `devices` is the Devices tab,
    which connects the selected device and opens its screen."""

    def __init__(self, eyes, events, engine=None, devices=None):
        super().__init__()
        self.eyes, self.events, self.engine, self.devices = eyes, events, engine, devices
        self.window = None
        self.before_open = None     # called before a Focus window opens (app.py: eye control off)
        self.show_devices = None    # called when a selection connects (app.py: shows the Devices tab)
        self.pending_connect = None # selection to connect once the Focus window has closed
        self.raise_target = None    # (device name, give up time): bring its screen window to the front
        self.last_selection = None
        self.last_seen = [0.0] * focus.N_TARGETS
        self.snap_id = None
        self.blinks_seen = eyes.double_blinks if eyes is not None else 0
        self.device_names = None
        self.choices = load_choices()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel("<b>Double blink</b>, then look at the Phone or Laptop box to control that "
                       "device. A box only appears while the camera sees the device.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        status_box = QGroupBox("Status")
        form = QFormLayout(status_box)
        self.channels = QLabel()
        self.channels.setWordWrap(True)
        self.model = QLabel()
        self.model.setWordWrap(True)
        self.camera = QLabel()
        self.display = QLabel("–")
        self.display.setWordWrap(True)
        self.last = QLabel("–")
        self.last.setWordWrap(True)
        form.addRow("Electrodes:", self.channels)
        form.addRow("Model:", self.model)
        form.addRow("Camera sees:", self.camera)
        form.addRow("Display:", self.display)
        form.addRow("Last selected:", self.last)
        layout.addWidget(status_box)

        devices_box = QGroupBox("What each box connects to")
        devices_form = QFormLayout(devices_box)
        self.combos = []
        for name in NAMES:
            combo = QComboBox()
            combo.currentIndexChanged.connect(self.save_choices)
            devices_form.addRow(f"{name}:", combo)
            self.combos.append(combo)
        hint = QLabel("Add phones and PCs in the Devices tab.")
        hint.setStyleSheet(f"color: {GREY};")
        devices_form.addRow(hint)
        layout.addWidget(devices_box)

        self.double_blink = QCheckBox("Double blink starts selecting")
        self.double_blink.setChecked(True)
        layout.addWidget(self.double_blink)
        self.confirm = QCheckBox(f"Require {CONFIRM} matching looks (fewer mistakes)")
        self.confirm.setChecked(True)
        layout.addWidget(self.confirm)
        self.fullscreen = QCheckBox("Full screen (recommended)")
        self.fullscreen.setChecked(True)
        layout.addWidget(self.fullscreen)
        self.train_button = QPushButton(f"Train ({focus.TRAINING_BLOCKS} blocks, about 1 minute)")
        self.train_button.clicked.connect(lambda: self.open_window("train"))
        self.select_button = QPushButton("Start selecting")
        self.select_button.setMinimumHeight(36)
        self.select_button.clicked.connect(lambda: self.open_window("select"))
        layout.addWidget(self.train_button)
        layout.addWidget(self.select_button)

        warning = QLabel("⚠ " + WARNING)
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {AMBER};")
        layout.addWidget(warning)
        layout.addStretch(1)
        self.refresh_device_choices()

    # ---- camera ----

    def update_seen(self):
        snap = self.engine.latest() if self.engine is not None else None
        if snap is None or snap.id == self.snap_id:
            return
        self.snap_id = snap.id
        now = time.monotonic()
        for name, _ in snap.detections:
            for i, word in enumerate(CAMERA_WORDS):
                if word in name.lower():
                    self.last_seen[i] = now

    def in_view(self):
        if self.engine is None:
            return set(range(focus.N_TARGETS))
        now = time.monotonic()
        return {i for i, t in enumerate(self.last_seen) if now - t < SEEN_FOR}

    # ---- devices ----

    def refresh_device_choices(self):
        manager = getattr(self.devices, "manager", None)
        devices = dict(manager.devices) if manager is not None else {}
        names = sorted(devices)
        if names == self.device_names:
            return
        self.device_names = names
        for i, combo in enumerate(self.combos):
            fitting = [n for n in names if type(devices[n]).__name__ == DEVICE_KINDS[i]]
            combo.blockSignals(True)
            combo.clear()
            if not fitting:
                combo.addItem(f"(no {NAMES[i].lower()} yet)", None)
            for n in fitting:
                combo.addItem(short(n), n)
            wanted = self.choices.get(NAMES[i])
            index = combo.findData(wanted) if wanted else -1
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)

    def target(self, box):
        return self.combos[box].currentData()

    def save_choices(self):
        for i, combo in enumerate(self.combos):
            if combo.currentData():
                self.choices[NAMES[i]] = combo.currentData()
        try:
            CHOICES_FILE.write_text(json.dumps(self.choices, indent=2), encoding="utf-8")
        except OSError as e:
            log.warning("Couldn't save the Focus device choices: %s", e)

    def connect_selected(self, event):
        name = event.get("connects_to")
        if self.show_devices:
            self.show_devices()
        if name is None:
            log.warning("Focus: no %s chosen for the %s box. Add one in the Devices tab, then pick it "
                        "in the Focus tab.", event["device"].lower(), event["device"])
            return
        if self.devices is not None and name in self.devices.windows:
            log.info("Focus: %s is already connected; showing its screen", short(name))
        else:
            log.info("Focus: connecting to %s", short(name))
            if self.devices is not None:
                self.devices.connect_device(name)   # may ask for the PC's password first
        self.raise_target = (name, time.monotonic() + 20)

    def raise_device_window(self):
        """Brings the selected device's screen window to the front once it exists."""
        name, give_up = self.raise_target
        window = self.devices.windows.get(name) if self.devices is not None else None
        if window is not None:
            window.showNormal()
            window.raise_()
            window.activateWindow()
            self.raise_target = None
        elif time.monotonic() > give_up or (self.devices is not None and self.devices.connecting is None
                                             and time.monotonic() > give_up - 18):
            self.raise_target = None   # connecting failed or was cancelled: the Devices tab says why

    # ---- windows ----

    def ready(self):
        e = self.eyes
        return e is not None and e.connected and e.use_focus and e.focus_ready and not e.error

    def trained(self):
        info = self.eyes.focus_model if self.eyes is not None else None
        return info is not None and "error" not in info

    def open_window(self, mode):
        if self.window is not None:
            self.window.close()
        if self.before_open:
            self.before_open()
        subtitles = [("→ " + short(self.target(i))) if self.target(i) else "no device chosen"
                     for i in range(focus.N_TARGETS)]
        self.window = FocusWindow(self.eyes, NAMES, mode, self.events,
                                  confirm=CONFIRM if self.confirm.isChecked() else 1,
                                  on_selected=self.selected, on_closed=self.closed,
                                  visible=self.in_view if mode == "select" else None,
                                  subtitles=subtitles if mode == "select" else None)
        self.window.showFullScreen() if self.fullscreen.isChecked() else self.window.show()

    def double_blinked(self):
        if not self.double_blink.isChecked() or self.window is not None:
            return
        if not (self.ready() and self.trained()):
            log.info("Double blink: Focus isn't ready yet (needs the electrodes on and a training)")
        elif not self.in_view():
            log.info("Double blink: the camera doesn't see a phone or a laptop")
        else:
            self.open_window("select")

    def selected(self, event):
        event["connects_to"] = self.target(event["box"])
        self.last_selection = event
        self.pending_connect = event   # connected once the Focus window has closed (see closed())

    def closed(self, window):
        if self.window is window:
            if window.failure:
                self.display.setText(dot(RED, window.failure))
            elif window.last_stats:
                hz, late = window.last_stats
                self.display.setText(dot(GREEN if late < 5 else AMBER, f"{hz:.0f} Hz, {late:.0f}% frames late"))
            self.window = None
            if self.pending_connect is not None:
                event, self.pending_connect = self.pending_connect, None
                QTimer.singleShot(200, lambda: self.connect_selected(event))

    def stop(self):
        if self.window is not None:
            self.window.close()

    # ---- display ----

    def refresh(self):
        e = self.eyes
        self.update_seen()
        self.refresh_device_choices()
        if self.raise_target is not None:
            self.raise_device_window()
        if e is not None and e.double_blinks > self.blinks_seen:
            self.blinks_seen = e.double_blinks
            self.double_blinked()

        if e is None or not e.use_focus:
            self.channels.setText(dot(GREY, "off: start the app with --eog-port COM10 --focus"))
        elif e.focus_error:
            self.channels.setText(dot(RED, e.focus_error))
        elif e.focus_ready:
            self.channels.setText(dot(GREEN, ", ".join(focus.ELECTRODES[ch] for ch in focus.FOCUS_CHANNELS) + " live"))
        else:
            self.channels.setText(dot(AMBER, "starting..."))
        info = e.focus_model if e is not None else None
        if info is None:
            self.model.setText("–")
        elif "error" in info:
            self.model.setText(dot(AMBER, info["error"]))
        else:
            acc = info["accuracy"]
            self.model.setText(dot(GREEN if (acc or 0) >= 0.8 else AMBER,
                                   (f"{acc:.0%} accurate" if acc is not None else "trained") +
                                   f", {info['blocks']} blocks ({info['session']})"))
        seen = self.in_view()
        self.camera.setText("  ".join(
            (f'<span style="color:{GREEN}">✔ {n}</span>' if i in seen else
             f'<span style="color:{GREY}">✗ {n}</span>') for i, n in enumerate(NAMES)))
        if self.last_selection:
            s = self.last_selection
            to = f" → {short(s['connects_to'])}" if s.get("connects_to") else ""
            self.last.setText(f"<b>{s['device']}</b>{to} at {s['time'][11:]}")
        if self.window is not None and self.window.last_stats:
            hz, late = self.window.last_stats
            self.display.setText(dot(GREEN if late < 5 else AMBER, f"{hz:.0f} Hz, {late:.0f}% frames late"))
        ok = self.ready()
        self.train_button.setEnabled(ok)
        self.select_button.setEnabled(ok and self.trained())


def listen_for_selections(host="127.0.0.1", port=EVENT_PORT):
    """Minimal listener for other programs: prints every selection the app announces."""
    with socket.create_connection((host, port)) as s:
        for line in s.makefile("r", encoding="utf-8"):
            print(json.loads(line))
