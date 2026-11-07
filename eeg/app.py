"""EEG gesture control. Reads a NeuroPawn Knight headset, learns your eye gestures in a
short calibration, and turns them into commands: left, right, up, down, squint. It can
send those commands to the Raspberry Pi.

    python eeg/app.py

No headset? Choose "Simulator", then use the arrow keys and S to make gestures.
"""

import logging
import queue
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtSerialPort import QSerialPortInfo
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

import signals
from signals import CLASSES, REST, WINDOW_S, Detector, StreamFilter
from sources import BrainFlowSource, SimulatedSource

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "laptop"))
from pi_link import PiLink  # noqa: E402  (shared with the vision app)

log = logging.getLogger("eeg")

MODELS = HERE / "models"
RECORDINGS = HERE / "recordings"
LATEST_MODEL = MODELS / "latest.joblib"

TICK_MS = 50            # read, filter and draw 20 times a second
CLASSIFY_EVERY_S = 0.1  # classify 10 times a second
DISPLAY_S = 5           # seconds of signal shown
SETTLE_S, READY_S, CUE_S = 3.0, 1.5, 2.5   # calibration timing

GREEN, AMBER, RED, BLUE, GREY = "#2e9e4f", "#d49a00", "#d64545", "#2f6fdf", "#888888"
SYMBOL = {REST: "•", "left": "←", "right": "→", "up": "↑", "down": "↓", "squint": "> <"}
INSTRUCTION = {
    REST: "Relax, look at the dot, and blink once or twice",
    "left": "Glance LEFT, then straight back",
    "right": "Glance RIGHT, then straight back",
    "up": "Glance UP, then straight back",
    "down": "Glance DOWN, then straight back",
    "squint": "SQUINT for about a second, then relax",
}
PLACEMENT_HELP = """For eye gestures, put electrodes close to the eyes:

  Ch1  right temple, beside the outer corner of the right eye
  Ch2  left temple, beside the outer corner of the left eye
  Ch3  forehead, just above the right eyebrow
  Ch4  cheekbone, just below the right eye
  Ch5-Ch8  optional: forehead or anywhere on the scalp

Reference and ground electrodes go where NeuroPawn's instructions say
(usually an earlobe or the bone behind the ear).

If your headset has fixed positions, that's fine: the app learns from whatever
placement you calibrate with. Use the channels nearest the eyes and forehead,
and recalibrate whenever you move the electrodes."""
PLOT_COLOURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#bfef45"]


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))


class Calibration:
    """The guided recording: a settle period, then trials in random order. Each trial is a
    get-ready pause (look at the centre) followed by the cue."""

    def __init__(self, trials_per_gesture):
        self.schedule = []
        for _ in range(trials_per_gesture):
            block = CLASSES[:]
            random.shuffle(block)
            self.schedule += block
        self.index = -1
        self.phase = "settle"
        self.phase_end = time.monotonic() + SETTLE_S
        self.cues = []         # (wall-clock time, gesture)
        self.ready_times = []  # wall-clock start of each get-ready pause
        self.raw, self.times = [], []

    def advance(self, now):
        """Moves to the next phase when it's time. Returns the new phase, or None."""
        if now < self.phase_end:
            return None
        if self.phase in ("settle", "cue"):
            self.index += 1
            if self.index >= len(self.schedule):
                self.phase = "done"
                return self.phase
            self.phase, self.phase_end = "ready", now + READY_S
            self.ready_times.append(time.time())
        else:
            self.phase, self.phase_end = "cue", now + CUE_S
            self.cues.append((time.time(), self.schedule[self.index]))
        return self.phase


class MainWindow(QMainWindow):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.source = None
        self.filter = None
        self.buffers = None
        self.bundle = None
        self.detector = None
        self.calibration = None
        self.link = None
        self.last_classify = 0.0
        self.last_pi_send = 0.0
        self.active_cmd, self.hold_until = None, 0.0
        self.manual_cmd = None
        self.pi_sending = "stop"
        self.detected, self.detected_until = None, 0.0
        self.message_until = 0.0
        self.shown_lead_off = None
        self.plot_items, self.curves = [], []

        self.setWindowTitle("Neurotech EEG")
        self.resize(1280, 820)
        self.build_ui()
        self.refresh_ports()
        if LATEST_MODEL.exists():
            try:
                self.bundle = joblib.load(LATEST_MODEL)
                log.info("Loaded the last calibration (%s, %.0f%% accurate)",
                         self.bundle["trained_at"], 100 * self.bundle["accuracy"])
            except Exception as e:
                log.warning("Couldn't load %s: %s", LATEST_MODEL, e)
        self.update_model_label()

        self.timer = QTimer(self, interval=TICK_MS, timeout=self.tick)
        self.timer.start()

    # ------------------------------------------------------------------ UI

    def build_ui(self):
        # Connection bar
        self.device = QComboBox()
        self.device.addItems(list(BrainFlowSource.BOARDS) + [SimulatedSource.name])
        self.device.currentTextChanged.connect(self.device_changed)
        self.port = QComboBox(editable=True)
        self.port.setMinimumWidth(230)
        self.port.lineEdit().setPlaceholderText("e.g. COM3")
        refresh = QPushButton("↻")
        refresh.setToolTip("Refresh the list of serial ports")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self.refresh_ports)
        self.connect_button = QPushButton("Connect")
        self.connect_button.setMinimumWidth(110)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.conn_status = QLabel(dot(GREY, "not connected"))
        self.scale = QComboBox()
        self.scale.addItems(["Auto", "±50 µV", "±100 µV", "±200 µV", "±500 µV", "±1000 µV"])
        self.scale.currentTextChanged.connect(self.apply_scale)
        placement = QPushButton("Electrode placement")
        placement.clicked.connect(lambda: QMessageBox.information(self, "Electrode placement", PLACEMENT_HELP))

        bar = QHBoxLayout()
        for widget in (QLabel("Device:"), self.device, QLabel("Port:"), self.port, refresh,
                       self.connect_button, self.conn_status):
            bar.addWidget(widget)
        bar.addStretch(1)
        for widget in (placement, QLabel("Scale:"), self.scale):
            bar.addWidget(widget)

        # Banner: calibration cues and detected gestures
        self.banner = QLabel()
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setMinimumHeight(90)
        self.banner_text = QLabel()
        self.banner_text.setAlignment(Qt.AlignCenter)
        self.banner_text.setStyleSheet("font-size: 16px;")
        self.set_banner("", "", GREY)

        # Signal plots
        pg.setConfigOptions(antialias=False)
        self.plots = pg.GraphicsLayoutWidget()

        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setFixedHeight(120)

        left = QVBoxLayout()
        left.addLayout(bar)
        left.addWidget(self.banner)
        left.addWidget(self.banner_text)
        left.addWidget(self.plots, 1)
        left.addWidget(self.log_view)

        # Detection
        detection_box = QGroupBox("Detection")
        grid = QGridLayout(detection_box)
        self.bars = {}
        for row, name in enumerate(CLASSES):
            bar_widget = QProgressBar()
            bar_widget.setRange(0, 100)
            bar_widget.setFormat("%p%")
            label = QLabel(f"{SYMBOL[name]}  {name}")
            grid.addWidget(label, row, 0)
            grid.addWidget(bar_widget, row, 1)
            self.bars[name] = bar_widget
        self.threshold_label = QLabel()
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(50, 95)
        self.threshold.valueChanged.connect(self.settings_changed)
        self.threshold.setValue(70)
        self.cooldown = QDoubleSpinBox(suffix=" s", decimals=1, minimum=0.3, maximum=3.0, singleStep=0.1, value=1.0)
        self.cooldown.valueChanged.connect(self.settings_changed)
        row = len(CLASSES)
        grid.addWidget(self.threshold_label, row, 0)
        grid.addWidget(self.threshold, row, 1)
        grid.addWidget(QLabel("Min. gap:"), row + 1, 0)
        grid.addWidget(self.cooldown, row + 1, 1)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        grid.addWidget(self.model_label, row + 2, 0, 1, 2)

        # Calibration
        calibration_box = QGroupBox("Calibration")
        calibration = QVBoxLayout(calibration_box)
        self.trials = QSpinBox(minimum=3, maximum=40, value=10, suffix=" per gesture")
        self.trials.valueChanged.connect(self.update_calibration_time)
        self.calibration_time = QLabel()
        self.calibration_time.setStyleSheet(f"color: {GREY};")
        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("Trials:"))
        trials_row.addWidget(self.trials)
        trials_row.addWidget(self.calibration_time)
        self.calibrate_button = QPushButton("Start calibration")
        self.calibrate_button.setMinimumHeight(36)
        self.calibrate_button.clicked.connect(self.toggle_calibration)
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setFormat("%v / %m trials")
        self.calibration_progress.setValue(0)
        hint = QLabel("Keep your head still. When cued, glance quickly and look straight back; "
                      "between cues, look at the centre of the screen.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {GREY};")
        calibration.addLayout(trials_row)
        calibration.addWidget(self.calibrate_button)
        calibration.addWidget(self.calibration_progress)
        calibration.addWidget(hint)
        self.update_calibration_time()

        # Raspberry Pi
        pi_box = QGroupBox("Raspberry Pi")
        pi = QFormLayout(pi_box)
        self.pi_enable = QCheckBox("Send detected commands to the Pi")
        self.pi_host = QLineEdit("aquapack")
        self.pi_port = QSpinBox(minimum=1, maximum=65535, value=5005)
        self.pi_hold = QDoubleSpinBox(suffix=" s", decimals=1, minimum=0.1, maximum=5.0, singleStep=0.1, value=0.5)
        self.pi_status = QLabel(dot(GREY, "off"))
        for widget in (self.pi_host, self.pi_port):
            (widget.editingFinished).connect(self.reconnect_pi)
        # Hold-to-spin buttons: send left/right while held, overriding detection.
        manual = QHBoxLayout()
        self.manual_buttons = []
        for text, command in (("◀  Left", "left"), ("Right  ▶", "right")):
            button = QPushButton(text)
            button.setMinimumHeight(34)
            button.setFocusPolicy(Qt.NoFocus)
            button.setEnabled(False)
            button.pressed.connect(lambda c=command: setattr(self, "manual_cmd", c))
            button.released.connect(lambda: setattr(self, "manual_cmd", None))
            manual.addWidget(button)
            self.manual_buttons.append(button)
        self.pi_enable.toggled.connect(self.pi_toggled)
        pi.addRow(self.pi_enable)
        pi.addRow("Pi address:", self.pi_host)
        pi.addRow("Port:", self.pi_port)
        pi.addRow("Each command lasts:", self.pi_hold)
        pi.addRow("Hold to spin:", manual)
        pi.addRow("Status:", self.pi_status)

        side = QWidget()
        side.setFixedWidth(360)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(detection_box)
        side_layout.addWidget(calibration_box)
        side_layout.addWidget(pi_box)
        side_layout.addStretch(1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addLayout(left, 1)
        layout.addWidget(side)
        self.setCentralWidget(central)

        # Simulator keys
        keys = {Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Up: "up", Qt.Key_Down: "down",
                Qt.Key_S: "squint", Qt.Key_B: "blink"}
        for key, gesture in keys.items():
            QShortcut(QKeySequence(key), self, activated=lambda g=gesture: self.simulate(g))

    def set_banner(self, big, small, color):
        if (big, small, color) == getattr(self, "_banner", None):
            return
        self._banner = (big, small, color)
        self.banner.setText(big)
        self.banner.setStyleSheet(f"font-size: 54px; font-weight: bold; color: {color};")
        self.banner_text.setText(small)

    def build_plots(self, n_channels):
        self.plots.clear()
        self.plot_items, self.curves = [], []
        for i in range(n_channels):
            plot = self.plots.addPlot(row=i, col=0)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setMenuEnabled(False)
            plot.setXRange(-DISPLAY_S, 0, padding=0)
            plot.getAxis("left").setWidth(70)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showAxis("bottom", i == n_channels - 1)
            if i:
                plot.setXLink(self.plot_items[0])
            self.curves.append(plot.plot(pen=pg.mkPen(PLOT_COLOURS[i % len(PLOT_COLOURS)], width=1)))
            self.plot_items.append(plot)
        self.shown_lead_off = None
        self.update_channel_labels()
        self.apply_scale()

    def update_channel_labels(self):
        lead_off = tuple(self.source.lead_off) if self.source else ()
        if lead_off == self.shown_lead_off:
            return
        self.shown_lead_off = lead_off
        for i, plot in enumerate(self.plot_items):
            off = i < len(lead_off) and lead_off[i]
            plot.setLabel("left", f"Ch{i + 1} (off?)" if off else f"Ch{i + 1}", units="µV",
                          color=RED if off else "#cccccc")

    def apply_scale(self, *_):
        text = self.scale.currentText()
        for plot in self.plot_items:
            if text == "Auto":
                plot.enableAutoRange(axis="y")
            else:
                limit = float(text.strip("±µV "))
                plot.setYRange(-limit, limit, padding=0)

    # ------------------------------------------------------------- settings

    def device_changed(self, name):
        self.port.setEnabled(name != SimulatedSource.name)

    def refresh_ports(self):
        current = self.port.currentText()
        self.port.clear()
        for info in QSerialPortInfo.availablePorts():
            self.port.addItem(f"{info.portName()}  {info.description()}".strip())
        if current:
            self.port.setEditText(current)

    def update_calibration_time(self):
        seconds = SETTLE_S + self.trials.value() * len(CLASSES) * (READY_S + CUE_S)
        self.calibration_time.setText(f"≈ {seconds / 60:.0f} min")
        self.calibration_progress.setMaximum(self.trials.value() * len(CLASSES))

    def settings_changed(self, *_):
        self.threshold_label.setText(f"Certainty ≥ {self.threshold.value()}%:")
        if self.detector:
            self.detector.threshold = self.threshold.value() / 100
            self.detector.cooldown = self.cooldown.value()

    def update_model_label(self):
        b = self.bundle
        if b is None:
            self.model_label.setText("No calibration yet. Connect, then start a calibration.")
            return
        per_class = ", ".join(f"{k} {v:.0%}" for k, v in b["per_class"].items())
        text = (f"<b>Calibration: {b['accuracy']:.0%} accurate</b> on trials it didn't train on<br>"
                f"<span style='color:{GREY}'>{per_class}<br>{b['n_trials']} trials, {b['source']}, "
                f"{b['trained_at'].replace('T', ' ')}</span>")
        if self.source and not self.model_fits():
            text += f"<br><span style='color:{RED}'>Made with a different device: recalibrate.</span>"
        elif b["accuracy"] < 0.8:
            text += (f"<br><span style='color:{AMBER}'>Low accuracy: check electrode contact "
                     f"and placement, then recalibrate.</span>")
        self.model_label.setText(text)

    def model_fits(self):
        b, s = self.bundle, self.source
        return b is not None and s is not None and (b["fs"], b["n_channels"], b["source"]) == (s.fs, s.n_channels, s.name)

    def use_model_if_it_fits(self):
        self.detector = None
        if self.model_fits():
            self.detector = Detector(self.bundle, self.threshold.value() / 100, self.cooldown.value())
        self.update_model_label()

    # ------------------------------------------------------------ connection

    def toggle_connection(self):
        if self.source:
            self.disconnect_source()
            return
        name = self.device.currentText()
        try:
            if name == SimulatedSource.name:
                source = SimulatedSource()
            else:
                port = self.port.currentText().split()[0] if self.port.currentText().strip() else ""
                if not port:
                    raise ValueError("Choose the headset's serial port first (plug it in, then press ↻).")
                source = BrainFlowSource(name, port)
            self.conn_status.setText(dot(AMBER, "connecting..."))
            QApplication.processEvents()
            source.start()
        except Exception as e:
            log.error("Couldn't connect to %s: %s", name, e)
            self.conn_status.setText(dot(RED, "not connected"))
            QMessageBox.warning(self, "Couldn't connect", f"Couldn't connect to {name}:\n\n{e}")
            return

        self.source = source
        self.filter = StreamFilter(source.fs)
        self.buffers = {k: np.zeros((source.n_channels, 0)) for k in ("raw", "eog", "emg", "display")}
        self.build_plots(source.n_channels)
        self.connect_button.setText("Disconnect")
        self.device.setEnabled(False)
        self.port.setEnabled(False)
        self.conn_status.setText(dot(GREEN, f"{source.name}: {source.n_channels} channels, {source.fs} Hz"))
        log.info("Connected to %s (%d channels, %d Hz)", source.name, source.n_channels, source.fs)
        if source.name == SimulatedSource.name:
            log.info("Simulator: arrow keys glance, S squints, B blinks")
        self.use_model_if_it_fits()

    def disconnect_source(self):
        if self.calibration:
            self.stop_calibration("disconnected")
        self.source.stop()
        log.info("Disconnected from %s", self.source.name)
        self.source = None
        self.detector = None
        self.connect_button.setText("Connect")
        self.device.setEnabled(True)
        self.device_changed(self.device.currentText())
        self.conn_status.setText(dot(GREY, "not connected"))
        self.update_model_label()

    def simulate(self, gesture):
        if isinstance(self.source, SimulatedSource) and not self.calibration:
            self.source.trigger(gesture)

    # ------------------------------------------------------------ main loop

    def tick(self):
        for _ in range(200):
            try:
                self.log_view.appendPlainText(self.log_queue.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic()

        if self.source:
            try:
                raw, times = self.source.read()
            except Exception as e:
                log.error("Lost the headset: %s", e)
                self.disconnect_source()
                return
            if raw.shape[1]:
                filtered = self.filter.process(raw)
                filtered["raw"] = raw
                keep = int(DISPLAY_S * self.source.fs)
                for name, buffer in self.buffers.items():
                    self.buffers[name] = np.concatenate([buffer, filtered[name]], axis=1)[:, -keep:]
                if self.calibration:
                    self.calibration.raw.append(raw)
                    self.calibration.times.append(times)
                self.draw()
            self.update_channel_labels()

        if self.calibration:
            self.run_calibration(now)
        elif self.detector and now - self.last_classify >= CLASSIFY_EVERY_S:
            self.classify(now)
        self.update_pi(now)
        if not self.calibration:
            self.show_idle_banner(now)

    def draw(self):
        display = self.buffers["display"]
        x = np.arange(-display.shape[1], 0) / self.source.fs
        for curve, y in zip(self.curves, display):
            curve.setData(x, y)

    def classify(self, now):
        win = int(WINDOW_S * self.source.fs)
        if self.buffers["raw"].shape[1] < win:
            return
        self.last_classify = now
        b = self.buffers
        probs, fired = self.detector.update(b["eog"][:, -win:], b["emg"][:, -win:], b["raw"][:, -win:],
                                            self.source.fs, now)
        for name, bar_widget in self.bars.items():
            bar_widget.setValue(round(100 * probs.get(name, 0)))
        if fired:
            log.info("Detected: %s", fired.upper())
            self.detected, self.detected_until = fired, now + 1.0
            self.active_cmd, self.hold_until = fired, now + self.pi_hold.value()

    def show_idle_banner(self, now):
        if now < self.message_until:
            return
        if self.detected and now < self.detected_until:
            self.set_banner(SYMBOL[self.detected], self.detected.upper(), GREEN)
        elif not self.source:
            self.set_banner("", "Connect a headset (or the Simulator) to begin", GREY)
        elif not self.detector:
            self.set_banner("", "Start a calibration so the app can learn your gestures", GREY)
        else:
            self.set_banner(SYMBOL[REST], "Ready: glance left, right, up or down, or squint", GREY)

    # ------------------------------------------------------------ calibration

    def toggle_calibration(self):
        if self.calibration:
            self.stop_calibration("cancelled")
            return
        if not self.source:
            QMessageBox.information(self, "Calibration", "Connect the headset first.")
            return
        self.calibration = Calibration(self.trials.value())
        self.calibration_progress.setMaximum(len(self.calibration.schedule))
        self.calibration_progress.setValue(0)
        self.calibrate_button.setText("Stop calibration")
        self.trials.setEnabled(False)
        self.set_banner("+", "Get comfortable and keep your head still. Starting in a moment...", GREY)
        log.info("Calibration started: %d trials", len(self.calibration.schedule))

    def stop_calibration(self, reason):
        self.calibration = None
        self.calibrate_button.setText("Start calibration")
        self.trials.setEnabled(True)
        log.info("Calibration %s", reason)

    def run_calibration(self, now):
        cal = self.calibration
        phase = cal.advance(now)
        if phase == "ready":
            upcoming = cal.schedule[cal.index]
            self.set_banner("+", f"Look here.  Next: {INSTRUCTION[upcoming]}", GREY)
            self.calibration_progress.setValue(cal.index)
        elif phase == "cue":
            gesture = cal.schedule[cal.index]
            self.set_banner(SYMBOL[gesture], INSTRUCTION[gesture], BLUE)
            if isinstance(self.source, SimulatedSource):
                # The simulator acts out the cue, with a human-like reaction time.
                self.source.trigger(gesture if gesture != REST else "blink", delay=random.uniform(0.25, 0.45))
        elif phase == "done":
            self.calibration_progress.setValue(len(cal.schedule))
            self.stop_calibration("finished")
            self.finish_calibration(cal)

    def finish_calibration(self, cal):
        source = self.source
        raw = np.concatenate(cal.raw, axis=1)
        times = np.concatenate(cal.times)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RECORDINGS.mkdir(exist_ok=True)
        recording = RECORDINGS / f"calibration_{stamp}.npz"
        np.savez_compressed(recording, raw=raw, times=times, fs=source.fs, source=source.name,
                            cue_times=[c[0] for c in cal.cues], cue_labels=[c[1] for c in cal.cues],
                            ready_times=cal.ready_times)
        log.info("Saved the recording to %s", recording)

        try:
            X, y, groups = signals.training_set(raw, times, cal.cues, source.fs, cal.ready_times)
            if len(set(y)) < len(CLASSES):
                raise ValueError("not enough data for every gesture; was the recording interrupted?")
            self.bundle = signals.train(X, y, groups, source.fs, source.n_channels, source.name)
        except Exception as e:
            log.exception("Training failed")
            QMessageBox.warning(self, "Calibration", f"Training failed: {e}")
            return
        MODELS.mkdir(exist_ok=True)
        joblib.dump(self.bundle, MODELS / f"model_{stamp}.joblib")
        joblib.dump(self.bundle, LATEST_MODEL)
        log.info("Calibration accuracy: %.0f%% (%s)", 100 * self.bundle["accuracy"],
                 ", ".join(f"{k} {v:.0%}" for k, v in self.bundle["per_class"].items()))
        for (true, guess), count in sorted(self.bundle["confusions"].items(), key=lambda kv: -kv[1]):
            log.info("  %s mistaken for %s: %d windows", true, guess, count)
        self.use_model_if_it_fits()
        accuracy = self.bundle["accuracy"]
        self.set_banner(f"{accuracy:.0%}", "Calibration done. Detection is on.", GREEN if accuracy >= 0.8 else AMBER)
        self.message_until = time.monotonic() + 3

    # ------------------------------------------------------------ Raspberry Pi

    def reconnect_pi(self):
        if self.link:
            self.link.close()
            self.link = None

    def pi_toggled(self, on):
        for button in self.manual_buttons:
            button.setEnabled(on)
        if not on:
            self.manual_cmd = None  # a disabled button never sends "released"

    def update_pi(self, now):
        if not self.pi_enable.isChecked():
            if self.link:
                self.link.send({"cmd": "stop"})
                self.link.close()
                self.link = None
            self.pi_status.setText(dot(GREY, "off"))
            return
        if self.link is None:
            self.link = PiLink(self.pi_host.text().strip(), self.pi_port.value())
        if now - self.last_pi_send >= 0.1:
            self.last_pi_send = now
            if self.manual_cmd:
                self.pi_sending = self.manual_cmd
            elif now < self.hold_until:
                self.pi_sending = self.active_cmd
            else:
                self.pi_sending = "stop"
            self.link.send({"cmd": self.pi_sending})
        if self.link.connected:
            self.pi_status.setText(dot(GREEN, f"connected, sending {self.pi_sending.upper()}"))
        else:
            self.pi_status.setText(dot(RED, "not connected"))

    def closeEvent(self, event):
        self.timer.stop()
        if self.link:
            self.link.send({"cmd": "stop"})
            self.link.close()
        if self.source:
            self.source.stop()
        event.accept()


def main():
    log_queue = queue.Queue()
    gui_handler = QueueLogHandler(log_queue)
    gui_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[console, gui_handler])

    app = QApplication(sys.argv)
    window = MainWindow(log_queue)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
