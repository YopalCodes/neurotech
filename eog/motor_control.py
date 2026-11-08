"""Control the stepper motor on the Raspberry Pi with your eyes, using the EOG classifier
from train.py.

    python motor_control.py --port COM10 --pi 10.153.4.29
    python motor_control.py --synthetic --pi 10.153.4.29    # no headset: use the buttons or keys

    Look right  -> the motor spins right, and keeps spinning
    Look left   -> the motor spins left, and keeps spinning
    Blink       -> the motor stops

The board is handled exactly as in live.py: board.connect() once at startup, the same
reads every 0.1 s, and stop_stream() + release_session() on exit. If connecting fails, or
the board stops sending data, close the program, unplug and replug the board, and run it
again. Close the NeuroPawn Visualizer first.

Detection uses live.py's classifier and timing: the newest 1.5 s window is classified every
0.1 s, and new events are ignored for 1.2 s after each one. The one difference: an event
fires as soon as a single check passes the certainty threshold (live.py waits for three
in a row).

Head tilt (IMU) works as in live.py: calibrated at startup (follow the prompts in the
terminal), reported as left/right/forward/back/neutral, and while the head turns faster than
HEAD_MOTION_GATE (60 deg/s) eye events are ignored for 1.5 s, because turning the head makes
the eyes counter-rotate, which looks like a glance. --no-imu skips it.

The motor state is sent to the Pi 10 times a second. If this program, the headset or the
Wi-Fi stops, the Pi's failsafe stops the motor within half a second.
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from brainflow.board_shim import BoardShim
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

import board as eog_board
import config
import features
from imu import HeadTilt
from live import BUFFER_SECONDS, MODEL_PATH, REFRACTORY, STEP

# Checks in a row that must pass the certainty threshold before an event fires.
# live.py uses 3; here one is enough, so a short blink isn't missed.
CONFIRM = 1

sys.path.append(str(Path(__file__).resolve().parent.parent / "laptop"))
from pi_link import PiLink  # noqa: E402  (shared with the vision app)

EVENT_TO_STATE = {"left": "left", "right": "right", "blink": "stop"}
STATE_TEXT = {"left": "◀  SPINNING LEFT", "right": "SPINNING RIGHT  ▶", "stop": "■  STOPPED"}
GREEN, RED, GREY = "#2e9e4f", "#d64545", "#888888"


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


class EventDetector:
    """live.py's detection loop, one step at a time, firing after CONFIRM passing checks."""

    def __init__(self, model, fs, threshold):
        self.model = model
        self.fs = fs
        self.threshold = threshold
        self.n = features.epoch_length(fs)
        self.streak_label, self.streak = None, 0
        self.quiet_until = 0.0

    def update(self, eog, now):
        """eog: (2, samples) raw [horizontal, vertical]. Returns ({label: p}, event or None)."""
        filtered = features.filter_eog(eog, self.fs)
        x = features.window_features(filtered[:, -self.n:], self.fs).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        best = int(np.argmax(proba))
        label, p = self.model.classes_[best], proba[best]
        probs = dict(zip(self.model.classes_, proba))

        if label == "rest" or p < self.threshold or now < self.quiet_until:
            self.streak_label, self.streak = None, 0
            return probs, None
        self.streak = self.streak + 1 if label == self.streak_label else 1
        self.streak_label = label
        if self.streak == CONFIRM:
            self.quiet_until = now + REFRACTORY
            self.streak_label, self.streak = None, 0
            return probs, (label, p)
        return probs, None


class MainWindow(QMainWindow):
    def __init__(self, board, board_id, model, args, tilt=None):
        super().__init__()
        self.setWindowTitle("EOG Motor Control")
        self.resize(760, 620)
        self.board = board
        self.tilt = tilt
        self.head_until = 0.0   # eye events paused until then because the head was turning
        self.fs = BoardShim.get_sampling_rate(board_id)
        self.rows = list(eog_board.eog_rows(board_id))
        self.detector = EventDetector(model, self.fs, args.threshold)
        self.state = "stop"

        source = "synthetic board" if args.synthetic else (
            f"{args.port}, channels {config.HORIZONTAL_CHANNEL} (horizontal) and "
            f"{config.VERTICAL_CHANNEL} (vertical)")
        self.headset_status = QLabel(dot(GREEN, f"Headset: {source}, {self.fs} Hz"))
        self.headset_status.setWordWrap(True)
        self.head_label = QLabel("Head tilt: off (--no-imu or synthetic board)" if tilt is None else "")

        # Motor state
        self.state_label = QLabel()
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setMinimumHeight(90)
        self.last_event = QLabel("Warming up for a few seconds, then: look right or left to spin "
                                 "the motor, blink to stop it.")
        self.last_event.setAlignment(Qt.AlignCenter)
        self.last_event.setWordWrap(True)
        self.last_event.setStyleSheet("font-size: 15px;")

        # Detection
        detection_box = QGroupBox("Detection")
        detection = QFormLayout(detection_box)
        self.bars = {}
        for label in ("left", "right", "blink"):   # "rest" (nothing happening) isn't shown
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            self.bars[label] = bar
            detection.addRow(f"{label}:", bar)
        self.threshold_label = QLabel()
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(50, 95)
        self.threshold.setValue(round(args.threshold * 100))
        self.threshold.valueChanged.connect(self.threshold_changed)
        self.threshold_changed(self.threshold.value())
        detection.addRow(self.threshold_label, self.threshold)

        # Manual control (also arrow keys and B / Space)
        manual_box = QGroupBox("Manual (also ← → keys, and B or Space to stop)")
        manual = QHBoxLayout(manual_box)
        for text, event in (("◀  Left", "left"), ("■  Stop", "blink"), ("Right  ▶", "right")):
            button = QPushButton(text)
            button.setMinimumHeight(40)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _=False, e=event: self.handle_event(e, None, "button"))
            manual.addWidget(button)

        # Pi
        pi_box = QGroupBox("Raspberry Pi")
        pi = QHBoxLayout(pi_box)
        self.pi_host = QLineEdit(args.pi)
        self.pi_port = QSpinBox(minimum=1, maximum=65535, value=args.cmd_port)
        self.pi_status = QLabel()
        for widget in (self.pi_host, self.pi_port):
            widget.editingFinished.connect(self.reconnect_pi)
        for w in (QLabel("Address:"), self.pi_host, QLabel("Port:"), self.pi_port, self.pi_status):
            pi.addWidget(w)

        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setFont(QFont("Consolas", 9))

        layout = QVBoxLayout()
        for widget in (self.headset_status, self.head_label, self.state_label, self.last_event,
                       detection_box, manual_box, pi_box):
            layout.addWidget(widget)
        layout.addWidget(self.log_view, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        for key, event in ((Qt.Key_Left, "left"), (Qt.Key_Right, "right"),
                           (Qt.Key_B, "blink"), (Qt.Key_Space, "blink")):
            QShortcut(QKeySequence(key), self, activated=lambda e=event: self.handle_event(e, None, "key"))

        self.show_state()
        self.link = None
        self.reconnect_pi()
        self.detect_timer = QTimer(self, interval=int(STEP * 1000), timeout=self.detect)
        self.detect_timer.start()
        self.send_timer = QTimer(self, interval=100, timeout=self.send_state)
        self.send_timer.start()

    def log(self, text):
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self.log_view.appendPlainText(line)
        print(line, flush=True)

    def threshold_changed(self, value):
        self.threshold_label.setText(f"Certainty ≥ {value}%:")
        self.detector.threshold = value / 100

    def show_state(self):
        color = GREY if self.state == "stop" else GREEN
        self.state_label.setText(STATE_TEXT[self.state])
        self.state_label.setStyleSheet(f"font-size: 40px; font-weight: bold; color: {color};")

    # ---------------------------------------------------------------- detection

    def detect(self):
        if self.detector is None:
            return
        now = time.monotonic()
        try:
            if self.tilt is not None:   # as in live.py
                change = self.tilt.update()
                pitch, roll = self.tilt.angles()
                if change:
                    self.log(f"HEAD {change.upper()} (forward/back {pitch:+.0f} deg, left/right {roll:+.0f} deg)")
                if self.tilt.rate > config.HEAD_MOTION_GATE:
                    self.head_until = now + config.EPOCH_END
                    self.detector.quiet_until = max(self.detector.quiet_until, self.head_until)
                moving = "  ·  head moving: eye events paused" if now < self.head_until else ""
                self.head_label.setText(f"Head: {self.tilt.state.upper()}  (forward/back {pitch:+.0f}°, "
                                        f"left/right {roll:+.0f}°){moving}")
            data = self.board.get_current_board_data(BUFFER_SECONDS * self.fs)
        except Exception as e:
            self.detector = None
            self.set_state("stop", "headset stopped")
            self.headset_status.setText(dot(RED, f"The headset stopped sending data ({e}). Close this "
                                                 "window, unplug and replug the board, and run it again."))
            self.log(f"Headset stopped: {e}")
            return
        if data.shape[1] < self.detector.n + 3 * self.fs:
            return  # warming up, as in live.py
        probs, event = self.detector.update(data[self.rows], now)
        for label, bar in self.bars.items():
            bar.setValue(round(100 * probs.get(label, 0)))
        if event:
            self.handle_event(event[0], event[1], "eyes")

    # ---------------------------------------------------------------- motor

    def handle_event(self, event, p, source):
        detail = f" (p={p:.2f})" if p is not None else f" ({source})"
        self.last_event.setText(f"Last: {event.upper()}{detail} at {time.strftime('%H:%M:%S')}")
        self.log(f"{event.upper()}{detail}")
        self.set_state(EVENT_TO_STATE[event], f"{event}{detail}")

    def set_state(self, state, why):
        if state != self.state:
            self.state = state
            self.log(f"Motor: {STATE_TEXT[state].strip('◀▶■ ').lower()}  <- {why}")
            self.show_state()

    def reconnect_pi(self):
        if self.link is not None:
            self.link.send({"cmd": "stop"})
            self.link.close()
        self.link = PiLink(self.pi_host.text().strip(), self.pi_port.value())

    def send_state(self):
        self.link.send({"cmd": self.state})
        self.pi_status.setText(dot(GREEN, "connected") if self.link.connected else dot(RED, "not connected"))

    def closeEvent(self, event):
        self.detect_timer.stop()
        self.send_timer.stop()
        self.link.send({"cmd": "stop"})
        self.link.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--no-imu", action="store_true", help="skip head tilt")
    parser.add_argument("--pi", default="raspberrypi.local", help="the Pi's IP address or hostname")
    parser.add_argument("--cmd-port", type=int, default=5005, help="the Pi's command server port")
    args = parser.parse_args()

    # As in live.py: load the model, connect once, and release the board on the way out.
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    print("Connecting to the board (it restarts when the port opens)...", flush=True)
    board, board_id = eog_board.connect(args.port, args.synthetic)
    fs = BoardShim.get_sampling_rate(board_id)
    if fs != saved["fs"]:
        print(f"Warning: model trained at {saved['fs']} Hz but board runs at {fs} Hz")
    try:
        tilt = None
        if not (args.no_imu or args.synthetic):   # as in live.py: calibrate head tilt at startup
            tilt = HeadTilt(board, board_id)
            tilt.calibrate()
        app = QApplication(sys.argv)
        window = MainWindow(board, board_id, model, args, tilt)
        window.show()
        app.exec()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
