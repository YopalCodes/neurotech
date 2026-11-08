"""Desktop app: live video from the Pi with AI detections, status, and controls, plus a
Teach tab for training the camera to recognise your own objects, an Eyes tab for steering
the motor with the NeuroPawn EOG headset, and a Devices tab for controlling other computers
and phones (remote/).

    python laptop/app.py --pi 192.168.1.50
    python laptop/app.py --source 0        # use this laptop's webcam instead of the Pi
    python laptop/app.py --eog-port COM10  # also connect the EOG headset (Eyes tab)
    python laptop/app.py --eog-port COM10 --dog   # and drive the MechDog with head tilt (Dog tab)
"""

import argparse
import logging
import queue
import sys
import time

import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPlainTextEdit, QPushButton, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from engine import Engine, describe
from eye_control import EyeControl, EyePanel
from devices_panel import DevicesPanel
from dog_panel import DogController, DogPanel
from library import DEFAULT_ROOT, Library
from teach import TeachPanel
from video_view import VideoView

GREEN, AMBER, RED, GREY = "#2e9e4f", "#d49a00", "#d64545", "#888888"


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


HOMING_TIMEOUT = 60   # seconds; a full turn back takes about 8 s at the default speed


def facing_text(angle):
    """35.2 -> '35° right of forward'."""
    if angle is None:
        return "unknown"
    if abs(angle) < 0.5:
        return "forward"
    return f"{abs(angle):.0f}° {'right' if angle > 0 else 'left'} of forward"


class StepperDial(QWidget):
    """A small dial: the tick at the top is forward, the arrow is where the stepper points."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(64, 64)
        self.angle = None

    def set_angle(self, angle):
        if angle != self.angle:
            self.angle = angle
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = min(self.width(), self.height()) / 2 - 4
        text = self.palette().windowText().color()
        p.setPen(QPen(QColor(GREY), 2))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(text, 3))
        p.drawLine(QPointF(c.x(), c.y() - r - 3), QPointF(c.x(), c.y() - r + 6))   # forward tick
        if self.angle is None:
            return
        a = math.radians(self.angle)   # 0 = up (forward), positive = clockwise (right)
        tip = QPointF(c.x() + math.sin(a) * (r - 6), c.y() - math.cos(a) * (r - 6))
        arrow = QPen(QColor(GREEN), 4)
        arrow.setCapStyle(Qt.RoundCap)
        p.setPen(arrow)
        p.drawLine(c, tip)


class QueueLogHandler(logging.Handler):
    """Hands log lines to the GUI thread, which shows them in the log panel."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))


class MainWindow(QMainWindow):
    def __init__(self, engine, log_queue, library, eyes=None, dog=None):
        super().__init__()
        self.engine = engine
        self.eyes = eyes
        self.dog = dog
        self.log_queue = log_queue
        self.shown_id = None
        self.names_version = 0
        self.setWindowTitle("Neurotech Vision")
        self.resize(1280, 800)

        # Left: video above, log below (drag the divider to resize).
        self.video = VideoView()

        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))

        left = QSplitter(Qt.Vertical)
        left.addWidget(self.video)
        left.addWidget(self.log_view)
        left.setSizes([620, 160])

        # Right: status, detections, controls.
        self.video_status = QLabel()
        self.pi_status = QLabel()
        self.ai_status = QLabel()
        self.ai_status.setWordWrap(True)
        self.speed_status = QLabel()
        self.stepper_status = QLabel()
        status_box = QGroupBox("Status")
        form = QFormLayout(status_box)
        form.addRow("Video:", self.video_status)
        form.addRow("Pi commands:", self.pi_status)
        form.addRow("AI:", self.ai_status)
        form.addRow("Speed:", self.speed_status)
        form.addRow("Stepper:", self.stepper_status)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Object", "Confidence"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        detections_box = QGroupBox("Detections (target in bold)")
        QVBoxLayout(detections_box).addWidget(self.table)

        self.sending = QLabel()
        font = self.sending.font()
        font.setPointSize(13)
        font.setBold(True)
        self.sending.setFont(font)

        self.ai_button = QPushButton()
        self.ai_button.setCheckable(True)
        self.ai_button.setMinimumHeight(40)
        self.ai_button.setFocusPolicy(Qt.NoFocus)
        self.ai_button.toggled.connect(self.set_ai)
        self.set_ai(False)

        self.target = QComboBox()
        self.target.setEnabled(False)
        self.target.addItem("(loading model...)")
        self.target.currentTextChanged.connect(self.set_target)

        self.conf_label = QLabel()
        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(5, 95)
        self.conf.valueChanged.connect(self.set_conf)
        self.conf.setValue(round(engine.conf * 100))
        self.set_conf(self.conf.value())

        # Stepper motor: hold a button to spin (sends left/right to the Pi), release to stop.
        motor_box = QGroupBox("Stepper motor")
        motor = QVBoxLayout(motor_box)
        motor_buttons = QHBoxLayout()
        for text, command in (("◀  Spin left", "left"), ("Spin right  ▶", "right")):
            button = QPushButton(text)
            button.setMinimumHeight(48)
            button.setFocusPolicy(Qt.NoFocus)
            button.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; }"
                                 f"QPushButton:pressed {{ background: {GREEN}; color: white; }}")
            button.pressed.connect(lambda c=command: self.set_manual(c))
            button.released.connect(lambda: self.set_manual(None))
            motor_buttons.addWidget(button)
        motor.addLayout(motor_buttons)
        self.motor_status = QLabel()
        self.motor_status.setWordWrap(True)
        motor.addWidget(self.motor_status)

        # Which way the stepper points (the Pi counts its steps; forward = 0).
        facing = QHBoxLayout()
        self.dial = StepperDial()
        facing.addWidget(self.dial)
        self.facing = QLabel()
        self.facing.setWordWrap(True)
        self.facing.setStyleSheet("font-weight: bold;")
        facing.addWidget(self.facing, 1)
        motor.addLayout(facing)
        facing_buttons = QHBoxLayout()
        self.zero_button = QPushButton("Set forward here")
        self.zero_button.setFocusPolicy(Qt.NoFocus)
        self.zero_button.setToolTip("The stepper's current direction becomes forward (0°)")
        self.zero_button.clicked.connect(self.set_forward)
        self.home_button = QPushButton("↺ Return to forward")
        self.home_button.setFocusPolicy(Qt.NoFocus)
        self.home_button.setToolTip("Spins back to forward (0°) the way it came, then stops")
        self.home_button.clicked.connect(self.return_forward)
        facing_buttons.addWidget(self.zero_button)
        facing_buttons.addWidget(self.home_button)
        motor.addLayout(facing_buttons)
        self.homing_until = 0.0

        hint = QLabel("Esc turns AI control and eye control off.")
        hint.setStyleSheet(f"color: {GREY};")

        control_box = QGroupBox("Control")
        control = QVBoxLayout(control_box)
        control.addWidget(QLabel("Sending to Pi:"))
        control.addWidget(self.sending)
        control.addWidget(self.ai_button)
        settings = QFormLayout()
        settings.addRow("Target:", self.target)
        settings.addRow(self.conf_label, self.conf)
        control.addLayout(settings)
        control.addWidget(hint)

        drive = QWidget()
        drive_layout = QVBoxLayout(drive)
        drive_layout.setContentsMargins(0, 0, 0, 0)
        drive_layout.addWidget(motor_box)
        drive_layout.addWidget(detections_box, 1)
        drive_layout.addWidget(control_box)

        self.teach = TeachPanel(library, engine, self.video)
        self.eye_panel = EyePanel(eyes)
        self.eye_panel.toggle.toggled.connect(self.set_eyes)
        self.tabs = QTabWidget()
        self.tabs.addTab(drive, "Drive")
        self.tabs.addTab(self.teach, "Teach")
        self.tabs.addTab(self.eye_panel, "Eyes")
        self.dog_panel = DogPanel(dog, eyes)
        self.dog_panel.toggle.toggled.connect(self.set_dog)
        self.dog_panel.stop_button.clicked.connect(self.stop_dog)
        self.tabs.addTab(self.dog_panel, "Dog")
        self.devices_panel = DevicesPanel()
        self.tabs.addTab(self.devices_panel, "Devices")
        self.tabs.currentChanged.connect(lambda _: self.teach.cancel_photo())

        side = QWidget()
        side.setFixedWidth(340)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(status_box)
        side_layout.addWidget(self.tabs, 1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(left, 1)
        layout.addWidget(side)
        self.setCentralWidget(central)

        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.escape)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.space)
        self.save_shortcuts = [QShortcut(QKeySequence(key), self, activated=self.teach.save_photo)
                               for key in (Qt.Key_Return, Qt.Key_Enter)]
        self.video.box_changed.connect(self.update_shortcuts)
        self.update_shortcuts()

        self.timer = QTimer(self, interval=33, timeout=self.refresh)
        self.timer.start()

    # ---- controls ----

    def set_ai(self, on):
        self.engine.ai_enabled = on
        if on and hasattr(self, "eye_panel"):
            self.eye_panel.toggle.setChecked(False)   # only one of AI and eyes steers the motor
        if on:
            self.ai_button.setText("AI control: ON  (click or Esc to stop)")
            self.ai_button.setStyleSheet(f"background: {GREEN}; color: white; font-weight: bold;")
        else:
            self.ai_button.setText("AI control: OFF  (click to start)")
            self.ai_button.setStyleSheet("")

    def set_target(self, name):
        if self.target.isEnabled():
            self.engine.target = name

    def set_conf(self, value):
        self.engine.conf = value / 100
        self.conf_label.setText(f"Confidence ≥ {value / 100:.2f}:")

    def set_eyes(self, on):
        if on:
            self.eyes.state = "stop"        # start still: only a new look left/right moves it
            self.ai_button.setChecked(False)
        self.engine.eye_control = on

    def set_dog(self, on):
        self.dog.set_enabled(on)

    def stop_dog(self):
        self.dog_panel.toggle.setChecked(False)
        if self.dog is not None:
            self.dog.stop()

    def set_forward(self):
        """'Set forward here': the stepper's current direction becomes 0°."""
        if self.engine.link is not None:
            self.engine.link.send({"cmd": "zero"})

    def return_forward(self):
        """'Return to forward': sends 'home' like a held button until the Pi reports it's back at
        0°, so the Pi's failsafe still stops it if the laptop or Wi-Fi drops."""
        self.homing_until = time.monotonic() + HOMING_TIMEOUT
        self.engine.manual_cmd = "home"

    def stepper_angle(self):
        """The stepper's angle from the Pi's latest status, or None if there's no recent one."""
        link = self.engine.link
        if link is None or not link.connected or time.monotonic() - link.status_time > 1.0:
            return None
        return link.status.get("stepper_angle")

    def set_manual(self, name):
        self.homing_until = 0.0   # a Spin button cancels Return to forward
        self.engine.manual_cmd = name
        if name is None and self.engine.eye_control:
            self.eyes.state = "stop"        # after a held Spin button, stay stopped

    def escape(self):
        self.ai_button.setChecked(False)
        self.eye_panel.toggle.setChecked(False)
        self.stop_dog()
        if self.engine.manual_cmd == "home":
            self.set_manual(None)   # also stops Return to forward
        self.teach.cancel_photo()

    def space(self):
        if self.tabs.currentWidget() is self.teach:
            self.teach.take_photo()

    def update_shortcuts(self):
        # Enter saves a photo only while one is frozen, so it still works in text boxes otherwise.
        for shortcut in self.save_shortcuts:
            shortcut.setEnabled(self.video.frozen is not None)

    # ---- display ----

    def fill_targets(self):
        """Lists the taught objects first, then the built-in ones."""
        e = self.engine
        custom = list(e.custom_names)
        built_in = [n for n in e.class_names if n not in custom]
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItems(custom)
        if custom and built_in:
            self.target.insertSeparator(len(custom))
        self.target.addItems(built_in)
        self.target.setCurrentText(e.target)
        self.target.blockSignals(False)
        self.target.setEnabled(True)

    def refresh(self):
        e = self.engine
        for _ in range(200):
            try:
                self.log_view.appendPlainText(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if e.names_version != self.names_version:
            self.names_version = e.names_version
            self.fill_targets()
        self.eye_panel.refresh()
        self.dog_panel.refresh()

        now = time.monotonic()
        video_live = e.stream is not None and now - e.stream.last_frame_time < 1.0
        snap = e.latest()

        if e.error:
            self.show_message(f"Error: {e.error}\n\nSee the log below.")
        elif not e.ready:
            self.show_message(e.status)
        elif not video_live:
            self.show_message(f"Waiting for video from {e.source} ...\n\n"
                              "If this is the Pi, is start_stream.sh running on it?")
        elif snap is not None and snap.id != self.shown_id:
            self.show_frame(snap)

        if video_live:
            self.video_status.setText(dot(GREEN, "live"))
        elif e.stream is not None:
            self.video_status.setText(dot(RED, "no signal"))
        else:
            self.video_status.setText(dot(AMBER, "starting"))

        if e.link is None:
            self.pi_status.setText(dot(AMBER, "starting"))
        elif e.link.connected:
            self.pi_status.setText(dot(GREEN, f"connected ({e.pi})"))
        else:
            self.pi_status.setText(dot(RED, f"not connected ({e.pi}:{e.cmd_port})"))

        if e.error:
            self.ai_status.setText(dot(RED, "crashed"))
        elif e.ready:
            taught = f" + {len(e.custom_names)} taught" if e.custom_names else ""
            self.ai_status.setText(dot(GREEN, f"{e.model_name}{taught} on {e.device_name}"))
        else:
            self.ai_status.setText(dot(AMBER, e.status))

        if video_live and snap is not None:
            self.speed_status.setText(f"{snap.fps:.1f} fps · AI {snap.ai_ms:.0f} ms/frame")
        else:
            self.speed_status.setText("–")

        angle = self.stepper_angle()
        self.dial.set_angle(angle)
        self.zero_button.setEnabled(angle is not None)
        self.home_button.setEnabled(angle is not None)
        if angle is not None:
            doing = e.link.status.get("stepper", "")
            self.facing.setText(f"Facing: {facing_text(angle)}")
            self.stepper_status.setText(facing_text(angle) + (f" · {doing}" if doing and doing != "stopped" else ""))
        elif e.link is not None and e.link.connected:
            self.facing.setText("Facing: unknown")
            self.stepper_status.setText(f"<span style='color:{AMBER}'>position unknown: update the Pi's code</span>")
        else:
            self.facing.setText("Facing: unknown")
            self.stepper_status.setText("–")
        if self.homing_until:
            if e.manual_cmd != "home":
                self.homing_until = 0.0   # cancelled (Spin button or Esc)
            elif angle is None or now > self.homing_until or (
                    abs(angle) < 0.05 and e.link.status.get("stepper") == "stopped"):
                self.homing_until = 0.0
                self.engine.manual_cmd = None
                if e.eye_control:
                    self.eyes.state = "stop"

        if e.link is None or not e.link.connected:
            self.motor_status.setText(
                f"<span style='color:{RED}'>Pi not connected, so the motor can't move.</span> "
                f"<span style='color:{GREY}'>Run command_server.py on the Pi, then open this app "
                f"with --pi &lt;the Pi's IP&gt;.</span>")
        elif e.manual_cmd == "home":
            self.motor_status.setText(f"<b style='color:{GREEN}'>Returning to forward...</b>")
        elif e.manual_cmd and e.sent and e.sent[1] == "manual":
            self.motor_status.setText(f"<b style='color:{GREEN}'>Spinning {e.manual_cmd}...</b>")
        elif e.eye_control:
            doing = "stopped" if self.eyes.state == "stop" else f"spinning {self.eyes.state}"
            self.motor_status.setText(f"Your eyes are steering the motor (Eyes tab): <b>{doing}</b>. "
                                      "Hold a button to override.")
        elif e.ai_enabled:
            self.motor_status.setText("The AI is steering the motor. Hold a button to override it.")
        else:
            self.motor_status.setText(f"<span style='color:{GREY}'>Stopped. Hold a button to spin; "
                                      f"let go to stop.</span>")

        if e.link is None or not e.link.connected:
            self.sending.setText("–  (Pi not connected)")
            self.sending.setStyleSheet(f"color: {GREY};")
        elif e.sent is None or now - e.sent_time > 0.5:
            self.sending.setText("nothing: no video, Pi stopped")
            self.sending.setStyleSheet(f"color: {AMBER};")
        else:
            command, reason = e.sent
            self.sending.setText(f"{describe(command).upper()}  ({reason})")
            self.sending.setStyleSheet(f"color: {GREEN if command['cmd'] != 'stop' else GREY};")

    def show_message(self, text):
        self.shown_id = None
        if self.video.frozen is None:   # while teaching, keep the frozen photo on screen
            self.video.show_message(text)

    def show_frame(self, snap):
        self.shown_id = snap.id
        if self.video.frozen is None:
            self.video.show_image(snap.image)

        self.table.setRowCount(len(snap.detections))
        bold = QFont()
        bold.setBold(True)
        for row, (name, conf) in enumerate(snap.detections):
            name_item, conf_item = QTableWidgetItem(name), QTableWidgetItem(f"{conf:.2f}")
            if name == self.engine.target:
                name_item.setFont(bold)
                conf_item.setFont(bold)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, conf_item)

    def closeEvent(self, event):
        self.timer.stop()
        self.devices_panel.shutdown()   # closes device windows and disconnects
        self.teach.shutdown()
        self.engine.stop()
        if self.dog is not None:
            self.dog.close()    # sends the dog a stop, then disconnects
        if self.eyes is not None:
            self.eyes.close()   # stop_stream() + release_session(), as live.py does on exit
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="Live AI view of the Pi's camera, with control of the Pi.")
    parser.add_argument("--pi", default="aquapack", help="the Pi's IP address or hostname")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--cmd-port", type=int, default=5005)
    parser.add_argument("--source",
                        help="use another video source instead of the Pi: a webcam index like 0, or a URL")
    parser.add_argument("--model", default="yolo11m.pt",
                        help="YOLO weights, downloaded automatically the first time. Sizes from fastest "
                             "to most accurate: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="size the AI looks at; raise to 960 or 1280 (with a higher stream "
                             "resolution) to catch small or distant objects")
    parser.add_argument("--conf", type=float, default=0.5, help="starting confidence threshold")
    parser.add_argument("--library", default=str(DEFAULT_ROOT),
                        help="folder that stores the objects you teach (default: library/)")
    parser.add_argument("--eog-port",
                        help="the NeuroPawn EOG headset's serial port, e.g. COM10, for the Eyes tab")
    parser.add_argument("--no-imu", action="store_true",
                        help="skip the headset's head tilt (no calibration, no head-movement filter)")
    parser.add_argument("--dog", action="store_true",
                        help="connect to the MechDog (running mechdog/main.py) over Bluetooth for the Dog tab")
    parser.add_argument("--dog-dry-run", action="store_true",
                        help="Dog tab without a dog: shows the commands but sends nothing")
    return parser.parse_args()


def main():
    args = parse_args()

    log_queue = queue.Queue()
    gui_handler = QueueLogHandler(log_queue)
    gui_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[console, gui_handler])

    # The dog connects over Bluetooth in the background, while the headset connects below.
    dog = DogController(dry_run=args.dog_dry_run) if (args.dog or args.dog_dry_run) else None

    # The EOG headset connects once, here, before the window opens (as Aram's live.py does).
    eyes = None
    if args.eog_port:
        eyes = EyeControl(args.eog_port, use_imu=not args.no_imu)
        if dog is not None:
            eyes.on_tilt = dog.on_tilt   # head tilt steers the dog
        print(f"Connecting to the EOG headset on {args.eog_port} (the board restarts when its port "
              "opens)...", flush=True)
        if eyes.use_imu:
            print("Then head tilt calibration: follow the prompts that appear here.", flush=True)
        try:
            eyes.connect()
        except Exception as e:
            eyes.error = str(e)
            logging.getLogger("eyes").error(eyes.error)

    app = QApplication(sys.argv)
    library = Library(args.library)
    engine = Engine(args.pi, args.stream_port, args.cmd_port, args.source,
                    args.model, args.imgsz, args.conf, custom_model=library.model_path)
    engine.eyes = eyes
    window = MainWindow(engine, log_queue, library, eyes, dog)
    window.show()
    engine.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
