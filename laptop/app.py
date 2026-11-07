"""Desktop app: live video from the Pi with AI detections, status, and controls, plus a
Teach tab for training the camera to recognise your own objects.

    python laptop/app.py --pi 192.168.1.50
    python laptop/app.py --source 0        # use this laptop's webcam instead of the Pi
"""

import argparse
import logging
import queue
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPlainTextEdit, QPushButton, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from engine import Engine, describe
from library import DEFAULT_ROOT, Library
from teach import TeachPanel
from video_view import VideoView

GREEN, AMBER, RED, GREY = "#2e9e4f", "#d49a00", "#d64545", "#888888"


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


class QueueLogHandler(logging.Handler):
    """Hands log lines to the GUI thread, which shows them in the log panel."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))


class MainWindow(QMainWindow):
    def __init__(self, engine, log_queue, library):
        super().__init__()
        self.engine = engine
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
        status_box = QGroupBox("Status")
        form = QFormLayout(status_box)
        form.addRow("Video:", self.video_status)
        form.addRow("Pi commands:", self.pi_status)
        form.addRow("AI:", self.ai_status)
        form.addRow("Speed:", self.speed_status)

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

        hint = QLabel("Esc turns AI control off.")
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
        self.tabs = QTabWidget()
        self.tabs.addTab(drive, "Drive")
        self.tabs.addTab(self.teach, "Teach")
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

    def set_manual(self, name):
        self.engine.manual_cmd = name

    def escape(self):
        self.ai_button.setChecked(False)
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

        if e.link is None or not e.link.connected:
            self.motor_status.setText(
                f"<span style='color:{RED}'>Pi not connected, so the motor can't move.</span> "
                f"<span style='color:{GREY}'>Run command_server.py on the Pi, then open this app "
                f"with --pi &lt;the Pi's IP&gt;.</span>")
        elif e.manual_cmd and e.sent and e.sent[1] == "manual":
            self.motor_status.setText(f"<b style='color:{GREEN}'>Spinning {e.manual_cmd}...</b>")
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
        self.teach.shutdown()
        self.engine.stop()
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
    return parser.parse_args()


def main():
    args = parse_args()

    log_queue = queue.Queue()
    gui_handler = QueueLogHandler(log_queue)
    gui_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[console, gui_handler])

    app = QApplication(sys.argv)
    library = Library(args.library)
    engine = Engine(args.pi, args.stream_port, args.cmd_port, args.source,
                    args.model, args.imgsz, args.conf, custom_model=library.model_path)
    window = MainWindow(engine, log_queue, library)
    window.show()
    engine.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
