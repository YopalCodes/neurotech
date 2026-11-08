"""A window showing a connected device's screen, driven with this computer's mouse and keyboard.

The Qt version of remote/controller.py's screen view, for the desktop app. The device comes
from remote/devices.py (an Android phone over ADB, or a computer running remote/agent.py).

Phones: click = tap, drag = swipe, right-click = Back, middle-click = Home, wheel = scroll.
"""

import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

KEYS = {
    Qt.Key_Return: "enter", Qt.Key_Enter: "enter", Qt.Key_Backspace: "backspace", Qt.Key_Tab: "tab",
    Qt.Key_Escape: "esc", Qt.Key_Delete: "delete", Qt.Key_Space: "space", Qt.Key_Up: "up",
    Qt.Key_Down: "down", Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Home: "home",
    Qt.Key_End: "end", Qt.Key_PageUp: "page_up", Qt.Key_PageDown: "page_down", Qt.Key_Insert: "insert",
    Qt.Key_Shift: "shift", Qt.Key_Control: "ctrl", Qt.Key_Alt: "alt", Qt.Key_Meta: "cmd",
    Qt.Key_CapsLock: "caps_lock", **{getattr(Qt, f"Key_F{i}"): f"f{i}" for i in range(1, 13)},
}
BUTTONS = {Qt.LeftButton: "left", Qt.RightButton: "right", Qt.MiddleButton: "middle"}
MOVE_INTERVAL = 0.015   # s; cap pointer-move messages at ~60 per second


def key_name(event):
    if event.key() in KEYS:
        return KEYS[event.key()]
    if Qt.Key_A <= event.key() <= Qt.Key_Z and event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
        return chr(event.key()).lower()   # Ctrl+C etc.: text() would be a control character
    text = event.text()
    return text if len(text) == 1 and text.isprintable() else None


class ScreenView(QWidget):
    """The device's screen, scaled to fit. Forwards mouse and keyboard to the device."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 240)
        self._image = self._array = None
        self.shown = -1
        self.last_move = 0.0

    def poll(self):
        d = self.device
        if d.frame is not None and d.frame_id != self.shown:
            self.shown = d.frame_id
            self._array = np.ascontiguousarray(np.asarray(d.frame.convert("RGB")))
            h, w = self._array.shape[:2]
            self._image = QImage(self._array.data, w, h, self._array.strides[0], QImage.Format.Format_RGB888)
            self.update()

    def _rect(self):
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        return QRectF((self.width() - iw * scale) / 2, (self.height() - ih * scale) / 2, iw * scale, ih * scale)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self._image is None:
            p.setPen(QColor("#bbb"))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for the screen...")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(self._rect(), self._image)
        if self.hasFocus():
            p.setPen(QColor("#4a9eff"))
            p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _pos(self, event):
        if self._image is None:
            return None
        r = self._rect()
        x, y = (event.position().x() - r.left()) / r.width(), (event.position().y() - r.top()) / r.height()
        return (x, y) if 0 <= x <= 1 and 0 <= y <= 1 else None

    def mouseMoveEvent(self, event):
        pos, now = self._pos(event), time.monotonic()
        if pos and now - self.last_move >= MOVE_INTERVAL:
            self.device.pointer_move(*pos)
            self.last_move = now

    def mousePressEvent(self, event):
        self.setFocus()
        pos = self._pos(event)
        if pos and event.button() in BUTTONS:
            self.device.pointer_down(*pos, BUTTONS[event.button()])

    def mouseReleaseEvent(self, event):
        pos = self._pos(event)
        if pos and event.button() in BUTTONS:
            self.device.pointer_up(*pos, BUTTONS[event.button()])

    def wheelEvent(self, event):
        pos = self._pos(event)
        if pos and event.angleDelta().y():
            self.device.scroll(1 if event.angleDelta().y() > 0 else -1, *pos)

    def keyPressEvent(self, event):
        key = key_name(event)
        if key:
            self.device.key_down(key)

    def keyReleaseEvent(self, event):
        key = key_name(event)
        if key and not event.isAutoRepeat():
            self.device.key_up(key)

    def focusNextPrevChild(self, forward):
        return False   # Tab goes to the device, not to the next button


class DeviceWindow(QWidget):
    closed = Signal(str)   # the device name, when the window is closed

    def __init__(self, device, manager):
        super().__init__()
        self.device, self.manager = device, manager
        self.setWindowTitle(f"{device.name} — device control")
        self.resize(520, 900) if getattr(device, "kind", "") == "android" else self.resize(1100, 700)

        toolbar = QHBoxLayout()
        for name in device.actions:
            button = QPushButton(name.replace("_", " ").title())
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _=False, n=name: device.action(n))
            toolbar.addWidget(button)
        for text, slot in (("Open app...", self.open_app), ("Type text...", self.type_text)):
            button = QPushButton(text)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        toolbar.addStretch(1)

        self.screen_view = ScreenView(device)
        self.status = QLabel("Click the screen to give it the keyboard.")
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.screen_view, 1)
        layout.addWidget(self.status)
        self.timer = QTimer(self, interval=15, timeout=self.poll)
        self.timer.start()

    def poll(self):
        self.screen_view.poll()
        if not self.device.connected:
            self.status.setText(f"<span style='color:#d64545'>Disconnected{': ' + self.device.error if self.device.error else ''}</span>")

    def open_app(self):
        if getattr(self.device, "kind", "") == "android":
            apps = self.device.list_apps()
            app, ok = QInputDialog.getItem(self, "Open app", "App:", apps, 0, False)
        else:
            app, ok = QInputDialog.getText(self, "Open app", "App name (as typed in the Start menu):")
        if ok and app:
            self.device.open_app(app)

    def type_text(self):
        text, ok = QInputDialog.getText(self, "Type text", "Text to type on the device:")
        if ok and text:
            self.device.type_text(text)

    def closeEvent(self, event):
        self.timer.stop()
        self.closed.emit(self.device.name)
        event.accept()
