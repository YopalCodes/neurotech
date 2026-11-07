"""The video panel. Shows live frames, or a frozen frame you drag a box on to teach an object."""

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

BOX_COLOR = QColor("#ffd400")
HINT = "Drag a box tightly around the object  ·  Enter = save  ·  Esc = cancel"


class VideoView(QWidget):
    box_changed = Signal()   # also sent when a frame is frozen or unfrozen

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._image = None      # QImage on screen
        self._array = None      # keeps the numpy buffer behind _image alive
        self._message = "Starting..."
        self.frozen = None      # BGR frame being labelled, or None while showing live video
        self.box = None         # (x1, y1, x2, y2) as fractions of the image size
        self._anchor = None     # where the current drag started

    # ---- what to show ----

    def show_message(self, text):
        if self._message != text or self._image is not None:
            self._message, self._image, self._array = text, None, None
            self.update()

    def show_image(self, bgr):
        self._message = None
        self._set_image(bgr)
        self.update()

    def freeze(self, bgr, box=None):
        """Holds one frame on screen so a box can be drawn on it. box is a starting box, or None."""
        self.frozen, self.box, self._anchor = bgr, box, None
        self._message = None
        self._set_image(bgr)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.box_changed.emit()
        self.update()

    def unfreeze(self):
        self.frozen, self.box, self._anchor = None, None, None
        self.unsetCursor()
        self.box_changed.emit()
        self.update()

    def _set_image(self, bgr):
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        self._array = bgr
        self._image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)

    # ---- drawing ----

    def _image_rect(self):
        """Where the image sits inside the widget: as large as fits, keeping its shape."""
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if self._image is None:
            font = p.font()
            font.setPointSize(11)
            p.setFont(font)
            p.setPen(QColor("#bbb"))
            p.drawText(self.rect().adjusted(20, 20, -20, -20),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message or "")
            return

        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self._image_rect()
        p.drawImage(target, self._image)
        if self.frozen is None:
            return

        if self.box is not None:
            x1, y1, x2, y2 = self.box
            box = QRectF(target.left() + x1 * target.width(), target.top() + y1 * target.height(),
                         (x2 - x1) * target.width(), (y2 - y1) * target.height())
            outside = QPainterPath()
            outside.addRect(target)
            outside.addRect(box)   # the default odd-even fill leaves the box itself clear
            p.fillPath(outside, QColor(0, 0, 0, 110))
            p.setPen(QPen(BOX_COLOR, 2))
            p.drawRect(box)

        banner = QRectF(target.left(), target.top(), target.width(), 30)
        p.fillRect(banner, QColor(0, 0, 0, 170))
        p.setPen(QColor("white"))
        p.drawText(banner, Qt.AlignmentFlag.AlignCenter, HINT)

    # ---- dragging a box ----

    def _to_image(self, pos):
        t = self._image_rect()
        x = (pos.x() - t.left()) / t.width()
        y = (pos.y() - t.top()) / t.height()
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def mousePressEvent(self, event):
        if self.frozen is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._anchor = self._to_image(event.position())
        self.box = None
        self.update()

    def mouseMoveEvent(self, event):
        if self._anchor is None:
            return super().mouseMoveEvent(event)
        (ax, ay), (x, y) = self._anchor, self._to_image(event.position())
        self.box = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
        self.update()

    def mouseReleaseEvent(self, event):
        if self._anchor is None:
            return super().mouseReleaseEvent(event)
        self.mouseMoveEvent(event)
        self._anchor = None
        if self.box is not None and (self.box[2] - self.box[0] < 0.01 or self.box[3] - self.box[1] < 0.01):
            self.box = None   # a click, not a drag
        self.box_changed.emit()
        self.update()
