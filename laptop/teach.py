"""The Teach tab: build a library of your own objects and train the camera to recognise them.

1. Add an object by name.
2. Take photos of it: each photo freezes the frame so you can drag a box around the object.
3. Press Train. When training finishes, the app starts recognising the object straight away.

Training runs train.py as a separate process, so the window stays responsive and a training
crash can't take the app down with it.
"""

import codecs
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from library import GOOD_PHOTOS, MIN_PHOTOS
from train import WORK_DIR

log = logging.getLogger("teach")

HERE = Path(__file__).resolve().parent
GREY = "#888888"


def button(text, slot, height=30):
    b = QPushButton(text)
    b.setMinimumHeight(height)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.clicked.connect(slot)
    return b


class TeachPanel(QWidget):
    def __init__(self, library, engine, video):
        super().__init__()
        self.library = library
        self.engine = engine
        self.video = video
        self.last_box = {}   # object name -> box from its previous photo, offered again for the next one
        self.process = None

        # Objects
        self.objects = QListWidget()
        self.objects.setMinimumHeight(90)
        self.objects.currentItemChanged.connect(lambda *_: self.update_buttons())
        self.new_name = QLineEdit(placeholderText="New object, e.g. red mug")
        self.new_name.returnPressed.connect(self.add_object)
        add_row = QHBoxLayout()
        add_row.addWidget(self.new_name, 1)
        add_row.addWidget(button("Add", self.add_object, 26))
        self.delete_button = button("Delete object", self.delete_object, 26)
        tools_row = QHBoxLayout()
        tools_row.addWidget(self.delete_button)
        tools_row.addWidget(button("Open folder", self.open_folder, 26))

        objects_box = QGroupBox("Your objects")
        layout = QVBoxLayout(objects_box)
        layout.addWidget(self.objects, 1)
        layout.addLayout(add_row)
        layout.addLayout(tools_row)

        # Photos
        self.photo_hint = QLabel()
        self.photo_hint.setWordWrap(True)
        self.photo_hint.setStyleSheet(f"color: {GREY};")
        self.take_button = button("Take photo  (Space)", self.take_photo, 36)
        self.save_button = button("Save  (Enter)", self.save_photo)
        self.cancel_button = button("Cancel  (Esc)", self.cancel_photo)
        self.undo_button = button("Undo last photo", self.undo_photo)
        grid = QGridLayout()
        grid.addWidget(self.take_button, 0, 0, 1, 2)
        grid.addWidget(self.save_button, 1, 0)
        grid.addWidget(self.cancel_button, 1, 1)
        grid.addWidget(self.undo_button, 2, 0, 1, 2)

        self.photos_box = QGroupBox("Photos")
        layout = QVBoxLayout(self.photos_box)
        layout.addWidget(self.photo_hint)
        layout.addLayout(grid)

        # Training
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.train_button = button("Train", self.train, 36)
        self.stop_button = button("Stop training", self.stop_training, 36)
        self.stop_button.setVisible(False)

        train_box = QGroupBox("Recognition model")
        layout = QVBoxLayout(train_box)
        layout.addWidget(self.model_status)
        layout.addWidget(self.progress)
        layout.addWidget(self.train_button)
        layout.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(objects_box, 1)
        layout.addWidget(self.photos_box)
        layout.addWidget(train_box)

        video.box_changed.connect(self.update_buttons)
        self.refresh()

    # ---- state ----

    def current_name(self):
        item = self.objects.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh(self, select=None):
        """Re-reads the library from disk and updates everything shown."""
        select = select or self.current_name()
        self.objects.blockSignals(True)
        self.objects.clear()
        for name, count in self.library.objects().items():
            label = f"{name}  —  {count} photo{'s' if count != 1 else ''}"
            if count < MIN_PHOTOS:
                label += f"  (needs {MIN_PHOTOS - count} more)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.objects.addItem(item)
            if name == select:
                self.objects.setCurrentItem(item)
        if self.objects.currentItem() is None and self.objects.count():
            self.objects.setCurrentRow(0)
        self.objects.blockSignals(False)
        self.update_buttons()
        self.update_model_status()

    def update_buttons(self):
        name = self.current_name()
        frozen = self.video.frozen is not None
        self.photos_box.setTitle(f"Photos of: {name}" if name else "Photos")
        self.delete_button.setEnabled(bool(name) and not frozen)
        self.take_button.setEnabled(bool(name) and not frozen)
        self.save_button.setEnabled(frozen and self.video.box is not None)
        self.cancel_button.setEnabled(frozen)
        self.undo_button.setEnabled(bool(name) and not frozen and self.library.objects().get(name, 0) > 0)
        self.objects.setEnabled(not frozen)

        if not name:
            hint = "Type a name for something you want the camera to learn, then press Add."
        elif frozen and self.video.box is None:
            hint = "Drag a box tightly around the object on the video."
        elif frozen:
            hint = "Press Save, or drag again to redo the box."
        else:
            hint = (f"Take {GOOD_PHOTOS}+ photos: turn the object, move it nearer and further, "
                    "change the background and lighting. Keep your other taught objects out of view.")
        self.photo_hint.setText(hint)

    def update_model_status(self):
        if self.process is not None:
            return   # training progress is shown instead
        info = self.library.model_info()
        if info is None:
            text = (f"Not trained yet. Give each object at least {MIN_PHOTOS} photos "
                    f"({GOOD_PHOTOS}+ is better), then press Train.")
        else:
            classes = ", ".join(info.get("classes", [])) or "?"
            score = info.get("score_map50")
            text = f"Recognises: {classes}"
            if info.get("trained_at"):
                text += f"\nTrained {info['trained_at']}"
            if score is not None:
                text += f", score {score:.2f} ({quality(score)})"
            if self.library.model_out_of_date():
                text += "\n⚠ Photos changed since then. Train again to use them."
        self.model_status.setText(text)
        self.train_button.setEnabled(bool(self.library.trainable()))

    # ---- objects ----

    def add_object(self):
        try:
            name = self.library.add_object(self.new_name.text())
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Can't add object", str(e))
            return
        self.new_name.clear()
        self.new_name.clearFocus()
        log.info("Added object %r. Now take photos of it.", name)
        self.refresh(select=name)

    def delete_object(self):
        name = self.current_name()
        if not name:
            return
        count = self.library.objects().get(name, 0)
        answer = QMessageBox.question(
            self, "Delete object",
            f"Delete {name!r} and its {count} photos?\n\nThis can't be undone. "
            "If it's in the trained model, it stays recognised until you train again.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.library.delete_object(name)
        self.last_box.pop(name, None)
        log.info("Deleted object %r", name)
        self.refresh()

    def open_folder(self):
        self.library.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.library.root)))

    # ---- photos ----

    def take_photo(self):
        name = self.current_name()
        if not name or self.video.frozen is not None:
            return
        snap = self.engine.latest()
        stream = self.engine.stream
        if snap is None or stream is None or time.monotonic() - stream.last_frame_time > 1.0:
            QMessageBox.information(self, "No video", "There's no live video to take a photo from.")
            return
        self.video.freeze(snap.raw, self.last_box.get(name))

    def save_photo(self):
        name = self.current_name()
        if self.video.frozen is None or self.video.box is None or not name:
            return
        try:
            count = self.library.save_photo(name, self.video.frozen, self.video.box)
        except (ValueError, OSError, RuntimeError) as e:
            QMessageBox.warning(self, "Couldn't save the photo", str(e))
            return
        self.last_box[name] = self.video.box
        self.video.unfreeze()
        log.info("Saved photo %d of %r", count, name)
        self.refresh()

    def cancel_photo(self):
        if self.video.frozen is not None:
            self.video.unfreeze()

    def undo_photo(self):
        name = self.current_name()
        if name and self.library.delete_last_photo(name):
            log.info("Deleted the last photo of %r", name)
            self.refresh()

    # ---- training ----

    def train(self):
        if self.process is not None:
            return
        if not self.library.trainable():
            QMessageBox.information(self, "Not enough photos",
                                    f"Each object needs at least {MIN_PHOTOS} photos before training.")
            return
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.train_log = open(WORK_DIR / "train.log", "w", encoding="utf-8")
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending, self.tail, self.outcome, self.stopped = "", deque(maxlen=15), None, False

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")         # Ultralytics prints non-ASCII characters
        env.insert("PYTHONUNBUFFERED", "1")
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([str(HERE / "train.py"), "--library", str(self.library.root)])
        process.setWorkingDirectory(str(HERE.parent))   # downloaded weights land next to yolo11m.pt
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process

        self.train_button.setVisible(False)
        self.stop_button.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)   # busy until the first round finishes
        self.model_status.setText("Starting training... (the first time, this downloads a 19 MB model)")
        log.info("Training started. The video may slow down while the GPU is busy.")
        process.start()

    def stop_training(self):
        if self.process is not None:
            self.stopped = True
            self.process.kill()

    def shutdown(self):
        """Called when the window closes."""
        if self.video.frozen is not None:
            self.video.unfreeze()
        if self.process is not None:
            self.stopped = True
            self.process.kill()
            self.process.waitForFinished(3000)

    def _read_output(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.train_log.write(text)
        self.train_log.flush()
        *lines, self.pending = re.split(r"[\r\n]", self.pending + text)
        for line in lines:
            line = line.strip()
            if line.startswith("@@PROGRESS"):
                _, epoch, total, score = line.split()
                self.progress.setRange(0, int(total))
                self.progress.setValue(int(epoch))
                self.model_status.setText(f"Training: round {epoch} of {total}, score {float(score):.2f}")
            elif line.startswith("@@DONE"):
                self.outcome = ("done", float(line.split()[1]))
            elif line.startswith("@@ERROR"):
                self.outcome = ("error", line[len("@@ERROR"):].strip())
            elif line.startswith("[train]"):
                log.info(line[len("[train]"):].strip())
            elif line:
                self.tail.append(line)

    def _finished(self, exit_code, exit_status):
        self._read_output()
        self.train_log.close()
        self.process.deleteLater()
        self.process = None
        self.progress.setVisible(False)
        self.stop_button.setVisible(False)
        self.train_button.setVisible(True)

        if self.outcome and self.outcome[0] == "done":
            score = self.outcome[1]
            log.info("Training finished: score %.2f (%s). Loading the new model...", score, quality(score))
            self.engine.reload_custom_model()
        elif self.stopped:
            log.info("Training stopped. The previous model (if any) is still in use.")
        else:
            reason = self.outcome[1] if self.outcome else f"exit code {exit_code}"
            log.error("Training failed: %s", reason)
            for line in self.tail:
                log.error("  %s", line)
            log.error("Full output: %s", WORK_DIR / "train.log")
            QMessageBox.warning(self, "Training failed",
                                f"{reason}\n\nThe full output is in:\n{WORK_DIR / 'train.log'}")
        self.update_model_status()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            log.error("Couldn't start training: %s", self.process.errorString())
            self._finished(-1, QProcess.ExitStatus.CrashExit)


def quality(score):
    if score >= 0.85:
        return "great"
    if score >= 0.6:
        return "good"
    return "weak: add more varied photos"
