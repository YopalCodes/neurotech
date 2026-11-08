"""MechDog control for the desktop app: steer Aram's MechDog with head tilt.

    tilt forward     -> start walking (keeps walking after you return to neutral)
    tilt back        -> stop
    tilt left/right  -> steer while walking, more tilt = sharper turn

Uses Aram's eog/dog_control.py as it is: Steering (head tilt -> stride and turn) and DogLink
(Bluetooth LE to mechdog/main.py running on the dog), sending a command every HEARTBEAT
(0.2 s). The dog stops by itself if commands stop for 1 s, so it halts if the app crashes or
the laptop goes out of range.

Head tilt comes from the headset process (eye_control.py), so this needs --eog-port with the
IMU on. Start the app with --dog to connect to the dog (--dog-dry-run to try it without one).
"""

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

sys.path.append(str(Path(__file__).resolve().parent.parent / "eog"))
import config as eog_config  # noqa: E402  (Aram's eog/ modules)
from dog_control import HEARTBEAT, DogLink, Steering  # noqa: E402

log = logging.getLogger("dog")
GREEN, AMBER, RED, GREY = "#2e9e4f", "#d49a00", "#d64545", "#888888"


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


def tighter_turn(cmd):
    """Shortens the stride while turning, so the dog turns in a tighter circle: from DOG_STRIDE
    when walking straight down to DOG_TURN_STRIDE at a full DOG_MAX_TURN turn."""
    stride, angle = cmd
    if stride == 0 or angle == 0:
        return cmd
    shrink = (eog_config.DOG_STRIDE - eog_config.DOG_TURN_STRIDE) * abs(angle) / eog_config.DOG_MAX_TURN
    return (int(round(stride - shrink)), angle)


def describe(cmd):
    """(stride, angle) -> what the dog is doing, as dog_control.py prints it."""
    stride, angle = cmd
    if stride == 0:
        return "STOPPED"
    if angle == 0:
        return "WALKING"
    return f"TURNING {'LEFT' if angle > 0 else 'RIGHT'} {abs(angle)}°"


class DogController:
    """Bluetooth link to the dog in a background thread. The headset's reader thread calls
    on_tilt() with every head-tilt update; the GUI sets `enabled` and calls stop()."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.enabled = False          # "Head drives the dog" switch
        self.steering = Steering()
        self.command = (0, 0)         # (stride, angle) the head is asking for
        self.sent = (0, 0)            # last command sent to the dog
        self.connected = False
        self.status = "Looking for the dog over Bluetooth..."
        self.error = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="dog", daemon=True)
        self._thread.start()

    def on_tilt(self, change, roll):
        """Called for every head-tilt update, exactly as dog_control.py's loop does."""
        cmd = tighter_turn(self.steering.update(change, roll))
        self.command = cmd if self.enabled else (0, 0)

    def set_enabled(self, on):
        self.steering.walking = False   # never resume walking on its own; tilt forward to walk
        self.command = (0, 0)
        self.enabled = on

    def stop(self):
        self.set_enabled(False)

    def close(self):
        self.stop()
        self._running = False
        self._thread.join(timeout=5)

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        link = DogLink(self.dry_run)
        try:
            await link.connect()
        except Exception as e:
            self.error = f"Couldn't connect to the dog: {e}"
            log.error(self.error)
            return
        self.connected = True
        self.status = "dry run (no dog): commands are only shown" if self.dry_run else \
            f"connected to {eog_config.DOG_NAME}"
        log.info("MechDog: %s", self.status)
        last_cmd, last_sent = None, 0.0
        try:
            while self._running:
                await asyncio.sleep(0.05)
                cmd = self.command if self.enabled else (0, 0)
                now = time.monotonic()
                if cmd != last_cmd or now - last_sent >= HEARTBEAT:   # as dog_control.py does
                    await link.send(*cmd)
                    last_sent, self.sent = now, cmd
                if cmd != last_cmd:
                    log.info("Dog: %s", describe(cmd))
                    last_cmd = cmd
        except Exception as e:
            self.connected = False
            self.error = f"Lost the dog: {e}. It stops by itself within 1 s. Restart the app to reconnect."
            log.error(self.error)
        finally:
            try:
                await link.close()   # sends a stop, then disconnects
            except Exception:
                pass


class DogPanel(QWidget):
    """The Dog tab. app.py connects `toggle` and `stop_button`."""

    def __init__(self, dog, eyes):
        super().__init__()
        self.dog, self.eyes = dog, eyes
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.toggle = QPushButton()
        self.toggle.setCheckable(True)
        self.toggle.setMinimumHeight(40)
        self.toggle.setFocusPolicy(Qt.NoFocus)
        self.toggle.toggled.connect(self.show_toggle)
        self.show_toggle(False)
        layout.addWidget(self.toggle)

        self.action = QLabel()
        self.action.setAlignment(Qt.AlignCenter)
        self.action.setMinimumHeight(60)
        layout.addWidget(self.action)
        self.head = QLabel()
        self.head.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.head)

        self.stop_button = QPushButton("■  STOP DOG")
        self.stop_button.setMinimumHeight(48)
        self.stop_button.setFocusPolicy(Qt.NoFocus)
        self.stop_button.setStyleSheet(f"QPushButton {{ font-size: 15px; font-weight: bold; color: {RED}; }}")
        layout.addWidget(self.stop_button)

        hint = QLabel("While on: tilt your head forward to walk (it keeps walking), back to stop, "
                      "and left or right to steer. STOP DOG or Esc stops it and turns head control off. "
                      "The dog also stops by itself if it hears nothing for 1 s.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {GREY};")
        layout.addWidget(hint)
        layout.addStretch(1)

        if dog is None or eyes is None or not eyes.connected or not eyes.use_imu:
            self.toggle.setEnabled(False)
            self.stop_button.setEnabled(dog is not None)

    def show_toggle(self, on):
        if on:
            self.toggle.setText("Head drives the dog: ON  (click or Esc to stop)")
            self.toggle.setStyleSheet(f"background: {GREEN}; color: white; font-weight: bold;")
        else:
            self.toggle.setText("Head drives the dog: OFF  (click to start)")
            self.toggle.setStyleSheet("")

    def refresh(self):
        dog, eyes = self.dog, self.eyes
        if dog is None:
            self.status.setText(dot(GREY, "No dog. Start the app with --dog (and --eog-port) to drive the "
                                          "MechDog with head tilt."))
            return
        if eyes is None or not eyes.use_imu:
            self.status.setText(dot(RED, "Head tilt is needed: start the app with --eog-port, without --no-imu."))
        elif dog.error:
            self.status.setText(dot(RED, dog.error))
        elif dog.connected:
            self.status.setText(dot(GREEN, f"Dog: {dog.status}"))
        else:
            self.status.setText(dot(AMBER, dog.status))
        if dog.connected and eyes is not None and eyes.connected and eyes.use_imu and not self.toggle.isEnabled():
            self.toggle.setEnabled(True)

        moving = dog.sent[0] != 0
        self.action.setText(describe(dog.sent) if self.toggle.isChecked() else "Head control is off")
        self.action.setStyleSheet(f"font-size: {22 if self.toggle.isChecked() else 15}px; font-weight: bold; "
                                  f"color: {GREEN if moving else GREY};")
        if eyes is not None and eyes.head_state is not None:
            self.head.setText(f"Head: {eyes.head_state.upper()}  (forward/back {eyes.pitch:+.0f}°, "
                              f"left/right {eyes.roll:+.0f}°)")
