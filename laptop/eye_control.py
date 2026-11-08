"""Eye control for the desktop app: steer the stepper motor with the NeuroPawn EOG headset.

    look right -> the motor spins right, and keeps spinning
    look left  -> the motor spins left, and keeps spinning
    blink      -> the motor stops

Uses Aram's EOG pipeline in eog/ as it is: board.connect(), features.py, the model made by
train.py, and live.py's timing. Detection is eog/motor_control.py's EventDetector, which fires
as soon as one check passes the certainty threshold.

The headset and its classifier run in a separate process. The classifier is CPU-heavy, and in
the app's own process it competed with the camera AI for Python's interpreter lock, which
delayed both the eye detection and the motor commands.

The board is handled like live.py: connect() once when the app starts, and stop_stream() +
release_session() when it closes. If connecting fails, or the board stops sending data, close
the app, unplug and replug the board, and start it again.

Head tilt (IMU) is handled as in live.py: calibrated at startup (follow the prompts in the
terminal), reported as left/right/forward/back/neutral, and while the head turns faster than
HEAD_MOTION_GATE (60 deg/s) eye events are ignored for 1.5 s, because turning the head makes the
eyes counter-rotate, which looks like a glance. Pass use_imu=False (--no-imu) to skip it.

Instead of a port, "playback:<recording.csv>" replays a recording made by record.py, to try
eye control without the headset (head tilt is off then: a recording can't be calibrated).
"""

import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QLabel, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

sys.path.append(str(Path(__file__).resolve().parent.parent / "eog"))
import board as eog_board  # noqa: E402  (Aram's eog/ modules)
import channel_check  # noqa: E402  (checks the EOG channels really came on)
import config as eog_config  # noqa: E402
import features  # noqa: E402  (Aram's EOG filter, reused to find blink peaks)
import focus  # noqa: E402  (SSVEP focus selection, after NeuroPawn's ssvep tutorial)
from imu import HeadTilt  # noqa: E402
from live import BUFFER_SECONDS, MODEL_PATH, STEP  # noqa: E402
from motor_control import EventDetector  # noqa: E402

log = logging.getLogger("eyes")

EVENT_TO_STATE = {"left": "left", "right": "right", "blink": "stop"}
STATE_TEXT = {"left": "◀  SPINNING LEFT", "right": "SPINNING RIGHT  ▶", "stop": "■  STOPPED"}
GREEN, RED, GREY = "#2e9e4f", "#d64545", "#888888"
RETRY_ADVICE = "Close the app, unplug and replug the board, and start it again."

# Double blink (starts a Focus selection). Aram's classifier decides a blink happened; then,
# DOUBLE_BLINK_WAIT later, the vertical channel's last DOUBLE_BLINK_LOOK seconds are checked
# for two blink peaks DOUBLE_BLINK_GAP apart. On the 20 cued blinks of the 2026-11-08
# recording no single blink counted as double; synthetic doubles 0.4-0.8 s apart were caught
# 70 of 71 times (0.3 s apart they merge into one).
DOUBLE_BLINK_WAIT = 1.0
DOUBLE_BLINK_LOOK = 2.5
DOUBLE_BLINK_GAP = (0.25, 1.0)
# A slower double blink: two blink events from the classifier this close together (it pauses
# 1.2 s after each event, so the second blink must come at least ~1.2 s after the first).
DOUBLE_BLINK_EVENTS_WITHIN = 2.5


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


def blink_check(vertical, fs):
    """vertical: the filtered vertical EOG channel around a detected blink. Returns
    (double?, description): double if it holds two blink peaks DOUBLE_BLINK_GAP apart, each at
    least half the size of the biggest."""
    from scipy.signal import find_peaks
    x = vertical - np.median(vertical)
    x = x * np.sign(x[np.argmax(np.abs(x))])   # blinks point the same way as the biggest one
    top = x.max()
    if top <= 0:
        return False, "no blink peak"
    low, high = DOUBLE_BLINK_GAP
    peaks, _ = find_peaks(x, height=0.5 * top, prominence=0.4 * top, distance=max(1, int(low * fs)))
    gaps = np.diff(peaks) / fs
    double = bool(np.any((gaps >= low) & (gaps <= high)))
    if len(peaks) < 2:
        return False, f"1 blink peak (size {top:.0f})"
    return double, f"{len(peaks)} blink peaks, " + ", ".join(f"{g:.2f}" for g in gaps) + " s apart"


def is_double_blink(vertical, fs):
    return blink_check(vertical, fs)[0]


def _playback(csv_path):
    """A BrainFlow board that replays a recording in real time, in place of the headset."""
    params = BrainFlowInputParams()
    params.file = csv_path
    params.master_board = eog_config.BOARD_ID
    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD.value, params)
    board.prepare_session()
    board.start_stream(45000)
    return board, eog_config.BOARD_ID


def _enable_focus(board, board_id, out):
    """Switches on the SSVEP channels (back of the head) and checks they're live. Returns True
    if they are; on failure reports it and eye tracking carries on without focus selection."""
    try:
        for ch in focus.FOCUS_CHANNELS:
            eog_board.send_command(board, f"chon_{ch}_{eog_config.CHANNEL_GAIN}")
            eog_board.send_command(board, f"rldadd_{ch}")
        channel_check.ensure_channels(board, board_id, focus.FOCUS_CHANNELS)
        out.put(("focus_ready", focus.FOCUS_CHANNELS))
        return True
    except Exception as e:
        out.put(("focus_error", f"Focus channels {focus.FOCUS_CHANNELS} didn't come on: {e}"))
        return False


def _focus_step(data, ts_row, focus_rows, pending, model, out):
    """Handles focus captures whose flicker window has arrived. `pending` holds the requests:
    (trial_id, onset wall time, mode, session_dir, block, target). Returns those still waiting.
    A trial that can't be used is reported as ("focus_miss", trial_id, reason)."""
    waiting = []
    timestamps = data[ts_row]
    for request in pending:
        trial_id, onset, mode, session_dir, block, target = request
        start = int(np.searchsorted(timestamps, onset))
        if 0 < start and start + focus.CAPTURE_SAMPLES <= len(timestamps):
            try:
                trial = focus.preprocess(data[focus_rows, start:start + focus.CAPTURE_SAMPLES])
                if mode == "collect":
                    focus.save_trial(session_dir, block, target, trial)
                    out.put(("focus_saved", trial_id))
                elif model is None:
                    out.put(("focus_miss", trial_id, "No focus model yet: run Train in the Focus tab first."))
                else:
                    out.put(("focus_pred", trial_id, focus.predict(model, trial)))
            except Exception as e:
                out.put(("focus_miss", trial_id, f"Focus trial {trial_id} failed: {e}"))
        elif time.time() - onset > 8 or (len(timestamps) and timestamps[0] > onset):
            out.put(("focus_miss", trial_id, f"Missed the EEG for focus trial {trial_id}: "
                                             "it's no longer in the buffer."))
        else:
            waiting.append(request)
    return waiting


def _detect(port, threshold, out, stop, use_imu, use_focus=False, commands=None):
    """The headset process: live.py's connect / calibrate / read / release, reporting to the app
    via `out`. With use_focus it also runs SSVEP focus selection on requests from `commands`."""
    saved = joblib.load(MODEL_PATH)
    try:
        if port.startswith("playback:"):
            board, board_id = _playback(port[len("playback:"):])
        else:
            board, board_id = channel_check.connect(port)
    except Exception as e:
        out.put(("error", f"Couldn't connect to the headset on {port}: {e}. {RETRY_ADVICE}"))
        return
    try:
        fs = BoardShim.get_sampling_rate(board_id)
        if fs != saved["fs"]:
            out.put(("log", f"Model trained at {saved['fs']} Hz but the board runs at {fs} Hz"))
        rows = list(eog_board.eog_rows(board_id))
        detector = EventDetector(saved["model"], fs, threshold.value)
        tilt, head_until = None, 0.0
        if use_imu:   # as in live.py: calibrate head tilt at startup, prompts in the terminal
            try:
                tilt = HeadTilt(board, board_id)
                tilt.calibrate()
            except Exception as e:
                out.put(("error", f"Head tilt calibration failed: {e} Start the app again, or use --no-imu."))
                return
        focus_rows, focus_model, pending = None, None, []
        blinks = []   # times of blink events still to be checked for a double blink
        last_blink, double_until = -1e9, 0.0
        if use_focus and _enable_focus(board, board_id, out):
            exg = BoardShim.get_exg_channels(board_id)
            focus_rows = [exg[ch - 1] for ch in focus.FOCUS_CHANNELS]
            try:
                focus.build_model()   # imports meegkit now (~10 s the first time), not mid-session
                focus_model, info = focus.fit()
            except Exception as e:
                info = {"error": f"Loading the focus model failed: {e}"}
            out.put(("focus_model", info))
        ts_row = BoardShim.get_timestamp_channel(board_id)
        out.put(("connected", f"{port}, channels {eog_config.HORIZONTAL_CHANNEL} (horizontal) and "
                              f"{eog_config.VERTICAL_CHANNEL} (vertical), {fs} Hz"))
        while not stop.is_set():
            time.sleep(STEP)
            now = time.monotonic()
            while focus_rows is not None and commands is not None and not commands.empty():
                command = commands.get_nowait()
                if command[0] == "capture":
                    pending.append(command[1:])
                elif command[0] == "fit":
                    try:
                        focus_model, info = focus.fit(command[1])
                    except Exception as e:
                        info = {"error": f"Training the focus model failed: {e}"}
                    out.put(("focus_model", info))
            try:
                if tilt is not None:   # as in live.py
                    change = tilt.update()
                    if tilt.rate > eog_config.HEAD_MOTION_GATE:
                        head_until = now + eog_config.EPOCH_END
                        detector.quiet_until = max(detector.quiet_until, head_until)
                    pitch, roll = tilt.angles()
                    out.put(("tilt", tilt.state, float(pitch), float(roll), change, now < head_until))
                data = board.get_current_board_data(BUFFER_SECONDS * fs)
            except Exception as e:
                out.put(("error", f"The headset stopped sending data ({e}). {RETRY_ADVICE}"))
                return
            if pending:
                pending = _focus_step(data, ts_row, focus_rows, pending, focus_model, out)
            if data.shape[1] < detector.n + 3 * fs:
                continue  # warming up, as in live.py
            detector.threshold = threshold.value
            probs, event = detector.update(data[rows], now)
            out.put(("probs", {str(k): float(v) for k, v in probs.items()}))
            if event:
                out.put(("event", str(event[0]), float(event[1])))
                if event[0] == "blink":
                    if now - last_blink <= DOUBLE_BLINK_EVENTS_WITHIN and now >= double_until:
                        out.put(("note", f"Blink check: two blinks {now - last_blink:.1f} s apart"))
                        out.put(("double_blink",))
                        double_until, blinks = now + 3.0, []
                    else:
                        blinks.append(now)
                    last_blink = now
            if blinks and now - blinks[0] >= DOUBLE_BLINK_WAIT:
                blinks.pop(0)
                vertical = features.filter_eog(data[rows], fs)[1][-int(DOUBLE_BLINK_LOOK * fs):]
                double, description = blink_check(vertical, fs)
                out.put(("note", f"Blink check: {description}" + (" -> double blink" if double else "")))
                if double and now >= double_until:
                    out.put(("double_blink",))
                    double_until, last_blink = now + 3.0, -1e9
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass


class EyeControl:
    """The app's side of the headset process. The GUI and engine read state, probs, last_event,
    error and connected, and set threshold.value."""

    def __init__(self, port, use_imu=True, use_focus=False):
        self.port = port
        self.use_imu = use_imu and not port.startswith("playback:")
        # Focus selection (SSVEP, focus.py). The GUI calls focus_capture()/focus_fit(); results
        # arrive in these attributes.
        self.use_focus = use_focus and not port.startswith("playback:")
        self.focus_ready = False        # back-of-head channels confirmed live
        self.focus_error = None
        self.focus_model = None         # {"session", "blocks", "accuracy"} or {"error"}
        self.focus_saved = set()        # trial ids saved during training
        self.focus_predictions = {}     # trial id -> target index
        self.focus_missed = {}          # trial id -> why it couldn't be used
        self._commands = mp.Queue()
        self.head_state = None     # left / right / forward / back / neutral, if the IMU is on
        self.pitch = self.roll = 0.0
        self.head_moving = False   # eye events paused because the head is turning
        self.on_tilt = None        # called with (change, roll) on every head-tilt update (dog_panel.py)
        self.description = ""
        self.error = None
        self.connected = False
        self.state = "stop"        # what the eyes last asked for
        self.probs = {}
        self.last_event = None     # (clock time, label, probability)
        self.double_blinks = 0     # goes up on every double blink (focus_panel.py watches it)
        self.threshold = mp.Value("d", 0.7)
        self._out = mp.Queue()
        self._stop = mp.Event()
        self._process = None

    def connect(self, timeout=300):
        """Starts the headset process and waits until it has connected (and, with the IMU, been
        calibrated). Call once, before the window opens. Raises RuntimeError if that fails."""
        self._process = mp.Process(target=_detect, name="eyes", daemon=True,
                                   args=(self.port, self.threshold, self._out, self._stop, self.use_imu,
                                         self.use_focus, self._commands))
        self._process.start()
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self._out.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                raise RuntimeError(f"The headset didn't respond within {timeout} s. {RETRY_ADVICE}")
            if message[0] == "connected":
                self.description, self.connected = message[1], True
                break
            if message[0] == "error":
                raise RuntimeError(message[1])
            self._handle(message)   # e.g. focus_ready / focus_model, sent while connecting
        threading.Thread(target=self._read, name="eyes-reader", daemon=True).start()

    def _read(self):
        while True:
            try:
                message = self._out.get(timeout=0.5)
            except queue.Empty:
                if not self._process.is_alive():
                    if not self.error and not self._stop.is_set():
                        self.error = f"The headset process stopped. {RETRY_ADVICE}"
                        self.state = "stop"
                    return
                continue
            self._handle(message)

    def _handle(self, message):
        """Applies one message from the headset process."""
        kind = message[0]
        if kind == "probs":
            self.probs = message[1]
        elif kind == "tilt":
            self.head_state, self.pitch, self.roll, change, self.head_moving = message[1:]
            if self.on_tilt is not None:
                self.on_tilt(change, self.roll)
            if change:
                log.info("Head: %s (forward/back %+.0f°, left/right %+.0f°)",
                         change.upper(), self.pitch, self.roll)
        elif kind == "event":
            label, p = message[1], message[2]
            self.state = EVENT_TO_STATE[label]
            self.last_event = (time.strftime("%H:%M:%S"), label, p)
            log.info("Eyes: %s (p=%.2f)", label.upper(), p)
        elif kind == "double_blink":
            self.double_blinks += 1
            log.info("Eyes: DOUBLE BLINK")
        elif kind == "error":
            self.error, self.state = message[1], "stop"
            log.error(message[1])
        elif kind == "focus_ready":
            self.focus_ready = True
            log.info("Focus: channels %s are live", message[1])
        elif kind == "focus_error":
            self.focus_error = message[1]
            log.warning(message[1])
        elif kind == "focus_model":
            self.focus_model = message[1]
            info = message[1]
            if "error" in info:
                log.info("Focus: %s", info["error"])
            else:
                acc = info["accuracy"]
                log.info("Focus model: %d blocks from %s, accuracy %s", info["blocks"], info["session"],
                         f"{acc:.0%}" if acc is not None else "n/a (needs 2+ blocks)")
        elif kind == "focus_saved":
            self.focus_saved.add(message[1])
        elif kind == "focus_pred":
            self.focus_predictions[message[1]] = message[2]
        elif kind == "focus_miss":
            self.focus_missed[message[1]] = message[2]
            log.warning(message[2])
        elif kind == "log":
            log.warning(message[1])
        elif kind == "note":
            log.info(message[1])

    def focus_capture(self, trial_id, onset, mode, session_dir=None, block=0, target=-1):
        """Asks the headset process for the 1.5 s of EEG that started at `onset` (time.time()).
        mode "collect" saves it as training data; "predict" classifies it."""
        self._commands.put(("capture", trial_id, onset, mode,
                            str(session_dir) if session_dir else None, block, target))

    def focus_fit(self, session_dir=None):
        """Retrains the focus model (on the latest training session by default)."""
        self.focus_model = None
        self._commands.put(("fit", str(session_dir) if session_dir else None))

    def close(self):
        """Stops the headset process, which runs stop_stream() + release_session() on its way out."""
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()


class EyePanel(QWidget):
    """The Eyes tab. app.py connects `toggle` (the Eye control on/off button)."""

    def __init__(self, eyes):
        super().__init__()
        self.eyes = eyes
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.headset = QLabel()
        self.headset.setWordWrap(True)
        layout.addWidget(self.headset)
        self.head = QLabel()
        self.head.setWordWrap(True)
        layout.addWidget(self.head)

        self.toggle = QPushButton()
        self.toggle.setCheckable(True)
        self.toggle.setMinimumHeight(40)
        self.toggle.setFocusPolicy(Qt.NoFocus)
        self.toggle.toggled.connect(self.show_toggle)
        self.show_toggle(False)
        layout.addWidget(self.toggle)

        self.state_label = QLabel()
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setMinimumHeight(60)
        layout.addWidget(self.state_label)
        self.last = QLabel("Look right or left to spin the motor; blink to stop it.")
        self.last.setAlignment(Qt.AlignCenter)
        self.last.setWordWrap(True)
        layout.addWidget(self.last)

        detection_box = QGroupBox("Detection")
        detection = QFormLayout(detection_box)
        self.bars = {}
        for label in ("left", "right", "blink"):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            self.bars[label] = bar
            detection.addRow(f"{label}:", bar)
        self.threshold_label = QLabel()
        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(50, 95)
        self.threshold.setValue(70)
        self.threshold.valueChanged.connect(self.set_threshold)
        self.set_threshold(70)
        detection.addRow(self.threshold_label, self.threshold)
        layout.addWidget(detection_box)

        hint = QLabel("While eye control is on: look right to spin right, look left to spin left, "
                      "blink to stop. Turning it on turns AI control off. A held Spin button "
                      "(Drive tab) overrides your eyes; Esc turns eye control off.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {GREY};")
        layout.addWidget(hint)
        layout.addStretch(1)

        if eyes is None or not eyes.connected:   # not started with --eog-port, or connecting failed
            self.toggle.setEnabled(False)
            self.threshold.setEnabled(False)
        if eyes is None:
            self.headset.setText(dot(GREY, "No headset. To use eye control, start the app with "
                                           "--eog-port COM10 (the headset's port)."))

    def show_toggle(self, on):
        if on:
            self.toggle.setText("Eye control: ON  (click or Esc to stop)")
            self.toggle.setStyleSheet(f"background: {GREEN}; color: white; font-weight: bold;")
        else:
            self.toggle.setText("Eye control: OFF  (click to start)")
            self.toggle.setStyleSheet("")

    def set_threshold(self, value):
        self.threshold_label.setText(f"Certainty ≥ {value}%:")
        if self.eyes is not None:
            self.eyes.threshold.value = value / 100

    def refresh(self):
        eyes = self.eyes
        if eyes is None:
            return
        if eyes.error:
            self.headset.setText(dot(RED, eyes.error))
        else:
            self.headset.setText(dot(GREEN, f"Headset: {eyes.description}"))
        if not eyes.use_imu:
            self.head.setText(f"<span style='color:{GREY}'>Head tilt: off</span>")
        elif eyes.head_state is not None:
            paused = (f"  <b style='color:{RED}'>head moving: eye events paused</b>"
                      if eyes.head_moving else "")
            self.head.setText(f"Head: <b>{eyes.head_state.upper()}</b>  (forward/back {eyes.pitch:+.0f}°, "
                              f"left/right {eyes.roll:+.0f}°){paused}")
        for label, bar in self.bars.items():
            bar.setValue(round(100 * eyes.probs.get(label, 0)))
        driving = self.toggle.isChecked()
        color = GREEN if driving and eyes.state != "stop" else GREY
        self.state_label.setText(STATE_TEXT[eyes.state] if driving else "Eye control is off")
        self.state_label.setStyleSheet(f"font-size: {22 if driving else 15}px; font-weight: bold; "
                                       f"color: {color};")
        if eyes.last_event:
            clock, label, p = eyes.last_event
            self.last.setText(f"Last: {label.upper()} (p={p:.2f}) at {clock}")
