"""Focus selection: pick one of two on-screen boxes (Phone, Laptop) by looking at it (SSVEP + TRCA).

Follows NeuroPawn's SSVEP-TRCA tutorial (github.com/NeuroPawn/ssvep): flicker frequencies
from its set, and the same trial timing, filters, analysis window and ensemble-TRCA
classifier (meegkit), adapted to this project:
  * the Knight IMU board (board.connect), with the eye electrodes kept on channels 5 and 6,
    so SSVEP uses the six channels 1-4, 7 and 8 over the back of the head;
  * each trial's EEG window is cut by timestamp from the moment the flicker starts, so a
    delay in passing the request between processes can't shift it.

How it works: each box flickers at its own frequency. Looking at one makes the visual cortex
(back of the head) ripple at that frequency; TRCA, trained on a short session of your own
data, decides which frequency is strongest.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes

FS = 125                                   # Knight sampling rate (Hz)
# One per box: Phone (left), Laptop (right). Two of NeuroPawn's four; on the first headset
# session these two were told apart best (pairs with 10 Hz scored highest) and were furthest apart.
FREQUENCIES = [6.67, 10.00]
N_TARGETS = len(FREQUENCIES)
# Sessions recorded before session.json existed used NeuroPawn's four boxes.
LEGACY_FREQUENCIES = [6.67, 8.57, 10.00, 12.00]

# Board channel -> electrode (10-20 system). Channels 5 and 6 stay on the eyes (EOG).
FOCUS_CHANNELS = [1, 2, 3, 4, 7, 8]
ELECTRODES = {1: "Oz", 2: "O1", 3: "O2", 4: "PO7", 7: "PO8", 8: "POz"}

# Trial timing (seconds), as in NeuroPawn's config.
CUE_S = 1.0          # training only: the box to look at is outlined in red
FLICKER_S = 1.5      # all boxes flicker
REST_S = 0.5         # blank, time to shift gaze
CAPTURE_SAMPLES = int(round(FLICKER_S * FS)) + 1   # 188 samples cover the flicker
LATENCY_S = 0.15     # the cortex takes ~150 ms to lock on; skipped
GAZE_S = 1.0         # EEG analysed per trial

# Filters, as in NeuroPawn's preprocessing.py (all zero-phase).
BANDPASS = (3.0, 48.0, 2)
NOTCHES = [(49.0, 51.0), (59.0, 61.0)]
NOTCH_ORDER = 4

# TRCA filter bank and ensemble setting, as in NeuroPawn's config.
FILTERBANK = [
    [(6, 48), (4, 60)], [(14, 48), (10, 60)], [(22, 48), (16, 60)],
    [(30, 48), (24, 60)], [(38, 48), (32, 60)], [(46, 48), (40, 60)],
]
ENSEMBLE = True

DATA_DIR = Path(__file__).resolve().parent / "focus_data"   # one folder per training session
TRAINING_BLOCKS = 10      # each block shows every box once, in random order (about 1 minute)


def preprocess(window):
    """window: (channels, samples) raw EEG -> (samples, channels) filtered, NeuroPawn's chain."""
    out = np.ascontiguousarray(window, dtype=np.float64).copy()
    for row in out:
        DataFilter.detrend(row, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(row, FS, BANDPASS[0], BANDPASS[1], BANDPASS[2],
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        for low, high in NOTCHES:
            DataFilter.perform_bandstop(row, FS, low, high, NOTCH_ORDER,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    return out.T


def crop():
    """Sample indices TRCA analyses: skip the visual latency, keep GAZE_S (samples 19-143)."""
    start = int(np.floor(LATENCY_S * FS + 0.5))
    return np.arange(start, start + int(np.floor(GAZE_S * FS + 0.5)))


def new_session_dir():
    path = DATA_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    (path / "session.json").write_text(json.dumps({"frequencies": FREQUENCIES}), encoding="utf-8")
    return path


def session_frequencies(session_dir):
    """The flicker frequencies a session was recorded with: file block_<b>_<i>.csv is
    frequency i-1 of this list."""
    try:
        return json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))["frequencies"]
    except (OSError, ValueError, KeyError):
        return LEGACY_FREQUENCIES


def target_files(session_dir):
    """For each of today's FREQUENCIES, its file number in the session; None if the session
    lacks one. Lets a session with more boxes train a model for fewer."""
    recorded = session_frequencies(session_dir)
    numbers = []
    for f in FREQUENCIES:
        match = [i for i, r in enumerate(recorded) if abs(r - f) < 0.01]
        if not match:
            return None
        numbers.append(match[0] + 1)
    return numbers


MIN_BLOCKS = 3            # a session stopped before this many blocks isn't preferred


def latest_session_dir():
    """The newest training session with at least MIN_BLOCKS complete blocks (so a training
    stopped early doesn't replace a full one), else the newest with any."""
    if not DATA_DIR.exists():
        return None
    sessions = [(count_blocks(p), p) for p in DATA_DIR.glob("*") if p.is_dir()]
    sessions = [(n, p) for n, p in sessions if n > 0]
    if not sessions:
        return None
    return max(sessions, key=lambda s: (s[0] >= MIN_BLOCKS, s[1].name))[1]


def save_trial(session_dir, block, target, trial):
    """trial: (samples, channels) filtered. Saved as NeuroPawn does: block_{block}_{target+1}.csv."""
    header = ",".join(ELECTRODES.get(ch, f"ch{ch}") for ch in FOCUS_CHANNELS[:trial.shape[1]])
    np.savetxt(Path(session_dir) / f"block_{block}_{target + 1}.csv", trial, delimiter=",",
               header=header, comments="", fmt="%.7f")


def complete_blocks(session_dir):
    """Block numbers in a session folder with every target saved (a missed trial drops only
    its own block)."""
    files = target_files(session_dir)
    if files is None:
        return []
    numbers = set()
    for path in Path(session_dir).glob("block_*_*.csv"):
        try:
            numbers.add(int(path.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(b for b in numbers if all(
        (Path(session_dir) / f"block_{b}_{n}.csv").exists() for n in files))


def count_blocks(session_dir):
    return len(complete_blocks(session_dir))


def load_training(session_dir):
    """-> eeg (gaze samples, channels, trials), labels (trials,), number of blocks."""
    blocks = complete_blocks(session_dir)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return None, None, 0
    files = target_files(session_dir)
    trials, labels = [], []
    for block in blocks:
        for target, number in enumerate(files):
            trials.append(np.loadtxt(Path(session_dir) / f"block_{block}_{number}.csv",
                                     delimiter=",", skiprows=1))
            labels.append(target)
    eeg = np.array(trials).transpose(1, 2, 0)[crop()]
    return eeg, np.array(labels), n_blocks


def build_model():
    from meegkit.trca import TRCA
    return TRCA(FS, FILTERBANK, ENSEMBLE)


def cross_validate(eeg, labels, n_blocks):
    """Leave-one-block-out accuracy (fraction), as NeuroPawn's evaluate_trca.py does."""
    if n_blocks < 2:
        return None
    correct = []
    for i in range(n_blocks):
        test = slice(i * N_TARGETS, (i + 1) * N_TARGETS)
        keep = np.ones(len(labels), bool)
        keep[test] = False
        model = build_model()
        model.fit(eeg[..., keep], labels[keep])
        correct.append(np.mean(model.predict(eeg[..., test]) == labels[test]))
    return float(np.mean(correct))


def fit(session_dir=None):
    """Fits TRCA on a training session (the latest by default).
    -> (model, info dict) or (None, info) if there is no training data yet."""
    session_dir = Path(session_dir) if session_dir else latest_session_dir()
    if session_dir is None:
        return None, {"error": "No focus training yet: run Train in the Focus tab."}
    eeg, labels, n_blocks = load_training(session_dir)
    if n_blocks == 0:
        return None, {"error": f"{session_dir.name} has no complete training block."}
    accuracy = cross_validate(eeg, labels, n_blocks)
    model = build_model()
    model.fit(eeg, labels)
    return model, {"session": session_dir.name, "blocks": n_blocks, "accuracy": accuracy}


def predict(model, trial):
    """trial: (samples, channels) filtered capture -> target index."""
    return int(model.predict(np.expand_dims(trial, axis=2)[crop()])[0])
