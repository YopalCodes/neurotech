"""Signal processing and the gesture classifier.

Calibration recordings and live data go through exactly the same steps, so what the
classifier learns is what it sees live.

What each gesture looks like on electrodes near the eyes:
  left / right   glance and back: a bump on the temple electrodes, opposite sign on each side (EOG)
  up / down      glance and back: a bump on the electrodes above and below the eye (EOG)
  squint         a burst of fast muscle activity, 20-45 Hz (EMG)
  rest           relaxed with normal blinking. Training this is what stops blinks counting as "up".
"""

from datetime import datetime

import numpy as np
from scipy.signal import butter, detrend, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REST = "rest"
COMMANDS = ["left", "right", "up", "down", "squint"]
CLASSES = [REST] + COMMANDS

WINDOW_S = 1.5
# Training windows end this long after each cue. A glance starts about 0.3 s after the cue,
# so these windows hold the whole glance at several positions, as the sliding live window will.
TRAIN_WINDOW_ENDS_S = (1.1, 1.3, 1.5, 1.7)
# Extra "rest" examples come from the get-ready pause before every cue (you're looking at
# the centre, maybe blinking), ending this long after the pause starts.
READY_WINDOW_END_S = 1.5


class StreamFilter:
    """Causal band-pass filters that keep their state between chunks. Filtering a recording
    all at once and filtering it chunk by chunk as it arrives give identical results."""

    BANDS = {
        "eog": (0.1, 10, 2),       # eye movements; a low cut keeps a glance's shape
        "emg": (20, 45, 4),        # muscle (squint); stops below 50/60 Hz mains hum
        "display": (0.3, 45, 2),   # what the plots show
    }

    def __init__(self, fs):
        self.sos = {name: butter(order, (lo, hi), btype="bandpass", fs=fs, output="sos")
                    for name, (lo, hi, order) in self.BANDS.items()}
        self.zi = None

    def process(self, raw):
        """raw: (channels, samples) -> {band name: filtered array of the same shape}."""
        if raw.shape[1] == 0:
            return {name: raw.copy() for name in self.sos}
        if self.zi is None:
            # Start as if the signal had always been at its first value: no start-up transient.
            first = raw[:, 0]
            self.zi = {name: sosfilt_zi(sos)[:, None, :] * first[None, :, None]
                       for name, sos in self.sos.items()}
        out = {}
        for name, sos in self.sos.items():
            out[name], self.zi[name] = sosfilt(sos, raw, axis=1, zi=self.zi[name])
        return out


def window_features(eog, emg, raw, fs):
    """Features for one window. Each argument is (channels, samples). Returns a 1-D array.

    Upward and downward swings are measured separately, so a blink (a short upward spike on
    the electrodes above the eye) can't hide a downward glance in the same window. Which
    swing comes first tells a glance from the smaller rebound that follows the opposite one.
    """
    n = eog.shape[1]
    high = eog.max(axis=1)
    low = eog.min(axis=1)
    order = (np.argmax(eog, axis=1) - np.argmin(eog, axis=1)) / n   # > 0: the downward swing came first
    # How long each swing lasts (a blink is brief, a glance longer).
    high_width = np.mean(eog > 0.5 * np.maximum(high, 1e-6)[:, None], axis=1)
    low_width = np.mean(eog < 0.5 * np.minimum(low, -1e-6)[:, None], axis=1)
    mean = eog.mean(axis=1)
    emg_power = np.log(np.mean(emg ** 2, axis=1) + 1e-6)

    spectrum = np.abs(np.fft.rfft(detrend(raw, axis=1) * np.hanning(raw.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(raw.shape[1], 1 / fs)
    bands = [np.log(spectrum[:, (freqs >= lo) & (freqs < hi)].mean(axis=1) + 1e-6)
             for lo, hi in ((4, 8), (8, 13), (13, 30))]
    return np.concatenate([high, low, order, high_width, low_width, mean, emg_power, *bands])


def training_set(raw, timestamps, cues, fs, ready_times=()):
    """Cuts a calibration recording into labelled windows.

    raw: (channels, samples); timestamps: (samples,) in seconds; cues: [(time, class name)];
    ready_times: start time of the get-ready pause before each cue, used as extra rest examples.
    Returns X (windows x features), y (class names), groups (trial number of each window).
    """
    filtered = StreamFilter(fs).process(raw)
    win = int(round(WINDOW_S * fs))
    X, y, groups = [], [], []

    def add(sample, end_s, label, trial):
        end = sample + int(round(end_s * fs))
        start = end - win
        if start >= 0 and end <= raw.shape[1]:
            X.append(window_features(filtered["eog"][:, start:end], filtered["emg"][:, start:end],
                                     raw[:, start:end], fs))
            y.append(label)
            groups.append(trial)

    for trial, (cue_time, label) in enumerate(cues):
        cue = int(np.searchsorted(timestamps, cue_time))
        for end_s in TRAIN_WINDOW_ENDS_S:
            add(cue, end_s, label, trial)
    for trial, ready_time in enumerate(ready_times):
        add(int(np.searchsorted(timestamps, ready_time)), READY_WINDOW_END_S, REST, trial)
    return np.array(X), np.array(y), np.array(groups)


def train(X, y, groups, fs, n_channels, source_name):
    """Trains the classifier and measures its accuracy on trials it never saw.

    Windows from the same trial always stay together in cross-validation; otherwise the
    accuracy would look better than it really is.
    """
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    fewest_trials = min(len(np.unique(groups[y == c])) for c in np.unique(y))
    cv = StratifiedGroupKFold(n_splits=max(2, min(5, fewest_trials)), shuffle=True, random_state=0)
    predicted = cross_val_predict(model, X, y, groups=groups, cv=cv)
    accuracy = float(np.mean(predicted == y))
    per_class = {c: float(np.mean(predicted[y == c] == c)) for c in CLASSES if np.any(y == c)}
    confusions = {}
    for true, guess in zip(y, predicted):
        if true != guess:
            confusions[(true, guess)] = confusions.get((true, guess), 0) + 1
    model.fit(X, y)
    return {
        "model": model,
        "fs": fs,
        "n_channels": n_channels,
        "source": source_name,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusions": confusions,
        "n_trials": int(len(np.unique(groups))),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


class Detector:
    """Classifies the live window about 10 times a second and fires a command when one
    gesture is clearly detected `consecutive` times in a row.

    After firing it waits until that gesture is no longer detected (so one glance gives one
    command, however long you hold it), and at least `cooldown` seconds, before firing again.
    """

    def __init__(self, bundle, threshold=0.7, cooldown=1.0, consecutive=2):
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        self.threshold = threshold
        self.cooldown = cooldown
        self.consecutive = consecutive
        self._streak_class, self._streak = None, 0
        self._armed = True
        self._last_fire = -np.inf

    def update(self, eog, emg, raw, fs, now):
        """Returns ({class: probability}, fired command or None)."""
        probs = self.model.predict_proba(window_features(eog, emg, raw, fs)[None, :])[0]
        best = int(np.argmax(probs))
        name = self.classes[best]

        if name != REST and probs[best] >= self.threshold:
            self._streak = self._streak + 1 if name == self._streak_class else 1
            self._streak_class = name
        else:
            self._streak, self._streak_class = 0, None
            self._armed = True

        fired = None
        if self._streak >= self.consecutive and self._armed and now - self._last_fire >= self.cooldown:
            fired = name
            self._armed = False
            self._last_fire = now
        return dict(zip(self.classes, probs)), fired
