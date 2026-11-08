"""Filtering, epoching, and feature extraction shared by train.py and live.py."""

import json
from pathlib import Path

import numpy as np
from brainflow.data_filter import DataFilter
from scipy import signal

import config


def make_filter(fs):
    """Causal band-pass, so offline training sees the same filter as live mode.

    The high-pass is 1st-order so a sustained look to one side stays a step
    instead of ringing back through zero.
    """
    hp = signal.butter(1, config.HIGHPASS, btype="highpass", fs=fs, output="sos")
    lp = signal.butter(4, config.LOWPASS, btype="lowpass", fs=fs, output="sos")
    return np.vstack([hp, lp])


def filter_eog(x, fs):
    """Band-pass each row of a (channels, samples) array."""
    x = x - x[:, :1]  # remove the DC offset so the filter starts without a big step
    return signal.sosfilt(make_filter(fs), x, axis=1)


def epoch_length(fs):
    return int(round((config.EPOCH_END - config.EPOCH_START) * fs))


def window_features(window, fs):
    """Features for one (2, n) window of filtered [horizontal, vertical] EOG."""
    baseline = window[:, : max(1, int(0.1 * fs))].mean(axis=1, keepdims=True)
    w = window - baseline
    d = np.diff(w, axis=1) * fs
    t = np.arange(w.shape[1]) / fs

    feats = []
    for ch, dch in zip(w, d):
        feats += [
            ch.max(), ch.min(), ch.max() - ch.min(), ch.mean(), ch.std(),
            t[ch.argmax()], t[ch.argmin()],
            dch.max(), dch.min(),
        ]
    # Coarse waveform shape: mean of 10 equal segments per channel.
    for ch in w:
        feats += [seg.mean() for seg in np.array_split(ch, 10)]
    # The horizontal/vertical ratio separates blinks (mostly vertical) from saccades.
    ptp_h, ptp_v = np.ptp(w, axis=1)
    feats.append(np.log((ptp_h + 1e-6) / (ptp_v + 1e-6)))
    return np.array(feats)


def load_recording(csv_path):
    """Return (eog, markers, fs) for one recording made by record.py."""
    csv_path = Path(csv_path)
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    data = DataFilter.read_file(str(csv_path))
    eog = data[[meta["horizontal_row"], meta["vertical_row"]]]
    markers = data[meta["marker_row"]]
    return eog, markers, meta["sampling_rate"]


def epochs_from_recording(csv_path, shifts=(0.0,), misaligned_shifts=()):
    """Cut labelled epochs around every cue marker.

    Windows at `misaligned_shifts` are labelled "rest" (see config).
    Returns (X, y, groups); `groups` gives each trial an ID so shifted copies
    of the same trial stay on one side of a cross-validation split.
    """
    eog, markers, fs = load_recording(csv_path)
    filtered = filter_eog(eog, fs)
    n = epoch_length(fs)
    X, y, groups = [], [], []
    for idx in np.flatnonzero(markers > 0):
        label = config.LABELS.get(int(markers[idx]))
        if label is None:
            continue
        windows = [(s, label) for s in shifts]
        if label != "rest":
            windows += [(s, "rest") for s in misaligned_shifts]
        for shift, window_label in windows:
            start = idx + int(round((config.EPOCH_START + shift) * fs))
            if start < 0 or start + n > filtered.shape[1]:
                continue
            X.append(window_features(filtered[:, start : start + n], fs))
            y.append(window_label)
            groups.append(f"{Path(csv_path).stem}:{idx}")
    return np.array(X), np.array(y), np.array(groups)
