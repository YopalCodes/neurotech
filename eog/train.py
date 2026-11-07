"""Train a rest/left/right/blink classifier on recordings in data/.

    python train.py                 # all recordings in data/
    python train.py data/eog_*.csv  # specific recordings
    python train.py --plot          # also save average waveforms per class
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import features

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "eog_classifier.joblib"


def candidate_models():
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0),
    }


def plot_averages(files, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    by_label = {label: [] for label in config.LABELS.values()}
    for f in files:
        eog, markers, fs = features.load_recording(f)
        filtered = features.filter_eog(eog, fs)
        n = features.epoch_length(fs)
        for idx in np.flatnonzero(markers > 0):
            start = idx + int(round(config.EPOCH_START * fs))
            label = config.LABELS.get(int(markers[idx]))
            if label and start + n <= filtered.shape[1]:
                w = filtered[:, start : start + n]
                by_label[label].append(w - w[:, :1])
    t = config.EPOCH_START + np.arange(n) / fs
    for label, epochs in by_label.items():
        if epochs:
            mean = np.mean(epochs, axis=0)
            axes[0].plot(t, mean[0], label=f"{label} (n={len(epochs)})")
            axes[1].plot(t, mean[1], label=label)
    axes[0].set_title("Horizontal channel")
    axes[1].set_title("Vertical channel")
    for ax in axes:
        ax.set_xlabel("Time from cue (s)")
        ax.axvline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("µV")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved average waveforms -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="recording CSVs (default: data/*.csv)")
    parser.add_argument("--plot", action="store_true", help="save average waveform plot")
    args = parser.parse_args()

    files = [Path(f) for f in args.files] or sorted((HERE / "data").glob("*.csv"))
    if not files:
        raise SystemExit("No recordings found. Run record.py first.")

    parts = [features.epochs_from_recording(f, config.SHIFTS, config.MISALIGNED_SHIFTS) for f in files]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    n_trials = len(set(groups))
    print(f"{len(files)} recording(s), {n_trials} trials, {len(X)} training windows")
    for label in config.LABELS.values():
        print(f"  {label:6s} {(y == label).sum()} windows")

    cv = StratifiedGroupKFold(n_splits=min(5, n_trials // len(config.LABELS)), shuffle=True, random_state=0)
    results = {}
    for name, model in candidate_models().items():
        pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
        results[name] = (balanced_accuracy_score(y, pred), pred)
        print(f"{name:13s} cross-validated balanced accuracy: {results[name][0]:.3f}")

    best = max(results, key=lambda k: results[k][0])
    labels = list(config.LABELS.values())
    print(f"\nBest: {best}\n")
    print(classification_report(y, results[best][1], labels=labels, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):", labels)
    print(confusion_matrix(y, results[best][1], labels=labels))

    model = candidate_models()[best].fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    fs = features.load_recording(files[0])[2]
    joblib.dump({"model": model, "fs": fs, "name": best}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    if args.plot:
        plot_averages(files, HERE / "models" / "average_waveforms.png")


if __name__ == "__main__":
    main()
