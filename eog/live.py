"""Classify EOG in real time with the model from train.py.

    python live.py --port COM5
    python live.py --synthetic

Every STEP seconds the newest epoch-length window is classified. An event is
reported once the same non-rest label wins CONFIRM windows in a row with
probability >= --threshold, then detection pauses for REFRACTORY seconds so
the return-to-center saccade isn't reported as a second event.
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from brainflow.board_shim import BoardShim

import board as eog_board
import config
import features

MODEL_PATH = Path(__file__).parent / "models" / "eog_classifier.joblib"
STEP = 0.1
CONFIRM = 3
REFRACTORY = 1.2
BUFFER_SECONDS = 15  # long enough for the 0.05 Hz high-pass to settle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    board, board_id = eog_board.connect(args.port, args.synthetic)
    fs = BoardShim.get_sampling_rate(board_id)
    if fs != saved["fs"]:
        print(f"Warning: model trained at {saved['fs']} Hz but board runs at {fs} Hz")
    rows = list(eog_board.eog_rows(board_id))
    n = features.epoch_length(fs)
    min_samples = n + 3 * fs

    print("Warming up... (Ctrl+C to stop)")
    streak_label, streak = None, 0
    quiet_until = 0.0
    try:
        while True:
            time.sleep(STEP)
            data = board.get_current_board_data(BUFFER_SECONDS * fs)
            if data.shape[1] < min_samples:
                continue
            filtered = features.filter_eog(data[rows], fs)
            x = features.window_features(filtered[:, -n:], fs).reshape(1, -1)
            proba = model.predict_proba(x)[0]
            best = int(np.argmax(proba))
            label, p = model.classes_[best], proba[best]

            now = time.monotonic()
            if label == "rest" or p < args.threshold or now < quiet_until:
                streak_label, streak = None, 0
                continue
            streak = streak + 1 if label == streak_label else 1
            streak_label = label
            if streak == CONFIRM:
                print(f"{time.strftime('%H:%M:%S')}  {label.upper():6s} (p={p:.2f})")
                quiet_until = now + REFRACTORY
                streak_label, streak = None, 0
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
