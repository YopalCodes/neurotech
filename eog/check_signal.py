"""Live view of all 8 Knight channels, filtered the same way as training.

Use it to find which channels your electrodes are on: look left, look right,
and blink, and watch which traces move. Then set HORIZONTAL_CHANNEL and
VERTICAL_CHANNEL in config.py.

    python check_signal.py --port COM5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from brainflow.board_shim import BoardShim
from matplotlib.animation import FuncAnimation
from scipy import signal

import board as eog_board
import config

SECONDS = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    board, board_id = eog_board.connect(args.port, args.synthetic, channels=range(1, 9))
    fs = BoardShim.get_sampling_rate(board_id)
    rows = BoardShim.get_exg_channels(board_id)[:8]
    sos = signal.butter(4, [0.3, config.LOWPASS], btype="bandpass", fs=fs, output="sos")

    fig, axes = plt.subplots(8, 1, sharex=True, figsize=(10, 9))
    t = np.arange(SECONDS * fs) / fs - SECONDS
    lines = [ax.plot(t, np.zeros_like(t), lw=0.8)[0] for ax in axes]
    labels = [ax.set_ylabel(f"ch{i + 1}", rotation=0, ha="right") for i, ax in enumerate(axes)]
    axes[-1].set_xlabel("seconds")
    fig.suptitle("Look left, right, and blink: the EOG channels show clear steps/spikes")

    def update(_frame):
        data = board.get_current_board_data(SECONDS * fs)
        if data.shape[1] < 2 * fs:
            return lines
        for line, label, ax, row in zip(lines, labels, axes, rows):
            x = signal.sosfiltfilt(sos, data[row] - data[row].mean())
            y = np.full_like(t, np.nan)
            y[-len(x):] = x
            line.set_ydata(y)
            label.set_text(f"ch{rows.index(row) + 1}\n±{x.std():.0f} µV")
            lim = max(50.0, 4 * x.std())
            ax.set_ylim(-lim, lim)
        return lines

    anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)  # noqa: F841 (keep a reference)
    try:
        plt.show()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
