"""Cued EOG recording session.

A dot sits at the center of the screen. On each trial it either jumps to the
left or right edge (look at it, then follow it back to center), turns red
(blink once), or stays put (rest, keep looking at center). A marker with the
trial label is written into BrainFlow's marker channel at cue onset.

    python record.py --port COM5
    python record.py --synthetic          # no hardware, for testing

Press SPACE to start, ESC to stop early. Data is saved either way.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter

import board as eog_board
import config

DATA_DIR = Path(__file__).parent / "data"
DOT_RADIUS = 18


class Session:
    def __init__(self, board, trials):
        self.board = board
        self.trials = trials
        self.index = 0

        self.root = tk.Tk()
        self.root.title("EOG recording")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill="white", outline="")
        self.text = self.canvas.create_text(
            self.width / 2, self.height * 0.25, fill="gray70", font=("Segoe UI", 20),
            text="Keep your head still and look at the dot.\n"
                 "Side jump: look at it, then follow it back.  Red: blink once.\n"
                 "Try to only blink between trials.\n\nPress SPACE to start.",
            justify="center",
        )
        self.move_dot(0)
        self.root.bind("<space>", self.start)
        self.root.bind("<Escape>", lambda _: self.root.destroy())

    def move_dot(self, offset, color="white"):
        x = self.width / 2 + offset * self.width * 0.4
        y = self.height / 2
        self.canvas.coords(self.dot, x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS)
        self.canvas.itemconfig(self.dot, fill=color)

    def start(self, _event):
        self.root.unbind("<space>")
        self.canvas.itemconfig(self.text, text="")
        self.root.after(2000, self.next_trial)

    def next_trial(self):
        if self.index >= len(self.trials):
            self.canvas.itemconfig(self.text, text="Done!")
            self.root.after(1500, self.root.destroy)
            return
        label = self.trials[self.index]
        self.index += 1

        self.board.insert_marker(config.MARKERS[label])
        if label == "left":
            self.move_dot(-1)
        elif label == "right":
            self.move_dot(1)
        elif label == "blink":
            self.move_dot(0, color="red")

        self.root.after(int(config.CUE_DURATION * 1000), self.end_trial)

    def end_trial(self):
        self.move_dot(0)
        self.canvas.itemconfig(self.text, text=f"{self.index}/{len(self.trials)}")
        iti = random.uniform(config.ITI_MIN, config.ITI_MAX)
        self.root.after(int(iti * 1000), self.next_trial)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    parser.add_argument("--synthetic", action="store_true", help="use BrainFlow's synthetic board")
    parser.add_argument("--trials", type=int, default=config.TRIALS_PER_CLASS, help="trials per class")
    parser.add_argument("--name", default="", help="optional tag added to the file name")
    args = parser.parse_args()

    trials = [label for label in config.LABELS.values() for _ in range(args.trials)]
    random.shuffle(trials)

    board, board_id = eog_board.connect(args.port, args.synthetic)
    try:
        Session(board, trials).run()
        time.sleep(config.EPOCH_END)  # let the last epoch finish streaming
        data = board.get_board_data()
    finally:
        board.stop_stream()
        board.release_session()

    DATA_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eog_{stamp}" + (f"_{args.name}" if args.name else "")
    DataFilter.write_file(data, str(DATA_DIR / f"{stem}.csv"), "w")

    h_row, v_row = eog_board.eog_rows(board_id)
    meta = {
        "board_id": board_id,
        "sampling_rate": BoardShim.get_sampling_rate(board_id),
        "horizontal_row": h_row,
        "vertical_row": v_row,
        "marker_row": BoardShim.get_marker_channel(board_id),
        "timestamp_row": BoardShim.get_timestamp_channel(board_id),
        "labels": config.LABELS,
        "trials_completed": int((data[BoardShim.get_marker_channel(board_id)] > 0).sum()),
    }
    (DATA_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {data.shape[1]} samples, {meta['trials_completed']} trials -> {DATA_DIR / stem}.csv")


if __name__ == "__main__":
    main()
