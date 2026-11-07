# EOG eye-movement classifier (NeuroPawn Knight + BrainFlow)

Records EOG from the NeuroPawn Knight IMU board (BrainFlow `NEUROPAWN_KNIGHT_BOARD_IMU`),
trains a classifier for
**rest / look left / look right / blink**, and runs it live.

```
pip install -r requirements.txt
python check_signal.py --port COM5   # live view of all 8 channels: find your EOG channels
python record.py --port COM5     # cued recording session (~6 min)
python train.py --plot           # train + cross-validate, saves models/
python live.py --port COM5       # real-time detection in the terminal
```

Every script accepts `--synthetic` to run on BrainFlow's fake board without hardware.

The repo includes two example sessions in `data/` and the model trained on them
in `models/`. They were recorded from one person (channel 6 horizontal, channel 5
vertical), so retrain on your own recordings for good live accuracy. `train.py`
uses every CSV in `data/`, so move the example sessions out first if you only
want your own.

## Close the NeuroPawn EXG Visualizer first

Only one program can hold the board's serial port. Use the Visualizer to check
electrode contact and signal quality, **close it**, and then run these scripts.

Opening the port reboots the board, so every script waits a few seconds
(`BOOT_SECONDS` in config.py) before it sends channel settings.

## Electrode placement

| Knight channel | Positive (P) | Negative (N) | Captures |
|---|---|---|---|
| 1 (horizontal) | outer corner of right eye | outer corner of left eye | left/right gaze |
| 2 (vertical) | above one eye (brow) | below the same eye (cheekbone) | blinks, up/down |
| Bias/RLD + ref | forehead center or mastoid | | |

To use different channels, change `HORIZONTAL_CHANNEL` and `VERTICAL_CHANNEL` in [config.py](config.py).
If left and right come out mirrored, swap P/N on channel 1; that doesn't affect
the classifier as long as you train and run with the same wiring.

## How a recording session works

A full-screen window shows a white dot in the center. Each trial does one of the following:

- **jumps left or right**: look at it, then follow it back to center
- **turns red**: blink once
- **stays put**: keep looking at center (this is the `rest` class)

The scripts mark each cue onset in BrainFlow's marker channel, with 1 = rest, 2 = left,
3 = right and 4 = blink. Recordings go to `data/` as BrainFlow CSVs, each with a JSON
sidecar. By default you get 20 trials per class in random order. Keep your head still,
and save blinks for the gaps between trials. Record 2–3 sessions, with the electrodes
re-applied between some of them; `train.py` uses every file in `data/`.

## Pipeline

- **Filter** ([features.py](features.py)): a causal 1st-order 0.05 Hz high-pass for electrode drift, then a 4th-order 10 Hz low-pass. Training and live mode use the same filter.
- **Epochs**: 0–1.5 s after each cue, plus copies shifted by ±0.2 s. Windows that only catch part of an event (for example, just the return to center) are labelled `rest`, so live mode doesn't report fragments.
- **Features**: the following, computed for each channel:
  - peak, trough, peak-to-peak, mean and standard deviation
  - the time of the peak and of the trough
  - the maximum and minimum slope
  - a 10-segment waveform shape
  - the log ratio of horizontal to vertical amplitude
- **Model**: `train.py` cross-validates shrinkage LDA and a random forest, grouping each trial's windows so they never land in both training and test folds. It keeps the better model and saves it to `models/eog_classifier.joblib`. `--plot` saves the average waveform per class, which is the quickest way to check the wiring.
- **Live**: every 0.1 s, `live.py` classifies the newest 1.5 s window. It reports an event once three windows in a row agree with p ≥ 0.7, and then ignores new detections for 1.2 s. Events therefore appear about 1 s after you move your eyes.
