"""Where the EEG data comes from: the NeuroPawn Knight board (through BrainFlow), or a
simulated headset for trying the app without hardware.

Every source has: name, fs (samples per second), n_channels, lead_off (bool per channel),
start(), read() -> (data in µV with shape (channels, samples), timestamps in seconds), stop().
"""

import time

import numpy as np


class BrainFlowSource:
    BOARDS = {"NeuroPawn Knight": 57, "NeuroPawn Knight IMU": 66}  # BrainFlow BoardIds

    def __init__(self, name, serial_port):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        board_id = self.BOARDS[name]
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self.name = name
        self.board = BoardShim(board_id, params)
        self.fs = BoardShim.get_sampling_rate(board_id)
        self.eeg_rows = BoardShim.get_eeg_channels(board_id)
        self.ts_row = BoardShim.get_timestamp_channel(board_id)
        self.loff_rows = BoardShim.get_other_channels(board_id)[:2]  # lead-off status bytes P and N
        self.n_channels = len(self.eeg_rows)
        self.lead_off = np.zeros(self.n_channels, bool)

    def start(self):
        self.board.prepare_session()
        try:
            self.board.start_stream(45000)
        except Exception:
            self.board.release_session()
            raise

    def read(self):
        data = self.board.get_board_data()
        if data.shape[1] and len(self.loff_rows) == 2:
            bits = int(data[self.loff_rows[0], -1]) | int(data[self.loff_rows[1], -1])
            self.lead_off = np.array([(bits >> i) & 1 for i in range(self.n_channels)], bool)
        return data[self.eeg_rows], data[self.ts_row]

    def stop(self):
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass


class SimulatedSource:
    """A fake 8-channel headset with electrodes around the eyes, laid out as the README
    recommends. It blinks on its own every few seconds. trigger() makes it glance or squint;
    the app calls it from the arrow keys and S, and during calibration it performs each
    cued gesture by itself."""

    name = "Simulator"
    fs = 125
    n_channels = 8
    GLANCE_UV = 150   # eye-movement signal at the temple electrodes
    BLINK_UV = 220
    SQUINT_UV = 35    # muscle noise level while squinting

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.lead_off = np.zeros(self.n_channels, bool)
        self.events = []  # (start time, kind)
        self.offsets = self.rng.uniform(-300, 300, self.n_channels)   # electrode DC offsets
        self.phase = self.rng.uniform(0, 2 * np.pi, self.n_channels)
        self.t0 = None
        self.sent = 0
        self.next_blink = None

    def start(self):
        self.t0 = time.time()
        self.sent = 0
        self.next_blink = self.t0 + self.rng.uniform(1, 4)

    def stop(self):
        pass

    def trigger(self, kind, delay=0.0, at=None):
        """kind: left, right, up, down, squint, or blink."""
        self.events.append(((time.time() if at is None else at) + delay, kind))

    def read(self):
        total = int((time.time() - self.t0) * self.fs)
        t = self.t0 + np.arange(self.sent, total) / self.fs
        self.sent = total
        return self.generate(t), t

    def generate(self, t):
        n = len(t)
        if n == 0:
            return np.zeros((self.n_channels, 0))
        while self.next_blink < t[-1]:
            self.events.append((self.next_blink, "blink"))
            self.next_blink += self.rng.uniform(2, 6)

        horizontal, vertical, blink, squint = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for start, kind in self.events:
            tau = t - start
            if kind in ("left", "right", "up", "down"):
                # Glance: eyes move in 50 ms, stay ~0.4 s, come back.
                shape = np.clip(tau / 0.05, 0, 1) * np.clip((0.5 - tau) / 0.05, 0, 1)
                sign = 1 if kind in ("left", "up") else -1
                if kind in ("left", "right"):
                    horizontal += sign * shape
                else:
                    vertical += sign * shape
            elif kind == "blink":
                blink += np.exp(-((tau - 0.1) / 0.05) ** 2)
            elif kind == "squint":
                squint += np.clip(tau / 0.1, 0, 1) * np.clip((1.0 - tau) / 0.1, 0, 1)
        self.events = [(s, k) for s, k in self.events if s > t[-1] - 2]

        g, b = self.GLANCE_UV, self.BLINK_UV
        eye = np.zeros((self.n_channels, n))
        eye[0] = -g * horizontal                      # Ch1 right temple
        eye[1] = g * horizontal                       # Ch2 left temple
        eye[2] = g * vertical + b * blink - 30 * squint   # Ch3 above the right eye
        eye[3] = -g * vertical - 0.3 * b * blink + 30 * squint  # Ch4 below the right eye
        eye[4] = 0.4 * (g * vertical + b * blink) - 0.2 * g * horizontal  # Ch5 forehead left
        eye[5] = 0.4 * (g * vertical + b * blink) + 0.2 * g * horizontal  # Ch6 forehead right
        # Ch7, Ch8: further back on the head; almost no eye signal.

        muscle_gain = np.array([1, 1, 0.9, 0.9, 0.6, 0.6, 0.2, 0.2])[:, None]
        muscle = self.SQUINT_UV * squint[None, :] * muscle_gain * self.rng.normal(0, 1, (self.n_channels, n))
        background = (self.rng.normal(0, 6, (self.n_channels, n))                                   # noise
                      + 4 * np.sin(2 * np.pi * 10 * t[None, :] + self.phase[:, None])               # alpha
                      + 15 * np.sin(2 * np.pi * 60 * t)[None, :]                                    # mains hum
                      + 20 * np.sin(2 * np.pi * 0.05 * t[None, :] + self.phase[:, None]))           # drift
        return self.offsets[:, None] + background + eye + muscle
