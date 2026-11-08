"""Head tilt (left/right/forward/back) from the Knight IMU.

Calibration, run at every startup:
  1. Hold still in neutral: the gravity direction becomes zero tilt and the
     average gyro reading becomes the gyro bias.
  2. Tilt forward, then right: these give the forward and right directions
     whichever way the board is mounted.

Tracking is a complementary filter on the gravity direction: the gyro rotates
the estimate every sample and the accelerometer slowly pulls it back, so it is
smooth during movement and doesn't drift.
"""

import time

import numpy as np
from brainflow.board_shim import BoardShim

import board as eog_board
import config


def unit(v):
    return v / np.linalg.norm(v)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)))


class HeadTilt:
    def __init__(self, board, board_id):
        self.board = board
        self.fs = BoardShim.get_sampling_rate(board_id)
        self.accel_rows, self.gyro_rows = eog_board.imu_rows(board_id)
        self.ts_row = BoardShim.get_timestamp_channel(board_id)
        self.last_ts = 0.0
        self.bias = np.zeros(3)
        self.up = self.forward = self.right = None  # calibrated basis
        self.gravity = None                         # current estimate (unit vector)
        self.rate = 0.0                             # head rotation speed, deg/s
        self.state = "neutral"

    # ---- reading ----------------------------------------------------------

    def read_new(self):
        """Accel (n, 3) and gyro (n, 3) samples that arrived since the last call."""
        data = self.board.get_current_board_data(2 * self.fs)
        new = data[self.ts_row] > self.last_ts
        if new.any():
            self.last_ts = data[self.ts_row][new][-1]
        return data[self.accel_rows][:, new].T, data[self.gyro_rows][:, new].T

    def collect(self, seconds):
        accel, gyro = [], []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(0.05)
            a, g = self.read_new()
            accel.append(a)
            gyro.append(g)
        return np.vstack(accel), np.vstack(gyro)

    def is_still(self, accel, gyro):
        return accel.std(axis=0).max() < 0.25 and np.linalg.norm(gyro - self.bias, axis=1).mean() < 0.15

    # ---- calibration ------------------------------------------------------

    def calibrate(self):
        print("\nHEAD TILT CALIBRATION")
        print("Sit upright, look straight ahead, and hold still...")
        self.read_new()  # discard anything buffered before now
        while True:
            accel, gyro = self.collect(config.CAL_HOLD_SECONDS)
            if accel.std(axis=0).max() < 0.25:
                break
            print("  Moved too much, hold still...")
        self.up = unit(accel.mean(axis=0))
        self.bias = gyro.mean(axis=0)
        print(f"  Zeroed. Gyro bias: {np.degrees(self.bias).round(2)} deg/s")

        fwd = self.capture_tilt("Tilt your head FORWARD (chin toward chest) and hold it there")
        self.wait_for_neutral()
        rgt = self.capture_tilt("Tilt your head to the RIGHT (ear toward right shoulder) and hold it there")
        self.wait_for_neutral()

        # Directions gravity moves in when tilting, with the vertical part removed.
        self.forward = unit(fwd - np.dot(fwd, self.up) * self.up)
        right = rgt - np.dot(rgt, self.up) * self.up
        right -= np.dot(right, self.forward) * self.forward
        if np.linalg.norm(right) < 0.1 * np.linalg.norm(rgt - np.dot(rgt, self.up) * self.up):
            raise RuntimeError("Forward and right tilts looked the same. Restart and tilt more clearly.")
        self.right = unit(right)
        self.gravity = self.up.copy()
        print("Calibration done.\n")

    def capture_tilt(self, prompt):
        print(prompt + "...")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            accel, gyro = self.collect(0.75)
            if len(accel) == 0:
                continue
            tilt = angle_between(accel.mean(axis=0), self.up)
            if tilt >= config.CAL_TILT_DEG and self.is_still(accel, gyro):
                print(f"  Got it ({tilt:.0f} deg).")
                return unit(accel.mean(axis=0))
        raise RuntimeError(f"Didn't see a steady tilt of at least {config.CAL_TILT_DEG} deg within 30 s.")

    def wait_for_neutral(self):
        print("Back to neutral...")
        while True:
            accel, gyro = self.collect(0.5)
            if len(accel) and angle_between(accel.mean(axis=0), self.up) < 5 and self.is_still(accel, gyro):
                return

    # ---- tracking ---------------------------------------------------------

    def update(self):
        """Process new samples. Returns the new state if it changed, else None."""
        accel, gyro = self.read_new()
        if len(gyro) == 0:
            return None
        dt = 1.0 / self.fs
        for a, g in zip(accel, gyro):
            w = g - self.bias
            # In the sensor frame, gravity rotates opposite to the head: dg/dt = -w x g.
            predicted = self.gravity - dt * np.cross(w, self.gravity)
            self.gravity = unit((1 - config.IMU_ALPHA) * predicted + config.IMU_ALPHA * unit(a))
        self.rate = np.degrees(np.linalg.norm(gyro - self.bias, axis=1).max())

        pitch, roll = self.angles()
        if self.state == "neutral":
            if max(abs(pitch), abs(roll)) > config.TILT_ENTER_DEG:
                if abs(pitch) >= abs(roll):
                    new = "forward" if pitch > 0 else "back"
                else:
                    new = "right" if roll > 0 else "left"
                self.state = new
                return new
        elif max(abs(pitch), abs(roll)) < config.TILT_EXIT_DEG:
            self.state = "neutral"
            return "neutral"
        return None

    def angles(self):
        """(pitch, roll) in degrees; positive pitch = forward, positive roll = right."""
        g = self.gravity
        down = np.dot(g, self.up)
        pitch = np.degrees(np.arctan2(np.dot(g, self.forward), down))
        roll = np.degrees(np.arctan2(np.dot(g, self.right), down))
        return pitch, roll
