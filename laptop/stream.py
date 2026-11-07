"""Reads a video stream in a background thread and keeps only the newest frame.

If the AI runs slower than the camera, reading frames one at a time falls further and
further behind. Reading continuously and keeping only the latest frame keeps the delay
low: the AI always works on what the camera sees right now.
"""

import logging
import threading
import time

import cv2

log = logging.getLogger("stream")

TIMEOUT_MS = 5000


def open_capture(source):
    if isinstance(source, str):
        # Network stream: use FFmpeg, and give up after a few seconds if the Pi is unreachable.
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS,
        ])
    return cv2.VideoCapture(source)  # a webcam index such as 0


class LatestFrameReader:
    def __init__(self, source):
        self.source = source
        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._running = True
        self.last_frame_time = 0.0  # time.monotonic() of the newest frame; 0 if none yet
        threading.Thread(target=self._run, name="stream", daemon=True).start()

    def read(self, last_id, timeout):
        """Waits for a frame newer than last_id. Returns (frame_id, frame), or (last_id, None) on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame_id != last_id, timeout):
                return last_id, None
            return self._frame_id, self._frame

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            cap = open_capture(self.source)
            if not cap.isOpened():
                log.warning("Can't open video stream %s, retrying...", self.source)
                cap.release()
                time.sleep(2)
                continue
            log.info("Video stream opened: %s", self.source)
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Video stream lost, reconnecting...")
                    break
                with self._cond:
                    self._frame = frame
                    self._frame_id += 1
                    self.last_frame_time = time.monotonic()
                    self._cond.notify_all()
            cap.release()
