"""The processing pipeline, with no GUI: reads the Pi's video, runs YOLO on the GPU, decides
on a command, and sends it to the Pi. app.py drives it and displays its output.

It runs in its own thread so the window never freezes. The GUI sets the controls
(ai_enabled, manual_cmd, target, conf) and reads results with latest().

Which command is sent:
  1. a manual button is held  -> that command, 20 times a second
  2. AI control is on         -> decide()'s command, after each new frame
                                 (nothing if the video or AI stalls, so the Pi's failsafe stops it)
  3. otherwise                -> "stop", 20 times a second
Manual and "stop" commands come from their own small thread, so a slow frame or a model
reload never makes a held button stutter.

Two models run on every frame: the main one (80 everyday objects) and, once you have trained
one in the Teach tab, the model of your own objects. Their detections are merged into one
result, so decide() and the display treat both the same way.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decide import TARGET, decide
from pi_link import PiLink
from stream import LatestFrameReader

log = logging.getLogger("engine")


@dataclass
class Snapshot:
    """One processed frame, ready to display."""
    id: int
    image: np.ndarray      # BGR, with detection boxes drawn
    raw: np.ndarray        # BGR, the camera frame as it arrived (used for teaching photos)
    detections: list       # [(class name, confidence)], most confident first
    fps: float             # frames processed per second
    ai_ms: float           # time the AI took on this frame


def describe(command):
    """{"cmd": "move", "args": {"speed": 0.4}} -> "move speed=0.4"."""
    args = command.get("args") or {}
    return command["cmd"] + "".join(f" {k}={v}" for k, v in args.items())


def merge(base, extra):
    """Combines two YOLO results for the same frame into one, renumbering extra's classes
    to follow base's."""
    import torch
    from ultralytics.engine.results import Results

    offset = max(base.names) + 1
    names = {**base.names, **{offset + i: name for i, name in extra.names.items()}}
    boxes = base.boxes.data.float()
    if len(extra.boxes):
        more = extra.boxes.data.float().clone()
        more[:, 5] += offset
        boxes = torch.cat([boxes, more.to(boxes.device)])
    return Results(base.orig_img, path=base.path, names=names, boxes=boxes)


class Engine:
    def __init__(self, pi, stream_port, cmd_port, source, model, imgsz, conf, custom_model=None):
        self.pi = pi
        self.cmd_port = cmd_port
        source = source or f"http://{pi}:{stream_port}/stream"
        self.source = int(source) if source.isdigit() else source
        self.model_name = model
        self.custom_model_path = Path(custom_model) if custom_model else None
        self.imgsz = imgsz

        # Controls, set by the GUI.
        self.ai_enabled = False   # off at start: nothing moves until you turn it on
        self.manual_cmd = None
        self.target = TARGET
        self.conf = conf

        # State, read by the GUI.
        self.status = "Starting..."
        self.error = None
        self.ready = False
        self.device_name = ""
        self.class_names = []     # everything that can be targeted: custom_names, then built-in
        self.custom_names = []    # objects from the Teach tab that the current model knows
        self.names_version = 0    # goes up whenever class_names changes
        self.stream = None
        self.link = None
        self.sent = None          # (command, reason) most recently sent
        self.sent_time = 0.0

        self._snapshot = None
        self._reload_custom = True   # load the taught-objects model, if there is one, at start
        self._send_lock = threading.Lock()
        self._last_logged = None
        self._running = True
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)

    def latest(self):
        return self._snapshot

    def reload_custom_model(self):
        """Loads the taught-objects model again, for example after training. Takes effect
        on the next frame."""
        self._reload_custom = True

    def _run(self):
        try:
            self._loop()
        except Exception as e:
            log.exception("The AI pipeline crashed")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._running = False   # also ends the command thread
            if self.link is not None:
                self.link.send({"cmd": "stop"})
                self.link.close()
            if self.stream is not None:
                self.stream.stop()

    def _loop(self):
        self.status = "Loading AI libraries..."
        import torch
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if torch.cuda.is_available():
            device, self.device_name = 0, torch.cuda.get_device_name(0)
            # FP16 on the GPU: about twice as fast, same accuracy. Newer Ultralytics calls it 'quantize'.
            precision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        else:
            device, self.device_name = "cpu", "CPU (slow: CUDA not available)"
            precision = {}
            log.warning("CUDA isn't available, so the AI runs on the CPU. See README.md.")

        def warm_up(m):
            m.predict(np.zeros((480, 640, 3), np.uint8), device=device, imgsz=self.imgsz,
                      verbose=False, **precision)

        self.status = f"Loading {self.model_name} (downloaded the first time)..."
        model = YOLO(self.model_name)
        base_names = sorted(model.names.values())
        self.status = "Warming up the GPU..."
        warm_up(model)
        custom = None

        self.stream = LatestFrameReader(self.source)
        self.link = PiLink(self.pi, self.cmd_port)
        threading.Thread(target=self._command_loop, name="commands", daemon=True).start()

        frame_id, snap_id, fps, last_frame = 0, 0, 0.0, None
        while self._running:
            if self._reload_custom:
                self._reload_custom = False
                custom = self._load_custom(YOLO, warm_up, custom)
                self._set_names(base_names, custom)
                if not self.ready:
                    log.info("AI ready: %s on %s", self.model_name, self.device_name)
                self.ready, self.status = True, "Running"

            frame_id, frame = self.stream.read(frame_id, timeout=0.05)
            ai_cmd = None
            if frame is not None:
                start = time.perf_counter()
                options = dict(device=device, imgsz=self.imgsz, conf=self.conf, verbose=False, **precision)
                result = model.predict(frame, **options)[0]
                if custom is not None:
                    result = merge(result, custom.predict(frame, **options)[0])
                ai_ms = (time.perf_counter() - start) * 1000
                ai_cmd = decide(result, self.target)

                if last_frame is not None:
                    instant = 1 / max(start - last_frame, 1e-6)
                    fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
                last_frame = start
                detections = sorted(
                    ((result.names[int(c)], float(p))
                     for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())),
                    key=lambda d: -d[1])
                snap_id += 1
                self._snapshot = Snapshot(snap_id, result.plot(), frame, detections, fps, ai_ms)

            if ai_cmd is not None:
                with self._send_lock:
                    if self.ai_enabled and not self.manual_cmd:
                        self._send(ai_cmd, "AI")

    def _command_loop(self):
        """Sends manual commands while a button is held, and "stop" while AI control is off."""
        while self._running:
            with self._send_lock:
                manual = self.manual_cmd
                if manual:
                    self._send({"cmd": manual}, "manual")
                elif not self.ai_enabled:
                    self._send({"cmd": "stop"}, "AI control off")
            time.sleep(0.05)

    def _send(self, command, reason):
        """Call with _send_lock held."""
        if self.link.send(command):
            self.sent, self.sent_time = (command, reason), time.monotonic()
            if (command, reason) != self._last_logged:
                log.info("Sending %s (%s)", describe(command), reason)
                self._last_logged = (command, reason)

    def _load_custom(self, YOLO, warm_up, current):
        """Returns the taught-objects model, None if there isn't one yet, or the current one if
        the new one won't load."""
        path = self.custom_model_path
        if path is None or not path.exists():
            return None
        self.status = "Loading your taught objects..."
        try:
            custom = YOLO(str(path))
            warm_up(custom)
        except Exception:
            log.exception("Couldn't load %s; keeping the previous one", path)
            return current
        log.info("Taught objects loaded: %s", ", ".join(custom.names.values()))
        return custom

    def _set_names(self, base_names, custom):
        custom_names = sorted(custom.names.values(), key=str.lower) if custom is not None else []
        names = custom_names + [n for n in base_names if n not in custom_names]
        if self.target not in names:
            fallback = TARGET if TARGET in names else names[0]
            log.warning("Nothing called %r can be detected now; targeting %r instead", self.target, fallback)
            self.target = fallback
        self.custom_names, self.class_names = custom_names, names
        self.names_version += 1
