"""The library of objects you have taught the camera. It is plain files, so you can browse it,
back it up, or share it by copying the folder.

    library/
      objects/
        mug/
          20261107-153012-481.jpg   a full camera frame
          20261107-153012-481.txt   where the mug is in it: "x_centre y_centre width height",
                                    as fractions of the image size
        keys/
          ...
      model.pt     the model trained on these objects (made by train.py)
      model.json   what that model was trained on, and how well it scored

To remove one photo by hand, delete its .jpg and .txt together.
"""

import json
import re
import shutil
import time
from pathlib import Path

import cv2

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "library"
MIN_PHOTOS = 10    # an object needs at least this many photos before it can be trained
GOOD_PHOTOS = 30   # recommended: more photos, from more angles, give better recognition

_VALID_NAME = re.compile(r"^\w[\w \-]*$")   # letters (any language), digits, _, space, -


class Library:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.model_path = self.root / "model.pt"
        self.info_path = self.root / "model.json"

    # ---- objects ----

    def objects(self):
        """{name: number of photos}, in alphabetical order."""
        if not self.objects_dir.is_dir():
            return {}
        folders = sorted((p for p in self.objects_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
        return {p.name: len(self.photos(p.name)) for p in folders}

    def trainable(self):
        """{name: number of photos} for the objects with enough photos to train."""
        return {name: n for name, n in self.objects().items() if n >= MIN_PHOTOS}

    def add_object(self, name):
        """Creates an empty object and returns its name. Raises ValueError if the name won't work."""
        name = " ".join(name.split())
        if not name:
            raise ValueError("Type a name first.")
        if len(name) > 40:
            raise ValueError("Use a name of 40 characters or fewer.")
        if not _VALID_NAME.match(name):
            raise ValueError("Use only letters, numbers, spaces, - and _ in the name.")
        for existing in self.objects():
            if existing.lower() == name.lower():
                raise ValueError(f"There is already an object called {existing!r}.")
        (self.objects_dir / name).mkdir(parents=True)
        return name

    def delete_object(self, name):
        if name not in self.objects():   # also stops names like ".." reaching rmtree
            raise ValueError(f"No object called {name!r}.")
        shutil.rmtree(self.objects_dir / name)

    # ---- photos ----

    def photos(self, name):
        """[(image path, (x_centre, y_centre, width, height))], oldest first."""
        folder = self.objects_dir / name
        found = []
        for image in sorted(folder.glob("*.jpg")):
            box = _read_box(image.with_suffix(".txt"))
            if box is not None:
                found.append((image, box))
        return found

    def save_photo(self, name, frame, box):
        """Saves a BGR camera frame with the object's box, given as (x1, y1, x2, y2) fractions
        of the image size. Returns how many photos the object has now."""
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in box)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise ValueError("The box is empty.")
        folder = self.objects_dir / name
        folder.mkdir(parents=True, exist_ok=True)

        now = time.time()
        stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        while (folder / f"{stem}.jpg").exists():
            stem += "x"

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Couldn't encode the photo as JPEG.")
        # Write the photo first: a photo only counts once its box file exists.
        (folder / f"{stem}.jpg").write_bytes(jpg.tobytes())
        (folder / f"{stem}.txt").write_text(
            f"{(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
        return len(self.photos(name))

    def delete_last_photo(self, name):
        """Deletes the newest photo of an object. Returns False if it has none."""
        photos = self.photos(name)
        if not photos:
            return False
        image, _ = photos[-1]
        image.unlink()
        image.with_suffix(".txt").unlink(missing_ok=True)
        return True

    # ---- trained model ----

    def model_info(self):
        """What the current model.pt was trained on, or None if there isn't one."""
        if not self.model_path.exists():
            return None
        try:
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def model_out_of_date(self):
        """True if photos were added or removed since the model was trained."""
        info = self.model_info()
        return info is not None and info.get("photos") != self.trainable()

    def write_model_info(self, photos, score, base_model, epochs):
        info = {
            "classes": list(photos),
            "photos": photos,
            "score_map50": round(score, 4),
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "base_model": base_model,
            "epochs": epochs,
        }
        self.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_box(path):
    """Reads a box file. Returns (x_centre, y_centre, width, height), or None if it's missing or invalid."""
    try:
        values = [float(v) for v in path.read_text().split()]
    except (OSError, ValueError):
        return None
    if len(values) == 5:   # also accept a full YOLO label line, which starts with a class number
        values = values[1:]
    if len(values) != 4:
        return None
    cx, cy, w, h = values
    if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
        return None
    return cx, cy, w, h
