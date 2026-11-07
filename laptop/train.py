"""Trains a model on the objects in the library, so the app can recognise them.

The app runs this when you press Train in the Teach tab. You can also run it yourself:

    python laptop/train.py                 # defaults
    python laptop/train.py --epochs 100    # train longer: slower, sometimes more accurate

It turns the library photos into a YOLO dataset (kept outside OneDrive, in
~/.neurotech/training), fine-tunes a small pretrained YOLO model on it with the GPU, and
saves the result as library/model.pt. The app runs that model alongside the main one, so
the 80 built-in objects still work.

Lines starting with @@ are for the app to read: @@PROGRESS epoch total score, then
@@DONE score or @@ERROR message.
"""

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

from library import DEFAULT_ROOT, MIN_PHOTOS, Library

WORK_DIR = Path.home() / ".neurotech" / "training"
VAL_EVERY = 5   # every 5th photo is held back to test the model on photos it never trained on


def say(message):
    print(f"[train] {message}", flush=True)


def report(*parts):
    print("@@" + " ".join(str(p) for p in parts), flush=True)


def build_dataset(library, out):
    """Writes the library's photos as a YOLO dataset. Returns ({name: photo count}, data.yaml path)."""
    from ultralytics.utils import YAML

    objects = library.trainable()
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    names = list(objects)
    for class_id, name in enumerate(names):
        for i, (image, (cx, cy, w, h)) in enumerate(library.photos(name)):
            split = "val" if i % VAL_EVERY == VAL_EVERY - 1 else "train"
            stem = f"{class_id}_{image.stem}"
            shutil.copyfile(image, out / "images" / split / f"{stem}.jpg")
            (out / "labels" / split / f"{stem}.txt").write_text(
                f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = out / "data.yaml"
    YAML.save(data_yaml, {"path": str(out), "train": "images/train", "val": "images/val", "names": names})
    return objects, data_yaml


def train(library, base_model, epochs, imgsz):
    too_few = {name: n for name, n in library.objects().items() if n < MIN_PHOTOS}
    for name, n in too_few.items():
        say(f"Skipping {name!r}: it has {n} photos and needs at least {MIN_PHOTOS}.")
    if not library.trainable():
        raise RuntimeError(f"Nothing to train yet: each object needs at least {MIN_PHOTOS} photos.")

    say("Preparing the photos...")
    objects, data_yaml = build_dataset(library, WORK_DIR / "dataset")
    summary = ", ".join(f"{name} ({n})" for name, n in objects.items())

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "the CPU (slow)"
    say(f"Training on {summary} with {device_name}, {epochs} rounds...")

    model = YOLO(base_model)

    def on_fit_epoch_end(trainer):
        # Also runs once more for the final check of the best model, hence the min().
        score = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        report("PROGRESS", min(trainer.epoch + 1, trainer.epochs), trainer.epochs, f"{score:.3f}")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=16, device=device,
        workers=2, project=str(WORK_DIR / "runs"), name="latest", exist_ok=True, plots=False,
    )

    trainer = model.trainer
    best = Path(trainer.best) if Path(trainer.best).exists() else Path(trainer.last)
    score = float(trainer.metrics.get("metrics/mAP50(B)", 0.0))

    # Copy, then rename: the app never sees a half-written model.
    temp = library.model_path.with_name("model.pt.tmp")
    shutil.copyfile(best, temp)
    os.replace(temp, library.model_path)
    library.write_model_info(objects, score, base_model, epochs)
    say(f"Saved {library.model_path}. Score {score:.2f} (1.00 = found every test object).")
    return score


def main():
    parser = argparse.ArgumentParser(description="Train a model on the objects in the library.")
    parser.add_argument("--library", default=str(DEFAULT_ROOT), help="library folder")
    parser.add_argument("--model", default="yolo11s.pt",
                        help="pretrained model to start from (downloaded the first time)")
    parser.add_argument("--epochs", type=int, default=60, help="rounds of training")
    parser.add_argument("--imgsz", type=int, default=640, help="image size to train at")
    args = parser.parse_args()

    try:
        score = train(Library(args.library), args.model, args.epochs, args.imgsz)
    except Exception as e:
        traceback.print_exc()
        report("ERROR", f"{type(e).__name__}: {e}")
        return 1
    report("DONE", f"{score:.3f}")
    return 0


if __name__ == "__main__":   # required on Windows: the data loader starts worker processes
    sys.exit(main())
