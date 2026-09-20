#!/usr/bin/env python3
"""Fine-tune a small YOLO model on the harvested UI dataset.

Short and bounded on purpose: a hackathon has hours, not days, and a ``yolov8n``
starting from COCO weights already knows edges, corners and boxes - all this run
has to teach it is what a button looks like next to a text field. Twenty-ish epochs
on a few hundred screenshots is minutes on an Apple GPU, and the validation numbers
printed at the end are what goes in the report.

Rebuilding from nothing
-----------------------
Trained weights are committed at ``data/models/ui_detector.pt``, so nobody needs to
run this to use the detector, and the shipped detector is unaffected by anything
below::

    uv run python scripts/train_detector.py --data <a dataset>/data.yaml  # ~8 min

**There is no in-repo way to produce that dataset any more.** It was built by
``scripts/build_ui_dataset.py``, which drove the local demo site, and it went with
that site. This script still trains on any YOLO dataset laid out the usual way and
labelled with the class map in :mod:`skillweaver.perception.labeling`, but supplying
one is now the caller's problem. Retraining therefore starts with building a
harvester against real pages, not with running a command that exists.

Training downloads ``yolov8n.pt`` once, which is the only step that needs a network.

The best checkpoint is copied to ``<models_dir>/ui_detector.pt``, which is exactly
where :class:`~skillweaver.perception.detect_yolo.YoloDetector` looks for it, so a
finished run replaces the committed weights in place.

Augmentation is turned almost all the way down. The usual COCO recipe flips images
horizontally and shuffles their colors, which is excellent for photographs of dogs
and actively wrong for user interfaces: a mirrored screenshot puts the scrollbar on
the left and the close button on the right, and no screen the agent will ever see
looks like that.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # allow `python scripts/...` without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from skillweaver.config import settings  # noqa: E402
from skillweaver.perception.detect_yolo import DEFAULT_WEIGHTS_NAME  # noqa: E402
from skillweaver.perception.labeling import CLASS_NAMES  # noqa: E402

DEFAULT_BASE = "yolov8n.pt"
"""The smallest YOLOv8. Downloaded once by ultralytics on first use."""

DEFAULT_EPOCHS = 24
DEFAULT_IMGSZ = 960
"""Bigger than YOLO's usual 640 because a 16-pixel checkbox on a 1280-wide screen
is four pixels across once the frame is squeezed to 640, and four pixels is not a
checkbox. This is the single setting that matters most for small controls."""


def train(
    data: Path,
    *,
    base: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    project: Path,
    name: str,
    patience: int,
) -> tuple[Path, dict[str, float], dict[str, float]]:
    """Run the fine-tune and return ``(best weights, overall metrics, per-class mAP50)``."""
    from ultralytics import YOLO

    # Absolute, always. Ultralytics resolves a relative ``project`` against its own
    # global ``runs_dir`` setting, which lives in the user's config and can point at
    # a completely different checkout - a training run that quietly writes its
    # weights into somebody else's working copy is a bad afternoon.
    project = project.resolve()

    model = YOLO(base)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=patience,
        pretrained=True,
        deterministic=True,
        seed=0,
        val=True,
        plots=False,
        # UI-specific augmentation: geometry must stay honest, so no mirroring and
        # no rotation; small translate/scale jitter still helps the model cope with
        # a layout that shifts a few pixels between viewport widths.
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
        translate=0.03,
        scale=0.15,
    )
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=f"{name}-val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    # The trainer knows where it actually put the checkpoint; guessing the path is
    # how a run reports success and leaves nothing behind.
    best = Path(model.trainer.best)

    overall = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    per_class: dict[str, float] = {}
    for index, class_index in enumerate(metrics.box.ap_class_index):
        per_class[CLASS_NAMES[int(class_index)]] = float(metrics.box.ap50[index])
    return best, overall, per_class


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small YOLO model on the harvested UI dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/models/ui-dataset/data.yaml"),
        help="dataset yaml: a YOLO dataset laid out the usual way, see the module docstring",
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="starting checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    parser.add_argument(
        "--device", default=None, help="torch device: mps, cpu, cuda:0; default is automatic"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to copy the best weights; default <models_dir>/{DEFAULT_WEIGHTS_NAME}",
    )
    parser.add_argument("--name", default="ui-detector", help="run name under the runs directory")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error(
            f"{args.data} does not exist. This repository no longer ships a harvester "
            "that builds one - see the module docstring - so point --data at a YOLO "
            "dataset you have built yourself."
        )

    runs = settings().models_dir / "runs"
    out = args.out or settings().models_dir / DEFAULT_WEIGHTS_NAME
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    best, overall, per_class = train(
        args.data,
        base=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=runs,
        name=args.name,
        patience=args.patience,
    )
    elapsed = time.monotonic() - started

    if not best.is_file():
        print(f"training finished but no checkpoint at {best}", file=sys.stderr)
        return 1
    shutil.copy2(best, out)

    print(f"\ntrained {args.epochs} epochs at imgsz={args.imgsz} in {elapsed / 60:.1f} min")
    print("validation:")
    for key, value in overall.items():
        print(f"  {key:<9} {value:.3f}")
    print("mAP50 per class (classes absent from the dataset are not listed):")
    for name in CLASS_NAMES:
        if name in per_class:
            print(f"  {name:<11} {per_class[name]:.3f}")
    print(f"\nweights -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
