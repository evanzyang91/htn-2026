# Detector test fixtures

Committed screenshots and their DOM ground truth, used by
`tests/perception/test_detect_yolo.py` to score the trained detector.

They come from a viewport - 1200x780 - that is in neither split of the training
dataset, so the recall measured here is generalization rather than recall.
`set_dialog` is captured at device scale 2.0, which is what keeps the test honest
about the physical-to-logical conversion.

Labels are ordinary YOLO label files: class id from
`skillweaver.perception.labeling.CLASS_NAMES`, then `cx cy w h` normalized against
the LOGICAL frame size in `frames.json`.

## What the shipped weights measure here

Recall at 0.5 IoU, matching element kind, for `yolov8n` fine-tuned 24 epochs at
`imgsz=960` on 200 training frames (`scripts/train_detector.py` defaults):

| frame | elements | recall |
| --- | --- | --- |
| `mail_inbox` (1x) | 55 | 1.000 |
| `rec_selected` (1x) | 183 | 0.940 |
| `set_dialog` (2x) | 46 | 0.783 |
| **mean** | | **0.908** |
| **interactive kinds only, pooled** | 93 | **0.957** |

`set_dialog` is the weak frame: its modal scrim dims the page behind the dialog and
about a third of the greyed-out body text goes unfound. Every control on it is
found. The floors pinned in the test sit below these numbers with room for the
run-to-run wobble of a short fine-tune.

## Regenerating

    uv run python scripts/build_ui_dataset.py --fixtures tests/perception/fixtures

The sandbox app is deterministic, so a regenerated frame is byte-identical unless
the app itself changed - in which case the pinned recall floors should be
remeasured, not adjusted until the test passes.
