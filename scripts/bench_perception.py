"""Measure the real perception stack on the checked-in fixture frames.

Latency: YOLO detection, OCR reading, fingerprinting and the full fused
observation, per frame, on this machine. Accuracy: detector recall against the
DOM-derived labels beside each frame, overall and over interactive controls.

Run it with:

    uv run python scripts/bench_perception.py [--rounds 3] [--json out.json]

No browser, no model and no network: everything comes from
``tests/perception/fixtures``. The numbers are wall-clock and machine-specific;
compare two runs on the same machine, not across machines.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# ultralytics installs its own top-level `tests` package into the venv, which
# shadows this repository's tests/ in a plain python process; bind ours first.
_tests = types.ModuleType("tests")
_tests.__path__ = [str(REPO / "tests")]
sys.modules["tests"] = _tests

from skillweaver.perception.detect_yolo import YoloDetector, default_weights_path  # noqa: E402
from skillweaver.perception.elements import build_index, merge_elements  # noqa: E402
from skillweaver.perception.fingerprint import StateFingerprinter  # noqa: E402
from skillweaver.perception.labeling import elements_from_label_text, recall_at_iou  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402
from skillweaver.perception.screenshot import load_screenshot, physical_size  # noqa: E402

FIXTURES = REPO / "tests" / "perception" / "fixtures"

INTERACTIVE = {"button", "link", "text_field", "checkbox", "radio", "tab", "menu", "row"}


def load_frames() -> list[dict]:
    manifest = json.loads((FIXTURES / "frames.json").read_text(encoding="utf-8"))
    frames = []
    for entry in manifest["frames"]:
        shot = load_screenshot(
            FIXTURES / entry["image"],
            scale=entry.get("scale", 1.0),
            width=entry.get("width"),
            height=entry.get("height"),
        )
        labels = elements_from_label_text(
            (FIXTURES / entry["labels"]).read_text(encoding="utf-8"),
            width=shot.width,
            height=shot.height,
        )
        frames.append({"name": entry["image"], "shot": shot, "expected": labels})
    return frames


def timed(fn, *args, rounds: int) -> tuple[float, object]:
    """Best-of-N wall time in ms, and the last result."""
    best = float("inf")
    result = None
    for _ in range(rounds):
        start = time.perf_counter()
        result = fn(*args)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best, result


def cold_timed(make, *args, rounds: int) -> tuple[float, object]:
    """Best-of-N over a FRESH object each round, so no cache carries between them.

    A reader that remembers the lines it has already read would otherwise report its
    second look at a frame as the cost of its first, which is the number that flatters
    it most and the one nobody actually pays.
    """
    best = float("inf")
    result = None
    for _ in range(rounds):
        subject = make()
        start = time.perf_counter()
        result = subject(*args)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds, best-of")
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = parser.parse_args()

    weights = default_weights_path()
    if not weights.exists():
        print(f"no detector weights at {weights}; run scripts/train_detector.py first")
        return 1

    frames = load_frames()
    detector = YoloDetector(weights)
    warm_reader = RapidOcrReader()
    fingerprinter = StateFingerprinter()

    # Load the models outside the timed region: a benchmark of the first call
    # would measure torch and onnxruntime initialisation, not perception. The engine
    # is shared with every cold reader below, so building one costs nothing either.
    detector.detect(frames[0]["shot"])
    warm_reader.read(frames[0]["shot"])
    engine = warm_reader._ensure_engine()  # noqa: SLF001 - a benchmark, sharing the load

    def cold_reader():
        return RapidOcrReader(engine=engine, cache=False).read

    rows = []
    for frame in frames:
        shot = frame["shot"]
        detect_ms, found = timed(detector.detect, shot, rounds=args.rounds)

        # Two honest numbers rather than one flattering one: what it costs to read a
        # screen never seen before, and what it costs to read one whose lines are
        # already known. A real run sits between them - an action changes part of a
        # screen, so some lines are new and most are not.
        cold_ocr_ms, text = cold_timed(cold_reader, shot, rounds=args.rounds)
        warm_reader.read(shot)
        warm_ocr_ms, _ = timed(warm_reader.read, shot, rounds=args.rounds)

        def observe(s=shot, reader=warm_reader):
            f = detector.detect(s)
            t = reader.read(s)
            merged = tuple(merge_elements(f, t))
            build_index(merged)
            return fingerprinter.fingerprint(s, merged, "http://bench.local/")

        observe_ms, _ = timed(observe, rounds=args.rounds)
        merged = merge_elements(found, text)
        fp_ms, _ = timed(
            fingerprinter.fingerprint,
            shot,
            tuple(merged),
            "http://bench.local/",
            rounds=args.rounds,
        )

        expected = frame["expected"]
        recall = recall_at_iou(expected, found, iou=0.5, match_kind=True)
        controls = [e for e in expected if e.kind.value in INTERACTIVE]
        control_recall = recall_at_iou(controls, found, iou=0.5, match_kind=True)

        rows.append(
            {
                "frame": frame["name"],
                "physical": "x".join(str(v) for v in physical_size(shot)),
                "detect_ms": round(detect_ms, 1),
                "ocr_ms": round(cold_ocr_ms, 1),
                "ocr_warm_ms": round(warm_ocr_ms, 1),
                "fingerprint_ms": round(fp_ms, 1),
                "observe_ms": round(observe_ms, 1),
                "detections": len(found),
                "recall": round(recall, 3),
                "control_recall": round(control_recall, 3),
                "controls": len(controls),
            }
        )

    header = (
        f"{'frame':<18} {'physical':>10} {'detect':>8} {'ocr cold':>9} {'ocr warm':>9} "
        f"{'fprint':>7} {'observe':>8} {'recall':>7} {'ctl-recall':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['frame']:<18} {row['physical']:>10} {row['detect_ms']:>7.1f}m "
            f"{row['ocr_ms']:>8.1f}m {row['ocr_warm_ms']:>8.1f}m {row['fingerprint_ms']:>6.1f}m "
            f"{row['observe_ms']:>7.1f}m {row['recall']:>7.3f} {row['control_recall']:>10.3f}"
        )
    print(
        "\nocr cold: a screen whose lines have never been read. ocr warm: the same screen "
        "again,\nwhich is what a re-observation after an action mostly is. A real run sits "
        "between them."
    )

    summary = {
        "mean_detect_ms": round(statistics.fmean(r["detect_ms"] for r in rows), 1),
        "mean_ocr_cold_ms": round(statistics.fmean(r["ocr_ms"] for r in rows), 1),
        "mean_ocr_warm_ms": round(statistics.fmean(r["ocr_warm_ms"] for r in rows), 1),
        "mean_observe_ms": round(statistics.fmean(r["observe_ms"] for r in rows), 1),
        "mean_recall": round(statistics.fmean(r["recall"] for r in rows), 3),
        "mean_control_recall": round(statistics.fmean(r["control_recall"] for r in rows), 3),
    }
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"frames": rows, "summary": summary}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
