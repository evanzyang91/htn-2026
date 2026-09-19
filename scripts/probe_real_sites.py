"""Measure the eyes on REAL websites, against the DOM as ground truth.

The detector shipped with this project was trained on screenshots of one application
(see ``scripts/build_ui_dataset.py``), so the question that decides whether any of this
generalises is not how it scores there - it is what it does on pages it has never seen
and that nobody built for it.

    uv run python scripts/probe_real_sites.py [--sites URL ...] [--shots DIR]

For each page it reports two numbers, and the second is the one that matters:

**control recall** - the fraction of the DOM's interactive elements the detector found
at IoU 0.5. A perception metric.

**reachability** - the fraction of the DOM's LABELLED interactive elements that the
agent could actually click, by asking for them the way a skill does: look the label up
with ``ElementIndex.find_text`` and check that the point it would click lands inside
the real control. An agent does not need to have detected a button to press it - it
needs the words on it to lead somewhere that hits it - so this is the number that
predicts whether a task can be done.

``BrowserGroundTruth`` reads the DOM and is an OFFLINE TEACHER: it is used here to
mark the answers and is never reachable from the agent.

Only public pages, loaded once each, read-only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from skillweaver.contracts import Box, Element, ElementKind, Navigate  # noqa: E402
from skillweaver.controllers.browser import BrowserController, BrowserGroundTruth  # noqa: E402
from skillweaver.orchestrator import ComposedPerceiver  # noqa: E402
from skillweaver.perception.detect_yolo import YoloDetector  # noqa: E402
from skillweaver.perception.labeling import recall_at_iou  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402
from skillweaver.perception.screenshot import save_screenshot  # noqa: E402

SITES = (
    "https://example.com",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://news.ycombinator.com",
    "https://httpbin.org/forms/post",
    "https://www.w3schools.com/html/html_forms.asp",
    "https://duckduckgo.com",
)

INTERACTIVE = frozenset(
    {
        ElementKind.button,
        ElementKind.text_field,
        ElementKind.checkbox,
        ElementKind.radio,
        ElementKind.link,
        ElementKind.menu,
        ElementKind.tab,
    }
)


def inside(box: Box, point) -> bool:
    return box.x <= point.x <= box.x + box.w and box.y <= point.y <= box.y + box.h


def reachable(truth: list[Element], observed) -> tuple[int, int, list[str]]:
    """How many labelled controls the agent could actually click, and which it could not.

    Asked exactly the way a skill asks: look the label up by text, take the first hit,
    and check the point that would be clicked lands inside the real control.
    """
    hits = misses = 0
    lost: list[str] = []
    for element in truth:
        label = element.text.strip()
        if element.kind not in INTERACTIVE or len(label) < 2:
            continue
        found = observed.index.find_text(label)
        if found and inside(element.box, found[0].box.center):
            hits += 1
        else:
            misses += 1
            if len(lost) < 4:
                lost.append(f"{element.kind.value} {label[:28]!r}")
    return hits, misses, lost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    parser.add_argument("--shots", type=Path, default=None, help="save each screenshot here")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "data" / "models" / "ui_detector.pt",
        help="which detector to measure, so two of them can be compared on one page set",
    )
    args = parser.parse_args()

    controller = BrowserController(headless=True)
    perceiver = ComposedPerceiver(YoloDetector(args.weights), RapidOcrReader())
    truth_source = BrowserGroundTruth(controller)

    rows = []
    try:
        for url in args.sites:
            result = controller.perform(Navigate(url))
            if not result.ok:
                print(f"  {url}: could not load ({result.error})")
                continue
            controller.perform(Navigate(url))  # settle a page that redirected
            time.sleep(1.0)
            started = time.perf_counter()
            observed = perceiver.observe(controller)
            ms = (time.perf_counter() - started) * 1000.0
            truth = list(truth_source.elements())
            controls = [e for e in truth if e.kind in INTERACTIVE]
            found = [e for e in observed.elements]

            recall = recall_at_iou(controls, found, iou=0.5, match_kind=False)
            hits, misses, lost = reachable(truth, observed)
            total = hits + misses
            rows.append(
                {
                    "url": url,
                    "dom_controls": len(controls),
                    "labelled": total,
                    "seen": len(found),
                    "control_recall": round(recall, 3),
                    "reachable": round(hits / total, 3) if total else None,
                    "observe_ms": round(ms, 1),
                    "unreachable": lost,
                }
            )
            if args.shots is not None:
                args.shots.mkdir(parents=True, exist_ok=True)
                name = url.split("//")[-1].replace("/", "_")[:50]
                save_screenshot(observed.screenshot, args.shots / f"{name}.png")
    finally:
        controller.close()

    header = (
        f"{'site':<44} {'DOM ctl':>8} {'seen':>6} {'recall':>7} {'reachable':>10} {'observe':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        reach = "—" if row["reachable"] is None else f"{row['reachable'] * 100:.0f}%"
        print(
            f"{row['url'][:44]:<44} {row['dom_controls']:>8} {row['seen']:>6} "
            f"{row['control_recall'] * 100:>6.0f}% {reach:>10} {row['observe_ms']:>8.0f}m"
        )
    usable = [r for r in rows if r["reachable"] is not None]
    if usable:
        recall = statistics.fmean(r["control_recall"] for r in rows) * 100
        reach = statistics.fmean(r["reachable"] for r in usable) * 100
        print(f"\nmean control recall {recall:.0f}%   mean reachable {reach:.0f}%")
    for row in rows:
        if row["unreachable"]:
            print(f"\n  could not reach on {row['url'][:40]}: {', '.join(row['unreachable'])}")

    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
