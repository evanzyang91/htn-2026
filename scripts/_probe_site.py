"""Print what the real perception stack sees on a sandbox application.

A development aid for writing a site's playbooks: it opens the page through the
project's own controller and perceiver and prints every merged element, so a step can
be written against what is ACTUALLY reported rather than against the markup.

    uv run python scripts/_probe_site.py http://127.0.0.1:8766/ [--click "Add to cart"]...

Each ``--click`` is a text to find and click before printing, applied in order, so a
screen two clicks deep can be inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from skillweaver.contracts import Click, Navigate, TypeText  # noqa: E402
from skillweaver.controllers.browser import BrowserController  # noqa: E402
from skillweaver.orchestrator import ComposedPerceiver  # noqa: E402
from skillweaver.perception.detect_yolo import YoloDetector  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--click", action="append", default=[], help="text to click, in order")
    parser.add_argument("--type", action="append", default=[], help="text to type after a click")
    parser.add_argument("--state", default="", help="dotted path into /__state to print")
    parser.add_argument("--reset", action="store_true", help="reset the app first")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    if args.reset:
        urllib.request.urlopen(f"{base}/__reset").read()

    controller = BrowserController(headless=True, start_url=f"{base}/")
    perceiver = ComposedPerceiver(
        YoloDetector(REPO / "data" / "models" / "ui_detector.pt"), RapidOcrReader()
    )
    observation = perceiver.observe(controller)

    typing = list(args.type)
    for wanted in args.click:
        hits = observation.index.find_text(wanted)
        if not hits:
            print(f"!! nothing matching {wanted!r}")
            break
        print(f">> click {wanted!r} -> {hits[0].kind.value} {hits[0].text[:50]!r} {hits[0].box}")
        controller.perform(Click(hits[0].box.center))
        if typing:
            controller.perform(TypeText(typing.pop(0)))
        observation = perceiver.observe(controller)

    print(f"\n=== {observation.url} : {len(observation.elements)} element(s)")
    for element in observation.elements:
        box = element.box
        print(
            f"  {element.kind.value:<11} {element.text[:66]!r:<70} "
            f"@({box.x},{box.y}) {box.w}x{box.h}"
        )

    if args.state:
        with urllib.request.urlopen(f"{base}/__state") as response:
            state = json.loads(response.read())
        for part in args.state.split("."):
            state = state[part]
        print(f"\n=== state.{args.state} = {json.dumps(state)[:600]}")

    controller.perform(Navigate(f"{base}/"))
    controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
