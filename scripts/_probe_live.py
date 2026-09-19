"""Probe the live sandbox through the real perception stack and dump what it sees.

A development aid for authoring the live-benchmark playbooks: walks the app's
screens the way the agent would - clicks found by text, typing into clicked
fields - and prints every merged element plus the server's own state after each
step. Not part of the benchmark itself.

    python3 apps/sandbox-site/serve.py --port 8765 &
    uv run python scripts/_probe_live.py > /tmp/probe.txt
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from skillweaver.contracts import Click, Point, TypeText  # noqa: E402
from skillweaver.controllers.browser import BrowserController  # noqa: E402
from skillweaver.orchestrator import ComposedPerceiver  # noqa: E402
from skillweaver.perception.detect_yolo import YoloDetector  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402

BASE = "http://127.0.0.1:8765"


def state() -> dict:
    with urllib.request.urlopen(f"{BASE}/__state") as r:
        return json.loads(r.read())


def reset() -> None:
    urllib.request.urlopen(f"{BASE}/__reset").read()


def main() -> int:
    ctl = BrowserController(headless=True, start_url=f"{BASE}/")
    per = ComposedPerceiver(YoloDetector("data/models/ui_detector.pt"), RapidOcrReader())

    def observe():
        time.sleep(0.25)
        return per.observe(ctl)

    def dump(tag, obs):
        ui = state()["ui"]
        print(f"\n===== {tag} | ui={json.dumps(ui)[:220]}")
        for e in obs.elements:
            print(
                f"  {e.kind.value:<10} {e.text[:70]!r} @({e.box.x},{e.box.y}) {e.box.w}x{e.box.h}"
            )

    def click_text(obs, query, *, kind=None, index=0):
        hits = obs.index.find_text(query, kind=kind)
        if not hits:
            print(f"  !! no hit for {query!r}")
            return obs
        ctl.perform(Click(hits[index].box.center))
        return observe()

    reset()
    ctl.perform(Click(Point(640, 400)))  # nothing; warm the pipeline
    obs = observe()
    dump("MAIL HOME", obs)

    # open the Billing message
    obs = click_text(obs, "Invoice 4471")
    dump("MESSAGE OPEN (m03)", obs)

    # label menu on the open message
    obs = click_text(obs, "Label")
    dump("LABEL MENU OPEN", obs)

    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    # search: click the search field then type
    obs = click_text(obs, "Search mail")
    ctl.perform(TypeText("invoice"))
    obs = observe()
    dump("MAIL SEARCHED 'invoice'", obs)

    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    obs = click_text(obs, "Compose")
    dump("COMPOSE OPEN", obs)

    # records
    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    obs = click_text(obs, "Records")
    dump("RECORDS", obs)

    obs = click_text(obs, "Filter records")
    ctl.perform(TypeText("storage"))
    obs = observe()
    dump("RECORDS FILTERED 'storage'", obs)

    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    obs = click_text(obs, "Records")
    obs = click_text(obs, "PRIORITY")
    dump("RECORDS SORT PRIORITY (1 click)", obs)
    obs = click_text(obs, "PRIORITY")
    dump("RECORDS SORT PRIORITY (2 clicks)", obs)

    # rename flow: double-click? records.editStart - inspect row click behavior
    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    obs = click_text(obs, "Records")
    obs = click_text(obs, "Aurora Ledger")
    dump("RECORDS AFTER CLICK ROW NAME", obs)

    # bulk menu
    obs = click_text(obs, "Bulk actions")
    dump("RECORDS BULK MENU", obs)

    # settings
    reset()
    ctl.perform(Click(Point(640, 16)))
    obs = observe()
    obs = click_text(obs, "Settings", kind=None)
    dump("SETTINGS", obs)

    obs = click_text(obs, "Avery Quinn")
    dump("SETTINGS AFTER CLICK NAME FIELD", obs)

    ctl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
