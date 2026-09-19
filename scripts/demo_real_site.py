"""Do one errand on a REAL website, and show what the detector's training decides.

    uv run python scripts/demo_real_site.py
    uv run python scripts/demo_real_site.py --weights /path/to/other.pt

The errand is a form on ``httpbin.org``, a public service that exists to be posted to:
type a customer name, choose a pizza size, press Submit. The skill below is an ordinary
one - the same shape the agent writes for itself - and it is run in the same sandbox,
through the same eyes, with no selectors and no DOM. Whether it worked is decided by
httpbin, which echoes the submitted form back as JSON.

Run with two different detectors and the point makes itself: the one trained only on
this repository's sandbox application cannot find a blank input on a plain real form,
and the one trained on live pages can. A blank input has no text for OCR to read, so it
is the one case where everything rests on detection - which is exactly where training
on one website shows up as not working anywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from skillweaver.contracts import Navigate, Provenance, utcnow  # noqa: E402
from skillweaver.controllers.browser import BrowserController  # noqa: E402
from skillweaver.orchestrator import ComposedPerceiver  # noqa: E402
from skillweaver.perception.detect_yolo import YoloDetector  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402
from skillweaver.skills.model import make_skill  # noqa: E402
from skillweaver.skills.sandbox import SkillRunner  # noqa: E402

FORM = "https://httpbin.org/forms/post"

CODE = '''
def run(ctx, customer, size):
    def beside(caption, gap):
        """The field a caption names, found by where it is rather than what it says.

        A blank input has no text at all, so nothing can be looked up about it. What
        can be said is where it sits: on the caption's line, to its right, and wide
        enough to type into.
        """
        labels = ctx.see.find_text(caption)
        ctx.expect(bool(labels), "no caption reading " + caption)
        label = labels[0]
        best = None
        for element in ctx.see.all():
            if abs(element.box.y - label.box.y) > gap:
                continue
            across = element.box.x - label.box.x
            if across < 20 or across > 400 or element.box.w < 60:
                continue
            if best is None or element.box.x < best.box.x:
                best = element
        ctx.expect(best is not None, "no field beside " + caption)
        return best

    ctx.ctl.click(beside("Customer name", 16))
    ctx.ctl.type_text(customer)
    ctx.ctl.click(ctx.see.find_text(size)[0])
    ctx.ctl.click(ctx.see.find_text("Submit order")[0])
    return customer
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "data" / "models" / "ui_detector.pt",
        help="the detector to use; point it at another to compare two",
    )
    parser.add_argument("--customer", default="Ada Lovelace")
    parser.add_argument("--size", default="Medium")
    parser.add_argument("--watch", action="store_true", help="open a real browser window")
    args = parser.parse_args()

    print(f"detector : {args.weights}")
    print(f"page     : {FORM}")
    print("task     : type a name, choose a size, submit - by sight, with no selectors\n")

    controller = BrowserController(headless=not args.watch)
    perceiver = ComposedPerceiver(YoloDetector(args.weights), RapidOcrReader())
    try:
        controller.perform(Navigate(FORM))
        time.sleep(1.2)
        skill = make_skill(
            name="order_pizza",
            domain="httpbin.org",
            summary="Fill in the order form and submit it.",
            docstring="Assumes the form is on screen. Ends on the posted result.",
            params={"customer": {"type": "string"}, "size": {"type": "string"}},
            code=CODE,
            verifier_code=None,
            provenance=Provenance("demo", "order a pizza", "hand-written", utcnow()),
        )
        runner = SkillRunner(None)
        outcome = runner.run(
            skill,
            {"customer": args.customer, "size": args.size},
            runner.context(controller, perceiver, domain="httpbin.org"),
        )
        print(f"skill    : ok={outcome.ok} actions={outcome.steps}")
        if outcome.error:
            print(f"           {outcome.error}")
        time.sleep(1.5)

        print(f"landed on: {controller.url()}")
        body = controller.page_text() if hasattr(controller, "page_text") else ""
        posted = _posted(body or _read_body(controller))
        if posted is None:
            print("\nthe form was NOT submitted - the page is still the form")
            return 1
        print("\nhttpbin echoed the submission back:")
        for key, value in posted.items():
            if value:
                print(f"    {key} = {value!r}")
        ok = posted.get("custname") == args.customer
        print(f"\n{'the name arrived at the server' if ok else 'the name did NOT arrive'}")
        return 0 if ok else 1
    finally:
        controller.close()


def _read_body(controller: BrowserController) -> str:
    page = controller._page  # noqa: SLF001 - a demo, reading what the server sent back
    return "" if page is None else page.inner_text("body")


def _posted(body: str) -> dict | None:
    """The ``form`` object out of httpbin's JSON reply, or ``None`` if there is none."""
    try:
        return json.loads(body).get("form")
    except (ValueError, AttributeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
