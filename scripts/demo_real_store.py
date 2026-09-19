"""Buy something on a real storefront, by sight, and let the shop say whether it worked.

    uv run python scripts/demo_real_store.py --weights /path/to/detector.pt
    uv run python scripts/demo_real_store.py --watch          # a browser you can see

Six screens end to end: sign in, find a named product among several, add it to the
cart, open a cart whose icon has no text at all, fill in a three-field checkout, and
confirm. Fourteen actions. Nothing is addressed by a selector, an id or the DOM - every
target is found in the screenshot, by the words printed on it or by where it sits.

Whether it worked is not this script's opinion. The shop lands on its own confirmation
page and says "Thank you for your order!", and the cart badge goes back to empty.

Why THIS shop
-------------

``saucedemo.com`` is Sauce Labs' demonstration storefront: a real website, built by
somebody else, in a real front-end framework, published specifically so that automation
can be pointed at it, with its own credentials printed on its login page. So the flow
is genuine and the purchase costs nobody anything.

A real merchant would be the wrong place to prove this and not much of a proof anyway:
it would spend someone's money, create an obligation with a third party that cannot be
withdrawn, and require credentials that are not mine to use. The interesting question -
can it drive six screens of a site nobody built for it, from pixels - is answered here
without any of that.

What this does and does not show
--------------------------------

It shows the WARM half of the claim on a real site: a stored skill, run in the sandbox,
through the eyes, doing a complete errand. It does not show the cold half - the agent
working the flow out for itself - because that needs a computer-use model and an API
key. The skill below is written the way the synthesizer writes them, but a person wrote
this one.
"""

from __future__ import annotations

import argparse
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
from skillweaver.skills.api import SkillLimits  # noqa: E402
from skillweaver.skills.model import make_skill  # noqa: E402
from skillweaver.skills.sandbox import SkillRunner  # noqa: E402

STORE = "https://www.saucedemo.com"

CODE = '''
def run(ctx, user, password, product, first, last, postcode):
    def says(text, what):
        hits = ctx.see.find_text(text)
        ctx.expect(bool(hits), "nothing on screen saying " + what)
        return hits[0]

    def put(text, value):
        """A box that shows its own placeholder is found by reading it."""
        ctx.ctl.click(says(text, text))
        ctx.ctl.type_text(value)

    def owned_by(label, title):
        """The control belonging to one product card, among several that match.

        A storefront repeats every control once per product, so the words never say
        which one. Below that product's name, and within its column, does.
        """
        best = None
        for element in ctx.see.find_text(label):
            drop = element.box.y - title.box.y
            across = element.box.x - title.box.x
            if drop < 20 or drop > 220 or across < 0 or across > 240:
                continue
            if best is None or element.box.y < best.box.y:
                best = element
        ctx.expect(best is not None, "no " + label + " belonging to " + title.text)
        return best

    def corner():
        """The cart: the control in the top-right, which carries no text at all.

        Its badge is a number in a coloured circle and OCR does not read it, so there
        is nothing to look up. Where it is, is the whole description - which is also
        how a person finds a cart icon.
        """
        best = None
        for element in ctx.see.all():
            if element.box.y > 70 or element.kind.value not in ("link", "button", "icon"):
                continue
            if best is None or element.box.x > best.box.x:
                best = element
        ctx.expect(best is not None, "nothing in the top-right corner")
        return best

    put("Username", user)
    put("Password", password)
    ctx.ctl.click(says("Login", "the login button"))

    ctx.ctl.click(owned_by("Add to cart", says(product, product)))
    ctx.ctl.click(corner())
    ctx.ctl.click(says("Checkout", "the checkout button"))

    put("First Name", first)
    put("Last Name", last)
    put("Zip/Postal Code", postcode)
    ctx.ctl.click(says("Continue", "the continue button"))
    ctx.ctl.click(says("Finish", "the finish button"))
    return product
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "data" / "models" / "ui_detector.pt",
        help="the detector to use. The sandbox-only one cannot do this page.",
    )
    parser.add_argument("--product", default="Sauce Labs Backpack")
    parser.add_argument("--watch", action="store_true", help="open a real browser window")
    args = parser.parse_args()

    print(f"shop     : {STORE}   (Sauce Labs' public demo storefront)")
    print(f"detector : {args.weights}")
    print(f"errand   : sign in, buy the {args.product!r}, check out - by sight\n")

    controller = BrowserController(headless=not args.watch)
    perceiver = ComposedPerceiver(YoloDetector(args.weights), RapidOcrReader())
    try:
        controller.perform(Navigate(STORE))
        time.sleep(1.5)
        skill = make_skill(
            name="buy_from_the_store",
            domain="saucedemo.com",
            summary="Sign in, add a named product to the cart and complete the checkout.",
            docstring="Assumes the sign-in page. Ends on the order confirmation.",
            params={
                key: {"type": "string"}
                for key in ("user", "password", "product", "first", "last", "postcode")
            },
            code=CODE,
            verifier_code=(
                "def verify(ctx, result):\n"
                '    return bool(ctx.see.find_text("Thank you for your order"))\n'
            ),
            provenance=Provenance("demo", "buy a backpack", "hand-written", utcnow()),
        )
        runner = SkillRunner(None)
        context = runner.context(
            controller,
            perceiver,
            domain="saucedemo.com",
            # A real site over the network is slower than a local sandbox, and this
            # errand is six screens rather than one.
            limits=SkillLimits(max_steps=40, max_seconds=180.0),
        )
        started = time.perf_counter()
        outcome = runner.run(
            skill,
            {
                "user": "standard_user",
                "password": "secret_sauce",
                "product": args.product,
                "first": "Ada",
                "last": "Lovelace",
                "postcode": "SW1A 1AA",
            },
            context,
        )
        seconds = time.perf_counter() - started

        print(f"skill    : ok={outcome.ok}  actions={outcome.steps}  {seconds:.1f}s")
        if outcome.error:
            print(f"           {outcome.error}")
        time.sleep(1.0)
        print(f"landed on: {controller.url()}")

        page = _text(controller)
        done = "thank you for your order" in page.casefold()
        print("\nwhat the shop says:")
        for line in [ln.strip() for ln in page.splitlines() if ln.strip()][:5]:
            print(f"    {line[:70]}")
        print(f"\n{'the order was placed' if done else 'the order was NOT placed'}")
        return 0 if done and outcome.ok else 1
    finally:
        controller.close()


def _text(controller: BrowserController) -> str:
    page = controller._page  # noqa: SLF001 - a demo, reading what the shop replied
    return "" if page is None else page.inner_text("body")


if __name__ == "__main__":
    raise SystemExit(main())
