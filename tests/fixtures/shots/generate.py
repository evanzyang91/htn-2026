"""Regenerate the committed perception fixtures in this directory.

    uv run python tests/fixtures/shots/generate.py

The perception tests need screenshots that are *real* pixels - something OCR can
actually be asked to read - while still having exact, trustworthy expectations. So the
fixtures are rendered from HTML written for the purpose in ``pages/``: Chromium draws
them, the DOM hands back every element's true box in logical pixels, and both the PNGs
and the expectations are committed.

The pages are absolutely positioned and use only Helvetica/Arial at large sizes, so the
ground-truth boxes do not drift between Chromium versions or machines.

Each page is rendered at every scale in its spec. A ``@2x`` render is the important one:
it is a Retina capture, its PNG is twice the logical size, and the expectations are
still in LOGICAL pixels - which is exactly the mistake this project cannot afford to
make, so a fixture exists to catch it.

Before writing anything the script runs the real OCR reader over each render and checks
that every string in ``expect_text`` comes back. A committed expectation is therefore
known to be achievable, not merely hoped for. Pass ``--no-verify`` to skip that (and
find out in CI instead).

Output, all in this directory:
    ``<name>@<scale>x.png``   the screenshots
    ``expectations.json``     sizes, scales, ground-truth elements, expected strings

``pairs/`` belongs to another piece of work and is never touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

EXPECTATIONS = HERE / "expectations.json"

# Every element's true box comes from the DOM, so this only has to list the strings OCR
# is REQUIRED to recover. They are checked at generation time.
_BOX_JS = """
() => Array.from(document.querySelectorAll('[data-kind]')).map((el) => {
  const r = el.getBoundingClientRect();
  return {
    kind: el.dataset.kind,
    text: (el.innerText || '').replace(/\\s+/g, ' ').trim(),
    box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
  };
});
"""


@dataclass(frozen=True)
class PageSpec:
    name: str
    html: str
    width: int
    height: int
    scales: tuple[float, ...]
    expect_text: tuple[str, ...]


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        name="invoices",
        html="pages/invoices.html",
        width=800,
        height=600,
        scales=(1.0, 2.0),
        expect_text=(
            "Open Invoices",
            "Acme Corporation billing",
            "Search invoices",
            "New Invoice",
            "INV-1042",
            "Globex",
            "Initech",
            "Umbrella",
            "Download report",
            "Billing settings",
            "Send Reminders",
            "invoices overdue",
        ),
    ),
    PageSpec(
        name="login",
        html="pages/login.html",
        width=640,
        height=480,
        scales=(1.0,),
        expect_text=(
            "Sign in to Ledger",
            "Email address",
            "Password",
            "Enter password",
            "Keep me signed in",
            "Forgot password",
        ),
    ),
)


def _shot_name(spec: PageSpec, scale: float) -> str:
    suffix = f"{scale:g}".replace(".", "_")
    return f"{spec.name}@{suffix}x"


def render(spec: PageSpec, scale: float, browser) -> dict:
    """Render one page at one device scale factor; write the PNG, return its record."""
    context = browser.new_context(
        viewport={"width": spec.width, "height": spec.height},
        device_scale_factor=scale,
    )
    page = context.new_page()
    page.goto((HERE / spec.html).resolve().as_uri())
    page.wait_for_load_state("load")
    elements = page.evaluate(_BOX_JS)
    png = page.screenshot(type="png")
    context.close()

    name = _shot_name(spec, scale)
    (HERE / f"{name}.png").write_bytes(png)
    return {
        "name": name,
        "png": f"{name}.png",
        "page": spec.html,
        "width": spec.width,
        "height": spec.height,
        "scale": scale,
        "expect_text": list(spec.expect_text),
        "elements": elements,
    }


def _ground_truth_box(record: dict, wanted: str) -> list[int]:
    """The DOM box of the element whose text contains ``wanted``.

    This is what turns a text expectation into a COORDINATE expectation: the test can
    then check that the OCR box for a string lands on the thing that prints it, which is
    the only way to catch a ``@2x`` fixture read without dividing by ``scale``.
    """
    from skillweaver.perception.elements import normalize_text

    target = normalize_text(wanted)
    hits = [el for el in record["elements"] if target in normalize_text(el["text"])]
    if not hits:
        raise SystemExit(f"{record['name']}: no element on the page contains {wanted!r}")
    # Prefer the tightest element that says it, so "Search" resolves to the button and
    # not to the "Search invoices" field beside it.
    hits.sort(key=lambda el: (len(normalize_text(el["text"])), el["box"][2] * el["box"][3]))
    return hits[0]["box"]


def verify(record: dict) -> list[str]:
    """Check the record against real OCR, and fill in each expectation's details.

    Expectations are checked the way a caller would use them - through
    :class:`ElementIndex.find_text` over the OCR output - because that is the behavior
    the pipeline depends on, and because real OCR drops the space between words often
    enough ("sendreminders") that a verbatim-only check would just be a worse test of a
    fuzzy matcher. ``verbatim`` records which strings OCR did get exactly right at
    generation time, so the committed expectations still pin down raw OCR quality.

    Returns the expectation strings that could not be recovered at all.
    """
    from skillweaver.perception.elements import ElementIndex
    from skillweaver.perception.ocr import RapidOcrReader
    from skillweaver.perception.screenshot import load_screenshot

    shot = load_screenshot(
        HERE / record["png"],
        scale=record["scale"],
        width=record["width"],
        height=record["height"],
    )
    found = RapidOcrReader().read(shot)
    index = ElementIndex(found)
    haystack = " | ".join(e.text.casefold() for e in found)

    missing: list[str] = []
    detailed: list[dict] = []
    for wanted in record["expect_text"]:
        hits = index.find_text(wanted)
        box = _ground_truth_box(record, wanted)
        ok = bool(hits) and _overlaps(hits[0].box, box)
        if not ok:
            missing.append(wanted)
        detailed.append(
            {
                "text": wanted,
                "near": box,
                "verbatim": wanted.casefold() in haystack,
            }
        )
    record["expect_text"] = detailed
    record["ocr_text"] = [e.text for e in found]
    verbatim = sum(1 for d in detailed if d["verbatim"])
    print(
        f"  {record['name']}: {len(found)} text elements, "
        f"{verbatim}/{len(detailed)} verbatim, {len(missing)} unrecoverable"
    )
    if missing:
        print(f"    OCR saw: {haystack}")
    return missing


def _overlaps(found, box: list[int]) -> bool:
    """Whether an OCR box lands on the ground-truth box that prints the string."""
    x, y, w, h = box
    ix = max(0, min(found.x + found.w, x + w) - max(found.x, x))
    iy = max(0, min(found.y + found.h, y + h) - max(found.y, y))
    smaller = min(found.w * found.h, w * h)
    return smaller > 0 and (ix * iy) / smaller >= 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-verify", action="store_true", help="skip the OCR check")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    records: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for spec in PAGES:
                for scale in spec.scales:
                    record = render(spec, scale, browser)
                    print(f"rendered {record['png']} ({len(record['elements'])} elements)")
                    records.append(record)
        finally:
            browser.close()

    failures: dict[str, list[str]] = {}
    if not args.no_verify:
        print("verifying with the real OCR reader...")
        for record in records:
            missing = verify(record)
            if missing:
                failures[record["name"]] = missing

    EXPECTATIONS.write_text(
        json.dumps(
            {
                "generated_by": "tests/fixtures/shots/generate.py",
                "note": (
                    "Boxes and sizes are LOGICAL pixels. Load a shot with "
                    "perception.screenshot.load_screenshot(path, scale=shot['scale'])."
                ),
                "shots": records,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    print(f"wrote {EXPECTATIONS.relative_to(REPO)}")

    if failures:
        print("\nOCR did not recover every expectation:")
        for name, missing in failures.items():
            print(f"  {name}: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
