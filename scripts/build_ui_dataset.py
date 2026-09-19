#!/usr/bin/env python3
"""Harvest a YOLO training set for the UI element detector, with no hand labelling.

The trick that makes this tractable in a hackathon is that the labels are free:
``BrowserGroundTruth`` reads the DOM and hands back exactly the boxes and kinds a
detector is supposed to learn, in the same logical-pixel space the agent clicks in.
So this script drives the sandbox app through a list of states, screenshots each
one, and writes the truth next to the pixels. Nobody draws a rectangle by hand.

``BrowserGroundTruth`` is an OFFLINE TEACHER. It is used here and in evaluation and
nowhere near the agent; that separation is the point of the project.

What varies, and why
--------------------
A detector trained on one screen learns that screen. Three axes of variation keep
the set honest:

* **App state** - :data:`RECIPES` drive the sandbox through menus, dialogs, empty
  filters, selections and banners, so controls move, appear and change enabled
  look. Each recipe is a list of ``/api/act`` calls, applied server-side and then
  loaded, which is far faster and more reproducible than clicking through the UI.
* **Viewport** - :data:`VIEWPORTS` change the layout: the mail list reflows, the
  records table gains and loses columns of whitespace.
* **Scroll** - long surfaces are captured at more than one offset, so an element
  is seen near the top, the middle and the bottom of a frame.

One viewport is held out entirely for validation, so the reported metric is
"unseen layout width", not "frames the model has already memorized". That is a
weaker claim than a fully unseen app would give; the caveat is deliberate and
stated rather than hidden in a shuffled split.

Run it::

    uv run python scripts/build_ui_dataset.py --out data/models/ui-dataset

It starts and stops its own copy of the sandbox server unless ``--url`` names one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # allow `python scripts/...` without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from skillweaver.contracts import (  # noqa: E402
    Box,
    Element,
    ElementKind,
    ElementSource,
    Navigate,
    Point,
    Screenshot,
    Scroll,
)
from skillweaver.controllers.browser import BrowserController, BrowserGroundTruth  # noqa: E402
from skillweaver.perception.labeling import (  # noqa: E402
    CLASS_NAMES,
    dataset_yaml,
    to_label_text,
)

SANDBOX = REPO_ROOT / "apps" / "sandbox-site" / "serve.py"

Act = tuple[str, dict]


@dataclass(frozen=True)
class Recipe:
    """One app state to photograph.

    Attributes:
        name: Goes into the file name, so a bad label can be traced back to a state.
        acts: ``(action, payload)`` pairs posted to ``/api/act`` in order, starting
            from a freshly reset app.
        scrolls: Vertical wheel offsets in logical pixels to capture at, ``0``
            meaning the top of the page. Every recipe is captured at every offset.
    """

    name: str
    acts: tuple[Act, ...] = ()
    scrolls: tuple[int, ...] = (0,)


_MAIL_IDS = ("m01", "m02", "m03", "m04", "m05", "m06")
_ROW_IDS = tuple(f"r{n:02d}" for n in range(1, 41))

RECIPES: tuple[Recipe, ...] = (
    # -- mail: sidebar, list, reading pane, compose, menus --------------------------
    Recipe("mail_inbox", (), (0, 240)),
    Recipe("mail_search_hit", (("mail.search", {"q": "invoice"}),)),
    Recipe("mail_search_empty", (("mail.search", {"q": "zzzz"}),)),
    Recipe(
        "mail_selected",
        (("mail.toggleSelect", {"id": "m01"}), ("mail.toggleSelect", {"id": "m03"})),
        (0, 200),
    ),
    Recipe("mail_select_all", (("mail.selectAll", {"ids": list(_MAIL_IDS), "checked": True}),)),
    Recipe(
        "mail_label_menu",
        (("mail.toggleSelect", {"id": "m02"}), ("mail.toggleLabelMenu", {})),
    ),
    Recipe("mail_reading", (("mail.open", {"id": "m01"}),), (0, 180)),
    Recipe("mail_reading_deep", (("mail.open", {"id": "m05"}),), (0, 320)),
    Recipe("mail_compose_blank", (("mail.composeOpen", {}),)),
    Recipe(
        "mail_compose_filled",
        (
            ("mail.composeOpen", {}),
            ("mail.composeField", {"field": "to", "value": "dana.whitfield@northwind.example"}),
            ("mail.composeField", {"field": "subject", "value": "Re: capacity plan"}),
            ("mail.composeField", {"field": "body", "value": "Approved - go ahead and lock it."}),
        ),
    ),
    Recipe(
        "mail_banner",
        (("mail.toggleSelect", {"id": "m04"}), ("mail.archive", {})),
    ),
    Recipe("mail_archived", (("mail.folder", {"folder": "archived"}),)),
    Recipe(
        "mail_sent",
        (
            ("mail.composeOpen", {}),
            ("mail.composeField", {"field": "to", "value": "priya.raman@northwind.example"}),
            ("mail.composeField", {"field": "subject", "value": "Checklist looks good"}),
            ("mail.send", {}),
            ("mail.folder", {"folder": "sent"}),
        ),
    ),
    # -- records: a dense table, sorting, inline edit, bulk menu --------------------
    Recipe("rec_default", (("nav", {"screen": "records"}),), (0, 400, 900)),
    Recipe(
        "rec_filtered",
        (("nav", {"screen": "records"}), ("records.filter", {"q": "north"})),
    ),
    Recipe(
        "rec_filter_empty",
        (("nav", {"screen": "records"}), ("records.filter", {"q": "qqqq"})),
    ),
    Recipe(
        "rec_sorted",
        (
            ("nav", {"screen": "records"}),
            ("records.sort", {"key": "status"}),
            ("records.sort", {"key": "status"}),
        ),
        (0, 520),
    ),
    Recipe(
        "rec_sorted_priority",
        (("nav", {"screen": "records"}), ("records.sort", {"key": "priority"})),
        (0, 700),
    ),
    Recipe(
        "rec_selected",
        (
            ("nav", {"screen": "records"}),
            ("records.toggleSelect", {"id": "r02"}),
            ("records.toggleSelect", {"id": "r05"}),
            ("records.toggleSelect", {"id": "r09"}),
        ),
        (0, 360),
    ),
    Recipe(
        "rec_bulk_menu",
        (
            ("nav", {"screen": "records"}),
            ("records.toggleSelect", {"id": "r03"}),
            ("records.toggleBulkMenu", {}),
        ),
    ),
    Recipe(
        "rec_select_all",
        (
            ("nav", {"screen": "records"}),
            ("records.selectAll", {"ids": list(_ROW_IDS), "checked": True}),
        ),
        (0, 640),
    ),
    Recipe(
        "rec_editing",
        (("nav", {"screen": "records"}), ("records.editStart", {"id": "r04"})),
    ),
    Recipe(
        "rec_banner",
        (
            ("nav", {"screen": "records"}),
            ("records.export", {"ids": list(_ROW_IDS[:6])}),
        ),
        (0, 300),
    ),
    # -- settings: cards, selects, toggles, a modal dialog --------------------------
    Recipe("set_default", (("nav", {"screen": "settings"}),), (0, 260)),
    Recipe(
        "set_edited",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "displayName", "value": "Avery Q."}),
            ("settings.field", {"field": "density", "value": "compact"}),
            ("settings.field", {"field": "notifyDesktop", "value": True}),
            ("settings.field", {"field": "autoArchive", "value": True}),
        ),
        (0, 260),
    ),
    Recipe(
        "set_dialog",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "timezone", "value": "Europe/Berlin"}),
            ("settings.save", {}),
        ),
    ),
    Recipe(
        "set_saved",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "weeklyDigest", "value": False}),
            ("settings.save", {}),
            ("settings.confirm", {}),
        ),
    ),
)
"""Every app state the dataset covers: 27 states across the three surfaces."""


@dataclass(frozen=True)
class Viewport:
    """One browser geometry, and whether its frames go to train or validation."""

    width: int
    height: int
    scale: float = 1.0
    split: str = "train"

    @property
    def tag(self) -> str:
        return f"{self.width}x{self.height}@{self.scale:g}x"


VIEWPORTS: tuple[Viewport, ...] = (
    Viewport(1280, 800),
    Viewport(1024, 768),
    Viewport(1440, 900),
    Viewport(1120, 700, scale=2.0),
    Viewport(1600, 1000),
    Viewport(1180, 860, split="val"),
)
"""Layouts to photograph. One is held out for validation, and one trains at 2x so
the model sees Retina-sharp pixels as well as ordinary ones - normalized labels are
identical either way, which is what makes mixing them free."""

MIN_SIDE = 4
"""Boxes thinner than this in logical pixels are noise: hairline rules, spacers."""

MAX_AREA_FRACTION = 0.75
"""Anything covering more of the frame than this is a layout wrapper, not a control.
Training on it teaches the model to predict the whole screen."""


# --------------------------------------------------------------------------------------
# The sandbox app
# --------------------------------------------------------------------------------------


class Sandbox:
    """The sandbox site: either one this script started, or one already running."""

    def __init__(self, url: str, process: subprocess.Popen | None = None) -> None:
        self.url = url.rstrip("/")
        self._process = process

    @classmethod
    def start(cls, port: int, *, timeout_s: float = 20.0) -> Sandbox:
        """Launch ``serve.py`` on ``port`` and wait until it answers.

        Raises:
            RuntimeError: if the server does not come up in time.
        """
        process = subprocess.Popen(
            [sys.executable, str(SANDBOX), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"sandbox server exited with code {process.returncode}")
            try:
                urllib.request.urlopen(f"{url}/__state", timeout=1.0).read()
                return cls(url, process)
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.1)
        process.terminate()
        raise RuntimeError(f"sandbox server did not start on {url} within {timeout_s}s")

    def reset(self) -> None:
        urllib.request.urlopen(f"{self.url}/__reset", timeout=5.0).read()

    def act(self, action: str, payload: dict) -> None:
        body = json.dumps({"action": action, "payload": payload}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/api/act", data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(request, timeout=5.0).read()

    def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
            self._process.kill()
        self._process = None


# --------------------------------------------------------------------------------------
# Filling a gap in the teacher
# --------------------------------------------------------------------------------------

_HIDDEN_CONTROL_JS = """
() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  for (const el of document.querySelectorAll('input[type="checkbox"], input[type="radio"]')) {
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility !== 'visible') continue;
    // Only the ones BrowserGroundTruth drops: a painted control it can see needs
    // no help from here, and picking it up twice would double-label it.
    if (parseFloat(style.opacity || '1') > 0.01) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (rect.right <= 0 || rect.bottom <= 0 || rect.left >= vw || rect.top >= vh) continue;
    out.push({
      kind: el.getAttribute('type').toLowerCase() === 'radio' ? 'radio' : 'checkbox',
      text: (el.getAttribute('aria-label') || '').trim(),
      x: rect.left, y: rect.top, w: rect.width, h: rect.height,
    });
  }
  return out;
}
"""


def hidden_controls(controller: BrowserController) -> list[Element]:
    """Recover checkboxes and toggles the DOM teacher cannot see.

    ``BrowserGroundTruth`` skips anything with ``opacity <= 0.01``, which is right
    for a fading dialog and wrong for the most common checkbox pattern on the web:
    a real ``<input type=checkbox>`` at ``opacity: 0`` laid exactly over a styled
    ``<span>`` that does the painting. The sandbox app uses it for every row
    checkbox and every settings toggle (``apps/sandbox-site/static/js/app.js``), so
    without this the teacher reports zero checkboxes on screens that visibly have a
    dozen, and a detector trained on it could never learn one.

    The input's own rectangle IS the visible control's rectangle in that pattern -
    the proxy is positioned on top of it - so the recovered box needs no adjustment.

    This belongs in ``BrowserGroundTruth``. ``browser.py`` is owned by another piece
    of work, so the gap is reported rather than edited and this script carries the
    workaround meanwhile; reaching for the controller's page is the price, and it
    stays confined to this offline dataset builder.
    """
    page = controller._live_page()  # noqa: SLF001 - see the docstring above
    found: list[Element] = []
    for raw in page.evaluate(_HIDDEN_CONTROL_JS):
        found.append(
            Element(
                box=Box(x=round(raw["x"]), y=round(raw["y"]), w=round(raw["w"]), h=round(raw["h"])),
                kind=ElementKind(raw["kind"]),
                text=raw["text"],
                confidence=1.0,
                source=ElementSource.dom,
            )
        )
    return found


# --------------------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------------------


@dataclass
class Counts:
    """What the run produced, for the summary line and for a sanity check."""

    frames: int = 0
    labels: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    per_split: dict[str, int] = field(default_factory=dict)

    def add(self, split: str, elements: Sequence[Element]) -> None:
        self.frames += 1
        self.labels += len(elements)
        self.per_split[split] = self.per_split.get(split, 0) + 1
        for element in elements:
            self.per_class[element.kind.value] = self.per_class.get(element.kind.value, 0) + 1


def usable_labels(elements: Sequence[Element], viewport: Box) -> list[Element]:
    """Drop the ground-truth boxes a detector should not be asked to learn.

    Three filters, in order of how much damage they would do: sub-pixel slivers
    (a hairline border reported as an element), page-sized wrappers (training on
    "the answer is the whole screen" poisons every other box), and exact duplicates
    where a control and its inner label share a rectangle.
    """
    limit = viewport.area * MAX_AREA_FRACTION
    seen: set[tuple[str, int, int, int, int]] = set()
    kept: list[Element] = []
    for element in elements:
        box = element.box
        if box.w < MIN_SIDE or box.h < MIN_SIDE or box.area > limit:
            continue
        key = (element.kind.value, box.x, box.y, box.w, box.h)
        if key in seen:
            continue
        seen.add(key)
        kept.append(element)
    return kept


def frames(
    sandbox: Sandbox,
    viewport: Viewport,
    *,
    headless: bool,
    recipes: Sequence[Recipe] = RECIPES,
) -> Iterator[tuple[str, Screenshot, list[Element]]]:
    """Yield ``(name, screenshot, labels)`` for every recipe and scroll offset.

    One browser per viewport: Playwright fixes a context's size at creation, and
    re-launching per frame would dominate the runtime.
    """
    with BrowserController(
        headless=headless,
        viewport=(viewport.width, viewport.height),
        device_scale_factor=viewport.scale,
    ) as controller:
        truth = BrowserGroundTruth(controller)
        bounds = Box(0, 0, viewport.width, viewport.height)
        middle = Point(viewport.width // 2, viewport.height // 2)
        for recipe in recipes:
            sandbox.reset()
            for action, payload in recipe.acts:
                sandbox.act(action, payload)
            result = controller.perform(Navigate(sandbox.url))
            if not result.ok:
                raise RuntimeError(f"could not load {sandbox.url}: {result.error}")
            offset = 0
            for target in recipe.scrolls:
                if target != offset:
                    controller.perform(Scroll(middle, dx=0, dy=target - offset))
                    offset = target
                name = f"{recipe.name}__{viewport.tag}__y{target}"
                labels = truth.elements() + hidden_controls(controller)
                yield name, controller.capture(), usable_labels(labels, bounds)


def write_frame(
    root: Path, split: str, name: str, screenshot: Screenshot, elements: Sequence[Element]
) -> None:
    """Write one image and its label file into the ultralytics directory layout.

    The PNG is saved at its native (physical) resolution, and the labels are
    normalized against the LOGICAL size. Those are the same numbers - normalization
    divides the scale out - so a 2x frame needs no special case anywhere.
    """
    (root / "images" / split / f"{name}.png").write_bytes(screenshot.png)
    (root / "labels" / split / f"{name}.txt").write_text(
        to_label_text(elements, screenshot.width, screenshot.height), encoding="utf-8"
    )


def build(out: Path, sandbox: Sandbox, *, headless: bool, viewports: Sequence[Viewport]) -> Counts:
    """Harvest every viewport into a fresh dataset directory under ``out``."""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    counts = Counts()
    for viewport in viewports:
        started = time.monotonic()
        before = counts.frames
        for name, screenshot, elements in frames(sandbox, viewport, headless=headless):
            write_frame(out, viewport.split, name, screenshot, elements)
            counts.add(viewport.split, elements)
        print(
            f"  {viewport.tag:>16} -> {viewport.split:<5} "
            f"{counts.frames - before:3d} frames in {time.monotonic() - started:5.1f}s",
            flush=True,
        )

    (out / "data.yaml").write_text(dataset_yaml(str(out.resolve())), encoding="utf-8")
    return counts


# --------------------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------------------

FIXTURE_VIEWPORTS: tuple[Viewport, ...] = (Viewport(1200, 780), Viewport(1200, 780, scale=2.0))
"""A geometry that appears in neither the training nor the validation split, at both
device scales. Frames captured here are what the committed detector test scores
against, so its recall floor is a claim about an unseen layout, not about a screen
the model has already been shown."""

FIXTURE_FRAMES: tuple[tuple[str, str, int], ...] = (
    ("mail_inbox", "mail_inbox", 0),
    ("rec_selected", "rec_selected", 0),
    ("set_dialog", "set_dialog", 1),
)
"""``(fixture name, recipe name, index into FIXTURE_VIEWPORTS)`` - one frame per
surface, and one of them at 2x so the test exercises the scale conversion."""

FIXTURE_README = """\
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
"""


def build_fixtures(out: Path, sandbox: Sandbox, *, headless: bool) -> None:
    """Write the committed test frames: PNG, YOLO labels, a manifest and a README."""
    out.mkdir(parents=True, exist_ok=True)
    by_recipe = {recipe.name: recipe for recipe in RECIPES}
    manifest: list[dict] = []
    for name, recipe_name, viewport_index in FIXTURE_FRAMES:
        viewport = FIXTURE_VIEWPORTS[viewport_index]
        recipe = replace(by_recipe[recipe_name], scrolls=(0,))
        bounds = Box(0, 0, viewport.width, viewport.height)
        for _, screenshot, elements in frames(
            sandbox, viewport, headless=headless, recipes=(recipe,)
        ):
            (out / f"{name}.png").write_bytes(screenshot.png)
            (out / f"{name}.txt").write_text(
                to_label_text(elements, screenshot.width, screenshot.height), encoding="utf-8"
            )
            manifest.append(
                {
                    "image": f"{name}.png",
                    "labels": f"{name}.txt",
                    "recipe": recipe_name,
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "scale": screenshot.scale,
                    "elements": len(elements),
                }
            )
            print(
                f"  {name:<14} {viewport.tag:>14} {len(elements):4d} labels "
                f"({bounds.w}x{bounds.h} logical)",
                flush=True,
            )
    (out / "frames.json").write_text(
        json.dumps({"frames": manifest}, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(FIXTURE_README, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest a YOLO training set from the sandbox app, labelled by the DOM."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/ui-dataset"),
        help="dataset directory to create (deleted first if it exists)",
    )
    parser.add_argument("--url", help="a sandbox site already running; default is to start one")
    parser.add_argument("--port", type=int, default=8791, help="port for the sandbox we start")
    parser.add_argument("--headed", action="store_true", help="show the browser while harvesting")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="instead of a dataset, write the committed detector test fixtures here",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="two viewports only - a smoke test of the pipeline, not a training set",
    )
    args = parser.parse_args(argv)

    sandbox = Sandbox(args.url) if args.url else Sandbox.start(args.port)
    if args.fixtures is not None:
        try:
            print(f"writing {len(FIXTURE_FRAMES)} test fixtures to {args.fixtures}")
            build_fixtures(args.fixtures, sandbox, headless=not args.headed)
        finally:
            sandbox.close()
        return 0

    viewports = (VIEWPORTS[0], VIEWPORTS[-1]) if args.quick else VIEWPORTS
    started = time.monotonic()
    try:
        print(
            f"harvesting {len(RECIPES)} app states x {len(viewports)} viewports from {sandbox.url}"
        )
        counts = build(args.out, sandbox, headless=not args.headed, viewports=viewports)
    finally:
        sandbox.close()

    print(
        f"\n{counts.frames} frames, {counts.labels} labels "
        f"({counts.labels / max(counts.frames, 1):.1f} per frame) "
        f"in {time.monotonic() - started:.1f}s -> {args.out}"
    )
    print("  split: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.per_split.items())))
    print("  class: " + ", ".join(f"{k}={counts.per_class.get(k, 0)}" for k in CLASS_NAMES))
    missing = [name for name in CLASS_NAMES if not counts.per_class.get(name)]
    if missing:
        print(f"  note: no examples of {', '.join(missing)} - the sandbox app has none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
