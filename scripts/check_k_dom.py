"""Check what ``perception/dom.py`` took from upstream Jev ``1489129..0da4053``, on real pages.

Two things. CONTEXT (``cbf517a``): a control whose label is shared carries the text of the
row or card around it, and one whose label is its own carries none. And the EFFECT WATCH
(``aa13d36``, ``0da4053``): a click or a type that shows no change is polled until the page
differs or the operation's budget is spent, instead of being given a quiet window, while a
scroll or a wait is never polled.

Every page is a small local HTML file read through the real ``DomPerceiver`` and a real
headless ``BrowserController``; the clicks, the typing and the wheel are real. The timings
are wall-clock and the bounds on them are wide on purpose - what is asserted is WHICH
budget was paid and whether the watch stopped early, not a number of milliseconds.

Needs NO network and no ``.env``; it does need the Playwright Chromium this project
already installs. Run: ``uv run python scripts/check_k_dom.py``. ``--harness`` runs the
same pages through ``HarnessBrowserController`` on a private headless ``ChromeProcess``,
exactly as ``scripts/check_a_dom.py`` does and never the person's own Chrome. Prints one
line per check and exits non-zero if any failed.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from skillweaver.contracts import Click, Navigate, PressKey, Scroll, TypeText, Wait
from skillweaver.perception.dom import (
    EFFECT_BUDGET_MS,
    MAX_CONTEXT_CHARS,
    MAX_DESCRIBED,
    DomControl,
    DomPerceiver,
    DomSnapshot,
)

_CARD = (
    '<div class="card" style="border:1px solid #999;margin:4px;padding:4px">'
    "<h3>{name}</h3><span>${price}.99</span> <button>Add item to cart</button></div>"
)
PRODUCTS = ("Sourdough", "Rye loaf", "Baguette", "Focaccia", "Ciabatta", "Brioche", "Pretzel")

CARDS = (
    """<!doctype html><title>cards</title><body style="margin:0">
<header><a href="#a">Deals</a> <a href="#b">Deals</a> <a href="#c">Stores</a>
<a href="#d">Help</a> <a href="#e">Account</a> today only</header>
<main>"""
    + "".join(_CARD.format(name=name, price=n + 2) for n, name in enumerate(PRODUCTS))
    + """<div class="card"><p>Your basket is ready to go</p><button>Checkout</button></div>
<div><button>Bare</button></div><div><button>Bare</button></div>
</main>"""
)
"""Seven same-label buttons in cards; a header whose two *Deals* share a label but sit in a
container of five controls; a unique *Checkout* with card text around it; and two *Bare*
buttons that share a label and have NOTHING around them to say."""

MANY = """<!doctype html><title>many</title><body style="margin:0;font-size:9px">""" + "".join(
    f"<div><span>row number {n}</span> <a href='#h{n}'>hide</a></div>" for n in range(40)
)
"""Forty rows with one shared label: more than :data:`MAX_DESCRIBED`."""

EFFECT = """<!doctype html><title>effect</title><body style="margin:0">
<p>Cart: <span id="badge">0</span></p>
<button id="add" onclick="setTimeout(() => { badge.textContent = +badge.textContent + 1; }, 800)">
Add</button>
<button id="nothing">Nothing</button>
<input id="dead" type="text" aria-label="Dead field"
  onkeydown="event.preventDefault()" onbeforeinput="event.preventDefault()">
"""
"""An effect that lands 800ms after the click with NO mutation before it - the shape of a
cart request being awaited - beside a true no-op and a field that refuses every key."""

_failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f'  [{detail}]' if detail != '' else ''}")
    if not ok:
        _failures.append(name)


def _named(snapshot: DomSnapshot, name: str) -> list[DomControl]:
    return [control for control in snapshot.controls if control.name == name]


def _snap(perceiver: DomPerceiver) -> DomSnapshot:
    assert perceiver.last is not None
    return perceiver.last


def constructors() -> None:
    from skillweaver.contracts import Box, ElementKind

    plain = DomControl(
        index=1,
        element_id="x",
        role="button",
        name="Add",
        kind=ElementKind.button,
        box=Box(0, 0, 9, 9),
    )
    check("a DomControl without context has none, and its name is its label", plain.context is None)
    check("... action_name == label", plain.action_name == "Add", plain.action_name)
    told = DomControl(
        index=1,
        element_id="x",
        role="button",
        name="Add",
        kind=ElementKind.button,
        box=Box(0, 0, 9, 9),
        context="Rye loaf $3.99",
    )
    check(
        "with context the name is upstream's: label, em dash with spaces, context",
        told.action_name == "Add — Rye loaf $3.99",
        told.action_name,
    )


def contexts(controller: Any, perceiver: DomPerceiver, urls: dict[str, str]) -> None:
    controller.perform(Navigate(urls["cards"]))
    perceiver.observe(controller)
    first = _snap(perceiver)
    adds = _named(first, "Add item to cart")
    check("seven same-label buttons are all offered", len(adds) == 7, len(adds))
    found = [control.context for control in adds]
    check(
        "each carries its own card's text, label stripped",
        all(
            context is not None and product in context and "Add item" not in context
            for context, product in zip(found, PRODUCTS, strict=False)
        ),
        found,
    )
    check("the seven contexts are distinct", len(set(found)) == 7)
    check(
        "so are the seven action names",
        len({control.action_name for control in adds}) == 7,
        adds[1].action_name if len(adds) > 1 else "",
    )
    checkout = _named(first, "Checkout")
    check(
        "a unique-label control gets none, whatever surrounds it",
        len(checkout) == 1 and checkout[0].context is None,
        [control.context for control in checkout],
    )
    deals = _named(first, "Deals")
    check(
        "a shared label in a header full of controls yields no context",
        len(deals) == 2 and all(control.context is None for control in deals),
        [control.context for control in deals],
    )
    bare = _named(first, "Bare")
    check(
        "a shared label with nothing around it yields no context",
        len(bare) == 2 and all(control.context is None for control in bare),
        [control.context for control in bare],
    )

    ids = [control.element_id for control in first.controls]
    check("element ids are unique on that screen", len(set(ids)) == len(ids))
    perceiver.observe(controller)
    again = _snap(perceiver)
    check(
        "a re-read of the same page gives the same ids, contexts and digest",
        [control.element_id for control in again.controls] == ids
        and [control.context for control in again.controls]
        == [control.context for control in first.controls]
        and again.digest == first.digest,
    )
    print(f"     context pass on cards: {first.context_ms:.2f}ms in-page")

    controller.perform(Navigate(urls["many"]))
    perceiver.observe(controller)
    many = _snap(perceiver)
    hides = _named(many, "hide")
    described = [control for control in hides if control.context]
    check(
        f"at most {MAX_DESCRIBED} controls are described in one snapshot",
        len(hides) == 40 and len(described) == MAX_DESCRIBED,
        (len(hides), len(described)),
    )
    check(
        f"no context is longer than {MAX_CONTEXT_CHARS}",
        all(len(control.context or "") <= MAX_CONTEXT_CHARS for control in many.controls),
    )
    print(f"     context pass on forty rows: {many.context_ms:.2f}ms in-page")


def _timed_observe(perceiver: DomPerceiver, controller: Any) -> float:
    began = time.monotonic()
    perceiver.observe(controller)
    return (time.monotonic() - began) * 1000.0


def effect_watch(controller: Any, perceiver: DomPerceiver, url: str) -> None:
    click_budget, type_budget = EFFECT_BUDGET_MS["CLICK"], EFFECT_BUDGET_MS["TYPE_TEXT"]

    def fresh() -> DomSnapshot:
        controller.perform(Navigate(url))
        perceiver.observe(controller)
        return _snap(perceiver)

    # The late effect: nothing moves for 800ms, then the badge does.
    basis = fresh()
    polls = perceiver.polls
    clicked = time.monotonic()
    controller.perform(Click(_named(basis, "Add")[0].box.center))
    perceiver.rest_after(1, basis, operation="CLICK")
    perceiver.observe(controller)
    since_click = (time.monotonic() - clicked) * 1000.0
    after = _snap(perceiver)
    used = perceiver.polls - polls
    check(
        "an effect landing 800ms after the click is observed as CHANGED",
        after.digest != basis.digest and "Cart:\n1" in after.text,
        after.text[:30].replace("\n", " "),
    )
    check(
        "... and the watch stopped near 800ms, not at the budget",
        700 <= since_click <= 1600 and 1 <= used <= 12,
        f"{since_click:.0f}ms since the click, {used} polls",
    )

    # A true no-op click pays the whole click budget.
    basis = fresh()
    polls = perceiver.polls
    controller.perform(Click(_named(basis, "Nothing")[0].box.center))
    perceiver.rest_after(1, basis, operation="CLICK")
    spent = _timed_observe(perceiver, controller)
    used = perceiver.polls - polls
    check(
        f"a no-op click pays ~{click_budget:.0f}ms and reads unchanged",
        click_budget - 100 <= spent <= click_budget + 1200
        and _snap(perceiver).digest == basis.digest,
        f"{spent:.0f}ms, {used} polls",
    )

    # A no-op type: three controller actions, and only the LAST observation is the armed one.
    basis = fresh()
    polls = perceiver.polls
    field = _named(basis, "Dead field")[0]
    perceiver.rest_after(3, basis, operation="TYPE_TEXT")
    spents = []
    for action in (Click(field.box.center), PressKey(("Control", "a")), TypeText("rye")):
        controller.perform(action)
        spents.append(_timed_observe(perceiver, controller))
    used = perceiver.polls - polls
    check(
        f"a no-op type pays ~{type_budget:.0f}ms, on its last observation only",
        spents[0] < 600
        and spents[1] < 600
        and type_budget - 100 <= spents[2] <= type_budget + 1000
        and _snap(perceiver).digest == basis.digest,
        f"{[round(s) for s in spents]}ms, {used} polls",
    )

    # Scroll and wait are never polled, changed or not. This page has nothing to scroll.
    for operation, action, waited in (
        ("SCROLL_DOWN", Scroll(controller.viewport().center, dy=400), False),
        ("WAIT", Wait(50), True),
    ):
        basis = fresh()
        polls, rests = perceiver.polls, perceiver.rests[0]
        controller.perform(action)
        perceiver.rest_after(1, basis, waited=waited, operation=operation)
        spent = _timed_observe(perceiver, controller)
        check(
            f"an unchanged page after {operation} is handed over: no poll, no rest",
            perceiver.polls == polls and perceiver.rests[0] == rests and spent < 600,
            f"{spent:.0f}ms",
        )

    # A caller that does not name its operation gets what it got before the watch existed.
    basis = fresh()
    polls, rests = perceiver.polls, perceiver.rests[0]
    controller.perform(Click(_named(basis, "Nothing")[0].box.center))
    perceiver.rest_after(1, basis)
    spent = _timed_observe(perceiver, controller)
    check(
        "rest_after without an operation: one quiet window, no poll",
        perceiver.polls == polls and perceiver.rests[0] == rests + 1 and spent < 1500,
        f"{spent:.0f}ms",
    )

    # An unarmed observation is never taxed - a warm replay, the gate, wait_for_text.
    polls, rests = perceiver.polls, perceiver.rests[0]
    spent = _timed_observe(perceiver, controller)
    check(
        "an unarmed observation neither polls nor rests",
        perceiver.polls == polls and perceiver.rests[0] == rests,
        f"{spent:.0f}ms",
    )


@contextlib.contextmanager
def _controller(harness: bool, folder: str):
    """The Playwright controller, or the harness on a Chrome of this script's own."""
    if not harness:
        from skillweaver.controllers.browser import BrowserController

        with BrowserController(headless=True, viewport=(1280, 1200)) as controller:
            yield controller
        return
    from skillweaver.controllers.chrome_launch import ChromeProcess

    with ChromeProcess(user_data_dir=Path(folder) / "profile", headless=True) as chrome:
        # Both are read when browser_harness is IMPORTED, so they are set before it is.
        os.environ["BU_CDP_URL"] = chrome.endpoint
        os.environ["BU_NAME"] = f"check-k-dom-{os.getpid()}"
        from skillweaver.controllers.harness import HarnessBrowserController

        with HarnessBrowserController(viewport=(1280, 1200)) as controller:
            yield controller


def main() -> int:
    harness = "--harness" in sys.argv[1:]
    constructors()
    with tempfile.TemporaryDirectory() as folder:
        urls = {}
        for name, html in {"cards": CARDS, "many": MANY, "effect": EFFECT}.items():
            path = Path(folder) / f"{name}.html"
            path.write_text(html, encoding="utf-8")
            urls[name] = path.as_uri()
        with _controller(harness, folder) as controller:
            print(f"--- through {controller.describe()}")
            perceiver = DomPerceiver()
            contexts(controller, perceiver, urls)
            effect_watch(controller, perceiver, urls["effect"])
            rests, rested_ms = perceiver.rests
            print(
                f"     {perceiver.polls} polls, {rests} rests, {rested_ms:.0f}ms rested, "
                f"site_ms={perceiver.site_ms:.0f}"
            )
    print(f"\n{len(_failures)} failed" if _failures else "\nall checks passed")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
