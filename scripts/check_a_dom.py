"""Check what ``perception/dom.py`` took from upstream Jev ``1489129``, on real pages.

Landmark ``section`` and ``aria-haspopup`` ``opens`` per control, the ENTER offer
(``DomSnapshot.can_press_enter`` / ``enter_label``), and ``DomSnapshot.blank`` with the
perceiver's sit-out. Every page is a small local HTML file read through the real
``DomPerceiver`` and a real headless ``BrowserController``, and focus is put where it is by
real clicks and real typing - no page is faked and no snapshot is hand-built, except the
two constructor checks at the top, which need no browser.

Needs NO network and no ``.env``; it does need the Playwright Chromium this project
already installs. Run: ``uv run python scripts/check_a_dom.py``. Prints one line per
check and exits non-zero if any failed.

``--harness`` runs the same pages through ``HarnessBrowserController`` instead, which is
the DEFAULT browser and the one that matters for the ENTER offer: its tab is a BACKGROUND
tab, and whether ``document.activeElement`` still names the typed-in field there is a
fact about Chrome, not about this code. It never touches the person's own Chrome: it
starts a private headless ``ChromeProcess`` on a throwaway profile and points the harness
at it with ``BU_CDP_URL`` and a ``BU_NAME`` of its own, as ``AGENTS.md`` describes. Needs
Google Chrome installed; still no network.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from skillweaver.contracts import Box, Click, ElementKind, Navigate, PressKey, TypeText
from skillweaver.perception.dom import (
    BLANK_SIT_OUTS,
    DomControl,
    DomPerceiver,
    DomSnapshot,
)

LANDMARKS = """<!doctype html><title>landmarks</title><body style="margin:0">
<header><button>In header</button></header>
<div role="banner"><button>In banner role</button></div>
<nav><a href="#n">In nav</a></nav>
<div role="search"><input type="search" aria-label="In search role"></div>
<main>
  <button>In main</button>
  <form><input type="text" aria-label="In form"><button type="button">Form button</button></form>
  <button aria-haspopup="menu">Opens menu</button>
  <button aria-haspopup="true">Opens true</button>
  <div role="combobox" tabindex="0" aria-haspopup="listbox" aria-label="Opens listbox">pick</div>
</main>
<aside><button>In aside</button></aside>
<div role="complementary"><button>In complementary role</button></div>
<div><button>In no landmark</button></div>
<dialog open style="position:static"><button>In dialog</button></dialog>
<div role="dialog"><button>In dialog role</button></div>
<footer><a href="#f">In footer</a></footer>
<div role="contentinfo"><a href="#c">In contentinfo role</a></div>
</body>"""

FIELDS = """<!doctype html><title>fields</title><body>
<p>Some page text.</p>
<input type="text" aria-label="Search repositories" id="q"><br><br>
<input type="text" aria-label="Already filled" value="kept query" id="kept"><br><br>
<textarea aria-label="Comment"></textarea><br><br>
<input type="password" aria-label="Password" id="pw"><br><br>
<input type="text" aria-label="Read only" value="fixed" readonly><br><br>
<label><input type="checkbox" id="box"> Tick me</label><br><br>
<button id="go">Plain button</button>
</body>"""

EMPTY = "<!doctype html><title>empty</title><body></body>"

# Blank when its load event fires, the real page 600ms later: the shape of a bot-check or
# splash shell that replaces itself. The style element is there so "blank" is tested on a
# document that is not literally empty.
INTERSTITIAL = """<!doctype html><title>shell</title><style>body{margin:0}</style><body>
<script>setTimeout(() => {
  document.body.innerHTML = '<h1>The real site</h1><button>Continue shopping</button>';
}, 600);</script></body>"""

_failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f'  [{detail}]' if detail != '' else ''}")
    if not ok:
        _failures.append(name)


def _by_name(snapshot: DomSnapshot) -> dict[str, DomControl]:
    return {control.name: control for control in snapshot.controls}


def _click(controller: Any, control: DomControl) -> None:
    result = controller.perform(Click(control.box.center))
    assert result.ok, result


def constructors() -> None:
    """Every call that predates the new fields still constructs, and they default off."""
    control = DomControl(1, "id", "button", "Go", ElementKind.button, Box(0, 0, 10, 10))
    check("DomControl defaults", control.section is None and control.opens is None)
    snapshot = DomSnapshot(url="u", title="t", text="hello", controls=(control,))
    check(
        "DomSnapshot defaults",
        snapshot.can_press_enter is False and snapshot.enter_label == "" and not snapshot.blank,
    )
    replaced = dataclasses.replace(snapshot, can_go_back=False)
    check("dataclasses.replace keeps working", replaced.digest == snapshot.digest)
    offered = dataclasses.replace(snapshot, can_press_enter=True, enter_label="Search")
    check("the ENTER offer is part of digest", offered.digest != snapshot.digest)
    unlabelled = dataclasses.replace(snapshot, enter_label="Search")
    check("a label with no offer is not", unlabelled.digest == snapshot.digest)
    check("blank: no text, no controls", DomSnapshot("u", "t", " \n", ()).blank)
    check("not blank: text alone", not DomSnapshot("u", "t", "Verify you are human", ()).blank)
    check("not blank: a control alone", not DomSnapshot("u", "t", "", (control,)).blank)


def landmarks(controller: Any, perceiver: DomPerceiver, url: str) -> None:
    controller.perform(Navigate(url))
    perceiver.observe(controller)
    found = _by_name(perceiver.last)
    expected = {
        "In header": "header",
        "In banner role": "header",
        "In nav": "nav",
        "In search role": "search",
        "In main": "main",
        "In form": "form",
        "Form button": "form",
        "In aside": "aside",
        "In complementary role": "aside",
        "In no landmark": None,
        "In dialog": "dialog",
        "In dialog role": "dialog",
        "In footer": "footer",
        "In contentinfo role": "footer",
    }
    for name, section in expected.items():
        control = found.get(name)
        got = control.section if control is not None else "<control missing>"
        check(f"section of '{name}' is {section}", got == section, got)
    for name, opens in {
        "Opens menu": "menu",
        "Opens true": "true",
        "Opens listbox": "listbox",
        "In main": None,
    }.items():
        control = found.get(name)
        got = control.opens if control is not None else "<control missing>"
        check(f"opens of '{name}' is {opens}", got == opens, got)


def enter_offer(controller: Any, perceiver: DomPerceiver, url: str) -> None:
    def look() -> DomSnapshot:
        perceiver.observe(controller)
        return perceiver.last

    controller.perform(Navigate(url))
    first = look()
    check("no ENTER with nothing focused", not first.can_press_enter, first.enter_label)
    found = _by_name(first)

    _click(controller, found["Search repositories"])
    focused_empty = look()
    check("no ENTER on a focused EMPTY field", not focused_empty.can_press_enter)
    check("focusing an empty field is not a change", focused_empty.digest == first.digest)

    controller.perform(TypeText("skillweaver"))
    typed = look()
    check("ENTER once the focused field holds text", typed.can_press_enter)
    check(
        "enter_label names the field", typed.enter_label == "Search repositories", typed.enter_label
    )

    _click(controller, found["Plain button"])
    away = look()
    check("no ENTER once focus leaves for a button", not away.can_press_enter)

    before = look()
    _click(controller, found["Already filled"])
    refocused = look()
    check(
        "ENTER on focusing a field that already holds a value",
        refocused.can_press_enter and refocused.enter_label == "Already filled",
        refocused.enter_label,
    )
    check("and that focus alone IS a change to digest", refocused.digest != before.digest)

    _click(controller, found["Comment"])
    controller.perform(TypeText("a note"))
    area = look()
    check(
        "ENTER on a non-empty TEXTAREA",
        area.can_press_enter and area.enter_label == "Comment",
        area.enter_label,
    )

    _click(controller, found["Read only"])
    check("no ENTER on a read-only field", not look().can_press_enter)

    _click(controller, found["Tick me"])
    ticked = look()
    check("the checkbox really took the click", _by_name(ticked)["Tick me"].checked is True)
    check("no ENTER on a focused checkbox (value 'on')", not ticked.can_press_enter)

    # A password field is never listed, so there is no box to click: reach it the way a
    # person would, with Tab from the field before it.
    _click(controller, found["Comment"])
    controller.perform(PressKey(("Tab",)))
    controller.perform(TypeText("hunter2"))
    focus = controller.evaluate(
        "() => document.activeElement.id + ':' + document.activeElement.value"
    )
    check("focus really is in the filled password field", focus == "pw:hunter2", focus)
    check("no ENTER on a password field", not look().can_press_enter)


def blank_pages(controller: Any, perceiver: DomPerceiver, urls: dict[str, str]) -> None:
    controller.perform(Navigate(urls["fields"]))
    before = perceiver.blank_waits
    perceiver.observe(controller)
    check("a normal page is not blank", not perceiver.last.blank)
    check("and is never sat out", perceiver.blank_waits == before)

    controller.perform(Navigate(urls["empty"]))
    before, site_before, began = perceiver.blank_waits, perceiver.site_ms, time.monotonic()
    perceiver.observe(controller)
    wall = (time.monotonic() - began) * 1000.0
    waits = perceiver.blank_waits - before
    check("an empty document is blank", perceiver.last.blank)
    check(f"it is sat out {BLANK_SIT_OUTS} times, then handed over", waits == BLANK_SIT_OUTS, waits)
    check("the sit-out is bounded by its caps", wall < BLANK_SIT_OUTS * 1200 + 3000, round(wall))
    site = perceiver.site_ms - site_before
    check("the wait is charged to site_ms", site >= BLANK_SIT_OUTS * 200 * 0.9, round(site))

    controller.perform(Navigate(urls["interstitial"]))
    before = perceiver.blank_waits
    observation = perceiver.observe(controller)
    waits = perceiver.blank_waits - before
    names = sorted(_by_name(perceiver.last))
    check("a shell that replaces itself is sat out", 1 <= waits <= BLANK_SIT_OUTS, waits)
    check("and the page handed over is the real one", names == ["Continue shopping"], names)
    check(
        "with the observation built from that same frame",
        any(e.text == "Continue shopping" for e in observation.elements),
    )


class _NoQuiesce:
    """A controller that cannot wait: everything else passes through."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "quiesce":
            raise AttributeError(name)
        return getattr(self._inner, name)


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
        os.environ["BU_NAME"] = f"check-a-dom-{os.getpid()}"
        from skillweaver.controllers.harness import HarnessBrowserController

        with HarnessBrowserController(viewport=(1280, 1200)) as controller:
            yield controller


def main() -> int:
    harness = "--harness" in sys.argv[1:]
    constructors()
    with tempfile.TemporaryDirectory() as folder:
        urls = {}
        for name, html in {
            "landmarks": LANDMARKS,
            "fields": FIELDS,
            "empty": EMPTY,
            "interstitial": INTERSTITIAL,
        }.items():
            path = Path(folder) / f"{name}.html"
            path.write_text(html, encoding="utf-8")
            urls[name] = path.as_uri()
        with _controller(harness, folder) as controller:
            print(f"--- through {controller.describe()}")
            perceiver = DomPerceiver()
            landmarks(controller, perceiver, urls["landmarks"])
            enter_offer(controller, perceiver, urls["fields"])
            blank_pages(controller, perceiver, urls)

            controller.perform(Navigate(urls["empty"]))
            plain = DomPerceiver()
            began = time.monotonic()
            plain.observe(_NoQuiesce(controller))
            spent = (time.monotonic() - began) * 1000.0
            check(
                "a controller with no quiesce is handed the blank frame at once",
                plain.last.blank and plain.blank_waits == 0 and spent < 1500,
                round(spent),
            )
    print(f"\n{len(_failures)} failed" if _failures else "\nall checks passed")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
