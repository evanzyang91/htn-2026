"""A small fake invoicing app for teaching and testing agents without a browser.

The task: *confirm payment of the Acme Corp invoice*. Four states lie on the path and
a fifth is a dead end::

    list --type "acme"--> searched --click Acme row--> selected --click Confirm--> done
      |                      |  ^                         |
      |                      |  +------- click Back ------+
      |                      +-- click Clear --> list
      +-- click Archive --> archive (DEAD END: no way out)   <-- also from searched

* ``list``: the invoice list. Acme is NOT on it (the list is long), so it has to be
  searched for. The search field has keyboard focus, so typing goes straight into it.
  Clicking any of the visible rows does nothing.
* ``searched``: the list filtered to the single Acme row.
* ``selected``: the Acme invoice page, with ``Confirm payment`` and ``Back``.
* ``done``: the confirmation page. Terminal; this is the goal.
* ``archive``: an empty archive page with no controls. ``Navigate`` is unsupported, so
  the only way out is ``controller.reset()`` - an agent must learn not to click it.

Use :func:`make_scenario` (or the ``scenario`` fixture) for a fresh, independent copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from skillweaver.contracts import (
    Action,
    Box,
    Click,
    Element,
    ElementKind,
    ElementSource,
    TaskSpec,
    TypeText,
)
from tests.fakes.controller import FakeController, FakeState, clicks, render_png, types
from tests.fakes.perception import FakePerceiver

DOMAIN = "fake.test"
WIDTH, HEIGHT = 800, 600
START, GOAL, DEAD_END = "list", "done", "archive"


def _el(kind: ElementKind, text: str, box: Box, stable_id: str) -> Element:
    return Element(box, kind, text, 0.95, stable_id, ElementSource.merged)


def _label(text: str, box: Box, stable_id: str) -> Element:
    return Element(box, ElementKind.text, text, 0.9, stable_id, ElementSource.ocr)


# Elements are module-level so tests and agents can refer to them by name.
HEADING = _label("Invoices", Box(20, 16, 200, 28), "heading")
SEARCH_EMPTY = _el(ElementKind.text_field, "Search invoices", Box(20, 60, 400, 32), "search")
SEARCH_FILLED = _el(ElementKind.text_field, "acme", Box(20, 60, 400, 32), "search")
ARCHIVE_LINK = _el(ElementKind.link, "Archive", Box(700, 60, 80, 32), "archive-link")
CLEAR_BUTTON = _el(ElementKind.button, "Clear", Box(430, 60, 70, 32), "clear")
ROW_GLOBEX = _el(ElementKind.row, "Globex INV-2041 $310.00", Box(20, 120, 760, 40), "row-2041")
ROW_INITECH = _el(ElementKind.row, "Initech INV-3377 $88.50", Box(20, 160, 760, 40), "row-3377")
ROW_UMBRELLA = _el(ElementKind.row, "Umbrella INV-0907 $4,020", Box(20, 200, 760, 40), "row-0907")
ROW_ACME = _el(ElementKind.row, "Acme Corp INV-1042 $1,200.00", Box(20, 120, 760, 40), "row-1042")
INVOICE_TITLE = _label("Invoice INV-1042", Box(20, 16, 300, 28), "invoice-title")
INVOICE_PARTY = _label("Acme Corp $1,200.00", Box(20, 60, 300, 24), "invoice-party")
CONFIRM_BUTTON = _el(ElementKind.button, "Confirm payment", Box(20, 120, 160, 40), "confirm")
BACK_BUTTON = _el(ElementKind.button, "Back", Box(200, 120, 80, 40), "back")
DONE_TITLE = _label("Payment confirmed", Box(20, 16, 300, 28), "done-title")
DONE_DETAIL = _label("INV-1042 paid to Acme Corp", Box(20, 60, 400, 24), "done-detail")
ARCHIVE_TITLE = _label("Archive is empty", Box(20, 16, 300, 28), "archive-title")

_LAYOUT: dict[str, tuple[tuple[Element, ...], str]] = {
    "list": (
        (HEADING, SEARCH_EMPTY, ARCHIVE_LINK, ROW_GLOBEX, ROW_INITECH, ROW_UMBRELLA),
        "https://fake.test/invoices",
    ),
    "searched": (
        (HEADING, SEARCH_FILLED, CLEAR_BUTTON, ARCHIVE_LINK, ROW_ACME),
        "https://fake.test/invoices?q=acme",
    ),
    "selected": (
        (INVOICE_TITLE, INVOICE_PARTY, CONFIRM_BUTTON, BACK_BUTTON),
        "https://fake.test/invoices/1042",
    ),
    "done": ((DONE_TITLE, DONE_DETAIL), "https://fake.test/invoices/1042/confirmed"),
    "archive": ((ARCHIVE_TITLE,), "https://fake.test/archive"),
}

TASK = TaskSpec(
    text="Confirm payment of the Acme Corp invoice.",
    domain=DOMAIN,
    target="browser",
    params={"start_url": "https://fake.test/invoices", "company": "Acme Corp"},
)

SOLUTION: tuple[Action, ...] = (
    TypeText("acme"),
    Click(ROW_ACME.box.center),
    Click(CONFIRM_BUTTON.box.center),
)
"""The shortest action sequence from ``list`` to ``done``."""

TRAP: tuple[Action, ...] = (Click(ARCHIVE_LINK.box.center),)
"""One action from ``list`` (or ``searched``) into the dead end."""


@dataclass(slots=True)
class Scenario:
    """A ready-to-drive copy of the fake app.

    ``controller`` starts in ``list``; ``perceiver`` is wired to it; ``task`` is what
    an agent is asked to do; ``solution`` and ``trap`` are reference action sequences
    for tests (an agent under test must not be given them).
    """

    controller: FakeController
    perceiver: FakePerceiver
    task: TaskSpec
    solution: tuple[Action, ...]
    trap: tuple[Action, ...]

    @property
    def solved(self) -> bool:
        """Whether the app is on the goal state."""
        return self.controller.state == GOAL

    @property
    def stuck(self) -> bool:
        """Whether the app is in the dead end."""
        return self.controller.state == DEAD_END


def make_controller() -> FakeController:
    """A fresh :class:`FakeController` for the app, in the ``list`` state."""
    states = {
        name: FakeState(render_png(elements, WIDTH, HEIGHT), elements, url)
        for name, (elements, url) in _LAYOUT.items()
    }
    transitions = {
        "list": [
            (types(contains="acme"), "searched"),
            (clicks(ARCHIVE_LINK), "archive"),
        ],
        "searched": [
            (clicks(ROW_ACME), "selected"),
            (clicks(CLEAR_BUTTON), "list"),
            (clicks(ARCHIVE_LINK), "archive"),
        ],
        "selected": [
            (clicks(CONFIRM_BUTTON), "done"),
            (clicks(BACK_BUTTON), "searched"),
        ],
    }
    return FakeController(
        states,
        transitions,
        start=START,
        viewport=Box(0, 0, WIDTH, HEIGHT),
        unsupported=("navigate",),
    )


def make_scenario() -> Scenario:
    """A fresh, independent :class:`Scenario`."""
    controller = make_controller()
    return Scenario(controller, FakePerceiver.for_controller(controller), TASK, SOLUTION, TRAP)
