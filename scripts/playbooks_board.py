"""Playbooks for the ten tasks of ``eval/board.yaml`` on ``apps/board-site``.

The tracker, whose idiom is four columns of cards: a control belongs to the card it is
drawn in, a menu opens inside that card, and the detail of a ticket lives in a panel
that appears beside the board. See ``scripts/playbooks_common.py`` for the shared
machinery and ``scripts/scripted_operator.py`` for what a playbook stands in for.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any

from playbooks_common import (  # type: ignore[import-not-found]
    bind,
    choose,
    fill,
    owned_by,
    press,
    replace_text,
    sees,
    skill,
)
from scripted_operator import (  # type: ignore[import-not-found]
    Element,
    Playbook,
    click,
    find,
)

__all__ = ["PLAYBOOKS", "playbooks"]

TOOLBAR = (0, 50, 1280, 104)
"""Where the search box and the three filters live. Every column name and assignee
also appears on the board below, so a filter click has to say which it means."""

PANEL = (860, 100, 1280, 800)
"""The detail panel, drawn over the right-hand columns."""

MODAL = (400, 160, 880, 640)
"""The create-ticket dialog."""


def card_menu(title: str, choice: str, *, done: bool = False):
    """Open a card's own menu and pick from it.

    The menu opens INSIDE the card, so its options land on top of that card's own
    text: the option is the one near the card, not the column heading of the same name
    at the top of the board.
    """

    def open_it(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        card = titled(elements, title)
        if card is None:
            return None
        button = owned_by(elements, "Move", card, below_by=(10, 90), across=260)
        return None if button is None else click(button)

    def pick(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        card = titled(elements, title)
        if card is None:
            return None
        option = owned_by(elements, choice, card, below_by=(10, 200), across=260)
        return None if option is None else click(option, done=done)

    return (open_it, pick)


def open_ticket(title: str, *, done: bool = False):
    """Click a card's title, which is what opens it."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        target = titled(elements, title)
        return None if target is None else click(target, done=done)

    return step


def titled(elements: Sequence[Element], title: str) -> Element | None:
    """The card headed ``title``, allowing for a list that truncates what it shows.

    A column of cards is narrow, so a long title is cut off and the whole of it is
    never on screen. Asking for less and less of it finds the card without needing to
    know where the cut falls.
    """
    words = title.split()
    for stop in range(len(words), 2, -1):
        found = find(elements, " ".join(words[:stop]))
        if found is not None:
            return found
    return None


def panel_field(caption: str, value: str, button: str, *, done: bool = False):
    """Replace a field in the detail panel and press the button that commits it.

    The button is looked up through ``ctx.see`` rather than by the id the prompt
    listed, because a detector will sometimes draw ONE box around a field and the
    button beside it. An id points at that whole box, whose middle is the field; a
    text lookup points at the part of it that says what was asked for.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        box = _under(elements, caption)
        if box is None or find(elements, button, within=PANEL) is None:
            return None
        return {
            "code": (
                replace_text(box.id, value)
                + f"hits = ctx.see.find_text({button!r})\n"
                + f'ctx.expect(bool(hits), "nothing saying {button}")\n'
                + "ctx.ctl.click(hits[0])\n"
            ),
            "done": done,
        }

    return step


def panel_select(done: bool = False, *, option: str = ""):
    """Choose from the detail panel's only dropdown.

    Found by what it IS rather than by the caption beside it: on this page the caption
    is small, grey and sometimes not read at all, while "the one dropdown in the
    panel" is never ambiguous.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        fields = [
            e
            for e in elements
            if e.kind == "text_field" and PANEL[0] <= e.cx <= PANEL[2] and e.cy >= PANEL[1]
        ]
        if not fields:
            return None
        target = min(fields, key=lambda e: e.cy)
        return {
            "code": (
                f'ctx.ctl.click(el["{target.id}"])\n'
                f"ctx.ctl.type_text({option!r})\n"
                'ctx.ctl.press("Enter")\n'
            ),
            "done": done,
        }

    return step


def comment_box(text: str, *, done: bool = False):
    """Type into the box above the Add comment button and press it.

    The list of comments is captioned with its own count, which changes, so the button
    is the fixed landmark: the box is what sits directly above it.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        button = find(elements, "Add comment", within=PANEL)
        if button is None:
            return None
        above = [
            e
            for e in elements
            if e.cx >= PANEL[0] and e.w >= 80 and 0 < button.y - e.y <= 120 and e is not button
        ]
        if not above:
            return None
        box = max(above, key=lambda e: e.y)
        return {
            "code": replace_text(box.id, text) + f'ctx.ctl.click(el["{button.id}"])\n',
            "done": done,
        }

    return step


def _under(elements: Sequence[Element], caption: str, *, region=PANEL) -> Element | None:
    """The field beneath a caption, inside one region of the screen."""
    label = find(elements, caption, within=region)
    if label is None:
        return None
    best: Element | None = None
    for element in elements:
        drop = element.cy - label.cy
        if not 8 <= drop <= 70 or not -12 <= element.x - label.x <= 90:
            continue
        if element.w < 80:
            continue
        if best is None or element.cy < best.cy:
            best = element
    return best


def compose_ticket(title: str, assignee: str, priority: str):
    """Fill the create dialog by tabbing through it.

    Its two dropdowns sit side by side under small grey captions over a dimmed board,
    and neither the captions nor the controls come back from perception at all - so
    there is nothing on screen to click at. Tabbing from the field that IS visible is
    what a keyboard user does, and typing an option's name into a focused dropdown
    picks it by name rather than by position.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        box = find(elements, "What needs doing", within=MODAL)
        if box is None:
            return None
        return {
            "code": (
                f'ctx.ctl.click(el["{box.id}"])\n'
                'ctx.ctl.press("Control", "a")\n'
                f"ctx.ctl.type_text({title!r})\n"
                'ctx.ctl.press("Tab")\n'
                'ctx.ctl.press("Tab")\n'
                f"ctx.ctl.type_text({assignee!r})\n"
                'ctx.ctl.press("Tab")\n'
                f"ctx.ctl.type_text({priority!r})\n"
            )
        }

    return step


# --------------------------------------------------------------------------------------
# The ten
# --------------------------------------------------------------------------------------

_ALL: tuple[Playbook, ...] = (
    # -- single-step -------------------------------------------------------------------
    Playbook(
        task='Search the board for "webhook".',
        steps=(fill("Search tickets", "webhook", within=TOOLBAR, done=True),),
        skill=skill(
            "search_the_board",
            "Search the board for a string.",
            "the board, filtered to the search.",
            ("query",),
            {"query": "webhook"},
            '    box = only(ctx, ctx.see.find_text("Search tickets"), "the search box")\n'
            "    fill_in(ctx, box, query)\n"
            "    return True\n",
            sees("Backlog"),
        ),
        bind=bind("search_the_board", {"query": "webhook"}, "the sentence quotes the search"),
    ),
    Playbook(
        task="Show only the tickets assigned to Ada Lindgren.",
        steps=(choose("All assignees", "Ada Lindgren", within=TOOLBAR, done=True),),
        skill=skill(
            "filter_by_assignee",
            "Filter the board to one assignee.",
            "the board, showing only that person's tickets.",
            ("assignee",),
            {"assignee": "Ada Lindgren"},
            '    hits = ctx.see.find_text("All assignees")\n'
            '    control = region(ctx, hits, 0, 50, 1280, 104, "the assignee filter")\n'
            "    pick(ctx, control, assignee)\n"
            "    return True\n",
            sees("Backlog"),
        ),
        bind=bind(
            "filter_by_assignee", {"assignee": "Ada Lindgren"}, "the sentence names the person"
        ),
    ),
    Playbook(
        task="Show only the High priority tickets.",
        steps=(choose("All priorities", "High", within=TOOLBAR, done=True),),
        skill=skill(
            "filter_by_priority",
            "Filter the board to one priority.",
            "the board, showing only tickets at that priority.",
            ("priority",),
            {"priority": "High"},
            '    hits = ctx.see.find_text("All priorities")\n'
            '    control = region(ctx, hits, 0, 50, 1280, 104, "the priority filter")\n'
            "    pick(ctx, control, priority)\n"
            "    return True\n",
            sees("Backlog"),
        ),
        bind=bind("filter_by_priority", {"priority": "High"}, "the sentence names the priority"),
    ),
    Playbook(
        task="Open the ticket about the flicker when the gauge crosses zero.",
        steps=(open_ticket("Fix flicker when the gauge crosses zero", done=True),),
        skill=skill(
            "open_ticket",
            "Open a ticket by its title.",
            "the board, with that ticket open in the detail panel.",
            ("title",),
            {"title": "Fix flicker when the gauge crosses zero"},
            "    ctx.ctl.click(only(ctx, ctx.see.find_text(title), title))\n    return title\n",
            "def verify(ctx, result):\n"
            "    return any(e.box.x > 880 for e in ctx.see.find_text(result))\n",
        ),
        bind=bind(
            "open_ticket",
            {"title": "Fix flicker when the gauge crosses zero"},
            "the sentence describes the ticket to open",
        ),
    ),
    # -- multi-step --------------------------------------------------------------------
    Playbook(
        task='Move the ticket "Fix flicker when the gauge crosses zero" to the Review column.',
        steps=card_menu("Fix flicker when the gauge crosses zero", "Review", done=True),
        skill=skill(
            "move_ticket",
            "Move a ticket to another column.",
            "the board, with the ticket in its new column.",
            ("title", "column"),
            {"title": "Fix flicker when the gauge crosses zero", "column": "Review"},
            "    card = titled(ctx, title, title)\n"
            '    ctx.ctl.click(owned_by(ctx, "Move", card, 10, 90, 260))\n'
            "    # The menu opens inside the card, so the option that belongs to this\n"
            "    # ticket is the one near it - not the column heading of the same name.\n"
            "    again = titled(ctx, title, title)\n"
            "    ctx.ctl.click(owned_by(ctx, column, again, 10, 200, 260))\n"
            "    return True\n",
            sees("Moved"),
        ),
        bind=bind(
            "move_ticket",
            {"title": "Fix flicker when the gauge crosses zero", "column": "Review"},
            "the sentence names the ticket and the column",
        ),
    ),
    Playbook(
        task='Reassign the ticket "Add bulk import for device registries" to June Nakagawa.',
        steps=(
            open_ticket("Add bulk import for device registries"),
            panel_select(done=True, option="June Nakagawa"),
        ),
        skill=skill(
            "reassign_ticket",
            "Open a ticket and assign it to someone else.",
            "the board, with the ticket reassigned.",
            ("title", "assignee"),
            {"title": "Add bulk import for device registries", "assignee": "June Nakagawa"},
            "    ctx.ctl.click(titled(ctx, title, title))\n"
            "    # The panel's only dropdown: its caption is small, grey and not\n"
            "    # always read, while there is never more than one of these.\n"
            "    boxes = [e for e in fields(ctx) if e.box.x > 880]\n"
            '    ctx.expect(bool(boxes), "no dropdown in the detail panel")\n'
            "    best = boxes[0]\n"
            "    for element in boxes:\n"
            "        if element.box.y < best.box.y:\n"
            "            best = element\n"
            "    pick(ctx, best, assignee)\n"
            "    return True\n",
            sees("Assigned"),
        ),
        bind=bind(
            "reassign_ticket",
            {"title": "Add bulk import for device registries", "assignee": "June Nakagawa"},
            "the sentence names the ticket and the person",
        ),
    ),
    Playbook(
        task=(
            'Rename the ticket "Set up nightly export of usage metrics" to "Nightly usage export".'
        ),
        steps=(
            open_ticket("Set up nightly export of usage metrics"),
            panel_field("Title", "Nightly usage export", "Save title", done=True),
        ),
        skill=skill(
            "rename_ticket",
            "Open a ticket and give it a new title.",
            "the board, with the ticket renamed.",
            ("title", "new_title"),
            {
                "title": "Set up nightly export of usage metrics",
                "new_title": "Nightly usage export",
            },
            "    ctx.ctl.click(titled(ctx, title, title))\n"
            '    fill_in(ctx, under(ctx, "Title", 70, 90), new_title)\n'
            "    # Not filtered to a control kind: a detector will sometimes draw one\n"
            "    # box around the field and the button beside it, and calling that a\n"
            "    # row does not stop the words on it leading to the right half.\n"
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Save title"), "the save button"))\n'
            "    return True\n",
            sees("Renamed"),
        ),
        bind=bind(
            "rename_ticket",
            {
                "title": "Set up nightly export of usage metrics",
                "new_title": "Nightly usage export",
            },
            "the sentence names the ticket and what to call it",
        ),
    ),
    Playbook(
        task=(
            'Add the comment "Checked on staging" to the ticket "Cache the schedule preview '
            'per workspace".'
        ),
        steps=(
            open_ticket("Cache the schedule preview per workspace"),
            comment_box("Checked on staging", done=True),
        ),
        skill=skill(
            "comment_on_ticket",
            "Open a ticket and leave a comment on it.",
            "the board, with the comment added.",
            ("title", "comment"),
            {"title": "Cache the schedule preview per workspace", "comment": "Checked on staging"},
            "    ctx.ctl.click(titled(ctx, title, title))\n"
            "    # The comment list is captioned with its own count, which changes, so\n"
            "    # the button is the fixed landmark and the box is what sits above it.\n"
            "    # Not filtered to a control kind: the detector does not always draw a\n"
            "    # box around this one, and a label that was only READ is still a label.\n"
            '    hits = ctx.see.find_text("Add comment")\n'
            '    button = region(ctx, hits, 860, 100, 1280, 800, "the comment button")\n'
            "    box = None\n"
            "    for element in ctx.see.all():\n"
            "        if element.box.x < 880 or element.box.w < 80:\n"
            "            continue\n"
            "        gap = button.box.y - element.box.y\n"
            "        if gap <= 0 or gap > 120:\n"
            "            continue\n"
            "        if box is None or element.box.y > box.box.y:\n"
            "            box = element\n"
            '    ctx.expect(box is not None, "no comment box in the panel")\n'
            "    fill_in(ctx, box, comment)\n"
            "    ctx.ctl.click(button)\n"
            "    return True\n",
            sees("Comment added"),
        ),
        bind=bind(
            "comment_on_ticket",
            {"title": "Cache the schedule preview per workspace", "comment": "Checked on staging"},
            "the sentence names the ticket and the comment",
        ),
    ),
    # -- composite ---------------------------------------------------------------------
    Playbook(
        task=(
            'Create a new ticket titled "Audit the export retry path", assigned to Theo '
            "Barros at High priority."
        ),
        steps=(
            press("New ticket", within=TOOLBAR),
            compose_ticket("Audit the export retry path", "Theo Barros", "High"),
            press("Create", within=MODAL, done=True),
        ),
        skill=skill(
            "create_ticket",
            "Create a ticket with a title, an assignee and a priority.",
            "the board, with the new ticket in the first column.",
            ("title", "assignee", "priority"),
            {
                "title": "Audit the export retry path",
                "assignee": "Theo Barros",
                "priority": "High",
            },
            '    ctx.ctl.click(only(ctx, controls(ctx, "New ticket"), "the new ticket button"))\n'
            "    # The dialog's two dropdowns do not come back from perception at all,\n"
            "    # so there is nothing on screen to click at: tab to them from the box\n"
            "    # that IS visible and choose each by typing its name.\n"
            '    box = only(ctx, ctx.see.find_text("What needs doing"), "the title box")\n'
            "    fill_in(ctx, box, title)\n"
            '    ctx.ctl.press("Tab")\n'
            '    ctx.ctl.press("Tab")\n'
            "    ctx.ctl.type_text(assignee)\n"
            '    ctx.ctl.press("Tab")\n'
            "    ctx.ctl.type_text(priority)\n"
            '    ctx.ctl.click(only(ctx, controls(ctx, "Create"), "the create button"))\n'
            "    return True\n",
            sees("Created"),
        ),
        bind=bind(
            "create_ticket",
            {
                "title": "Audit the export retry path",
                "assignee": "Theo Barros",
                "priority": "High",
            },
            "the sentence names the title, the person and the priority",
        ),
    ),
    Playbook(
        task="Archive everything in the Done column.",
        steps=(
            press("Archive done", within=TOOLBAR),
            press("Archive", control=True, within=MODAL, done=True),
        ),
        skill=skill(
            "archive_done",
            "Archive every ticket in the Done column.",
            "the board, with the Done column empty.",
            (),
            {},
            '    ctx.ctl.click(only(ctx, controls(ctx, "Archive done"), "the archive button"))\n'
            "    # The dialog's own BUTTON: its heading asks 'Archive 3 tickets?', so\n"
            "    # the word alone names the question rather than the answer.\n"
            '    hits = controls(ctx, "Archive")\n'
            '    ctx.ctl.click(region(ctx, hits, 400, 160, 880, 640, "the confirm button"))\n'
            "    return True\n",
            sees("Archived"),
        ),
        bind=bind("archive_done", {}, "the library already knows how to archive the Done column"),
    ),
)


def playbooks() -> dict[str, Playbook]:
    """Every task of the board suite, keyed by the sentence the agent is given."""
    return {playbook.task: playbook for playbook in _ALL}


PLAYBOOKS = playbooks()


if __name__ == "__main__":
    for book in _ALL:
        if book.skill is not None:
            ast.parse(book.skill["code"])
            ast.parse(book.skill["verifier_code"])
    print(json.dumps({"playbooks": len(PLAYBOOKS)}, indent=2))
