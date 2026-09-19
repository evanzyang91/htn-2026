"""Playbooks for the fourteen tasks of ``eval/tasks.yaml`` on ``apps/sandbox-site``.

One per task. Each says how a competent operator does the task by clicking what is on
screen, and what skill the run teaches - the same two things a real computer-use model
supplies, written down so the rest of the system can be measured without one. See
``scripts/scripted_operator.py`` for what that does and does not prove.

Everything here is written against what PERCEPTION reports, not against the DOM. The
steps act by the element ids the prompt lists, and the skills find their targets with
``ctx.see``, so a control the detector misses or OCR garbles is a control neither can
click - which is the point. The app's own rules are obeyed as a user must obey them: a
message is selected before it is archived, the label menu is opened after the selection
and not before, the rename is committed rather than left sitting in its box.

Regions, and why nearly every lookup has one
--------------------------------------------

A real page repeats its words. ``Archive`` is a toolbar button and ``Archived`` is a
folder beneath it; ``Travel`` is a menu item and also a label in the sidebar legend;
``Records`` is a nav button, a heading, a column header and a line in the footer;
``Paused`` is a menu item and also the status of two rows already on screen. Clicking
the wrong one does not fail loudly - it quietly does something else - so a lookup that
could be ambiguous is bounded by where the thing actually is. The skills carry the same
bounds, because they will be replayed on the same screens.

The one thing these may use that a model could not is certainty about WHICH task is
being asked. That is deliberate: it is the model's judgment being replaced, not its
eyes or its hands.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any

from playbooks_common import (  # type: ignore[import-not-found]
    bind,
    go,
    press,
    sees,
    skill,
)
from scripted_operator import (  # type: ignore[import-not-found]
    Element,
    Playbook,
    click,
    find,
    scroll,
    squash,
)

__all__ = ["PLAYBOOKS", "playbooks"]

# Regions, as (x0, y0, x1, y1) around an element's centre. The viewport is 1280x800.
TOP_BAR = (0, 0, 2000, 52)
MAIL_TOOLBAR = (200, 50, 900, 112)
LABEL_MENU = (300, 115, 760, 320)
BULK_MENU = (280, 170, 560, 350)
COLUMN_HEADERS = (0, 180, 1280, 228)
COMPOSE = (760, 430, 1280, 800)


# --------------------------------------------------------------------------------------
# Step builders
# --------------------------------------------------------------------------------------


def fill(label: str, value: str, *, kind: str = "text_field", done: bool = False):
    """Click a field and replace its contents.

    One move, not two. Focusing a field changes nothing a critic can see - this app
    even hides the caret - so a bare click on a text box is judged as a move that did
    nothing, and the next one is refused as a repeat. Clicking and typing IS one
    decision, and a code block is how the agent expresses one.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        field = find(elements, label, kind=kind)
        if field is None:
            return None
        return {"code": _replace(field.id, value), "done": done}

    return step


def _replace(element_id: str, value: str) -> str:
    return (
        f'ctx.ctl.click(el["{element_id}"])\n'
        'ctx.ctl.press("Control", "a")\n'
        f"ctx.ctl.type_text({value!r})\n"
    )


def select_row_then(row_text: str, button: str, *, region=MAIL_TOOLBAR, done: bool = False):
    """Tick the checkbox on the row naming ``row_text``, then press a toolbar button.

    Both in one move: the app disables ``Archive`` and ``Label`` until something is
    selected, and ticking a box without then acting on it is not a thing anyone means.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        row = find(elements, row_text)
        target = find(elements, button, within=region)
        if row is None or target is None:
            return None
        box = _checkbox_on_row(elements, row)
        if box is None:
            return None
        return {
            "code": f'ctx.ctl.click(el["{box.id}"])\nctx.ctl.click(el["{target.id}"])\n',
            "done": done,
        }

    return step


def _checkbox_on_row(elements: Sequence[Element], row: Element) -> Element | None:
    """The checkbox belonging to a list row: leftmost, vertically aligned with it."""
    candidates = [
        e
        for e in elements
        if e.kind == "checkbox" and abs(e.cy - row.cy) <= max(30, row.h // 2) and e.cx < row.cx
    ]
    return min(candidates, key=lambda e: e.cx) if candidates else None


def _pencil_on_row(elements: Sequence[Element], row: Element) -> Element | None:
    """The edit button at the right-hand end of a records row. It carries no text, so
    it is found by where it is: the rightmost button on this row."""
    candidates = [
        e for e in elements if e.kind == "button" and abs(e.cy - row.cy) <= 26 and e.cx > 1100
    ]
    return max(candidates, key=lambda e: e.cx) if candidates else None


def _header_checkbox(elements: Sequence[Element]) -> Element | None:
    """The select-all checkbox: topmost in the table's left-hand column."""
    candidates = [e for e in elements if e.kind == "checkbox" and e.cx < 60]
    return min(candidates, key=lambda e: e.cy) if candidates else None


def rename_record(name: str, new_name: str):
    """Open a row's inline editor with its pencil, then replace and commit the name."""

    def start(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        row = find(elements, name, kind=None)
        if row is None:
            return None
        pencil = _pencil_on_row(elements, row)
        return None if pencil is None else click(pencil)

    def commit(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        # The editor is a bordered box with the old name in it, and the detector calls
        # that a button as often as a text field - so it is identified by its text and
        # its size (a row is far taller) rather than by what it was classified as.
        boxes = [e for e in elements if squash(name) in squash(e.text) and e.h <= 34]
        if not boxes:
            return None
        editor = min(boxes, key=lambda e: e.w * e.h)
        return {"code": _replace(editor.id, new_name) + 'ctx.ctl.press("Enter")\n', "done": True}

    return (start, commit)


def save_settings(elements: Sequence[Element], failures: int) -> dict[str, Any]:
    """Press Save, scrolling the settings pane first when the save bar is below it.

    Written as one code block rather than a scroll move and a click move because
    whether the scroll is needed depends on the window, and a scroll that was not
    needed changes nothing - which a critic reads as a move that failed.
    """
    # A code block is a standalone snippet - it has none of the helpers a stored skill
    # carries - so finding somewhere to put the pointer is written out in full here.
    return {
        "code": (
            'hits = [h for h in ctx.see.find_text("Save changes") '
            'if h.kind.value == "button"]\n'
            "if not hits:\n"
            "    big = None\n"
            "    for element in ctx.see.all():\n"
            "        if big is None or element.box.area > big.box.area:\n"
            "            big = element\n"
            '    ctx.expect(big is not None, "the settings screen appears to be empty")\n'
            "    ctx.ctl.scroll(big, 0, 400)\n"
            '    hits = [h for h in ctx.see.find_text("Save changes") '
            'if h.kind.value == "button"]\n'
            'ctx.expect(bool(hits), "no save button on the settings screen")\n'
            "ctx.ctl.click(hits[0])\n"
        )
    }


def search_then_archive(query: str, subject: str):
    """Find a message with the search box, then select it and archive it.

    Not by scrolling. The message this is for sits below the fold of a 1280x800
    window, and the application clips it: the screen element is ``overflow: hidden``
    and the list inside it is not a scroller either, so no amount of wheeling brings
    it into view. Searching is what the task means by "find" anyway - it is what a
    person does - and it works whatever the window size.
    """

    def look(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        field = find(elements, "Search mail", kind="text_field")
        if field is None:
            return None
        return {"code": _replace(field.id, query)}

    return (look, select_row_then(subject, "Archive", done=True))


def select_all_then_bulk(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
    """Tick the table's select-all box and open the bulk-actions menu."""
    box = _header_checkbox(elements)
    bulk = find(elements, "Bulk actions")
    if box is None or bulk is None:
        return None
    return {"code": f'ctx.ctl.click(el["{box.id}"])\nctx.ctl.click(el["{bulk.id}"])\n'}


def compose(to: str, subject: str):
    """Fill the compose window's address and subject boxes.

    They show a placeholder and nothing else, and the placeholder disappears as soon
    as anything is typed, so they are found by where the window puts them: the two
    topmost fields inside it.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        x0, y0, x1, y1 = COMPOSE
        fields = [
            e for e in elements if e.kind == "text_field" and x0 <= e.cx <= x1 and y0 <= e.cy <= y1
        ]
        if len(fields) < 2:
            return None
        ordered = sorted(fields, key=lambda e: e.y)
        return {
            "code": (
                f'ctx.ctl.click(el["{ordered[0].id}"])\n'
                f"ctx.ctl.type_text({to!r})\n"
                f'ctx.ctl.click(el["{ordered[1].id}"])\n'
                f"ctx.ctl.type_text({subject!r})\n"
            )
        }

    return step


def scroll_down(anchor_text: str, dy: int = 400):
    """Scroll the list under ``anchor_text``. A move like any other: it changes the
    screen, so the critic accepts it and the next step finds what it wanted."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        anchor = find(elements, anchor_text)
        return scroll(anchor, dy)

    return step


# --------------------------------------------------------------------------------------
# Skills: what each run teaches
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# The fourteen
# --------------------------------------------------------------------------------------

_ALL: tuple[Playbook, ...] = (
    # -- single-step -------------------------------------------------------------------
    Playbook(
        task="Open the Records screen.",
        steps=(go("Records", done=True),),
        skill=skill(
            "open_records_screen",
            "Open the Records screen from the application bar.",
            "the Records screen.",
            (),
            {},
            '    go_to(ctx, "Records")\n    return True\n',
            sees("Bulk actions"),
        ),
        bind=bind("open_records_screen", {}, "the library already knows how to open Records"),
    ),
    Playbook(
        task="Open the Settings screen.",
        steps=(go("Settings", done=True),),
        skill=skill(
            "open_settings_screen",
            "Open the Settings screen from the application bar.",
            "the Settings screen.",
            (),
            {},
            '    go_to(ctx, "Settings")\n    return True\n',
            sees("Display name"),
        ),
        bind=bind("open_settings_screen", {}, "the library already knows how to open Settings"),
    ),
    Playbook(
        task='Search the mailbox for "invoice".',
        steps=(fill("Search mail", "invoice", done=True),),
        skill=skill(
            "search_mailbox",
            "Search the mailbox for a string.",
            "the mail screen, filtered to the search.",
            ("query",),
            {"query": "invoice"},
            '    field = only(ctx, ctx.see.find_text("Search mail"), "the mail search box")\n'
            "    fill_in(ctx, field, query)\n"
            "    return True\n",
            sees("conversations"),
        ),
        bind=bind(
            "search_mailbox",
            {"query": "invoice"},
            'the sentence quotes the string to search for: "invoice"',
        ),
    ),
    Playbook(
        task='Open the message from Billing titled "Invoice 4471 is ready".',
        steps=(press("Invoice 4471 is ready", done=True),),
        skill=skill(
            "open_message",
            "Open a message in the mailbox by its subject.",
            "the mail screen with that message open in the reading pane.",
            ("subject",),
            {"subject": "Invoice 4471 is ready"},
            "    ctx.ctl.click(only(ctx, ctx.see.find_text(subject), subject))\n"
            "    return subject\n",
            # Not the close button's label: that is an accessible name and nothing on
            # screen prints it - the button shows a cross. What IS visible is the
            # message itself, now in the reading pane on the right-hand side.
            "def verify(ctx, result):\n"
            "    return any(e.box.x > 600 for e in ctx.see.find_text(result))\n",
        ),
        bind=bind(
            "open_message",
            {"subject": "Invoice 4471 is ready"},
            "the sentence names the subject of the message to open",
        ),
    ),
    Playbook(
        task='Filter the records table for "storage".',
        steps=(go("Records"), fill("Filter records", "storage", done=True)),
        skill=skill(
            "filter_records",
            "Filter the records table for a string.",
            "the Records screen, filtered.",
            ("query",),
            {"query": "storage"},
            '    go_to(ctx, "Records")\n'
            '    field = only(ctx, ctx.see.find_text("Filter records"), "the records filter")\n'
            "    fill_in(ctx, field, query)\n"
            "    return True\n",
            sees("datasets"),
        ),
        bind=bind(
            "filter_records",
            {"query": "storage"},
            'the sentence quotes the string to filter for: "storage"',
        ),
    ),
    # -- multi-step --------------------------------------------------------------------
    Playbook(
        task='Archive the message from Billing titled "Invoice 4471 is ready".',
        steps=(select_row_then("Invoice 4471 is ready", "Archive", done=True),),
        skill=skill(
            "archive_message",
            "Archive one message in the mailbox by its subject.",
            "the mail screen, with that conversation archived.",
            ("subject",),
            {"subject": "Invoice 4471 is ready"},
            "    row = only(ctx, ctx.see.find_text(subject), subject)\n"
            "    ctx.ctl.click(checkbox_for(ctx, row, subject))\n"
            '    toolbar = ctx.see.find_text("Archive")\n'
            '    ctx.ctl.click(region(ctx, toolbar, 200, 50, 900, 112, "the Archive button"))\n'
            "    return True\n",
            sees("Archived 1"),
        ),
        bind=bind(
            "archive_message",
            {"subject": "Invoice 4471 is ready"},
            "the sentence names the subject of the message to archive",
        ),
    ),
    Playbook(
        task='Add the "Travel" label to Helen Okafor\'s message "Lunch on Thursday?".',
        steps=(
            select_row_then("Lunch on Thursday", "Label"),
            press("Travel", within=LABEL_MENU, done=True),
        ),
        skill=skill(
            "label_message",
            "Add a label to one message in the mailbox.",
            "the mail screen, with the label applied.",
            ("subject", "label"),
            {"subject": "Lunch on Thursday?", "label": "Travel"},
            "    row = only(ctx, ctx.see.find_text(subject), subject)\n"
            "    ctx.ctl.click(checkbox_for(ctx, row, subject))\n"
            '    toolbar = ctx.see.find_text("Label")\n'
            '    ctx.ctl.click(region(ctx, toolbar, 200, 50, 900, 112, "the Label button"))\n'
            "    menu = ctx.see.find_text(label)\n"
            '    ctx.ctl.click(region(ctx, menu, 300, 115, 760, 320, label + " in the menu"))\n'
            "    return True\n",
            sees("Labelled 1"),
        ),
        bind=bind(
            "label_message",
            {"subject": "Lunch on Thursday?", "label": "Travel"},
            "the sentence names both the message and the label to add",
        ),
    ),
    Playbook(
        task="Sort the records table by Priority, descending.",
        steps=(
            go("Records"),
            press("Priority", within=COLUMN_HEADERS),
            press("Priority", within=COLUMN_HEADERS, done=True),
        ),
        skill=skill(
            "sort_records",
            "Sort the records table by a column, in a direction.",
            "the Records screen, sorted.",
            ("column", "direction"),
            {"column": "Priority", "direction": "descending"},
            '    go_to(ctx, "Records")\n'
            "    headers = ctx.see.find_text(column)\n"
            '    ctx.ctl.click(region(ctx, headers, 0, 180, 1280, 228, column + " header"))\n'
            '    if direction.lower().startswith("desc"):\n'
            "        # The first click sorts ascending; clicking the header again reverses it.\n"
            "        again = ctx.see.find_text(column)\n"
            '        ctx.ctl.click(region(ctx, again, 0, 180, 1280, 228, column + " header"))\n'
            "    return True\n",
            sees("Sorted by"),
        ),
        bind=bind(
            "sort_records",
            {"column": "Priority", "direction": "descending"},
            "the sentence names the column and the direction",
        ),
    ),
    Playbook(
        task='Rename the record "Aurora Ledger" to "Aurora Ledger v2".',
        steps=(go("Records"), *rename_record("Aurora Ledger", "Aurora Ledger v2")),
        skill=skill(
            "rename_record",
            "Rename one row of the records table.",
            "the Records screen, with the row renamed and the editor closed.",
            ("record", "new_name"),
            {"record": "Aurora Ledger", "new_name": "Aurora Ledger v2"},
            '    go_to(ctx, "Records")\n'
            "    row = only(ctx, ctx.see.find_text(record), record)\n"
            "    middle = row.box.y + row.box.h // 2\n"
            "    pencil = None\n"
            "    for element in ctx.see.all():\n"
            '        if element.kind.value != "button" or element.box.x < 1100:\n'
            "            continue\n"
            "        if abs(element.box.y + element.box.h // 2 - middle) > 26:\n"
            "            continue\n"
            "        if pencil is None or element.box.x > pencil.box.x:\n"
            "            pencil = element\n"
            '    ctx.expect(pencil is not None, "no edit button on the row for " + record)\n'
            "    ctx.ctl.click(pencil)\n"
            "    editor = None\n"
            "    for element in ctx.see.find_text(record):\n"
            "        # The editor is a bordered box holding the old name. Whether the\n"
            "        # detector calls that a text field or a button varies, so it is\n"
            "        # identified by its text and its size - a row is far taller.\n"
            "        if element.box.h > 34:\n"
            "            continue\n"
            "        if editor is None or element.box.area < editor.box.area:\n"
            "            editor = element\n"
            '    ctx.expect(editor is not None, "the rename box did not open")\n'
            "    fill_in(ctx, editor, new_name)\n"
            '    ctx.ctl.press("Enter")\n'
            "    return True\n",
            sees("Renamed to"),
        ),
        bind=bind(
            "rename_record",
            {"record": "Aurora Ledger", "new_name": "Aurora Ledger v2"},
            "the sentence names the record and what to call it",
        ),
    ),
    Playbook(
        task='Change the display name in Settings to "Avery Q." and save it.',
        steps=(
            go("Settings"),
            fill("Avery", "Avery Q."),
            save_settings,
            press("Confirm and save", done=True),
        ),
        skill=skill(
            "change_display_name",
            "Change the display name in Settings and confirm the save.",
            "the Settings screen, saved.",
            ("display_name",),
            {"display_name": "Avery Q."},
            '    go_to(ctx, "Settings")\n'
            "    box = None\n"
            "    for element in fields(ctx):\n"
            "        if box is None or element.box.y < box.box.y:\n"
            "            box = element\n"
            '    ctx.expect(box is not None, "no text box on the settings screen")\n'
            "    fill_in(ctx, box, display_name)\n"
            '    hits = controls(ctx, "Save changes")\n'
            "    if not hits:\n"
            "        # The save bar sits below the fold on a short window.\n"
            "        ctx.ctl.scroll(biggest(ctx), 0, 400)\n"
            '        hits = controls(ctx, "Save changes")\n'
            '    ctx.ctl.click(only(ctx, hits, "the save button"))\n'
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Confirm and save"), "the confirm"))\n'
            "    return True\n",
            sees("Settings saved"),
        ),
        bind=bind(
            "change_display_name",
            {"display_name": "Avery Q."},
            "the sentence quotes the new display name",
        ),
    ),
    # -- composite ---------------------------------------------------------------------
    Playbook(
        task="Find the message about scheduled maintenance for the file server and archive it.",
        steps=search_then_archive("maintenance", "Scheduled maintenance"),
        skill=skill(
            "find_and_archive_message",
            "Find a message anywhere in the mailbox and archive it.",
            "the mail screen, with that conversation archived.",
            ("query", "subject"),
            {"query": "maintenance", "subject": "Scheduled maintenance"},
            "    # Searching, not scrolling: this application clips its message list,\n"
            "    # so a conversation past the bottom of the window cannot be reached by\n"
            "    # wheeling at all. Searching is what finding one means here.\n"
            '    box = only(ctx, ctx.see.find_text("Search mail"), "the mail search box")\n'
            "    fill_in(ctx, box, query)\n"
            "    row = only(ctx, ctx.see.find_text(subject), subject)\n"
            "    ctx.ctl.click(checkbox_for(ctx, row, subject))\n"
            '    toolbar = ctx.see.find_text("Archive")\n'
            '    ctx.ctl.click(region(ctx, toolbar, 200, 50, 900, 112, "the Archive button"))\n'
            "    return True\n",
            sees("Archived 1"),
        ),
        bind=bind(
            "find_and_archive_message",
            {"query": "maintenance", "subject": "Scheduled maintenance"},
            "the sentence describes the message to find and archive",
        ),
    ),
    Playbook(
        task=(
            "Filter the records table to the Security category, then set every row shown to "
            "the Paused status."
        ),
        steps=(
            go("Records"),
            fill("Filter records", "Security"),
            select_all_then_bulk,
            press("Paused", within=BULK_MENU, done=True),
        ),
        skill=skill(
            "set_status_of_filtered_records",
            "Filter the records table and set every row shown to one status.",
            "the Records screen, with the filtered rows at the new status.",
            ("category", "status"),
            {"category": "Security", "status": "Paused"},
            '    go_to(ctx, "Records")\n'
            '    field = only(ctx, ctx.see.find_text("Filter records"), "the records filter")\n'
            "    fill_in(ctx, field, category)\n"
            "    top = None\n"
            "    for element in boxes(ctx):\n"
            "        if element.box.x > 60:\n"
            "            continue\n"
            "        if top is None or element.box.y < top.box.y:\n"
            "            top = element\n"
            '    ctx.expect(top is not None, "no select-all checkbox on the records table")\n'
            "    ctx.ctl.click(top)\n"
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Bulk actions"), "the bulk menu"))\n'
            "    menu = ctx.see.find_text(status)\n"
            '    ctx.ctl.click(region(ctx, menu, 280, 170, 560, 350, status + " in the menu"))\n'
            "    return True\n",
            sees("records to"),
        ),
        bind=bind(
            "set_status_of_filtered_records",
            {"category": "Security", "status": "Paused"},
            "the sentence names the category to filter to and the status to set",
        ),
    ),
    Playbook(
        task='Filter the records table for "ledger" and export the shown rows to CSV.',
        steps=(
            go("Records"),
            fill("Filter records", "ledger"),
            press("Export", kind="button", done=True),
        ),
        skill=skill(
            "export_filtered_records",
            "Filter the records table and export the rows shown to CSV.",
            "the Records screen, with the export recorded.",
            ("query",),
            {"query": "ledger"},
            '    go_to(ctx, "Records")\n'
            '    field = only(ctx, ctx.see.find_text("Filter records"), "the records filter")\n'
            "    fill_in(ctx, field, query)\n"
            "    # The BUTTON: the footer also counts 'Exports this session', and the\n"
            "    # tightest thing that says the word is that line rather than the control.\n"
            '    ctx.ctl.click(only(ctx, controls(ctx, "Export"), "the export button"))\n'
            "    return True\n",
            sees("Exported"),
        ),
        bind=bind(
            "export_filtered_records",
            {"query": "ledger"},
            'the sentence quotes the filter: "ledger"',
        ),
    ),
    Playbook(
        task=(
            "Open Dana Whitfield's message about the Q3 capacity plan, then compose and send "
            'a message to dana.whitfield@northwind.example with the subject "Signed off".'
        ),
        steps=(
            press("Q3 capacity plan"),
            press("Compose"),
            compose("dana.whitfield@northwind.example", "Signed off"),
            press("Send", within=COMPOSE, done=True),
        ),
        skill=skill(
            "read_then_send",
            "Open a message, then compose and send a new one.",
            "the mail screen, with the message sent.",
            ("subject_to_open", "to", "subject"),
            {
                "subject_to_open": "Q3 capacity plan",
                "to": "dana.whitfield@northwind.example",
                "subject": "Signed off",
            },
            "    ctx.ctl.click(only(ctx, ctx.see.find_text(subject_to_open), subject_to_open))\n"
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Compose"), "the compose button"))\n'
            "    window = []\n"
            "    for element in fields(ctx):\n"
            "        if element.box.x + element.box.w // 2 < 760:\n"
            "            continue\n"
            "        if element.box.y + element.box.h // 2 < 430:\n"
            "            continue\n"
            "        window.append(element)\n"
            '    ctx.expect(len(window) >= 2, "the compose window has no address box")\n'
            "    first = window[0]\n"
            "    second = window[1]\n"
            "    for element in window:\n"
            "        if element.box.y < first.box.y:\n"
            "            second = first\n"
            "            first = element\n"
            "        elif element.box.y < second.box.y and element.box.y > first.box.y:\n"
            "            second = element\n"
            "    ctx.ctl.click(first)\n"
            "    ctx.ctl.type_text(to)\n"
            "    ctx.ctl.click(second)\n"
            "    ctx.ctl.type_text(subject)\n"
            "    send = ctx.see.find_text('Send')\n"
            '    ctx.ctl.click(region(ctx, send, 760, 430, 1280, 800, "the send button"))\n'
            "    return True\n",
            sees("Message sent"),
        ),
        bind=bind(
            "read_then_send",
            {
                "subject_to_open": "Q3 capacity plan",
                "to": "dana.whitfield@northwind.example",
                "subject": "Signed off",
            },
            "the sentence names the message to open, the address and the subject",
        ),
    ),
)


def playbooks() -> dict[str, Playbook]:
    """Every task of the shipped suite, keyed by the sentence the agent is given."""
    return {playbook.task: playbook for playbook in _ALL}


PLAYBOOKS = playbooks()


if __name__ == "__main__":  # a quick check that every skill at least parses
    for book in _ALL:
        if book.skill is not None:
            ast.parse(book.skill["code"])
            ast.parse(book.skill["verifier_code"])
    print(json.dumps({"playbooks": len(PLAYBOOKS)}, indent=2))
