"""What every site's playbooks share: the step builders and the skill scaffolding.

One playbook module per application (``playbooks_sandbox``, ``playbooks_shop``,
``playbooks_board``) describes how that application's tasks are done and what skill
each run teaches. This module holds the parts that are the same everywhere, because
they are about driving a WEB PAGE rather than about driving any particular one:
clicking a labelled control, replacing the contents of a field, choosing from a
native select, finding the control that belongs to a card or a row.

That split is the point. If a new application needed new machinery here, the machinery
would be the thing being tested rather than the agent; what a new application needs is
its own task list and its own knowledge of what its buttons are called.

See ``scripts/scripted_operator.py`` for what a playbook is standing in for, and what
that does and does not prove.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scripted_operator import (  # type: ignore[import-not-found]
    Element,
    click,
    find,
    squash,
)

__all__ = [
    "HELPERS",
    "TOP_BAR",
    "below",
    "bind",
    "choose",
    "fill",
    "go",
    "owned_by",
    "press",
    "replace_text",
    "sees",
    "skill",
    "twice",
]

TOP_BAR = (0, 0, 2000, 52)
"""The application bar, where a site's navigation lives. Most pages repeat their
section names as headings and footers, so a navigation click is bounded to the strip
where the navigation actually is."""


# --------------------------------------------------------------------------------------
# Step builders: one move each, expressed against what perception reports
# --------------------------------------------------------------------------------------


def go(label: str, *, done: bool = False):
    """Click a top-bar navigation control."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        target = find(elements, label, within=TOP_BAR)
        return None if target is None else click(target, done=done)

    return step


def press(label: str, *, kind: str | None = None, within=None, nth: int = 0, done: bool = False):
    """Click the first thing whose text contains ``label``, optionally in a region."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        target = find(elements, label, kind=kind, within=within, nth=nth)
        return None if target is None else click(target, done=done)

    return step


def twice(label: str, *, within=None, done: bool = False):
    """Click the same control twice in one move.

    For a control whose whole purpose is repetition - a quantity that goes up by one
    per press, a sort header that reverses on the second click. Two presses are one
    decision, and splitting them would have the critic judge a half-finished intent.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        target = find(elements, label, within=within)
        if target is None:
            return None
        return {
            "code": f'ctx.ctl.click(el["{target.id}"])\nctx.ctl.click(el["{target.id}"])\n',
            "done": done,
        }

    return step


def replace_text(element_id: str, value: str) -> str:
    """Code for "click this field and make it say exactly this".

    One move, not two. Focusing a field changes nothing a critic can see - some
    applications even hide the caret - so a bare click on a text box is judged a move
    that did nothing and the next one is refused as a repeat. Clicking and typing IS
    one decision.
    """
    return (
        f'ctx.ctl.click(el["{element_id}"])\n'
        'ctx.ctl.press("Control", "a")\n'
        f"ctx.ctl.type_text({value!r})\n"
    )


def fill(label: str, value: str, *, kind: str | None = None, within=None, done: bool = False):
    """Click the field found by ``label`` and replace its contents."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        field = find(elements, label, kind=kind, within=within)
        if field is None:
            return None
        return {"code": replace_text(field.id, value), "done": done}

    return step


def choose(shown: str, option: str, *, within=None, done: bool = False):
    """Pick an option from a native select by TYPING its label.

    A dropdown list is drawn by the operating system, not by the page, so it is not in
    any screenshot and cannot be clicked at. What a keyboard user does instead works
    perfectly: focus the control, type enough of the option to identify it, and press
    Enter. That is label-driven rather than position-driven, so it survives an option
    list in a different order - which is the whole difference between working on one
    site and working on a site.

    Args:
        shown: Text of the option currently selected, which is what is on screen.
        option: Enough of the wanted option's label to identify it by prefix.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        target = find(elements, shown, within=within)
        if target is None:
            return None
        return {
            "code": (
                f'ctx.ctl.click(el["{target.id}"])\n'
                f"ctx.ctl.type_text({option!r})\n"
                'ctx.ctl.press("Enter")\n'
            ),
            "done": done,
        }

    return step


# --------------------------------------------------------------------------------------
# Finding the control that belongs to something
# --------------------------------------------------------------------------------------


def owned_by(
    elements: Sequence[Element],
    label: str,
    title: Element,
    *,
    below_by: tuple[int, int] = (20, 130),
    across: int = 240,
) -> Element | None:
    """The control named ``label`` that belongs to the card or row headed ``title``.

    A grid repeats every one of its controls - twenty-four "Add to cart" buttons, one
    per product - so naming the control is not enough to say WHICH one. What
    distinguishes them is where they are: the one that belongs to a card is below that
    card's name and within that card.

    Bounded from the title's LEFT EDGE rightwards, not around its centre. A card's
    title starts at the card's left edge while its button is usually pushed to the
    right edge, so a window around the centre reaches into the neighbouring card and
    can be nearer to the wrong one - which is how "two of the Brass Level" put two
    hammers in the cart.
    """
    wanted = squash(label)
    best: Element | None = None
    for element in elements:
        if wanted not in squash(element.text):
            continue
        drop = element.cy - title.cy
        if not below_by[0] <= drop <= below_by[1]:
            continue
        if not 0 <= element.x - title.x <= across:
            continue
        if best is None or element.cy < best.cy:
            best = element
    return best


def below(elements: Sequence[Element], label: str, *, gap: int = 60, across: int = 90):
    """The element directly underneath a label - the field a caption names.

    A form field shows a placeholder until something is typed into it, and then shows
    what was typed, so it cannot reliably be found by its own text. Its caption does
    not move.
    """
    caption = find(elements, label)
    if caption is None:
        return None
    best: Element | None = None
    for element in elements:
        drop = element.cy - caption.cy
        if not 8 <= drop <= gap or not -8 <= element.x - caption.x <= across:
            continue
        if element.w < 80:
            continue
        if best is None or element.cy < best.cy:
            best = element
    return best


# --------------------------------------------------------------------------------------
# Skills: what a run teaches, in the shape the synthesizer is asked to reply with
# --------------------------------------------------------------------------------------

HELPERS = '''
def only(ctx, hits, what):
    """The first hit, or a clean failure naming what was wanted."""
    ctx.expect(bool(hits), "nothing on screen matching " + what)
    return hits[0]


def titled(ctx, text, what):
    """The smallest thing on screen that says this, allowing for truncation.

    A narrow column cuts a long title off, so the whole of it is never on screen;
    asking for less and less of it finds the card without needing to know where the
    cut falls. The smallest match is then where the words actually are - a click
    inside it is inside every larger thing containing it.
    """
    words = text.split()
    for stop in range(len(words), 2, -1):
        hits = ctx.see.find_text(" ".join(words[:stop]))
        if hits:
            best = hits[0]
            for hit in hits:
                if hit.box.area < best.box.area:
                    best = hit
            return best
    ctx.expect(False, "nothing on screen matching " + what)


def tightest(ctx, text, what):
    """The SMALLEST thing on screen that says this.

    A page nests its text: a card's title is inside the card, and a detector will
    sometimes draw one band across a whole row of columns that contains several
    cards' worth of words. The smallest match is where the words actually are, and a
    click inside it is inside every larger one too.
    """
    hits = ctx.see.find_text(text)
    ctx.expect(bool(hits), "nothing on screen matching " + what)
    best = hits[0]
    for hit in hits:
        if hit.box.area < best.box.area:
            best = hit
    return best


def region(ctx, hits, x0, y0, x1, y1, what):
    """The first hit whose middle lies in a part of the screen.

    Pages repeat their words: a toolbar has an Archive button and a sidebar an
    Archived folder, and clicking the wrong one quietly does the wrong thing.
    """
    for hit in hits:
        cx = hit.box.x + hit.box.w // 2
        cy = hit.box.y + hit.box.h // 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return hit
    ctx.expect(False, "nothing matching " + what + " where it was expected")


def boxes(ctx):
    """Every checkbox on screen."""
    return [e for e in ctx.see.all() if e.kind.value == "checkbox"]


def fields(ctx):
    """Every text box on screen."""
    return [e for e in ctx.see.all() if e.kind.value == "text_field"]


def controls(ctx, text):
    """Controls whose label matches - not the prose that happens to read like it."""
    return [e for e in ctx.see.find_text(text) if e.kind.value in ("button", "link", "menu")]


def biggest(ctx):
    """The largest thing on screen: where to put the pointer before scrolling.

    A wheel event scrolls whatever is under it, and the first element in reading order
    is in the application bar, which scrolls nothing at all.
    """
    best = None
    for element in ctx.see.all():
        if best is None or element.box.area > best.box.area:
            best = element
    ctx.expect(best is not None, "the screen appears to be empty")
    return best


def checkbox_for(ctx, row, what):
    """The checkbox belonging to a list row: leftmost, aligned with it."""
    middle = row.box.y + row.box.h // 2
    best = None
    for element in boxes(ctx):
        centre = element.box.y + element.box.h // 2
        if abs(centre - middle) > 30 or element.box.x > row.box.x:
            continue
        if best is None or element.box.x < best.box.x:
            best = element
    ctx.expect(best is not None, "no checkbox on the row for " + what)
    return best


def owned_by(ctx, label, title, low, high, across):
    """The control named `label` belonging to the card or row headed `title`.

    A grid repeats every control it has, once per card, so the name alone does not say
    which one. Where it is does: below that card's title and inside that card, which
    is measured rightwards from the title's left edge - a card's title starts at its
    left edge while its button is pushed to the right one.
    """
    best = None
    for element in ctx.see.find_text(label):
        drop = element.box.y - title.box.y
        if drop < low or drop > high:
            continue
        if element.box.x < title.box.x or element.box.x - title.box.x > across:
            continue
        if best is None or element.box.y < best.box.y:
            best = element
    ctx.expect(best is not None, "no " + label + " belonging to " + title.text)
    return best


def under(ctx, label, gap, across):
    """The field a caption names, found by where it is rather than what it says.

    A form field shows a placeholder until it is typed into and its own text after,
    so its own text cannot identify it. Its caption does not move.
    """
    caption = only(ctx, ctx.see.find_text(label), label)
    best = None
    for element in ctx.see.all():
        drop = element.box.y - caption.box.y
        if drop < 8 or drop > gap:
            continue
        # A field is left-aligned with its caption and is field-sized. Without both,
        # the nearest thing under a caption can be a fragment of whatever the panel
        # happens to be drawn on top of.
        if element.box.x - caption.box.x < -8 or element.box.x - caption.box.x > across:
            continue
        if element.box.w < 80:
            continue
        if best is None or element.box.y < best.box.y:
            best = element
    ctx.expect(best is not None, "nothing under " + label)
    return best


def fill_in(ctx, field, value):
    """Replace a field's contents. The caret may be invisible, so select all first."""
    ctx.ctl.click(field)
    ctx.ctl.press("Control", "a")
    ctx.ctl.type_text(value)


def pick(ctx, control, option):
    """Choose an option from a native select by typing its label.

    The dropdown is drawn by the operating system and is in no screenshot, so it
    cannot be clicked at. Typing enough of the option to identify it is what a
    keyboard user does, and it depends on the option's NAME rather than its position.
    """
    ctx.ctl.click(control)
    ctx.ctl.type_text(option)
    ctx.ctl.press("Enter")


def go_to(ctx, screen):
    """Click a top-bar navigation control, not a heading of the same name."""
    ctx.ctl.click(region(ctx, ctx.see.find_text(screen), 0, 0, 2000, 52, screen))
'''


def skill(
    name: str,
    summary: str,
    ends_on: str,
    params: Sequence[str],
    example: dict[str, Any],
    body: str,
    verifier: str,
) -> dict[str, Any]:
    """One skill in the shape the synthesizer is asked to reply with."""
    return {
        "name": name,
        "summary": summary,
        "docstring": (
            f"{summary}\n\nAssumes: the application is open on its first screen.\n"
            f"Ends on: {ends_on}"
        ),
        "params": {
            parameter: {"type": "string", "description": f"The {parameter.replace('_', ' ')}."}
            for parameter in params
        },
        "example_args": example,
        "requires": [],
        "code": f"{HELPERS}\ndef run({', '.join(['ctx', *params])}):\n{body}\n",
        "verifier_code": verifier,
    }


def bind(name: str, args: dict[str, Any], why: str) -> dict[str, Any]:
    """The composer's reply: which stored skill, with what read out of the sentence."""
    return {"steps": [{"skill": name, "args": args}], "why": why}


def sees(text: str) -> str:
    """A verifier that passes when something is visibly on screen afterwards."""
    return f'def verify(ctx, result):\n    return bool(ctx.see.find_text("{text}"))\n'
