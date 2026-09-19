"""The hardening pass, and in particular the rule that decides whether a stored skill
survives its second run: anchor on MEANING, never on position.

``ctx.see.by_kind("text")[1]`` is how a model writes "the search box" after watching
one page. It is true of that page and of nothing else - an advert, a banner, one more
caption read by OCR, and the index points somewhere else. This module tests two
things and they are not the same thing. :func:`positional_lookups` is the DETECTOR: it
must spot every shape of the defect and must not cry wolf over ``find_text(...)[0]``,
where the index means "the best match". :func:`harden` is the REPAIR: it rewrites what
the recording can ground and, where nothing names the element, keeps the positional
lookup and makes the skill log that it has one.

That last case is the one to be careful about. A skill that works positionally and
says so is better than no skill, so nothing here may turn a brittle skill into a
rejected one. The admission gate in ``test_synthesize.py`` is what decides whether a
skill is any good; this pass only decides what it is asked to judge.

No browser, no model, no network: the fake invoicing app of ``tests/fakes/scenario.py``
is the recording, and the elements with no text are built by hand, because that app
labels everything and the interesting case is a control that nothing labels.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from skillweaver.contracts import (
    Box,
    Click,
    Element,
    ElementKind,
    ElementSource,
    Trajectory,
)
from skillweaver.skills.refactor import PositionalLookup, harden, positional_lookups
from tests.fakes import InMemoryTrajectoryRecorder, Scenario, SimpleElementIndex

DOMAIN = "fake.test"


@pytest.fixture
def trajectory(scenario: Scenario) -> Trajectory:
    """The recorded solution of the fake app, replayed from its start state."""
    recorder = InMemoryTrajectoryRecorder()
    recorder.start(scenario.task.text, DOMAIN)
    for action in scenario.solution:
        before = scenario.perceiver.observe(scenario.controller)
        result = scenario.controller.perform(action)
        after = scenario.perceiver.observe(scenario.controller)
        recorder.step(action, before, after, result)
    finished = recorder.finish(ok=True, note="solved by exploration")
    scenario.controller.reset()
    return finished


# --------------------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------------------


def test_the_detector_spots_a_positional_subscript_written_directly():
    code = 'def run(ctx):\n    ctx.ctl.click(ctx.see.by_kind("text")[1])\n'
    assert positional_lookups(code) == (PositionalLookup("by_kind", "text", 1, line=2),)


def test_the_detector_spots_a_positional_subscript_through_a_variable():
    code = (
        "def run(ctx):\n"
        '    rows = ctx.see.by_kind("row")\n'
        '    ctx.expect(len(rows) >= 3, "too few rows")\n'
        "    ctx.ctl.click(rows[2])\n"
    )
    assert positional_lookups(code) == (PositionalLookup("by_kind", "row", 2, via="rows", line=4),)


def test_the_detector_spots_a_subscript_of_all_and_an_index_from_the_end():
    code = (
        "def run(ctx):\n"
        "    fifth = ctx.see.all()[4]\n"
        "    last = ctx.see.all()[-1]\n"
        "    ctx.ctl.click(fifth)\n"
        "    ctx.ctl.click(last)\n"
    )
    found = positional_lookups(code)
    assert [(f.query, f.kind, f.index) for f in found] == [("all", None, 4), ("all", None, -1)]
    assert found[1].position == "1 from the end"


def test_the_detector_spots_one_inside_a_loop_or_a_branch():
    code = (
        "def run(ctx):\n"
        "    for _ in range(2):\n"
        '        ctx.ctl.click(ctx.see.by_kind("row")[0])\n'
        "    if ctx.see.find_text('Next'):\n"
        "        ctx.ctl.click(ctx.see.all()[3])\n"
    )
    assert [str(f) for f in positional_lookups(code)] == [
        "ctx.see.by_kind('row')[0]",
        "ctx.see.all()[3]",
    ]


def test_the_detector_leaves_a_named_lookup_alone():
    """``find_text(...)[0]`` is "the best match", not "the first thing on screen".
    Flagging it would make the pass rewrite its own output forever."""
    code = (
        "def run(ctx):\n"
        '    ctx.ctl.click(ctx.see.find_text("Search", "text_field")[0])\n'
        '    ctx.ctl.click(ctx.see.best("blue submit button")[0])\n'
        "    row = ctx.see.find_text('Acme')\n"
        "    ctx.ctl.click(ctx.see.nearest(row[0].box.center, 'checkbox')[0])\n"
        "    ctx.ctl.click(ctx.see.containing(row[0].box.center)[0])\n"
    )
    assert positional_lookups(code) == ()


def test_the_detector_leaves_an_index_that_is_not_a_fixed_position():
    """A computed index, a slice and a loop variable are not positions a page can
    move out from under - and rewriting them would change what the skill does."""
    code = (
        "def run(ctx, n):\n"
        '    rows = ctx.see.by_kind("row")\n'
        "    ctx.ctl.click(rows[n])\n"
        "    for row in rows[1:]:\n"
        "        ctx.ctl.click(row)\n"
        "    for index in range(len(rows)):\n"
        "        ctx.ctl.click(rows[index])\n"
    )
    assert positional_lookups(code) == ()


def test_the_detector_forgets_a_variable_that_was_rebound():
    """``rows`` holding a ``find_text`` result by the time it is subscripted is not a
    positional lookup, whatever it held earlier."""
    code = (
        "def run(ctx):\n"
        '    rows = ctx.see.by_kind("row")\n'
        '    rows = ctx.see.find_text("Acme Corp", "row")\n'
        "    ctx.ctl.click(rows[0])\n"
    )
    assert positional_lookups(code) == ()


def test_the_detector_reports_nothing_for_source_it_cannot_read():
    assert positional_lookups("def run(ctx:\n") == ()
    assert positional_lookups("x = ctx.see.all()[0]\n") == ()


# --------------------------------------------------------------------------------------
# The repair: anchoring on meaning
# --------------------------------------------------------------------------------------


def test_a_positional_pick_becomes_a_named_lookup(trajectory):
    """The defect the live Wikipedia run shipped: reach the thing by counting. The
    recording says which element that count landed on, so it is named instead."""
    draft = (
        "def run(ctx):\n"
        '    rows = ctx.see.by_kind("row")\n'
        '    ctx.expect(len(rows) >= 1, "no rows")\n'
        "    ctx.ctl.click(rows[0])\n"
        '    ctx.ctl.click(ctx.see.by_kind("button")[0])\n'
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.positions_anchored == 2
    assert hardened.positions_announced == ()
    assert positional_lookups(hardened.code) == ()
    assert "ctx.see.find_text('Acme Corp INV-1042 $1,200.00', 'row')" in hardened.code
    assert "ctx.see.find_text('Confirm payment', 'button')" in hardened.code
    assert "ctx.ctl.click(acme_corp_inv_row[0])" in hardened.code
    assert "ctx.ctl.click(confirm_payment_button[0])" in hardened.code
    # Every introduced lookup is checked before it is indexed.
    assert hardened.code.count("ctx.expect(bool(") == 2


def test_an_anchored_lookup_is_reported_in_the_change_log(trajectory):
    draft = 'def run(ctx):\n    ctx.ctl.click(ctx.see.by_kind("button")[0])\n'
    hardened = harden(draft, trajectory)
    assert any("anchored" in change and "Confirm payment" in change for change in hardened.changes)


def test_a_position_the_recording_cannot_ground_is_kept_and_announced(trajectory):
    """The rule that must NOT become a rejection: nothing in this run is a checkbox,
    so there is no anchor to find. The lookup stays, and the skill says so."""
    draft = (
        "def run(ctx):\n"
        '    boxes = ctx.see.by_kind("checkbox")\n'
        '    ctx.expect(len(boxes) >= 2, "no checkboxes")\n'
        "    ctx.ctl.click(boxes[1])\n"
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.positions_anchored == 0
    assert hardened.positions_unanchored == 1
    assert "ctx.ctl.click(boxes[1])" in hardened.code
    assert "ctx.log(" in hardened.code
    assert "checkbox number 2 in reading order" in hardened.code
    assert any("navigates by position" in change for change in hardened.changes)


def test_a_skill_that_already_owned_up_is_not_made_to_own_up_twice(trajectory):
    """The archive skill's own line - "sender name not readable" - is the model doing
    what the prompt asks. Injecting a second confession would only add noise."""
    draft = (
        "def run(ctx):\n"
        '    ctx.log("sender name not readable; taking the second row")\n'
        '    ctx.ctl.click(ctx.see.by_kind("checkbox")[1])\n'
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.code.count("ctx.log(") == 1
    assert "sender name not readable" in hardened.code
    assert hardened.positions_unanchored == 1


def test_an_ambiguous_position_is_not_guessed_at(trajectory):
    """``by_kind("text")[0]`` is a different element on every screen of this run and
    nothing acted on any of them. A guess rewritten into a skill is worse than the
    index it replaced, so the index stays - and announces itself."""
    draft = (
        "def run(ctx):\n"
        '    heading = ctx.see.by_kind("text")[0]\n'
        "    ctx.ctl.scroll(heading, dy=120)\n"
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.positions_anchored == 0
    assert hardened.positions_unanchored == 1
    assert "heading = ctx.see.by_kind('text')[0]" in hardened.code


def test_an_already_anchored_skill_is_left_exactly_as_written(trajectory):
    """Hardening is not a tax: code that already names what it wants is untouched."""
    draft = (
        "def run(ctx, company):\n"
        "    rows = ctx.see.find_text(company, 'row')\n"
        "    ctx.expect(bool(rows), 'no row for the company')\n"
        "    ctx.ctl.click(rows[0])\n"
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.positions_anchored == 0
    assert hardened.positions_announced == ()
    assert "ctx.log(" not in hardened.code
    assert not hardened.changed


def test_an_anchored_lookup_is_placed_inside_the_block_that_needs_it(trajectory):
    """A lookup hoisted out of an ``if`` would run when the skill said not to, and
    its ``ctx.expect`` would fail a run that was fine."""
    draft = (
        "def run(ctx):\n"
        "    if ctx.see.find_text('Confirm payment'):\n"
        '        ctx.ctl.click(ctx.see.by_kind("button")[0])\n'
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    lines = hardened.code.splitlines()
    wanted = "find_text('Confirm payment', 'button')"
    lookup = next(i for i, line in enumerate(lines) if wanted in line)
    assert lines[lookup].startswith("        "), hardened.code


# --------------------------------------------------------------------------------------
# Anchoring an element that carries no text of its own
# --------------------------------------------------------------------------------------


def _element(kind: ElementKind, text: str, box: Box, stable_id: str) -> Element:
    return Element(box, kind, text, 0.9, stable_id, ElementSource.merged)


@pytest.fixture
def inbox(scenario: Scenario) -> Trajectory:
    """One recorded click on an UNLABELLED checkbox that sits inside a labelled row.

    Built by hand because the fake app labels everything, and the case that matters
    is the one the live archive run hit: OCR reads the row and not the control in it.
    """
    row = _element(ElementKind.row, "Dana Whitfield  Re: invoice", Box(20, 120, 760, 40), "row-1")
    checkbox = _element(ElementKind.checkbox, "", Box(28, 132, 16, 16), "check-1")
    other = _element(ElementKind.checkbox, "", Box(28, 172, 16, 16), "check-2")
    second = _element(ElementKind.row, "Kim Lau  Lunch?", Box(20, 160, 760, 40), "row-2")

    recorder = InMemoryTrajectoryRecorder()
    recorder.start("Archive the message from Dana Whitfield.", DOMAIN)
    elements = (row, checkbox, second, other)
    before = replace(
        scenario.perceiver.observe(scenario.controller),
        elements=elements,
        index=SimpleElementIndex(elements),
        url="https://mail.test/inbox",
    )
    action = Click(checkbox.box.center)
    result = scenario.controller.perform(action)
    after = scenario.perceiver.observe(scenario.controller)
    recorder.step(action, before, after, result)
    return recorder.finish(ok=True)


def test_a_textless_element_is_anchored_on_the_labelled_thing_it_sits_in(inbox):
    """No text on the checkbox, so it is found FROM the row that reads a name - which
    survives the row moving, and its position in the checkbox list does not."""
    draft = 'def run(ctx):\n    ctx.ctl.click(ctx.see.by_kind("checkbox")[0])\n    return True\n'
    hardened = harden(draft, inbox)

    assert hardened.positions_anchored == 1
    assert hardened.positions_announced == ()
    assert positional_lookups(hardened.code) == ()
    assert "ctx.see.find_text('Dana Whitfield  Re: invoice', 'row')" in hardened.code
    assert "ctx.see.nearest(dana_whitfield_re_row[0].box.center, 'checkbox')" in hardened.code
    assert "ctx.ctl.click(checkbox[0])" in hardened.code
    assert hardened.code.count("ctx.expect(bool(") == 2


def test_a_literal_coordinate_on_a_textless_element_is_anchored_too(inbox):
    """The coordinate pass alone would write ``by_kind("checkbox")[0]`` - its own
    fallback when an element has no text. That is the same defect, and the anchoring
    pass runs after it for exactly this reason."""
    draft = "def run(ctx):\n    ctx.ctl.click(Point(36, 140))\n    return True\n"
    hardened = harden(draft, inbox)

    assert hardened.coordinates_replaced == 1
    assert hardened.positions_anchored == 1
    assert "Point(" not in hardened.code
    assert positional_lookups(hardened.code) == ()
    assert "ctx.see.nearest(" in hardened.code
