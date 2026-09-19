"""The panel that answers "what will calling this skill actually DO?".

It reads the stored source of every skill and lays it out as an ordered procedure.
Three properties are what make it worth having, and each is tested here:

*It says what a step ANCHORS ON.* ``click`` is not information; ``click the element
whose text is "Confirm payment"`` is, and ``click the fixed point (400, 140)`` is the
warning that the skill will not survive a redesign.

*It is total.* Stored code the reader has no phrasing for still produces a step, and
code that does not parse at all produces a sentence saying so - never an exception and
never a skill that silently vanishes from the page.

*It correlates with the recording, honestly.* Where the run a skill was written from is
still on disk, its screens appear beside the steps they match, matched by verb in
order. Where the run is gone the steps still render, and the page says the run is gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillweaver.dashboard.build import (
    build_dashboard,
    build_trajectory_panel,
    collect,
    read_skill_steps,
)

FULL = Path(__file__).parents[1] / "dashboard_fixtures" / "full"


@pytest.fixture(scope="module")
def full_page(tmp_path_factory: pytest.TempPathFactory) -> str:
    out = tmp_path_factory.mktemp("dash") / "index.html"
    return build_dashboard(FULL, out).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# Reading a skill's steps out of its source
# --------------------------------------------------------------------------------------


def test_each_action_is_described_by_what_it_anchors_on() -> None:
    steps, _, problem = read_skill_steps(
        "def run(ctx, company):\n"
        "    field = ctx.see.find_text('Search', 'text_field')\n"
        "    ctx.ctl.click(field[0])\n",
        params=("company",),
    )
    assert problem == ""
    assert [(s.kind, s.verb) for s in steps] == [("look", "look for"), ("act", "click")]
    assert steps[0].anchor == 'text "Search" (text_field)'
    assert steps[1].anchor == 'text "Search" (text_field)', "the click inherits what was found"
    assert not any(s.brittle for s in steps)


def test_a_click_on_a_raw_coordinate_is_flagged_as_brittle() -> None:
    """The single most useful thing to see when deciding whether to trust a skill."""
    steps, _, _ = read_skill_steps("def run(ctx):\n    ctx.ctl.click(Point(400, 140))\n")
    assert steps[0].brittle
    assert "fixed point" in steps[0].anchor


def test_a_parameter_is_named_as_one_rather_than_as_a_value() -> None:
    steps, _, _ = read_skill_steps(
        "def run(ctx, company):\n"
        "    ctx.ctl.type_text(company)\n"
        "    rows = ctx.see.find_text(company, 'row')\n"
    )
    assert steps[0].detail == "the company parameter"
    assert steps[1].anchor == "text from the company parameter (row)", (
        "reading the kind filter as the search text would misstate what it looks for"
    )


def test_an_expectation_carries_the_sentence_it_fails_with() -> None:
    steps, _, _ = read_skill_steps(
        "def run(ctx):\n"
        "    rows = ctx.see.by_kind('row')\n"
        "    ctx.expect(bool(rows), 'the search returned no row')\n"
    )
    check = steps[-1]
    assert check.kind == "check"
    assert check.anchor == "any row"
    assert check.why == "the search returned no row"


def test_nesting_is_kept_so_a_branch_reads_as_one() -> None:
    steps, _, _ = read_skill_steps(
        "def run(ctx):\n"
        "    if ctx.see.all():\n"
        "        ctx.ctl.press('Enter')\n"
        "    else:\n"
        "        ctx.ctl.wait(250)\n"
    )
    assert [(s.verb, s.depth) for s in steps] == [
        ("if", 0),
        ("press", 1),
        ("otherwise", 0),
        ("wait", 1),
    ]
    assert steps[-1].detail == "250ms"


def test_code_with_no_phrasing_is_still_a_step_in_its_own_words() -> None:
    steps, _, problem = read_skill_steps(
        "def run(ctx):\n    total = sum(len(e.text) for e in ctx.see.all())\n"
    )
    assert problem == ""
    assert steps and "kept as total" in steps[0].detail


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("", "stores no run()"),
        ("def run(ctx", "does not parse"),
        ("x = 1", "defines no run() function"),
        ("def run(ctx):\n    pass\n", "empty body"),
    ],
)
def test_unreadable_code_is_reported_rather_than_raised(code: str, expected: str) -> None:
    steps, _, problem = read_skill_steps(code)
    assert steps == ()
    assert expected in problem


def test_a_verifier_reads_as_the_check_it_is() -> None:
    """``return bool(ctx.see.find_text(...))`` is where the skill judges its own work,
    and burying it as a returned value would hide the line a reader most wants."""
    checks, _, problem = read_skill_steps(
        "def verify(ctx, result):\n    return bool(ctx.see.find_text('Payment confirmed'))\n",
        function="verify",
    )
    assert problem == ""
    assert checks[-1].kind == "check"
    assert checks[-1].anchor == 'text "Payment confirmed"'


def test_a_url_read_out_of_stored_code_loses_its_scheme() -> None:
    """The page claims to need no network; nothing it prints may look like a live link."""
    steps, _, _ = read_skill_steps(
        "def run(ctx):\n    ctx.ctl.perform(Navigate('https://sandbox.test/records'))\n"
    )
    assert "https://" not in steps[0].detail
    assert "sandbox.test/records" in steps[0].detail


def test_a_very_long_skill_is_truncated_rather_than_swallowing_the_page() -> None:
    body = "\n".join("    ctx.ctl.press('Enter')" for _ in range(200))
    steps, truncated, _ = read_skill_steps(f"def run(ctx):\n{body}\n")
    assert truncated
    assert 0 < len(steps) <= 60


# --------------------------------------------------------------------------------------
# The panel, against stored skills
# --------------------------------------------------------------------------------------


def test_every_stored_skill_gets_a_procedure(full_page: str) -> None:
    panel = build_trajectory_panel(*_skills_and_dir(FULL))
    assert [r.name for r in panel.routines] == [
        "export_csv",
        "send_reply",
        "open_record_detail",
        "search_records",
        "open_records",
        "old_search",
    ]
    assert all(r.steps for r in panel.routines)
    for name in ("export_csv", "search_records", "old_search"):
        assert f">{name}</span>" in full_page


def test_a_skill_whose_recording_is_gone_still_shows_its_steps(full_page: str) -> None:
    panel = build_trajectory_panel(*_skills_and_dir(FULL))
    search = next(r for r in panel.routines if r.name == "search_records")
    assert search.matched == 0
    assert "no longer stored" in search.run_note
    assert [s.verb for s in search.steps] == ["click", "type", "press", "wait"]
    assert "is no longer stored" in full_page


def test_a_skill_whose_recording_is_on_disk_gets_its_real_screens(recorded_dir: Path) -> None:
    panel = collect(recorded_dir).trajectories
    routine = panel.routines[0]
    assert routine.run_id == "rec0000001"
    assert routine.run_steps == 3
    assert routine.matched == 3, "two clicks and one type, matched by verb in order"
    assert routine.run_note == ""
    shots = [s for s in routine.steps if s.shot]
    assert [s.verb for s in shots] == ["click", "type", "click"]
    assert all(s.shot.startswith("data:image/png;base64,") for s in shots)
    assert all(s.shot_caption.startswith("recorded: ") for s in shots)
    assert not any(s.shot for s in routine.steps if s.kind in {"look", "check"}), (
        "a lookup performed nothing, so no recorded screen belongs beside it"
    )


def test_the_recorded_screens_reach_the_page_embedded(recorded_dir: Path, tmp_path: Path) -> None:
    page = build_dashboard(recorded_dir, tmp_path / "out.html").read_text(encoding="utf-8")
    assert page.count('class="tshot" src="data:image/png;base64,') == 3
    assert "3 screen(s) matched" in page
    assert "http://" not in page and "https://" not in page


def test_an_empty_library_gets_the_panels_own_empty_state(tmp_path: Path) -> None:
    panel = build_trajectory_panel([], tmp_path / "trajectories", None)
    assert panel.routines == ()
    assert "Nothing to lay out yet" in panel.empty_reason
    page = build_dashboard(tmp_path, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "No procedures to read yet" in page


def test_a_trajectory_directory_that_cannot_be_read_costs_only_the_screens(
    recorded_dir: Path, tmp_path: Path
) -> None:
    """One unreadable store must not cost the reader every skill's steps."""
    skills, _ = _skills_and_dir(recorded_dir)[:2]
    broken = tmp_path / "not-a-directory"
    broken.write_text("", encoding="utf-8")
    panel = build_trajectory_panel(skills, broken, None)
    assert panel.routines and panel.routines[0].steps
    assert panel.routines[0].matched == 0
    assert panel.routines[0].run_note


def _skills_and_dir(data_dir: Path) -> tuple[list, Path, None]:
    from skillweaver.dashboard.build import read_skills

    skills, _ = read_skills(data_dir / "skills")
    return skills, data_dir / "trajectories", None
