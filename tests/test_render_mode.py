"""Headed or headless: choosing it, and never comparing across it by accident.

Two claims are under test, and the second is the one that protects a user.

**The mode is a setting with a flag over it.** ``SKILLWEAVER_HEADLESS`` picks it and
``skillweaver --headless`` / ``--headed`` overrides that for one invocation, through the
one configuration mechanism this project has. HEADED stays the default, because the
demo depends on a browser somebody can watch, and a flag that was not written never
resets a configured mode to a default.

**A crossing is explained, not suffered.** A skill learned with a visible window
records a starting screen a headless run may not recognise - measured 0.126 and 0.421
similarity across modes on two real pages against a same-state cut of 0.26 - so the warm
path retrieves the right skill, loses the screen, and falls through to a model as though
the library had never learned anything. Here the mode is recorded beside the skill and
the warm attempt says which two modes it was caught between.

It EXPLAINS rather than refuses, and that is the measured half. The same comparison on
the sandbox ordering app scores 0.819, well above the cut: a small purpose-built page
survives the crossing where a real one does not. A pre-emptive refusal would therefore
break the sandbox to protect Wikipedia, so the comparison is always attempted and the
crossing is offered as the likely reason it failed.

Nothing here launches a browser except the two tests marked as doing so, which only
read back the mode the browser was asked for.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from skillweaver import cli
from skillweaver.agent.planner import PlanFailure
from skillweaver.cli import app
from skillweaver.config import DEFAULT_HEADLESS, Settings, load_settings
from skillweaver.contracts import Observation, Provenance, Skill, TaskSpec
from skillweaver.errors import ConfigError, SkillWeaverError
from skillweaver.orchestrator import Agent, RunOutcome
from skillweaver.render_mode import HEADED, HEADLESS, crossing, mode_name, mode_of
from skillweaver.skills.retrieve import SkillRetriever
from skillweaver.skills.store import RENDER_MODE_FILE, FileSkillStore
from tests.fakes.scenario import Scenario

_NOW = datetime(2026, 9, 19, tzinfo=UTC)
DOMAIN = "fake.test"
runner = CliRunner()


# --------------------------------------------------------------------------------------
# choosing the mode
# --------------------------------------------------------------------------------------


def test_the_default_is_headed_because_the_demo_is_watched() -> None:
    """Not an accident of a boolean's default: a visible browser is the product."""
    assert DEFAULT_HEADLESS is False
    assert load_settings(env={}, env_file=None).headless is False
    assert Settings().headless is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
        ("", False),
        ("   ", False),
    ],
)
def test_the_environment_spells_the_mode_any_of_the_usual_ways(raw: str, expected: bool) -> None:
    """A shell, a Makefile and a CI file each spell a boolean differently; a blank
    value is not a spelling of either, so it leaves the default alone."""
    assert load_settings(env={"SKILLWEAVER_HEADLESS": raw}, env_file=None).headless is expected


def test_a_mode_that_is_not_a_mode_stops_the_command_rather_than_being_guessed() -> None:
    with pytest.raises(ConfigError, match="SKILLWEAVER_HEADLESS"):
        load_settings(env={"SKILLWEAVER_HEADLESS": "maybe"}, env_file=None)


def test_the_env_file_can_set_it_and_the_environment_still_wins(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("SKILLWEAVER_HEADLESS=true\n")
    assert load_settings(env={}, env_file=dotenv).headless is True
    assert load_settings(env={"SKILLWEAVER_HEADLESS": "0"}, env_file=dotenv).headless is False


class _Spy:
    """Captures the settings the root callback resolved, instead of opening a world."""

    def __init__(self) -> None:
        self.config: Settings | None = None

    def __call__(self, config: Settings | None = None) -> Any:
        self.config = config
        raise SystemExit(0)  # nothing after this call is under test


def _resolved(monkeypatch: pytest.MonkeyPatch, *args: str) -> Settings:
    spy = _Spy()
    monkeypatch.setattr(cli, "build_workbench", spy)
    monkeypatch.delenv("SKILLWEAVER_HEADLESS", raising=False)
    runner.invoke(app, [*args, "skills", "ls"])
    assert spy.config is not None, "the root callback did not resolve any settings"
    return spy.config


def test_the_flag_overrides_the_environment_for_one_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _resolved(monkeypatch, "--headless").headless is True
    monkeypatch.setenv("SKILLWEAVER_HEADLESS", "1")
    spy = _Spy()
    monkeypatch.setattr(cli, "build_workbench", spy)
    runner.invoke(app, ["--headed", "skills", "ls"])
    assert spy.config is not None and spy.config.headless is False


def test_a_flag_that_was_not_written_leaves_a_configured_mode_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract at the top of ``cli``: no flag silently resets a configured value.

    A plain ``bool`` default would make every invocation that says nothing about the
    mode an invocation that says ``--headed``, which is how a CI file's
    ``SKILLWEAVER_HEADLESS=1`` would stop working without anyone editing it.
    """
    monkeypatch.setenv("SKILLWEAVER_HEADLESS", "yes")
    spy = _Spy()
    monkeypatch.setattr(cli, "build_workbench", spy)
    runner.invoke(app, ["skills", "ls"])
    assert spy.config is not None and spy.config.headless is True


def test_the_flag_is_written_before_the_subcommand_and_so_covers_all_three() -> None:
    """``learn``, ``run`` and ``eval run`` open their world through one call, so one
    global flag reaches all of them - and the help says where to write it."""
    top = runner.invoke(app, ["--help"])
    assert "--headless" in top.output and "--headed" in top.output
    for command in (["learn", "--help"], ["run", "--help"], ["eval", "run", "--help"]):
        assert runner.invoke(app, command).exit_code == 0


# --------------------------------------------------------------------------------------
# opening the world in the mode that was chosen
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("headless", [True, False])
def test_the_world_is_opened_in_the_configured_mode(
    monkeypatch: pytest.MonkeyPatch, headless: bool
) -> None:
    """``_open_world`` is the ONE place this project constructs a browser, which is why
    one setting is enough to reach ``learn``, ``run`` and ``eval run`` alike. It used to
    hardcode a visible window, against a ``BrowserController`` that defaults to the
    opposite - so anything building one directly silently got the other mode.
    """
    import skillweaver.controllers.browser as browser_module
    from skillweaver.orchestrator import _open_world

    asked: dict[str, Any] = {}

    class _Recorded:
        def __init__(self, **kwargs: Any) -> None:
            asked.update(kwargs)
            self.headless = kwargs["headless"]

    monkeypatch.setattr(browser_module, "BrowserController", _Recorded)
    config = load_settings(env={"SKILLWEAVER_HEADLESS": str(headless)}, env_file=None)
    controller, _perceiver = _open_world(config, TaskSpec(text="t", domain=DOMAIN))
    assert asked["headless"] is headless
    assert mode_of(controller) == mode_name(headless)


def test_a_real_browser_reports_the_mode_it_was_asked_for() -> None:
    """Headless only: a headed browser in a test suite steals focus, which is the whole
    reason the mode had to become selectable."""
    from skillweaver.controllers.browser import BrowserController

    with BrowserController(headless=True) as controller:
        assert controller.headless is True
        assert mode_of(controller) == HEADLESS
        assert "headed" not in controller.describe()


# --------------------------------------------------------------------------------------
# naming a mode
# --------------------------------------------------------------------------------------


class _Rendered:
    """Any object that says which renderer it is - what ``mode_of`` duck-types on."""

    def __init__(self, headless: bool) -> None:
        self.headless = headless


def test_a_mode_has_one_name_each_way_round() -> None:
    assert (mode_name(True), mode_name(False)) == (HEADLESS, HEADED)
    assert mode_of(_Rendered(headless=True)) == HEADLESS
    assert mode_of(_Rendered(headless=False)) == HEADED


def test_a_controller_with_no_opinion_is_not_given_one(scenario: Scenario) -> None:
    """A desktop's screen is always visible and has no second mode to be confused
    with, so the answer is "no claim" rather than "headed". Inventing one would put a
    wrong sentence in a report."""
    assert mode_of(object()) is None
    assert mode_of(scenario.controller) is None


def test_a_mode_that_is_not_a_bool_is_not_a_mode() -> None:
    thing = _Rendered(headless=True)
    thing.headless = "yes"  # type: ignore[assignment]
    assert mode_of(thing) is None


class TestCrossing:
    def test_two_of_the_same_mode_have_nothing_to_say(self) -> None:
        assert crossing(HEADED, HEADED) is None
        assert crossing(HEADLESS, HEADLESS) is None

    def test_an_unknown_mode_on_either_side_claims_nothing(self) -> None:
        assert crossing(None, HEADED) is None
        assert crossing(HEADED, None) is None
        assert crossing(None, None) is None

    def test_a_crossing_names_both_modes_and_the_flag_that_fixes_it(self) -> None:
        said = crossing(HEADED, HEADLESS)
        assert said is not None
        assert "recorded headed" in said and "headless" in said
        assert "--headed" in said, "the fix is to match what the library was recorded in"

    def test_it_is_offered_as_a_likely_reason_rather_than_a_verdict(self) -> None:
        """The gap is not a constant - 0.819 on the sandbox, 0.126 on Wikipedia - so a
        run that lost its screen for some other reason is not told a falsehood."""
        said = crossing(HEADED, HEADLESS)
        assert said is not None and "likely" in said
        assert "0.819" in said, "the page that survived a crossing is quoted too"
        assert "the library is intact" in said.lower()

    def test_the_other_direction_points_the_other_way(self) -> None:
        said = crossing(HEADLESS, HEADED)
        assert said is not None and "--headless" in said


# --------------------------------------------------------------------------------------
# recording the mode beside the skill
# --------------------------------------------------------------------------------------


def _skill(name: str = "search_wikipedia") -> Skill:
    return Skill(
        name=name,
        domain=DOMAIN,
        summary="Search Wikipedia and open the article.",
        docstring="Searches for `query` and opens the article.",
        params={},
        code="def run(ctx):\n    return True\n",
        requires=(),
        precondition=None,
        verifier_code=None,
        provenance=Provenance("run-taught", "Search Wikipedia", "m", _NOW),
    )


class TestRecordedMode:
    def test_a_store_stamps_what_it_writes_and_reads_it_back(self, tmp_path: Path) -> None:
        store = FileSkillStore(tmp_path, render_mode=HEADLESS)
        stored = store.put(_skill())
        assert store.recorded_render_mode(stored.name, DOMAIN) == HEADLESS
        assert store.recorded_render_modes(DOMAIN) == {stored.name: HEADLESS}
        assert (store.version_dir(stored.name, DOMAIN, 1) / RENDER_MODE_FILE).is_file()

    def test_a_store_that_never_opened_a_world_claims_no_mode(self, tmp_path: Path) -> None:
        """``skills ls`` builds a store and renders nothing; it must not assert a mode."""
        store = FileSkillStore(tmp_path)
        stored = store.put(_skill())
        assert store.render_mode is None
        assert store.recorded_render_mode(stored.name, DOMAIN) is None
        assert store.recorded_render_modes() == {}

    def test_it_survives_the_statistics_and_the_retirement(self, tmp_path: Path) -> None:
        """The reason it is not a key in ``meta.json``: that file is regenerated whole
        from a ``Skill``, which has no field for the mode, so a key beside it would
        vanish on the skill's first recorded run."""
        store = FileSkillStore(tmp_path, render_mode=HEADED)
        store.put(_skill())
        store.record_run("search_wikipedia", DOMAIN, ok=True, ms=120.0)
        assert store.recorded_render_mode("search_wikipedia", DOMAIN) == HEADED
        store.demote("search_wikipedia", DOMAIN, "it ran and did not work")
        assert store.recorded_render_mode("search_wikipedia", DOMAIN) == HEADED

    def test_each_version_keeps_its_own_answer(self, tmp_path: Path) -> None:
        """A library relearned in the other mode does not retroactively rewrite what
        the old version was recorded in."""
        FileSkillStore(tmp_path, render_mode=HEADED).put(_skill())
        later = FileSkillStore(tmp_path, render_mode=HEADLESS)
        later.put(_skill())
        assert later.recorded_render_mode("search_wikipedia", DOMAIN, 1) == HEADED
        assert later.recorded_render_mode("search_wikipedia", DOMAIN, 2) == HEADLESS
        assert later.recorded_render_mode("search_wikipedia", DOMAIN) == HEADLESS

    def test_an_unknown_skill_is_unknown_rather_than_an_error(self, tmp_path: Path) -> None:
        """This is asked while deciding whether to bother comparing, not to read the
        library, so it answers instead of raising."""
        store = FileSkillStore(tmp_path, render_mode=HEADED)
        assert store.recorded_render_mode("nothing_like_this", DOMAIN) is None
        assert store.recorded_render_mode("nothing_like_this", DOMAIN, 3) is None

    def test_a_file_this_build_cannot_read_is_unknown_not_wrong(self, tmp_path: Path) -> None:
        store = FileSkillStore(tmp_path, render_mode=HEADED)
        store.put(_skill())
        mode_file = store.version_dir("search_wikipedia", DOMAIN, 1) / RENDER_MODE_FILE
        mode_file.write_text("goggles\n")
        assert store.recorded_render_mode("search_wikipedia", DOMAIN) is None

    def test_a_misspelled_mode_is_refused_before_it_is_written_to_anything(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SkillWeaverError, match="render_mode"):
            FileSkillStore(tmp_path, render_mode="headfull")


# --------------------------------------------------------------------------------------
# the warm attempt refuses a comparison it cannot win, and says why
# --------------------------------------------------------------------------------------


class _Lost:
    """A planner that declines at one named stage, the way the real one does."""

    def __init__(self, stage: str = "no_route", reason: str = "no known route") -> None:
        self.last_failure = PlanFailure(stage=stage, reason=reason, skill="search_wikipedia")  # type: ignore[arg-type]

    def plan(self, task: TaskSpec, observation: Observation) -> None:  # pragma: no cover
        raise AssertionError("the agent plans through attempt()")

    def attempt(self, task: TaskSpec, observation: Observation) -> RunOutcome | None:
        return None


class _Explorer:
    """Reports a failed exploration, so the run ends without a model."""

    def explore(self, task: TaskSpec, **_: Any) -> Any:  # pragma: no cover
        raise AssertionError("this test never reaches the cold path")


class _Store:
    """A library of ``{name: mode}``, where a name mapped to ``None`` never said."""

    def __init__(self, **modes: str | None) -> None:
        self._modes = modes

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[Skill]:
        return [_skill(name) for name in self._modes] if domain in (None, DOMAIN) else []

    def recorded_render_modes(self, domain: str | None = None) -> dict[str, str]:
        return {n: m for n, m in self._modes.items() if m is not None}


def _agent(scenario: Scenario, *, mode: bool | None, store: Any, planner: Any = None) -> Agent:
    controller = scenario.controller
    if mode is not None:
        controller.headless = mode  # type: ignore[attr-defined]
    return Agent(
        controller=controller,
        perceiver=scenario.perceiver,
        store=store,
        retriever=SkillRetriever(store),
        planner=planner if planner is not None else _Lost(),  # type: ignore[arg-type]
        explorer=_Explorer(),  # type: ignore[arg-type]
    )


def _warm(scenario: Scenario, *, mode: bool | None, store: Any, planner: Any = None) -> Any:
    report = _agent(scenario, mode=mode, store=store, planner=planner).run(
        TaskSpec(text="Search Wikipedia", domain=DOMAIN), learn=False, cold=False
    )
    return report.warm


class TestTheWarmAttemptSaysSo:
    def test_a_library_recorded_in_the_other_mode_explains_a_lost_screen(
        self, scenario: Scenario
    ) -> None:
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=HEADED))
        assert warm is not None and not warm.ok
        assert warm.cross_mode is not None
        assert "recorded headed" in warm.cross_mode and "headless" in warm.cross_mode

    def test_the_stage_still_says_where_it_actually_stopped(self, scenario: Scenario) -> None:
        """The mode is why, not where. A stage rewritten to name the mode would lose the
        one fact a reader needs to tell this apart from any other routing failure."""
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=HEADED))
        assert warm is not None and warm.stage == "no_route"
        assert "no known route" in warm.reason
        assert str(warm).endswith("]"), "the line carries the crossing in brackets"

    def test_the_comparison_is_attempted_rather_than_refused(self, scenario: Scenario) -> None:
        """The measured half: the crossing is survivable on a small clean page (0.819 on
        the sandbox app), so refusing it up front would break what works today."""
        before = scenario.controller.captures
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=HEADED))
        assert warm is not None and warm.cross_mode is not None
        assert scenario.controller.captures > before, "the screen was actually read"

    def test_a_warm_attempt_that_succeeded_is_not_second_guessed(self, scenario: Scenario) -> None:
        """A cross-mode run whose screens matched anyway is simply a success."""
        warm = _warm(
            scenario,
            mode=True,
            store=_Store(search_wikipedia=HEADED),
            planner=_Lost(stage="skill_failed", reason="the skill raised"),
        )
        assert warm is not None and warm.cross_mode is None

    @pytest.mark.parametrize(
        "stage", ["no_candidates", "unbindable_args", "unaccounted", "vanished", "route_failed"]
    )
    def test_a_failure_the_mode_cannot_explain_is_left_alone(
        self, scenario: Scenario, stage: str
    ) -> None:
        """Words, a missing argument, a vanished skill and a broken controller are not
        pixels. Offering the mode for those would be a confident falsehood in the one
        place a user is already confused."""
        warm = _warm(
            scenario,
            mode=True,
            store=_Store(search_wikipedia=HEADED),
            planner=_Lost(stage=stage, reason="something else"),
        )
        assert warm is not None and warm.cross_mode is None

    def test_one_matching_skill_is_enough_to_say_nothing(self, scenario: Scenario) -> None:
        """A library that is not uniformly unreadable has a different problem, and
        blaming the mode would send the reader after the wrong thing."""
        warm = _warm(
            scenario, mode=True, store=_Store(search_wikipedia=HEADED, open_article=HEADLESS)
        )
        assert warm is not None and warm.cross_mode is None

    def test_a_skill_that_never_said_is_not_evidence_against_itself(
        self, scenario: Scenario
    ) -> None:
        """A library stored before the mode was recorded claims nothing either way."""
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=None))
        assert warm is not None and warm.cross_mode is None

    def test_the_same_mode_has_nothing_to_explain(self, scenario: Scenario) -> None:
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=HEADLESS))
        assert warm is not None and warm.cross_mode is None

    def test_a_controller_that_claims_no_mode_accuses_nothing(self, scenario: Scenario) -> None:
        warm = _warm(scenario, mode=None, store=_Store(search_wikipedia=HEADED))
        assert warm is not None and warm.cross_mode is None

    def test_a_store_that_does_not_keep_the_answer_is_not_second_guessed(
        self, scenario: Scenario
    ) -> None:
        class _Older:
            def list(self, domain: str | None = None, *, include_demoted: bool = False) -> Any:
                return [_skill()]

        warm = _warm(scenario, mode=True, store=_Older())
        assert warm is not None and warm.cross_mode is None

    def test_an_empty_library_is_still_reported_as_empty(self, scenario: Scenario) -> None:
        warm = _warm(scenario, mode=True, store=_Store())
        assert warm is not None and warm.stage == "empty_library" and warm.cross_mode is None

    def test_nothing_is_demoted_for_a_crossing(self, scenario: Scenario) -> None:
        """Nothing had to be added to guarantee this: a crossing is lost at ``no_route``,
        which is reached while planning, and only a skill that RAN and failed is ever
        retired. Asserted because a cross-mode run that retired a working library would
        be far worse than one that quietly missed."""
        warm = _warm(scenario, mode=True, store=_Store(search_wikipedia=HEADED))
        assert warm is not None and warm.demoted is None
        assert warm.performed_nothing, "the screen is untouched, so the cold path may have it"


# --------------------------------------------------------------------------------------
# what a person sees
# --------------------------------------------------------------------------------------


def _workbench(store: Any, settings: Settings) -> Any:
    from skillweaver.graph.model import InMemorySiteGraph
    from skillweaver.orchestrator import Workbench
    from skillweaver.trajectory.store import TrajectoryFileStore

    return Workbench(
        settings=settings,
        store=store,
        retriever=SkillRetriever(store),
        graph=InMemorySiteGraph(),
        trajectories=TrajectoryFileStore(settings.trajectories_dir),
        session=lambda task, budget: None,
    )


def test_a_skill_says_which_mode_it_was_recorded_in(tmp_path: Path) -> None:
    """ "A skill that says 'I was recorded headed' is far better than one that quietly
    never matches" - printed against the screen it qualifies, because that identity is
    only comparable to a screen the same renderer drew."""
    config = load_settings(env={"SKILLWEAVER_DATA_DIR": str(tmp_path)}, env_file=None)
    store = FileSkillStore(config.skills_dir, render_mode=HEADLESS)
    store.put(_skill())
    result = runner.invoke(
        app, ["skills", "show", "search_wikipedia"], obj=_workbench(store, config)
    )
    assert result.exit_code == 0, result.output
    assert "recorded headless" in result.output
    assert "starts on" in result.output


def test_a_skill_stored_before_the_mode_was_recorded_says_nothing(tmp_path: Path) -> None:
    config = load_settings(env={"SKILLWEAVER_DATA_DIR": str(tmp_path)}, env_file=None)
    store = FileSkillStore(config.skills_dir)
    store.put(_skill())
    result = runner.invoke(
        app, ["skills", "show", "search_wikipedia"], obj=_workbench(store, config)
    )
    assert result.exit_code == 0, result.output
    assert "recorded" not in result.output.split("parameters")[0].split("starts on")[1]


def test_the_command_line_hints_that_the_library_is_intact(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    """A warm attempt that did nothing is the moment to say that one word on the
    command line fixes it - not that the task must be learned again."""
    report = _agent(scenario, mode=True, store=_Store(search_wikipedia=HEADED)).run(
        TaskSpec(text="Search Wikipedia", domain=DOMAIN), learn=False, cold=False
    )
    assert report.warm is not None and report.warm.cross_mode is not None
    cli._hint_at_cross_mode(report)
    said = capsys.readouterr().err
    assert said.startswith("\nhint:"), "a reader looks for hints, not for a dense attempt line"
    assert "the library is intact" in said.lower()
    assert "--headed" in said


def test_a_run_that_did_not_cross_modes_is_not_hinted_at(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _agent(scenario, mode=True, store=_Store()).run(
        TaskSpec(text="Search Wikipedia", domain=DOMAIN), learn=False, cold=False
    )
    assert report.warm is not None and report.warm.cross_mode is None
    cli._hint_at_cross_mode(report)
    assert capsys.readouterr().err == ""


def test_the_report_a_person_reads_carries_the_sentence(scenario: Scenario) -> None:
    report = _agent(scenario, mode=True, store=_Store(search_wikipedia=HEADED)).run(
        TaskSpec(text="Search Wikipedia", domain=DOMAIN), learn=False, cold=False
    )
    explained = report.explain()
    assert "no_route" in explained, "where it stopped"
    assert "recorded headed" in explained, "and why it most likely stopped there"
    assert report.decision == "none" and not report.ok
