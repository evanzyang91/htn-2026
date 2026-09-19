"""Running the suite: the reset, the report, and the things that must not be smoothed.

Three claims are defended here, in descending order of how badly the project needs
them to be true.

**The application is reset before every run.** Nine of the fourteen shipped tasks
mutate state, so a run that starts on the previous run's leftovers measures a
different and easier task. That failure is completely silent - the numbers still come
out, they are just wrong - so ``test_the_app_is_reset_before_every_single_run``
asserts the exact interleaving of resets and runs, not merely a count.

**The report is one the dashboard can read.** Asserted by parsing it with
``skillweaver.dashboard.build``'s own reader and building the real page from it, not
by checking that a file exists or by re-reading the harness's own idea of the format.

**Ground truth beats the agent's opinion.** The referee decides success; the agent's
own critic is recorded beside it and ignored when they disagree.

Nothing here touches a network, a browser or a real model. ``FakeApp`` is a dict with
a log, and one test drives the genuine :class:`~skillweaver.orchestrator.Agent` over
``tests/fakes`` so the harness is known to work against the real thing and not just
against a shape that resembles it.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pytest

from skillweaver.contracts import Budget, Usage
from skillweaver.dashboard.build import (
    build_dashboard,
    build_speedup_panel,
    read_eval_reports,
)
from skillweaver.errors import ConfigError, SkillWeaverError
from skillweaver.eval.harness import (
    BrowserReferee,
    Check,
    EvalTask,
    HttpReferee,
    LiveReferee,
    Referee,
    Suite,
    build_referee,
    load_suite,
    run,
    run_suite,
    run_task,
)
from skillweaver.orchestrator import ResetRefused, world_reset_from_url

SEED: dict[str, Any] = {"screen": "mail", "archived": [], "saved": 0}

COLD_MS = 0.020
"""Seconds a cold run "spends". Real sleeps, so the harness's clock is really being
measured rather than a number the fake handed it."""

WARM_MS = 0.004


# --------------------------------------------------------------------------------------
# The world, the referee that watches it, and the agent that acts on it
# --------------------------------------------------------------------------------------


START_URL = "https://site.test/start"


@dataclass(slots=True)
class FakeDom:
    """One screen, as the DOM reader script reports it. See :class:`FakePage`."""

    url: str
    elements: list[dict[str, Any]] = field(default_factory=list)


def an_element(kind: str, text: str, y: int = 0) -> dict[str, Any]:
    """One entry of the reader script's payload."""
    return {"kind": kind, "text": text, "x": 0, "y": y, "w": 200, "h": 24}


def a_page(url: str, headings: Sequence[str] = (), links: Sequence[str] = ()) -> FakeDom:
    return FakeDom(
        url=url,
        elements=[an_element("text", t, i * 30) for i, t in enumerate(headings)]
        + [an_element("link", t, 400 + i * 30) for i, t in enumerate(links)],
    )


class FakePage:
    """The one Playwright object :class:`BrowserGroundTruth` touches: a page.

    It answers ``evaluate`` with the payload the real reader script returns, so the
    GENUINE ``BrowserGroundTruth`` runs in these tests - the parsing, the clipping and
    the kind mapping are the shipped ones - without a browser or a socket anywhere.
    """

    def __init__(self, dom: FakeDom) -> None:
        self._dom = dom

    @property
    def url(self) -> str:
        return self._dom.url

    def evaluate(self, script: str) -> dict[str, Any]:
        return {"viewport": [1280, 800], "elements": list(self._dom.elements)}


class FakeBrowserControl:
    """Stands in for ``BrowserController`` where ground truth reaches into it.

    Reads :attr:`FakeApp.page` every time, so a referee shown this controller after a
    run sees the screen the run ENDED on rather than the one it started from.
    """

    def __init__(self, app: FakeApp) -> None:
        self.app = app

    def _live_page(self) -> FakePage:
        return FakePage(self.app.page)


@dataclass(slots=True)
class FakeApp:
    """The application under evaluation: some state, and a log of what happened to it.

    ``events`` is the point of this class. It records ``"reset"`` and ``"run:<task>"``
    in order, so a test can assert that every run was preceded by a reset rather than
    counting resets and hoping they landed in the right places.

    ``page`` is the same world seen the other way: what a referee that can only read a
    live screen would find. Only the live-referee tests look at it.
    """

    state: dict[str, Any] = field(default_factory=lambda: deepcopy(SEED))
    page: FakeDom = field(default_factory=lambda: a_page(START_URL, ["Start"]))
    events: list[str] = field(default_factory=list)
    reset_fails_on: set[int] = field(default_factory=set)
    resets: int = 0

    def reset(self) -> None:
        self.resets += 1
        if self.resets in self.reset_fails_on:
            raise SkillWeaverError("the reset endpoint refused")
        self.state = deepcopy(SEED)
        self.page = a_page(START_URL, ["Start"])
        self.events.append("reset")


class FakeReferee:
    """A :class:`Referee` over :class:`FakeApp`. The harness's only door to truth."""

    def __init__(self, app: FakeApp) -> None:
        self.app = app

    def reset(self) -> None:
        self.app.reset()

    def state(self) -> dict[str, Any]:
        return self.app.state


@dataclass(slots=True)
class Behaviour:
    """What the fake agent does on one attempt of one task."""

    mutate: Callable[[dict[str, Any]], None] | None = None
    seconds: float = WARM_MS
    claims_ok: bool = True
    llm_calls: int = 0
    usd: float = 0.0
    skill_used: str | None = "a_stored_skill"
    decision: str = "warm"
    learned: str | None = None
    raises: str = ""


def solve(app_key: str = "screen", value: Any = "records") -> Callable[[dict[str, Any]], None]:
    """A mutation that makes the task's check pass."""

    def mutate(state: dict[str, Any]) -> None:
        state[app_key] = value

    return mutate


COLD = Behaviour(
    mutate=solve(),
    seconds=COLD_MS,
    llm_calls=9,
    usd=0.27,
    skill_used=None,
    decision="cold",
    learned="a_stored_skill",
)
WARM = Behaviour(mutate=solve(), seconds=WARM_MS, llm_calls=0, decision="warm")


@dataclass(slots=True)
class FakeReport:
    """The slice of :class:`~skillweaver.orchestrator.RunReport` the harness reads."""

    ok: bool
    steps: int = 3
    llm_calls: int = 0
    decision: str = "warm"
    rescued: bool = False
    run_id: str = "run-1"
    learned: Any = None
    attempts: tuple[Any, ...] = ()
    outcome: Any = None


@dataclass(slots=True)
class Named:
    """Anything the harness reads a ``name`` or a ``usd`` or a ``skill_used`` off."""

    name: str = ""
    usd: float = 0.0
    skill_used: str | None = None


@dataclass(slots=True)
class FakeAgent:
    app: FakeApp
    behaviour: Behaviour
    task_id: str
    has_controller: bool = True

    @property
    def controller(self) -> FakeBrowserControl:
        """The open world, as :attr:`skillweaver.orchestrator.Agent.controller` is.

        A session that opened no browser has none, and a referee that needed to look
        at one has to say so rather than guess; ``has_controller=False`` is that case.
        """
        if not self.has_controller:
            raise AttributeError("this session opened no controller")
        return FakeBrowserControl(self.app)

    def run(self, spec: Any, *, learn: bool = True, warm: bool = True, cold: bool = True) -> Any:
        self.app.events.append(f"run:{self.task_id}")
        if self.behaviour.raises:
            raise SkillWeaverError(self.behaviour.raises)
        time.sleep(self.behaviour.seconds)
        if self.behaviour.mutate is not None:
            self.behaviour.mutate(self.app.state)
        return FakeReport(
            ok=self.behaviour.claims_ok,
            llm_calls=self.behaviour.llm_calls,
            decision=self.behaviour.decision,
            attempts=(Named(usd=self.behaviour.usd),),
            outcome=Named(skill_used=self.behaviour.skill_used),
            learned=None if self.behaviour.learned is None else Named(name=self.behaviour.learned),
        )


@dataclass(slots=True)
class FakeStore:
    """Just enough skill library for the harness's library-growth probe."""

    skills: list[str] = field(default_factory=list)

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[str]:
        return list(self.skills)


@dataclass(slots=True)
class FakeWorkbench:
    """A workbench whose session yields a scripted agent instead of opening a browser."""

    app: FakeApp
    script: dict[tuple[str, int], Behaviour]
    store: FakeStore = field(default_factory=FakeStore)
    data_dir: Path = Path("data")
    attempts: dict[str, int] = field(default_factory=dict)
    seen_specs: list[Any] = field(default_factory=list)
    seen_budgets: list[Any] = field(default_factory=list)
    opens_a_controller: bool = True

    @contextlib.contextmanager
    def session(self, spec: Any, budget: Any) -> Iterator[FakeAgent]:
        self.seen_specs.append(spec)
        self.seen_budgets.append(budget)
        task_id = str(getattr(spec, "text", ""))
        n = self.attempts.get(task_id, 0) + 1
        self.attempts[task_id] = n
        behaviour = self.script.get((task_id, n), WARM if n > 1 else COLD)
        if behaviour.learned:
            self.store.skills.append(behaviour.learned)
        yield FakeAgent(self.app, behaviour, task_id, self.opens_a_controller)


def a_task(task_id: str = "open_records", text: str | None = None) -> EvalTask:
    return EvalTask(
        id=task_id,
        text=text or task_id,
        tags=("single-step",),
        expect=(Check("screen", "equals", "records"),),
    )


def a_suite(*tasks: EvalTask, warm_runs: int = 3) -> Suite:
    return Suite(
        tasks=tasks or (a_task(),),
        name="fake-suite",
        domain="fake.test",
        base_url="http://127.0.0.1:0",
        warm_runs=warm_runs,
        source="tests/eval/test_harness.py",
    )


# --------------------------------------------------------------------------------------
# The reset: the assertion the whole report rests on
# --------------------------------------------------------------------------------------


def test_the_app_is_reset_before_every_single_run() -> None:
    """Strict interleaving, not a count.

    An un-reset run silently measures the previous run's leftovers - "archive the
    Billing message" is free once the first run archived it - so four resets landing
    anywhere near four runs is not good enough. They must alternate.
    """
    app = FakeApp()
    bench = FakeWorkbench(app, script={})

    record = run_task(
        a_task(),
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(),
        warm_runs=3,
    )

    assert len(record.runs) == 4, "one cold run and three warm ones"
    assert app.events == [
        "reset",
        "run:open_records",
        "reset",
        "run:open_records",
        "reset",
        "run:open_records",
        "reset",
        "run:open_records",
    ]


def test_every_run_starts_from_identical_state_even_when_the_previous_one_mutated_it() -> None:
    """The cold run archives something; the warm runs must each start clean."""
    app = FakeApp()

    def archive(state: dict[str, Any]) -> None:
        state["archived"].append("m03")
        state["screen"] = "records"

    seen: list[int] = []

    def observe(state: dict[str, Any]) -> None:
        seen.append(len(state["archived"]))
        archive(state)

    behaviour = Behaviour(mutate=observe, seconds=WARM_MS)
    bench = FakeWorkbench(app, script={("open_records", n): behaviour for n in (1, 2, 3, 4)})

    run_task(a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=3)

    assert seen == [0, 0, 0, 0], "each run saw an empty archive: no leftovers carried over"


def test_a_reset_that_failed_is_recorded_rather_than_assumed() -> None:
    """A run on a dirty application is not a failed run - it is a run whose number
    must not be believed, and the two need different words in the report."""
    app = FakeApp(reset_fails_on={3})
    bench = FakeWorkbench(app, script={})

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=3
    )

    assert [r.reset_ok for r in record.runs] == [True, True, False, True]
    assert app.resets == 4, "the suite kept going rather than stopping at the bad reset"


def test_the_run_before_the_first_one_is_reset_too() -> None:
    """A suite whose first task inherits a hand-driven browser would otherwise start
    the cold run part-way through the site."""
    app = FakeApp()
    app.state["screen"] = "records"  # left over from something else
    bench = FakeWorkbench(app, script={("open_records", 1): Behaviour(mutate=None, seconds=0.0)})

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=0
    )

    assert record.runs[0].ok is False, "the stale 'records' screen was cleared before the run"


# --------------------------------------------------------------------------------------
# The ground-truth boundary
# --------------------------------------------------------------------------------------


def test_the_referee_is_never_handed_to_the_agent() -> None:
    """The agent may RESET the world; it may never READ the truth about it.

    The two halves of the referee are not alike and the boundary runs between them,
    not around the object:

    * **Resetting is the agent's business.** The admission gate has to put the world
      back to re-run a candidate skill, so ``reset_url`` is passed on the TaskSpec
      deliberately - see :data:`~skillweaver.orchestrator.WorldReset`. Withholding it
      is how a state-changing task ends up never being learned at all.
    * **Reading ground truth is not.** ``/__state``, the referee object and the task's
      ``expect`` checks stay on this side. Scoring against perfect knowledge is
      legitimate; letting the agent ACT on it would make every number here a
      measurement of nothing.
    """
    app = FakeApp()
    referee = FakeReferee(app)
    bench = FakeWorkbench(app, script={})

    run_task(a_task(), workbench=bench, referee=referee, suite=a_suite(), warm_runs=1)

    assert bench.seen_specs, "the session was opened at least once"
    for spec in bench.seen_specs:
        held = [getattr(spec, f.name) for f in fields(spec)] + list(spec.params.values())
        assert not any(isinstance(v, FakeReferee | HttpReferee) for v in held)
        assert "referee" not in spec.params
        # No door to the truth: the state endpoint is never named anywhere on the spec.
        assert not any("__state" in str(v) for v in held)
        # Nor the checks that decide the verdict.
        assert not any("expect" in str(k) for k in spec.params)
        assert not any("screen" in str(v) for v in spec.params.values())
        # The agent is given the task's words, and a way to undo what it does.
        assert spec.text == "open_records"
        assert spec.params["reset_url"].endswith("/__reset")


def test_the_agent_is_told_how_to_put_the_world_back() -> None:
    """Without this the library never grows, and the whole report measures nothing.

    Most of the shipped suite CHANGES something, and the admission gate can only
    admit a skill it has re-run from where the recording started. No reset, no
    re-run; no re-run, no stored skill; no stored skill, every "warm" run explores
    again and the suite reports a speedup of about 1.0 while looking healthy.
    """
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    suite = a_suite()

    run_task(a_task(), workbench=bench, referee=FakeReferee(app), suite=suite, warm_runs=1)

    for spec in bench.seen_specs:
        assert spec.params["reset_url"] == f"{suite.base_url}{suite.reset_path}"


def test_ground_truth_overrules_the_agents_own_claim_of_success() -> None:
    """A critic that reports success on a task that did not happen is a more valuable
    finding than any speedup, so it is recorded rather than trusted."""
    app = FakeApp()
    lying = Behaviour(mutate=None, seconds=0.0, claims_ok=True)
    bench = FakeWorkbench(app, script={("open_records", n): lying for n in (1, 2)})

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=1
    )

    assert all(r.agent_claimed for r in record.runs), "the agent said it worked"
    assert not any(r.ok for r in record.runs), "ground truth says otherwise, and it wins"
    assert not any(r.agreed for r in record.runs)
    assert record.metrics.cold_ok is False


def test_the_scores_per_check_detail_is_kept_so_a_failure_can_be_diagnosed() -> None:
    app = FakeApp()
    bench = FakeWorkbench(app, script={("open_records", 1): Behaviour(mutate=None, seconds=0.0)})

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=0
    )

    assert record.runs[0].score is not None
    assert "screen" in record.runs[0].score.reason
    assert "'mail'" in record.runs[0].score.reason


# --------------------------------------------------------------------------------------
# A run that goes wrong
# --------------------------------------------------------------------------------------


def test_a_run_that_raises_is_one_failed_measurement_not_a_lost_suite() -> None:
    app = FakeApp()
    bench = FakeWorkbench(
        app,
        script={("open_records", 2): Behaviour(raises="the browser went away", seconds=0.0)},
    )

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=2
    )

    assert len(record.runs) == 3, "the suite finished the remaining runs"
    assert record.runs[1].ok is False
    assert "the browser went away" in record.runs[1].error
    assert record.runs[2].ok is True, "and the run after the broken one was fine"


def test_a_warm_run_that_explored_is_recorded_as_warm_in_name_only() -> None:
    app = FakeApp()
    explored = Behaviour(mutate=solve(), seconds=WARM_MS, skill_used=None, decision="cold")
    bench = FakeWorkbench(app, script={("open_records", 2): explored})

    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=1
    )

    assert record.runs[1].skill_used is None
    assert record.runs[1].to_json()["used_library"] is False
    assert record.metrics.warm_explored == 1


def test_a_run_the_library_got_wrong_and_exploration_rescued_credits_no_skill() -> None:
    """Saying a skill carried the run here would report the library as working on
    exactly the runs that prove it was not."""
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    record = run_task(
        a_task(), workbench=bench, referee=FakeReferee(app), suite=a_suite(), warm_runs=1
    )
    assert record.runs[0].skill_used is None, "the cold run explored"


# --------------------------------------------------------------------------------------
# The report the dashboard reads
# --------------------------------------------------------------------------------------


def test_the_full_suite_writes_a_report_the_dashboards_own_reader_accepts(
    tmp_path: Path,
) -> None:
    """The hard coordination constraint: the metrics file has to load through
    ``skillweaver.dashboard.build``, not merely exist."""
    app = FakeApp()
    tasks = (a_task("open_records"), a_task("open_settings"), a_task("archive_billing"))
    bench = FakeWorkbench(app, script={})

    path = run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(*tasks),
        out_dir=tmp_path / "eval",
        warm_runs=3,
        stamp="20260919T120000Z",
    )
    assert path.name == "20260919T120000Z.json"

    reports, problems = read_eval_reports(tmp_path / "eval")
    assert problems == [], "the dashboard's reader found nothing it could not understand"
    assert len(reports) == 1

    report = reports[0]
    assert report.suite == "fake-suite"
    assert {t.task_id for t in report.tasks} == {"open_records", "open_settings", "archive_billing"}

    for task in report.tasks:
        assert task.cold is not None and task.cold.attempt == 1
        assert len(task.warm) == 3
        assert task.cold.ok and all(w.ok for w in task.warm)
        assert task.cold.wall_ms > 0 and all(w.wall_ms > 0 for w in task.warm)
        assert task.cold.llm_calls == 9
        assert all(w.llm_calls == 0 for w in task.warm), "the warm path consults no model"
        assert task.cold.skill_used is None, "the cold run explored"
        assert all(w.skill_used == "a_stored_skill" for w in task.warm)


def test_the_dashboards_speedup_panel_fills_in_from_that_report(tmp_path: Path) -> None:
    """Not just parseable: chartable. The panel's empty state is what a file the
    dashboard could read but not use would produce."""
    app = FakeApp()
    bench = FakeWorkbench(app, script={})

    run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(a_task("open_records"), a_task("open_settings")),
        out_dir=tmp_path / "eval",
        warm_runs=3,
        stamp="20260919T120000Z",
    )

    panel = build_speedup_panel(*read_eval_reports(tmp_path / "eval"))
    assert panel.empty_reason == ""
    assert len(panel.rows) == 2
    assert panel.time_factor is not None and panel.time_factor > 1.0
    assert panel.notes == ()


def test_the_real_dashboard_page_builds_from_the_report(tmp_path: Path) -> None:
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(a_task("open_records", text="Open the Records screen.")),
        out_dir=tmp_path / "eval",
        warm_runs=3,
        stamp="20260919T120000Z",
    )

    out = build_dashboard(tmp_path, tmp_path / "dashboard.html")
    page = out.read_text(encoding="utf-8")

    assert "Open the Records screen." in page
    assert "No evaluation results yet" not in page


def test_the_report_carries_the_suite_domain_and_the_warm_run_count(tmp_path: Path) -> None:
    """A report can never be ambiguous about how many warm runs stand behind it."""
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    path = run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(),
        out_dir=tmp_path,
        warm_runs=2,
        stamp="s",
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["warm_runs"] == 2
    assert data["tasks"][0]["domain"] == "fake.test"
    assert data["tasks"][0]["tags"] == ["single-step"]
    assert len(data["tasks"][0]["runs"]) == 3


def test_only_runs_the_tasks_it_was_asked_for(tmp_path: Path) -> None:
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    path = run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(a_task("one"), a_task("two"), a_task("three")),
        out_dir=tmp_path,
        warm_runs=1,
        stamp="s",
        only=["two"],
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [t["task_id"] for t in data["tasks"]] == ["two"]


# --------------------------------------------------------------------------------------
# The markdown summary: caveats above the table, never below it
# --------------------------------------------------------------------------------------


def summary_of(tmp_path: Path, **kwargs: Any) -> str:
    app: FakeApp = kwargs.pop("app")
    bench: FakeWorkbench = kwargs.pop("bench")
    run_suite(
        workbench=bench,
        referee=FakeReferee(app),
        suite=kwargs.pop("suite", a_suite()),
        out_dir=tmp_path,
        stamp="s",
        **kwargs,
    )
    return (tmp_path / "s.md").read_text(encoding="utf-8")


def test_the_summary_reports_the_headline_and_the_table(tmp_path: Path) -> None:
    app = FakeApp()
    text = summary_of(tmp_path, app=app, bench=FakeWorkbench(app, script={}), warm_runs=3)

    assert "# Evaluation: fake-suite" in text
    assert "## Headline" in text
    assert "Speedup (pooled)" in text
    assert "Success delta" in text
    assert "`open_records`" in text
    assert "1 cold run then 3 warm run(s) each" in text


def test_a_suite_where_nothing_entered_the_library_opens_with_that_warning(
    tmp_path: Path,
) -> None:
    """The failure mode firstmate warned about: synthesis stores nothing, every warm
    run quietly explores, and the speedup lands near 1.0 looking perfectly healthy."""
    app = FakeApp()
    explored = Behaviour(mutate=solve(), seconds=WARM_MS, skill_used=None, decision="cold")
    bench = FakeWorkbench(app, script={("open_records", n): explored for n in (2, 3, 4)})

    text = summary_of(tmp_path, app=app, bench=bench, warm_runs=3)

    assert "measured cold-versus-cold" in text
    headline = text.index("## Headline")
    assert text.index("cold-versus-cold") < headline, "the caveat is above the numbers"


def test_a_warm_failure_is_named_rather_than_averaged_away(tmp_path: Path) -> None:
    app = FakeApp()
    broken = Behaviour(mutate=None, seconds=0.0, claims_ok=False)
    bench = FakeWorkbench(app, script={("open_records", 2): broken})

    text = summary_of(tmp_path, app=app, bench=bench, warm_runs=2)

    assert "Tasks whose warm run failed" in text
    assert "`open_records`" in text


def test_a_warm_run_slower_than_its_cold_one_is_called_a_regression(tmp_path: Path) -> None:
    """If warm is not faster, the number says so. It is not smoothed."""
    app = FakeApp()
    bench = FakeWorkbench(
        app,
        script={
            ("open_records", 1): Behaviour(mutate=solve(), seconds=WARM_MS, skill_used=None),
            ("open_records", 2): Behaviour(mutate=solve(), seconds=COLD_MS * 2),
        },
    )

    text = summary_of(tmp_path, app=app, bench=bench, warm_runs=1)

    assert "SLOWER" in text
    assert "Tasks where warm was SLOWER than cold" in text


def test_a_disagreement_between_the_agent_and_ground_truth_is_surfaced(tmp_path: Path) -> None:
    app = FakeApp()
    lying = Behaviour(mutate=None, seconds=0.0, claims_ok=True)
    bench = FakeWorkbench(app, script={("open_records", n): lying for n in (1, 2)})

    text = summary_of(tmp_path, app=app, bench=bench, warm_runs=1)

    assert "disagreed with ground truth" in text


def test_a_failed_reset_is_surfaced_in_the_summary(tmp_path: Path) -> None:
    app = FakeApp(reset_fails_on={2})
    text = summary_of(tmp_path, app=app, bench=FakeWorkbench(app, script={}), warm_runs=1)
    assert "without a successful reset" in text


def test_a_clean_suite_says_so_plainly(tmp_path: Path) -> None:
    app = FakeApp()
    text = summary_of(tmp_path, app=app, bench=FakeWorkbench(app, script={}), warm_runs=2)
    assert "Every task succeeded cold and warm" in text


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------


def test_tokens_are_null_when_nobody_was_counting_rather_than_zero(tmp_path: Path) -> None:
    """``0`` would read as "this run used no tokens", which is a different and much
    stronger claim than "no meter was supplied"."""
    app = FakeApp()
    record = run_task(
        a_task(),
        workbench=FakeWorkbench(app, script={}),
        referee=FakeReferee(app),
        suite=a_suite(),
        warm_runs=1,
    )
    assert record.runs[0].input_tokens is None
    assert record.runs[0].to_json()["output_tokens"] is None


def test_a_meter_is_delta_counted_per_run() -> None:
    app = FakeApp()
    total = Usage()

    def meter() -> Usage:
        return total

    class Counting(FakeAgent):
        def run(self, spec: Any, **kwargs: Any) -> Any:
            nonlocal total
            total = total + Usage(input_tokens=100, output_tokens=20, calls=1, cost_usd=0.01)
            return super().run(spec, **kwargs)

    bench = FakeWorkbench(app, script={})
    original = FakeWorkbench.session

    @contextlib.contextmanager
    def session(self: FakeWorkbench, spec: Any, budget: Any) -> Iterator[FakeAgent]:
        with original(self, spec, budget) as agent:
            yield Counting(agent.app, agent.behaviour, agent.task_id)

    FakeWorkbench.session = session  # type: ignore[assignment]
    try:
        record = run_task(
            a_task(),
            workbench=bench,
            referee=FakeReferee(app),
            suite=a_suite(),
            warm_runs=2,
            meter=meter,
        )
    finally:
        FakeWorkbench.session = original  # type: ignore[assignment]

    assert [r.input_tokens for r in record.runs] == [100, 100, 100], "per run, not cumulative"
    assert [r.output_tokens for r in record.runs] == [20, 20, 20]


# --------------------------------------------------------------------------------------
# The HTTP referee, without a server
# --------------------------------------------------------------------------------------


def test_the_http_referee_satisfies_the_protocol() -> None:
    assert isinstance(HttpReferee("http://127.0.0.1:8765"), Referee)
    assert isinstance(FakeReferee(FakeApp()), Referee)


class FakeResponse:
    """What ``urlopen`` hands back: a context manager over some bytes."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def fake_http(
    monkeypatch: pytest.MonkeyPatch, body: str = "{}", *, raises: Exception | None = None
) -> list[str]:
    """Replace ``urlopen`` and collect the requests the referee would have sent.

    Every HTTP test here is offline. Provoking a real connection refusal - even to
    localhost - is a socket call inside ``make test``, and the whole reason this file
    exists in its current shape is that a test which quietly reached a real service
    went unnoticed until something else changed. Injecting the failure is also
    stricter: it pins the URL and the method rather than only the error text.
    """
    sent: list[str] = []

    def urlopen(target: Any, timeout: float | None = None) -> FakeResponse:
        # The reset goes through the orchestrator's own world_reset_from_url, which
        # passes a plain URL; the state read builds a Request. Record both the same way.
        method = target.get_method() if isinstance(target, urllib.request.Request) else "GET"
        url = target.full_url if isinstance(target, urllib.request.Request) else str(target)
        sent.append(f"{method} {url}")
        if raises is not None:
            raise raises
        return FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return sent


def test_the_referee_resets_through_the_apps_reset_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing call, pinned to the suite's own reset endpoint."""
    sent = fake_http(monkeypatch, '{"ok": true}')

    HttpReferee("http://127.0.0.1:8765").reset()

    assert sent == ["GET http://127.0.0.1:8765/__reset"]


def test_the_referee_resets_with_the_same_callable_the_admission_gate_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One mechanism, driven from both ends.

    The harness resets between runs and the admission gate resets before re-running a
    candidate. Those are the same job, so this referee delegates to
    ``orchestrator.world_reset_from_url`` rather than shipping a second implementation
    that could drift from it - and ``reset_url`` is handed to the agent so the gate
    calls the very same endpoint.
    """
    sent = fake_http(monkeypatch)
    referee = HttpReferee("http://127.0.0.1:8765")

    referee.reset()
    world_reset_from_url(referee.reset_url)()

    assert sent == ["GET http://127.0.0.1:8765/__reset"] * 2, "identical calls"


def test_a_suite_may_name_a_different_reset_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam is not sandbox-specific: any "restore the demo data" URL fits."""
    sent = fake_http(monkeypatch)

    HttpReferee("https://demo.example/app", reset_path="/admin/restore").reset()

    assert sent == ["GET https://demo.example/app/admin/restore"]


def test_the_referee_reads_ground_truth_from_the_state_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = fake_http(monkeypatch, '{"ui": {"screen": "records"}}')

    state = HttpReferee("http://127.0.0.1:8765/").state()

    assert state == {"ui": {"screen": "records"}}
    assert sent == ["GET http://127.0.0.1:8765/__state"]


def test_an_unreachable_sandbox_says_how_to_start_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message a stranger acts on, not a bare connection error."""
    fake_http(monkeypatch, raises=urllib.error.URLError("Connection refused"))
    referee = HttpReferee("http://127.0.0.1:8765")

    with pytest.raises(SkillWeaverError, match="serve.py"):
        referee.reset()
    with pytest.raises(SkillWeaverError, match="did not answer GET /__state"):
        referee.state()


def test_a_sandbox_answering_with_something_other_than_json_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http(monkeypatch, "<html>a proxy login page</html>")
    with pytest.raises(SkillWeaverError, match="did not return JSON"):
        HttpReferee("http://127.0.0.1:8765").state()


def test_a_sandbox_answering_with_a_json_list_is_not_mistaken_for_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http(monkeypatch, "[1, 2, 3]")
    with pytest.raises(SkillWeaverError, match="did not return an object"):
        HttpReferee("http://127.0.0.1:8765").state()


def test_a_budget_is_passed_through_to_every_session() -> None:
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    limits = Budget(max_steps=7, max_seconds=11.0)

    run_task(
        a_task(),
        workbench=bench,
        referee=FakeReferee(app),
        suite=a_suite(),
        budget=limits,
        warm_runs=1,
    )

    assert bench.seen_budgets == [limits, limits], "the same limits bound every run"


# --------------------------------------------------------------------------------------
# Which referee: the suite's decision, and the live-page one
# --------------------------------------------------------------------------------------
#
# The bug this section pins: `run()` built an HttpReferee whatever the suite said, so
# evaluating a public website asked en.wikipedia.org for /__state and the shipped
# command could not start. The live numbers this project quotes had to be produced by
# a rig somebody rebuilt by hand, which is the same as saying nobody else could check
# them.
#
# Nothing here opens a socket or a browser. `urlopen` is replaced, and the referee's
# DOM reading runs the GENUINE BrowserGroundTruth against a fake page object.


def a_live_suite(*tasks: EvalTask, warm_runs: int = 1) -> Suite:
    return Suite(
        tasks=tasks or (a_live_task(),),
        name="live-suite",
        domain="site.test",
        base_url="https://site.test",
        reset_path="/start",
        referee="dom",
        warm_runs=warm_runs,
        source="tests/eval/test_harness.py",
    )


def a_live_task(task_id: str = "open_records") -> EvalTask:
    return EvalTask(
        id=task_id,
        text=task_id,
        tags=("single-step",),
        expect=(
            Check("url", "contains", "/records"),
            Check("headings", "contains", "Records"),
        ),
    )


def land_on(app: FakeApp, url: str, headings: Sequence[str] = ()) -> Callable[[Any], None]:
    """A run that navigates. Takes the state dict it is handed and ignores it: this
    world's ground truth is the screen, not a server's mapping."""

    def mutate(_state: dict[str, Any]) -> None:
        app.page = a_page(url, headings)

    return mutate


def test_a_suite_says_which_referee_it_needs_and_http_is_the_default(tmp_path: Path) -> None:
    """The default is the better referee, and it is what every existing suite gets."""
    document = {
        "suite": "s",
        "base_url": "http://127.0.0.1:8765",
        "tasks": [
            {
                "id": "t",
                "text": "do it",
                "tags": ["single-step"],
                "expect": [{"path": "screen", "equals": "records"}],
            }
        ],
    }
    default = load_suite(_suite_file(tmp_path, document, "default.yaml"))
    live = load_suite(_suite_file(tmp_path, {**document, "referee": "dom"}, "live.yaml"))

    assert default.referee == "http"
    assert live.referee == "dom"
    assert isinstance(build_referee(default), HttpReferee)
    assert isinstance(build_referee(live), BrowserReferee)


def test_a_suite_naming_a_referee_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """Defaulting would be the dangerous answer: a suite that meant 'dom' and was
    given 'http' because of a typo asks a public website for /__state, fails every
    reset, and reports a table of zeroes as though the agent had done the work badly."""
    document = {
        "suite": "s",
        "referee": "playwright",
        "tasks": [
            {
                "id": "t",
                "text": "do it",
                "tags": ["single-step"],
                "expect": [{"path": "screen", "equals": "records"}],
            }
        ],
    }
    with pytest.raises(ConfigError, match="'referee' must be"):
        load_suite(_suite_file(tmp_path, document, "typo.yaml"))


def _suite_file(tmp_path: Path, document: Any, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")  # JSON is valid YAML
    return path


def test_the_browser_referee_is_a_referee_and_asks_to_be_shown_the_screen() -> None:
    live = BrowserReferee("https://site.test", reset_path="/start")

    assert isinstance(live, Referee)
    assert isinstance(live, LiveReferee), "it cannot be asked once the browser has gone"
    assert not isinstance(HttpReferee("http://127.0.0.1:8765"), LiveReferee), (
        "an application that reports its own state outlives the window; do not go and look"
    )


def test_the_browser_referee_reads_url_headings_and_links_off_the_page() -> None:
    """The three keys the live suite's checks are written against, and no others."""
    app = FakeApp()
    app.page = a_page("https://site.test/records?from=search", ["Records", "42 rows"], ["Home"])
    live = BrowserReferee("https://site.test", reset_path="/start")

    live.observe(FakeBrowserControl(app))

    assert live.state() == {
        "url": "https://site.test/records?from=search",
        "headings": ["Records", "42 rows"],
        "links": ["Home"],
    }


def test_the_browser_referee_reset_loads_the_page_the_suite_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = fake_http(monkeypatch)

    BrowserReferee("https://site.test", reset_path="/start").reset()

    assert sent == ["GET https://site.test/start"]


def test_a_site_that_refuses_to_be_reset_has_not_failed_to_be_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 from a real website means there is no reset hook there, which is already
    known: a suite judged from the DOM is read-only and nothing needed putting back.
    Calling it a failed reset would mark every run in the report as untrustworthy for
    doing exactly what it was designed to do."""
    fake_http(monkeypatch, raises=ResetRefused("403"))

    BrowserReferee("https://site.test", reset_path="/start").reset()  # does not raise


def test_a_site_that_cannot_be_reached_at_all_is_a_failed_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http(monkeypatch, raises=urllib.error.URLError("Connection refused"))

    with pytest.raises(SkillWeaverError, match="did not answer"):
        BrowserReferee("https://site.test", reset_path="/start").reset()


def test_the_live_referee_is_shown_the_screen_before_the_browser_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole mechanism, end to end offline.

    A referee over a live page has one moment to read ground truth: after the run and
    before the session tears the browser down. Score it afterwards, as an application
    that reports its own state is scored, and there is nothing left to read.
    """
    fake_http(monkeypatch)
    app = FakeApp()
    arrive = Behaviour(mutate=land_on(app, "https://site.test/records", ["Records"]))
    bench = FakeWorkbench(app, script={("open_records", n): arrive for n in (1, 2)})
    live = BrowserReferee("https://site.test", reset_path="/start")

    record = run_task(
        a_live_task(), workbench=bench, referee=live, suite=a_live_suite(), warm_runs=1
    )

    assert [r.ok for r in record.runs] == [True, True]
    assert all(r.reset_ok for r in record.runs)


def test_a_run_the_referee_could_not_see_is_not_scored_on_the_last_ones_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent-wrong-number case, refused.

    A referee that kept the previous run's screen would score a session that never
    opened as a success, because the screen it remembered was the one that passed.
    """
    fake_http(monkeypatch)
    app = FakeApp()
    app.page = a_page("https://site.test/records", ["Records"])
    bench = FakeWorkbench(app, script={}, opens_a_controller=False)
    live = BrowserReferee("https://site.test", reset_path="/start")
    live.observe(FakeBrowserControl(app))
    assert live.state()["url"].endswith("/records"), "a passing screen, remembered"

    record = run_task(
        a_live_task(), workbench=bench, referee=live, suite=a_live_suite(), warm_runs=0
    )

    assert record.runs[0].ok is False
    assert "never saw the page" in record.runs[0].error


def test_the_live_referee_is_never_handed_to_the_agent_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Showing the referee a controller is a ONE-WAY read, and this is the assertion
    that keeps it one: the controller travels to the referee, and the referee travels
    nowhere. A referee the agent could reach would make every number here a
    measurement of nothing - see the sandbox-referee test of the same name."""
    fake_http(monkeypatch)
    app = FakeApp()
    live = BrowserReferee("https://site.test", reset_path="/start")
    bench = FakeWorkbench(app, script={})

    run_task(a_live_task(), workbench=bench, referee=live, suite=a_live_suite(), warm_runs=1)

    assert bench.seen_specs
    for spec in bench.seen_specs:
        held = [getattr(spec, f.name) for f in fields(spec)] + list(spec.params.values())
        assert not any(isinstance(v, BrowserReferee | FakeReferee | HttpReferee) for v in held)
        assert "referee" not in spec.params
        assert not any("headings" in str(v) for v in held), "nor the shape of the truth"


def test_a_suite_over_a_public_site_never_asks_it_for_a_state_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE regression test: the shipped entry point, on a suite that names a website.

    `run()` used to build an HttpReferee whatever the suite said, so this exact call
    asked the site for /__state and the command failed before the first task. The
    report it now writes records which referee decided `ok`, because a reader of these
    numbers is entitled to know.
    """
    sent = fake_http(monkeypatch)
    app = FakeApp()
    arrive = Behaviour(mutate=land_on(app, "https://site.test/records", ["Records"]))
    bench = FakeWorkbench(app, script={("open_records", 1): arrive})
    document = {
        "suite": "live-suite",
        "domain": "site.test",
        "base_url": "https://site.test",
        "reset_path": "/start",
        "referee": "dom",
        "tasks": [
            {
                "id": "open_records",
                "text": "open_records",
                "tags": ["single-step"],
                "expect": [
                    {"path": "url", "contains": "/records"},
                    {"path": "headings", "contains": "Records"},
                ],
            }
        ],
    }

    written = run(
        workbench=bench,
        suite=_suite_file(tmp_path, document, "live.yaml"),
        out=tmp_path / "out",
        repeat=0,
    )

    report = json.loads(written.read_text(encoding="utf-8"))
    assert report["referee"] == "dom"
    assert not any("__state" in url for url in sent), sent
    assert sent == ["GET https://site.test/start"] * 2, "one startup check, one per run"
    assert report["tasks"][0]["runs"][0]["ok"] is True


def test_only_runs_the_named_tasks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """How a run against somebody else's website is bounded to what it needs to show."""
    fake_http(monkeypatch, '{"screen": "records"}')
    app = FakeApp()
    bench = FakeWorkbench(app, script={})
    document = {
        "suite": "s",
        "base_url": "http://127.0.0.1:8765",
        "tasks": [
            {
                "id": t,
                "text": t,
                "tags": ["single-step"],
                "expect": [{"path": "screen", "equals": "records"}],
            }
            for t in ("first", "second", "third")
        ],
    }

    written = run(
        workbench=bench,
        suite=_suite_file(tmp_path, document, "three.yaml"),
        out=tmp_path / "out",
        repeat=0,
        only=["second"],
    )

    report = json.loads(written.read_text(encoding="utf-8"))
    assert [t["task_id"] for t in report["tasks"]] == ["second"]
