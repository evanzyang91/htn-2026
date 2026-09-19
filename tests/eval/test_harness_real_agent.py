"""The harness driving the REAL agent, cold then warm, over the fake app.

Everywhere else in ``tests/eval`` the agent is a scripted double, which proves the
harness's bookkeeping but not that it can drive the thing it was built to measure.
Here the agent is assembled by :func:`~skillweaver.orchestrator.build_agent` - the one
place the real agent is wired, and the same function the shipped workbench calls - so
the planner, the explorer, both critics and the admission gate are the shipped ones.
Only the leaves are fakes: ``tests/fakes``' four-state invoicing app stands in for a
browser and a scripted :class:`~tests.fakes.llm.FakeLLM` stands in for the model.

What this file is really defending is that the harness measures a difference that is
actually there. The model's script holds exactly the calls one cold run costs; a ninth
call raises ``ScriptExhausted``. So the three warm runs are model-free by construction
rather than by a counter that could be wrong, and the speedup the harness reports comes
out of a warm path that genuinely skipped the model while still doing the work.

It also demonstrates the :class:`~skillweaver.eval.harness.Referee` seam away from
HTTP: the fake app resets through ``controller.reset()`` and its ground truth is the
controller's own state, which the agent reaches only through pixels.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from skillweaver.contracts import Budget, LLMResponse, TaskSpec, Trajectory, Usage
from skillweaver.dashboard.build import build_speedup_panel, read_eval_reports
from skillweaver.eval.harness import Check, EvalTask, Suite, run_suite, run_task
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.orchestrator import Agent, build_agent
from skillweaver.skills.retrieve import SkillRetriever
from skillweaver.skills.synthesize import EnvironmentFactory, ReplayEnvironment
from tests.fakes import (
    FakeLLM,
    InMemorySkillStore,
    InMemoryTrajectoryRecorder,
    InMemoryTrajectoryStore,
    Scenario,
    make_scenario,
)
from tests.fakes.scenario import DOMAIN, GOAL

TASK = "Confirm payment of the Acme Corp invoice."


# --------------------------------------------------------------------------------------
# What the scripted model says. Mirrors tests/test_cli.py, which owns the canonical copy.
# --------------------------------------------------------------------------------------


def _answer(**fields: Any) -> str:
    fields.setdefault("thought", "the next move")
    fields.setdefault("expect", "the screen advances to the next step")
    fields.setdefault("done", False)
    return json.dumps(fields)


def click(element_id: str, **extra: Any) -> str:
    return _answer(action={"kind": "click", "element_id": element_id}, **extra)


def type_text(text: str, **extra: Any) -> str:
    return _answer(action={"kind": "type_text", "text": text}, **extra)


YES = json.dumps(
    {
        "ok": True,
        "evidence": "the screen advanced as expected",
        "reason": "the move did what it said",
        "confidence": 0.9,
    }
)

SKILL_CODE = (
    "def run(ctx, company):\n"
    '    ctx.ctl.type_text("acme")\n'
    '    rows = ctx.see.find_text("Acme Corp", fuzzy=False)\n'
    '    ctx.expect(bool(rows), "the search returned no row for " + company)\n'
    "    ctx.ctl.click(rows[0])\n"
    '    buttons = ctx.see.find_text("Confirm payment", fuzzy=False)\n'
    '    ctx.expect(bool(buttons), "no Confirm payment button on the invoice")\n'
    "    ctx.ctl.click(buttons[0])\n"
    "    return True\n"
)


def skill_reply() -> str:
    """The synthesizer's reply: the JSON object its prompt asks for, in a fence."""
    draft = {
        "name": "confirm_invoice_payment",
        "summary": "Confirm payment of a company's invoice from the invoice list.",
        "docstring": (
            "Searches the invoice list for the company and confirms payment.\n\n"
            "Assumes: the invoice list is on screen with its search field focused.\n"
            "Ends on: the payment-confirmed page."
        ),
        "params": {"company": {"type": "string", "description": "Company to pay."}},
        "example_args": {"company": "Acme Corp"},
        "requires": [],
        "code": SKILL_CODE,
        "verifier_code": (
            'def verify(ctx, result):\n    return bool(ctx.see.find_text("Payment confirmed"))\n'
        ),
    }
    return "```json\n" + json.dumps(draft) + "\n```"


BIND = json.dumps(
    {
        "steps": [{"skill": "confirm_invoice_payment", "args": {"company": "Acme Corp"}}],
        "why": "the library already knows this errand; the company comes from the sentence",
    }
)
"""ONE model call: the composer reading "the Acme Corp invoice" out of the sentence and
binding it to the stored skill's ``company`` parameter.

This is the whole difference between the two warm shapes. Asked in plain English the
warm path pays this call; handed ``company="Acme Corp"`` as a parameter it pays
nothing. Both are real, and quoting the zero without naming its condition would be the
most flattering number in the report bought with a hidden footnote."""


LEARN_SCRIPT: tuple[str, ...] = (
    type_text("acme"),
    YES,
    click("row-1042"),
    YES,
    click("confirm", done=True),
    YES,
    YES,
    skill_reply(),
)
"""Everything ONE cold run costs: three moves, each judged, the final judgment of the
``done`` claim, and one call to write the skill. Eight calls and not a ninth - the
warm runs that follow have no budget at the model at all."""


# --------------------------------------------------------------------------------------
# A workbench over the fake app
# --------------------------------------------------------------------------------------


class FakeWorld:
    """One persistent installation, and a workbench that opens sessions onto it.

    The library, graph and recorded runs live here and survive across sessions exactly
    as a data directory would, which is what makes the cold run's skill available to
    the warm ones.
    """

    def __init__(self, scenario: Scenario, replies: Sequence[str]) -> None:
        self.scenario = scenario
        self.llm = FakeLLM(list(replies))
        self.store = InMemorySkillStore()
        self.graph = InMemorySiteGraph()
        self.trajectories = InMemoryTrajectoryStore()
        self.data_dir = Path("data")

    def _environment(self, trajectory: Trajectory) -> EnvironmentFactory:
        """The admission gate's world, put back where the recording started.

        The same "put the world back" capability the evaluation needs between runs -
        here reaching it through the fake app's own reset.
        """

        def factory() -> ReplayEnvironment:
            self.scenario.controller.reset()
            return ReplayEnvironment(self.scenario.controller, self.scenario.perceiver, self.graph)

        return factory

    @contextlib.contextmanager
    def session(self, task: TaskSpec, budget: Budget) -> Iterator[Agent]:
        yield build_agent(
            task,
            controller=self.scenario.controller,
            perceiver=self.scenario.perceiver,
            llm=self.llm,
            store=self.store,
            retriever=SkillRetriever(self.store),
            graph=self.graph,
            trajectories=self.trajectories,
            recorder=InMemoryTrajectoryRecorder(),
            environment=self._environment,
            budget=budget,
        )


class ControllerReferee:
    """Ground truth for the fake app: reset it, and read the state it is really in.

    An OFFLINE TEACHER in exactly the sense the project means. The agent sees this app
    only as rendered pixels through its perceiver; this object asks the controller
    which state it is in, which no part of the agent may do.
    """

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def reset(self) -> None:
        self._scenario.controller.reset()

    def state(self) -> dict[str, Any]:
        return {"app": {"screen": self._scenario.controller.state}}


EVAL_TASK = EvalTask(
    id="confirm_invoice_payment",
    text=TASK,
    tags=("multi-step",),
    expect=(Check("app.screen", "equals", GOAL),),
    params={"company": "Acme Corp"},
)


def a_suite() -> Suite:
    return Suite(
        tasks=(EVAL_TASK,),
        name="fake-app",
        domain=DOMAIN,
        base_url="",
        warm_runs=3,
        source="tests/eval/test_harness_real_agent.py",
    )


@pytest.fixture
def world() -> FakeWorld:
    """A fresh installation with exactly enough model for one cold run and three
    plain-English warm runs. A parameters-supplied run has no budget here at all, so
    it proving model-free is a fact about the script rather than about a counter."""
    return FakeWorld(make_scenario(), (*LEARN_SCRIPT, BIND, BIND, BIND))


# --------------------------------------------------------------------------------------
# THE HEADLINE, MEASURED
# --------------------------------------------------------------------------------------


def test_the_three_shapes_cold_then_warm_in_english_then_warm_with_parameters(
    world: FakeWorld,
) -> None:
    """WHAT THE PROJECT CLAIMS, WITH THE CONDITION ON THE ZERO MADE EXPLICIT.

    The cold run explores an app it has never seen, succeeds, and the admission gate
    stores the skill. Each later run retrieves that skill and does the same work - but
    what it COSTS depends on what it was given:

    * asked in plain English, one model call to bind ``company`` from the sentence;
    * handed the parameter, no model call whatsoever.

    Both numbers are asserted here rather than only the flattering one. The model's
    script holds exactly the calls the first two shapes need, so the parameters-
    supplied run has no budget at the model at all: a single call would raise
    ``ScriptExhausted`` rather than be quietly miscounted.
    """
    world.llm = FakeLLM([*LEARN_SCRIPT, BIND, BIND, BIND])
    record = run_task(
        EVAL_TASK,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=3,
        bound_runs=1,
    )

    assert [r.phase for r in record.runs] == ["cold", "warm", "warm", "warm", "warm"]
    assert [r.binding for r in record.runs] == ["plain", "plain", "plain", "plain", "bound"]
    assert all(r.error == "" for r in record.runs), [r.error for r in record.runs]
    assert all(r.ok for r in record.runs), "ground truth says every run reached the goal"

    cold, *warm = record.runs
    plain, bound = warm[:3], warm[3]

    assert cold.llm_calls == 7, "seven calls to explore the task and judge the result"
    assert cold.skill_used is None, "it got there by exploring"
    assert cold.learned == "confirm_invoice_payment"

    assert [r.llm_calls for r in plain] == [1, 1, 1], "one call to read the parameter"
    assert bound.llm_calls == 0, "THE CLAIM, and what it costs: the parameter, supplied"

    assert [r.skill_used for r in warm] == ["confirm_invoice_payment"] * 4
    assert world.llm.calls == len(LEARN_SCRIPT) + 3, "the bound run spent nothing"


def test_the_two_warm_shapes_are_reported_separately_not_averaged(world: FakeWorld) -> None:
    """A single "warm" number covering both shapes would be neither figure."""
    world.llm = FakeLLM([*LEARN_SCRIPT, BIND, BIND, BIND])
    record = run_task(
        EVAL_TASK,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=3,
        bound_runs=1,
    )

    assert record.metrics.warm_llm_calls == 1.0, "the plain-English shape stands alone"
    bound = record.bound_metrics
    assert bound is not None and bound.warm_llm_calls == 0.0
    assert bound.cold_ms == record.metrics.cold_ms, "both compare against the same cold run"


def test_a_task_with_no_parameters_has_no_second_shape(world: FakeWorld) -> None:
    """``None``, not zero: nothing was asked, rather than supplying them not helping."""
    world.llm = FakeLLM([*LEARN_SCRIPT, BIND])
    plain_task = dataclasses.replace(EVAL_TASK, params={})

    record = run_task(
        plain_task,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=1,
        bound_runs=1,
    )

    assert [r.binding for r in record.runs] == ["plain", "plain"]
    assert record.bound_metrics is None


PER_CALL = Usage(input_tokens=100, output_tokens=20, calls=1, cost_usd=0.01)


def metered(replies: Sequence[str]) -> list[LLMResponse]:
    """The same script, with a price on every reply, so a meter has something to read."""
    return [LLMResponse(text=r, usage=PER_CALL) for r in replies]


def test_the_cold_runs_own_accounting_leaves_out_what_learning_cost() -> None:
    """``RunReport.llm_calls`` counts the calls charged to the run's ATTEMPTS, and
    writing the skill afterwards is not one of them.

    So the eighth call - the synthesiser's - is real money that the run's own books do
    not show. That is why :func:`~skillweaver.eval.harness.run_task` accepts a meter:
    a caller that can reach the model client gets the true per-run cost, learning
    included, instead of the agent's self-report.
    """
    world = FakeWorld(make_scenario(), metered([*LEARN_SCRIPT, BIND]))
    record = run_task(
        EVAL_TASK,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=1,
        bound_runs=0,
        meter=world.llm.total_usage,
    )

    cold, warm = record.runs
    assert cold.llm_calls == 7, "what the run charged itself"

    # The meter counted all eight, because it reads the model client rather than the
    # run's self-report: 8 x 100 input tokens, not 7.
    assert cold.input_tokens == 800
    assert cold.output_tokens == 160
    assert warm.input_tokens == 100, "the one binding call, correctly attributed"


def test_the_referee_agrees_with_the_agent_on_every_run(world: FakeWorld) -> None:  # noqa: D401
    """An independent check of the same runs. The two verdicts are computed from
    entirely different evidence - the critic from pixels, the referee from the
    controller's true state - so agreement is worth something."""
    record = run_task(
        EVAL_TASK,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=3,
    )
    assert all(r.agreed for r in record.runs)


def test_the_measured_speedup_is_real_because_the_warm_runs_did_the_same_work(
    world: FakeWorld,
) -> None:
    """Guards the cheapest way to fake this result: a warm path that "succeeds" by
    doing nothing would also report zero model calls and a marvellous speedup."""
    controller = world.scenario.controller
    run_task(
        EVAL_TASK,
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        warm_runs=3,
        bound_runs=0,
    )
    # Three moves to do the errand. The cold run performs them twice - once while
    # exploring and once when the admission gate replays the candidate skill - and
    # each of the three warm runs performs them once: 6 + 9.
    assert len(controller.actions) == 15


def test_each_run_starts_from_the_apps_first_screen(world: FakeWorld) -> None:
    """The fake app has a terminal goal state and an inescapable dead end, so a second
    run that inherited the first one's screen would either be already finished or be
    unable to move at all. Both would be measured as a warm run."""
    referee = ControllerReferee(world.scenario)
    record = run_task(EVAL_TASK, workbench=world, referee=referee, suite=a_suite(), warm_runs=3)
    assert all(r.reset_ok for r in record.runs)
    assert all(r.ok for r in record.runs), (
        "every run reached the goal from the start screen, which only a reset provides"
    )


def test_the_whole_thing_writes_a_report_the_dashboard_charts(
    world: FakeWorld, tmp_path: Path
) -> None:
    """End to end: the real agent, measured by the harness, read back by the dashboard."""
    run_suite(
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        out_dir=tmp_path,
        warm_runs=3,
        stamp="20260919T120000Z",
    )

    reports, problems = read_eval_reports(tmp_path)
    assert problems == []
    task = reports[0].tasks[0]
    assert task.cold is not None and task.cold.llm_calls == 7
    assert [w.llm_calls for w in task.warm] == [1, 1, 1, 0], (
        "three plain-English warm runs at one binding call each, then the "
        "parameters-supplied run at none"
    )

    panel = build_speedup_panel(reports, problems)
    assert panel.empty_reason == ""
    assert panel.call_factor is not None and panel.call_factor > 1.0


def test_a_library_that_learned_nothing_is_reported_as_cold_versus_cold(
    tmp_path: Path,
) -> None:
    """Firstmate's live run hit exactly this: the cold run worked, synthesis stored
    nothing, and every later run explored again. Without this detection the suite
    would report a speedup near 1.0 and look like an ordinary null result.

    Here the model is given enough script to explore four times over but its
    synthesiser reply is malformed, so nothing is ever admitted to the library.
    """
    explore_only = (
        type_text("acme"),
        YES,
        click("row-1042"),
        YES,
        click("confirm", done=True),
        YES,
        YES,
    )
    world = FakeWorld(make_scenario(), (*explore_only, "not json at all") * 5)

    path = run_suite(
        workbench=world,
        referee=ControllerReferee(world.scenario),
        suite=a_suite(),
        out_dir=tmp_path,
        warm_runs=3,
        stamp="s",
    )

    assert world.store.list(domain=DOMAIN) == [], "nothing entered the library"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"]["library_was_used"] is False
    assert all(run["skill_used"] is None for run in data["tasks"][0]["runs"])

    text = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "measured cold-versus-cold" in text
