"""The command line and the cold-versus-warm decision underneath it.

What is being proved here, in the order the file is written.

**The headline.** ``test_learn_then_run_is_cold_then_warm_with_zero_model_calls`` runs
the real ``learn`` command and then the real ``run`` command on the same task, through
the real Typer app, and asserts that the second one consults the model **exactly zero
times**. Not "fewer", not "cheaper": the ``FakeLLM`` handed to the warm run has an
exhausted script, so a single call raises ``ScriptExhausted`` and fails the test loudly
rather than quietly inflating a counter.

**The fall-through is never hidden.** A stored skill that runs cleanly and does not
finish the errand is rejected by the critic, the run falls through to exploration, and
the report says so - ``decision == "cold"`` with the failed warm attempt still in
``attempts`` and ``rescued`` set. A skill library that reports "fine" when it was wrong
is how this kind of system rots, so that is asserted rather than assumed.

**Every subcommand actually runs.** Each one is invoked end to end against the fakes in
``tests/fakes``, and ``--help`` for every command and subcommand is asserted non-empty.
Nothing in this file touches a network, a browser or a real model: the workbench is
injected through the Typer context, so the commands under test are the shipped ones and
only their leaves are doubles.
"""

from __future__ import annotations

import contextlib
import json
import xml.dom.minidom
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from skillweaver.cli import app, main
from skillweaver.config import Settings, load_settings
from skillweaver.contracts import Provenance, Skill, TaskSpec, Trajectory
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.orchestrator import (
    Agent,
    AttemptRecord,
    RunReport,
    Workbench,
    build_agent,
    recall_end_state,
    task_spec,
)
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
from tests.fakes.scenario import DOMAIN

TASK = "Confirm payment of the Acme Corp invoice."
COMPANY = "company=Acme Corp"

runner = CliRunner()


# --------------------------------------------------------------------------------------
# What the scripted model says
# --------------------------------------------------------------------------------------


def _answer(**fields: Any) -> str:
    """One acting reply, in the shape the explorer's prompt asks for."""
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
"""A critic escalation answering yes. The cold path's critic has no evidence check to
run on a screen that merely changed, so it asks the model - once per move, plus once
more for the final 'is the task done?'. Those calls are what make exploration the
expensive path, and they are in the script rather than hidden behind a fake critic."""

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

HALF_DONE_CODE = (
    "def run(ctx, company):\n"
    '    ctx.ctl.type_text("acme")\n'
    '    rows = ctx.see.find_text("Acme Corp", fuzzy=False)\n'
    '    ctx.expect(bool(rows), "the search returned no row for " + company)\n'
    "    return True\n"
)
"""A skill that runs cleanly and leaves the errand unfinished: it filters the list and
stops. Its own ``ctx.expect`` passes, so nothing about it looks broken from inside -
only the critic, comparing the end screen with where the task is known to finish,
can tell. This is the shape of a skill that rots when a site changes."""


def skill_reply(code: str = SKILL_CODE) -> str:
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
        "code": code,
        "verifier_code": (
            'def verify(ctx, result):\n    return bool(ctx.see.find_text("Payment confirmed"))\n'
        ),
    }
    return "```json\n" + json.dumps(draft) + "\n```"


SOLVE_FROM_LIST = (
    type_text("acme"),
    YES,
    click("row-1042"),
    YES,
    click("confirm", done=True),
    YES,
    YES,
)
"""Three moves from the invoice list to the confirmation page, each judged by the
model, plus the final judgment of the ``done`` claim: seven calls."""

SOLVE_FROM_SEARCHED = (
    click("row-1042"),
    YES,
    click("confirm", done=True),
    YES,
    YES,
)
"""The same errand finished from the already-filtered list - where a half-done warm
attempt leaves the screen: five calls."""

LEARN_SCRIPT = (*SOLVE_FROM_LIST, skill_reply())
"""Everything one ``learn`` of this task costs: the run, then one call to write the
skill. Eight calls, and the admission gate re-runs the result without another."""


# --------------------------------------------------------------------------------------
# The world one invocation runs against
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class World:
    """The fakes behind the command line, and a way to invoke it against them.

    One ``World`` is one persistent installation: its library, site graph and recorded
    runs survive across invocations exactly as a data directory would, so ``learn``
    followed by ``run`` is genuinely two separate commands over shared memory.
    """

    scenario: Scenario
    settings: Settings
    store: InMemorySkillStore = field(default_factory=InMemorySkillStore)
    graph: InMemorySiteGraph = field(default_factory=InMemorySiteGraph)
    trajectories: InMemoryTrajectoryStore = field(default_factory=InMemoryTrajectoryStore)
    llm: FakeLLM = field(default_factory=FakeLLM)

    def script(self, replies: Sequence[str]) -> FakeLLM:
        """Give the model a script. Anything beyond it raises ``ScriptExhausted``."""
        self.llm = FakeLLM(replies)
        return self.llm

    def agent(self, task: TaskSpec, budget: Any) -> Agent:
        """The REAL agent wiring, with fakes only at the leaves.

        Built through :func:`~skillweaver.orchestrator.build_agent`, which is the same
        function the shipped workbench calls, so these tests exercise the planner, the
        explorer, the two critics and the admission gate as they are actually
        assembled - not a test-local approximation of them.
        """
        return build_agent(
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

    def _environment(self, trajectory: Trajectory) -> EnvironmentFactory:
        """The admission gate's world, put back where the recording started.

        The fake app resets to its first screen, which is this world's only way back -
        the equivalent of a browser that can navigate. When a run began somewhere
        else, the reset lands on the wrong screen and the gate refuses the candidate
        at its precondition check, which is the behaviour being relied on, not a
        limitation being worked around.
        """

        def factory() -> ReplayEnvironment:
            self.scenario.controller.reset()
            return ReplayEnvironment(self.scenario.controller, self.scenario.perceiver, self.graph)

        return factory

    def workbench(self) -> Workbench:
        """A workbench whose session opens the fake app instead of a browser."""

        @contextlib.contextmanager
        def session(task: TaskSpec, budget: Any) -> Iterator[Agent]:
            self.scenario.controller.reset()
            yield self.agent(task, budget)

        return Workbench(
            settings=self.settings,
            store=self.store,
            retriever=SkillRetriever(self.store),
            graph=self.graph,
            trajectories=self.trajectories,
            session=session,
        )

    def invoke(self, *args: str) -> Result:
        """Run one command against this world, as a person would type it."""
        return runner.invoke(app, list(args), obj=self.workbench())

    def json(self, *args: str) -> Any:
        """Run one command with ``--json`` and parse what it printed."""
        result = self.invoke(*args, "--json")
        assert result.exception is None, result.exception
        return json.loads(result.output)


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh installation: an empty library, an empty graph, nothing recorded."""
    config = load_settings(env={"SKILLWEAVER_DATA_DIR": str(tmp_path)}, env_file=None)
    return World(scenario=make_scenario(), settings=config)


def learn(world: World, *extra: str) -> Result:
    """Teach the world this task. Costs the eight scripted calls and nothing else."""
    world.script(LEARN_SCRIPT)
    return world.invoke("learn", TASK, "--domain", DOMAIN, "-p", COMPANY, *extra)


# --------------------------------------------------------------------------------------
# THE HEADLINE
# --------------------------------------------------------------------------------------


def test_learn_then_run_is_cold_then_warm_with_zero_model_calls(world: World) -> None:
    """THIS IS WHAT THE PROJECT CLAIMS, DRIVEN THE WAY A JUDGE WILL DRIVE IT.

    ``learn`` explores the task with a computer-use model, succeeds, and the admission
    gate stores the resulting skill. ``run`` of the SAME task then takes the warm path:
    it retrieves that skill, routes to the screen it starts on, runs its Python and
    verifies the result - and consults the model **exactly zero times**.

    The zero is asserted three ways, because one counter can always be wrong:

    * ``world.llm.calls`` does not move across the second command, and the script it
      is holding is exhausted, so a call would raise rather than be miscounted;
    * the run's own accounting reports ``llm_calls == 0``;
    * the fake app really did reach its confirmation page, so the zero is the cost of
      doing the work rather than the cost of doing nothing.
    """
    first = learn(world)
    assert first.exit_code == 0, first.output
    assert world.llm.calls == len(LEARN_SCRIPT), "the cold run cost more than it should"
    assert [s.name for s in world.store.list()] == ["confirm_invoice_payment"]

    spent_learning = world.llm.calls
    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY)

    assert report["decision"] == "warm"
    assert report["ok"]
    assert world.scenario.solved, "the app did not actually reach the confirmation page"
    # The claim, stated at the model and in the run's own books.
    assert world.llm.calls == spent_learning, "the warm run consulted the model"
    assert report["llm_calls"] == 0
    assert report["attempts"][0]["llm_calls"] == 0
    assert report["attempts"][0]["skills_used"] == ["confirm_invoice_payment"]
    assert report["learned"] is None and report["rescued"] is False


def test_the_warm_run_is_faster_because_it_skips_the_model_not_the_work(world: World) -> None:
    """The warm run performs the same three actions the cold run discovered.

    Guards the cheapest way to fake this result: a warm path that "succeeds" by doing
    nothing would also report zero model calls.
    """
    learn(world)
    cold_actions = list(world.scenario.controller.actions)
    world.scenario.controller.actions.clear()

    world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY)

    warm_actions = list(world.scenario.controller.actions)
    assert len(warm_actions) == 3
    assert [type(a) for a in warm_actions] == [type(a) for a in cold_actions[-3:]]


# --------------------------------------------------------------------------------------
# The fall-through, reported rather than hidden
# --------------------------------------------------------------------------------------


def _plant_half_done_skill(world: World) -> Skill:
    """Teach the world a skill that used to work and no longer finishes the job.

    The trajectory is real - it records the errand being completed - so
    :func:`~skillweaver.orchestrator.recall_end_state` can tell the warm critic where
    this task is supposed to end. The skill's code only gets halfway there, which is
    exactly the situation the critic exists to catch.
    """
    learn(world)
    stored = world.store.get("confirm_invoice_payment", DOMAIN)
    world.store.put(
        Skill(
            name=stored.name,
            domain=stored.domain,
            summary=stored.summary,
            docstring=stored.docstring,
            params={"company": {"type": "string"}},
            code=HALF_DONE_CODE,
            requires=(),
            precondition=stored.precondition,
            verifier_code=None,
            provenance=stored.provenance,
        )
    )
    world.scenario.controller.reset()
    return world.store.get("confirm_invoice_payment", DOMAIN)


def test_a_warm_attempt_that_fails_verification_falls_through_to_cold(world: World) -> None:
    """A stored skill that runs cleanly and does not finish the task must not pass.

    The warm path retrieves it, runs it without error - and the critic, comparing the
    screen it ended on with the screen the recorded run ended on, says no. The run
    then explores from where the skill left the app and finishes the errand.

    Crucially the report does not present this as an ordinary success. Both attempts
    are in it, the warm one records the stage it was rejected at, and ``rescued`` is
    set: the library was WRONG about this task, which is information, not a detail.
    """
    _plant_half_done_skill(world)
    world.script(SOLVE_FROM_SEARCHED)

    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY, "--no-learn")

    assert report["ok"] and report["decision"] == "cold"
    assert report["rescued"] is True, "a warm attempt that ran and failed was flattened away"
    warm, cold = report["attempts"]
    assert warm["path"] == "warm" and warm["ok"] is False
    assert warm["stage"] == "rejected"
    assert cold["path"] == "cold" and cold["ok"] is True
    assert world.scenario.solved


def test_the_prose_report_names_the_rescue_too(world: World) -> None:
    """The same finding has to survive into what a human reads, not only the JSON."""
    _plant_half_done_skill(world)
    world.script(SOLVE_FROM_SEARCHED)

    result = world.invoke("run", TASK, "--domain", DOMAIN, "-p", COMPANY, "--no-learn")

    assert result.exit_code == 0
    assert "warm path: failed at rejected" in result.output
    assert "WRONG about this task" in result.output


def test_a_rescued_run_is_not_admitted_when_it_cannot_be_replayed(world: World) -> None:
    """A rescue teaches nothing when the world cannot be put back where it started.

    Exploration that rescued a half-done warm attempt began on the FILTERED list, not
    on the invoice list, and this fake app can only be reset to its first screen. The
    admission gate therefore refuses the candidate at its precondition check rather
    than proving it on the wrong screen - the right answer, and worth pinning down,
    because the tempting alternative is to admit it and call the rescue a success.

    In a browser this case usually resolves itself: ``navigating_environment`` returns
    to the URL the RUN started on, which is reachable.
    """
    _plant_half_done_skill(world)
    world.script((*SOLVE_FROM_SEARCHED, skill_reply(), skill_reply(), skill_reply()))

    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY)

    assert report["ok"] and report["decision"] == "cold"
    assert report["learned"] is None
    assert "precondition" in report["learning_note"]


def test_a_skill_that_fails_outright_is_demoted_and_reported(world: World) -> None:
    """A skill that RAN and broke is retired, and the report says which one.

    This is the other half of the distinction the planner draws: a skill whose own
    expectation failed is evidence against that skill, unlike one the critic merely
    judged incomplete.
    """
    learn(world)
    stored = world.store.get("confirm_invoice_payment", DOMAIN)
    world.store.put(
        Skill(
            name=stored.name,
            domain=stored.domain,
            summary=stored.summary,
            docstring=stored.docstring,
            params={"company": {"type": "string"}},
            code=(
                "def run(ctx, company):\n"
                '    ctx.ctl.type_text("acme")\n'
                '    ctx.expect(False, "the confirmation dialog never appeared")\n'
            ),
            requires=(),
            precondition=stored.precondition,
            verifier_code=None,
            provenance=stored.provenance,
        )
    )
    world.scenario.controller.reset()
    world.script(SOLVE_FROM_SEARCHED)

    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY, "--no-learn")

    warm = report["attempts"][0]
    assert warm["stage"] == "skill_failed"
    assert warm["demoted"] == "confirm_invoice_payment"
    assert world.store.get("confirm_invoice_payment", DOMAIN).demoted_reason


# --------------------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------------------


def test_skills_ls_lists_what_was_learned(world: World) -> None:
    learn(world)

    result = world.invoke("skills", "ls")

    assert result.exit_code == 0
    assert "confirm_invoice_payment" in result.output
    assert DOMAIN in result.output


def test_skills_ls_on_an_empty_library_says_how_to_fill_it(world: World) -> None:
    result = world.invoke("skills", "ls")

    assert result.exit_code == 0
    assert "no skills stored" in result.output
    assert "skillweaver learn" in result.output


def test_skills_show_prints_the_python_the_agent_wrote(world: World) -> None:
    """The generated source is the artefact this project produces; --code prints it."""
    learn(world)

    result = world.invoke("skills", "show", "confirm_invoice_payment", "--code")

    assert result.exit_code == 0
    assert "def run(ctx" in result.output
    assert "def verify(ctx, result)" in result.output
    assert "taught by" in result.output


def test_skills_show_of_an_unknown_skill_fails_cleanly(world: World) -> None:
    """Exit 1 and one readable line - not a traceback, and not exit 0."""
    result = world.invoke("skills", "show", "no_such_skill")

    assert result.exit_code == 1
    assert "no skill named" in result.output
    assert "Traceback" not in result.output


def test_skills_rm_then_run_returns_to_the_cold_path(world: World) -> None:
    """Retiring a skill takes it out of retrieval, so the next run explores again.

    The skill is not deleted - its version is still in the store - but nothing on the
    warm path will reach for it, which is what "rm" has to mean for a library that is
    allowed to be wrong but not allowed to lose its history.
    """
    learn(world)
    removed = world.invoke("skills", "rm", "confirm_invoice_payment", "--reason", "site changed")
    assert removed.exit_code == 0
    assert world.store.get("confirm_invoice_payment", DOMAIN).demoted_reason == "site changed"
    assert world.store.list() == [], "a retired skill is still being offered"

    world.script((*SOLVE_FROM_LIST, skill_reply()))
    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY)

    assert report["decision"] == "cold"
    assert report["attempts"][0]["stage"] == "empty_library"
    assert report["rescued"] is False, "a cold start is not a rescue"
    assert report["learned"] == {"name": "confirm_invoice_payment", "version": 2}


def test_skills_ls_all_shows_the_retired_one_with_its_reason(world: World) -> None:
    learn(world)
    world.invoke("skills", "rm", "confirm_invoice_payment", "--reason", "site changed")

    result = world.invoke("skills", "ls", "--all")

    assert "confirm_invoice_payment" in result.output
    assert "RETIRED" in result.output


# --------------------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------------------


def test_graph_show_lists_the_screens_and_edges_a_run_discovered(world: World) -> None:
    """The map is a by-product of working: one learn run and the site graph exists."""
    learn(world)

    result = world.invoke("graph", "show", DOMAIN)

    assert result.exit_code == 0
    assert "4 screen(s), 3 edge(s)" in result.output
    assert "Payment confirmed" in result.output


def test_graph_show_on_an_unknown_domain_explains_rather_than_erroring(world: World) -> None:
    result = world.invoke("graph", "show", "never-visited.test")

    assert result.exit_code == 0
    assert "nothing known" in result.output


def test_graph_show_svg_is_well_formed_and_self_contained(world: World) -> None:
    """The SVG has to open in a browser and drop into a slide, so it is parsed here
    and checked for anything that would need fetching."""
    learn(world)

    result = world.invoke("graph", "show", DOMAIN, "--svg")

    assert result.exit_code == 0
    xml.dom.minidom.parseString(result.output)  # raises if it is not well-formed
    assert "Payment confirmed" in result.output
    assert "<script" not in result.output
    assert "http://" not in result.output.replace("http://www.w3.org/2000/svg", "")


def test_graph_show_svg_writes_to_a_file_when_asked(world: World, tmp_path: Path) -> None:
    learn(world)
    out = tmp_path / "nested" / "graph.svg"

    result = world.invoke("graph", "show", DOMAIN, "--svg", "--out", str(out))

    assert result.exit_code == 0
    assert out.read_text().startswith("<svg")
    assert str(out) in result.output


def test_graph_show_json_names_every_edge(world: World) -> None:
    learn(world)

    data = world.json("graph", "show", DOMAIN)

    assert len(data["states"]) == 4
    assert len(data["edges"]) == 3
    assert all(e["successes"] == 1 for e in data["edges"])


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------


def test_replay_shows_a_recorded_run_action_by_action(world: World) -> None:
    learn(world)
    run_id = world.trajectories.list()[0]

    result = world.invoke("replay", run_id)

    assert result.exit_code == 0
    assert "SOLVED" in result.output
    assert TASK in result.output
    assert result.output.count("ok") >= 3


def test_replay_with_no_argument_lists_the_runs_on_record(world: World) -> None:
    learn(world)

    result = world.invoke("replay")

    assert result.exit_code == 0
    assert world.trajectories.list()[0] in result.output


def test_replay_of_an_unknown_run_fails_cleanly(world: World) -> None:
    result = world.invoke("replay", "0000deadbeef")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "skillweaver replay" in result.output


def test_replay_json_carries_every_step(world: World) -> None:
    learn(world)
    run_id = world.trajectories.list()[0]

    data = world.json("replay", run_id)

    assert data["ok"] and len(data["steps"]) == 3
    assert [s["index"] for s in data["steps"]] == [0, 1, 2]
    assert all(s["ok"] for s in data["steps"])


# --------------------------------------------------------------------------------------
# dashboard and eval
# --------------------------------------------------------------------------------------


def test_dashboard_build_writes_one_self_contained_page(world: World, tmp_path: Path) -> None:
    out = tmp_path / "dash" / "index.html"

    result = world.invoke("dashboard", "build", "--out", str(out))

    assert result.exit_code == 0
    page = out.read_text()
    assert page.lstrip().startswith("<!")
    assert "<html" in page


def test_dashboard_build_works_before_anything_has_run(world: World, tmp_path: Path) -> None:
    """The page has to be openable before the first run, or it cannot be demoed."""
    result = world.invoke("--data-dir", str(tmp_path / "empty"), "dashboard", "build")

    assert result.exit_code == 0


def test_eval_run_says_it_is_not_built_yet_instead_of_crashing(world: World) -> None:
    """The seam is wired; the harness is another piece of work and has not landed.

    A missing module must produce a sentence a stranger can act on and exit code 2 -
    "could not run at all" - not an ImportError traceback.
    """
    result = world.invoke("eval", "run")

    assert result.exit_code == 2
    assert "not built yet" in result.output
    assert "Traceback" not in result.output
    assert "skillweaver.eval" in result.output


# --------------------------------------------------------------------------------------
# Budgets, flags and failure modes
# --------------------------------------------------------------------------------------


def test_a_model_call_budget_stops_the_cold_run(world: World) -> None:
    """A flag beats configuration, and the limit really stops the loop."""
    world.script(LEARN_SCRIPT)

    result = world.invoke(
        "learn", TASK, "--domain", DOMAIN, "-p", COMPANY, "--max-llm-calls", "2", "--json"
    )

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["ok"] is False and report["decision"] == "none"
    assert world.llm.calls < len(LEARN_SCRIPT)
    assert world.store.list() == [], "a run that ran out of budget must not store a skill"


def test_the_configured_budget_is_used_when_no_flag_overrides_it(world: World) -> None:
    """Configuration is the floor and there is no second configuration mechanism."""
    world.settings = load_settings(
        env={"SKILLWEAVER_MAX_LLM_CALLS": "2", "SKILLWEAVER_DATA_DIR": "unused"},
        env_file=None,
    )
    world.script(LEARN_SCRIPT)

    result = world.invoke("learn", TASK, "--domain", DOMAIN, "-p", COMPANY)

    assert result.exit_code == 1
    assert world.llm.calls < len(LEARN_SCRIPT)


def test_library_only_never_explores(world: World) -> None:
    """A run told not to explore fails honestly rather than paying a model."""
    result = world.invoke("run", TASK, "--domain", DOMAIN, "--library-only", "--json")

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["decision"] == "none"
    assert [a["path"] for a in report["attempts"]] == ["warm"]
    assert world.llm.calls == 0


def test_no_learn_runs_the_task_without_growing_the_library(world: World) -> None:
    world.script(SOLVE_FROM_LIST)

    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY, "--no-learn")

    assert report["ok"] and report["decision"] == "cold"
    assert report["learned"] is None
    assert "switched off" in report["learning_note"]
    assert world.store.list() == []


def test_an_unknown_target_is_refused_before_anything_opens(world: World) -> None:
    result = world.invoke("run", TASK, "--target", "telepathy")

    assert result.exit_code == 2
    assert "--target must be" in result.output


def test_a_malformed_param_is_refused_with_the_offending_text(world: World) -> None:
    result = world.invoke("run", TASK, "--domain", DOMAIN, "-p", "company")

    assert result.exit_code == 2
    assert "KEY=VALUE" in result.output


def test_params_keep_their_json_types(world: World) -> None:
    """``-p count=3`` is a number and ``-p company=Acme`` is a string, with no second
    flag to say which - a skill declaring a numeric parameter binds either way."""
    spec = task_spec("t", domain=DOMAIN, params={"count": 3, "company": "Acme"})

    assert spec.params["count"] == 3
    world.script(SOLVE_FROM_LIST + (skill_reply(),))
    report = world.json("run", TASK, "--domain", DOMAIN, "-p", COMPANY, "-p", "count=3")

    assert report["ok"]


# --------------------------------------------------------------------------------------
# --help, everywhere
# --------------------------------------------------------------------------------------

COMMANDS = [
    (),
    ("learn",),
    ("run",),
    ("skills",),
    ("skills", "ls"),
    ("skills", "show"),
    ("skills", "rm"),
    ("graph",),
    ("graph", "show"),
    ("replay",),
    ("dashboard",),
    ("dashboard", "build"),
    ("eval",),
    ("eval", "run"),
]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c) or "root")
def test_every_command_has_help_a_stranger_can_act_on(
    world: World, command: tuple[str, ...]
) -> None:
    """Help is the whole interface for someone meeting this project at a demo table."""
    result = world.invoke(*command, "--help")

    assert result.exit_code == 0
    body = result.output.strip()
    assert body, f"`{' '.join(command)} --help` printed nothing"
    assert len(body) > 200, f"`{' '.join(command)} --help` is too thin to act on"
    assert "Usage:" in result.output


def test_a_bare_invocation_shows_help_rather_than_doing_something(world: World) -> None:
    result = world.invoke()

    assert "Usage:" in result.output


# --------------------------------------------------------------------------------------
# The report type itself
# --------------------------------------------------------------------------------------


def test_a_cold_start_is_not_reported_as_a_rescue() -> None:
    """``rescued`` must mean "the library was wrong", not "the library was empty".

    If every first run cried rescue, the one signal that matters would be noise.
    """
    spec = TaskSpec(text="t", domain="d")
    declined = AttemptRecord("warm", False, "nothing stored", stage="empty_library")
    explored = AttemptRecord("cold", True, "explored it", steps=3, llm_calls=7)

    report = RunReport(True, spec, "cold", (declined, explored))

    assert report.rescued is False
    assert report.llm_calls == 7 and report.steps == 3
    assert "NOT SOLVED" not in report.explain()


def test_a_warm_attempt_that_acted_and_failed_is_a_rescue() -> None:
    spec = TaskSpec(text="t", domain="d")
    ran_and_failed = AttemptRecord(
        "warm", False, "the critic said no", stage="rejected", performed_nothing=False
    )
    explored = AttemptRecord("cold", True, "explored it")

    report = RunReport(True, spec, "cold", (ran_and_failed, explored))

    assert report.rescued is True
    assert "WRONG about this task" in report.explain()


def test_the_explanation_names_the_path_that_answered() -> None:
    spec = TaskSpec(text="pay it", domain="d")
    warm = AttemptRecord("warm", True, "ran pay_invoice", skills_used=("pay_invoice",), steps=3)

    text = RunReport(True, spec, "warm", (warm,)).explain()

    assert "SOLVED by the warm path" in text
    assert "0 model call(s)" in text
    assert "pay_invoice" in text


def test_recall_end_state_finds_where_the_task_finishes(world: World) -> None:
    """The fact that makes a warm verdict free: where the taught run ended."""
    learn(world)
    spec = task_spec(TASK, domain=DOMAIN)

    recalled = recall_end_state(world.store, world.trajectories, spec)

    assert recalled is not None
    trajectory: Trajectory = world.trajectories.load(world.trajectories.list()[0])
    assert recalled == trajectory.steps[-1].after.fingerprint


def test_recall_end_state_is_none_when_nothing_has_been_recorded(world: World) -> None:
    """And the warm path then pays a model for its verdict rather than guessing."""
    assert recall_end_state(world.store, world.trajectories, task_spec(TASK, domain=DOMAIN)) is None
    assert recall_end_state(world.store, None, task_spec(TASK, domain=DOMAIN)) is None


def test_task_spec_files_a_browser_task_under_the_host_of_its_url() -> None:
    spec = task_spec("pay it", url="https://acme.test/invoices?q=1")

    assert spec.domain == "acme.test"
    assert spec.params["start_url"] == "https://acme.test/invoices?q=1"
    assert spec.target == "browser"


def test_task_spec_falls_back_to_the_target_when_there_is_no_url() -> None:
    assert task_spec("open finder", target="desktop").domain == "desktop"
    assert task_spec("do a thing").domain == "browser"


def test_an_explicit_domain_always_wins() -> None:
    spec = task_spec("pay it", domain="chosen.test", url="https://acme.test/x")

    assert spec.domain == "chosen.test"


def test_provenance_is_carried_into_what_skills_show_prints(world: World) -> None:
    """A stored skill can always say which run taught it - that link is what
    :func:`recall_end_state` walks, so it is asserted rather than assumed."""
    learn(world)

    skill = world.store.get("confirm_invoice_payment", DOMAIN)

    assert isinstance(skill.provenance, Provenance)
    assert skill.provenance.trajectory_id in world.trajectories.list()
    assert skill.provenance.task_text == TASK


# --------------------------------------------------------------------------------------
# The process entry point
# --------------------------------------------------------------------------------------


def test_main_returns_the_exit_code_the_command_chose(tmp_path: Path) -> None:
    """``main`` is the path a real shell takes, and ``CliRunner`` does not exercise it.

    It once returned ``0`` for everything - including a usage error it had just
    printed - because it asked Typer not to handle exits and then caught the
    ``SystemExit`` that consequently never arrived. A command line that reports
    success while printing a failure is worse than one that crashes, so the three
    codes are pinned here against the real dispatch.
    """
    data = ("--data-dir", str(tmp_path))

    assert main([*data, "skills", "ls"]) == 0
    assert main([*data, "skills", "show", "definitely_not_a_skill"]) == 1
    assert main([*data, "eval", "run"]) == 2
    assert main([*data, "--help"]) == 0
    assert main([*data, "no-such-command"]) == 2
