"""The explorer: the loop that learns a task nobody has done before.

What is being proved here, in the order the file is written.

**It can actually solve something.** A scripted model drives the three-step task in
``tests/fakes/scenario.py`` to its goal, and the run is checked end to end: the outcome,
the shape of the trajectory, and the exact edges left in the site graph. The critic used
is a real :class:`~skillweaver.agent.critic.TieredCritic` configured to decide
programmatically, so ``fake_llm.calls`` counts acting calls only and the "three moves,
three model calls" claim is measured rather than asserted.

**It stops.** Each of the four limits in :class:`~skillweaver.contracts.Budget` gets its
own test and is shown stopping the loop ON ITS OWN, with the other three left wide open.

**It does not loop forever.** The headline behavioral test: a move that failed on a screen
is refused the second time it is proposed on that same screen, before it is performed, and
the model is told why. The assertion is on ``controller.actions`` - the dead end reaches
the screen exactly once.

**It fails honestly.** A run that cannot succeed comes back with a diagnosis naming the
state it was stuck in, and listing what it had tried there.

**It survives a bad answer.** Unparseable output and an element id that is not on the
screen are recoverable step failures: nothing is performed, the reason is remembered, and
the run carries on.

Nothing here touches a network, a browser or a real model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import pytest

from skillweaver.agent import checks as C
from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import (
    Attempt,
    Diagnosis,
    ElementCatalog,
    ExplorationOutcome,
    Explorer,
    FailureMemory,
    load_prompt,
)
from skillweaver.contracts import (
    Budget,
    Click,
    LLMResponse,
    Observation,
    RunOutcome,
    TypeText,
    Usage,
    Verdict,
)
from skillweaver.contracts import (
    Explorer as ExplorerProtocol,
)
from skillweaver.graph.model import InMemorySiteGraph
from tests.fakes import (
    FakeCritic,
    FakeLLM,
    InMemoryTrajectoryRecorder,
    Scenario,
    make_scenario,
)
from tests.fakes.scenario import (
    ARCHIVE_LINK,
    CONFIRM_BUTTON,
    DOMAIN,
    ROW_ACME,
    ROW_GLOBEX,
    TASK,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def answer(**fields: Any) -> str:
    """One model reply, as the acting prompt asks for it."""
    fields.setdefault("thought", "the next move")
    fields.setdefault("expect", "the screen changes to the next step of the task")
    fields.setdefault("done", False)
    return json.dumps(fields)


def click(element_id: str, **extra: Any) -> str:
    return answer(action={"kind": "click", "element_id": element_id}, **extra)


def type_text(text: str, **extra: Any) -> str:
    return answer(action={"kind": "type_text", "text": text}, **extra)


SOLVE = (
    type_text("acme"),
    click("row-1042"),
    click("confirm", done=True),
)
"""The three answers that solve the fake invoicing app, by element id."""


def changed_critic() -> TieredCritic:
    """A real critic that decides every question programmatically.

    ``state_changed`` is promoted from veto to evidence, which makes "the screen became a
    different screen" a decisive yes and "it did not" a decisive no. That is exactly right
    for the fake app, where every useful move moves the state machine - and it means no
    model call is ever made for judging, so every call this file counts is an ACTING call.
    ``require_change=False`` only stops the same check also running as a veto and saying
    everything twice.
    """
    return TieredCritic(None, evidence=(C.state_changed(),), require_change=False)


def build(
    scenario: Scenario,
    replies: Sequence[str | LLMResponse],
    **kwargs: Any,
) -> tuple[Explorer, FakeLLM, InMemorySiteGraph, InMemoryTrajectoryRecorder]:
    """An explorer wired to one scenario, with a scripted model and in-memory memory."""
    llm = FakeLLM(replies)
    graph = InMemorySiteGraph()
    recorder = InMemoryTrajectoryRecorder()
    explorer = Explorer(
        llm,
        scenario.perceiver,
        critic=kwargs.pop("critic", None) or changed_critic(),
        graph=graph,
        recorder=recorder,
        **kwargs,
    )
    return explorer, llm, graph, recorder


def fingerprints() -> dict[str, str]:
    """The fingerprint value of every state of the fake app, from a throwaway copy."""
    other = make_scenario()
    values = {}
    for name in other.controller.states:
        other.controller.state = name
        values[name] = other.perceiver.observe(other.controller).fingerprint.value
    return values


def edges(graph: InMemorySiteGraph) -> list[tuple[str, str, tuple]]:
    """Every edge as ``(src, dst, actions)``, so a test can name the exact set."""
    return sorted((t.src.value, t.dst.value, t.actions) for t in graph.transitions(DOMAIN))


BUDGET = Budget(max_steps=20, max_seconds=60.0, max_usd=10.0, max_llm_calls=20)
"""Generous on every axis, so a test about something else is never stopped by a limit."""


# --------------------------------------------------------------------------------------
# It solves the task
# --------------------------------------------------------------------------------------


def test_implements_the_explorer_protocol(scenario: Scenario) -> None:
    explorer, _, _, _ = build(scenario, SOLVE)
    assert isinstance(explorer, ExplorerProtocol)


def test_solves_the_three_step_task(scenario: Scenario) -> None:
    explorer, llm, _, _ = build(scenario, SOLVE)

    outcome = explorer.explore(TASK, scenario.controller, BUDGET)

    assert isinstance(outcome, RunOutcome)
    assert outcome.ok
    assert outcome.diagnosis is None
    assert scenario.solved
    assert scenario.controller.actions == list(scenario.solution)
    # Three moves, three model calls: the critic decided every verdict for free.
    assert llm.calls == 3
    assert outcome.spend.steps == 3
    assert outcome.spend.llm_calls == 3


def test_the_trajectory_has_the_shape_synthesis_expects(scenario: Scenario) -> None:
    explorer, _, _, _ = build(scenario, SOLVE)

    trajectory = explorer.explore(TASK, scenario.controller, BUDGET).trajectory

    assert trajectory.ok
    assert trajectory.task == TASK.text
    assert trajectory.domain == DOMAIN
    assert [s.index for s in trajectory.steps] == [0, 1, 2]
    assert [s.action for s in trajectory.steps] == list(scenario.solution)
    assert all(s.result.ok for s in trajectory.steps)
    assert all(s.verdict is not None and s.verdict.ok for s in trajectory.steps)
    # Each step's "after" is the next step's "before": one unbroken walk of the app.
    for earlier, later in zip(trajectory.steps, trajectory.steps[1:], strict=False):
        assert earlier.after.fingerprint == later.before.fingerprint
    assert trajectory.steps[-1].after.url == "https://fake.test/invoices/1042/confirmed"


def test_writes_exactly_the_transitions_it_walked(scenario: Scenario) -> None:
    explorer, _, graph, _ = build(scenario, SOLVE)
    fp = fingerprints()

    explorer.explore(TASK, scenario.controller, BUDGET)

    assert edges(graph) == sorted(
        [
            (fp["list"], fp["searched"], (TypeText("acme"),)),
            (fp["searched"], fp["selected"], (Click(ROW_ACME.box.center),)),
            (fp["selected"], fp["done"], (Click(CONFIRM_BUTTON.box.center),)),
        ]
    )
    for transition in graph.transitions(DOMAIN):
        assert (transition.attempts, transition.successes) == (1, 1)
        assert transition.mean_ms == 5.0  # the controller's own reported latency
        assert transition.last_verified is not None
    assert {s.fingerprint.value for s in graph.states(DOMAIN)} == {
        fp["list"],
        fp["searched"],
        fp["selected"],
        fp["done"],
    }


def test_states_are_labelled_and_carry_their_url(scenario: Scenario) -> None:
    explorer, _, graph, _ = build(scenario, SOLVE)
    fp = fingerprints()

    explorer.explore(TASK, scenario.controller, BUDGET)

    by_value = {s.fingerprint.value: s for s in graph.states(DOMAIN)}
    assert by_value[fp["done"]].label == "Payment confirmed"
    assert by_value[fp["done"]].url_pattern == "https://fake.test/invoices/1042/confirmed"


def test_a_failed_transition_is_written_down_too(scenario: Scenario) -> None:
    """The graph only ever grows here, so a move that led nowhere has to be recorded.

    Note that the Globex and Acme rows occupy the same box, so this is literally the same
    click that later works on the filtered list: a self-loop on ``list`` and an edge out of
    ``searched``. Which is why failure is a property of a move AT a state, never of a move.
    """
    explorer, _, graph, _ = build(scenario, [click("row-2041"), *SOLVE])
    fp = fingerprints()

    explorer.explore(TASK, scenario.controller, BUDGET)

    dead = [t for t in graph.transitions(DOMAIN) if t.src.value == t.dst.value == fp["list"]]
    assert len(dead) == 1
    assert dead[0].actions == (Click(ROW_GLOBEX.box.center),)
    assert (dead[0].attempts, dead[0].successes) == (1, 0)
    assert dead[0].last_verified is None
    assert dead[0].mean_ms == 0.0  # a failure's duration says nothing about the edge


# --------------------------------------------------------------------------------------
# Each of the four budget limits, on its own
# --------------------------------------------------------------------------------------


def test_max_steps_alone_stops_the_run(scenario: Scenario) -> None:
    explorer, llm, _, _ = build(scenario, SOLVE)

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0)
    )

    assert not outcome.ok
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.limit == "max_steps"
    assert outcome.spend.steps == 1
    assert llm.calls == 1
    assert scenario.controller.state == "searched"  # it got one move in, and no more


def test_max_llm_calls_alone_stops_the_run(scenario: Scenario) -> None:
    explorer, llm, _, _ = build(scenario, SOLVE)

    outcome = explorer.explore(
        TASK,
        scenario.controller,
        Budget(max_steps=99, max_seconds=60.0, max_usd=10.0, max_llm_calls=2),
    )

    assert not outcome.ok
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.limit == "max_llm_calls"
    assert llm.calls == 2
    assert not scenario.solved


def test_max_usd_alone_stops_the_run(scenario: Scenario) -> None:
    pricey = [LLMResponse(text=reply, usage=Usage(calls=1, cost_usd=0.6)) for reply in SOLVE]
    explorer, llm, _, _ = build(scenario, pricey)

    outcome = explorer.explore(
        TASK,
        scenario.controller,
        Budget(max_steps=99, max_seconds=60.0, max_usd=1.0, max_llm_calls=99),
    )

    assert not outcome.ok
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.limit == "max_usd"
    assert outcome.spend.usd == pytest.approx(1.2)
    assert llm.calls == 2
    assert not scenario.solved


def test_max_seconds_alone_stops_the_run(scenario: Scenario) -> None:
    class SlowLLM(FakeLLM):
        """A model that takes 50ms to answer, so wall-clock is the only limit in play."""

        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            time.sleep(0.05)
            return super().complete(*args, **kwargs)

    llm = SlowLLM(SOLVE)
    explorer = Explorer(
        llm,
        scenario.perceiver,
        critic=changed_critic(),
        graph=InMemorySiteGraph(),
        recorder=InMemoryTrajectoryRecorder(),
    )

    outcome = explorer.explore(
        TASK,
        scenario.controller,
        Budget(max_steps=99, max_seconds=0.08, max_usd=10.0, max_llm_calls=99),
    )

    assert not outcome.ok
    assert outcome.diagnosis is not None
    assert outcome.diagnosis.limit == "max_seconds"
    assert not scenario.solved
    assert outcome.spend.elapsed_seconds() >= 0.08


# --------------------------------------------------------------------------------------
# A dead end is not walked into twice
# --------------------------------------------------------------------------------------


def test_a_repeated_dead_end_is_refused_before_it_is_performed(scenario: Scenario) -> None:
    """The headline: the same failed move, proposed again on the same screen, is refused.

    Clicking a row on the unfiltered list does nothing at all, so the critic calls it a
    failure for free. When the model proposes it a second time the explorer must not spend
    a step re-learning that; it must refuse, explain, and ask again.
    """
    dead_end = click("row-2041")
    explorer, llm, _, _ = build(scenario, [dead_end, dead_end, type_text("acme")])

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=2, max_seconds=60.0, max_usd=10.0)
    )

    # The dead end reached the screen once; the second attempt was something else.
    assert scenario.controller.actions == [Click(ROW_GLOBEX.box.center), TypeText("acme")]
    assert llm.calls == 3
    assert outcome.spend.steps == 2

    # The model was asked again, and told what it had already tried.
    third = llm.requests[2].messages[0].text
    assert "already tried" in third
    assert "row-2041" in third

    # And the memory counted the repeat rather than forgetting it.
    tried = [a for a in (outcome.diagnosis.tried if outcome.diagnosis else ()) if a.count > 1]
    assert [a.count for a in tried] == [2]


def test_a_refused_repeat_is_counted_once_not_filed_twice(scenario: Scenario) -> None:
    """One stubborn idea must read as one problem in the diagnosis, not as two."""
    dead_end = click("row-2041")
    explorer, _, _, _ = build(scenario, [dead_end] * 4)

    outcome = explorer.explore(
        TASK,
        scenario.controller,
        Budget(max_steps=9, max_seconds=60.0, max_usd=10.0, max_llm_calls=4),
    )

    assert outcome.diagnosis is not None
    tried = outcome.diagnosis.tried
    assert len(tried) == 1
    assert tried[0].count == 4
    assert tried[0].summary.startswith("click [row-2041]")
    # The first reason is the observed one, not the memory's own later refusal.
    assert "the screen did not change" in tried[0].reason
    assert "already tried" not in tried[0].reason


def test_the_prompt_lists_what_failed_on_this_screen(scenario: Scenario) -> None:
    explorer, llm, _, _ = build(scenario, [click("row-2041"), type_text("acme")])

    explorer.explore(TASK, scenario.controller, Budget(max_steps=2, max_seconds=60.0, max_usd=10.0))

    first, second = (r.messages[0].text for r in llm.requests[:2])
    assert "(nothing yet on this screen)" in first
    assert "DO NOT PROPOSE ANY OF THESE AGAIN" in second
    assert "click [row-2041] row" in second
    assert "the screen did not change" in second


# --------------------------------------------------------------------------------------
# A run that cannot succeed says where it got stuck
# --------------------------------------------------------------------------------------


def test_a_hopeless_run_diagnoses_the_state_it_was_stuck_in(scenario: Scenario) -> None:
    """The archive is a dead end with no controls and no way out but a reload."""
    explorer, _, _, _ = build(
        scenario,
        [
            click("archive-link"),
            click("archive-title"),
            answer(action={"kind": "wait", "ms": 100}),
        ],
    )
    fp = fingerprints()

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=3, max_seconds=60.0, max_usd=10.0)
    )

    assert not outcome.ok
    assert scenario.stuck
    diagnosis = outcome.diagnosis
    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.state == fp["archive"]
    assert diagnosis.label == "Archive is empty"
    assert diagnosis.url == "https://fake.test/archive"
    assert (diagnosis.stopped_by, diagnosis.limit) == ("budget", "max_steps")
    assert diagnosis.moves == 3
    assert diagnosis.steps == 3

    rendered = diagnosis.render()
    assert fp["archive"][:12] in rendered
    assert "Archive is empty" in rendered
    assert "wait 100ms" in rendered
    assert "click [archive-title]" in rendered
    # The same account reaches a caller that only knows the Protocol.
    assert outcome.note == rendered
    assert outcome.verdict.reason == rendered
    assert not outcome.verdict.ok


def test_the_diagnosis_separates_this_screen_from_the_others(scenario: Scenario) -> None:
    explorer, _, _, _ = build(
        scenario, [click("row-2041"), click("archive-link"), click("archive-title")]
    )
    fp = fingerprints()

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=3, max_seconds=60.0, max_usd=10.0)
    )

    assert outcome.diagnosis is not None
    assert outcome.diagnosis.state == fp["archive"]
    states = {a.state for a in outcome.diagnosis.tried}
    assert states == {fp["list"], fp["archive"]}
    assert "1 further failed attempt(s) on other screens." in outcome.diagnosis.render()


# --------------------------------------------------------------------------------------
# Bad model output is a step failure, not a crash
# --------------------------------------------------------------------------------------


def test_unparseable_output_and_a_phantom_element_are_both_recoverable(
    scenario: Scenario,
) -> None:
    explorer, llm, _, _ = build(
        scenario,
        [
            "I think we should click on the Acme row.",
            click("no-such-element"),
            type_text("acme"),
        ],
    )

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0)
    )

    # Nothing was performed for either bad answer, and the run carried on regardless.
    assert scenario.controller.actions == [TypeText("acme")]
    assert scenario.controller.state == "searched"
    assert llm.calls == 3
    assert outcome.spend.steps == 1

    reasons = " ".join(a.reason for a in (outcome.diagnosis.tried if outcome.diagnosis else ()))
    assert "was not a JSON object" in reasons
    assert "no element 'no-such-element'" in reasons
    # And the model was told, in the turn right after each one.
    assert "was not a JSON object" in llm.requests[1].messages[0].text
    assert "no element 'no-such-element'" in llm.requests[2].messages[0].text


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (answer(action={"kind": "teleport", "element_id": "confirm"}), "unknown action kind"),
        (answer(action={"kind": "click"}), "needs an 'element_id'"),
        (answer(action={"kind": "type_text"}), "non-empty 'text' string"),
        (answer(action={"kind": "press_key", "keys": []}), "non-empty list of key names"),
        (answer(action={"kind": "scroll", "dx": 0, "dy": 0}), "non-zero 'dx' or 'dy'"),
        (answer(action={"kind": "navigate", "url": "https://x/"}), "cannot perform 'navigate'"),
        (answer(action="click the button"), "must be an object"),
        (answer(thought="no idea"), "contained no decision"),
        (answer(action={"kind": "click", "element_id": "confirm"}, code="pass"), "not both"),
    ],
)
def test_every_unusable_answer_is_explained_rather_than_raised(
    scenario: Scenario, reply: str, expected: str
) -> None:
    explorer, llm, _, _ = build(scenario, [reply, type_text("acme")])

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0)
    )

    assert llm.calls == 2
    assert expected in llm.requests[1].messages[0].text
    assert scenario.controller.actions == [TypeText("acme")]
    assert outcome.spend.steps == 1


# --------------------------------------------------------------------------------------
# Code blocks
# --------------------------------------------------------------------------------------


def test_a_code_block_is_one_move_and_several_recorded_actions(scenario: Scenario) -> None:
    """The other half of "the model may answer with a primitive or a short code block"."""
    explorer, llm, graph, _ = build(
        scenario,
        [
            answer(
                code='ctx.ctl.type_text("acme")\nctx.ctl.click(ctx.see.find_text("Acme Corp")[0])',
                expect="the Acme invoice page opens",
            ),
            answer(code='ctx.ctl.click(el["confirm"])', done=True),
        ],
    )
    fp = fingerprints()

    outcome = explorer.explore(TASK, scenario.controller, BUDGET)

    assert outcome.ok
    assert scenario.solved
    assert llm.calls == 2  # two moves, three actions
    assert scenario.controller.actions == list(scenario.solution)

    steps = outcome.trajectory.steps
    assert len(steps) == 3
    # The verdict belongs to the move, so it sits on the move's last action.
    assert steps[0].verdict is None and steps[0].note.startswith("part of: code block")
    assert steps[1].verdict is not None and steps[1].verdict.ok
    assert steps[2].verdict is not None and steps[2].verdict.ok
    # Still one edge per action, so the intermediate state is on the map.
    assert edges(graph) == sorted(
        [
            (fp["list"], fp["searched"], (TypeText("acme"),)),
            (fp["searched"], fp["selected"], (Click(ROW_ACME.box.center),)),
            (fp["selected"], fp["done"], (Click(CONFIRM_BUTTON.box.center),)),
        ]
    )


def test_a_code_block_that_breaks_is_a_step_failure(scenario: Scenario) -> None:
    """A block that reaches outside the sandbox keeps what it did and is told why."""
    explorer, llm, _, _ = build(
        scenario,
        [answer(code='import os\nctx.ctl.type_text("acme")'), type_text("acme")],
    )

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0)
    )

    assert scenario.controller.actions == [TypeText("acme")]  # the block never ran
    assert llm.calls == 2
    assert "may not import modules" in llm.requests[1].messages[0].text
    assert outcome.spend.steps == 1


def test_a_code_block_keeps_the_actions_it_managed_before_failing(scenario: Scenario) -> None:
    explorer, _, graph, _ = build(
        scenario,
        [
            answer(code='ctx.ctl.type_text("acme")\nctx.expect(False, "nothing is ever right")'),
            *SOLVE[1:],
        ],
    )
    fp = fingerprints()

    outcome = explorer.explore(TASK, scenario.controller, BUDGET)

    assert outcome.ok  # the run recovered: the typing had in fact worked
    assert scenario.controller.actions == list(scenario.solution)
    assert (fp["list"], fp["searched"], (TypeText("acme"),)) in edges(graph)


# --------------------------------------------------------------------------------------
# Claiming the task is done
# --------------------------------------------------------------------------------------


def test_a_done_claim_the_critic_rejects_does_not_end_the_run(scenario: Scenario) -> None:
    """``done`` is a claim. The critic compares the first screen with this one and decides."""

    class NeverSatisfied(TieredCritic):
        """Says yes to every step and no to every whole-task claim."""

        def judge(self, goal, before, after, expectation=None):  # type: ignore[override]
            if goal == TASK.text:
                return super().judge("this is not the task", before, before, expectation)
            return super().judge(goal, before, after, expectation)

    critic = NeverSatisfied(None, evidence=(C.state_changed(),))
    explorer, llm, _, _ = build(scenario, [*SOLVE, click("back")], critic=critic)

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=3, max_seconds=60.0, max_usd=10.0)
    )

    assert not outcome.ok
    assert scenario.solved  # it did reach the goal screen; the critic would not confirm it
    assert llm.calls == 3
    assert outcome.diagnosis is not None
    assert any("claim the task is complete" in a.summary for a in outcome.diagnosis.tried)


def test_done_on_its_own_finishes_a_run_without_acting(scenario: Scenario) -> None:
    """A run may already be finished when it starts; saying so costs one call and no step."""
    scenario.controller.state = "done"
    explorer, llm, _, _ = build(
        scenario,
        [answer(done=True, expect="the confirmation is already on screen")],
        critic=FakeCritic(default=Verdict(True, "the confirmation is on screen")),
    )

    outcome = explorer.explore(TASK, scenario.controller, BUDGET)

    assert outcome.ok
    assert scenario.controller.actions == []
    assert outcome.trajectory.steps == ()
    assert outcome.spend.steps == 0
    assert llm.calls == 1


# --------------------------------------------------------------------------------------
# The pieces, on their own
# --------------------------------------------------------------------------------------


def observe(scenario: Scenario, state: str) -> Observation:
    scenario.controller.state = state
    return scenario.perceiver.observe(scenario.controller)


def test_the_catalog_names_elements_by_their_stable_id(scenario: Scenario) -> None:
    catalog = ElementCatalog(observe(scenario, "selected").elements)

    assert set(catalog.ids) == {"invoice-title", "invoice-party", "confirm", "back"}
    assert catalog.get("confirm").text == "Confirm payment"
    assert "confirm" in catalog
    rendered = catalog.render()
    assert "[confirm] button 'Confirm payment' at (20,120) 160x40" in rendered


def test_the_catalog_refuses_an_id_that_is_not_on_the_screen(scenario: Scenario) -> None:
    catalog = ElementCatalog(observe(scenario, "selected").elements)

    with pytest.raises(Exception) as caught:  # noqa: B017 - the type is module-private
        catalog.get("search")

    assert "no element 'search'" in str(caught.value)
    assert "confirm" in str(caught.value)  # it says what IS there


def test_the_catalog_falls_back_to_positional_ids(scenario: Scenario) -> None:
    from skillweaver.contracts import Box, Element, ElementKind, ElementSource

    plain = tuple(
        Element(Box(0, y, 10, 10), ElementKind.other, "", 1.0, None, ElementSource.yolo)
        for y in (0, 10)
    )
    assert ElementCatalog(plain).ids == ("e0", "e1")


def test_the_failure_memory_is_per_screen_and_keeps_the_first_reason() -> None:
    memory = FailureMemory()
    memory.remember("state-a", "click:x", "click [x]", "the screen did not change")
    memory.remember("state-a", "click:x", "click [x]", "refused: already tried")
    memory.remember("state-b", "click:x", "click [x]", "an error appeared")

    here = memory.at("state-a")
    assert [(a.count, a.reason) for a in here] == [(2, "the screen did not change")]
    assert memory.seen("state-b", "click:x") == Attempt(
        "state-b", "click:x", "click [x]", "an error appeared", 1
    )
    assert memory.seen("state-c", "click:x") is None
    assert len(memory.all()) == 2


def test_the_prompt_demands_grounding_and_an_expectation() -> None:
    prompt = load_prompt()
    assert "element_id" in prompt
    assert "expect" in prompt
    assert "Act by id" in prompt


def test_the_outcome_is_a_run_outcome(scenario: Scenario) -> None:
    explorer, _, _, _ = build(scenario, SOLVE)
    outcome = explorer.explore(TASK, scenario.controller, BUDGET)
    assert isinstance(outcome, ExplorationOutcome)
    assert isinstance(outcome, RunOutcome)
    assert outcome.skill_used is None  # exploration, not a stored skill


def test_the_graph_neighbourhood_is_quoted_by_element_id_not_by_pixel(
    scenario: Scenario,
) -> None:
    """The prompt may not show the model a target it is forbidden to use."""
    graph = InMemorySiteGraph()
    recorder = InMemoryTrajectoryRecorder()
    first = Explorer(
        FakeLLM([click("row-2041"), type_text("acme")]),
        scenario.perceiver,
        critic=changed_critic(),
        graph=graph,
        recorder=recorder,
    )
    first.explore(TASK, scenario.controller, Budget(max_steps=2, max_seconds=60.0, max_usd=10.0))

    # A second run over the graph the first one left behind, from the same screen.
    again = make_scenario()
    llm = FakeLLM([type_text("acme")])
    Explorer(
        llm,
        again.perceiver,
        critic=changed_critic(),
        graph=graph,
        recorder=InMemoryTrajectoryRecorder(),
    ).explore(TASK, again.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0))

    known = llm.requests[0].messages[0].text.split("KNOWS ABOUT THIS SCREEN:\n")[1]
    known = known.split("\n\n")[0]
    assert "click [row-2041]" in known
    assert "(0/1 worked" in known
    assert "type_text 'acme'" in known  # a targetless action stays as it is
    assert "(400, 140)" not in known  # and no pixel is ever offered as a target


def test_the_prompt_carries_the_screenshot_and_the_graph(scenario: Scenario) -> None:
    explorer, llm, _, _ = build(scenario, SOLVE)

    explorer.explore(TASK, scenario.controller, BUDGET)

    first, third = llm.requests[0], llm.requests[2]
    assert first.system == load_prompt()
    assert first.messages[0].images == (observe(scenario, "list").screenshot.png,)
    assert "this screen is new" in first.messages[0].text
    assert TASK.text in first.messages[0].text
    assert "company='Acme Corp'" in first.messages[0].text
    assert "[confirm] button" in third.messages[0].text


def test_without_a_graph_nothing_is_remembered_but_the_run_still_works(
    scenario: Scenario,
) -> None:
    explorer = Explorer(
        FakeLLM(SOLVE),
        scenario.perceiver,
        critic=changed_critic(),
        graph=None,
        recorder=InMemoryTrajectoryRecorder(),
    )

    assert explorer.explore(TASK, scenario.controller, BUDGET).ok


def test_a_broken_controller_is_raised_not_diagnosed(scenario: Scenario) -> None:
    from skillweaver.errors import ControllerError

    explorer, _, _, _ = build(scenario, SOLVE)
    scenario.controller.close()

    with pytest.raises(ControllerError):
        explorer.explore(TASK, scenario.controller, BUDGET)


def test_the_archive_link_is_still_offered_after_the_trap_is_learned(
    scenario: Scenario,
) -> None:
    """A sanity check on the scenario itself: the trap is reachable and is not the goal."""
    explorer, _, _, _ = build(scenario, [click("archive-link")])

    outcome = explorer.explore(
        TASK, scenario.controller, Budget(max_steps=1, max_seconds=60.0, max_usd=10.0)
    )

    assert scenario.stuck
    assert not outcome.ok
    assert scenario.controller.actions == [Click(ARCHIVE_LINK.box.center)]
