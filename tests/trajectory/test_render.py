"""The brief a skill is written from, on a run that FUMBLED.

A clean run renders the same either way, so nothing here is driven by one. The
fixture is a real recording of the fake app made by the real
:class:`~skillweaver.agent.explorer.Explorer` and judged by a real
:class:`~skillweaver.agent.critic.TieredCritic`, with a scripted model that gets it
wrong first: two clicks on rows that are not the one wanted (the screen does not
move, so the critic rejects the move), then a code block that searches and opens the
right row, then a detour through Back and in again, then the confirmation. Seven
actions, five moves, one rejected - which is what a messy run actually looks like.

What is being proved, and why each one is not cosmetic:

**A rejected move is marked as rejected.** The recording knows - the critic's verdict
is on the step - and until this rendering said so, a move the critic threw out was
shown to the model as one more step of the procedure.

**A move's stated reason is rendered before the actions it explains.** The explorer
puts the reason on the move's LAST step, because that is the step the verdict belongs
to. Read in place, that says the agent pressed Enter because it wanted to search, and
the click that opened the search box was unexplained.

**The grouping is structural.** Moves are found by the verdicts, not by parsing the
notes, and a recording with no verdicts at all falls back to one move per step rather
than inventing a twelve-action move nobody performed.

Nothing here touches a network, a browser or a real model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from skillweaver.agent import checks as C
from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import Explorer
from skillweaver.contracts import (
    ActionResult,
    Budget,
    Click,
    Point,
    Trajectory,
    TrajectoryStep,
    Verdict,
)
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.trajectory.render import Move, describe_trajectory, moves_of
from tests.fakes import FakeLLM, InMemoryTrajectoryRecorder, Scenario
from tests.fakes.scenario import TASK

# --------------------------------------------------------------------------------------
# The fumbling run
# --------------------------------------------------------------------------------------

WRONG_ROWS = 'ctx.ctl.click(el["row-2041"])\nctx.ctl.click(el["row-3377"])'
SEARCH_AND_OPEN = 'ctx.ctl.type_text("acme")\nctx.ctl.click(ctx.see.find_text("Acme Corp")[0])'

FIRST_REASON = (
    "The Acme invoice should already be one of the rows on this list, so I will click "
    "the Globex row and then the Initech row if that was wrong."
)
SECOND_REASON = (
    "The rows on screen are not Acme, so I will type 'acme' into the focused search "
    "field and then click the Acme Corp row it filters down to."
)


def answer(**fields: Any) -> str:
    fields.setdefault("expect", "the screen changes to the next step of the task")
    fields.setdefault("done", False)
    return json.dumps(fields)


FUMBLE = (
    answer(thought=FIRST_REASON, code=WRONG_ROWS),
    answer(thought=SECOND_REASON, code=SEARCH_AND_OPEN),
    answer(
        thought="Check the filtered list again.",
        action={"kind": "click", "element_id": "back"},
    ),
    answer(
        thought="Open the Acme invoice again.",
        action={"kind": "click", "element_id": "row-1042"},
    ),
    answer(
        thought="Confirm the payment.",
        action={"kind": "click", "element_id": "confirm"},
        done=True,
    ),
)


@pytest.fixture
def fumbled(scenario: Scenario) -> Trajectory:
    """A real recording of a run that tried something useless before it worked."""
    recorder = InMemoryTrajectoryRecorder()
    explorer = Explorer(
        FakeLLM(FUMBLE),
        scenario.perceiver,
        # Programmatic only: "the screen did not change" is the whole judgment here, so
        # the rejection in this recording is a real critic's and costs no model call.
        critic=TieredCritic(None, evidence=(C.state_changed(),), require_change=False),
        graph=InMemorySiteGraph(),
        recorder=recorder,
    )
    outcome = explorer.explore(TASK, scenario.controller, Budget(20, 60.0, 10.0, 20))
    assert outcome.ok, "the fixture is a run that fumbled and then SUCCEEDED"
    return outcome.trajectory


def line_of(brief: str, needle: str) -> int:
    """Where ``needle`` first appears, by line, so order can be asserted."""
    for index, line in enumerate(brief.splitlines()):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} is not in the brief")


# --------------------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------------------


class TestMoves:
    def test_the_steps_of_one_move_are_grouped_by_its_verdict(self, fumbled: Trajectory) -> None:
        moves = moves_of(fumbled)

        assert [len(move.steps) for move in moves] == [2, 2, 1, 1, 1]
        assert [move.index for move in moves] == [1, 2, 3, 4, 5]
        assert sum(len(move.steps) for move in moves) == len(fumbled.steps)

    def test_a_move_carries_the_reason_its_last_step_was_given(self, fumbled: Trajectory) -> None:
        first, second = moves_of(fumbled)[:2]

        assert first.reason == FIRST_REASON
        assert second.reason == SECOND_REASON
        # And the boilerplate the other steps of a code block carry is kept apart.
        assert all("part of:" in aside for aside in first.asides)

    def test_the_rejected_move_is_the_one_the_critic_rejected(self, fumbled: Trajectory) -> None:
        moves = moves_of(fumbled)

        assert [move.rejected for move in moves] == [True, False, False, False, False]
        assert moves[0].verdict is not None and not moves[0].verdict.ok

    def test_a_recording_with_no_verdicts_is_one_move_per_step(self, scenario: Scenario) -> None:
        """An older file or a hand-built run has no boundaries to read. Inventing one
        long move out of that would be a claim the recording does not make."""
        recorder = InMemoryTrajectoryRecorder()
        recorder.start(TASK.text, TASK.domain)
        for action in scenario.solution:
            before = scenario.perceiver.observe(scenario.controller)
            result = scenario.controller.perform(action)
            after = scenario.perceiver.observe(scenario.controller)
            recorder.step(action, before, after, result, None, "no verdict here")
        trajectory = recorder.finish(ok=True)

        assert [len(move.steps) for move in moves_of(trajectory)] == [1, 1, 1]

    def test_steps_after_the_last_verdict_are_a_move_of_their_own(
        self, fumbled: Trajectory
    ) -> None:
        """A run cut off mid-move still recorded what it did."""
        cut = Trajectory(
            run_id=fumbled.run_id,
            task=fumbled.task,
            domain=fumbled.domain,
            steps=(*fumbled.steps, _unjudged(fumbled)),
            ok=False,
            started_at=fumbled.started_at,
            finished_at=fumbled.finished_at,
        )

        moves = moves_of(cut)

        assert len(moves[-1].steps) == 1
        assert moves[-1].verdict is None


def _unjudged(trajectory: Trajectory) -> TrajectoryStep:
    """One more recorded action, on the run's last screen, that nobody judged."""
    screen = trajectory.steps[-1].after
    return TrajectoryStep(
        len(trajectory.steps),
        Click(Point(1, 1)),
        screen,
        screen,
        ActionResult(True),
        None,
        "unjudged",
    )


# --------------------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------------------


class TestDescribeTrajectory:
    def test_the_rejected_move_is_marked_and_the_accepted_ones_are_not(
        self, fumbled: Trajectory
    ) -> None:
        brief = describe_trajectory(fumbled)

        assert "MOVE 1 - the critic REJECTED it" in brief
        assert "THE CRITIC SAID NO" in brief
        assert brief.count("THE CRITIC SAID NO") == 1
        assert "MOVE 2 - the critic ACCEPTED it" in brief

    def test_the_headline_says_the_run_fumbled_before_any_of_it_is_read(
        self, fumbled: Trajectory
    ) -> None:
        brief = describe_trajectory(fumbled)

        assert "7 action(s) in 5 move(s)" in brief
        assert "REJECTED 1 (move(s) 1)" in brief
        assert "FUMBLED" in brief

    def test_a_reason_is_rendered_before_the_actions_it_explains(self, fumbled: Trajectory) -> None:
        """The defect this file exists for: the reason was attached to the move's LAST
        action, so a two-action move read as if only its second action was intended."""
        brief = describe_trajectory(fumbled)

        assert line_of(brief, SECOND_REASON) < line_of(brief, "STEP 2: type_text 'acme'")
        assert line_of(brief, FIRST_REASON) < line_of(brief, "STEP 0:")

    def test_every_recorded_action_is_still_shown(self, fumbled: Trajectory) -> None:
        """Marking a move is not hiding it: what the run did is still the recording."""
        brief = describe_trajectory(fumbled)

        for step in fumbled.steps:
            assert f"STEP {step.index}:" in brief

    def test_it_is_deterministic(self, fumbled: Trajectory) -> None:
        assert describe_trajectory(fumbled) == describe_trajectory(fumbled)

    def test_a_clean_run_says_nothing_about_fumbling(self, scenario: Scenario) -> None:
        recorder = InMemoryTrajectoryRecorder()
        recorder.start(TASK.text, TASK.domain)
        for action in scenario.solution:
            before = scenario.perceiver.observe(scenario.controller)
            result = scenario.controller.perform(action)
            after = scenario.perceiver.observe(scenario.controller)
            recorder.step(action, before, after, result, Verdict(True, "it worked", 1.0), "go")
        trajectory = recorder.finish(ok=True)

        brief = describe_trajectory(trajectory)

        assert "FUMBLED" not in brief
        assert "REJECTED" not in brief
        assert "STEPS: 3 action(s) in 3 move(s)" in brief

    def test_an_action_the_controller_refused_is_named_as_refused(self, scenario: Scenario) -> None:
        recorder = InMemoryTrajectoryRecorder()
        recorder.start(TASK.text, TASK.domain)
        screen = scenario.perceiver.observe(scenario.controller)
        recorder.step(
            Click(Point(1, 1)),
            screen,
            screen,
            ActionResult(False, "the controller cannot navigate"),
            Verdict(False, "nothing happened", 1.0),
            "try navigating there",
        )
        trajectory = recorder.finish(ok=False)

        brief = describe_trajectory(trajectory)

        assert "the controller REFUSED step 0: the controller cannot navigate" in brief

    def test_a_run_that_did_not_succeed_does_not_call_its_last_screen_the_goal(
        self, scenario: Scenario
    ) -> None:
        recorder = InMemoryTrajectoryRecorder()
        recorder.start(TASK.text, TASK.domain)
        before = scenario.perceiver.observe(scenario.controller)
        result = scenario.controller.perform(scenario.solution[0])
        after = scenario.perceiver.observe(scenario.controller)
        recorder.step(scenario.solution[0], before, after, result, Verdict(True, "ok", 1.0), "go")
        trajectory = recorder.finish(ok=False, note="out of budget")

        brief = describe_trajectory(trajectory)

        assert "FINAL SCREEN (the goal)" not in brief
        assert "FINAL SCREEN (where the run STOPPED" in brief

    def test_a_long_critic_reason_is_cut_rather_than_quoted_whole(self, scenario: Scenario) -> None:
        """A vision critic's reason carries its whole account of the screen, and the
        model already has the screen."""
        recorder = InMemoryTrajectoryRecorder()
        recorder.start(TASK.text, TASK.domain)
        before = scenario.perceiver.observe(scenario.controller)
        result = scenario.controller.perform(scenario.solution[0])
        after = scenario.perceiver.observe(scenario.controller)
        recorder.step(
            scenario.solution[0], before, after, result, Verdict(True, "x" * 600, 1.0), "go"
        )
        trajectory = recorder.finish(ok=True)

        brief = describe_trajectory(trajectory)

        assert "x" * 600 not in brief
        assert "…" in brief

    def test_the_brief_still_names_the_element_each_click_landed_on(
        self, fumbled: Trajectory
    ) -> None:
        """Rule 6 of the synthesis prompt depends on it: a coordinate is a screenshot,
        and the element under it is the only handle a skill can name again."""
        brief = describe_trajectory(fumbled)

        assert "this landed on the button reading 'Confirm payment'" in brief
        assert "in reading order" in brief


class TestMoveObject:
    def test_a_move_needs_at_least_one_step(self) -> None:
        with pytest.raises(IndexError):
            Move(1, [])
