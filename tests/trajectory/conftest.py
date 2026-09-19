"""Fixtures for the trajectory tests: a store rooted in ``tmp_path`` and one
recorded run that exercises every ``Action`` variant.

Nothing here touches ``settings()``, the network or a real browser: the store and
the recorder take their root explicitly so the tests never write outside
``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillweaver.contracts import (
    ACTION_TYPES,
    Action,
    Click,
    Drag,
    Move,
    Navigate,
    Observation,
    Point,
    PressKey,
    Scroll,
    Trajectory,
    TypeText,
    Verdict,
    Wait,
)
from skillweaver.trajectory.record import Recorder
from skillweaver.trajectory.store import TrajectoryFileStore
from tests.fakes import Scenario
from tests.fakes.scenario import CONFIRM_BUTTON, ROW_ACME

# The three actions that solve the fake app, then one of every other Action variant
# performed on the terminal screen - so a round trip proves EVERY variant survives.
# ``Navigate`` is unsupported by the fake controller on purpose: a recorded failure
# has to round-trip too.
RUN_ACTIONS: tuple[Action, ...] = (
    TypeText("acme"),
    Click(ROW_ACME.box.center),
    Click(CONFIRM_BUTTON.box.center),
    Click(Point(400, 300), button="right", clicks=2),
    Move(Point(400, 300)),
    Drag(Point(10, 20), Point(300, 400)),
    PressKey(("Meta", "a")),
    Scroll(Point(400, 300), dx=-20, dy=120),
    Wait(15),
    Navigate("https://fake.test/archive"),
)


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    """Where runs are written; stands in for ``settings().trajectories_dir``."""
    return tmp_path / "trajectories"


@pytest.fixture
def store(store_root: Path) -> TrajectoryFileStore:
    return TrajectoryFileStore(store_root)


@pytest.fixture
def recorder(store_root: Path) -> Recorder:
    return Recorder(store_root)


def drive(scenario: Scenario, recorder: Recorder, actions: tuple[Action, ...]) -> None:
    """Perform ``actions`` on ``scenario`` and record each one, as an agent loop would."""
    perceiver = scenario.perceiver
    controller = scenario.controller
    before: Observation = perceiver.observe(controller)
    for position, action in enumerate(actions):
        result = controller.perform(action)
        after = perceiver.observe(controller)
        verdict = Verdict(result.ok, "clicked the confirm button", 0.8) if position == 2 else None
        recorder.step(action, before, after, result, verdict, note=f"step {position}")
        before = after


@pytest.fixture
def recorded(scenario: Scenario, recorder: Recorder) -> Trajectory:
    """A finished run covering every ``Action`` variant, already on disk."""
    recorder.start(scenario.task.text, scenario.task.domain)
    drive(scenario, recorder, RUN_ACTIONS)
    assert scenario.solved, "the recorded run is supposed to reach the goal screen"
    trajectory = recorder.finish(ok=True, note="reached the confirmation page")
    assert {a.kind for a in RUN_ACTIONS} == set(ACTION_TYPES), "the run must cover every variant"
    return trajectory
