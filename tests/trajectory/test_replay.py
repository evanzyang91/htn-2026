"""Replay: a recorded run re-issued against a fresh controller, and the report that
says exactly where a changed app stops matching it."""

from __future__ import annotations

from dataclasses import replace

import pytest

from skillweaver.contracts import ActionResult, Click, Point, Trajectory
from skillweaver.trajectory.record import Recorder
from skillweaver.trajectory.replay import replay, validate
from skillweaver.trajectory.store import TrajectoryFileStore
from tests.fakes import Scenario, clicks, make_scenario
from tests.fakes.scenario import ARCHIVE_LINK, CLEAR_BUTTON, ROW_ACME
from tests.trajectory.conftest import RUN_ACTIONS, drive

# -- a clean replay ---------------------------------------------------------------------


def test_replaying_a_recorded_run_against_a_fresh_app_is_clean(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)
    fresh = make_scenario()

    report = replay(loaded, fresh.controller, perceiver=fresh.perceiver)

    assert report.ok, str(report)
    assert report.divergences == () and report.first_divergence is None
    assert report.steps_replayed == report.steps_total == len(RUN_ACTIONS)
    assert report.compared == "fingerprint"
    assert fresh.solved, "replay should have driven the fresh app to the goal"
    assert fresh.controller.actions == list(RUN_ACTIONS)
    assert "clean" in str(report)


def test_replay_falls_back_to_urls_without_a_perceiver(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()

    report = replay(store.load(recorded.run_id), fresh.controller)

    assert report.ok and report.compared == "url"
    assert fresh.solved


# -- divergence -------------------------------------------------------------------------


def test_a_second_step_that_goes_somewhere_else_is_reported_as_step_two(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()
    # the app changed: clicking the Acme row now lands in the archive, not the invoice
    fresh.controller.transitions["searched"] = [(clicks(ROW_ACME), "archive")]

    report = replay(store.load(recorded.run_id), fresh.controller, perceiver=fresh.perceiver)

    assert not report.ok
    diverged = report.first_divergence
    assert diverged is not None
    assert diverged.index == 1  # the second step, the click on the Acme row
    assert diverged.reason == "fingerprint"
    assert diverged.action == RUN_ACTIONS[1]
    assert diverged.expected == recorded.steps[1].after.fingerprint.value
    assert diverged.observed != diverged.expected
    assert diverged.similarity is not None and diverged.similarity < 1.0
    assert report.steps_replayed == 2, "replay stops at the divergence"
    assert fresh.stuck
    assert "step 1 fingerprint" in str(report)


def test_a_refused_action_is_reported_with_the_controllers_reason(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()
    fresh.controller.fail_next("the search box moved")

    report = replay(store.load(recorded.run_id), fresh.controller, perceiver=fresh.perceiver)

    diverged = report.first_divergence
    assert diverged is not None
    assert (diverged.index, diverged.reason) == (0, "action_result")
    assert diverged.expected == "ok=True" and "the search box moved" in diverged.observed
    assert report.steps_replayed == 1


def test_a_url_divergence_is_reported_without_a_perceiver(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()
    fresh.controller.transitions["searched"] = [(clicks(ROW_ACME), "archive")]

    report = replay(store.load(recorded.run_id), fresh.controller)

    diverged = report.first_divergence
    assert diverged is not None
    assert (diverged.index, diverged.reason) == (1, "url")
    assert diverged.expected == "https://fake.test/invoices/1042"
    assert diverged.observed == "https://fake.test/archive"
    assert diverged.similarity is None


def test_a_zero_similarity_threshold_tolerates_a_changed_screen(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()
    fresh.controller.transitions["searched"] = [(clicks(ROW_ACME), "archive")]

    report = replay(
        store.load(recorded.run_id),
        fresh.controller,
        perceiver=fresh.perceiver,
        min_similarity=0.0,
    )

    assert report.ok, "min_similarity=0.0 accepts any screen the actions still reach"
    assert report.steps_replayed == report.steps_total


def test_collecting_every_divergence(recorded: Trajectory, store: TrajectoryFileStore) -> None:
    fresh = make_scenario()
    # clicking the Acme row does nothing now, so every later step is wrong too
    fresh.controller.transitions["searched"] = [(clicks(CLEAR_BUTTON), "list")]

    report = replay(
        store.load(recorded.run_id),
        fresh.controller,
        perceiver=fresh.perceiver,
        stop_on_divergence=False,
    )

    assert not report.ok
    assert report.steps_replayed == report.steps_total
    assert [d.index for d in report.divergences] == list(range(1, len(RUN_ACTIONS)))
    assert {d.reason for d in report.divergences} == {"fingerprint"}


def test_a_dead_controller_becomes_a_divergence_not_an_exception(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()
    fresh.controller.close()

    report = replay(store.load(recorded.run_id), fresh.controller, perceiver=fresh.perceiver)

    diverged = report.first_divergence
    assert diverged is not None
    assert (diverged.index, diverged.reason) == (0, "controller_error")
    assert "closed" in diverged.observed
    assert report.steps_replayed == 0


# -- dry run ----------------------------------------------------------------------------


def test_a_dry_run_validates_without_touching_the_controller(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    fresh = make_scenario()

    report = replay(store.load(recorded.run_id), fresh.controller, dry_run=True)

    assert report.ok and report.dry_run
    assert report.steps_replayed == 0 and report.steps_total == len(RUN_ACTIONS)
    assert fresh.controller.actions == [] and fresh.controller.state == "list"
    assert "dry run" in str(report)


def test_a_dry_run_rejects_a_trajectory_this_controller_cannot_replay(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)
    # a click 50 logical pixels past the right edge, recorded as having SUCCEEDED
    off_screen = replace(
        loaded.steps[0],
        action=Click(Point(loaded.steps[0].before.screenshot.width + 50, 10)),
        result=ActionResult(ok=True),
    )
    broken = replace(loaded, steps=(off_screen, *loaded.steps[1:]))
    fresh = make_scenario()

    report = replay(broken, fresh.controller, dry_run=True)

    assert not report.ok
    assert report.first_divergence is not None
    assert report.first_divergence.reason == "out_of_viewport"
    assert "(850, 10)" in report.first_divergence.observed
    assert fresh.controller.actions == []


def test_validate_exempts_steps_the_recording_already_shows_failing(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)
    fresh = make_scenario()

    # the last step is a Navigate the fake controller refuses, and the recording
    # says so: re-refusing it is faithful replay, not divergence
    assert loaded.steps[-1].action.kind == "navigate"
    assert not fresh.controller.supports("navigate")
    assert validate(loaded, fresh.controller) == []


def test_out_of_order_steps_are_reported(recorded: Trajectory, store: TrajectoryFileStore) -> None:
    loaded = store.load(recorded.run_id)
    scrambled = replace(loaded, steps=(loaded.steps[1], loaded.steps[0], *loaded.steps[2:]))

    report = replay(scrambled, make_scenario().controller, dry_run=True)

    assert not report.ok
    assert [d.reason for d in report.divergences][:2] == ["bad_index", "bad_index"]


def test_an_unreplayable_trajectory_never_touches_the_controller(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)
    scrambled = replace(loaded, steps=(loaded.steps[1], loaded.steps[0], *loaded.steps[2:]))
    fresh = make_scenario()

    report = replay(scrambled, fresh.controller, perceiver=fresh.perceiver)

    assert not report.ok and not report.dry_run
    assert report.steps_replayed == 0
    assert fresh.controller.actions == []


# -- partial and empty runs ---------------------------------------------------------------


def test_a_crashed_run_replays_as_far_as_it_got(
    scenario: Scenario, recorder: Recorder, store: TrajectoryFileStore
) -> None:
    run_id = recorder.start(scenario.task.text, scenario.task.domain)
    drive(scenario, recorder, RUN_ACTIONS[:2])  # killed before finish()

    fresh = make_scenario()
    report = replay(store.load(run_id), fresh.controller, perceiver=fresh.perceiver)

    assert report.ok and report.steps_replayed == 2
    assert fresh.controller.state == "selected"


def test_a_failed_run_replays_faithfully_into_the_dead_end(
    scenario: Scenario, recorder: Recorder, store: TrajectoryFileStore
) -> None:
    run_id = recorder.start("click around", "fake.test")
    drive(scenario, recorder, (Click(ARCHIVE_LINK.box.center),))
    recorder.finish(ok=False, note="walked into the archive")

    fresh = make_scenario()
    report = replay(store.load(run_id), fresh.controller, perceiver=fresh.perceiver)

    assert report.ok, "a failed run is still a faithful recording"
    assert fresh.stuck


@pytest.mark.parametrize("dry_run", [True, False])
def test_an_empty_trajectory_replays_cleanly(
    scenario: Scenario, recorder: Recorder, store: TrajectoryFileStore, dry_run: bool
) -> None:
    run_id = recorder.start("do nothing", "fake.test")
    recorder.finish(ok=True)

    report = replay(store.load(run_id), scenario.controller, dry_run=dry_run)

    assert report.ok and report.steps_total == 0 and report.steps_replayed == 0
