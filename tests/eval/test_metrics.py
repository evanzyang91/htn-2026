"""The arithmetic behind the headline numbers, against fixed fixtures.

Every number this project puts on a slide comes out of
:mod:`skillweaver.eval.metrics`, so this file is mostly about the cases where the
honest answer is "no number": a run that took zero milliseconds, a task the cold run
could not do, a task the warm run broke. In each of those a plausible-looking figure
could be produced, and producing it would be the single easiest way for this project
to report a success it did not have. So the assertions below are as much about what
is ``None`` as about what is computed.

The fixtures are fixed integers rather than anything measured: 1000ms against 250ms is
4x, and a test that has to be recalculated when it fails is not a test.
"""

from __future__ import annotations

import pytest

from skillweaver.eval.metrics import (
    RunPoint,
    call_reduction,
    speedup,
    success_delta,
    suite_metrics,
    suite_metrics_from_report,
    task_metrics,
)


def cold(ms: float, *, ok: bool = True, calls: int = 12, usd: float = 0.25) -> RunPoint:
    """The first encounter: attempt 1, solved by exploring."""
    return RunPoint(attempt=1, ok=ok, wall_ms=ms, llm_calls=calls, usd=usd, used_library=False)


def warm(
    attempt: int, ms: float, *, ok: bool = True, calls: int = 0, used_library: bool = True
) -> RunPoint:
    """A later run, carried by the library unless told otherwise."""
    return RunPoint(
        attempt=attempt, ok=ok, wall_ms=ms, llm_calls=calls, usd=0.0, used_library=used_library
    )


# --------------------------------------------------------------------------------------
# speedup
# --------------------------------------------------------------------------------------


def test_speedup_is_cold_over_warm_so_above_one_means_faster() -> None:
    assert speedup(1000.0, 250.0) == 4.0
    assert speedup(250.0, 1000.0) == 0.25


@pytest.mark.parametrize(
    ("cold_ms", "warm_ms"),
    [(0.0, 250.0), (1000.0, 0.0), (0.0, 0.0), (-5.0, 250.0), (1000.0, -5.0)],
)
def test_a_zero_or_negative_duration_has_no_speedup_rather_than_an_enormous_one(
    cold_ms: float, warm_ms: float
) -> None:
    """A 0ms run is a broken clock, not an infinitely fast run.

    This is the case that would otherwise produce the largest number in the report out
    of the least reliable measurement in it.
    """
    assert speedup(cold_ms, warm_ms) is None


def test_a_missing_side_has_no_speedup() -> None:
    assert speedup(None, 250.0) is None
    assert speedup(1000.0, None) is None


# --------------------------------------------------------------------------------------
# success delta
# --------------------------------------------------------------------------------------


def test_success_delta_is_warm_minus_cold() -> None:
    assert success_delta(False, [True, True, True]) == 1.0
    assert success_delta(True, [False, False, False]) == -1.0
    assert success_delta(True, [True, True, True]) == 0.0


def test_success_delta_averages_a_flaky_warm_path() -> None:
    assert success_delta(True, [True, False, True, False]) == -0.5


def test_no_warm_run_means_no_delta_rather_than_zero() -> None:
    """``0.0`` would read as "the library neither helped nor hurt". Nothing was asked."""
    assert success_delta(True, []) is None
    assert success_delta(None, [True]) is None


def test_call_reduction_is_an_absolute_count() -> None:
    """A ratio against zero calls is either undefined or misleadingly enormous, and
    zero calls is precisely the result the warm path exists to produce."""
    assert call_reduction(12, 0.0) == 12.0
    assert call_reduction(None, 0.0) is None
    assert call_reduction(12, None) is None


# --------------------------------------------------------------------------------------
# One task
# --------------------------------------------------------------------------------------


def test_a_healthy_task_reports_the_speedup_and_the_calls_it_saved() -> None:
    metrics = task_metrics(
        "open_records",
        [cold(1200.0, calls=9), warm(2, 300.0), warm(3, 300.0), warm(4, 600.0)],
    )
    assert metrics.runs == 4
    assert metrics.cold_ok is True
    assert metrics.warm_ok_rate == 1.0
    assert metrics.success_delta == 0.0
    assert metrics.cold_ms == 1200.0
    assert metrics.warm_ms == 400.0
    assert metrics.speedup == 3.0
    assert metrics.call_reduction == 9.0
    assert metrics.comparable is True
    assert metrics.regressed is False
    assert metrics.note == ""


def test_the_lowest_attempt_is_the_cold_run_whatever_order_the_runs_arrive_in() -> None:
    shuffled = [warm(3, 300.0), cold(1200.0), warm(2, 300.0)]
    assert task_metrics("t", shuffled).cold_ms == 1200.0


def test_a_task_that_failed_cold_has_no_baseline_to_be_faster_than() -> None:
    """The awkward case: the warm runs worked, so there ARE warm timings, and dividing
    the failed cold run's duration by them would invent a speedup from a run that never
    did the task."""
    metrics = task_metrics(
        "pause_security_records",
        [cold(400.0, ok=False), warm(2, 200.0), warm(3, 200.0)],
    )
    assert metrics.cold_ok is False
    assert metrics.cold_ms is None
    assert metrics.warm_ms == 200.0
    assert metrics.speedup is None
    assert metrics.comparable is False
    assert "cold run failed" in metrics.note
    # The success story is still told: the library can do what exploration could not.
    assert metrics.success_delta == 1.0


def test_a_task_whose_warm_runs_all_failed_reports_the_regression_not_a_speedup() -> None:
    """The other awkward case, and the worse one: a warm run that gives up in 20ms
    against a cold run that spent 1200ms succeeding would otherwise read as 60x."""
    metrics = task_metrics(
        "rename_aurora_ledger",
        [cold(1200.0), warm(2, 20.0, ok=False), warm(3, 20.0, ok=False)],
    )
    assert metrics.speedup is None
    assert metrics.comparable is False
    assert metrics.warm_failures == 2
    assert metrics.warm_ok_rate == 0.0
    assert metrics.success_delta == -1.0
    assert "every warm run failed" in metrics.note


def test_a_partly_failing_warm_path_is_timed_on_its_successes_only() -> None:
    metrics = task_metrics(
        "flaky", [cold(1000.0), warm(2, 250.0), warm(3, 10.0, ok=False), warm(4, 350.0)]
    )
    assert metrics.warm_ms == 300.0, "the 10ms give-up is not a measurement of doing the task"
    assert metrics.speedup == pytest.approx(1000.0 / 300.0)
    assert metrics.warm_failures == 1
    assert metrics.warm_ok_rate == pytest.approx(2 / 3)
    assert metrics.success_delta == pytest.approx(-1 / 3)


def test_a_zero_duration_warm_run_makes_the_task_incomparable() -> None:
    metrics = task_metrics("instant", [cold(1000.0), warm(2, 0.0)])
    assert metrics.speedup is None
    assert "broken clock" in metrics.note


def test_a_cold_only_task_says_it_is_awaiting_its_warm_run() -> None:
    metrics = task_metrics("never_repeated", [cold(1000.0)])
    assert metrics.warm_ok_rate is None
    assert metrics.success_delta is None
    assert metrics.speedup is None
    assert "no warm run" in metrics.note


def test_a_task_with_no_runs_says_so_instead_of_raising() -> None:
    metrics = task_metrics("nothing", [])
    assert metrics.runs == 0
    assert metrics.note == "no runs were recorded"


def test_a_slower_warm_path_is_flagged_as_a_regression_not_averaged_away() -> None:
    metrics = task_metrics("slower", [cold(200.0), warm(2, 800.0)])
    assert metrics.speedup == 0.25
    assert metrics.regressed is True


def test_a_warm_run_that_explored_is_counted_as_warm_in_name_only() -> None:
    metrics = task_metrics(
        "not_really_warm",
        [cold(1000.0), warm(2, 900.0, used_library=False), warm(3, 900.0, used_library=False)],
    )
    assert metrics.warm_explored == 2


# --------------------------------------------------------------------------------------
# A whole suite
# --------------------------------------------------------------------------------------


def test_the_pooled_speedup_weights_a_task_by_how_long_it_actually_takes() -> None:
    """One 10-second task that halved matters more than one 10ms task that did not,
    and the pooled figure is the one that says so."""
    fast = task_metrics("fast", [cold(10.0), warm(2, 10.0)])
    slow = task_metrics("slow", [cold(10_000.0), warm(2, 5_000.0)])
    summary = suite_metrics([fast, slow])

    assert summary.pooled_speedup == pytest.approx(10_010.0 / 5_010.0)
    assert summary.mean_speedup == pytest.approx(1.5), "the unweighted mean disagrees, as it should"
    assert summary.comparable_tasks == 2


def test_an_incomparable_task_is_excluded_from_the_speedup_and_named_in_the_failures() -> None:
    good = task_metrics("good", [cold(1000.0), warm(2, 250.0)])
    broke = task_metrics("broke", [cold(1000.0), warm(2, 30.0, ok=False)])
    never = task_metrics("never", [cold(800.0, ok=False), warm(2, 100.0, ok=False)])
    summary = suite_metrics([good, broke, never])

    assert summary.comparable_tasks == 1
    assert summary.pooled_speedup == 4.0, "only the comparable task contributes"
    assert summary.warm_failures == ("broke", "never")
    assert summary.cold_failures == ("never",)


def test_a_suite_where_no_warm_run_used_the_library_says_it_measured_cold_versus_cold() -> None:
    """The failure mode that invalidates every other number: synthesis stored nothing,
    so each "warm" run explored from scratch and the speedup is roughly 1.0 while
    everything appears to work."""
    tasks = [
        task_metrics(
            name,
            [cold(1000.0), warm(2, 990.0, used_library=False), warm(3, 1010.0, used_library=False)],
        )
        for name in ("a", "b")
    ]
    summary = suite_metrics(tasks)

    assert summary.warm_runs == 4
    assert summary.warm_runs_that_explored == 4
    assert summary.library_was_used is False
    assert summary.pooled_speedup == pytest.approx(1.0, abs=0.01), (
        "and it looks like a perfectly ordinary null result, which is the whole danger"
    )


def test_one_warm_run_on_the_library_is_enough_for_the_suite_to_count_as_warm() -> None:
    summary = suite_metrics(
        [task_metrics("a", [cold(1000.0), warm(2, 200.0, used_library=False), warm(3, 200.0)])]
    )
    assert summary.warm_runs == 2
    assert summary.warm_runs_that_explored == 1
    assert summary.library_was_used is True


def test_an_empty_suite_answers_with_empty_metrics_rather_than_raising() -> None:
    summary = suite_metrics([])
    assert summary.tasks == ()
    assert summary.pooled_speedup is None
    assert summary.library_was_used is False


# --------------------------------------------------------------------------------------
# Re-deriving the numbers from a written report
# --------------------------------------------------------------------------------------


def test_the_metrics_can_be_recomputed_from_a_report_file_rather_than_trusted() -> None:
    report = {
        "tasks": [
            {
                "task_id": "open_records",
                "runs": [
                    {"attempt": 1, "ok": True, "wall_ms": 1200.0, "llm_calls": 9},
                    {
                        "attempt": 2,
                        "ok": True,
                        "wall_ms": 300.0,
                        "llm_calls": 0,
                        "skill_used": "open_records",
                    },
                ],
            }
        ]
    }
    summary = suite_metrics_from_report(report)

    assert summary.comparable_tasks == 1
    assert summary.tasks[0].speedup == 4.0
    assert summary.tasks[0].call_reduction == 9.0
    assert summary.library_was_used is True, "skill_used implies the library carried it"


def test_a_run_missing_its_attempt_falls_back_to_its_position() -> None:
    report = {
        "tasks": [
            {
                "task_id": "t",
                "runs": [{"ok": True, "wall_ms": 900.0}, {"ok": True, "wall_ms": 300.0}],
            }
        ]
    }
    assert suite_metrics_from_report(report).tasks[0].speedup == 3.0


@pytest.mark.parametrize("report", [{}, {"tasks": "not a list"}, {"tasks": [{"no": "id"}]}])
def test_a_malformed_report_yields_empty_metrics_rather_than_a_traceback(report: dict) -> None:
    """This is a reporting path: a summary that says "nothing to report" beats a crash."""
    assert suite_metrics_from_report(report).tasks == ()
