"""The panel that prices the library: tokens and dollars saved, conservatively.

This number is the one a stranger will try to poke a hole in, so the tests are written
against the ways it could flatter:

*A missing measurement is never a saving of zero.* The harness writes ``None`` for
tokens when no usage meter was attached; ``None`` has to survive the reader, the
arithmetic and the page.

*A task nobody solved cold has no baseline* and is excluded and named, rather than
credited with whatever its repeats happened to cost.

*A repeat that FAILED is charged its own cost and credited nothing*, because the work
still had to be done afterwards. Averaging it away would hide the one case where the
library made things worse.

The arithmetic is also checked end to end against a report in the exact shape
``skillweaver.eval.harness.RunRecord.to_json`` emits, computed by hand in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillweaver.dashboard.build import (
    EvalRun,
    build_dashboard,
    build_savings_panel,
    collect,
    human_count,
    human_usd,
    read_eval_reports,
)

FIXTURES = Path(__file__).parents[1] / "dashboard_fixtures"
FULL = FIXTURES / "full"


def _report(tasks: list[dict], path: Path) -> Path:
    (path / "eval").mkdir(parents=True, exist_ok=True)
    (path / "eval" / "suite.json").write_text(
        json.dumps({"schema_version": 1, "suite": "unit", "tasks": tasks}), encoding="utf-8"
    )
    return path


def _panel(tasks: list[dict], path: Path):
    return build_savings_panel(*read_eval_reports(_report(tasks, path) / "eval"))


# --------------------------------------------------------------------------------------
# Reading the two new fields
# --------------------------------------------------------------------------------------


def test_the_token_fields_are_read_off_the_harnesss_own_keys(tmp_path: Path) -> None:
    _report(
        [
            {
                "task_id": "t",
                "runs": [{"attempt": 1, "ok": True, "input_tokens": 900, "output_tokens": 120}],
            }
        ],
        tmp_path,
    )
    run = read_eval_reports(tmp_path / "eval")[0][0].tasks[0].runs[0]
    assert (run.input_tokens, run.output_tokens, run.tokens) == (900, 120, 1020)


@pytest.mark.parametrize(
    "run",
    [{"attempt": 1}, {"attempt": 1, "input_tokens": None, "output_tokens": None}],
)
def test_an_unmeasured_run_keeps_none_rather_than_becoming_zero(run: dict, tmp_path: Path) -> None:
    """The whole point of the distinction: nobody counted is not the same as counted none."""
    _report([{"task_id": "t", "runs": [run]}], tmp_path)
    parsed = read_eval_reports(tmp_path / "eval")[0][0].tasks[0].runs[0]
    assert parsed.input_tokens is None and parsed.output_tokens is None
    assert parsed.tokens is None


def test_a_genuine_zero_is_kept_as_a_zero(tmp_path: Path) -> None:
    """A warm run that made no model call at all really did spend nothing."""
    _report(
        [
            {
                "task_id": "t",
                "runs": [{"attempt": 1, "input_tokens": 0, "output_tokens": 0}],
            }
        ],
        tmp_path,
    )
    assert read_eval_reports(tmp_path / "eval")[0][0].tasks[0].runs[0].tokens == 0


@pytest.mark.parametrize("value", ["lots", True, None, [], 1e400])
def test_a_token_field_of_any_shape_never_raises(value: object, tmp_path: Path) -> None:
    _report([{"task_id": "t", "runs": [{"attempt": 1, "input_tokens": value}]}], tmp_path)
    parsed = read_eval_reports(tmp_path / "eval")[0][0].tasks[0].runs[0]
    assert parsed.input_tokens is None
    build_dashboard(tmp_path, tmp_path / "out.html")  # and the whole build survives it


def test_half_a_measurement_counts_the_half_that_is_there() -> None:
    assert EvalRun(attempt=1, input_tokens=400).tokens == 400
    assert EvalRun(attempt=1, output_tokens=7).tokens == 7


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------


def test_the_saving_is_the_first_attempt_minus_every_repeat(tmp_path: Path) -> None:
    panel = _panel(
        [
            {
                "task_id": "t",
                "runs": [
                    {"attempt": 1, "ok": True, "llm_calls": 20, "usd": 0.40,
                     "input_tokens": 9000, "output_tokens": 1000},
                    {"attempt": 2, "ok": True, "llm_calls": 2, "usd": 0.04,
                     "input_tokens": 900, "output_tokens": 100},
                    {"attempt": 3, "ok": True, "llm_calls": 2, "usd": 0.04,
                     "input_tokens": 900, "output_tokens": 100},
                ],
            }
        ],
        tmp_path,
    )  # fmt: skip
    assert panel.saved_tokens == 2 * (10000 - 1000)
    assert panel.saved_usd == pytest.approx(2 * (0.40 - 0.04))
    assert panel.saved_calls == 2 * (20 - 2)
    assert panel.repeats == 2 and panel.token_runs == 2 and panel.usd_runs == 2


def test_a_repeat_that_failed_subtracts_instead_of_earning_a_credit(tmp_path: Path) -> None:
    panel = _panel(
        [
            {
                "task_id": "t",
                "runs": [
                    {"attempt": 1, "ok": True, "llm_calls": 20, "usd": 0.40},
                    {"attempt": 2, "ok": False, "llm_calls": 18, "usd": 0.36},
                ],
            }
        ],
        tmp_path,
    )
    assert panel.saved_calls == -18, "the run cost 18 calls and did not do the task"
    assert panel.saved_usd == pytest.approx(-0.36)
    assert panel.rows[0].spent_width > 0 and panel.rows[0].saved_width == 0, (
        "a saving that is really a loss must look like one"
    )


def test_a_task_never_solved_cold_is_excluded_and_named(tmp_path: Path) -> None:
    panel = _panel(
        [
            {
                "task_id": "never_worked",
                "task_text": "a task the agent never managed",
                "runs": [
                    {"attempt": 1, "ok": False, "llm_calls": 30, "usd": 0.6},
                    {"attempt": 2, "ok": True, "llm_calls": 1, "usd": 0.01},
                ],
            },
            {
                "task_id": "fine",
                "runs": [
                    {"attempt": 1, "ok": True, "llm_calls": 10},
                    {"attempt": 2, "ok": True, "llm_calls": 1},
                ],
            },
        ],
        tmp_path,
    )
    assert [r.task_id for r in panel.rows] == ["fine"]
    excluded = {e.task_id: e.reason for e in panel.excluded}
    assert "never_worked" in excluded
    assert "no baseline" in excluded["never_worked"]
    assert panel.saved_usd is None, "the only priced task was the excluded one"


def test_a_task_with_no_repeat_yet_is_excluded_and_named(tmp_path: Path) -> None:
    panel = _panel(
        [
            {"task_id": "once", "runs": [{"attempt": 1, "ok": True, "llm_calls": 9}]},
            {
                "task_id": "twice",
                "runs": [
                    {"attempt": 1, "ok": True, "llm_calls": 9},
                    {"attempt": 2, "ok": True, "llm_calls": 1},
                ],
            },
        ],
        tmp_path,
    )
    assert [e.task_id for e in panel.excluded] == ["once"]
    assert "nothing to price" in panel.excluded[0].reason


def test_tokens_missing_on_one_side_are_not_counted_at_all(tmp_path: Path) -> None:
    """Subtracting a measured warm run from an unmeasured cold one would invent a saving."""
    panel = _panel(
        [
            {
                "task_id": "t",
                "runs": [
                    {"attempt": 1, "ok": True, "llm_calls": 20},
                    {"attempt": 2, "ok": True, "llm_calls": 2,
                     "input_tokens": 900, "output_tokens": 100},
                ],
            }
        ],
        tmp_path,
    )  # fmt: skip
    assert panel.saved_tokens is None and panel.token_runs == 0
    assert panel.token_gap == 1
    assert panel.saved_calls == 18, "the measure that WAS recorded still counts"


def test_the_cumulative_curve_follows_the_order_the_suite_ran(tmp_path: Path) -> None:
    def task(name: str, saving: int) -> dict:
        return {
            "task_id": name,
            "runs": [
                {"attempt": 1, "ok": True, "llm_calls": saving + 1},
                {"attempt": 2, "ok": True, "llm_calls": 1},
            ],
        }

    panel = _panel([task("a", 10), task("b", 4), task("c", 6)], tmp_path)
    assert [p.task_id for p in panel.chart_points] == ["a", "b", "c"]
    assert [p.value for p in panel.chart_points] == [10.0, 14.0, 20.0]
    assert panel.chart_metric == "model calls"
    assert panel.chart_line and panel.chart_area


# --------------------------------------------------------------------------------------
# Against the committed corpus, and against a report in the harness's own shape
# --------------------------------------------------------------------------------------


def test_the_fixture_suite_totals_what_it_should() -> None:
    """Checked by hand: find_invoice 2 x ($0.42 - $0.03); calls 2 x (23-2) + (11-2) +
    (31-5) - 18 for the warm run that failed; export_csv has no repeat and is excluded."""
    panel = build_savings_panel(*read_eval_reports(FULL / "eval"))
    assert panel.saved_usd == pytest.approx(0.78)
    assert panel.saved_calls == 59
    assert panel.saved_tokens is None and panel.token_runs == 0
    assert panel.headline == "$0.78"
    assert [e.task_id for e in panel.excluded] == ["export_csv"]


def test_the_page_says_plainly_that_tokens_were_not_measured(tmp_path: Path) -> None:
    page = build_dashboard(FULL, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "not measured" in page
    assert "no usage meter was attached" in page
    assert "0 tokens" not in page and "0 token saved" not in page


def test_the_panel_labels_what_it_actually_compares(tmp_path: Path) -> None:
    """Nobody may read a per-task saving on one suite as a fleet-wide guarantee."""
    page = build_dashboard(FULL, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "What this compares:" in page
    assert "not a rate that carries over to other work or other sites" in page
    assert "task(s) excluded" in page


def test_a_report_in_the_harnesss_own_shape_adds_up(tmp_path: Path) -> None:
    """The keys here are exactly what ``RunRecord.to_json`` writes, extras included, and
    the expected numbers are computed in the test rather than read off the panel."""
    cold = {
        "attempt": 1,
        "phase": "cold",
        "ok": True,
        "wall_ms": 8.0,
        "llm_calls": 7,
        "steps": 3,
        "skill_used": None,
        "usd": 0.217,
        "run_id": "r1",
        "started_at": "2026-09-19T08:33:20Z",
        "input_tokens": 19200,
        "output_tokens": 2480,
        "decision": "cold",
        "library_size": 0,
    }
    warm = {
        "attempt": 2,
        "phase": "warm",
        "ok": True,
        "wall_ms": 3.0,
        "llm_calls": 1,
        "skill_used": "confirm_invoice_payment",
        "usd": 0.031,
        "input_tokens": 2400,
        "output_tokens": 310,
        "decision": "warm",
    }
    free = dict(warm, attempt=3, llm_calls=0, usd=0.0, input_tokens=0, output_tokens=0)
    panel = _panel(
        [{"task_id": "confirm", "task_text": "Confirm payment", "runs": [cold, warm, free]}],
        tmp_path,
    )  # fmt: skip

    cold_tokens = 19200 + 2480
    assert panel.saved_tokens == (cold_tokens - 2710) + cold_tokens
    assert panel.saved_usd == pytest.approx((0.217 - 0.031) + 0.217)
    assert panel.saved_calls == (7 - 1) + 7
    assert panel.baseline_tokens == cold_tokens
    assert panel.headline == "$0.40"
    assert panel.excluded == ()


def test_no_evaluation_data_gives_the_panels_own_empty_state(tmp_path: Path) -> None:
    page = build_dashboard(tmp_path, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "Nothing to price yet" in page
    assert collect(tmp_path).savings.empty_reason


def test_the_headline_reaches_the_masthead(tmp_path: Path) -> None:
    """Put where it cannot be missed: the first tile of the ten-second read."""
    page = build_dashboard(FULL, tmp_path / "out.html").read_text(encoding="utf-8")
    tile = collect(FULL).stats[0]
    assert tile.value == "$0.78"
    assert "saved by the library" in page


@pytest.mark.parametrize(
    ("value", "shown"),
    [(None, "not measured"), (0, "0"), (940, "940"), (12345, "12.3k"), (1_400_000, "1.40M")],
)
def test_human_count(value: float | None, shown: str) -> None:
    assert human_count(value) == shown


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (None, "not priced"),
        (0.0, "$0.0000"),
        (0.0042, "$0.0042"),
        (0.78, "$0.78"),
        (-1.5, "-$1.50"),
    ],
)
def test_human_usd(value: float | None, shown: str) -> None:
    assert human_usd(value) == shown
