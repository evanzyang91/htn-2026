"""Reading ground truth, scoring a run against it, and refusing a broken suite.

Scoring is the part of the harness that decides whether the project's claim is true,
so the thing being defended here is that it cannot be talked into a yes. A path that
does not exist is not a match; a check with no operator is a suite that does not load;
a task with no checks scores ``False`` rather than passing by default.

:func:`skillweaver.eval.harness.score` takes a plain mapping, which is why none of
these tests needs an application at all - and is also why the only object that can
reach the live application stays in the harness's own hands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillweaver.errors import ConfigError
from skillweaver.eval.harness import (
    Check,
    EvalTask,
    load_suite,
    resolve_path,
    score,
)

STATE: dict[str, Any] = {
    "ui": {"screen": "mail", "mail": {"search": "", "openId": None}},
    "mail": {
        "labels": ["Work", "Personal"],
        "messages": [
            {"id": "m01", "archived": False, "labels": ["Work"], "subject": "Q3 capacity plan"},
            {"id": "m03", "archived": True, "labels": ["Finance"], "subject": "Invoice 4471"},
        ],
        "sent": [],
    },
    "settings": {"displayName": "Avery Quinn", "savedCount": 0},
}


def task(*checks: Check) -> EvalTask:
    return EvalTask(id="t", text="do the thing", tags=("single-step",), expect=checks)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def test_a_dotted_path_walks_nested_mappings() -> None:
    assert resolve_path(STATE, "ui.screen") == "mail"
    assert resolve_path(STATE, "settings.savedCount") == 0


def test_a_list_can_be_indexed_by_position() -> None:
    assert resolve_path(STATE, "mail.messages[0].id") == "m01"
    assert resolve_path(STATE, "mail.labels[1]") == "Personal"


def test_a_list_can_be_indexed_by_a_matching_field() -> None:
    """The interesting facts live in lists of records whose positions shift as the
    application is used, so a check written against position 2 would start testing a
    different message the moment a task reordered anything."""
    assert resolve_path(STATE, "mail.messages[id=m03].archived") is True
    assert resolve_path(STATE, "mail.messages[id=m01].archived") is False


def test_a_null_value_is_returned_as_null_and_a_missing_one_is_not() -> None:
    """The sandbox uses ``null`` meaningfully - ``openId`` is null when no message is
    open - so "the value there is None" and "there is nothing there" cannot be the
    same answer."""
    assert resolve_path(STATE, "ui.mail.openId") is None
    assert resolve_path(STATE, "ui.mail.nothingHere") is not None


@pytest.mark.parametrize(
    "path",
    [
        "nope",
        "ui.nope",
        "ui.screen.nope",
        "mail.messages[id=m99].archived",
        "mail.messages[9].id",
        "mail.messages[notanumber]",
        "settings[0]",
    ],
)
def test_a_path_that_does_not_resolve_never_pretends_to(path: str) -> None:
    check = Check(path=path, op="equals", value="anything")
    assert score(task(check), STATE).ok is False


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


def test_equals_compares_exactly() -> None:
    assert score(task(Check("ui.screen", "equals", "mail")), STATE).ok is True
    assert score(task(Check("ui.screen", "equals", "records")), STATE).ok is False


def test_equals_does_not_confuse_false_with_missing() -> None:
    assert score(task(Check("mail.messages[id=m01].archived", "equals", False)), STATE).ok is True


def test_contains_is_membership_for_a_list_and_substring_for_a_string() -> None:
    assert score(task(Check("mail.messages[id=m01].labels", "contains", "Work")), STATE).ok is True
    assert (
        score(task(Check("mail.messages[id=m01].labels", "contains", "Travel")), STATE).ok is False
    )
    assert (
        score(task(Check("mail.messages[id=m01].subject", "contains", "capacity")), STATE).ok
        is True
    )


def test_contains_on_something_with_no_members_fails_rather_than_raising() -> None:
    result = score(task(Check("settings.savedCount", "contains", "1")), STATE)
    assert result.ok is False
    assert "not a list or string" in result.reason


def test_count_and_at_least_measure_length() -> None:
    assert score(task(Check("mail.labels", "count", 2)), STATE).ok is True
    assert score(task(Check("mail.labels", "count", 3)), STATE).ok is False
    assert score(task(Check("mail.labels", "at_least", 1)), STATE).ok is True
    assert score(task(Check("mail.sent", "at_least", 1)), STATE).ok is False


def test_a_length_check_on_something_with_no_length_fails_rather_than_raising() -> None:
    result = score(task(Check("settings.savedCount", "at_least", 1)), STATE)
    assert result.ok is False
    assert "no length" in result.reason


def test_absent_treats_a_null_value_and_a_missing_key_alike() -> None:
    assert score(task(Check("ui.mail.openId", "absent", True)), STATE).ok is True
    assert score(task(Check("ui.mail.neverExisted", "absent", True)), STATE).ok is True
    assert score(task(Check("ui.screen", "absent", True)), STATE).ok is False
    assert score(task(Check("ui.screen", "absent", False)), STATE).ok is True


# --------------------------------------------------------------------------------------
# Scoring a whole task
# --------------------------------------------------------------------------------------


def test_every_check_must_pass() -> None:
    both = task(Check("ui.screen", "equals", "mail"), Check("settings.savedCount", "equals", 0))
    assert score(both, STATE).ok is True

    one_wrong = task(
        Check("ui.screen", "equals", "mail"), Check("settings.savedCount", "equals", 9)
    )
    assert score(one_wrong, STATE).ok is False


def test_a_task_with_no_checks_scores_false_because_nothing_was_verified() -> None:
    assert score(EvalTask(id="t", text="x"), STATE).ok is False


def test_every_check_is_evaluated_even_after_one_has_failed() -> None:
    """ "Three of five checks failed, and here they are" is more use than stopping at
    the first."""
    result = score(
        task(
            Check("ui.screen", "equals", "records"),
            Check("settings.savedCount", "equals", 9),
            Check("mail.labels", "count", 2),
        ),
        STATE,
    )
    assert len(result.results) == 3
    assert [r.ok for r in result.results] == [False, False, True]


def test_the_failure_reason_names_the_path_and_both_values() -> None:
    reason = score(task(Check("ui.screen", "equals", "records")), STATE).reason
    assert "ui.screen" in reason and "'mail'" in reason and "'records'" in reason


# --------------------------------------------------------------------------------------
# Loading a suite: strict, because a silently shortened suite flatters itself
# --------------------------------------------------------------------------------------


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(body, encoding="utf-8")
    return path


GOOD = """
suite: demo
domain: demo.test
base_url: http://127.0.0.1:9999
warm_runs: 2
tasks:
  - id: one
    text: Open the records screen.
    tags: [single-step]
    expect:
      - path: ui.screen
        equals: records
  - id: two
    text: Archive a message and label it.
    tags: [multi-step, composite]
    expect:
      - path: mail.messages[id=m03].archived
        equals: true
"""


def test_a_good_suite_loads_with_its_tasks_tags_and_settings(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, GOOD))
    assert suite.name == "demo"
    assert suite.domain == "demo.test"
    assert suite.base_url == "http://127.0.0.1:9999"
    assert suite.warm_runs == 2
    assert [t.id for t in suite.tasks] == ["one", "two"]
    assert suite.tasks[0].expect[0] == Check("ui.screen", "equals", "records")
    assert [t.id for t in suite.tagged("composite")] == ["two"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("tasks: []", "no 'tasks' list"),
        ("suite: x", "no 'tasks' list"),
        ("- just\n- a list\n", "must be a mapping at the top level"),
        ("tasks:\n  - text: no id\n    tags: [single-step]\n", "has no 'id'"),
        ("tasks:\n  - id: a\n    tags: [single-step]\n", "has no 'text'"),
        ("tasks:\n  - id: a\n    text: t\n", "must carry at least one"),
        ("tasks:\n  - id: a\n    text: t\n    tags: [quick]\n", "unknown tag"),
        ("tasks:\n  - id: a\n    text: t\n    tags: [single-step]\n", "has no 'expect'"),
    ],
)
def test_a_broken_suite_is_rejected_rather_than_quietly_shortened(
    tmp_path: Path, body: str, expected: str
) -> None:
    """A suite that dropped the three tasks it could not parse would report a
    flattering success rate over the eleven that were easy enough to spell."""
    with pytest.raises(ConfigError, match=expected):
        load_suite(write(tmp_path, body))


def test_a_check_must_name_exactly_one_operator(tmp_path: Path) -> None:
    none_named = """
tasks:
  - id: a
    text: t
    tags: [single-step]
    expect:
      - path: ui.screen
"""
    with pytest.raises(ConfigError, match="exactly one of"):
        load_suite(write(tmp_path, none_named))

    two_named = """
tasks:
  - id: a
    text: t
    tags: [single-step]
    expect:
      - path: ui.screen
        equals: records
        contains: rec
"""
    with pytest.raises(ConfigError, match="exactly one of"):
        load_suite(write(tmp_path, two_named))


def test_a_length_operator_given_a_non_integer_is_rejected(tmp_path: Path) -> None:
    body = """
tasks:
  - id: a
    text: t
    tags: [single-step]
    expect:
      - path: mail.sent
        at_least: "one"
"""
    with pytest.raises(ConfigError, match="takes an integer"):
        load_suite(write(tmp_path, body))


def test_a_duplicate_task_id_is_rejected(tmp_path: Path) -> None:
    """Two tasks with one id would overwrite each other in every report keyed by it."""
    body = (
        GOOD
        + """
  - id: one
    text: A different task wearing the same name.
    tags: [single-step]
    expect:
      - path: ui.screen
        equals: mail
"""
    )
    with pytest.raises(ConfigError, match="duplicate task id"):
        load_suite(write(tmp_path, body))


def test_a_missing_suite_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the task suite"):
        load_suite(tmp_path / "nope.yaml")


def test_a_file_that_is_not_yaml_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_suite(write(tmp_path, "tasks: [\n  unclosed"))
