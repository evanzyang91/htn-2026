"""The shipped suite, ``eval/tasks.yaml``, checked against the app it claims to test.

A task suite can be wrong in ways no amount of harness correctness catches. A check
naming a message id that is not in the seed can never pass, and would be read as "the
agent failed" forever. A task that is already satisfied the moment the app loads can
never fail, and would pad the success rate with a free point. Both are silent.

So this file loads the real suite, loads the real application's seed state - through
``apps/sandbox-site/serve.py`` itself, so the state is built the way the server builds
it rather than the way this test imagines it - and checks the suite against it:

* every task fails on a freshly reset app, so nothing passes for free;
* every record a check refers to by id actually exists in the seed;
* the suite is varied, and the tags are the ones the harness knows about.

No server is started and no port is bound: ``fresh_state()`` is a pure function of
``seed.json``.
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from skillweaver.eval.harness import TAGS, EvalTask, Suite, load_suite, resolve_path, score

ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = ROOT / "eval" / "tasks.yaml"
SERVE = ROOT / "apps" / "sandbox-site" / "serve.py"

SELECTOR = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)=([^\]]+)\]")
"""``[id=m03]`` in a check path: the field and the value it must match."""


@pytest.fixture(scope="module")
def suite() -> Suite:
    return load_suite(SUITE_PATH)


@pytest.fixture(scope="module")
def fresh() -> dict[str, Any]:
    """The sandbox's starting state, built by the server's own ``fresh_state``.

    Imported by path because ``apps/sandbox-site`` is a standalone script rather than
    a package. It imports nothing but the standard library and binds no port.
    """
    spec = importlib.util.spec_from_file_location("sandbox_serve", SERVE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fresh_state()


# --------------------------------------------------------------------------------------
# The shape of the suite
# --------------------------------------------------------------------------------------


def test_the_shipped_suite_loads(suite: Suite) -> None:
    assert suite.name == "sandbox-site"
    assert suite.domain == "sandbox.test"
    assert suite.base_url.startswith("http://")
    assert suite.warm_runs == 3, "cold once, then warm three times"


def test_the_suite_holds_between_twelve_and_fifteen_tasks(suite: Suite) -> None:
    assert 12 <= len(suite.tasks) <= 15


def test_all_three_shapes_are_genuinely_represented(suite: Suite) -> None:
    """A suite where every task is the same shape proves nothing: single-step tasks
    cannot show composition, and composite ones cannot show a clean warm hit."""
    counts = Counter(tag for task in suite.tasks for tag in task.tags)
    assert set(counts) == set(TAGS)
    for tag in TAGS:
        assert counts[tag] >= 3, f"only {counts[tag]} {tag} task(s) is not a category"


def test_the_tasks_are_distinct_errands(suite: Suite) -> None:
    assert len({t.id for t in suite.tasks}) == len(suite.tasks)
    assert len({t.text for t in suite.tasks}) == len(suite.tasks)


def test_the_tasks_reach_across_the_whole_application(suite: Suite) -> None:
    """Mail, records and settings all get exercised. A suite that only ever opened
    the mail screen would report a library that is good at one screen."""
    paths = " ".join(check.path for task in suite.tasks for check in task.expect)
    for area in ("mail.", "records.", "settings.", "ui.screen"):
        assert area in paths, f"nothing in the suite touches {area}"


def test_every_task_is_scored_by_at_least_one_check(suite: Suite) -> None:
    for task in suite.tasks:
        assert task.expect, f"{task.id} could never fail"


# --------------------------------------------------------------------------------------
# What the agent is told, and what it is not
# --------------------------------------------------------------------------------------


def test_the_task_text_never_leaks_how_the_run_will_be_scored(suite: Suite) -> None:
    """The agent is given the errand in words, not the ground-truth paths that decide
    whether it did it. A task text mentioning ``ui.screen`` or ``/__state`` would be
    handing the answer to the thing being measured."""
    for task in suite.tasks:
        lowered = task.text.lower()
        assert "__state" not in lowered and "__reset" not in lowered
        assert "ui." not in lowered
        for check in task.expect:
            assert check.path.lower() not in lowered


def test_the_task_text_reads_like_an_instruction_to_a_person(suite: Suite) -> None:
    for task in suite.tasks:
        assert len(task.text) > 15, f"{task.id}: too terse to be an errand"
        assert task.text.strip().endswith("."), f"{task.id}: not written as a sentence"


# --------------------------------------------------------------------------------------
# The suite against the real application state
# --------------------------------------------------------------------------------------


def test_no_task_passes_on_a_freshly_reset_app(suite: Suite, fresh: dict[str, Any]) -> None:
    """The one that matters most.

    A task already satisfied at startup would be scored as a success by every run,
    cold and warm alike, and would quietly add a free point to both success rates and
    a meaningless entry to the speedup table.
    """
    passing = [task.id for task in suite.tasks if score(task, fresh).ok]
    assert passing == [], f"these tasks are already done before the agent starts: {passing}"


def test_every_record_a_check_names_by_id_exists_in_the_seed(
    suite: Suite, fresh: dict[str, Any]
) -> None:
    """``mail.messages[id=m99]`` can never match, so the task can never pass, and the
    report would blame the agent forever."""
    missing: list[str] = []
    for task in suite.tasks:
        for check in task.expect:
            for field, wanted in SELECTOR.findall(check.path):
                container = check.path[: check.path.index(f"[{field}={wanted}]")]
                rows = resolve_path(fresh, container)
                if not isinstance(rows, list) or not any(
                    str(r.get(field)) == wanted for r in rows if isinstance(r, dict)
                ):
                    missing.append(f"{task.id}: {check.path}")
    assert missing == [], f"checks referring to records that do not exist: {missing}"


def test_every_check_path_starts_somewhere_real(suite: Suite, fresh: dict[str, Any]) -> None:
    """Catches a typo in the top-level area of a path (``setting.`` for ``settings.``),
    which would otherwise read as an agent that never manages to save anything."""
    for task in suite.tasks:
        for check in task.expect:
            root = SELECTOR.sub("", check.path).split(".")[0]
            assert root in fresh, f"{task.id}: {check.path} starts at unknown '{root}'"


def test_the_mutating_tasks_are_the_reason_the_reset_matters(
    suite: Suite, fresh: dict[str, Any]
) -> None:
    """Most of this suite changes stored data rather than just the current screen, so
    a run starting on the previous run's leftovers would be doing a different task.
    This is the fact the harness's reset-before-every-run exists to handle."""
    mutating = [
        task.id
        for task in suite.tasks
        if any(not check.path.startswith("ui.") for check in task.expect)
    ]
    assert len(mutating) >= len(suite.tasks) // 2, (
        f"only {len(mutating)} of {len(suite.tasks)} tasks change stored state"
    )


def test_a_task_is_scored_against_state_alone_so_the_suite_needs_no_browser(
    suite: Suite, fresh: dict[str, Any]
) -> None:
    """Scoring is a pure function of the application's state, which is what lets the
    referee be the offline teacher rather than something the agent shares."""
    task: EvalTask = next(t for t in suite.tasks if t.id == "open_records")
    assert score(task, fresh).ok is False
    done = {**fresh, "ui": {**fresh["ui"], "screen": "records"}}
    assert score(task, done).ok is True


# --------------------------------------------------------------------------------------
# params: the third shape, and the condition on the model-free number
# --------------------------------------------------------------------------------------


def test_most_tasks_declare_the_parameters_a_skill_would_take(suite: Suite) -> None:
    """Without these the suite can only measure the plain-English shape, and the
    report could quote "no model calls" without ever having measured it."""
    with_params = [t.id for t in suite.tasks if t.params]
    assert len(with_params) >= len(suite.tasks) - 3, (
        f"only {len(with_params)} of {len(suite.tasks)} tasks can be run with "
        "parameters supplied, so the third shape is barely measured"
    )


def test_every_declared_parameter_value_appears_in_the_task_text(suite: Suite) -> None:
    """A parameter is the value a caller supplies INSTEAD of the agent reading it out
    of the sentence. If it is not in the sentence, the two shapes are not the same
    errand and the comparison between them means nothing.
    """
    stray: list[str] = []
    for task in suite.tasks:
        lowered = task.text.lower()
        for key, value in task.params.items():
            if str(value).lower() not in lowered:
                stray.append(f"{task.id}: {key}={value!r} is not in the task text")
    assert stray == [], stray


def test_parameters_never_smuggle_in_the_answer(suite: Suite, fresh: dict[str, Any]) -> None:
    """Supplying parameters must make the task cheaper to READ, never easier to DO.

    A parameter naming a ground-truth path, or a record id the agent would otherwise
    have to find on screen, would turn the parameters-supplied run into a different
    and easier task - and its speedup into a measurement of the hint.
    """
    for task in suite.tasks:
        for key, value in task.params.items():
            text = str(value)
            assert not text.startswith(("ui.", "mail.", "records.", "settings.")), (
                f"{task.id}: {key} looks like a ground-truth path"
            )
            assert not SELECTOR.search(text), f"{task.id}: {key} carries a record selector"
            assert "__state" not in text and "__reset" not in text
            for check in task.expect:
                assert check.path not in text
