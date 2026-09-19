"""``eval/wikipedia.yaml``, checked WITHOUT touching en.wikipedia.org.

The suite it describes is the one part of this project that deliberately runs against
somebody else's live website. That run is by hand, it costs money, and its result is
a fact about the internet on the day it was run. None of which belongs in ``make
test``, so this file asserts only what can be known offline:

* the file parses and validates under the same ``load_suite`` the sandbox suite uses;
* it describes live, read-only Wikipedia rather than the sandbox by accident;
* the checks are written in the five operators the harness already has, against a
  state shape a referee over a live page can actually produce;
* and malformed variants of it are REJECTED - because the value of a strict loader is
  entirely in what it refuses, and a suite this file only ever fed valid YAML to
  would be tested for nothing.

The last point is why the rejection tests build their variants by mutating the real
file rather than by inventing a small one: a mistake somebody makes while editing
this suite is an edit to this suite.

Nothing here opens a socket. ``load_suite`` reads a file and parses YAML; the URLs in
it are strings until a browser is pointed at them.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from skillweaver.errors import ConfigError
from skillweaver.eval.harness import OPERATORS, TAGS, Suite, load_suite, score

ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = ROOT / "eval" / "wikipedia.yaml"

STATE_ROOTS = ("url", "headings", "links")
"""The top-level keys a referee over a live page can supply; see the suite's own
comments. A check reaching outside these could never be evaluated."""


@pytest.fixture(scope="module")
def suite() -> Suite:
    return load_suite(SUITE_PATH)


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    return yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))


def _written(tmp_path: Path, document: Any, name: str = "variant.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# It loads, and it is the suite it says it is
# --------------------------------------------------------------------------------------


def test_the_wikipedia_suite_loads(suite: Suite) -> None:
    assert suite.name == "wikipedia"
    assert suite.domain == "en.wikipedia.org"
    assert suite.base_url == "https://en.wikipedia.org"
    assert suite.source.endswith("wikipedia.yaml")


def test_it_is_a_handful_of_tasks_not_a_bill(suite: Suite) -> None:
    """Every task is three shapes of live traffic against a site we do not own."""
    assert 4 <= len(suite.tasks) <= 8
    assert suite.warm_runs <= 2, "warm variance is not worth somebody else's bandwidth"


def test_it_is_a_second_suite_and_not_a_rewrite_of_the_first(suite: Suite) -> None:
    sandbox = load_suite(ROOT / "eval" / "tasks.yaml")
    assert suite.domain != sandbox.domain
    assert not {t.id for t in suite.tasks} & {t.id for t in sandbox.tasks}


def test_every_task_is_search_or_navigation(suite: Suite) -> None:
    """The brief's ask, in one assertion: this suite is about getting around a real
    encyclopedia, so every task must be scored on where the browser ended up."""
    for task in suite.tasks:
        assert any(check.path == "url" for check in task.expect), task.id


def test_the_shapes_the_harness_reports_on_are_all_present(suite: Suite) -> None:
    counts = Counter(tag for task in suite.tasks for tag in task.tags)
    assert set(counts) <= set(TAGS)
    for tag in TAGS:
        assert counts[tag] >= 1, f"no {tag} task, so that column of the report is empty"


def test_the_tasks_are_distinct_errands(suite: Suite) -> None:
    assert len({t.id for t in suite.tasks}) == len(suite.tasks)
    assert len({t.text for t in suite.tasks}) == len(suite.tasks)


def test_a_parameterised_search_is_asked_for_twice_with_different_terms(
    suite: Suite,
) -> None:
    """The measured failure was a replay with a DIFFERENT query timing out. A suite
    that searched for one term could not have caught it, and cannot show it fixed."""
    queries = {str(task.params["query"]) for task in suite.tasks if "query" in task.params}
    assert len(queries) >= 2, f"only ever searches for {queries}"


# --------------------------------------------------------------------------------------
# Read-only, and honest about its reset
# --------------------------------------------------------------------------------------


def test_nothing_in_the_suite_asks_wikipedia_to_change(suite: Suite) -> None:
    """A task that mutated state would need an undo, and there is no undoing
    Wikipedia. See WorldReset in orchestrator.py."""
    forbidden = ("edit", "create", "delete", "save", "publish", "log in", "sign in")
    for task in suite.tasks:
        lowered = task.text.lower()
        assert not any(word in lowered for word in forbidden), task.id


def test_the_reset_is_a_real_page_rather_than_a_sandbox_endpoint(suite: Suite) -> None:
    """Left at the default, every run would GET https://en.wikipedia.org/__reset,
    take a 404, and record reset_ok=False for a world that needed no resetting."""
    assert suite.reset_path != "/__reset"
    assert suite.reset_path.startswith("/wiki/")


def test_every_start_path_is_an_article_or_the_main_page(suite: Suite) -> None:
    for task in suite.tasks:
        assert task.start_path.startswith("/wiki/"), task.id


# --------------------------------------------------------------------------------------
# The checks: existing vocabulary, reachable state, no free points
# --------------------------------------------------------------------------------------


def test_no_new_check_vocabulary_was_invented(suite: Suite) -> None:
    for task in suite.tasks:
        for check in task.expect:
            assert check.op in OPERATORS, f"{task.id}: {check.describe()}"


def test_every_check_path_is_something_a_live_referee_can_answer(suite: Suite) -> None:
    for task in suite.tasks:
        for check in task.expect:
            root = check.path.split(".")[0].split("[")[0]
            assert root in STATE_ROOTS, f"{task.id}: {check.path} is not in {STATE_ROOTS}"


def test_no_task_scores_against_an_empty_page(suite: Suite) -> None:
    """A run that crashed before navigating leaves nothing behind. If a task passed
    on that, the suite would be handing out points for failure."""
    nothing: dict[str, Any] = {"url": "", "headings": [], "links": []}
    for task in suite.tasks:
        assert not score(task, nothing).ok, task.id


def test_no_task_passes_merely_by_sitting_on_its_start_page(suite: Suite) -> None:
    """Every task must require the agent to GO somewhere."""
    for task in suite.tasks:
        start = {
            "url": f"https://en.wikipedia.org{task.start_path}",
            "headings": [],
            "links": [],
        }
        assert not score(task, start).ok, task.id


def test_a_task_passes_when_the_browser_really_did_land_there(suite: Suite) -> None:
    """The mirror image: the checks are satisfiable, so a green run is possible.

    Built from the task's own expectations rather than from a hand-written page, so
    this stays true when somebody adds a task.
    """
    for task in suite.tasks:
        state: dict[str, Any] = {"url": "https://en.wikipedia.org", "headings": [], "links": []}
        for check in task.expect:
            if check.path == "url":
                state["url"] += str(check.value)
            else:
                state[check.path].append(check.value)
        assert score(task, state).ok, f"{task.id} cannot be passed: {score(task, state).reason}"


def test_the_url_checks_survive_a_tracking_parameter(suite: Suite) -> None:
    """Wikipedia's search suggestions add ?wprov=...; a check that broke on that
    would be measuring analytics rather than this agent."""
    for task in suite.tasks:
        state: dict[str, Any] = {"url": "https://en.wikipedia.org", "headings": [], "links": []}
        for check in task.expect:
            if check.path == "url":
                state["url"] += str(check.value)
            else:
                state[check.path].append(check.value)
        state["url"] += "?wprov=acrw1_0"
        assert score(task, state).ok, task.id


def test_the_task_text_never_leaks_the_url_it_is_scored_on(suite: Suite) -> None:
    """The agent is given `text` and nothing else. If the answer is written in it,
    the task measures reading comprehension rather than navigation."""
    for task in suite.tasks:
        for check in task.expect:
            if check.path == "url":
                assert str(check.value) not in task.text, task.id


# --------------------------------------------------------------------------------------
# It says out loud that it hits the internet
# --------------------------------------------------------------------------------------


def test_the_file_warns_that_it_is_not_part_of_make_test() -> None:
    """The one thing a reader must not have to discover by being billed for it."""
    comments = "\n".join(
        line for line in SUITE_PATH.read_text(encoding="utf-8").splitlines() if line.startswith("#")
    ).lower()
    assert "live" in comments
    assert "internet" in comments
    assert "make test" in comments


def test_no_test_in_this_repository_runs_the_suite_for_real() -> None:
    """A grep, deliberately: the protection is that nobody wired it up, and the way
    to keep it is to notice when somebody does."""
    for path in (ROOT / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "wikipedia.yaml" in text:
            assert path.name == "test_wikipedia_suite.py", (
                f"{path} mentions the live suite; it must not run it"
            )
    import tests.eval.test_wikipedia_suite as this

    for name in ("run", "run_suite", "run_task", "HttpReferee", "BrowserController"):
        assert not hasattr(this, name), (
            f"this module imported {name}; it is supposed to read a file, not drive a browser"
        )


# --------------------------------------------------------------------------------------
# What the loader refuses
# --------------------------------------------------------------------------------------


def test_a_task_with_no_checks_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    del broken["tasks"][0]["expect"]
    with pytest.raises(ConfigError, match="no 'expect' checks"):
        load_suite(_written(tmp_path, broken))


def test_a_check_naming_two_operators_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"][0]["expect"][0]["equals"] = "https://en.wikipedia.org"
    with pytest.raises(ConfigError, match="must name exactly one of"):
        load_suite(_written(tmp_path, broken))


def test_a_check_naming_no_operator_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"][0]["expect"][0] = {"path": "url"}
    with pytest.raises(ConfigError, match="must name exactly one of"):
        load_suite(_written(tmp_path, broken))


def test_a_check_with_no_path_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"][0]["expect"][0].pop("path")
    with pytest.raises(ConfigError, match="has no 'path'"):
        load_suite(_written(tmp_path, broken))


def test_a_duplicate_task_id_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"].append(copy.deepcopy(broken["tasks"][0]))
    with pytest.raises(ConfigError, match="duplicate task id"):
        load_suite(_written(tmp_path, broken))


def test_a_task_with_no_text_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"][1]["text"] = "   "
    with pytest.raises(ConfigError, match="no 'text' to give the agent"):
        load_suite(_written(tmp_path, broken))


def test_an_unknown_tag_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    """'navigation' is exactly the tag somebody will reach for while editing this
    file, and a fourth category nobody reports on is worse than no tag at all."""
    broken = copy.deepcopy(raw)
    broken["tasks"][0]["tags"] = ["navigation"]
    with pytest.raises(ConfigError, match="unknown tag"):
        load_suite(_written(tmp_path, broken))


def test_params_that_are_not_a_mapping_are_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"][0]["params"] = ["Charles Babbage"]
    with pytest.raises(ConfigError, match="'params' must be a mapping"):
        load_suite(_written(tmp_path, broken))


def test_a_negative_warm_run_count_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["warm_runs"] = -1
    with pytest.raises(ConfigError, match="'warm_runs' must be a non-negative integer"):
        load_suite(_written(tmp_path, broken))


def test_a_suite_with_no_tasks_is_rejected(raw: dict[str, Any], tmp_path: Path) -> None:
    broken = copy.deepcopy(raw)
    broken["tasks"] = []
    with pytest.raises(ConfigError, match="has no 'tasks' list"):
        load_suite(_written(tmp_path, broken))


def test_a_file_that_is_not_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("tasks: [\n  - id: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_suite(path)


def test_a_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the task suite"):
        load_suite(tmp_path / "absent.yaml")
