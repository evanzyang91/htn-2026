"""The evaluation harness: run every task cold once and warm N times, and measure it.

This is the module that turns the project's claim - *the agent gets permanently
faster at things it has done before* - into numbers somebody else can check. It runs
a suite of tasks against a real application, scores each run against ground truth the
agent cannot see, and writes a machine-readable report plus a markdown summary.

    load_suite(eval/tasks.yaml)
            |
            v
    for each task:  reset -> COLD run    -> score   (explore; learn a skill)
                    reset -> WARM run 1  -> score   (the sentence alone)
                    reset -> WARM run 2  -> score
                    reset -> WARM run 3  -> score
                    reset -> BOUND run   -> score   (its parameters supplied too)
            |
            v
    data/eval/<stamp>.json   the shape skillweaver.dashboard.build reads
    data/eval/<stamp>.md     the same thing for a human

The reset, and why it is the load-bearing part
----------------------------------------------

Every run - including the first - starts by putting the application back to a
byte-identical state through :meth:`Referee.reset`. This is not tidiness. Nine of the
fourteen tasks in the shipped suite MUTATE state: they archive a message, rename a
row, save a setting. A second run that starts on the wreckage of the first is not
repeating the task, it is doing a different and usually easier one - "archive the
Billing message" is free when the first run already archived it - and the resulting
speedup would be a measurement of leftovers. That failure is silent, which is what
makes it dangerous, so :func:`run_task` resets before every single run and the report
records ``reset_ok`` per run so a reset that did not happen is visible rather than
assumed.

:class:`Referee` is a general seam, not a sandbox-specific one: anything that can put
its world back and describe the result satisfies it, and
:class:`HttpReferee` is merely the implementation that speaks to
``apps/sandbox-site``. The admission gate in ``skills/synthesize.py`` needs the same
"put the world back" capability, and this Protocol is offered as the shared shape for
it rather than a second competing mechanism.

The ground-truth boundary
-------------------------

:class:`Referee` is an **OFFLINE TEACHER**. It reads the application's authoritative
state - for the sandbox, the server's own ``GET /__state``, which is exact rather than
inferred - and the harness uses it to decide whether a run reached the goal. The agent
must never see it, and in this module it cannot: the referee is a local in
:func:`run_task`, it is never placed on the :class:`~skillweaver.contracts.TaskSpec`,
never passed into ``workbench.session``, and never reachable from the ``Agent`` the
session yields. :func:`score` is a free function taking a plain state mapping, so the
only object holding the door open is the one the harness keeps to itself.

Because of that separation the harness can - and does - disagree with the agent. Each
run records ``ok`` (what the referee saw) and ``agent_claimed`` (what the agent's own
critic believed). ``ok`` is what counts. Where the two differ the summary names the
task, because a critic that reports success on a task that did not happen is a more
valuable finding than any speedup.

Reading a number out of this report
-----------------------------------

Three things can make the headline figure meaningless, so the harness detects all
three itself rather than leaving them to be noticed later:

* **Nothing entered the library.** If synthesis never stores a skill, every "warm" run
  quietly explores from scratch, the suite reports a speedup near 1.0, and everything
  looks like it worked. :attr:`~skillweaver.eval.metrics.SuiteMetrics.library_was_used`
  is ``False`` in that case and the summary opens with a warning, not a table.
* **A failed run is not a fast run.** Timings come from successful runs only; see
  :mod:`skillweaver.eval.metrics`.
* **A warm run that was slower** is reported as a regression by name. It is not
  averaged away.

And one thing that would make it too flattering: **"no model calls" has a condition.**
A warm run is free only once the skill's parameters are known; asked in plain English
it still spends one small call reading them out of the sentence. Both are measured -
see the three shapes in :func:`run_task` - and the summary prints them in one table so
the zero is never quoted without what it costs to get there.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Protocol, runtime_checkable

import yaml

from skillweaver.contracts import Budget, Usage, utcnow
from skillweaver.errors import ConfigError, SkillWeaverError
from skillweaver.eval.metrics import (
    RunPoint,
    SuiteMetrics,
    TaskMetrics,
    suite_metrics,
    task_metrics,
)
from skillweaver.logging_ import get_logger
from skillweaver.orchestrator import task_spec, world_reset_from_url

__all__ = [
    "Check",
    "EvalTask",
    "HttpReferee",
    "Referee",
    "RunRecord",
    "SCHEMA_VERSION",
    "Suite",
    "TaskRecord",
    "WARM_RUNS",
    "load_suite",
    "render_summary",
    "resolve_path",
    "run",
    "run_suite",
    "run_task",
    "score",
    "write_report",
]

log = get_logger(__name__)

SCHEMA_VERSION = 1
"""The report format version, as read by :mod:`skillweaver.dashboard.build`."""

WARM_RUNS = 3
"""Warm runs per task when nobody says otherwise. Three, not one: a single warm run
cannot distinguish a library that is reliably fast from one that got lucky once, and
the variance across three is what the ``warm_ok_rate`` column reports."""

BOUND_RUNS = 1
"""Parameters-supplied warm runs per task that declares ``params``. One, not three:
this shape exists to establish that the model-free number is real, and its variance
matters far less than the plain-English shape a user actually types."""

Binding = Literal["plain", "bound"]
"""How the task reached the agent. ``"plain"`` is the sentence alone, which the warm
path must still read to bind a skill's parameters; ``"bound"`` supplies them, which is
what makes a warm run cost nothing at all."""

DEFAULT_SUITE = Path("eval/tasks.yaml")
"""The shipped suite, relative to the repository root."""

TAGS = ("single-step", "multi-step", "composite")
"""The shapes a task may be tagged with. A suite is rejected if it uses another, so a
typo cannot quietly create a fourth category nobody reports on."""

OPERATORS = ("equals", "contains", "count", "at_least", "absent")
"""The check vocabulary. Deliberately tiny - see ``eval/tasks.yaml`` for what each
means. A check naming no operator, or two, is a broken suite and is rejected at load."""

_MISSING = object()
"""Distinguishes "no value at that path" from "the value there is ``None``". The
sandbox uses ``null`` meaningfully - ``ui.mail.openId`` is null when no message is
open - so the two cannot be conflated."""


# --------------------------------------------------------------------------------------
# The suite: what gets run, and what counts as having done it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    """One assertion about the application's state after a run.

    ``path`` is the dotted path into the state mapping (see :func:`resolve_path`),
    ``op`` the operator and ``value`` its argument. Held as data rather than a closure
    so a failed check can explain itself in the report.
    """

    path: str
    op: Literal["equals", "contains", "count", "at_least", "absent"]
    value: Any

    def describe(self) -> str:
        """The check as one line, for the report's ``detail`` field."""
        return f"{self.path} {self.op} {self.value!r}"


@dataclass(frozen=True, slots=True)
class EvalTask:
    """One task in the suite: what to ask for, and how to know it was done.

    Attributes:
        id: Stable identifier, used as ``task_id`` in the report and to correlate
            runs across evaluations.
        text: The natural-language instruction handed to the agent. This is ALL the
            agent gets; ``expect`` is never shown to it.
        tags: One or more of :data:`TAGS`.
        expect: Every check that must pass for the run to count as a success.
        params: Extra task parameters passed through to the
            :class:`~skillweaver.contracts.TaskSpec`.
        start_path: Path on the suite's ``base_url`` the run starts at.
    """

    id: str
    text: str
    tags: tuple[str, ...] = ()
    expect: tuple[Check, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    start_path: str = "/"


@dataclass(frozen=True, slots=True)
class Suite:
    """A loaded ``tasks.yaml``: the tasks plus where to run them."""

    tasks: tuple[EvalTask, ...]
    name: str = "sandbox-site"
    domain: str = "sandbox.test"
    base_url: str = "http://127.0.0.1:8765"
    reset_path: str = "/__reset"
    warm_runs: int = WARM_RUNS
    bound_runs: int = BOUND_RUNS
    source: str = ""

    def tagged(self, tag: str) -> tuple[EvalTask, ...]:
        """Every task carrying ``tag``, in suite order."""
        return tuple(t for t in self.tasks if tag in t.tags)


def load_suite(path: Path | str | None = None) -> Suite:
    """Read a task suite from YAML.

    Strict on purpose. A suite is the definition of what this project claims to do,
    so a malformed one is an error rather than a silently shortened evaluation: an
    unknown tag, a check with no operator or two, a duplicate task id and a missing
    ``expect`` all raise instead of being skipped. A suite that quietly drops the
    three tasks it could not parse would report a flattering success rate over the
    eleven that were easy enough to spell correctly.

    Args:
        path: The YAML file. ``None`` uses :data:`DEFAULT_SUITE`.

    Returns:
        The parsed :class:`Suite`.

    Raises:
        ConfigError: if the file is missing, is not YAML, or does not describe a
            valid suite.
    """
    resolved = Path(path) if path is not None else DEFAULT_SUITE
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read the task suite at {resolved}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"the task suite at {resolved} is not valid YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"the task suite at {resolved} must be a mapping at the top level")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ConfigError(f"the task suite at {resolved} has no 'tasks' list")

    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_tasks):
        task = _task_from(item, index, resolved)
        if task.id in seen:
            raise ConfigError(f"{resolved}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)

    warm = _count(raw, "warm_runs", WARM_RUNS, resolved)

    return Suite(
        tasks=tuple(tasks),
        name=str(raw.get("suite") or resolved.stem),
        domain=str(raw.get("domain") or "sandbox.test"),
        base_url=str(raw.get("base_url") or "http://127.0.0.1:8765").rstrip("/"),
        reset_path=str(raw.get("reset_path") or "/__reset"),
        warm_runs=warm,
        bound_runs=_count(raw, "bound_runs", BOUND_RUNS, resolved),
        source=str(resolved),
    )


def _count(raw: Mapping[str, Any], key: str, default: int, source: Path) -> int:
    """A non-negative integer setting, or a ``ConfigError`` naming it."""
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{source}: '{key}' must be a non-negative integer")
    return value


def _task_from(item: Any, index: int, source: Path) -> EvalTask:
    """One ``tasks:`` entry, validated. Raises ``ConfigError`` naming the entry."""
    where = f"{source}: task #{index + 1}"
    if not isinstance(item, Mapping):
        raise ConfigError(f"{where} is not a mapping")
    task_id = str(item.get("id") or "").strip()
    if not task_id:
        raise ConfigError(f"{where} has no 'id'")
    text = str(item.get("text") or "").strip()
    if not text:
        raise ConfigError(f"{where} ({task_id}) has no 'text' to give the agent")

    raw_tags = item.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ConfigError(f"{where} ({task_id}) must carry at least one of {TAGS}")
    tags = tuple(str(t) for t in raw_tags)
    unknown = [t for t in tags if t not in TAGS]
    if unknown:
        raise ConfigError(f"{where} ({task_id}) has unknown tag(s) {unknown}; allowed: {TAGS}")

    raw_expect = item.get("expect")
    if not isinstance(raw_expect, list) or not raw_expect:
        raise ConfigError(
            f"{where} ({task_id}) has no 'expect' checks, so nothing could score it. "
            "A task nobody can fail is not evidence of anything."
        )
    checks = tuple(
        _check_from(c, task_id, position, source) for position, c in enumerate(raw_expect)
    )

    params = item.get("params") or {}
    if not isinstance(params, Mapping):
        raise ConfigError(f"{where} ({task_id}): 'params' must be a mapping")

    return EvalTask(
        id=task_id,
        text=text,
        tags=tags,
        expect=checks,
        params=dict(params),
        start_path=str(item.get("start_path") or "/"),
    )


def _check_from(raw: Any, task_id: str, position: int, source: Path) -> Check:
    """One ``expect:`` entry, validated. Exactly one operator, always a path."""
    where = f"{source}: {task_id} check #{position + 1}"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} is not a mapping")
    path = str(raw.get("path") or "").strip()
    if not path:
        raise ConfigError(f"{where} has no 'path'")
    present = [op for op in OPERATORS if op in raw]
    if len(present) != 1:
        raise ConfigError(
            f"{where} must name exactly one of {OPERATORS}, found {present or 'none'}"
        )
    op = present[0]
    value = raw[op]
    if op in ("count", "at_least") and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{where}: '{op}' takes an integer, got {value!r}")
    return Check(path=path, op=op, value=value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Ground truth: the offline teacher, and the scoring that reads it
# --------------------------------------------------------------------------------------


@runtime_checkable
class Referee(Protocol):
    """Puts the world back, and says what is true about it. **OFFLINE TEACHER.**

    This is the evaluation's ground-truth source, and it exists under exactly the same
    rule as :class:`~skillweaver.contracts.GroundTruthSource`: the agent's action path
    - perceiver, explorer, planner, skill runner, skill code - MUST NEVER reach it.
    Scoring a run against perfect knowledge is legitimate; letting the agent act on
    perfect knowledge would make every number in the report a measurement of nothing.
    The harness keeps its referee in a local variable and hands it to no one.

    The seam is deliberately general - reset the world, describe the world - so that
    anything resettable can be evaluated, and so that the admission gate's need to put
    the world back can be served by the same shape rather than a competing one.
    """

    def reset(self) -> None:
        """Return the application to a byte-identical starting state.

        Called before EVERY run, so that each run measures the task rather than the
        previous run's leftovers.

        Raises:
            SkillWeaverError: if the world could not be reset. The harness treats this
                as a failed run rather than continuing against a dirty state.
        """
        ...

    def state(self) -> Mapping[str, Any]:
        """The application's authoritative state, as plain JSON-like data.

        Raises:
            SkillWeaverError: if the state cannot be read.
        """
        ...


class HttpReferee:
    """A :class:`Referee` over ``apps/sandbox-site``'s own control endpoints.

    The reset endpoint rebuilds the server's state from ``seed.json`` and
    ``GET /__state`` returns it. Both are server-side and exact: the sandbox routes
    every mutation through ``POST /api/act``, so its state is a precise description of
    what is on screen rather than something scraped back out of the DOM. That makes it
    a far better referee than reading pixels, and an absolutely disqualifying thing to
    let the agent see.

    **The reset is not this class's own invention.** It is
    :func:`~skillweaver.orchestrator.world_reset_from_url`, the very callable the
    admission gate uses to put a world back before re-running a candidate skill. One
    mechanism, used from both ends: if the evaluation can reset this application then
    so can the gate, and a reset that starts failing fails visibly in both places at
    once instead of the two drifting apart.

    Args:
        base_url: Where the sandbox is serving, e.g. ``http://127.0.0.1:8765``.
        reset_path: The endpoint that restores the seed state.
        timeout: Seconds to wait on either endpoint.
    """

    __slots__ = ("_base", "_reset", "_reset_url", "_timeout")

    def __init__(
        self, base_url: str, *, reset_path: str = "/__reset", timeout: float = 10.0
    ) -> None:
        self._base = base_url.rstrip("/")
        self._reset_url = f"{self._base}{reset_path}"
        self._reset = world_reset_from_url(self._reset_url, timeout=timeout)
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"HttpReferee({self._base!r})"

    @property
    def reset_url(self) -> str:
        """The endpoint that restores this world, for handing to a task that must be
        learned: a state-changing task cannot be admitted without one."""
        return self._reset_url

    def reset(self) -> None:
        """Restore the seed state. Raises :class:`SkillWeaverError` if it does not.

        ``world_reset_from_url`` raises ``OSError`` by contract; that is translated
        here into the project's own error carrying the sentence a stranger can act on.
        """
        try:
            self._reset()
        except OSError as exc:
            raise SkillWeaverError(
                f"the sandbox at {self._base} did not answer the reset at "
                f"{self._reset_url}: {exc}. "
                "Start it with: python3 apps/sandbox-site/serve.py --port 8765"
            ) from exc

    def state(self) -> Mapping[str, Any]:
        """GET ``/__state``, parsed."""
        payload = self._call("/__state", method="GET")
        if not isinstance(payload, Mapping):
            raise SkillWeaverError(f"{self._base}/__state did not return an object")
        return payload

    def _call(self, path: str, *, method: str) -> Any:
        url = f"{self._base}{path}"
        request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, OSError) as exc:
            raise SkillWeaverError(
                f"the sandbox at {self._base} did not answer {method} {path}: {exc}. "
                "Start it with: python3 apps/sandbox-site/serve.py --port 8765"
            ) from exc
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise SkillWeaverError(f"{url} did not return JSON: {exc}") from exc


def resolve_path(state: Any, path: str) -> Any:
    """Follow a dotted path into a JSON-like structure, or return the missing sentinel.

    Two kinds of index are understood, both written inside square brackets on a
    segment::

        mail.messages[id=m03].archived    the first list element whose "id" is "m03"
        mail.sent[0].subject              the element at that position

    The key-match form exists because the interesting facts about the sandbox live in
    lists of records whose positions shift as the application is used, and a check
    written against position 2 would start testing a different message the moment a
    task reordered anything.

    Returns the module's ``_MISSING`` sentinel - not ``None`` - when the path does not
    resolve, so that a genuinely-null value can be told apart from an absent one.
    """
    current: Any = state
    for segment in path.split("."):
        name, _, index = segment.partition("[")
        if name:
            if not isinstance(current, Mapping) or name not in current:
                return _MISSING
            current = current[name]
        if index:
            current = _index_into(current, index.rstrip("]"))
            if current is _MISSING:
                return _MISSING
    return current


def _index_into(current: Any, selector: str) -> Any:
    """``[0]`` by position, ``[key=value]`` by first match. Missing sentinel if neither."""
    if not isinstance(current, Sequence) or isinstance(current, str | bytes):
        return _MISSING
    key, sep, wanted = selector.partition("=")
    if sep:
        for element in current:
            if isinstance(element, Mapping) and str(element.get(key)) == wanted:
                return element
        return _MISSING
    try:
        return current[int(selector)]
    except (ValueError, IndexError):
        return _MISSING


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check, and whether the finished application satisfied it."""

    check: Check
    ok: bool
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"check": self.check.describe(), "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Score:
    """The referee's verdict on one run: did it do the task, and how is that known."""

    ok: bool
    results: tuple[CheckResult, ...] = ()

    @property
    def reason(self) -> str:
        """One line naming what failed, or confirming what passed."""
        if self.ok:
            return f"all {len(self.results)} ground-truth check(s) passed"
        failed = [r for r in self.results if not r.ok]
        return "; ".join(r.detail for r in failed[:3]) or "no checks were run"

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "checks": [r.to_json() for r in self.results]}


def score(task: EvalTask, state: Mapping[str, Any]) -> Score:
    """Judge a finished run against ground truth.

    A free function over a plain mapping rather than a method on the referee, so that
    the scoring rules can be tested without a world at all - and so the only object
    that can reach the live application stays in the harness's own hands.

    Every check is evaluated even after one has failed: a report that says "three of
    five checks failed, and here they are" is more use than one that stops at the
    first. A task with no checks scores ``False``, because nothing was verified.

    Args:
        task: The task, carrying its ``expect`` checks.
        state: The application's ground-truth state, from :meth:`Referee.state`.
    """
    results = tuple(_evaluate(check, state) for check in task.expect)
    return Score(ok=bool(results) and all(r.ok for r in results), results=results)


def _evaluate(check: Check, state: Mapping[str, Any]) -> CheckResult:
    """Apply one check, and phrase the failure so a human can act on it."""
    found = resolve_path(state, check.path)

    if check.op == "absent":
        absent = found is _MISSING or found is None
        ok = absent if check.value else not absent
        return CheckResult(check, ok, f"{check.path} is {'absent' if absent else 'present'}")

    if found is _MISSING:
        return CheckResult(check, False, f"{check.path} is not present in the application state")

    if check.op == "equals":
        ok = found == check.value
        return CheckResult(check, ok, f"{check.path} is {found!r}, expected {check.value!r}")

    if check.op == "contains":
        if isinstance(found, str):
            ok = str(check.value) in found
        elif isinstance(found, Sequence):
            ok = check.value in found
        else:
            return CheckResult(
                check, False, f"{check.path} is {type(found).__name__}, not a list or string"
            )
        return CheckResult(
            check, ok, f"{check.path} is {found!r}, expected it to contain {check.value!r}"
        )

    size = len(found) if isinstance(found, Sequence | Mapping) else None
    if size is None:
        return CheckResult(
            check, False, f"{check.path} is {type(found).__name__}, which has no length"
        )
    ok = size == check.value if check.op == "count" else size >= check.value
    return CheckResult(
        check, ok, f"{check.path} has {size} item(s), expected {check.op} {check.value}"
    )


# --------------------------------------------------------------------------------------
# What one run, and one task, cost
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One measured run, with everything needed to distrust it.

    ``ok`` is the REFEREE's verdict and is what every metric is computed from.
    ``agent_claimed`` is what the agent's own critic believed; the two differing is a
    finding, and :func:`render_summary` names the task when they do.

    Attributes:
        attempt: Counts from ``1``; attempt ``1`` is the cold run.
        phase: ``"cold"`` or ``"warm"``.
        binding: ``"plain"`` when the agent got only the sentence and had to read the
            skill's parameters out of it, ``"bound"`` when they were supplied. The
            difference is normally one model call, and it is the difference between
            "almost free" and "free".
        ok: Whether ground truth says the task was done.
        agent_claimed: Whether the agent said it did the task.
        wall_ms: Wall-clock milliseconds spent inside ``agent.run``. Excludes opening
            the browser and the reset, which are fixed overhead that would swamp the
            comparison; ``session_ms`` carries the whole thing for transparency.
        session_ms: Milliseconds for the whole session, world setup included.
        steps: Controller actions performed.
        llm_calls: Model calls made.
        input_tokens / output_tokens: Tokens, when a meter was supplied. ``None``
            means nobody was counting - not that the answer was zero.
        usd: Model spend in US dollars.
        skill_used: The stored skill that carried the run, or ``None`` when it
            explored. ``None`` on a WARM run means the run was warm in name only.
        decision: Which path the orchestrator says answered: warm, cold or none.
        rescued: A stored skill ran, was wrong, and exploration saved the run.
        learned: The skill this run added to the library, if any.
        library_size: Skills stored for this domain when the run finished.
        reset_ok: Whether the pre-run reset succeeded. ``False`` means this run
            started on the previous run's leftovers and its timing is not comparable.
        error: The exception that ended the run, when one did.
        score: The referee's per-check detail.
    """

    attempt: int
    phase: Literal["cold", "warm"]
    ok: bool
    binding: Binding = "plain"
    agent_claimed: bool = False
    wall_ms: float = 0.0
    session_ms: float = 0.0
    steps: int = 0
    llm_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float = 0.0
    skill_used: str | None = None
    decision: str = "none"
    rescued: bool = False
    run_id: str | None = None
    learned: str | None = None
    library_size: int = 0
    reset_ok: bool = True
    error: str = ""
    started_at: datetime = field(default_factory=utcnow)
    score: Score | None = None

    @property
    def agreed(self) -> bool:
        """Whether the agent's own verdict matched the referee's."""
        return self.ok == self.agent_claimed

    def to_json(self) -> dict[str, Any]:
        """The run in the report format :mod:`skillweaver.dashboard.build` reads.

        The first block of keys is exactly what the dashboard consumes; everything
        after it is extra, which that reader ignores by design.
        """
        return {
            "attempt": self.attempt,
            "ok": self.ok,
            "wall_ms": round(self.wall_ms, 1),
            "llm_calls": self.llm_calls,
            "steps": self.steps,
            "skill_used": self.skill_used,
            "usd": round(self.usd, 6),
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat().replace("+00:00", "Z"),
            # -- beyond the dashboard's format, for anyone auditing the numbers --
            "phase": self.phase,
            "binding": self.binding,
            "agent_claimed": self.agent_claimed,
            "agreed": self.agreed,
            "used_library": self.skill_used is not None,
            "session_ms": round(self.session_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "decision": self.decision,
            "rescued": self.rescued,
            "learned": self.learned,
            "library_size": self.library_size,
            "reset_ok": self.reset_ok,
            "error": self.error,
            "score": None if self.score is None else self.score.to_json(),
        }

    def point(self) -> RunPoint:
        """This run reduced to the numbers :mod:`skillweaver.eval.metrics` works on."""
        return RunPoint(
            attempt=self.attempt,
            ok=self.ok,
            wall_ms=self.wall_ms,
            llm_calls=self.llm_calls,
            usd=self.usd,
            used_library=self.skill_used is not None,
        )


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """Every run of one task, plus the comparison they add up to."""

    task: EvalTask
    runs: tuple[RunRecord, ...] = ()

    @property
    def metrics(self) -> TaskMetrics:
        """Cold versus warm, in the shape a user actually types: the sentence alone.

        The parameters-supplied runs are deliberately EXCLUDED. They are the faster
        shape, and folding them into the headline would let the report quote a number
        nobody gets by asking in English.
        """
        return task_metrics(self.task.id, (r.point() for r in self._by("plain")))

    @property
    def bound_metrics(self) -> TaskMetrics | None:
        """The same comparison for the parameters-supplied runs, or ``None``.

        ``None`` when the task declares no parameters, so there is no second shape to
        report - not a zero, which would read as "supplying them did not help".
        """
        bound = [r for r in self.runs if r.binding == "bound"]
        if not bound:
            return None
        cold = [r for r in self.runs if r.phase == "cold"]
        return task_metrics(self.task.id, (r.point() for r in cold + bound))

    def _by(self, binding: Binding) -> list[RunRecord]:
        """The cold run plus the warm runs of one binding shape, in attempt order."""
        return [r for r in self.runs if r.phase == "cold" or r.binding == binding]

    def to_json(self) -> dict[str, Any]:
        """The task in the report format the dashboard reads, metrics attached."""
        bound = self.bound_metrics
        return {
            "task_id": self.task.id,
            "task_text": self.task.text,
            "domain": "",  # filled in by run_suite, which knows the suite's domain
            "runs": [r.to_json() for r in self.runs],
            "tags": list(self.task.tags),
            "params": dict(self.task.params),
            "metrics": self.metrics.to_json(),
            "metrics_bound": None if bound is None else bound.to_json(),
        }


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------

Meter = Callable[[], Usage]
"""Reads the model client's cumulative usage. Supplied by a caller that can reach the
client - the harness cannot, since ``Workbench.session`` builds and closes it - and
its absence is why ``input_tokens`` is ``None`` rather than ``0`` on a default run."""


def run_task(
    task: EvalTask,
    *,
    workbench: Any,
    referee: Referee,
    suite: Suite,
    budget: Budget | None = None,
    warm_runs: int = WARM_RUNS,
    bound_runs: int = BOUND_RUNS,
    meter: Meter | None = None,
) -> TaskRecord:
    """Run one task cold once, warm ``warm_runs`` times, resetting before each.

    The cold run is forced to explore (``warm=False``) and allowed to learn, so it
    measures the first encounter honestly even when the library already holds
    something. Each warm run prefers the library but may still fall through to
    exploration, because that is what the product does - and when it happens the run
    records ``skill_used=None`` and ``rescued``, so the report shows a warm run that
    was not warm rather than a mysteriously slow one.

    Three shapes, not two
    ---------------------

    "The warm path makes no model calls" is true only once the skill's parameters are
    known, and a plain-English task still has to be read to find them - one model call
    to notice that "Dana Whitfield's message" means ``sender="Dana Whitfield"``. That
    call is small and real, and quoting the zero without it would be the most
    flattering number in this report bought with a hidden footnote. So a task that
    declares ``params`` is ALSO run with those parameters supplied, and the two are
    recorded separately as ``binding="plain"`` and ``binding="bound"``:

    ====================  ==========================================================
    cold                  explore from scratch, learn a skill
    warm, plain English   retrieve the skill, one call to bind its parameters
    warm, parameters      retrieve the skill, no model at all
    ====================  ==========================================================

    A run that raises is recorded as a failure carrying the error and the suite
    continues. Losing thirteen good measurements because the fourteenth hit a stale
    browser is not a trade worth making.

    Args:
        task: The task to run.
        workbench: A :class:`~skillweaver.orchestrator.Workbench`; only its
            ``session`` and ``store`` are used. **The referee is never given to it.**
        referee: Ground truth. Reset before every run, read after every run.
        suite: The suite the task came from, for the domain, start URL and reset URL.
        budget: Limits for each run. ``None`` uses the contract default.
        warm_runs: How many plain-English warm runs to do after the cold one.
        bound_runs: How many parameters-supplied warm runs to add, for a task that
            declares ``params``. ``0`` skips the third shape.
        meter: Optional cumulative-usage reader, for token counts.
    """
    limits = budget if budget is not None else Budget()
    plain = _spec_for(task, suite, bound=False)
    records: list[RunRecord] = []

    def go(attempt: int, phase: Literal["cold", "warm"], binding: Binding) -> None:
        records.append(
            _one_run(
                task,
                spec=plain if binding == "plain" else _spec_for(task, suite, bound=True),
                attempt=attempt,
                phase=phase,
                binding=binding,
                workbench=workbench,
                referee=referee,
                budget=limits,
                meter=meter,
                domain=suite.domain,
            )
        )

    go(1, "cold", "plain")
    for attempt in range(2, warm_runs + 2):
        go(attempt, "warm", "plain")

    # Only worth running when the task actually names values a skill takes as
    # parameters; for a task with none, "bound" and "plain" are the same run twice.
    if task.params:
        for offset in range(bound_runs):
            go(warm_runs + 2 + offset, "warm", "bound")

    return TaskRecord(task=task, runs=tuple(records))


def _spec_for(task: EvalTask, suite: Suite, *, bound: bool) -> Any:
    """The task as the agent receives it, in one of the two binding shapes.

    ``reset_url`` rides along in both. Without it the admission gate cannot put the
    world back to re-run a candidate, so a task that CHANGES anything - which is most
    of this suite - is never admitted, the library never grows, and every warm run
    quietly explores again. The evaluation would then report a speedup of roughly 1.0
    and look like a clean null result. See
    :data:`~skillweaver.orchestrator.WorldReset`.
    """
    return task_spec(
        task.text,
        domain=suite.domain,
        target="browser",
        url=f"{suite.base_url}{task.start_path}",
        reset_url=f"{suite.base_url}{suite.reset_path}",
        params=dict(task.params) if bound else None,
    )


def _one_run(
    task: EvalTask,
    *,
    spec: Any,
    attempt: int,
    phase: Literal["cold", "warm"],
    binding: Binding,
    workbench: Any,
    referee: Referee,
    budget: Budget,
    meter: Meter | None,
    domain: str,
) -> RunRecord:
    """Reset, run, score. The unit the whole report is built out of.

    The reset comes FIRST and its failure is recorded rather than raised, because a
    run that started on a dirty application is not a run that failed - it is a run
    whose number must not be believed, and the two need different words in the report.
    """
    started = utcnow()
    reset_ok = True
    try:
        referee.reset()
    except SkillWeaverError as exc:
        reset_ok = False
        log.warning("eval.reset.failed", task=task.id, attempt=attempt, error=str(exc))

    before = meter() if meter is not None else None
    session_start = time.perf_counter()
    wall_ms = 0.0
    report: Any = None
    error = ""

    # Ground truth as it stood the moment the task finished. Learning is not a passive
    # observer: the admission gate puts the world BACK to re-run the candidate skill,
    # so a run that archived a message and then taught a skill about it leaves the
    # message un-archived if that replay fails. Scoring after ``run`` returned therefore
    # scored the gate's leftovers and reported a task that was done as not done.
    finished_state: dict[str, Any] = {}

    def snapshot() -> None:
        # The agent is handed this callable, not the referee. It takes nothing and
        # returns nothing, so no fact about the world can travel through it INTO the
        # agent - the ground-truth boundary is unchanged.
        if finished_state:
            return
        try:
            finished_state["state"] = dict(referee.state())
        except SkillWeaverError as exc:
            log.warning("eval.snapshot.failed", task=task.id, attempt=attempt, error=str(exc))

    try:
        with workbench.session(spec, budget) as agent:
            run_start = time.perf_counter()
            # The referee is NOT in scope for the agent: it is never placed on the
            # spec, never passed to the session, and never reachable from `agent`.
            report = agent.run(
                spec,
                learn=(phase == "cold"),
                warm=(phase == "warm"),
                cold=True,
                on_finished=snapshot,
            )
            wall_ms = (time.perf_counter() - run_start) * 1000.0
    except Exception as exc:  # a broken run is one failed measurement, not a lost suite
        error = f"{type(exc).__name__}: {exc}"
        log.warning("eval.run.raised", task=task.id, attempt=attempt, error=error)
        if wall_ms == 0.0:
            wall_ms = (time.perf_counter() - session_start) * 1000.0

    session_ms = (time.perf_counter() - session_start) * 1000.0
    after = meter() if meter is not None else None

    try:
        verdict = score(task, finished_state.get("state") or referee.state())
    except SkillWeaverError as exc:
        verdict = Score(ok=False, results=())
        error = error or f"the referee could not read the application state: {exc}"

    record = RunRecord(
        attempt=attempt,
        phase=phase,
        binding=binding,
        ok=verdict.ok,
        agent_claimed=bool(getattr(report, "ok", False)),
        wall_ms=wall_ms,
        session_ms=session_ms,
        steps=int(getattr(report, "steps", 0) or 0),
        llm_calls=int(getattr(report, "llm_calls", 0) or 0),
        input_tokens=None
        if before is None or after is None
        else after.input_tokens - before.input_tokens,
        output_tokens=None
        if before is None or after is None
        else after.output_tokens - before.output_tokens,
        usd=_spend_of(report),
        skill_used=_skill_of(report),
        decision=str(getattr(report, "decision", "none")),
        rescued=bool(getattr(report, "rescued", False)),
        run_id=getattr(report, "run_id", None),
        learned=_learned_of(report),
        library_size=_library_size(workbench, domain),
        reset_ok=reset_ok,
        error=error,
        started_at=started,
        score=verdict,
    )
    log.info(
        "eval.run",
        task=task.id,
        attempt=attempt,
        phase=phase,
        binding=binding,
        ok=record.ok,
        wall_ms=round(record.wall_ms, 1),
        llm_calls=record.llm_calls,
        skill_used=record.skill_used,
    )
    return record


def _spend_of(report: Any) -> float:
    """Dollars across every attempt of a run; ``0.0`` when the run never started."""
    attempts = getattr(report, "attempts", ()) or ()
    return float(sum(getattr(a, "usd", 0.0) for a in attempts))


def _skill_of(report: Any) -> str | None:
    """The stored skill that carried the run, or ``None`` when it explored.

    Read from the winning attempt only. A warm attempt that ran a skill, failed, and
    was rescued by exploration did not have its skill carry the run, and saying
    otherwise here would report the library as working on exactly the runs that prove
    it was not.
    """
    outcome = getattr(report, "outcome", None)
    if outcome is None or not getattr(report, "ok", False):
        return None
    return getattr(outcome, "skill_used", None)


def _learned_of(report: Any) -> str | None:
    """The name of the skill this run added to the library, if it added one."""
    learned = getattr(report, "learned", None)
    return None if learned is None else str(getattr(learned, "name", "") or "") or None


def _library_size(workbench: Any, domain: str) -> int:
    """How many skills the library holds for this domain. ``0`` if it cannot be read.

    Recorded per run because it is the cheapest possible detector for the failure mode
    that invalidates everything else: a library that never grows means every warm run
    explored, and the speedup column is then measuring noise.
    """
    store = getattr(workbench, "store", None)
    if store is None:
        return 0
    try:
        return len(store.list(domain=domain))
    except (SkillWeaverError, OSError, TypeError, ValueError):
        return 0


def run_suite(
    *,
    workbench: Any,
    referee: Referee,
    suite: Suite,
    out_dir: Path | str,
    budget: Budget | None = None,
    warm_runs: int | None = None,
    bound_runs: int | None = None,
    meter: Meter | None = None,
    stamp: str | None = None,
    only: Iterable[str] | None = None,
) -> Path:
    """Run every task in the suite and write the report and its summary.

    Args:
        workbench: The :class:`~skillweaver.orchestrator.Workbench` to run through.
        referee: Ground truth. Never handed to the agent.
        suite: The loaded task suite.
        out_dir: Directory for ``<stamp>.json`` and ``<stamp>.md``. Created if needed.
        budget: Limits per run.
        warm_runs: Plain-English warm runs per task. ``None`` uses the suite's own.
        bound_runs: Parameters-supplied warm runs per task that declares ``params``.
            ``None`` uses the suite's own; ``0`` skips that shape entirely.
        meter: Optional cumulative-usage reader, for token counts.
        stamp: Report basename. ``None`` uses UTC ``YYYYmmddTHHMMSSZ``.
        only: Run just these task ids. ``None`` runs the whole suite.

    Returns:
        The path of the JSON report. The markdown summary sits beside it as ``.md``.

    Raises:
        SkillWeaverError: if the report cannot be written.
    """
    warm = suite.warm_runs if warm_runs is None else warm_runs
    bound = suite.bound_runs if bound_runs is None else bound_runs
    wanted = set(only) if only is not None else None
    tasks = [t for t in suite.tasks if wanted is None or t.id in wanted]
    log.info("eval.suite.start", suite=suite.name, tasks=len(tasks), warm_runs=warm)

    records = [
        run_task(
            task,
            workbench=workbench,
            referee=referee,
            suite=suite,
            budget=budget,
            warm_runs=warm,
            bound_runs=bound,
            meter=meter,
        )
        for task in tasks
    ]

    summary = suite_metrics(r.metrics for r in records)
    bound_summary = suite_metrics(m for r in records if (m := r.bound_metrics) is not None)
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite.name,
        "generated_at": utcnow().isoformat().replace("+00:00", "Z"),
        "warm_runs": warm,
        "bound_runs": bound,
        "source": suite.source,
        "base_url": suite.base_url,
        "tasks": [dict(r.to_json(), domain=suite.domain) for r in records],
        "metrics": summary.to_json(),
        "metrics_bound": bound_summary.to_json() if bound_summary.tasks else None,
    }
    log.info(
        "eval.suite.done",
        suite=suite.name,
        pooled_speedup=summary.pooled_speedup,
        library_was_used=summary.library_was_used,
    )
    return write_report(report, summary, out_dir, stamp=stamp, bound=bound_summary)


def write_report(
    report: Mapping[str, Any],
    summary: SuiteMetrics,
    out_dir: Path | str,
    *,
    stamp: str | None = None,
    bound: SuiteMetrics | None = None,
) -> Path:
    """Write ``<out_dir>/<stamp>.json`` and ``<out_dir>/<stamp>.md``.

    Returns the JSON path. Raises :class:`SkillWeaverError` if either cannot be
    written: a measurement nobody can read is not a measurement.
    """
    directory = Path(out_dir)
    name = stamp or utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
        md_path.write_text(render_summary(report, summary, bound), encoding="utf-8")
    except OSError as exc:
        raise SkillWeaverError(f"cannot write the evaluation report to {directory}: {exc}") from exc
    log.info("eval.report.written", json=str(json_path), markdown=str(md_path))
    return json_path


# --------------------------------------------------------------------------------------
# The summary a human reads
# --------------------------------------------------------------------------------------


def render_summary(
    report: Mapping[str, Any], summary: SuiteMetrics, bound: SuiteMetrics | None = None
) -> str:
    """The markdown summary: the headline first, then everything that undercuts it.

    Written in that order deliberately. Anything that makes the headline untrustworthy
    - a library that never grew, a cold run that failed, a warm run that was slower,
    a critic that disagreed with ground truth - appears ABOVE the results table, not
    in a footnote below it, because a caveat nobody scrolls to is not a caveat.
    """
    tasks = report.get("tasks") or []
    by_id = {t.get("task_id"): t for t in tasks if isinstance(t, Mapping)}
    lines: list[str] = [
        f"# Evaluation: {report.get('suite', 'suite')}",
        "",
        f"Generated {report.get('generated_at', '')} · "
        f"{len(tasks)} task(s) · {_run_plan(report)}, "
        "with the application reset before every run.",
        "",
    ]

    lines += _warnings(summary, by_id)
    lines += _headline(summary, bound)
    lines += _table(summary, by_id)
    lines += _notes(summary)
    return "\n".join(lines) + "\n"


def _run_plan(report: Mapping[str, Any]) -> str:
    """How many runs of each shape stand behind this report, spelled out.

    A report can never be ambiguous about what its speedup is an average of.
    """
    warm = report.get("warm_runs", 0)
    bound = report.get("bound_runs", 0) if report.get("metrics_bound") else 0
    plan = f"1 cold run then {warm} warm run(s) each"
    if bound:
        plan += f", plus {bound} run(s) with the task's parameters supplied"
    return plan


def _warnings(summary: SuiteMetrics, by_id: Mapping[Any, Any]) -> list[str]:
    """Everything that would make a reader distrust the numbers, before the numbers."""
    out: list[str] = []
    if summary.warm_runs and not summary.library_was_used:
        out += [
            "> **This run measured cold-versus-cold, not cold-versus-warm.**",
            f"> All {summary.warm_runs} warm run(s) solved their task by exploring; no stored",
            "> skill ever carried one. Nothing entered the library, so the speedup figures",
            "> below compare exploration against exploration and mean nothing.",
            "",
        ]
    elif summary.warm_runs_that_explored:
        out += [
            f"> **{summary.warm_runs_that_explored} of {summary.warm_runs} warm run(s) explored "
            "instead of using the library.** Those runs were warm in name only; their timings "
            "are exploration timings.",
            "",
        ]

    dirty = [
        (task.get("task_id"), run.get("attempt"))
        for task in by_id.values()
        for run in (task.get("runs") or [])
        if isinstance(run, Mapping) and run.get("reset_ok") is False
    ]
    if dirty:
        listed = ", ".join(f"{tid} attempt {n}" for tid, n in dirty[:6])
        out += [
            f"> **{len(dirty)} run(s) started without a successful reset** ({listed}). Those runs "
            "began on the previous run's leftovers and their timings are not comparable.",
            "",
        ]

    disagreed = [
        (task.get("task_id"), run.get("attempt"))
        for task in by_id.values()
        for run in (task.get("runs") or [])
        if isinstance(run, Mapping) and run.get("agreed") is False
    ]
    if disagreed:
        listed = ", ".join(f"{tid} attempt {n}" for tid, n in disagreed[:6])
        out += [
            f"> **The agent's own verdict disagreed with ground truth on {len(disagreed)} run(s)** "
            f"({listed}). Ground truth is what the table below reports.",
            "",
        ]
    return out


def _shapes(summary: SuiteMetrics, bound: SuiteMetrics | None) -> list[str]:
    """The three shapes, side by side, so the zero is never quoted on its own.

    "The warm path makes no model calls" is the most quotable sentence this project
    has, and on its own it is misleading: getting to zero needs the skill's parameters
    supplied. Asked in plain English the warm path still spends one small call working
    out that "Dana Whitfield's message" means ``sender="Dana Whitfield"``. Both numbers
    are real and they belong in the same table, with the cheaper one carrying the
    condition that buys it.
    """
    if bound is None or not bound.tasks:
        return []
    return [
        "## The three shapes",
        "",
        "| shape | what the agent is given | model calls per run | speedup |",
        "| --- | --- | --- | --- |",
        f"| cold | the sentence, and no library | {_fmt(_cold_calls(summary), '')} "
        "| 1.00x (baseline) |",
        f"| warm, plain English | the sentence alone | {_fmt(summary.warm_call_mean, '')} "
        f"| {_fmt(summary.pooled_speedup, 'x')} |",
        f"| warm, parameters supplied | the sentence and its parameters "
        f"| {_fmt(bound.warm_call_mean, '')} | {_fmt(bound.pooled_speedup, 'x')} |",
        "",
        "The plain-English row is the one a person gets by typing a sentence; the row "
        "below it is what the same skill costs once its parameters are known. Quote "
        "the zero only with that condition attached.",
        "",
    ]


def _cold_calls(summary: SuiteMetrics) -> float | None:
    """Mean model calls of the successful cold runs across the comparable tasks."""
    values = [t.cold_llm_calls for t in summary.tasks if t.cold_llm_calls is not None]
    return fmean(values) if values else None


def _headline(summary: SuiteMetrics, bound: SuiteMetrics | None = None) -> list[str]:
    """The four numbers this project exists to produce, and the three shapes."""
    return _shapes(summary, bound) + [
        "## Headline",
        "",
        f"- **Speedup (pooled): {_fmt(summary.pooled_speedup, 'x')}** "
        f"— total cold time over total warm time across the "
        f"{summary.comparable_tasks} comparable task(s)",
        f"- Speedup (mean per task): {_fmt(summary.mean_speedup, 'x')}",
        f"- **Success delta: {_fmt(summary.mean_success_delta, '', plus=True)}** "
        "— warm success rate minus cold, averaged over tasks",
        f"- Cold success rate: {_pct(summary.cold_success_rate)} · "
        f"Warm success rate: {_pct(summary.warm_success_rate)}",
        f"- Model calls saved per warm run: {_fmt(summary.total_call_reduction, '')} "
        "across the comparable tasks",
        "",
    ]


def _table(summary: SuiteMetrics, by_id: Mapping[Any, Any]) -> list[str]:
    """Per task, with the reason in place of any number that could not be computed."""
    out = [
        "## Per task",
        "",
        "| task | tags | cold ok | warm ok | cold ms | warm ms | speedup | cold calls | "
        "warm calls | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in summary.tasks:
        source = by_id.get(task.task_id) or {}
        tags = ", ".join(source.get("tags") or []) if isinstance(source, Mapping) else ""
        speed = _fmt(task.speedup, "x")
        if task.regressed:
            speed = f"**{speed} SLOWER**"
        out.append(
            f"| `{task.task_id}` | {tags} | {_tick(task.cold_ok)} | {_pct(task.warm_ok_rate)} | "
            f"{_fmt(task.cold_ms, '')} | {_fmt(task.warm_ms, '')} | {speed} | "
            f"{_fmt(task.cold_llm_calls, '')} | {_fmt(task.warm_llm_calls, '')} | {task.note} |"
        )
    out.append("")
    return out


def _notes(summary: SuiteMetrics) -> list[str]:
    """Named failures. Nothing here is aggregated away into a percentage."""
    out: list[str] = []
    if summary.cold_failures:
        out += [
            "## Tasks the cold run could not do",
            "",
            *[f"- `{t}`" for t in summary.cold_failures],
            "",
        ]
    if summary.warm_failures:
        out += [
            "## Tasks whose warm run failed",
            "",
            "The library was asked to repeat something it had already done and did not.",
            "",
            *[f"- `{t}`" for t in summary.warm_failures],
            "",
        ]
    if summary.regressions:
        out += [
            "## Tasks where warm was SLOWER than cold",
            "",
            *[f"- `{t}`" for t in summary.regressions],
            "",
        ]
    if not (summary.cold_failures or summary.warm_failures or summary.regressions):
        out += [
            "Every task succeeded cold and warm, and no warm run was slower than its cold one.",
            "",
        ]
    return out


def _fmt(value: float | int | None, unit: str, *, plus: bool = False) -> str:
    """A number, or an em dash where a number would have to be invented."""
    if value is None:
        return "—"
    sign = "+" if plus and value > 0 else ""
    return f"{sign}{value:,.2f}{unit}" if isinstance(value, float) else f"{sign}{value}{unit}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _tick(value: bool | None) -> str:
    return "—" if value is None else ("yes" if value else "**no**")


# --------------------------------------------------------------------------------------
# The command line's entry point
# --------------------------------------------------------------------------------------


def run(
    *,
    workbench: Any,
    suite: Path | str | None = None,
    out: Path | str | None = None,
    repeat: int = WARM_RUNS,
    bound_runs: int | None = None,
    budget: Budget | None = None,
    meter: Meter | None = None,
) -> Path:
    """Entry point for ``skillweaver eval run``. Loads the suite and runs it live.

    The referee is built here, from the suite's ``base_url``, and never leaves this
    function's frame except into :func:`run_suite`, which likewise hands it to nothing
    the agent can see.

    ``repeat`` is passed through exactly as given. Note that ``cli.py`` - which this
    harness does not own - defaults that flag to ``1``, while this suite's own default
    and :data:`WARM_RUNS` are ``3``; the number actually used is recorded in the report
    and printed at the top of the summary, so a report can never be ambiguous about how
    many warm runs stand behind its speedup.

    Args:
        workbench: The :class:`~skillweaver.orchestrator.Workbench` to run through.
        suite: Path to the task suite. ``None`` uses ``eval/tasks.yaml``.
        out: Where the report is written. ``None`` uses ``<data-dir>/eval``.
        repeat: Warm runs per task, after the cold one.
        budget: Limits per run.
        meter: Optional cumulative-usage reader, for token counts.

    Returns:
        The path of the JSON report.

    Raises:
        ConfigError: if the suite cannot be loaded.
        SkillWeaverError: if the sandbox is not reachable or the report cannot be
            written.
    """
    loaded = load_suite(suite)
    referee = HttpReferee(loaded.base_url, reset_path=loaded.reset_path)
    # Fail before running fourteen tasks against a world that is not there.
    referee.reset()
    destination = (
        Path(out) if out is not None else Path(getattr(workbench, "data_dir", "data")) / "eval"
    )
    return run_suite(
        workbench=workbench,
        referee=referee,
        suite=loaded,
        out_dir=destination,
        budget=budget,
        warm_runs=repeat,
        bound_runs=bound_runs,
        meter=meter,
    )
