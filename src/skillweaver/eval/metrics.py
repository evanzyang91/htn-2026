"""Speedup and success delta, computed from recorded runs.

This module is pure arithmetic over numbers somebody else measured. It opens no
browser, calls no model and reads no file; hand it :class:`RunPoint` values - or the
``runs`` list straight out of a report written by
:mod:`skillweaver.eval.harness` - and it answers.

It is deliberately separate from the harness because the harness is the part that can
be wrong in interesting ways (a browser that hung, a reset that did not take) while
this part must never be wrong at all. Every number below is a headline number for this
project, so each one says what it means and, more importantly, **says nothing when it
would have to invent something to speak**.

What "undefined" means here, and why there is so much of it
-----------------------------------------------------------

Every quantity in this module is ``| None``, and ``None`` is returned rather than a
plausible-looking substitute in exactly the cases where a substitute would be a lie:

* **A run that failed is not a measurement of how long the task takes.** A warm run
  that gave up in 40ms did not do the task 900x faster than the cold run that spent
  36 seconds doing it. So :func:`task_metrics` computes timings from **successful runs
  only**; if the cold run failed, or every warm run failed, there is no honest
  comparison and ``speedup`` is ``None`` with a ``note`` saying which side was missing.
* **A zero-length measurement cannot be divided by.** ``speedup`` of a 0ms warm run is
  not infinity, it is a broken clock, and :func:`speedup` answers ``None``.
* **A task with no warm run has no success delta.** Not ``0.0`` - which would read as
  "the library neither helped nor hurt" - but ``None``: nothing was asked of it.

A ``None`` that shows up in a report is therefore a fact about the evaluation, and the
markdown summary prints it as an explicit reason rather than a blank cell.

Which direction the numbers point
---------------------------------

``speedup`` is **cold / warm**, so greater than 1.0 means the warm run was faster and
the project's claim holds for that task; below 1.0 means the library made it *slower*
and :attr:`SuiteMetrics.regressions` names it. ``success_delta`` is **warm minus
cold**, in ``-1.0..1.0``, so a negative number means the library broke a task that
exploration could do - the single worst thing this suite can find, and
:attr:`SuiteMetrics.warm_failures` names those too.

The one door a run gets in through
----------------------------------

A wrong answer is the CHEAPEST answer this architecture can give. On 2026-09-19 the
ordering suite's ``order_appears_in_history`` retrieved a trivial navigation skill,
ran it in 4 to 8 seconds with zero model calls, and got the task wrong all four
times - faster and cheaper than any honest run in the suite. Every efficiency figure
this project reports gets BETTER when a skill does the wrong thing quickly, so the
metric and the truth point in opposite directions and only ground truth separates
them.

That is why every timing and every saving below is computed from :func:`_measured`
and from nowhere else. It is one function, four lines long, and it admits a run only
when :attr:`RunPoint.ok` - the OFFLINE REFEREE's verdict, never the agent's own -
says the task was achieved. There is deliberately no second path to a sample: a
future edit that wants to average something has to go through that door or write a
new one in plain sight.

Excluding a run is not enough on its own, because a number that quietly drops its
most flattering runs is still a number nobody can audit. So the failures are counted
and NAMED rather than averaged away: :attr:`TaskMetrics.warm_failed_ms` says how fast
the excluded runs were, :attr:`TaskMetrics.warm_failures_from_library` says how many
of them a stored skill carried - the fast-free-and-wrong shape exactly - and
:attr:`TaskMetrics.flattered_by_failures` is ``True`` when counting them would have
made the task look better. :attr:`TaskMetrics.note` says so in the sentence the
markdown summary already prints next to the speedup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

__all__ = [
    "RunPoint",
    "SuiteMetrics",
    "TaskMetrics",
    "call_reduction",
    "speedup",
    "success_delta",
    "suite_metrics",
    "suite_metrics_from_report",
    "task_metrics",
]


# --------------------------------------------------------------------------------------
# The input: one measured run, reduced to the numbers this module needs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunPoint:
    """One measured attempt, reduced to what the arithmetic needs.

    This is a view over a run, not a record of one - the harness's own
    ``RunRecord`` carries far more. ``attempt`` counts from ``1`` and the lowest
    attempt of a task is its COLD run, matching the report format the dashboard reads.

    Attributes:
        attempt: Which attempt this was, counting from ``1``.
        ok: Whether the run achieved the task, **as the offline referee scored it**
            and not as the agent claimed.
        wall_ms: Wall-clock milliseconds the run took.
        llm_calls: Model calls the run made. Zero on a true warm run is the claim
            this whole project exists to make.
        usd: Model spend for the run, in US dollars.
        used_library: Whether a stored skill carried the run. A "warm" run with this
            ``False`` explored instead, which means it was not warm at all - see
            :attr:`SuiteMetrics.warm_runs_that_explored`.
        verified: Whether ``ok`` came from a referee's per-check score rather than
            from a bare boolean somebody wrote into the report. A run constructed
            directly is trusted, because the only thing that does that is the
            harness, which holds the referee; a run read back out of a report is
            trusted only when the report carries the checks that prove it.
    """

    attempt: int
    ok: bool
    wall_ms: float
    llm_calls: int = 0
    usd: float = 0.0
    used_library: bool = False
    verified: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], position: int = 0) -> RunPoint:
        """One run out of a report's ``runs`` list.

        Tolerant in the same way the dashboard's reader is: a missing ``attempt``
        falls back to the position in the list, and anything unparsable becomes the
        field's default rather than raising, because a report that is half readable
        is worth more than an exception.

        One place it is NOT tolerant. A report carries two verdicts per run: ``ok``,
        which the harness writes from the referee, and ``score``, the referee's own
        per-check detail. Where a report has both, the ``score`` wins, because ``ok``
        is one boolean that any writer could have filled in from the agent's own
        claim while ``score`` is the evidence. A run whose score is absent is still
        read - most reports are honest and refusing them all would answer nothing -
        but it is marked ``verified=False`` and counted in
        :attr:`TaskMetrics.unverified_runs` so a reader knows which numbers rest on
        an unshown check.
        """
        scored = raw.get("score")
        checked = isinstance(scored, Mapping) and isinstance(scored.get("ok"), bool)
        return cls(
            attempt=_as_int(raw.get("attempt"), position + 1),
            ok=bool(scored["ok"]) if checked else bool(raw.get("ok", False)),  # type: ignore[index]
            wall_ms=_as_float(raw.get("wall_ms"), 0.0),
            llm_calls=_as_int(raw.get("llm_calls"), 0),
            usd=_as_float(raw.get("usd"), 0.0),
            used_library=_as_bool(raw.get("used_library"), raw.get("skill_used") is not None),
            verified=checked,
        )


def _as_int(value: Any, default: int) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _as_float(value: Any, default: float) -> float:
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
    )


def _as_bool(value: Any, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


# --------------------------------------------------------------------------------------
# The gate: the only way a run reaches a timing or a saving
# --------------------------------------------------------------------------------------


def _measured(runs: Iterable[RunPoint]) -> list[RunPoint]:
    """The runs a timing or a saving may be computed from: the ones that WORKED.

    Every mean in this module starts here, and nothing else in it filters a sample.
    A run that did not achieve the task is not a measurement of doing the task - it
    is a measurement of giving up, and the cheapest way to give up is to confidently
    do the wrong thing fast. Averaging one in does not merely add noise, it moves the
    headline in the flattering direction, which is why this is a gate and not a
    guideline.

    ``ok`` is the offline referee's verdict. Nothing here consults what the agent
    believed about itself.
    """
    return [run for run in runs if run.ok]


# --------------------------------------------------------------------------------------
# The two headline numbers, each as a function you can read in one sitting
# --------------------------------------------------------------------------------------


def speedup(cold_ms: float | None, warm_ms: float | None) -> float | None:
    """How many times faster the warm run was than the cold one, or ``None``.

    ``2.0`` means warm took half the time. Below ``1.0`` means the library made the
    task slower, which is a real result and is reported as such.

    Returns ``None`` - never ``0.0``, never ``inf`` - when either side is missing,
    zero or negative. A zero-millisecond run is a clock that did not work, not a run
    that took no time, and dividing by it would manufacture the largest number in the
    report out of the least reliable measurement in it.

    Args:
        cold_ms: Wall-clock milliseconds of the successful cold run.
        warm_ms: Wall-clock milliseconds of the successful warm run, or their mean.
    """
    if cold_ms is None or warm_ms is None:
        return None
    if cold_ms <= 0.0 or warm_ms <= 0.0:
        return None
    return cold_ms / warm_ms


def success_delta(cold_ok: bool | None, warm_oks: Sequence[bool]) -> float | None:
    """The warm success rate minus the cold one, in ``-1.0..1.0``, or ``None``.

    ``+1.0`` is a task exploration could not do and the library can; ``-1.0`` is a
    task exploration could do and the library broke. ``0.0`` means both sides agreed,
    whether they agreed on success or on failure - those two are very different
    outcomes, so read this next to :attr:`TaskMetrics.cold_ok`.

    Returns ``None`` when there were no warm runs at all, because a delta against
    nothing is not zero, it is unasked.

    Args:
        cold_ok: Whether the cold run succeeded, or ``None`` if there was no cold run.
        warm_oks: Whether each warm run succeeded, in attempt order.
    """
    if cold_ok is None or not warm_oks:
        return None
    return fmean(1.0 if ok else 0.0 for ok in warm_oks) - (1.0 if cold_ok else 0.0)


def call_reduction(cold_calls: int | None, warm_calls: float | None) -> float | None:
    """Model calls saved per warm run: cold minus the warm mean, or ``None``.

    Reported as an absolute count rather than a ratio because the number that matters
    is ``0`` - a warm run that consults no model at all - and every ratio against zero
    is either undefined or misleadingly enormous.
    """
    if cold_calls is None or warm_calls is None:
        return None
    return float(cold_calls) - warm_calls


# --------------------------------------------------------------------------------------
# Per task
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """What the recorded runs of one task add up to.

    Attributes:
        task_id: The task these runs belong to.
        runs: How many runs were recorded, cold and warm together.
        cold_ok: Whether the cold run succeeded, or ``None`` if there was none.
        warm_ok_rate: Fraction of warm runs that succeeded, or ``None`` if there
            were none.
        success_delta: See :func:`success_delta`.
        cold_ms: Wall-clock milliseconds of the cold run, **if it succeeded**.
        warm_ms: Mean wall-clock milliseconds of the **successful** warm runs.
        speedup: See :func:`speedup`.
        cold_llm_calls: Model calls of the successful cold run.
        warm_llm_calls: Mean model calls of the successful warm runs.
        call_reduction: See :func:`call_reduction`.
        cold_usd / warm_usd: Model spend of the cold run and the warm mean.
        warm_failures: How many warm runs failed.
        warm_explored: How many warm runs solved the task by exploring rather than
            from the library - runs that were warm in name only.
        warm_failed_ms: Mean wall-clock of the warm runs that FAILED, which no figure
            above is computed from. Reported so the excluded runs can be seen rather
            than merely counted: next to :attr:`warm_ms` it says whether the library
            was wrong slowly or wrong fast.
        warm_failed_llm_calls: Mean model calls of those same failed warm runs. A
            zero here is the worst reading in this class - free, and wrong.
        warm_failures_from_library: How many of the failed warm runs a stored skill
            carried. This is the fast-free-and-wrong shape the 2026-09-19 ordering
            suite found: retrieval answered confidently with an unrelated skill.
        unverified_runs: Runs whose ``ok`` arrived without the referee's checks
            behind it. See :meth:`RunPoint.from_mapping`.
        comparable: Whether :attr:`speedup` could be computed at all.
        note: Why it could not, when it could not - and, when it could, what was
            left out of it. Empty only when nothing was excluded and nothing failed.
        regressed: Whether the warm runs were SLOWER than the cold one.
        flattered_by_failures: Whether counting the failed warm runs would have made
            this task look BETTER than the truth.
    """

    task_id: str
    runs: int = 0
    cold_ok: bool | None = None
    warm_ok_rate: float | None = None
    success_delta: float | None = None
    cold_ms: float | None = None
    warm_ms: float | None = None
    speedup: float | None = None
    cold_llm_calls: int | None = None
    warm_llm_calls: float | None = None
    call_reduction: float | None = None
    cold_usd: float | None = None
    warm_usd: float | None = None
    warm_failures: int = 0
    warm_explored: int = 0
    warm_failed_ms: float | None = None
    warm_failed_llm_calls: float | None = None
    warm_failures_from_library: int = 0
    unverified_runs: int = 0
    comparable: bool = False
    note: str = ""

    @property
    def regressed(self) -> bool:
        """Whether the library made this task slower. Never true when incomparable."""
        return self.speedup is not None and self.speedup < 1.0

    @property
    def flattered_by_failures(self) -> bool:
        """Whether the runs this task's numbers EXCLUDE were faster than the ones
        they are made of.

        True is the dangerous shape stated exactly: the library answered wrong, and
        answered wrong faster than it answers right, so any figure that counted those
        runs would have rewarded the wrong answer. The comparison is against the warm
        mean where there is one and against the cold run where every warm run failed
        - in the second case nothing honest survives to compare, and the failures
        being faster than the cold baseline is precisely what would have made the task
        look like the best result in the suite.
        """
        if self.warm_failed_ms is None or self.warm_failed_ms <= 0.0:
            return False
        reference = self.warm_ms if self.warm_ms is not None else self.cold_ms
        return reference is not None and self.warm_failed_ms < reference

    def to_json(self) -> dict[str, Any]:
        """The metrics as plain data, for the report file and the markdown summary."""
        return {
            "task_id": self.task_id,
            "runs": self.runs,
            "cold_ok": self.cold_ok,
            "warm_ok_rate": _round(self.warm_ok_rate, 4),
            "success_delta": _round(self.success_delta, 4),
            "cold_ms": _round(self.cold_ms, 1),
            "warm_ms": _round(self.warm_ms, 1),
            "speedup": _round(self.speedup, 3),
            "cold_llm_calls": self.cold_llm_calls,
            "warm_llm_calls": _round(self.warm_llm_calls, 2),
            "call_reduction": _round(self.call_reduction, 2),
            "cold_usd": _round(self.cold_usd, 6),
            "warm_usd": _round(self.warm_usd, 6),
            "warm_failures": self.warm_failures,
            "warm_explored": self.warm_explored,
            "warm_failed_ms": _round(self.warm_failed_ms, 1),
            "warm_failed_llm_calls": _round(self.warm_failed_llm_calls, 2),
            "warm_failures_from_library": self.warm_failures_from_library,
            "unverified_runs": self.unverified_runs,
            "comparable": self.comparable,
            "regressed": self.regressed,
            "flattered_by_failures": self.flattered_by_failures,
            "note": self.note,
        }


def task_metrics(task_id: str, runs: Iterable[RunPoint]) -> TaskMetrics:
    """Reduce every recorded run of one task to its comparison.

    The run with the lowest ``attempt`` is the cold one and every later run is warm,
    matching the report format the dashboard reads. Timings and call counts come from
    **successful runs only**: a failed run is a measurement of giving up, and averaging
    it into a "how fast is this" number is how a suite ends up reporting a speedup it
    did not earn.

    Args:
        task_id: The task's stable id.
        runs: Its runs, in any order.
    """
    ordered = sorted(runs, key=lambda r: r.attempt)
    if not ordered:
        return TaskMetrics(task_id=task_id, note="no runs were recorded")

    cold, warm = ordered[0], ordered[1:]
    # The one gate. Every mean below is over a list that came out of it, and the
    # failed runs are kept in `warm_failed` to be REPORTED, never to be averaged in.
    counted_cold = _measured([cold])  # empty unless the cold run achieved the task
    warm_ok = _measured(warm)
    warm_failed = [r for r in warm if not r.ok]

    cold_ms = cold.wall_ms if counted_cold else None
    warm_ms = fmean(r.wall_ms for r in warm_ok) if warm_ok else None
    warm_calls = fmean(r.llm_calls for r in warm_ok) if warm_ok else None
    factor = speedup(cold_ms, warm_ms)

    computed = TaskMetrics(
        task_id=task_id,
        runs=len(ordered),
        cold_ok=cold.ok,
        warm_ok_rate=fmean(1.0 if r.ok else 0.0 for r in warm) if warm else None,
        success_delta=success_delta(cold.ok, [r.ok for r in warm]),
        cold_ms=cold_ms,
        warm_ms=warm_ms,
        speedup=factor,
        cold_llm_calls=cold.llm_calls if counted_cold else None,
        warm_llm_calls=warm_calls,
        call_reduction=call_reduction(cold.llm_calls if counted_cold else None, warm_calls),
        cold_usd=cold.usd if counted_cold else None,
        warm_usd=fmean(r.usd for r in warm_ok) if warm_ok else None,
        warm_failures=len(warm_failed),
        warm_explored=sum(1 for r in warm if not r.used_library),
        warm_failed_ms=fmean(r.wall_ms for r in warm_failed) if warm_failed else None,
        warm_failed_llm_calls=fmean(r.llm_calls for r in warm_failed) if warm_failed else None,
        warm_failures_from_library=sum(1 for r in warm_failed if r.used_library),
        unverified_runs=sum(1 for r in ordered if not r.verified),
        comparable=factor is not None,
    )
    return replace(computed, note=_note_for(computed, cold, warm, warm_ok, cold_ms, warm_ms))


def _note_for(
    computed: TaskMetrics,
    cold: RunPoint,
    warm: Sequence[RunPoint],
    warm_ok: Sequence[RunPoint],
    cold_ms: float | None,
    warm_ms: float | None,
) -> str:
    """The sentence the markdown summary prints beside this task's speedup.

    Two jobs, in this order. When there is no speedup it says why - that is the old
    behaviour and the more urgent message. When there IS one it says what the number
    left out, because a figure computed from three of five runs and presented like a
    figure computed from five is the quiet half of the same problem. A task whose
    excluded runs were FASTER than the ones that counted gets told so in as many
    words, since that is the reading under which the exclusion is what saved the
    report.
    """
    parts = [
        _why_incomparable(cold, warm, warm_ok, cold_ms, warm_ms, computed.speedup),
        _what_was_left_out(computed),
    ]
    return "; ".join(part for part in parts if part)


def _what_was_left_out(computed: TaskMetrics) -> str:
    """What the numbers above exclude, or ``""`` when they exclude nothing.

    Silent on a task where nothing failed, so the note column stays empty on a
    healthy task and every word in it means something.
    """
    failed = computed.warm_failures
    if not failed:
        return "" if not computed.unverified_runs else _unverified(computed)
    total = max(computed.runs - 1, failed)
    said = (
        f"{failed} of {total} warm run(s) failed ground truth and are excluded from "
        "every figure here"
    )
    if computed.warm_failures_from_library:
        said += f" ({computed.warm_failures_from_library} carried by a stored skill)"
    if computed.flattered_by_failures:
        reference = computed.warm_ms if computed.warm_ms is not None else computed.cold_ms
        kind = "warm" if computed.warm_ms is not None else "cold"
        said += (
            f" - they averaged {computed.warm_failed_ms:,.0f}ms against "
            f"{reference:,.0f}ms for the {kind} run(s) that counted, so counting them "
            "would have flattered this task"
        )
    extra = _unverified(computed)
    return f"{said}; {extra}" if extra else said


def _unverified(computed: TaskMetrics) -> str:
    """The note for runs whose verdict arrived without the referee's checks."""
    if not computed.unverified_runs:
        return ""
    return (
        f"{computed.unverified_runs} run(s) carried no referee checks, so their "
        "verdict is taken on trust"
    )


def _why_incomparable(
    cold: RunPoint,
    warm: Sequence[RunPoint],
    warm_ok: Sequence[RunPoint],
    cold_ms: float | None,
    warm_ms: float | None,
    factor: float | None,
) -> str:
    """The sentence the summary prints where a speedup would have gone.

    Ordered most-important-first: a broken side is more interesting than a broken
    clock, and "nothing to compare against" is more interesting than either.
    """
    if factor is not None:
        return ""
    if not warm:
        return "no warm run: the task was run cold only"
    if not cold.ok:
        return "the cold run failed, so there is no baseline to be faster than"
    if not warm_ok:
        return f"every warm run failed ({len(warm)}/{len(warm)}), so there is nothing to compare"
    if cold_ms is not None and cold_ms <= 0.0:
        return "the cold run measured 0ms, which is a broken clock rather than a fast run"
    if warm_ms is not None and warm_ms <= 0.0:
        return "the warm runs measured 0ms, which is a broken clock rather than a fast run"
    return "the comparison could not be computed"


# --------------------------------------------------------------------------------------
# Per suite
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuiteMetrics:
    """What a whole evaluation run adds up to: the numbers that go on the slide.

    Attributes:
        tasks: Per-task metrics, in the order the tasks were recorded.
        pooled_speedup: Total successful cold milliseconds over total successful warm
            milliseconds, across the comparable tasks. **This is the headline
            number**, because it weights a task by how long it actually takes rather
            than letting a 200ms task count as much as a 40-second one.
        mean_speedup: The unweighted mean of the per-task speedups. Reported next to
            the pooled figure because the two disagreeing is itself informative: it
            means the wins and the long tasks are not the same tasks.
        mean_success_delta: Mean of the per-task success deltas.
        cold_success_rate / warm_success_rate: Fraction of cold / warm runs that
            achieved the task, across the whole suite.
        total_call_reduction: Model calls saved per warm run, summed over the
            comparable tasks.
        comparable_tasks: How many tasks yielded a speedup at all.
        regressions: Tasks where the warm runs were SLOWER than the cold one.
        warm_failures: Tasks where at least one warm run failed.
        cold_failures: Tasks where the cold run failed.
        warm_runs_that_explored: Warm runs across the suite that solved the task by
            exploring instead of from the library. **When this equals the number of
            warm runs, the suite measured cold-versus-cold**: nothing was ever in the
            library, and every speedup below is noise. :attr:`library_was_used` says
            so in one boolean.
        warm_runs: How many warm runs the suite recorded in total.
        warm_runs_excluded: Warm runs across the suite that failed ground truth and
            are therefore in none of the figures above. The difference between this
            and :attr:`warm_runs` is what the speedup is actually an average of.
        wrong_skill_runs: Excluded runs that a STORED SKILL carried. The library did
            not decline, it answered, and it was wrong - the failure mode a
            wall-clock table rewards.
        flattered_tasks: Tasks whose excluded runs were faster than the runs that
            counted. Every one of these would have raised the headline had ground
            truth not been consulted, which is what makes the list worth printing.
        unverified_runs: Runs whose verdict arrived without the referee's checks.
    """

    tasks: tuple[TaskMetrics, ...] = ()
    pooled_speedup: float | None = None
    mean_speedup: float | None = None
    mean_success_delta: float | None = None
    cold_success_rate: float | None = None
    warm_success_rate: float | None = None
    total_call_reduction: float | None = None
    comparable_tasks: int = 0
    regressions: tuple[str, ...] = ()
    warm_failures: tuple[str, ...] = ()
    cold_failures: tuple[str, ...] = ()
    warm_runs_that_explored: int = 0
    warm_runs: int = 0
    warm_runs_excluded: int = 0
    wrong_skill_runs: int = 0
    flattered_tasks: tuple[str, ...] = ()
    unverified_runs: int = 0

    @property
    def warm_call_mean(self) -> float | None:
        """Mean model calls of a successful warm run, across the comparable tasks.

        The number the project's headline claim rests on, and the one that differs
        between asking in plain English and supplying the skill's parameters. ``None``
        when no task was comparable, rather than ``0.0`` - which would read as the
        very claim being made.
        """
        values = [t.warm_llm_calls for t in self.tasks if t.warm_llm_calls is not None]
        return fmean(values) if values else None

    @property
    def library_was_used(self) -> bool:
        """Whether ANY warm run was actually carried by a stored skill.

        ``False`` is the result that invalidates every other number here, and it is
        the failure mode this project is most exposed to: if synthesis never stores a
        skill, every "warm" run quietly explores from scratch and the suite reports a
        speedup of roughly 1.0 while appearing to work perfectly. The harness prints
        this as a warning at the top of the summary rather than at the bottom.
        """
        return self.warm_runs > 0 and self.warm_runs_that_explored < self.warm_runs

    def to_json(self) -> dict[str, Any]:
        """The metrics as plain data, for the report file and the markdown summary."""
        return {
            "pooled_speedup": _round(self.pooled_speedup, 3),
            "mean_speedup": _round(self.mean_speedup, 3),
            "mean_success_delta": _round(self.mean_success_delta, 4),
            "cold_success_rate": _round(self.cold_success_rate, 4),
            "warm_success_rate": _round(self.warm_success_rate, 4),
            "total_call_reduction": _round(self.total_call_reduction, 2),
            "warm_call_mean": _round(self.warm_call_mean, 3),
            "comparable_tasks": self.comparable_tasks,
            "regressions": list(self.regressions),
            "warm_failures": list(self.warm_failures),
            "cold_failures": list(self.cold_failures),
            "warm_runs": self.warm_runs,
            "warm_runs_that_explored": self.warm_runs_that_explored,
            "warm_runs_excluded": self.warm_runs_excluded,
            "wrong_skill_runs": self.wrong_skill_runs,
            "flattered_tasks": list(self.flattered_tasks),
            "unverified_runs": self.unverified_runs,
            "library_was_used": self.library_was_used,
            "tasks": [t.to_json() for t in self.tasks],
        }


def suite_metrics(tasks: Iterable[TaskMetrics]) -> SuiteMetrics:
    """Roll per-task metrics up into the suite's headline numbers.

    Tasks that are not :attr:`TaskMetrics.comparable` are excluded from both speedup
    figures and counted in the lists that name what went wrong, so an incomparable
    task drags the report's credibility down rather than silently dropping out of it.
    """
    ordered = tuple(tasks)
    if not ordered:
        return SuiteMetrics()

    comparable = [t for t in ordered if t.comparable]
    cold_total = sum(t.cold_ms or 0.0 for t in comparable)
    warm_total = sum(t.warm_ms or 0.0 for t in comparable)
    deltas = [t.success_delta for t in ordered if t.success_delta is not None]
    cold_oks = [t.cold_ok for t in ordered if t.cold_ok is not None]
    warm_rates = [t.warm_ok_rate for t in ordered if t.warm_ok_rate is not None]
    reductions = [t.call_reduction for t in comparable if t.call_reduction is not None]

    return SuiteMetrics(
        tasks=ordered,
        pooled_speedup=speedup(cold_total, warm_total),
        mean_speedup=fmean(t.speedup for t in comparable if t.speedup is not None)
        if comparable
        else None,
        mean_success_delta=fmean(deltas) if deltas else None,
        cold_success_rate=fmean(1.0 if ok else 0.0 for ok in cold_oks) if cold_oks else None,
        warm_success_rate=fmean(warm_rates) if warm_rates else None,
        total_call_reduction=sum(reductions) if reductions else None,
        comparable_tasks=len(comparable),
        regressions=tuple(t.task_id for t in ordered if t.regressed),
        warm_failures=tuple(t.task_id for t in ordered if t.warm_failures),
        cold_failures=tuple(t.task_id for t in ordered if t.cold_ok is False),
        warm_runs_that_explored=sum(t.warm_explored for t in ordered),
        warm_runs=sum(max(t.runs - 1, 0) for t in ordered),
        warm_runs_excluded=sum(t.warm_failures for t in ordered),
        wrong_skill_runs=sum(t.warm_failures_from_library for t in ordered),
        flattered_tasks=tuple(t.task_id for t in ordered if t.flattered_by_failures),
        unverified_runs=sum(t.unverified_runs for t in ordered),
    )


def suite_metrics_from_report(report: Mapping[str, Any]) -> SuiteMetrics:
    """Compute the suite's metrics from a report file's contents.

    Takes the parsed JSON of a ``data/eval/<stamp>.json`` file - the same bytes the
    dashboard reads - so the numbers in a summary can always be re-derived from the
    artifact rather than trusted because the harness said so.

    A malformed report yields empty metrics rather than raising: this is a reporting
    path, and a summary that says "nothing to report" beats a traceback.
    """
    raw_tasks = report.get("tasks") if isinstance(report, Mapping) else None
    if not isinstance(raw_tasks, list):
        return SuiteMetrics()
    computed: list[TaskMetrics] = []
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("task_id") or raw.get("id") or "")
        if not task_id:
            continue
        raw_runs = raw.get("runs")
        points = (
            [
                RunPoint.from_mapping(item, index)
                for index, item in enumerate(raw_runs)
                if isinstance(item, Mapping)
            ]
            if isinstance(raw_runs, list)
            else []
        )
        computed.append(task_metrics(task_id, points))
    return suite_metrics(computed)


def _round(value: float | None, digits: int) -> float | None:
    """``round`` that passes ``None`` through, so a JSON field stays honestly null."""
    return None if value is None else round(value, digits)
