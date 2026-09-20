"""Who decides a run is finished on the ``--perception dom --policy jev`` path: Jev, alone.

The vision critic answered a ``DONE`` claim in 3.8-9.7s (live splitkb, 2026-09-20: two
1280x800 PNGs up, ~200 tokens of JSON back), which was most of what a fast cold run spent
after its last real move, and the same call is what reported correct Walmart warm replays
as failures at ~$0.055 each. On this path the claim is now settled by ONE Jev question
over the DOM text the run already holds - :meth:`skillweaver.llm.jev_.JevPolicy.judge_done`,
a separate question from the policy's own ``DONE`` head - and **Claude is never asked,
not as a fallback, not on a weak answer, not on an error.** The owner's decision, and the
reason there are no tiers here: a second judge that is consulted only when the first is
unsure makes the fast path's cost unpredictable, and this path's promise is that it is not.

What keeps that from being "trust the policy":

* ``satisfied`` at or above :data:`MIN_SATISFIED` is the ONLY yes. ``not_satisfied``,
  ``cannot_tell``, a weak ``satisfied`` and a provider error all REFUSE the claim, and a
  refused claim goes back to the policy exactly as it always did; ``NO_CHANGE_LIMIT`` in
  ``agent/jev_driver.py`` still ends a run that can only repeat itself.
* the free checks run FIRST and can refuse without any model: an error page, and a page
  that is literally the page the run started on (:attr:`DomSnapshot.digest`). They never
  say yes on the cold path. On the warm path they are ``TieredCritic``'s own three roles
  with no model behind them, so a recalled end screen can still corroborate for free.
* a warm replay whose instant end screen does not earn a free yes is re-read AT REST
  (:func:`observe_at_rest`) before anything else: a code-speed skill returns while the
  page is still answering, and the instant read is the "empty with a loading spinner"
  frame ``AGENTS.md`` records.

The pixel path and the default explorer never construct this class:
``orchestrator._open_done_critic`` is the one door, and :data:`ENV_SWITCH` closes it.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from skillweaver.agent.critic import CriticVerdict, TieredCritic
from skillweaver.contracts import Controller, Observation, Perceiver
from skillweaver.errors import ProviderError
from skillweaver.logging_ import get_logger

__all__ = [
    "ENV_SWITCH",
    "MIN_SATISFIED",
    "REST_QUIET_MS",
    "DoneJudge",
    "JevDoneCritic",
    "enabled",
    "observe_at_rest",
    "quoted_arguments",
]

log = get_logger(__name__)

ENV_SWITCH = "SKILLWEAVER_JEV_DONE"
"""Set to ``0`` to get the vision critic back as the judge of a Jev run's ``done`` claim
and of its warm replays. Read here, from the process environment only, because
``config.py`` is not this module's to edit; default ON."""

MIN_SATISFIED = 0.8
"""The least probability at which ``satisfied`` is believed. It can only turn a weak yes
into a refusal - nothing is escalated anywhere.

Where it sits against what was measured, live, 2026-09-20 (``scripts/check_h_done.py
--jev-pages``, ``P(satisfied)`` per page, 186-390ms a call):

  splitkb cart holding the named product            1.000   true done
  Wikipedia article named by the goal               1.000   true done
  splitkb product page BEFORE the add               0.010   not done
  same page, action list CLAIMING the click         0.030   not done
  splitkb cart, product name swapped in the text    0.000   not done
  Wikipedia article, goal naming another person     0.000   not done

Six pages is a small table and the two populations are 0.97 apart in it, so the floor is
not what separates them today; it is there for the page nobody has measured yet.
Re-measure rather than nudge."""

REST_QUIET_MS = 500.0
"""The quiet window :func:`observe_at_rest` asks for: the admission gate's
``REST_WINDOW_MS``, and wide enough to span the 25-45ms a page takes to START answering a
click, which is the measured reason a zero-window check reads quiet too early."""

_QUOTED = re.compile(r"\"([^\"]{2,})\"|“([^”]{2,})”|'([^']{2,})'")


def enabled() -> bool:
    """Whether :data:`ENV_SWITCH` leaves the Jev judge on (the default)."""
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in ("0", "false", "off", "no")


def quoted_arguments(goal: str) -> tuple[str, ...]:
    """The strings a goal puts in quotes - the task's own arguments, verbatim."""
    return tuple(next(g for g in match.groups() if g) for match in _QUOTED.finditer(goal))


class _Snapshot(Protocol):
    @property
    def url(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def digest(self) -> str: ...


class _Source(Protocol):
    @property
    def last(self) -> Any: ...


class DoneJudge(Protocol):
    """What :class:`JevDoneCritic` needs of ``JevPolicy``."""

    def judge_done(
        self,
        goal: str,
        end: Any,
        *,
        start_url: str = "",
        start_title: str = "",
        actions: Sequence[str] = (),
    ) -> Any: ...


def observe_at_rest(perceiver: Perceiver, controller: Controller) -> Observation:
    """Observe once the page has STOPPED ANSWERING, for the reader that judges a screen a
    code-speed skill has just left.

    On a controller that can ``quiesce`` - both browser controllers - this is upstream's
    quiet window (no DOM mutation and no finished request for :data:`REST_QUIET_MS`),
    re-armed when the document goes away underneath it, because the answer to the last
    click is often a redirect and the page worth judging is the one it lands on; then ONE
    read. The first version of this polled the fingerprint the way the admission gate's
    ``_observe_at_rest`` does, and on live splitkb.com it cost 5.6s for three reads that
    never came to rest, each one blocked behind the redirect it was trying to outlast.
    That loop is kept only for a controller with no ``quiesce``.
    """
    from skillweaver.skills.synthesize import REST_BUDGET_MS, REST_POLL_MS, REST_WINDOW_MS

    started = time.monotonic()

    def spent_ms() -> float:
        return (time.monotonic() - started) * 1000.0

    quiesce = getattr(controller, "quiesce", None)
    if callable(quiesce):
        waits = 0
        while spent_ms() < REST_BUDGET_MS:
            waits += 1
            settled = quiesce(REST_QUIET_MS, max(REST_BUDGET_MS - spent_ms(), REST_QUIET_MS))
            if settled is not None:
                break
            time.sleep(REST_POLL_MS / 1000.0)  # the document went away; let the next one exist
        seen = perceiver.observe(controller)
        log.info("critic.final.at_rest", how="quiesce", waits=waits, waited_ms=round(spent_ms()))
        return seen

    seen = perceiver.observe(controller)
    still_since, reads = time.monotonic(), 1
    while True:
        rested = (time.monotonic() - still_since) * 1000.0 >= REST_WINDOW_MS
        if rested or spent_ms() >= REST_BUDGET_MS:
            log.info(
                "critic.final.at_rest",
                how="fingerprint",
                rested=rested,
                reads=reads,
                waited_ms=round(spent_ms()),
            )
            return seen
        time.sleep(REST_POLL_MS / 1000.0)
        latest = perceiver.observe(controller)
        reads += 1
        if latest.fingerprint.value != seen.fingerprint.value:
            still_since = time.monotonic()
        seen = latest


class JevDoneCritic:
    """A ``Critic`` for the claim that a TASK is complete, decided by Jev and nobody else.

    Args:
        judge: The run's ``JevPolicy`` - its session, its validation, its meter.
        source: The ``DomPerceiver`` the run observes through; ``last`` is the END page.
        free: The model-free checks that run first. ``None`` is the cold path's: the
            ``no_error_state`` veto, plus the literal start-page veto this class adds.
            The warm path passes ``_warm_critic``'s ``TieredCritic`` built with NO model,
            whose evidence or corroboration can also say yes for free.
        rest: ``(perceiver, controller)`` to re-read the end screen at rest when the
            instant read did not earn a free yes - the warm path. ``None`` judges the
            observation it is handed, which on the cold path is the fresh look
            ``JevDriver`` already rested.
        floor: :data:`MIN_SATISFIED`.

    The explorer tells it where the run started (:meth:`open_run`) and what was done
    (:meth:`brief_final`); without either it still judges, on less.
    """

    def __init__(
        self,
        judge: DoneJudge,
        source: _Source,
        *,
        free: TieredCritic | None = None,
        rest: tuple[Perceiver, Controller] | None = None,
        floor: float = MIN_SATISFIED,
    ) -> None:
        self._judge = judge
        self._source = source
        self._free = free if free is not None else TieredCritic(None, require_change=False)
        self._literal_veto = free is None
        self._rest = rest
        self._floor = floor
        self._start: tuple[int, _Snapshot] | None = None
        self._actions: tuple[str, ...] = ()

    def open_run(self, first: Observation) -> None:
        """Remember the page the run started on, as the perceiver read it."""
        snapshot = self._source.last
        self._start = None if snapshot is None else (id(first), snapshot)
        self._actions = ()

    def brief_final(self, actions: Sequence[str]) -> None:
        """The literal actions performed so far, in order, for the judge to read."""
        self._actions = tuple(actions)

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> CriticVerdict:
        """Whether ``goal`` is achieved on the screen the run now stands on."""
        free = self._free.judge(goal, before, after, expectation)
        if self._rest is not None and not free.ok:
            # Only a free YES is believed on the instant read - it is what a warm replay
            # has always been judged on, and live splitkb paid 3.6s of rest to be told
            # the same thing. A miss or a veto may just be the page still answering, so
            # THAT is re-read at rest before anyone is asked anything.
            after = observe_at_rest(*self._rest)
            free = self._free.judge(goal, before, after, expectation)
        if free.policy != "inconclusive-no-model":
            log.info("critic.final", judge="checks", policy=free.policy, ok=free.ok)
            return free
        end = self._source.last
        if end is None:
            return self._refused("jev-unread", "the end page's DOM was never read", free, 0.0)
        start = self._start[1] if self._start and self._start[0] == id(before) else None
        if self._literal_veto and start is not None and start.digest == end.digest:
            log.info("critic.final", judge="checks", policy="literal-unchanged", ok=False)
            return self._refused(
                "literal-unchanged",
                "the page is literally the page the run started on: same address, text, "
                "control values and scroll position",
                free,
                1.0,
                source="programmatic",
            )
        try:
            answer = self._judge.judge_done(
                goal,
                end,
                start_url=start.url if start is not None else (before.url or ""),
                start_title=start.title if start is not None else "",
                actions=self._actions,
            )
        except ProviderError as exc:
            log.warning("critic.final", judge="jev", choice="error", error=str(exc))
            return self._refused(
                "jev-error", f"the Jev judge could not be asked, so the claim is refused: {exc}",
                free, 0.0,
            )  # fmt: skip
        choice, p = str(answer.choice), float(answer.probability)
        arrived = self._arrived(goal, start, end)
        log.info(
            "critic.final",
            judge="jev",
            choice=choice,
            p=round(p, 3),
            probabilities={k: round(float(v), 3) for k, v in answer.probabilities.items()},
            ms=round(float(answer.ms)),
            floor=self._floor,
            quoted=arrived,
        )
        told = f"Jev judged the end page {choice} (p={p:.2f}, {answer.ms:.0f}ms, DOM text only)"
        if choice == "satisfied" and p >= self._floor:
            return CriticVerdict(
                ok=True,
                reason=f"{told}{f'; {arrived}' if arrived else ''}",
                confidence=p,
                source="model",
                escalated=True,
                policy="jev-satisfied",
                checks=free.checks,
            )
        if choice == "satisfied":
            return self._refused(
                "jev-weak", f"{told}, under the {self._floor:.2f} it is believed at", free, p
            )
        return self._refused(f"jev-{choice.replace('_', '-')}", told, free, p)

    @staticmethod
    def _arrived(goal: str, start: _Snapshot | None, end: _Snapshot) -> str:
        """What became of the goal's quoted arguments - evidence for a reader, never a vote."""
        found = []
        for argument in quoted_arguments(goal):
            needle = " ".join(argument.lower().split())
            at_end = needle in " ".join(f"{end.title} {end.text}".lower().split())
            at_start = start is not None and needle in " ".join(
                f"{start.title} {start.text}".lower().split()
            )
            state = (
                "absent from the end page"
                if not at_end
                else "already on the start page"
                if at_start
                else "new on the end page"
            )
            found.append(f"{argument!r} is {state}")
        return "; ".join(found)

    @staticmethod
    def _refused(
        policy: str,
        reason: str,
        free: CriticVerdict,
        confidence: float,
        *,
        source: str = "model",
    ) -> CriticVerdict:
        return CriticVerdict(
            ok=False,
            reason=reason,
            confidence=confidence,
            source=source,  # type: ignore[arg-type]
            escalated=source == "model" and policy not in ("jev-unread",),
            policy=policy,
            checks=free.checks,
        )


DoneCriticFactory = Callable[[TieredCritic | None, bool], JevDoneCritic]
"""``(free checks or None for the cold path, read the end screen at rest?)`` -> the judge.
What ``orchestrator._open_done_critic`` hands ``build_agent``."""
