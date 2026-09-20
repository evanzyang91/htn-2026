"""The per-move judge of the fast cold path: did the page LITERALLY change, and no more.

``TieredCritic`` has no evidence to check a single move against, so its tiers collapse to
one rule: a move whose screen did not move is failed for free by the ``state_changed``
veto, and EVERY move whose screen did move escalates to a vision-model call. That is the
dominant cost of a cold Jev run - measured on a 21-move live run, 1.3-2.0s for a move
judged programmatically against 4.8-10s for one judged by the model, on roughly half the
moves - and it buys a verdict ``AGENTS.md`` already says nothing may be keyed on, because
it is wrong on a page that answers in place: it called correct *Add to cart* clicks
failures and the policy went on to add twelve breads.

:class:`LiteralMoveCritic` answers the question that CAN be answered for free and
truthfully - ``DomSnapshot.digest`` before against after, upstream Jev's ``page_changed``
and the same signal ``JevDriver`` already keeps its own history on - and says so in its
reason rather than dressing "the page answered" up as "the move did what it was for".

**What it must never judge is the ``done`` claim.** ``Explorer._judge`` gives it per-move
calls only; the claim that a TASK is complete still goes to the full critic, vision model
included, because the cheapest answer this architecture can give is a wrong one and a
change detector cannot tell a full cart from a wrong one.

What each consumer of a per-move verdict gets from it:

* ``FailureMemory`` / ``_exclusions``: a dead end is now "this move left the page exactly
  as it was", which is what ``JevDriver._spent`` already withholds on. A wrong-but-changing
  move is no longer filed - the policy's ``refused`` relay dropped those already.
* the site graph: an edge succeeds when its action moved the page, which is what routing
  wants to know about an edge.
* the trajectory: every move still carries a verdict, which ``moves_of`` needs as the move
  boundary, and the synthesizer reads ACCEPTED/REJECTED with a reason that says exactly
  how little was checked.

The ``no_error_state`` veto still runs: it is free, and an error page that "changed" is
not progress.
"""

from __future__ import annotations

from typing import Protocol

from skillweaver.agent.checks import CheckVerdict, Outcome, no_error_state
from skillweaver.agent.critic import CriticVerdict
from skillweaver.contracts import Observation
from skillweaver.logging_ import get_logger

__all__ = ["LiteralMoveCritic", "SnapshotSource"]

log = get_logger(__name__)


class _Digested(Protocol):
    @property
    def digest(self) -> str: ...

    @property
    def url(self) -> str: ...


class SnapshotSource(Protocol):
    """What :class:`LiteralMoveCritic` needs of a perceiver: ``DomPerceiver.last``."""

    @property
    def last(self) -> _Digested | None: ...


class LiteralMoveCritic:
    """A ``Critic`` for ONE MOVE that never calls a model.

    Args:
        source: The ``DomPerceiver`` the explorer observes through. It holds ONE snapshot
            slot - the last observation's - so the screen a move starts from has to be
            read BEFORE the move is performed: :meth:`open_move`, which the explorer calls.

    A move whose digests cannot be read (no ``open_move``, or a perceiver with no snapshot)
    is answered ``ok=False, confidence=0.0`` - "I do not know" - and never guessed.
    """

    def __init__(self, source: SnapshotSource) -> None:
        self._source = source
        self._opened: tuple[int, str, str] | None = None
        self._errors = no_error_state()

    def open_move(self, before: Observation) -> None:
        """Remember what the page literally was as a move starts."""
        snapshot = self._source.last
        self._opened = (
            None if snapshot is None else (id(before), snapshot.digest, snapshot.url or "")
        )

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> CriticVerdict:
        """Whether the page literally changed between ``before`` and ``after``."""
        del goal, expectation  # a change detector reads neither, and says so in its reason
        errors: CheckVerdict = self._errors(before, after)
        opened, self._opened = self._opened, None
        snapshot = self._source.last
        if errors.outcome is Outcome.failed:
            return self._verdict(False, f"{errors.name}: {errors.reason}", 1.0, "error", errors)
        if opened is None or opened[0] != id(before) or snapshot is None:
            log.warning("move_critic.unread", opened=opened is not None)
            return self._verdict(
                False,
                "the page's literal state before or after this move could not be read, so "
                "the move is not judged either way",
                0.0,
                "literal-unread",
                errors,
            )
        _, digest, url = opened
        if (snapshot.url or "") != url:
            return self._verdict(
                True,
                f"the page moved to {snapshot.url} (literal change only: whether that is "
                "what the move was for was NOT judged)",
                1.0,
                "literal-arrived",
                errors,
            )
        if snapshot.digest != digest:
            return self._verdict(
                True,
                "the page's text, controls or scroll position changed in place (literal "
                "change only: whether that is what the move was for was NOT judged)",
                1.0,
                "literal-changed",
                errors,
            )
        return self._verdict(
            False,
            "the page is literally what it was before this move: same address, text, "
            "control values and scroll position",
            1.0,
            "literal-unchanged",
            errors,
        )

    @staticmethod
    def _verdict(
        ok: bool, reason: str, confidence: float, policy: str, errors: CheckVerdict
    ) -> CriticVerdict:
        return CriticVerdict(
            ok=ok,
            reason=reason,
            confidence=confidence,
            source="programmatic",
            escalated=False,
            policy=policy,
            checks=(errors,),
        )
