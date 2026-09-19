"""The critic: "did that actually work?", answered as cheaply as it honestly can be.

Nothing else in this project can learn without an answer to that question. The explorer
needs it to know which action to keep, synthesis needs it to admit a skill, and the warm
path - replaying a saved skill instead of thinking - is only trustworthy because a critic
re-checks the result. So the critic is on the hot path of every run, which makes its
second job cost: *most of this project's efficiency claim is not asking a vision model a
question a free check already answered.*

The cost policy
---------------

:class:`TieredCritic` decides in four tiers, and stops at the first one that reaches a
conclusion:

1. **Run the programmatic checks** from :mod:`skillweaver.agent.checks`. They are
   deterministic, take microseconds, and each can return ``passed``, ``failed`` or
   ``unknown``.
2. **If they are decisive, return that verdict and make NO model call at all.** Decisive
   means either of:

   * any check FAILED - a failing check is proof of failure, so nothing is gained by
     paying a model to agree. This covers the two commonest outcomes in practice: the
     screen did not change, and an error message appeared.
   * every *evidence* check PASSED, and there was at least one. See below for why a
     passing veto is not enough.

3. **If a corroborating check passed**, return a decisive yes, again with no model
   call. Corroboration is evidence that can only say yes (see below).
4. **Only when the checks are genuinely inconclusive**, send the before and after
   screenshots plus the goal to the vision model, and return its judgment with
   ``source="model"``. Inconclusive means no evidence or veto failed AND there was
   nothing that could establish success - no evidence check was configured, one of
   them returned ``unknown``, or the corroboration did not come through.

Evidence, corroboration and vetoes
----------------------------------

The three roles are not symmetric, and conflating them is how a critic starts lying.

An **evidence** check can prove success *and* failure: ``element_present("Payment
confirmed")``, ``matches_state(the one screen this can possibly end on)``, whatever the
caller passes in. When all of them pass, the goal was reached; when one fails, it was
not. It is authority, so only give it a check whose failure really is the task failing.

A **veto** can only prove failure. ``state_changed()`` failing means the step did nothing;
``state_changed()`` *passing* means only that something moved, which is not success -
clicking the wrong button also changes the screen. ``no_error_state()`` is the same shape.
So vetoes are consulted for a decisive no and are worth nothing towards a yes, and a
critic configured with vetoes alone escalates every time it is asked about a screen that
merely changed.

A **corroboration** check is the mirror image of a veto: it can only prove success.
Passing is a decisive yes; failing decides nothing and falls through to the model. It
exists because of a measured bug. The warm path used to hand :meth:`judge` the
fingerprint of the screen the ORIGINAL learned run ended on as ``expected_state`` -
decisive evidence - and for any task whose end screen depends on its argument that turns
a correct replay into a failure. A skill learned from "Search Wikipedia for computer
vision" and replayed for "machine learning" ran clean, passed its own verifier, and was
still rejected: one run ends on /wiki/Computer_vision and the other on
/wiki/Machine_learning, similarity 0.120. A recalled end screen is a *shortcut to a free
yes*, never a reason to call a working skill broken, and ``corroborating_state`` is that
shortcut with its authority removed. What still fails such a run is unchanged: the
vetoes, any real evidence check, and the model that is now asked when the shortcut misses.

Which path was taken
--------------------

:meth:`TieredCritic.judge` returns a :class:`CriticVerdict` - a real
:class:`~skillweaver.contracts.Verdict`, so any caller typed against the Protocol is
unaffected - carrying ``escalated``, a ``policy`` slug naming the rule that fired, and
``checks``, the full verdict of every check that ran. A reader can therefore see not only
what the critic decided but what it cost and why, which is the only way a cost policy
stays honest once it is out of sight.

Failure behavior
----------------

* A model reply that cannot be parsed, or that claims success without naming any visible
  evidence, **degrades to an honest low-confidence "I do not know"** (``ok=False``,
  ``confidence=0.0``, ``source="model"``) rather than to a false success.
* A model *call* that fails raises :class:`~skillweaver.errors.ProviderError`, per the
  :class:`~skillweaver.contracts.Critic` Protocol. It is not degraded: a provider outage
  is an infrastructure problem the caller must see, not a judgment about the screen.
* With no model configured at all (``llm=None``), an inconclusive case returns
  ``ok=False, confidence=0.0`` and says so, and never pretends a model was asked.

The adapters in :mod:`skillweaver.llm` are verified against recorded cassettes and have
not yet made a live call; this module codes against the
:class:`~skillweaver.contracts.LLMClient` interface and is tested the same way.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from skillweaver.agent.checks import Check, CheckVerdict, Outcome, no_error_state, run_all
from skillweaver.agent.checks import matches_state as matches_state_check
from skillweaver.agent.checks import state_changed as state_changed_check
from skillweaver.contracts import (
    Fingerprint,
    LLMClient,
    LLMMessage,
    Observation,
    Verdict,
)
from skillweaver.logging_ import get_logger

__all__ = [
    "MIN_EVIDENCE_CHARS",
    "MODEL_CONFIDENCE_CAP",
    "PROMPT_PATH",
    "CriticVerdict",
    "TieredCritic",
    "load_prompt",
]

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.md"
"""The judging prompt, used as the system prompt of the escalated call."""

MODEL_CONFIDENCE_CAP = 0.9
"""Ceiling applied to the model's self-reported confidence.

A model is only ever consulted here because the deterministic evidence was absent, so by
construction its answer rests on a reading of pixels rather than on a match. Capping it
below the ``1.0`` a fingerprint match earns keeps ``confidence`` comparable across the two
sources - a caller sorting verdicts by confidence should never see a model's opinion
outrank a measurement.
"""

MIN_EVIDENCE_CHARS = 12
"""Shortest ``evidence`` string accepted alongside a claimed success.

The prompt demands something concrete and visible. ``"yes"``, ``"done"`` and ``"it
worked"`` are all shorter than this, and all of them are the model agreeing rather than
looking. A success claim with less evidence than this degrades to "I do not know".
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@cache
def load_prompt() -> str:
    """The contents of ``prompts/critic.md``. Read once per process.

    Raises:
        OSError: if the prompt file is missing from the installed package.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class CriticVerdict(Verdict):
    """A :class:`~skillweaver.contracts.Verdict` that shows its own cost.

    Attributes:
        escalated: ``True`` exactly when a model call was made. ``False`` on every
            verdict the programmatic checks decided - assert this to prove a path is free.
        policy: A slug naming the rule that fired, one of ``"check-failed"``,
            ``"evidence-passed"``, ``"corroborated"``, ``"inconclusive-no-model"``,
            ``"model"``, ``"model-unparseable"`` or ``"model-unevidenced"``.
        checks: Every check that ran, in order, with its own verdict. Present on an
            escalated verdict too, so the reader can see what was inconclusive.
    """

    escalated: bool = False
    policy: str = ""
    checks: tuple[CheckVerdict, ...] = ()

    def summary(self) -> str:
        """A one-line, log-friendly account of the decision and what it cost."""
        cost = "1 model call" if self.escalated else "no model call"
        return (
            f"{'ok' if self.ok else 'not ok'} ({self.confidence:.2f}, {self.source}) "
            f"via {self.policy}, {len(self.checks)} check(s), {cost}: {self.reason}"
        )


class TieredCritic:
    """A :class:`~skillweaver.contracts.Critic` that pays for a model only when it must.

    Args:
        llm: The vision model consulted when the checks are inconclusive. ``None`` means
            programmatic-only: an inconclusive case then returns an honest
            ``ok=False, confidence=0.0`` instead of asking anyone.
        evidence: Checks that can prove success. When every one of them passes the
            verdict is a decisive yes; when any fails it is a decisive no.
        expected_state: Shorthand for adding ``matches_state(expected_state)`` to
            ``evidence`` - the way a run that can only end on ONE screen says so.
            A task whose end screen depends on its argument wants
            ``corroborating_state`` instead; see the module docstring.
        corroboration: Checks that can only prove success. All of them passing is a
            decisive yes; one failing decides nothing and escalates.
        corroborating_state: Shorthand for adding ``matches_state(corroborating_state)``
            to ``corroboration`` - a recalled end screen offered as a free shortcut to
            "yes" with no power to say "no".
        require_change: Adds ``state_changed()`` as a veto (default). Turn it off for a
            step that is meant to leave the screen alone, or the critic will call every
            such step a failure.
        check_errors: Adds ``no_error_state()`` as a veto (default).
        vetoes: Extra checks that can only prove failure.
        max_tokens: Cap on the escalated reply. The prompt asks for a small JSON object,
            so this is deliberately tight.

    The configuration is per-critic because :meth:`judge`'s signature is fixed by the
    Protocol. Use :meth:`expecting` to derive a critic for one particular step.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        evidence: Sequence[Check] = (),
        expected_state: Fingerprint | None = None,
        corroboration: Sequence[Check] = (),
        corroborating_state: Fingerprint | None = None,
        require_change: bool = True,
        check_errors: bool = True,
        vetoes: Sequence[Check] = (),
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._require_change = require_change
        self._check_errors = check_errors
        derived: list[Check] = []
        if expected_state is not None:
            derived.append(matches_state_check(expected_state))
        self._evidence: tuple[Check, ...] = (*derived, *evidence)
        hinted: list[Check] = []
        if corroborating_state is not None:
            hinted.append(matches_state_check(corroborating_state))
        self._corroboration: tuple[Check, ...] = (*hinted, *corroboration)
        built: list[Check] = []
        if require_change:
            built.append(state_changed_check())
        if check_errors:
            built.append(no_error_state())
        self._vetoes: tuple[Check, ...] = (*built, *vetoes)

    def expecting(
        self,
        *evidence: Check,
        expected_state: Fingerprint | None = None,
        corroboration: Sequence[Check] = (),
        corroborating_state: Fingerprint | None = None,
    ) -> TieredCritic:
        """A copy of this critic with extra evidence for one step. Shares the model."""
        clone = TieredCritic(
            self._llm,
            evidence=(*self._evidence, *evidence),
            expected_state=expected_state,
            corroboration=(*self._corroboration, *corroboration),
            corroborating_state=corroborating_state,
            require_change=self._require_change,
            check_errors=self._check_errors,
            max_tokens=self._max_tokens,
        )
        clone._vetoes = self._vetoes
        return clone

    @property
    def evidence_checks(self) -> tuple[Check, ...]:
        """The checks that can prove success AND prove failure, in the order they run."""
        return self._evidence

    @property
    def corroboration_checks(self) -> tuple[Check, ...]:
        """The checks that can only prove success, in the order they run."""
        return self._corroboration

    @property
    def veto_checks(self) -> tuple[Check, ...]:
        """The checks that can only prove failure, in the order they run."""
        return self._vetoes

    # -- the Protocol ------------------------------------------------------------------

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> CriticVerdict:
        """Judge whether ``goal`` was achieved going from ``before`` to ``after``.

        Follows the cost policy in the module docstring. The returned verdict says which
        tier decided it (``escalated``, ``policy``) and carries every check's own verdict
        (``checks``).

        Raises:
            ProviderError: if a model was needed and the call failed.
        """
        evidence = run_all(self._evidence, before, after)
        corroboration = run_all(self._corroboration, before, after)
        vetoes = run_all(self._vetoes, before, after)
        results = tuple(evidence + corroboration + vetoes)

        # Corroboration is deliberately absent from this list. A check that can only
        # prove success has no vote on failure; letting one in here is exactly the bug
        # that demoted a warm replay for landing on the right screen for a DIFFERENT
        # argument than the one the skill was learned with.
        failures = [r for r in evidence + vetoes if r.outcome is Outcome.failed]
        if failures:
            return CriticVerdict(
                ok=False,
                reason="; ".join(f"{r.name}: {r.reason}" for r in failures),
                confidence=min(r.confidence for r in failures),
                source="programmatic",
                escalated=False,
                policy="check-failed",
                checks=results,
            )

        if evidence and all(r.outcome is Outcome.passed for r in evidence):
            return CriticVerdict(
                ok=True,
                reason="; ".join(f"{r.name}: {r.reason}" for r in evidence),
                confidence=min(r.confidence for r in evidence),
                source="programmatic",
                escalated=False,
                policy="evidence-passed",
                checks=results,
            )

        if corroboration and all(r.outcome is Outcome.passed for r in corroboration):
            return CriticVerdict(
                ok=True,
                reason="; ".join(f"{r.name}: {r.reason}" for r in corroboration),
                confidence=min(r.confidence for r in corroboration),
                source="programmatic",
                escalated=False,
                policy="corroborated",
                checks=results,
            )

        if self._llm is None:
            return CriticVerdict(
                ok=False,
                reason=(
                    "the programmatic checks were inconclusive and no model is configured: "
                    + (self._trail(results) or "no checks were applicable")
                ),
                confidence=0.0,
                source="programmatic",
                escalated=False,
                policy="inconclusive-no-model",
                checks=results,
            )

        return self._ask_model(goal, before, after, expectation, results, tuple(corroboration))

    # -- the escalated path ------------------------------------------------------------

    def _ask_model(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None,
        results: tuple[CheckVerdict, ...],
        corroboration: tuple[CheckVerdict, ...] = (),
    ) -> CriticVerdict:
        """Exactly one model call, then a verdict that never over-claims."""
        log.debug("critic.escalate", model=self._llm.name(), goal=goal)  # type: ignore[union-attr]
        response = self._llm.complete(  # type: ignore[union-attr]
            [self._message(goal, before, after, expectation, results, corroboration)],
            system=load_prompt(),
            max_tokens=self._max_tokens,
        )
        parsed = _parse_reply(response.text)
        if parsed is None:
            return self._degraded(
                "model-unparseable",
                "the model's reply could not be parsed as a verdict, so it is being read "
                f"as 'I do not know' rather than as a success: {_snippet(response.text)}",
                results,
            )

        ok = bool(parsed.get("ok"))
        evidence_text = str(parsed.get("evidence") or "").strip()
        reason = str(parsed.get("reason") or "").strip() or evidence_text
        if ok and len(evidence_text) < MIN_EVIDENCE_CHARS:
            return self._degraded(
                "model-unevidenced",
                "the model claimed success without naming anything it could see "
                f"(evidence: {evidence_text!r}), so it is being read as 'I do not know': "
                f"{reason or _snippet(response.text)}",
                results,
            )

        confidence = min(_as_confidence(parsed.get("confidence")), MODEL_CONFIDENCE_CAP)
        return CriticVerdict(
            ok=ok,
            reason=f"{reason} [seen: {evidence_text}]" if evidence_text else reason,
            confidence=confidence,
            source="model",
            escalated=True,
            policy="model",
            checks=results,
        )

    def _degraded(
        self, policy: str, reason: str, results: tuple[CheckVerdict, ...]
    ) -> CriticVerdict:
        """An escalated verdict that refuses to conclude anything. Never ``ok``."""
        log.warning("critic.degrade", policy=policy, reason=reason)
        return CriticVerdict(
            ok=False,
            reason=reason,
            confidence=0.0,
            source="model",
            escalated=True,
            policy=policy,
            checks=results,
        )

    def _message(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None,
        results: tuple[CheckVerdict, ...],
        corroboration: tuple[CheckVerdict, ...] = (),
    ) -> LLMMessage:
        """The one user turn: the goal, what the cheap checks found, and both frames.

        The corroboration is quoted in a section of its own, and said to be worthless
        as evidence of failure. Folding it into the main trail would hand the model a
        line reading "the screen is not the expected state" and invite it to agree -
        which is the veto this role exists to remove, laundered through the model.
        """
        lines = [f"GOAL: {goal}"]
        if expectation:
            lines.append(f"EXPECTED AFTER THE ATTEMPT: {expectation}")
        if before.url or after.url:
            lines.append(f"URL BEFORE: {before.url or '(none)'}")
            lines.append(f"URL AFTER: {after.url or '(none)'}")
        decisive = tuple(r for r in results if r not in corroboration)
        trail = self._trail(decisive)
        lines.append(
            "DETERMINISTIC CHECKS (they could not decide, which is why you are being "
            f"asked):\n{trail}"
            if trail
            else "DETERMINISTIC CHECKS: none were applicable."
        )
        if corroboration:
            lines.append(
                "CORROBORATION - a shortcut to a free 'yes' that did not fire. It is "
                "NOT evidence of failure: it compares this screen against the one a "
                "PREVIOUS run of this task ended on, and a task run with different "
                "arguments correctly ends somewhere else. Judge the goal against what "
                "you can see, and ignore this section unless it passed:\n"
                f"{self._trail(corroboration)}"
            )
        lines.append(
            "The first image is the screen BEFORE the attempt; the second is the screen "
            "AFTER it. Reply with the JSON object described in your instructions."
        )
        return LLMMessage(
            role="user",
            text="\n\n".join(lines),
            images=(before.screenshot.png, after.screenshot.png),
        )

    @staticmethod
    def _trail(results: Sequence[CheckVerdict]) -> str:
        return "\n".join(f"- {r.name} [{r.outcome.value}]: {r.reason}" for r in results)


# --------------------------------------------------------------------------------------
# Parsing a model reply, defensively
# --------------------------------------------------------------------------------------


def _parse_reply(text: str) -> dict[str, Any] | None:
    """The JSON object in a model reply, or ``None`` when there is not one.

    Tolerates a code fence and surrounding prose - both are things models do despite
    being asked not to, and neither is a reason to throw away a real answer. Returns
    ``None`` for anything else, including valid JSON that is not an object or that never
    says ``ok``, because a verdict without a decision in it is not a verdict.
    """
    if not text or not text.strip():
        return None
    candidates = [text.strip()]
    match = _JSON_BLOCK.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and "ok" in data:
            return data
    return None


def _as_confidence(value: Any) -> float:
    """A model-supplied confidence clamped into ``0.0..1.0``; ``0.5`` when unusable.

    A missing or nonsense confidence is not a reason to discard an otherwise good
    judgment, but it must not become a confident one either, so it lands in the middle.
    """
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5


def _snippet(text: str, limit: int = 160) -> str:
    """A short, single-line quote of a model reply, for a human reading a log."""
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if not flat:
        return "(empty reply)"
    return repr(flat if len(flat) <= limit else flat[: limit - 3] + "...")
