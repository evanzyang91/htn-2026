"""The critic: "did that actually work?", answered as cheaply as it honestly can be.

Most of this project's efficiency claim is not asking a vision model a question a free
check already answered. :class:`TieredCritic` stops at the first tier that concludes:
run the programmatic checks; if any FAILED, or every EVIDENCE check passed and there was
one, return that with no model call; if a CORROBORATING check passed, a free yes; only
then escalate to the vision model.

**The three roles are not symmetric, and conflating them is how a critic starts lying.**
EVIDENCE proves success and failure, so only give it a check whose failure really is the
task failing. A VETO can only prove failure - ``state_changed()`` passing means only
that something moved, and clicking the wrong button also changes the screen.
CORROBORATION is the mirror: passing is decisive, failing decides nothing.

Corroboration exists because of a measured bug. The warm path handed :meth:`judge` the
fingerprint of the ORIGINAL learned run's end screen as ``expected_state`` - decisive
evidence - which fails any task whose end screen depends on its argument: a skill
learned from "Search Wikipedia for computer vision" and replayed for "machine learning"
ran clean, passed its own verifier and was still rejected at similarity 0.120.

An unparseable model reply, or one claiming success without naming visible evidence,
degrades to an honest ``ok=False, confidence=0.0``. A failing model CALL raises
ProviderError instead: an outage is infrastructure, not a judgment about the screen.

**The judge is never handed a tool, and a reply that is not a verdict is asked for once
more before it degrades.** A run's client is built with ``computer_use=True``, which
appends the computer tool to every request, and a judge shown two screenshots beside a
screenshot tool sometimes CALLS it and says nothing: the ``(empty reply)`` degradations
that reported finished errands NOT SOLVED, and on the Jev path sent a correct ``DONE``
back through the policy to be claimed again (~5s a time, measured). :func:`text_only` is
the door every ask goes through, and :data:`REASK_MAX_TOKENS` is why the second ask is
given more room than the first. The re-ask is the SAME evidence and the SAME gates - it
can turn "I do not know" into a verdict, never a no into a yes.
"""

from __future__ import annotations

import copy
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
    "REASK_MAX_TOKENS",
    "CriticVerdict",
    "TieredCritic",
    "load_prompt",
    "text_only",
]

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.md"
"""The judging prompt, used as the system prompt of the escalated call."""

MODEL_CONFIDENCE_CAP = 0.9
"""Ceiling applied to the model's self-reported confidence.

A model is consulted here only because the deterministic evidence was absent, so its
answer rests on a reading of pixels rather than a match. Capping it below the ``1.0`` a
fingerprint match earns keeps ``confidence`` comparable across the two sources."""

MIN_EVIDENCE_CHARS = 12
"""Shortest ``evidence`` string accepted alongside a claimed success. ``"yes"``,
``"done"`` and ``"it worked"`` are all shorter, and all of them are the model agreeing
rather than looking; a claim with less evidence degrades to "I do not know"."""

REASK_MAX_TOKENS = 2048
"""``max_tokens`` for the ONE re-ask a reply that is not a verdict earns.

``claude-opus-5`` thinks by default and thinking is billed against ``max_tokens``, so a
first ask at 512 can end on ``max_tokens`` with no text at all - the same empty reply a
tool call produces, for a different reason. The re-ask is given room for both; the first
ask keeps its small ceiling because that is what bounds the common case."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@cache
def load_prompt() -> str:
    """The contents of ``prompts/critic.md``. Read once per process.

    Raises:
        OSError: if the prompt file is missing from the installed package.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


def text_only(llm: LLMClient) -> LLMClient:
    """``llm`` as a client whose requests carry no tool the caller did not pass.

    A client exposing ``text_only()`` is asked for its own. Otherwise a client that
    appends the computer tool to every request (``AnthropicClient(computer_use=True)``,
    recognised by that flag) is shallow-copied with the flag off: the copy SHARES the
    SDK client and the usage meter, so every call the judge makes is still charged to the
    run through ``total_usage`` - the standing rule that cost is read from the meter.
    Anything else is returned as it came.
    """
    own = getattr(llm, "text_only", None)
    if callable(own):
        return own()
    if getattr(llm, "_computer_use", False) is True:
        judge = copy.copy(llm)
        judge._computer_use = False  # type: ignore[attr-defined]  # noqa: SLF001
        return judge
    return llm


@dataclass(frozen=True, slots=True)
class CriticVerdict(Verdict):
    """A Verdict that shows its own cost.

    Attributes:
        escalated: ``True`` exactly when a model call was made - assert this to prove a
            path is free.
        policy: A slug naming the rule that fired, one of ``"check-failed"``,
            ``"evidence-passed"``, ``"corroborated"``, ``"inconclusive-no-model"``,
            ``"model"``, ``"model-unparseable"`` or ``"model-unevidenced"``.
        checks: Every check that ran, in order. Present on an escalated verdict too.
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
    """A Critic that pays for a model only when it must.

    Args:
        llm: Consulted when the checks are inconclusive. ``None`` means
            programmatic-only, returning an honest ``ok=False, confidence=0.0``.
        evidence: Checks that can prove success. All passing is a decisive yes; one
            failing is a decisive no.
        expected_state: Shorthand for ``matches_state(...)`` in ``evidence`` - how a run
            that can only end on ONE screen says so. A task whose end screen depends on
            its argument wants ``corroborating_state``.
        corroboration: Checks that can only prove success. One failing escalates.
        corroborating_state: ``matches_state(...)`` in ``corroboration``: a recalled end
            screen as a free shortcut to "yes" with no power to say "no".
        require_change: Adds ``state_changed()`` as a veto (default). Turn it off for a
            step meant to leave the screen alone.
        check_errors: Adds ``no_error_state()`` as a veto (default).

    The configuration is per-critic because :meth:`judge`'s signature is fixed by the
    Protocol; use :meth:`expecting` to derive a critic for one step.
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
        self._llm = text_only(llm) if llm is not None else None
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

        The verdict says which tier decided it (``escalated``, ``policy``) and carries
        every check's own verdict.

        Raises:
            ProviderError: if a model was needed and the call failed.
        """
        evidence = run_all(self._evidence, before, after)
        corroboration = run_all(self._corroboration, before, after)
        vetoes = run_all(self._vetoes, before, after)
        results = tuple(evidence + corroboration + vetoes)

        # Corroboration is deliberately absent: a check that can only prove success has
        # no vote on failure. Letting one in here is the bug that demoted a warm replay
        # for landing on the right screen for a DIFFERENT argument.
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
        """One model call - two when the first reply is not a verdict - and a verdict that
        never over-claims.

        The re-ask is against the SAME message: same goal, same frames, same check trail.
        Without it a degraded ``done`` verdict goes back to the policy as a refusal, the
        policy claims ``DONE`` again on the unchanged screen, and the run pays a policy
        call and a fresh escalation to ask the identical question.
        """
        log.debug("critic.escalate", model=self._llm.name(), goal=goal)  # type: ignore[union-attr]
        message = self._message(goal, before, after, expectation, results, corroboration)
        verdict = self._ask_once(message, results, self._max_tokens)
        if verdict.policy == "model":
            return verdict
        log.warning("critic.reask", policy=verdict.policy, reason=verdict.reason)
        return self._ask_once(message, results, max(self._max_tokens, REASK_MAX_TOKENS))

    def _ask_once(
        self, message: LLMMessage, results: tuple[CheckVerdict, ...], max_tokens: int
    ) -> CriticVerdict:
        """Ask, parse, and apply the evidence gate. Degrades; never raises on a bad reply."""
        response = self._llm.complete(  # type: ignore[union-attr]
            [message], system=load_prompt(), max_tokens=max_tokens
        )
        parsed = _parse_reply(response.text)
        if parsed is None:
            return self._degraded(
                "model-unparseable",
                "the model's reply could not be parsed as a verdict, so it is being read "
                f"as 'I do not know' rather than as a success: {_snippet(response.text)} "
                f"[stop_reason={response.stop_reason}, "
                f"tool_calls={[call.name for call in response.tool_calls]}]",
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

        The corroboration is quoted separately and said to be worthless as evidence of
        failure: folding it into the main trail would hand the model a line reading "the
        screen is not the expected state" and invite it to agree, which is the veto this
        role exists to remove, laundered through the model.
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

    Tolerates a code fence and surrounding prose. ``None`` for anything else, including
    valid JSON that is not an object or never says ``ok``, because a verdict without a
    decision in it is not a verdict.
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
    """A model-supplied confidence clamped into ``0.0..1.0``; ``0.5`` when unusable. A
    nonsense confidence is not a reason to discard an otherwise good judgment, but it
    must not become a confident one either."""
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
