"""Skill synthesis: turning one successful fumble into a skill the library can trust.

This is the self-growing half of skillweaver, and its whole value rests on one
property: **nothing enters the library without proving it works**. A model that has
watched a run will happily write plausible code for it. Plausible code that does not
run is the known failure mode of this entire approach - a library of it is worse than
an empty one, because a planner picks from it.

So there are two doors and only one of them opens::

    synthesizer = Synthesizer(llm, store, critic)

    candidate = synthesizer.synthesize(trajectory)     # a draft. NOT stored. Ever.
    admission = synthesizer.admit(trajectory, environment)
    admission.ok, admission.skill, admission.attempts  # stored only if ok

:meth:`Synthesizer.synthesize` is the ``Synthesizer`` Protocol: it writes a candidate
and hands it back. It cannot store, because it does not store - the only ``put`` in
this module is inside :meth:`Synthesizer.admit`, after the gate, and the store is
private to the synthesizer. There is no third path.

The gate, in order, for every attempt:

1. **Structure.** :func:`~skillweaver.skills.model.validate_skill` - a name, a
   one-line summary, a docstring, params that agree with ``run``, a verifier.
2. **The sandbox's static scan.** An import, an ``open``, a dunder: refused here,
   before anything executes, and never admitted.
3. **The precondition.** The environment is put back to the recorded starting screen
   and must actually be there; a skill proved against the wrong screen proves nothing.
4. **Execution.** The skill is RE-RUN through
   :class:`~skillweaver.skills.sandbox.SkillRunner` against that environment with the
   model's own example arguments, verifier included.
5. **The critic.** :class:`~skillweaver.contracts.Critic` judges the screen before
   against the screen after, for the task the trajectory was solving.

Only then is it stored. A failure at any stage is fed back to the model - the error
AND the sandbox's trace, which lists every action and log line in order - and the
skill is rewritten, up to ``max_repairs`` times. Exhausting them is a clean
``Admission(ok=False)`` with every attempt attached, not an exception: a synthesizer
that could not write this skill has not broken, it has simply not written it.

Before any of that the draft goes through :mod:`skillweaver.skills.refactor`, which
replaces literal coordinates with perception lookups and lifts this run's data into
parameters. Hardening first, admission second: what is judged is what is stored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from skillweaver.contracts import (
    Controller,
    Critic,
    Element,
    GraphView,
    LLMClient,
    LLMMessage,
    Observation,
    Perceiver,
    Provenance,
    Skill,
    SkillResult,
    SkillStore,
    Trajectory,
    Verdict,
    utcnow,
)
from skillweaver.errors import SandboxViolation
from skillweaver.logging_ import get_logger
from skillweaver.skills.api import SkillLimits, describe_action
from skillweaver.skills.model import SkillInvalid, make_skill
from skillweaver.skills.refactor import Hardening, harden
from skillweaver.skills.sandbox import SkillRunner, scan_code

__all__ = [
    "PROMPT",
    "Admission",
    "Attempt",
    "ReplayEnvironment",
    "Stage",
    "Synthesizer",
    "load_prompt",
]

log = get_logger(__name__)

PROMPT = "synthesize.md"
"""The generation prompt, next to this project's other prompts in
``skillweaver/agent/prompts/``. It states the published ``ctx`` surface exactly,
forbids imports, and demands a verifier; :func:`load_prompt` reads it."""

Stage = Literal["generation", "structure", "sandbox", "precondition", "execution", "critic"]
"""Where an attempt stopped. Everything before ``execution`` is decided without
touching the environment at all."""

_MAX_ELEMENTS = 18
"""Elements described per recorded screen. Enough to write a lookup against, short
enough that a long list does not bury the ones that were acted on."""

_MAX_TEXT = 80
_FENCE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


@lru_cache(maxsize=4)
def load_prompt(name: str = PROMPT) -> str:
    """The text of a prompt shipped in ``skillweaver.agent.prompts``.

    Cached: the file does not change while a process runs, and synthesis reads it
    on every call.

    Raises:
        FileNotFoundError: if no such prompt is packaged.
    """
    return (resources.files("skillweaver.agent.prompts") / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# What the gate runs against
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayEnvironment:
    """A world to re-execute a candidate skill in, standing at the recorded start.

    ``controller`` and ``perceiver`` are what the skill acts and sees through;
    ``graph`` is the read-only site graph it may consult, if there is one.

    The gate asks for a FRESH one per attempt - see ``environment`` in
    :meth:`Synthesizer.admit` - so a repair never inherits the half-finished screen
    its predecessor left behind, which would let a broken skill pass by accident.
    """

    controller: Controller
    perceiver: Perceiver
    graph: GraphView | None = None


EnvironmentFactory = Callable[[], ReplayEnvironment]
"""Called once per admission attempt; must return the recorded environment reset to
the trajectory's first screen (``controller.reset()``, a fresh page, a new browser)."""


# --------------------------------------------------------------------------------------
# What came of it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attempt:
    """One pass through the gate: what was tried and how far it got.

    Attributes:
        index: ``0`` for the first generation, ``1..n`` for repairs.
        stage: Where it stopped, or ``"critic"`` when it passed everything.
        ok: Whether this attempt was admitted.
        skill: The candidate as it was judged - hardened, validated, NOT stored.
            ``None`` when the model returned nothing usable.
        error: Why it failed, in the words the model is shown. ``None`` when ``ok``.
        trace: The sandbox trace of the run, when it got as far as running.
        result: The raw :class:`~skillweaver.contracts.SkillResult`, when it ran.
        verdict: The critic's judgment, when one was asked for.
        hardening: What the hardening pass changed before this attempt.
    """

    index: int
    stage: Stage
    ok: bool
    skill: Skill | None = None
    error: str | None = None
    trace: tuple[str, ...] = ()
    result: SkillResult | None = None
    verdict: Verdict | None = None
    hardening: Hardening | None = None

    def __str__(self) -> str:
        head = f"attempt {self.index} ({self.stage})"
        return f"{head}: admitted" if self.ok else f"{head}: {self.error}"


@dataclass(frozen=True, slots=True)
class Admission:
    """The outcome of :meth:`Synthesizer.admit`.

    Attributes:
        ok: Whether a skill entered the library.
        skill: The STORED skill (``version >= 1``) when ``ok``, else ``None``. A
            non-``None`` value here is a promise: this code ran, and a critic agreed.
        attempts: Every attempt, in order. ``attempts[0]`` is the first generation.
        reason: One line saying why, for a log or a status line.
    """

    ok: bool
    skill: Skill | None
    attempts: tuple[Attempt, ...] = ()
    reason: str = ""

    @property
    def repairs(self) -> int:
        """How many times the skill was rewritten after the first draft."""
        return max(len(self.attempts) - 1, 0)

    def __str__(self) -> str:
        name = self.skill.name if self.skill else "no skill"
        return f"admission {'ok' if self.ok else 'rejected'} ({name}): {self.reason}"


@dataclass(frozen=True, slots=True)
class _Draft:
    """One parsed model reply."""

    raw: str
    name: str
    summary: str
    docstring: str
    code: str
    verifier_code: str | None
    params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    example_args: Mapping[str, Any] = field(default_factory=dict, hash=False)
    requires: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# Describing the run to the model
# --------------------------------------------------------------------------------------


def _describe_element(element: Element) -> str:
    text = element.text.strip().replace("\n", " ")
    if len(text) > _MAX_TEXT:
        text = text[: _MAX_TEXT - 1] + "…"
    box = element.box
    return f"  - {element.kind.value} {text!r} at ({box.x}, {box.y}) {box.w}x{box.h}"


def _describe_screen(observation: Observation, label: str) -> str:
    elements: Sequence[Element] = observation.elements[:_MAX_ELEMENTS]
    url = observation.url or "none"
    lines = [f"{label} (url: {url}, screen id: {observation.fingerprint.value})"]
    lines.extend(_describe_element(e) for e in elements)
    if len(observation.elements) > _MAX_ELEMENTS:
        lines.append(f"  - ... and {len(observation.elements) - _MAX_ELEMENTS} more elements")
    return "\n".join(lines)


def describe_trajectory(trajectory: Trajectory) -> str:
    """The recorded run as the text the model is asked to write a skill from.

    Deterministic for a given trajectory - no timestamps, no ordering by dict - so
    the same run produces the same request, which is what makes an LLM cassette
    replay and a repair prompt diffable.
    """
    parts = [
        f"TASK: {trajectory.task}",
        f"DOMAIN: {trajectory.domain}",
        f"STEPS: {len(trajectory.steps)}",
        "",
    ]
    if trajectory.steps:
        parts.append(_describe_screen(trajectory.steps[0].before, "STARTING SCREEN"))
        parts.append("")
    for step in trajectory.steps:
        parts.append(f"STEP {step.index}: {describe_action(step.action)}")
        if step.note:
            parts.append(f"  reason given at the time: {step.note}")
        parts.append(_describe_screen(step.after, "  screen after"))
        parts.append("")
    if trajectory.steps:
        parts.append(_describe_screen(trajectory.steps[-1].after, "FINAL SCREEN (the goal)"))
    parts.append("")
    parts.append("Write the skill for this task as the JSON object described above.")
    return "\n".join(parts)


def _repair_brief(attempt: Attempt) -> str:
    """What the model is shown after a rejection: the stage, the error, the trace."""
    lines = [
        "Your skill was REJECTED and has not been stored.",
        f"Stage: {attempt.stage}",
        f"Error: {attempt.error}",
    ]
    if attempt.trace:
        lines.append("")
        lines.append("Trace of the run, in order:")
        lines.extend(f"  {line}" for line in attempt.trace)
    if attempt.verdict is not None and not attempt.verdict.ok:
        lines.append("")
        lines.append(f"The critic said: {attempt.verdict.reason}")
    lines.append("")
    lines.append(
        "Fix exactly this and return the same JSON object shape again. "
        "Keep what was working; change what the error names."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Reading the reply
# --------------------------------------------------------------------------------------


def _json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in a model reply, fenced or bare, or ``None``."""
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group("body"))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_draft(text: str) -> tuple[_Draft | None, str]:
    """A draft from one reply, or ``(None, why)``."""
    data = _json_object(text)
    if data is None:
        return None, "the reply was not a JSON object; return only the JSON object"
    missing = [key for key in ("name", "summary", "docstring", "code") if not data.get(key)]
    if missing:
        return None, f"the JSON object is missing {', '.join(missing)}"
    verifier = data.get("verifier_code")
    if not verifier or not str(verifier).strip():
        return None, (
            "the skill has no verifier: return verifier_code defining "
            "def verify(ctx, result) that checks the end state with ctx.see"
        )
    requires = data.get("requires") or ()
    if isinstance(requires, str):
        requires = (requires,)
    params = data.get("params") or {}
    example = data.get("example_args") or {}
    if not isinstance(params, dict) or not isinstance(example, dict):
        return None, "params and example_args must both be JSON objects"
    return (
        _Draft(
            raw=text,
            name=str(data["name"]).strip(),
            summary=str(data["summary"]).strip(),
            docstring=str(data["docstring"]).strip(),
            code=str(data["code"]),
            verifier_code=str(verifier),
            params=params,
            example_args=example,
            requires=tuple(str(r) for r in requires),
        ),
        "",
    )


# --------------------------------------------------------------------------------------
# The synthesizer
# --------------------------------------------------------------------------------------


class Synthesizer:
    """A :class:`~skillweaver.contracts.Synthesizer` with an admission gate.

    Args:
        llm: Writes and repairs the code. Its ``name()`` is recorded in provenance.
        store: Where an ADMITTED skill is put. Private: the only call to ``put`` in
            this class is the one after the gate, so there is no way to store a skill
            that has not proved itself through this object.
        critic: Judges the replayed run. Required, for the same reason.
        max_repairs: How many times a rejected skill may be rewritten. ``0`` means
            one attempt and no repairs. Bounded because a model that cannot fix its
            code in three goes will not fix it in thirty, and every go costs money.
        limits: Sandbox limits for the admission run.
        min_steps: Runs shorter than this are not worth a skill; ``synthesize``
            returns ``None`` for them.
        min_similarity: How like the recorded starting screen the environment must
            be before a candidate is run in it (``1.0`` is the same screen).
        max_tokens, temperature: Passed to ``llm.complete``.

    Not thread-safe, and one instance may be reused across trajectories.
    """

    __slots__ = (
        "_critic",
        "_limits",
        "_llm",
        "_max_repairs",
        "_max_tokens",
        "_min_similarity",
        "_min_steps",
        "_prompt",
        "_store",
        "_temperature",
    )

    def __init__(
        self,
        llm: LLMClient,
        store: SkillStore,
        critic: Critic,
        *,
        max_repairs: int = 2,
        limits: SkillLimits | None = None,
        min_steps: int = 1,
        min_similarity: float = 1.0,
        max_tokens: int = 8000,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> None:
        if max_repairs < 0:
            raise ValueError(f"max_repairs must not be negative, got {max_repairs}")
        self._llm = llm
        self._store = store
        self._critic = critic
        self._max_repairs = max_repairs
        self._limits = limits if limits is not None else SkillLimits()
        self._min_steps = min_steps
        self._min_similarity = min_similarity
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._prompt = prompt if prompt is not None else load_prompt()

    def __repr__(self) -> str:
        return (
            f"Synthesizer(model={self._llm.name()!r}, "
            f"store={type(self._store).__name__}, max_repairs={self._max_repairs})"
        )

    # -- the Synthesizer protocol ----------------------------------------------------------

    def synthesize(self, trajectory: Trajectory) -> Skill | None:
        """Write a candidate skill from ``trajectory``: hardened, validated,
        ``version=0``, and **not stored**.

        ``None`` when the run failed, was too short to be worth a skill, or the model
        did not return usable code. Storing happens only in :meth:`admit`.

        Raises:
            ProviderError: if the model call fails.
        """
        if not self._worth_keeping(trajectory):
            return None
        response = self._llm.complete(
            [LLMMessage(role="user", text=describe_trajectory(trajectory))],
            system=self._prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        draft, why = _parse_draft(response.text)
        if draft is None:
            log.info("skill.synthesize.unusable", run_id=trajectory.run_id, why=why)
            return None
        candidate, error, _ = self._build(draft, trajectory)
        if candidate is None:
            log.info("skill.synthesize.invalid", run_id=trajectory.run_id, error=error)
            return None
        return candidate

    # -- the gate --------------------------------------------------------------------------

    def admit(self, trajectory: Trajectory, environment: EnvironmentFactory) -> Admission:
        """Synthesize a skill and store it ONLY if it proves itself.

        The candidate is hardened, structurally validated, statically scanned,
        re-executed through the sandbox against a fresh ``environment`` standing at
        the recorded starting screen, and judged by the critic. A rejection is fed
        back to the model with its error and trace and the skill is rewritten, up to
        ``max_repairs`` times.

        Args:
            trajectory: The successful run to learn from.
            environment: Called once per attempt; returns the recorded environment
                reset to the trajectory's first screen.

        Returns:
            An :class:`Admission`. ``ok`` means - and only means - that this exact
            code ran to completion in the replayed environment, its verifier agreed
            and the critic agreed, and ``skill`` is what the store now holds.
            Otherwise nothing was stored and ``attempts`` says why.

        Raises:
            ProviderError: if a model call fails.
            BudgetExceeded: if the sandbox run exhausts an agent-level budget.
        """
        if not self._worth_keeping(trajectory):
            reason = (
                "the run did not succeed"
                if not trajectory.ok
                else f"the run is {len(trajectory.steps)} step(s) long; too short to be a skill"
            )
            log.info("skill.admit.skipped", run_id=trajectory.run_id, reason=reason)
            return Admission(ok=False, skill=None, reason=reason)

        messages = [LLMMessage(role="user", text=describe_trajectory(trajectory))]
        attempts: list[Attempt] = []
        for index in range(self._max_repairs + 1):
            response = self._llm.complete(
                messages,
                system=self._prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            attempt = self._attempt(index, response.text, trajectory, environment)
            attempts.append(attempt)
            if attempt.ok and attempt.skill is not None:
                stored = self._store.put(attempt.skill)
                log.info(
                    "skill.admit.stored",
                    run_id=trajectory.run_id,
                    name=stored.name,
                    domain=stored.domain,
                    version=stored.version,
                    repairs=index,
                )
                return Admission(
                    ok=True,
                    skill=stored,
                    attempts=tuple(attempts),
                    reason=f"admitted after {index} repair(s)",
                )
            log.info(
                "skill.admit.rejected",
                run_id=trajectory.run_id,
                attempt=index,
                stage=attempt.stage,
                error=attempt.error,
            )
            if index < self._max_repairs:
                messages = [
                    *messages,
                    LLMMessage(role="assistant", text=response.text),
                    LLMMessage(role="user", text=_repair_brief(attempt)),
                ]
        last = attempts[-1]
        return Admission(
            ok=False,
            skill=None,
            attempts=tuple(attempts),
            reason=(
                f"rejected at {last.stage} after {len(attempts)} attempt(s) "
                f"({self._max_repairs} repair(s) allowed): {last.error}"
            ),
        )

    # -- one attempt -----------------------------------------------------------------------

    def _attempt(
        self,
        index: int,
        reply: str,
        trajectory: Trajectory,
        environment: EnvironmentFactory,
    ) -> Attempt:
        """Everything that has to be true before a skill may be stored."""
        draft, why = _parse_draft(reply)
        if draft is None:
            return Attempt(index, "generation", False, error=why)

        candidate, error, hardening = self._build(draft, trajectory)
        if candidate is None:
            return Attempt(index, "structure", False, error=error, hardening=hardening)

        try:
            scan_code(candidate.code, what=f"skill {candidate.name!r}")
            if candidate.verifier_code:
                scan_code(candidate.verifier_code, what=f"the verifier for {candidate.name!r}")
        except SandboxViolation as exc:
            return Attempt(
                index, "sandbox", False, skill=candidate, error=str(exc), hardening=hardening
            )

        env = environment()
        before = env.perceiver.observe(env.controller)
        if candidate.precondition is not None:
            similarity = before.fingerprint.similarity(candidate.precondition)
            if similarity < self._min_similarity:
                return Attempt(
                    index,
                    "precondition",
                    False,
                    skill=candidate,
                    error=(
                        "the environment is not on the recorded starting screen "
                        f"(similarity {similarity:.2f} to {candidate.precondition.value}); "
                        "the skill cannot be proved here"
                    ),
                    hardening=hardening,
                )

        runner = SkillRunner(self._store, limits=self._limits)
        ctx = runner.context(
            env.controller,
            env.perceiver,
            graph=env.graph,
            domain=candidate.domain,
            limits=self._limits,
        )
        result = runner.run(candidate, dict(draft.example_args), ctx)
        if not result.ok:
            return Attempt(
                index,
                "execution",
                False,
                skill=candidate,
                error=result.error,
                trace=result.trace,
                result=result,
                hardening=hardening,
            )

        after = env.perceiver.observe(env.controller)
        verdict = self._critic.judge(trajectory.task, before, after, candidate.docstring)
        if not verdict.ok:
            return Attempt(
                index,
                "critic",
                False,
                skill=candidate,
                error=f"the critic rejected the result: {verdict.reason}",
                trace=result.trace,
                result=result,
                verdict=verdict,
                hardening=hardening,
            )
        return Attempt(
            index,
            "critic",
            True,
            skill=candidate,
            trace=result.trace,
            result=result,
            verdict=verdict,
            hardening=hardening,
        )

    # -- building a candidate --------------------------------------------------------------

    def _build(
        self, draft: _Draft, trajectory: Trajectory
    ) -> tuple[Skill | None, str | None, Hardening | None]:
        """Harden the draft and validate it into a :class:`Skill`.

        The precondition is the trajectory's FIRST screen: where the recording
        started is where the skill may be used, and the planner routes there.
        """
        hardening = harden(draft.code, trajectory, params=draft.params)
        params = {**dict(draft.params), **dict(hardening.added_params)}
        if hardening.changed:
            log.debug("skill.harden.applied", name=draft.name, changes=len(hardening.changes))
        try:
            candidate = make_skill(
                name=draft.name,
                domain=trajectory.domain,
                summary=draft.summary,
                docstring=draft.docstring,
                code=hardening.code,
                params=params,
                requires=draft.requires,
                precondition=trajectory.steps[0].before.fingerprint if trajectory.steps else None,
                verifier_code=draft.verifier_code,
                provenance=Provenance(
                    trajectory_id=trajectory.run_id,
                    task_text=trajectory.task,
                    model=self._llm.name(),
                    created_at=utcnow(),
                ),
            )
        except SkillInvalid as exc:
            return None, f"{type(exc).__name__}: {exc}", hardening
        return candidate, None, hardening

    def _worth_keeping(self, trajectory: Trajectory) -> bool:
        """A failed run teaches nothing a skill can repeat, and a one-action run is
        not a procedure worth a name."""
        return trajectory.ok and len(trajectory.steps) >= self._min_steps
