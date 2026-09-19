"""The explorer: the cold path, where a task nobody has done before gets done anyway.

Everything else in this project either feeds this loop or lives off what it leaves
behind. Synthesis compiles a verified trajectory from here into a stored skill; the site
graph grows only because this loop writes down every transition it sees. There is no
separate crawler and no offline indexing pass: the map is a by-product of working.

The loop is six steps, bounded four ways::

    observe -> prompt -> ask the model -> ground and perform -> judge -> continue?

**Observe.** One :class:`~skillweaver.contracts.Perceiver` call. The action path of this
module sees the screenshot, the detected elements and the fingerprint, and nothing else.
There is deliberately no route from here to a DOM or any other ground truth: that exists
in this project solely as an offline teacher for labelling and scoring, and reaching it
from the acting loop would make every number the project reports meaningless.

**Prompt.** :class:`ElementCatalog` gives every element on the current screen a short id,
and the model is required to act by id. Around that go the goal, the graph neighbourhood
of the current state, any retrieved skills, the run so far - and the failure memory.

**The failure memory is the part that makes this terminate.** An explorer that forgets
what it just tried proposes it again, because the screen that prompted it has not changed,
and then loops until its budget runs out with nothing to show. So every failed attempt is
recorded against the fingerprint of the screen it was tried on (:class:`FailureMemory`),
listed back in the prompt, and - because a model that is told not to repeat itself still
sometimes does - *enforced*: a move whose signature is already a dead end at this exact
state is refused without being performed, and the model is asked again knowing why. The
guard is what the test asserts; the prompt section is what usually makes the guard
unnecessary.

**Ground and perform.** The model may answer with one primitive action or with a short
code block. Both go through the same grounding: an action names an element id that must be
in the catalogue of the screen in front of it, and code blocks run in the same sandbox
that stored skills run in (:class:`~skillweaver.skills.sandbox.SkillRunner`), with the
catalogue passed in as ``el``. Whatever a move performs is taped
(:class:`_TapedController`) so that one action, one trajectory step and one graph edge
stay the same thing whether it came from a primitive or from a block.

**Judge.** Every verdict comes from the :class:`~skillweaver.contracts.Critic`, never from
this module. :class:`~skillweaver.agent.critic.TieredCritic` answers most step questions -
"the screen did not change" and "an error appeared" - with no model call at all, which is
most of why this loop is affordable.

**Continue?** :class:`~skillweaver.contracts.Budget` bounds the run four independent ways:
steps, wall-clock seconds, dollars and model calls. Each one alone can stop the loop.
Money and calls are charged from the difference in
:meth:`~skillweaver.contracts.LLMClient.total_usage` across each model call, so an
escalation the critic makes on the same client is charged to the run that caused it; a
client that under-reports is still charged for the call this loop made itself.

Failure is a result, not an exception
-------------------------------------

Running out of budget, a model that will not produce a usable answer, a screen with no way
out: all of these return :class:`ExplorationOutcome` - a real
:class:`~skillweaver.contracts.RunOutcome`, so any caller typed against the Protocol is
unaffected - with ``ok=False`` and a :class:`Diagnosis` naming the state it was stuck in,
the limit that stopped it, and everything it had tried there. A failed run that says where
it got stuck is worth keeping; one that says "failed" is not. The only exception allowed
out is :class:`~skillweaver.errors.ControllerError`, per the
:class:`~skillweaver.contracts.Explorer` Protocol: a broken controller is not a judgment
about the task.

A malformed reply, an id that is not on the screen, an action the controller cannot
perform, a code block that raises - each is a recoverable step failure. It is remembered,
explained back to the model, and the loop goes round again.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from skillweaver.agent.critic import TieredCritic
from skillweaver.contracts import (
    ACTION_TYPES,
    Action,
    ActionKind,
    ActionResult,
    Box,
    Budget,
    Candidate,
    Click,
    Controller,
    Critic,
    Drag,
    Element,
    LLMClient,
    LLMMessage,
    Navigate,
    Observation,
    Perceiver,
    Point,
    PressKey,
    Provenance,
    RunOutcome,
    Screenshot,
    Scroll,
    SiteGraph,
    Skill,
    SkillRetriever,
    Spend,
    TaskSpec,
    Trajectory,
    TrajectoryRecorder,
    TypeText,
    UIState,
    Usage,
    Verdict,
    Wait,
    utcnow,
)
from skillweaver.contracts import Move as PointerMove
from skillweaver.errors import BudgetExceeded, ControllerError, SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.skills.api import SkillLimits, describe_action
from skillweaver.skills.sandbox import SkillRunner

__all__ = [
    "MAX_BLOCK_ACTIONS",
    "PROMPT_PATH",
    "Attempt",
    "Diagnosis",
    "ElementCatalog",
    "ExplorationOutcome",
    "Explorer",
    "FailureMemory",
    "Move",
    "load_prompt",
]

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "explore.md"
"""The acting prompt, used as the system prompt of every model call this loop makes."""

MAX_BLOCK_ACTIONS = 8
"""Controller actions one model-written code block may perform.

A block is a move, not a program: the loop has to see the screen again to stay grounded,
and a block that acts more than this has stopped being one decision. The sandbox stops it
at the limit and the actions it already performed are kept and judged.
"""

RECENT_MOVES = 8
"""How many past moves of this run are quoted back to the model."""

NEIGHBOURS = 6
"""How many known outgoing edges of the current state are quoted back to the model."""

DEFAULT_MAX_TOKENS = 1024
"""Cap on an acting reply. The answer is a small JSON object; a long one is a symptom."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_TARGETLESS: frozenset[ActionKind] = frozenset({"type_text", "press_key", "wait", "navigate"})


@cache
def load_prompt() -> str:
    """The contents of ``prompts/explore.md``. Read once per process.

    Raises:
        OSError: if the prompt file is missing from the installed package.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


class _Invalid(Exception):
    """A model answer that cannot be turned into a grounded move.

    Private and never allowed out of this module: the loop catches it and turns it into a
    recoverable step failure, because being unable to use one answer is not being unable
    to do the task. The message is written to be read by the model that will be asked
    again, so it names what was wrong and what would have been right.

    ``record`` is ``False`` when the refusal came from the failure memory itself, which
    has already counted the repeat. Filing it a second time would make the diagnosis
    report one stubborn idea as two different problems.
    """

    def __init__(self, message: str, *, record: bool = True) -> None:
        super().__init__(message)
        self.record = record


# --------------------------------------------------------------------------------------
# What the model is allowed to point at
# --------------------------------------------------------------------------------------


class ElementCatalog:
    """The elements of ONE observation, each under a short id the model acts by.

    An element's :attr:`~skillweaver.contracts.Element.stable_id` is its id when it has
    one and it is unique on this screen, so the same control keeps the same name across
    observations of the same state and the failure memory can recognise a repeat. Anything
    else gets a positional ``e0``, ``e1`` id, which is stable within this screen only.

    The catalogue is the grounding boundary: an id that is not in it is not on the screen,
    and :meth:`get` refuses it with a message naming what is available instead.
    """

    __slots__ = ("_by_id", "_ids")

    def __init__(self, elements: Sequence[Element]) -> None:
        counts: dict[str, int] = {}
        for element in elements:
            if element.stable_id:
                counts[element.stable_id] = counts.get(element.stable_id, 0) + 1
        by_id: dict[str, Element] = {}
        for position, element in enumerate(elements):
            name = element.stable_id or ""
            if not name or counts.get(name, 0) > 1 or name in by_id:
                name = f"e{position}"
            while name in by_id:  # pragma: no cover - only if a stable_id is literally "eN"
                name += "_"
            by_id[name] = element
        self._by_id = by_id
        self._ids = tuple(by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, element_id: object) -> bool:
        return element_id in self._by_id

    @property
    def ids(self) -> tuple[str, ...]:
        """Every id, in the reading order the elements came in."""
        return self._ids

    def as_mapping(self) -> dict[str, Element]:
        """A fresh ``{id: element}`` dict - what a code block receives as ``el``."""
        return dict(self._by_id)

    def id_for(self, element: Element) -> str | None:
        """The id this catalogue gave ``element``, or ``None`` if it is not on it."""
        for name, known in self._by_id.items():
            if known == element:
                return name
        return None

    def get(self, element_id: object) -> Element:
        """The element with this id.

        Raises:
            _Invalid: if the id is missing, not a string, or not on this screen. The
                message lists the ids that ARE on it, truncated, because that is the
                correction the model needs.
        """
        if not isinstance(element_id, str) or not element_id:
            raise _Invalid(
                "this action needs an 'element_id' naming one of the elements listed on "
                f"the current screen, but got {element_id!r}"
            )
        try:
            return self._by_id[element_id]
        except KeyError:
            raise _Invalid(
                f"there is no element {element_id!r} on the screen in front of you. "
                f"The ids on it are: {self._quote_ids()}"
            ) from None

    def _quote_ids(self, limit: int = 24) -> str:
        shown = ", ".join(self._ids[:limit])
        return shown + (", ..." if len(self._ids) > limit else "") if shown else "(none)"

    def render(self, limit: int = 60) -> str:
        """The element list as the prompt shows it, one line per element."""
        if not self._by_id:
            return "  (perception found nothing on this screen)"
        lines = []
        for name, element in list(self._by_id.items())[:limit]:
            box = element.box
            text = f" {element.text!r}" if element.text else ""
            lines.append(
                f"  [{name}] {element.kind.value}{text} at ({box.x},{box.y}) {box.w}x{box.h}"
            )
        if len(self._by_id) > limit:
            lines.append(f"  ... and {len(self._by_id) - limit} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# One move
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Move:
    """One grounded decision: what to do next, and what it is supposed to achieve.

    Attributes:
        thought: The model's stated reasoning, for the trajectory and the log.
        expect: What the model said would be visibly different afterwards. Handed to the
            critic as the expectation, which is the whole reason the prompt demands it.
        done: The model's claim that this move completes the task. A claim only: the
            critic decides, comparing the run's first screen with the one this leaves.
        action: The single primitive action, or ``None`` for a code move.
        code: The model's code block, or ``None`` for a primitive move.
        summary: One human-readable line, used in prompts, logs and the diagnosis.
        signature: The move's identity for :class:`FailureMemory`. Built from what was
            ASKED for - the element id, not the resolved pixel - so that "the same move
            again" means what a reader would mean by it.
    """

    thought: str
    expect: str
    done: bool
    action: Action | None
    code: str | None
    summary: str
    signature: str

    @property
    def acts(self) -> bool:
        """Whether this move touches the screen at all (a bare ``done`` does not)."""
        return self.action is not None or self.code is not None


# --------------------------------------------------------------------------------------
# The failure memory
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attempt:
    """One thing that was tried on one screen and did not work.

    ``state`` is the :class:`~skillweaver.contracts.Fingerprint` value of the screen it
    was tried on - failure is a property of a move AT a state, not of the move - and
    ``count`` is how many times it has been proposed since.
    """

    state: str
    signature: str
    summary: str
    reason: str
    count: int = 1


class FailureMemory:
    """What this run has already tried, per screen, and why it did not work.

    Keyed by ``(state fingerprint, move signature)``, because the same click is a fresh
    idea on a different screen and a dead end on this one. Two readers:

    * the prompt, through :meth:`at`, so the model can avoid repeating itself;
    * the loop, through :meth:`seen`, so that when it repeats itself anyway the move is
      refused before it is performed rather than after.
    """

    __slots__ = ("_by_key",)

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Attempt] = {}

    def __len__(self) -> int:
        return len(self._by_key)

    def remember(self, state: str, signature: str, summary: str, reason: str) -> Attempt:
        """Record a failed attempt, or count one more of an attempt already known.

        The FIRST reason is kept: it is the one observed when the move was actually
        performed, while a later one is usually this memory's own refusal.
        """
        key = (state, signature)
        known = self._by_key.get(key)
        entry = (
            Attempt(state, signature, summary, reason)
            if known is None
            else Attempt(state, signature, known.summary, known.reason, known.count + 1)
        )
        self._by_key[key] = entry
        return entry

    def seen(self, state: str, signature: str) -> Attempt | None:
        """The failed attempt matching this move at this state, or ``None``."""
        return self._by_key.get((state, signature))

    def at(self, state: str) -> list[Attempt]:
        """Every failed attempt on one screen, most-repeated first."""
        found = [a for a in self._by_key.values() if a.state == state]
        return sorted(found, key=lambda a: -a.count)

    def all(self) -> tuple[Attempt, ...]:
        """Every failed attempt of the run, in the order each was first seen."""
        return tuple(self._by_key.values())


# --------------------------------------------------------------------------------------
# What a failed run says
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Where a run got stuck and what it had tried there.

    The point of this type is that a failed run is still useful. ``state`` is the
    fingerprint value of the screen the loop was on when it stopped - the single most
    important fact, because it says whether the agent was lost, in a dead end, or one move
    from the goal - with ``label`` and ``url`` to make it legible to a human.

    ``stopped_by`` is ``"budget"`` (a limit was reached, named in ``limit``), ``"error"``
    (something broke; ``detail`` says what) or ``"gave-up"``.
    """

    stopped_by: str
    state: str
    label: str
    url: str | None
    steps: int
    moves: int
    limit: str = ""
    detail: str = ""
    tried: tuple[Attempt, ...] = ()

    def render(self) -> str:
        """The paragraph a human reads, and the ``reason`` of the run's final verdict."""
        where = f"state {self.state[:12]}"
        if self.label:
            where += f" ({self.label})"
        if self.url:
            where += f" at {self.url}"
        cause = {
            "budget": f"the run ran out of budget: {self.detail or self.limit}",
            "error": f"the run could not continue: {self.detail}",
        }.get(self.stopped_by, self.detail or "the run gave up")
        lines = [
            f"Stuck on {where} after {self.moves} move(s) and {self.steps} action(s); {cause}."
        ]
        here = [a for a in self.tried if a.state == self.state]
        if here:
            lines.append(f"Tried on that screen, without success ({len(here)}):")
            lines += [f"  - {a.summary} (x{a.count}) -> {_short(a.reason, 160)}" for a in here]
        elsewhere = len(self.tried) - len(here)
        if elsewhere:
            lines.append(f"{elsewhere} further failed attempt(s) on other screens.")
        if not self.tried:
            lines.append("Nothing had been tried yet, so the run never got started.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ExplorationOutcome(RunOutcome):
    """A :class:`~skillweaver.contracts.RunOutcome` that says where it got stuck.

    ``diagnosis`` is ``None`` on a successful run and a :class:`Diagnosis` on every failed
    one; :attr:`~skillweaver.contracts.RunOutcome.note` carries the same thing rendered,
    so a caller that only knows the Protocol still sees it.
    """

    diagnosis: Diagnosis | None = None


# --------------------------------------------------------------------------------------
# Taping what a move actually did
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Performed:
    """One action that reached the screen, with the observations either side of it."""

    action: Action
    before: Observation
    after: Observation
    result: ActionResult
    ms: float


class _TapedController:
    """A :class:`~skillweaver.contracts.Controller` that observes after every action.

    This is what makes a primitive move and a code block the same thing downstream: both
    perform through this, so both leave a list of :class:`_Performed` entries, and one
    entry is one trajectory step and one graph edge either way. The observation after an
    action becomes the one before the next, so a block of ``n`` actions costs ``n``
    observations rather than ``2n``.

    It delegates everything else to the real controller and owns none of it.
    """

    __slots__ = ("_inner", "_perceiver", "last", "tape")

    def __init__(self, inner: Controller, perceiver: Perceiver, seed: Observation) -> None:
        self._inner = inner
        self._perceiver = perceiver
        self.last = seed
        self.tape: list[_Performed] = []

    def perform(self, action: Action) -> ActionResult:
        before = self.last
        started = time.perf_counter()
        result = self._inner.perform(action)
        measured = (time.perf_counter() - started) * 1000
        after = self._perceiver.observe(self._inner)
        self.last = after
        self.tape.append(_Performed(action, before, after, result, result.elapsed_ms or measured))
        return result

    def capture(self) -> Screenshot:
        return self._inner.capture()

    def viewport(self) -> Box:
        return self._inner.viewport()

    def supports(self, action_kind: ActionKind) -> bool:
        return self._inner.supports(action_kind)

    def url(self) -> str | None:
        return self._inner.url()

    def describe(self) -> str:
        return self._inner.describe()

    def close(self) -> None:
        self._inner.close()


# --------------------------------------------------------------------------------------
# One run's accumulating state
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Run:
    """Everything one call to :meth:`Explorer.explore` accumulates. Private and mutable."""

    task: TaskSpec
    spend: Spend
    memory: FailureMemory
    usage_mark: Usage
    first: Observation
    current: Observation
    skills: tuple[Candidate, ...] = ()
    history: list[str] = field(default_factory=list)
    rejection: str | None = None
    steps: int = 0
    moves: int = 0
    ok: bool = False
    verdict: Verdict | None = None
    stopped_by: str = "gave-up"
    limit: str = ""
    detail: str = ""


# --------------------------------------------------------------------------------------
# The explorer
# --------------------------------------------------------------------------------------


class Explorer:
    """An :class:`~skillweaver.contracts.Explorer`: learn one task by trying things.

    Args:
        llm: The computer-use model that proposes each move. Its
            :meth:`~skillweaver.contracts.LLMClient.total_usage` is what the dollar and
            call budgets are charged from, so a critic sharing this client has its
            escalations charged to the run that caused them.
        perceiver: Eyes. Called once before the loop and once after every action.
        critic: Who decides whether a move worked. Defaults to a
            :class:`~skillweaver.agent.critic.TieredCritic` over ``llm``, which answers
            most steps without a model call. Never re-implement judging here.
        graph: Where transitions are written. ``None`` means this run teaches the project
            nothing about the site, so pass one unless you have a reason not to.
        recorder: Where the trajectory is built. ``None`` means
            :class:`skillweaver.trajectory.record.Recorder`, which writes to the
            configured data directory; tests pass an in-memory one.
        retriever: Consulted ONCE per run for skills worth reusing, which are described to
            the model. ``None`` skips it.
        runner: The sandbox code blocks execute in. ``None`` builds one with no skill
            store, so a block can act but cannot call a stored skill.
        max_tokens: Cap on each acting reply.
        max_block_actions: Actions one code block may perform.

    One explorer is reusable across runs; :meth:`explore` keeps no state between them.
    """

    def __init__(
        self,
        llm: LLMClient,
        perceiver: Perceiver,
        *,
        critic: Critic | None = None,
        graph: SiteGraph | None = None,
        recorder: TrajectoryRecorder | None = None,
        retriever: SkillRetriever | None = None,
        runner: SkillRunner | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_block_actions: int = MAX_BLOCK_ACTIONS,
    ) -> None:
        self._llm = llm
        self._perceiver = perceiver
        self._critic: Critic = critic if critic is not None else TieredCritic(llm)
        self._graph = graph
        self._retriever = retriever
        self._runner = runner if runner is not None else SkillRunner(None)
        self._max_tokens = max_tokens
        self._max_block_actions = max_block_actions
        self._recorder = recorder if recorder is not None else _default_recorder()

    def __repr__(self) -> str:
        return f"Explorer(model={self._llm.name()!r}, graph={self._graph is not None})"

    # -- the Protocol ------------------------------------------------------------------

    def explore(self, task: TaskSpec, controller: Controller, budget: Budget) -> ExplorationOutcome:
        """Attempt ``task`` within ``budget`` and return what happened.

        Neither running out of budget nor failing the task raises: both come back as
        ``ok=False`` with the partial trajectory and a :class:`Diagnosis`.

        Raises:
            ControllerError: if the controller breaks mid-run.
        """
        spend = Spend(budget).start()
        self._recorder.start(task.text, task.domain)
        self._go_to_start(task, controller)
        observation = self._perceiver.observe(controller)
        run = _Run(
            task=task,
            spend=spend,
            memory=FailureMemory(),
            usage_mark=self._llm.total_usage(),
            first=observation,
            current=observation,
        )
        try:
            self._loop(task, controller, run)
        except ControllerError:
            self._recorder.finish(False, "the controller broke mid-run")
            raise
        except BudgetExceeded as exc:
            run.stopped_by, run.limit, run.detail = "budget", _limit_of(exc), str(exc)
        except SkillWeaverError as exc:
            run.stopped_by = "error"
            run.detail = f"{type(exc).__name__}: {exc}"
            log.warning("explore.error", task=task.text, error=run.detail)
        return self._finish(run)

    # -- the loop ----------------------------------------------------------------------

    def _loop(self, task: TaskSpec, controller: Controller, run: _Run) -> None:
        """Observe, ask, act, judge, repeat - until solved or out of budget."""
        self._remember_state(task, run.current)
        run.skills = self._retrieve(task)

        while True:
            if self._exhausted(run):
                return
            catalog = ElementCatalog(run.current.elements)
            answer = self._ask(task, run, catalog)
            try:
                move = self._ground(answer, catalog, controller)
                self._refuse_repeat(move, run)
            except _Invalid as exc:
                self._reject(run, exc, answer)
                continue
            if self._exhausted(run):
                return
            self._make_move(task, move, catalog, controller, run)
            if run.ok:
                return

    def _make_move(
        self,
        task: TaskSpec,
        move: Move,
        catalog: ElementCatalog,
        controller: Controller,
        run: _Run,
    ) -> None:
        """Perform one move, judge it, write it down - and check a ``done`` claim."""
        run.moves += 1
        before = run.current
        performed = self._perform(move, catalog, controller, run)
        if performed:
            after = performed[-1].after
            verdict = self._judge(move.expect or move.summary, before, after, move, run)
            self._write_down(task, move, performed, verdict, run)
            run.current = after
            run.steps += len(performed)
            run.spend.add_step(len(performed))
            run.history.append(
                f"{run.moves}. {move.summary} -> "
                f"{'ok' if verdict.ok else 'FAILED'}: {_short(verdict.reason)}"
            )
            if not verdict.ok:
                run.memory.remember(
                    before.fingerprint.value, move.signature, move.summary, verdict.reason
                )
                run.rejection = f"your last move ({move.summary}) did not work: {verdict.reason}"
            else:
                run.rejection = None
        elif move.acts:
            # A move that reached nothing is still a move that did not work, and it is
            # the one most likely to be proposed again verbatim: the screen that
            # suggested it is untouched. Remember it so the guard catches the repeat.
            reason = run.rejection or "it performed no action at all"
            run.memory.remember(before.fingerprint.value, move.signature, move.summary, reason)
            run.history.append(f"{run.moves}. {move.summary} -> performed nothing: {reason}")

        if move.done:
            self._settle(task, move, run)

    def _settle(self, task: TaskSpec, move: Move, run: _Run) -> None:
        """Test the model's claim that the task is finished, and believe only the critic."""
        verdict = self._judge(task.text, run.first, run.current, move, run)
        if verdict.ok:
            run.ok, run.verdict, run.stopped_by = True, verdict, "solved"
            log.info("explore.solved", task=task.text, steps=run.steps, moves=run.moves)
            return
        run.memory.remember(
            run.current.fingerprint.value,
            f"done:{move.signature}",
            f"claim the task is complete after {move.summary}",
            verdict.reason,
        )
        run.rejection = (
            "you said the task was complete, but the critic compared the first screen of "
            f"the run with this one and disagreed: {verdict.reason}"
        )

    # -- asking the model --------------------------------------------------------------

    def _ask(self, task: TaskSpec, run: _Run, catalog: ElementCatalog) -> str:
        """One model call: the whole situation in one user turn, plus the screenshot.

        The conversation is rebuilt each time rather than grown. What has been tried is
        then an explicit, curated section of the prompt instead of something the model has
        to infer from a transcript - which is the difference between an explorer that
        learns from its failures and one that merely has them in its context.
        """
        message = LLMMessage(
            role="user",
            text=self._compose(task, run, catalog),
            images=(run.current.screenshot.png,),
        )
        response = self._llm.complete([message], system=load_prompt(), max_tokens=self._max_tokens)
        self._charge(run, at_least=1)
        return response.text

    def _compose(self, task: TaskSpec, run: _Run, catalog: ElementCatalog) -> str:
        """The user turn: goal, budget, screen, what is known, and what has failed."""
        spend, budget = run.spend, run.spend.budget
        sections = [
            f"TASK: {task.text}",
            f"DOMAIN: {task.domain}   TARGET: {task.target}",
        ]
        if task.params:
            sections.append(
                "TASK PARAMETERS (use these values literally): "
                + ", ".join(f"{k}={v!r}" for k, v in task.params.items())
            )
        sections.append(
            f"MOVE {run.moves + 1}. Budget left: "
            f"{max(budget.max_steps - spend.steps, 0)} action(s), "
            f"{max(budget.max_llm_calls - spend.llm_calls, 0)} model call(s), "
            f"${max(budget.max_usd - spend.usd, 0.0):.4f}, "
            f"{max(budget.max_seconds - spend.elapsed_seconds(), 0.0):.0f}s."
        )
        sections.append(
            "CURRENT SCREEN\n"
            f"  url: {run.current.url or '(the controller has no URL)'}\n"
            f"  state id: {run.current.fingerprint.value[:12]}"
        )
        sections.append("ELEMENTS ON SCREEN (act by id):\n" + catalog.render())
        sections.append(
            "WHAT THE SITE GRAPH KNOWS ABOUT THIS SCREEN:\n" + self._known(run, catalog)
        )
        if run.skills:
            sections.append(
                "STORED SKILLS THAT MAY ALREADY DO PART OF THIS:\n"
                + "\n".join(
                    f"  - {c.skill.name}({', '.join(c.skill.params)}) - {c.skill.summary} [{c.why}]"
                    for c in run.skills
                )
            )
        sections.append(
            "ALREADY TRIED ON THIS EXACT SCREEN AND FAILED - DO NOT PROPOSE ANY OF "
            "THESE AGAIN:\n" + self._dead_ends(run)
        )
        if run.history:
            sections.append(
                "THIS RUN SO FAR:\n"
                + "\n".join(f"  {line}" for line in run.history[-RECENT_MOVES:])
            )
        if run.rejection:
            sections.append(f"ABOUT YOUR LAST ANSWER: {run.rejection}")
        sections.append(
            "The image is the CURRENT screen, the one the element ids above refer to. "
            "Reply with the JSON object described in your instructions."
        )
        return "\n\n".join(sections)

    def _known(self, run: _Run, catalog: ElementCatalog) -> str:
        """The graph neighbourhood of the current state, as the prompt shows it."""
        if self._graph is None:
            return "  (no site graph is wired up, so nothing is remembered between runs)"
        edges = self._graph.neighbors(run.current.fingerprint)[:NEIGHBOURS]
        if not edges:
            return "  (this screen is new: nothing has ever been tried here before)"
        return "\n".join(
            f"  - {'; '.join(_retarget(a, catalog, run.current) for a in e.actions)} -> "
            f"state {e.dst.value[:12]} ({e.successes}/{e.attempts} worked, ~{e.mean_ms:.0f}ms)"
            for e in edges
        )

    def _dead_ends(self, run: _Run) -> str:
        """The failure memory for the current screen, as the prompt shows it."""
        here = run.memory.at(run.current.fingerprint.value)
        if not here:
            return "  (nothing yet on this screen)"
        return "\n".join(f"  - {a.summary} (tried {a.count}x) -> {_short(a.reason)}" for a in here)

    # -- grounding ---------------------------------------------------------------------

    def _ground(self, text: str, catalog: ElementCatalog, controller: Controller) -> Move:
        """Turn one model reply into a :class:`Move`, or refuse it.

        Raises:
            _Invalid: for anything unusable - not JSON, no decision in it, an action kind
                that does not exist, an element id that is not on the screen, an action
                this controller cannot perform. The loop turns each into a recoverable
                step failure.
        """
        data = _parse_answer(text)
        if data is None:
            raise _Invalid(
                "your reply was not a JSON object. Reply with exactly the object described "
                f"in your instructions and nothing else. What you sent was: {_snippet(text)}"
            )
        thought = str(data.get("thought") or "").strip()
        expect = str(data.get("expect") or data.get("expectation") or "").strip()
        done = bool(data.get("done"))
        spec = data.get("action")
        code = data.get("code")

        if spec is not None and code is not None:
            raise _Invalid("send either 'action' or 'code' for one move, not both")
        if spec is not None:
            action, signature, summary = _resolve(spec, catalog, controller)
            return Move(thought, expect, done, action, None, summary, signature)
        if code is not None:
            if not isinstance(code, str) or not code.strip():
                raise _Invalid("'code' must be a non-empty string of Python statements")
            body = code.strip()
            digest = hashlib.sha256(_normalize_code(body).encode()).hexdigest()[:10]
            first = body.splitlines()[0].strip()
            return Move(
                thought,
                expect,
                done,
                None,
                body,
                f"code block ({len(body.splitlines())} line(s)): {_short(first, 60)}",
                f"code:{digest}",
            )
        if done:
            return Move(thought, expect, True, None, None, "declare the task complete", "done")
        raise _Invalid(
            "your reply contained no decision: give an 'action', or a 'code' block, or 'done': true"
        )

    def _refuse_repeat(self, move: Move, run: _Run) -> None:
        """Refuse a move already known to fail on this exact screen.

        The prompt asks the model not to repeat itself and usually that is enough. This is
        what happens when it is not: the same move on the same screen has the same
        outcome, so performing it would spend a step to learn nothing. Refusing costs the
        model call that proposed it and returns a message naming the previous reason,
        which is the input it needs to propose something else.

        Raises:
            _Invalid: when the move is a known dead end here.
        """
        state = run.current.fingerprint.value
        known = run.memory.seen(state, move.signature)
        if known is None:
            return
        run.memory.remember(state, move.signature, move.summary, known.reason)
        raise _Invalid(
            f"you have already tried {known.summary} on this exact screen "
            f"({known.count}x) and it did not work: {known.reason}. The screen has not "
            "changed since, so it will not work now either. Propose something different - "
            "a different element, a different kind of action, or a move that leaves this "
            "screen.",
            record=False,
        )

    def _reject(self, run: _Run, refusal: _Invalid, answer: str) -> None:
        """Record an unusable answer as a recoverable step failure and tell the model."""
        why = str(refusal)
        if refusal.record:
            run.memory.remember(
                run.current.fingerprint.value,
                f"malformed:{hashlib.sha256(answer.encode()).hexdigest()[:10]}",
                "an answer that could not be used",
                why,
            )
        run.rejection = why
        run.history.append(f"{run.moves + 1}. (answer refused) {_short(why, 110)}")
        log.info("explore.reject", why=_short(why, 160))

    # -- performing --------------------------------------------------------------------

    def _perform(
        self, move: Move, catalog: ElementCatalog, controller: Controller, run: _Run
    ) -> list[_Performed]:
        """Run the move against the screen and return every action that reached it.

        A primitive action goes straight through the tape. A code block goes through the
        same sandbox that stored skills run in, with the current screen's catalogue bound
        to ``el``: a block that raises, violates the sandbox or hits its own limits still
        keeps whatever it managed to do, because those actions really did happen and the
        critic has to judge the screen they left behind.
        """
        tape = _TapedController(controller, self._perceiver, run.current)
        if move.action is not None:
            tape.perform(move.action)
        elif move.code is not None:
            self._run_block(move, catalog, tape, run)
        return tape.tape

    def _run_block(
        self, move: Move, catalog: ElementCatalog, tape: _TapedController, run: _Run
    ) -> None:
        """Execute a code move in the skill sandbox against ``tape``."""
        remaining = max(run.spend.budget.max_steps - run.spend.steps, 1)
        limits = SkillLimits(
            max_steps=min(self._max_block_actions, remaining),
            max_seconds=max(
                min(30.0, run.spend.budget.max_seconds - run.spend.elapsed_seconds()), 0.5
            ),
            max_depth=1,
        )
        skill = _block_skill(move, run.task, self._llm.name())
        context = self._runner.context(
            tape, self._perceiver, graph=self._graph, domain=run.task.domain, limits=limits
        )
        result = self._runner.run(skill, {"el": catalog.as_mapping()}, context)
        if not result.ok:
            run.rejection = f"your code block failed: {result.error}"
            log.info("explore.block.failed", error=result.error)

    # -- judging and writing down ------------------------------------------------------

    def _judge(
        self, goal: str, before: Observation, after: Observation, move: Move, run: _Run
    ) -> Verdict:
        """One critic call, with whatever it cost charged to this run's budget."""
        verdict = self._critic.judge(goal, before, after, move.expect or None)
        self._charge(run, at_least=1 if getattr(verdict, "escalated", False) else 0)
        return verdict

    def _write_down(
        self,
        task: TaskSpec,
        move: Move,
        performed: Sequence[_Performed],
        verdict: Verdict,
        run: _Run,
    ) -> None:
        """Record the move in the trajectory and in the site graph.

        One performed action is one trajectory step and one graph edge, whether it came
        from a primitive move or from inside a code block. The verdict belongs to the
        move, so it is attached to the move's LAST action - the one the critic actually
        looked at - and the earlier ones are marked as what they are.

        Every edge is written, failures included: an edge that led nowhere is exactly what
        stops a later run walking into the same dead end, and the graph's persistence sums
        statistics rather than overwriting them, so recording a failure is never a loss.
        """
        last = len(performed) - 1
        for index, step in enumerate(performed):
            note = move.thought if index == last else f"part of: {move.summary}"
            if not step.result.ok:
                note = f"{note} [controller refused it: {step.result.error}]"
            self._recorder.step(
                step.action,
                step.before,
                step.after,
                step.result,
                verdict if index == last else None,
                note,
            )
            self._remember_state(task, step.before)
            self._remember_state(task, step.after)
            if self._graph is not None:
                self._graph.observe_transition(
                    step.before.fingerprint,
                    (step.action,),
                    step.after.fingerprint,
                    verdict.ok and step.result.ok,
                    step.ms,
                )
        log.info(
            "explore.move",
            move=move.summary,
            actions=len(performed),
            ok=verdict.ok,
            reason=_short(verdict.reason, 120),
        )

    def _remember_state(self, task: TaskSpec, observation: Observation) -> None:
        """Add or refresh the graph node for one observed screen."""
        if self._graph is None:
            return
        self._graph.upsert_state(
            UIState(
                fingerprint=observation.fingerprint,
                domain=task.domain,
                label=_label_of(observation),
                url_pattern=observation.url,
            )
        )

    # -- budget ------------------------------------------------------------------------

    def _exhausted(self, run: _Run) -> bool:
        """Whether any of the four limits has been reached; records which one."""
        try:
            run.spend.check()
        except BudgetExceeded as exc:
            run.stopped_by, run.limit, run.detail = "budget", _limit_of(exc), str(exc)
            log.info("explore.budget", limit=run.limit, detail=run.detail)
            return True
        return False

    def _charge(self, run: _Run, *, at_least: int) -> None:
        """Charge model spend since the last charge against the run's budget.

        Taken as the difference in the client's own running total rather than from one
        response, so a model call the critic made on the same client is charged here too -
        it was made because of this run. ``at_least`` floors the call count, so a client
        that does not report ``calls`` still cannot make ``max_llm_calls`` unenforceable.
        """
        total = self._llm.total_usage()
        mark = run.usage_mark
        run.usage_mark = total
        run.spend.add_usage(
            Usage(
                max(total.input_tokens - mark.input_tokens, 0),
                max(total.output_tokens - mark.output_tokens, 0),
                max(total.calls - mark.calls, at_least),
                max(total.cost_usd - mark.cost_usd, 0.0),
            )
        )

    # -- odds and ends -----------------------------------------------------------------

    def _go_to_start(self, task: TaskSpec, controller: Controller) -> None:
        """Load ``params["start_url"]`` when the controller can and is not already there.

        Best effort by design: a desktop controller cannot navigate and a browser may
        already be on the page, and neither is a reason not to attempt the task.
        """
        url = task.params.get("start_url")
        if not isinstance(url, str) or not url or not controller.supports("navigate"):
            return
        if controller.url() == url:
            return
        result = controller.perform(Navigate(url))
        if not result.ok:
            log.info("explore.start_url.failed", url=url, error=result.error)

    def _retrieve(self, task: TaskSpec) -> tuple[Candidate, ...]:
        """Skills worth reusing, fetched once. A retrieval failure is not a run failure."""
        if self._retriever is None:
            return ()
        try:
            return tuple(self._retriever.search(task.text, task.domain, 4))
        except SkillWeaverError as exc:
            log.warning("explore.retrieve.failed", error=str(exc))
            return ()

    def _finish(self, run: _Run) -> ExplorationOutcome:
        """Close the trajectory and build the outcome, with a diagnosis when it failed."""
        if run.ok and run.verdict is not None:
            trajectory = self._recorder.finish(True, run.verdict.reason)
            return ExplorationOutcome(
                ok=True,
                trajectory=trajectory,
                verdict=run.verdict,
                spend=run.spend,
                skill_used=None,
                note=run.verdict.reason,
            )
        diagnosis = Diagnosis(
            stopped_by=run.stopped_by,
            state=run.current.fingerprint.value,
            label=_label_of(run.current),
            url=run.current.url,
            steps=run.steps,
            moves=run.moves,
            limit=run.limit,
            detail=run.detail,
            tried=run.memory.all(),
        )
        note = diagnosis.render()
        trajectory: Trajectory = self._recorder.finish(False, note)
        log.info(
            "explore.failed",
            stopped_by=run.stopped_by,
            limit=run.limit,
            state=diagnosis.state[:12],
            steps=run.steps,
        )
        return ExplorationOutcome(
            ok=False,
            trajectory=trajectory,
            verdict=Verdict(False, note, 1.0, "programmatic"),
            spend=run.spend,
            skill_used=None,
            note=note,
            diagnosis=diagnosis,
        )


# --------------------------------------------------------------------------------------
# Parsing and grounding a model answer
# --------------------------------------------------------------------------------------


def _parse_answer(text: str) -> dict[str, Any] | None:
    """The JSON object in a model reply, or ``None``.

    Tolerates a code fence and prose around the object, because models do both despite
    being asked not to and neither is a reason to throw away a usable answer. Anything
    else is ``None`` and becomes a recoverable step failure.
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
        if isinstance(data, dict):
            return data
    return None


def _resolve(spec: Any, catalog: ElementCatalog, controller: Controller) -> tuple[Action, str, str]:
    """``(action, signature, summary)`` for one action spec from the model.

    This is where grounding is enforced: every action with a target names an element id
    that is in ``catalog``, and the resulting point is that element's center in LOGICAL
    pixels. The model never sends coordinates, so it can never send stale ones.

    Raises:
        _Invalid: for a spec that is not an object, an unknown kind, a missing or wrong
            field, an id that is not on the screen, or a kind this controller refuses.
    """
    if not isinstance(spec, Mapping):
        raise _Invalid(f"'action' must be an object such as {{'kind': 'click', ...}}, got {spec!r}")
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in ACTION_TYPES:
        raise _Invalid(
            f"unknown action kind {kind!r}; it must be one of: {', '.join(sorted(ACTION_TYPES))}"
        )
    if not controller.supports(kind):  # type: ignore[arg-type]
        raise _Invalid(f"this controller cannot perform {kind!r} actions; use something else")

    match kind:
        case "click":
            element_id = spec.get("element_id")
            element = catalog.get(element_id)
            button = str(spec.get("button") or "left")
            if button not in ("left", "right", "middle"):
                raise _Invalid(f"'button' must be left, right or middle, got {button!r}")
            clicks = _as_int(spec.get("clicks"), 1, "clicks")
            action: Action = Click(element.box.center, button, clicks)  # type: ignore[arg-type]
            return (
                action,
                f"click:{element_id}:{button}:{clicks}",
                _describe("click", element_id, element),
            )
        case "move":
            element_id = spec.get("element_id")
            element = catalog.get(element_id)
            return (
                PointerMove(element.box.center),
                f"move:{element_id}",
                _describe("move to", element_id, element),
            )
        case "drag":
            element_id = spec.get("element_id") or spec.get("from_element_id")
            to_id = spec.get("to_element_id") or spec.get("to")
            start, end = catalog.get(element_id), catalog.get(to_id)
            return (
                Drag(start.box.center, end.box.center),
                f"drag:{element_id}->{to_id}",
                f"drag [{element_id}] onto [{to_id}]",
            )
        case "type_text":
            text = spec.get("text")
            if not isinstance(text, str) or not text:
                raise _Invalid("'type_text' needs a non-empty 'text' string")
            return TypeText(text), f"type_text:{text}", f"type {text!r}"
        case "press_key":
            keys = spec.get("keys") or spec.get("key")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
                raise _Invalid("'press_key' needs 'keys', a non-empty list of key names")
            chord = tuple(str(k) for k in keys)
            return PressKey(chord), f"press_key:{'+'.join(chord)}", f"press {'+'.join(chord)}"
        case "scroll":
            element_id = spec.get("element_id")
            dx = _as_int(spec.get("dx"), 0, "dx")
            dy = _as_int(spec.get("dy"), 0, "dy")
            if dx == 0 and dy == 0:
                raise _Invalid("'scroll' needs a non-zero 'dx' or 'dy'")
            if element_id is None:
                viewport = controller.viewport()
                point, where = viewport.center, "the middle of the screen"
            else:
                element = catalog.get(element_id)
                point, where = element.box.center, f"[{element_id}]"
            return (
                Scroll(point, dx, dy),
                f"scroll:{element_id}:{dx},{dy}",
                f"scroll {where} by ({dx}, {dy})",
            )
        case "wait":
            ms = _as_int(spec.get("ms"), 500, "ms")
            return Wait(ms), f"wait:{ms}", f"wait {ms}ms"
        case "navigate":
            url = spec.get("url")
            if not isinstance(url, str) or not url:
                raise _Invalid("'navigate' needs a 'url' string")
            return Navigate(url), f"navigate:{url}", f"navigate to {url}"
    raise _Invalid(f"action kind {kind!r} is not supported here")  # pragma: no cover


def _retarget(action: Action, catalog: ElementCatalog, observation: Observation) -> str:
    """:func:`~skillweaver.skills.api.describe_action`, with pixels named as element ids.

    The graph stores what was performed, which is a point, because a point is what a
    controller takes. Quoting that point back at a model which is required to act by id
    would be showing it something it is not allowed to use and cannot check. So a point is
    named by the element under it ON THE CURRENT SCREEN, and stays a point when the screen
    has nothing there - which is itself the useful signal that the remembered move does
    not apply here any more.
    """
    point = _target_of(action)
    if point is None:
        return describe_action(action)
    under = observation.index.containing(point)
    name = catalog.id_for(under[0]) if under else None
    if name is None:
        return describe_action(action)
    return f"{action.kind} [{name}]"


def _target_of(action: Action) -> Point | None:
    """The point an action aims at, or ``None`` for one that aims at nothing."""
    if isinstance(action, Click | PointerMove | Scroll):
        return action.point
    if isinstance(action, Drag):
        return action.start
    return None


def _describe(verb: str, element_id: Any, element: Element) -> str:
    text = f" {_short(element.text, 40)!r}" if element.text else ""
    return f"{verb} [{element_id}] {element.kind.value}{text}"


def _as_int(value: Any, default: int, name: str) -> int:
    """An integer field of an action spec, with a default.

    Raises:
        _Invalid: if the value is present but not a whole number.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _Invalid(f"{name!r} must be a whole number, got {value!r}") from None


def _normalize_code(code: str) -> str:
    """A code block reduced to what it does, so re-indented whitespace is not a new idea."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())


def _block_skill(move: Move, task: TaskSpec, model: str) -> Skill:
    """The model's code block as a throwaway, unstored skill the sandbox can run.

    The block is plain statements, so it is wrapped in the ``run(ctx, el)`` the runner
    calls. It goes through exactly the same static scan and limits as a stored skill:
    the explorer must not be a way to run code a skill would not be allowed to run.
    """
    body = "\n".join(f"    {line}" for line in move.code.splitlines()) if move.code else "    pass"
    return Skill(
        name="explore_block",
        domain=task.domain,
        summary=move.summary,
        docstring=move.thought or move.summary,
        params={"el": {"type": "object"}},
        code=f"def run(ctx, el):\n{body}\n",
        requires=(),
        precondition=None,
        verifier_code=None,
        provenance=Provenance("", task.text, model, utcnow()),
    )


def _label_of(observation: Observation) -> str:
    """A short human name for a screen: its most heading-like text. ``""`` when silent."""
    for element in observation.elements:
        if element.text.strip():
            return _short(element.text.strip(), 40)
    return ""


def _limit_of(exc: BudgetExceeded) -> str:
    """The limit named by a :class:`~skillweaver.errors.BudgetExceeded` message."""
    head = str(exc).split(" ", 1)[0]
    return head if head.startswith("max_") else ""


def _short(text: str, limit: int = 90) -> str:
    """One line of ``text``, at most ``limit`` characters."""
    flat = re.sub(r"\s+", " ", (text or "").strip())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _snippet(text: str, limit: int = 160) -> str:
    """A short quote of a model reply, for a message the model will read back."""
    flat = _short(text, limit)
    return repr(flat) if flat else "(an empty reply)"


def _default_recorder() -> TrajectoryRecorder:
    """The on-disk recorder, imported late so this module stays importable without it."""
    from skillweaver.trajectory.record import Recorder

    return Recorder()
