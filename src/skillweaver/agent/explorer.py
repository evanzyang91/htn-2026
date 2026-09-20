"""The explorer: the cold path, where a task nobody has done before gets done anyway.

observe -> prompt -> ask the model -> ground and perform -> judge -> continue?
Synthesis compiles a verified trajectory from here into a stored skill, and the site
graph grows only because this loop writes down every transition it sees.

This module sees exactly what the perceiver it was handed returns and must never reach
past it - it has no route to BrowserGroundTruth, and reaching that from here would make
every number the project reports meaningless. Every verdict comes from the Critic, never
from here. Failure is a result, not an exception: running out of budget, an unusable
model answer or a screen with no way out all return an ``ok=False``
:class:`ExplorationOutcome` carrying a :class:`Diagnosis`. Only ControllerError is
allowed out - a broken controller is not a judgment about the task.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from skillweaver.agent.critic import CriticVerdict, TieredCritic
from skillweaver.contracts import (
    ACTION_TYPES,
    Action,
    ActionKind,
    ActionResult,
    Back,
    Box,
    Budget,
    Candidate,
    Click,
    Controller,
    Critic,
    Drag,
    Element,
    Fingerprint,
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
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD
from skillweaver.skills.api import SkillLimits, describe_action
from skillweaver.skills.family import Subgoal, nearest_workflow, skeleton, tokens_of
from skillweaver.skills.family import render as render_signature
from skillweaver.skills.sandbox import SkillRunner

__all__ = [
    "MAX_BLOCK_ACTIONS",
    "PROMPT_PATH",
    "ActingPolicy",
    "Attempt",
    "Diagnosis",
    "ElementCatalog",
    "ExplorationOutcome",
    "Explorer",
    "FailureMemory",
    "Move",
    "PolicyBlocked",
    "load_prompt",
    "signature_move",
]

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "explore.md"
"""The acting prompt, used as the system prompt of every model call this loop makes."""

MAX_STRAYS = 2
"""How many moves in a row may fail to fit a borrowed workflow before it is dropped.
Two, because one is a cookie banner."""

LOOKAHEAD = 1
"""How many steps of a borrowed workflow a move may skip and still be following it.
One site submits a search with a button and the next submits on Enter inside the typing
block, so the step after the current one is the furthest a fitting move can land."""

MAX_BLOCK_ACTIONS = 8
"""Controller actions one model-written code block may perform.

A block is a move, not a program: the loop has to see the screen again to stay grounded.
The sandbox stops it at the limit and the actions it already performed are kept and judged.
"""

RECENT_MOVES = 8
"""How many past moves of this run are quoted back to the model."""

NEIGHBOURS = 6
"""How many known outgoing edges of the current state are quoted back to the model."""

DEFAULT_MAX_TOKENS = 1024
"""Cap on an acting reply. The answer is a small JSON object; a long one is a symptom."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_TARGETLESS: frozenset[ActionKind] = frozenset(
    {"type_text", "press_key", "wait", "navigate", "back"}
)


@cache
def load_prompt() -> str:
    """The contents of ``prompts/explore.md``. Read once per process.

    Raises:
        OSError: if the prompt file is missing from the installed package.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


class PolicyBlocked(Exception):
    """An acting policy reporting that nothing on this screen can make progress.

    The policy's own ``BLOCKED`` answer, which upstream Jev treats as terminal - not a
    failure of the loop. The run is diagnosed as blocked at the screen it was on, which
    beats burning the remaining budget on the same question.
    """


# --------------------------------------------------------------------------------------
# Who decides the next move
# --------------------------------------------------------------------------------------


@runtime_checkable
class ActingPolicy(Protocol):
    """Whatever decides the next move, when it is not this module's own model call.

    The DEFAULT is ``None`` and that path is unchanged. A policy replaces that one step
    and NOTHING else - failure memory, grounding, the taped controller, the critic, the
    graph writing, the trajectory and all four budget limits are the same code either
    way, which is why a skill learned through a policy is an ordinary stored skill.

    The return value is the explorer's ANSWER PROTOCOL: the same JSON object
    ``Explorer._ground`` parses out of a model reply.

    Raises:
        PolicyBlocked: when the policy reports no supported operation can progress.
        ProviderError: on a provider failure, which the loop records as a step failure.
    """

    def propose(
        self,
        task: TaskSpec,
        observation: Observation,
        catalog: ElementCatalog,
        history: Sequence[str],
        dead_ends: Sequence[Attempt],
        rejection: str | None,
    ) -> str:
        """The next move, as an answer object.

        Args:
            catalog: The screen's elements under the ids an answer must name.
            history: One line per move so far, oldest first.
            dead_ends: What was already tried ON THIS SCREEN and did not work. A policy
                that re-proposes one is refused and asked again, so honouring them is how
                it avoids paying for the same answer twice; ``signature_move`` reads the
                move and its element back out of one.
            rejection: Why the previous answer was refused, when it was.
        """
        ...

    def name(self) -> str:
        """The policy identifier, for logs and provenance."""
        ...


def signature_move(signature: str) -> tuple[str, str] | None:
    """The ``(kind, element_id)`` a :attr:`Move.signature` aims at, or ``None``.

    Signatures are built in ``_resolve`` as ``<kind>:<element_id>[:...]``; this is where
    that format is READ, so a policy pruning known-dead targets need not know how it is
    spelled. The KIND comes back with the id because a dead end is a move at an element,
    not an element: a click that did nothing says nothing about typing into the same
    field, and dropping the field on the strength of the click can take the task's only
    way forward away.
    """
    kind, _, rest = signature.partition(":")
    if kind in _TARGETLESS or kind in ("code", "done", "malformed") or not rest:
        return None
    target = rest.partition(":")[0]
    return (kind, target) if target else None


class _Invalid(Exception):
    """A model answer that cannot be turned into a grounded move.

    Never allowed out of this module: the loop turns it into a recoverable step failure.
    The message is written to be read by the model that will be asked again.

    ``record`` is ``False`` when the refusal came from the failure memory itself, which
    has already counted the repeat - filing it twice would report one stubborn idea as
    two problems.
    """

    def __init__(self, message: str, *, record: bool = True) -> None:
        super().__init__(message)
        self.record = record


# --------------------------------------------------------------------------------------
# What the model is allowed to point at
# --------------------------------------------------------------------------------------


class ElementCatalog:
    """The elements of ONE observation, each under a short id the model acts by.

    An element's ``stable_id`` is its id when it has one and is unique on this screen, so
    the same control keeps the same name across observations and the failure memory can
    recognise a repeat; anything else gets a positional ``e0``, ``e1``.

    The catalogue is the grounding boundary: an id not in it is not on the screen.
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
                message lists the ids that ARE on it, because that is the correction the
                model needs.
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
        expect: What the model said would be visibly different afterwards. Handed to the
            critic as the expectation, which is why the prompt demands it.
        done: A CLAIM that this move completes the task; the critic decides.
        summary: One human-readable line, used in prompts, logs and the diagnosis.
        signature: The move's identity for :class:`FailureMemory`, built from what was
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

    ``state`` is the fingerprint value of the screen it was tried on - failure is a
    property of a move AT a state - and ``count`` is how often it has been proposed since.
    """

    state: str
    signature: str
    summary: str
    reason: str
    count: int = 1


class FailureMemory:
    """What this run has already tried, per screen, and why it did not work.

    Keyed by ``(state fingerprint, move signature)``: the same click is a fresh idea on a
    different screen and a dead end on this one. Read by the prompt through :meth:`at`,
    by the repeat guard through :meth:`seen`, and by an acting policy through
    :meth:`near`.

    :meth:`near` exists because the other two key on the fingerprint VALUE, exact
    equality, while a move that fails usually repaints the page without changing it - so
    the dead end is filed under a screen the next step is not standing on. Measured on
    one 33-action sandbox run: of 11 critic-failed moves, SEVEN left a screen that is the
    same state by ``SAME_STATE_THRESHOLD`` and exactly ONE of those fingerprinted
    identically. :meth:`at` and :meth:`seen` are LEFT exact, because the default
    explorer's prompt and repeat guard are calibrated against them.
    """

    __slots__ = ("_by_key", "_screens")

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Attempt] = {}
        self._screens: dict[str, Fingerprint] = {}

    def __len__(self) -> int:
        return len(self._by_key)

    def remember(self, state: Fingerprint, signature: str, summary: str, reason: str) -> Attempt:
        """Record a failed attempt, or count one more of an attempt already known.

        The FIRST reason is kept - observed when the move was actually performed, where a
        later one is usually this memory's own refusal. The whole Fingerprint is taken
        because :meth:`near` needs its ``parts``; only the value is stored on the Attempt.
        """
        key = (state.value, signature)
        known = self._by_key.get(key)
        entry = (
            Attempt(state.value, signature, summary, reason)
            if known is None
            else Attempt(state.value, signature, known.summary, known.reason, known.count + 1)
        )
        self._by_key[key] = entry
        self._screens.setdefault(state.value, state)
        return entry

    def seen(self, state: str, signature: str) -> Attempt | None:
        """The failed attempt matching this move at this state, or ``None``."""
        return self._by_key.get((state, signature))

    def at(self, state: str) -> list[Attempt]:
        """Every failed attempt on one screen, most-repeated first."""
        found = [a for a in self._by_key.values() if a.state == state]
        return sorted(found, key=lambda a: -a.count)

    def near(self, state: Fingerprint) -> list[Attempt]:
        """Every failed attempt on this screen OR one indistinguishable from it.

        Same order as :meth:`at`, and the same answer whenever the fingerprints agree
        exactly. The class docstring holds the measurement saying they usually do not.
        """
        same = {
            value
            for value, known in self._screens.items()
            if value == state.value or known.similarity(state) >= SAME_STATE_THRESHOLD
        }
        found = [a for a in self._by_key.values() if a.state in same]
        return sorted(found, key=lambda a: -a.count)

    def all(self) -> tuple[Attempt, ...]:
        """Every failed attempt of the run, in the order each was first seen."""
        return tuple(self._by_key.values())


# --------------------------------------------------------------------------------------
# What a failed run says
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Where a run got stuck and what it had tried there, so a failed run is still useful.

    ``state`` is the fingerprint of the screen the loop stopped on - it says whether the
    agent was lost, in a dead end, or one move from the goal. ``stopped_by`` is
    ``"budget"`` (a limit named in ``limit``), ``"error"`` (``detail`` says what) or
    ``"gave-up"``.
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
    """A RunOutcome that says where it got stuck.

    ``diagnosis`` is ``None`` on success; ``note`` carries the same thing rendered, so a
    caller that only knows the Protocol still sees it.
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
    """A Controller that observes after every action.

    What makes a primitive move and a code block the same thing downstream: both leave a
    list of :class:`_Performed`, and one entry is one trajectory step and one graph edge
    either way. The observation after an action becomes the one before the next, so a
    block of ``n`` actions costs ``n`` observations rather than ``2n``.
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
    skeleton: tuple[Subgoal, ...] = ()
    skeleton_from: str = ""
    at: int = 0
    strays: int = 0
    history: list[str] = field(default_factory=list)
    # What was literally DONE, with no verdict attached: what a final judge is shown.
    actions: list[str] = field(default_factory=list)
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
    """An Explorer: learn one task by trying things. Reusable; keeps no state between runs.

    Args:
        llm: The computer-use model that proposes each move. Its ``total_usage`` is what
            the dollar and call budgets are charged from, so a critic sharing this client
            has its escalations charged to the run that caused them.
        critic: Defaults to a :class:`TieredCritic` over ``llm``, which answers most steps
            without a model call. Never re-implement judging here.
        graph: ``None`` means this run teaches the project nothing about the site.
        retriever: Consulted ONCE per run for skills worth reusing.
        runner: The sandbox code blocks execute in. ``None`` builds one with no skill
            store, so a block can act but cannot call a stored skill.
        policy: ``None`` - the DEFAULT - is ``llm`` asked with the acting prompt and the
            screenshot, the path every stored skill was learned on. See
            :class:`ActingPolicy` for what it does NOT replace.
        library: Read ONCE per run to find a workflow worth aiming at
            (:meth:`_adopt_skeleton`).
        move_critic: Who judges each MOVE. ``None`` - the DEFAULT - is ``critic``, so the
            path every stored skill was learned on is unchanged. The ``done`` claim is
            ALWAYS judged by ``critic``, whatever this is: a cheap per-move judge must
            never be what decides a run was solved. If it has an ``open_move`` method it
            is told the screen each move starts from, BEFORE the move is performed - see
            :class:`~skillweaver.agent.move_critic.LiteralMoveCritic` for why it needs to be.
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
        policy: ActingPolicy | None = None,
        library: Callable[[], Sequence[Skill]] | None = None,
        move_critic: Critic | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_block_actions: int = MAX_BLOCK_ACTIONS,
    ) -> None:
        self._library = library
        self._move_critic = move_critic
        self._llm = llm
        self._perceiver = perceiver
        self._critic: Critic = critic if critic is not None else TieredCritic(llm)
        # The free tiers alone, for the one move that has nothing else to be judged on:
        # see _judge_wait.
        self._wait_critic = TieredCritic(None)
        self._graph = graph
        self._retriever = retriever
        self._runner = runner if runner is not None else SkillRunner(None)
        self._policy = policy
        self._max_tokens = max_tokens
        self._max_block_actions = max_block_actions
        self._recorder = recorder if recorder is not None else _default_recorder()

    def __repr__(self) -> str:
        decides = self._policy.name() if self._policy is not None else self._llm.name()
        return f"Explorer(decides={decides!r}, graph={self._graph is not None})"

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
        # A critic that judges from the page's own text is told where the run began;
        # see agent/critic_final.py. The default critic has no such method.
        open_run = getattr(self._critic, "open_run", None)
        if callable(open_run):
            open_run(observation)
        run = _Run(
            task=task,
            spend=spend,
            memory=FailureMemory(),
            usage_mark=self._spent(),
            first=observation,
            current=observation,
        )
        try:
            self._loop(task, controller, run)
        except ControllerError:
            self._recorder.finish(False, "the controller broke mid-run")
            raise
        except PolicyBlocked as exc:
            # The policy's own terminal answer, not a limit and not a fault; recorded as
            # where the run stopped so the diagnosis names the screen.
            run.stopped_by = "blocked"
            run.detail = str(exc) or "the policy reported no supported operation could progress"
            log.info("explore.blocked", task=task.text, detail=run.detail)
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
        self._adopt_skeleton(task, run)

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
            if self._only_the_subgoal_is_done(move, run):
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
        open_move = getattr(self._move_critic, "open_move", None)
        if callable(open_move):
            open_move(before)
        performed = self._perform(move, catalog, controller, run)
        if performed:
            after = performed[-1].after
            verdict = self._judge(
                move.expect or move.summary, before, after, move, run, per_move=True
            )
            self._write_down(task, move, performed, verdict, run)
            run.current = after
            run.steps += len(performed)
            run.spend.add_step(len(performed))
            run.history.append(
                f"{run.moves}. {move.summary} -> "
                f"{'ok' if verdict.ok else 'FAILED'}: {_short(verdict.reason)}"
            )
            literal = (
                move.summary
                if move.action is not None
                else "; ".join(describe_action(step.action) for step in performed)
            )
            run.actions.append(f"{run.moves}. {literal}")
            if not verdict.ok:
                run.memory.remember(
                    before.fingerprint, move.signature, move.summary, verdict.reason
                )
                run.rejection = f"your last move ({move.summary}) did not work: {verdict.reason}"
            else:
                run.rejection = None
            self._follow(run, performed, verdict.ok)
        elif move.acts:
            # A move that reached nothing did not work, and is the one most likely to be
            # proposed again verbatim: the screen that suggested it is untouched.
            reason = run.rejection or "it performed no action at all"
            run.memory.remember(before.fingerprint, move.signature, move.summary, reason)
            run.history.append(f"{run.moves}. {move.summary} -> performed nothing: {reason}")

        if move.done:
            self._settle(task, move, run)

    def _settle(self, task: TaskSpec, move: Move, run: _Run) -> None:
        """Test the model's claim that the task is finished, and believe only the critic."""
        brief = getattr(self._critic, "brief_final", None)
        if callable(brief):
            brief(tuple(run.actions))
        verdict = self._judge(task.text, run.first, run.current, move, run)
        if verdict.ok:
            run.ok, run.verdict, run.stopped_by = True, verdict, "solved"
            log.info(
                "explore.solved",
                task=task.text,
                steps=run.steps,
                moves=run.moves,
                judged_by=getattr(verdict, "policy", "") or verdict.source,
            )
            return
        run.memory.remember(
            run.current.fingerprint,
            f"done:{move.signature}",
            f"claim the task is complete after {move.summary}",
            verdict.reason,
        )
        run.rejection = (
            "you said the task was complete, but the critic compared the first screen of "
            f"the run with this one and disagreed: {verdict.reason}"
        )

    # -- a relative's workflow, as something to aim at ---------------------------------

    def _adopt_skeleton(self, task: TaskSpec, run: _Run) -> None:
        """Give a cold run the workflow of the nearest stored relative, if there is one.

        An acting policy is handed the WHOLE errand on every decision, so every decision
        re-plans it: measured on walmart.com, asked to *add X to the cart, then open the
        cart*, the policy searched, opened the empty cart, went back, searched again,
        added, opened the cart - DONE at action 10 - then typed the product in again, 17
        actions for a five-step errand. A skill that has already done this kind of errand
        knows the order, and its signature says it with no site's labels in it.

        A PRIOR, NEVER A RAIL: nothing here constrains what may be proposed, grounded or
        performed. :meth:`_follow` drops it after :data:`MAX_STRAYS` moves that do not
        fit, and a run never fails for leaving it.
        """
        if self._library is None:
            return
        try:
            source = nearest_workflow(task.text, task.domain, self._library())
        except SkillWeaverError as exc:
            log.warning("explore.skeleton.unreadable", error=str(exc))
            return
        if source is None:
            log.info("explore.skeleton.none", task=task.text)
            return
        run.skeleton = skeleton(source.action_signature)
        run.skeleton_from = f"{source.name}@{source.domain}"
        log.info(
            "explore.skeleton.adopted",
            task=task.text,
            source=run.skeleton_from,
            learned=source.provenance.task_text,
            signature=render_signature(source.action_signature),
        )

    def _subgoal(self, run: _Run) -> Subgoal | None:
        """The step being aimed at, or ``None`` once the skeleton is spent or dropped."""
        return run.skeleton[run.at] if run.at < len(run.skeleton) else None

    def _aimed(self, task: TaskSpec, run: _Run) -> TaskSpec:
        """``task`` as an acting policy should be shown it RIGHT NOW.

        While a skeleton is followed the goal LEADS with the current subgoal and carries
        the errand after it - the errand cannot be left out, because it names the value to
        type and the thing to click. Otherwise this is ``task`` itself.
        """
        subgoal = self._subgoal(run)
        if subgoal is None:
            return task
        goal = (
            f"{subgoal.text.capitalize()}. This is step {run.at + 1} of "
            f"{len(run.skeleton)} of the errand, and the only step to do now. "
            f"The errand: {task.text}"
        )
        return replace(task, text=goal)

    def _workflow(self, run: _Run) -> str:
        """The skeleton as a prompt section, for the default (prompted) policy."""
        lines = [
            "A WORKFLOW THAT ALREADY WORKED FOR THIS KIND OF ERRAND "
            f"(from the stored skill {run.skeleton_from}; its labels and values were left "
            "out on purpose). It is a PRIOR, not a rule: aim at the step marked NEXT, and "
            "depart from it the moment this screen disagrees:"
        ]
        for index, step in enumerate(run.skeleton):
            mark = "done" if index < run.at else ("NEXT" if index == run.at else "later")
            lines.append(f"  {index + 1}. [{mark}] {step.text}")
        if run.at >= len(run.skeleton):
            lines.append("  Every step is done: check the whole task and say so if it is complete.")
        return "\n".join(lines)

    def _follow(self, run: _Run, performed: Sequence[_Performed], ok: bool) -> None:
        """Move along the skeleton, or away from it, after one judged move.

        A move that REACHED THE SCREEN and is the current step (or the one after it -
        sites skip steps) advances past it, whatever the critic said. One that worked and
        fits nothing is a stray, and :data:`MAX_STRAYS` in a row drop the skeleton.

        The verdict deliberately does not advance it, measured on live splitkb.com
        2026-09-20: requiring ``ok`` stalled the skeleton at step 0 of 4 after two moves,
        because ``state_changed`` calls a filled search field a failure (0.898 similar to
        an empty one) and an AJAX *Add to cart* a failure every time. A stalled skeleton
        tells the policy to REDO the step it just did.
        """
        if self._subgoal(run) is None:
            return
        tokens = tokens_of((p.action, p.before) for p in performed if p.result.ok)
        if not tokens:
            return
        fitted = False
        for token in tokens:
            for ahead in range(run.at, min(run.at + 1 + LOOKAHEAD, len(run.skeleton))):
                if run.skeleton[ahead].matches(token):
                    run.at, fitted = ahead + 1, True
                    break
        if fitted:
            run.strays = 0
            log.info("explore.skeleton.advanced", at=run.at, of=len(run.skeleton), judged_ok=ok)
            return
        if not ok:
            return
        run.strays += 1
        if run.strays >= MAX_STRAYS:
            log.info(
                "explore.skeleton.dropped",
                source=run.skeleton_from,
                at=run.at,
                of=len(run.skeleton),
                why=f"{run.strays} working move(s) in a row did not fit it",
            )
            run.skeleton, run.at = (), 0

    def _only_the_subgoal_is_done(self, move: Move, run: _Run) -> bool:
        """Whether a ``done`` claim was about the STEP the policy was aimed at.

        A policy shown a subgoal answers DONE when the subgoal is satisfied, which says
        nothing about the errand; settling it as a task claim would spend a critic call
        and file a dead end against a policy that was correct. So while steps remain, a
        bare DONE advances the skeleton and the policy is asked again.
        """
        if not move.done or move.acts or self._subgoal(run) is None:
            return False
        run.at += 1
        run.history.append(
            f"(step {run.at} of the borrowed workflow was already satisfied on this screen)"
        )
        log.info("explore.skeleton.step_satisfied", at=run.at, of=len(run.skeleton))
        return True

    # -- asking the model --------------------------------------------------------------

    def _ask(self, task: TaskSpec, run: _Run, catalog: ElementCatalog) -> str:
        """The next move, as an answer object: from the policy, or from the model.

        The policy branch is charged exactly as the model branch is (``at_least=1``), so a
        Jev step counts against ``max_llm_calls`` and the dollar budget on the same terms
        a Claude step does, even if its provider under-reports.
        """
        if self._policy is not None:
            try:
                answer = self._propose(self._aimed(task, run), run, catalog)
            except PolicyBlocked:
                if self._subgoal(run) is None:
                    raise
                # BLOCKED answers the question the policy was ASKED, and it was asked
                # about a borrowed step, so it may not end the run: the skeleton goes and
                # the whole errand is asked about the same screen. Measured on the sandbox
                # shop 2026-09-20, the second ask was BLOCKED too - the skeleton had not
                # caused the failure, only been in a position to.
                self._charge(run, at_least=1)
                log.info(
                    "explore.skeleton.dropped",
                    source=run.skeleton_from,
                    at=run.at,
                    of=len(run.skeleton),
                    why="the policy reported BLOCKED on a borrowed step",
                )
                run.skeleton, run.at = (), 0
                answer = self._propose(task, run, catalog)
            self._charge(run, at_least=1)
            return answer
        return self._ask_model(task, run, catalog)

    def _propose(self, task: TaskSpec, run: _Run, catalog: ElementCatalog) -> str:
        assert self._policy is not None
        return self._policy.propose(
            task,
            run.current,
            catalog,
            run.history,
            run.memory.near(run.current.fingerprint),
            run.rejection,
        )

    def _ask_model(self, task: TaskSpec, run: _Run, catalog: ElementCatalog) -> str:
        """One model call: the whole situation in one user turn, plus the screenshot.

        The conversation is rebuilt each time rather than grown, so what has been tried is
        a curated prompt section instead of something to infer from a transcript.
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
        if run.skeleton:
            sections.append(self._workflow(run))
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
            _Invalid: for anything unusable - not JSON, no decision in it, an unknown
                action kind, an element id not on the screen, an action this controller
                cannot perform. The loop turns each into a recoverable step failure.
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

        The same move on the same screen has the same outcome, so performing it would
        spend a step to learn nothing. The refusal names the previous reason, which is
        the input the model needs to propose something else.

        Raises:
            _Invalid: when the move is a known dead end here.
        """
        state = run.current.fingerprint
        known = run.memory.seen(state.value, move.signature)
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
                run.current.fingerprint,
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

        A code block goes through the same sandbox stored skills run in, with the
        catalogue bound to ``el``; a block that raises or hits its limits still keeps
        whatever it managed to do, because those actions really did happen.
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
        self,
        goal: str,
        before: Observation,
        after: Observation,
        move: Move,
        run: _Run,
        *,
        per_move: bool = False,
    ) -> Verdict:
        """One critic call, with whatever it cost charged to this run's budget.

        ``per_move`` is the ONLY door to ``move_critic``: a ``done`` claim never passes
        it, so the run's final verdict is the full critic's under every configuration.
        """
        if per_move and self._move_critic is None and self._policy_waited(move):
            return self._judge_wait(goal, before, after, move)
        critic = self._move_critic if per_move and self._move_critic is not None else self._critic
        verdict = critic.judge(goal, before, after, move.expect or None)
        self._charge(run, at_least=1 if getattr(verdict, "escalated", False) else 0)
        return verdict

    def _policy_waited(self, move: Move) -> bool:
        """Whether ``move`` is an acting policy letting time pass, and nothing else.

        The default explorer is excluded on purpose: its model WRITES an expectation for
        its wait and its prompt is calibrated against the full critic's answer to it.
        """
        return self._policy is not None and not move.done and isinstance(move.action, Wait)

    def _judge_wait(
        self, goal: str, before: Observation, after: Observation, move: Move
    ) -> Verdict:
        """A policy's bare wait, judged by the free checks alone - never by the model.

        A wait claims nothing about the task, so the only honest question is whether the
        screen moved while time passed, and the vetoes answer that for free. The full
        critic answered the same thing for one escalated call: measured on live splitkb
        (2026-09-20, slow moves), the ``wait 150ms`` fresh look a ``DONE`` is answered
        with cost 6.3s and 5.6s of vision model to be told "the pending add-to-cart
        completed" - on the way to a ``done`` claim the full critic then judges anyway,
        on the very screen this wait produced. An unchanged or error screen still fails
        exactly as before, by the same checks.
        """
        verdict = self._wait_critic.judge(goal, before, after, move.expect or None)
        if verdict.policy != "inconclusive-no-model":
            return verdict
        return CriticVerdict(
            ok=True,
            reason=(
                "the screen changed while the policy waited and shows no error (a wait "
                "claims nothing about the task, so nothing more was judged)"
            ),
            confidence=1.0,
            source="programmatic",
            escalated=False,
            policy="wait-changed",
            checks=verdict.checks,
        )

    def _write_down(
        self,
        task: TaskSpec,
        move: Move,
        performed: Sequence[_Performed],
        verdict: Verdict,
        run: _Run,
    ) -> None:
        """Record the move in the trajectory and in the site graph.

        One performed action is one trajectory step and one graph edge, primitive or
        block. The verdict belongs to the move, so it is attached to the move's LAST
        action - the one the critic looked at.

        Every edge is written, failures included: an edge that led nowhere is what stops a
        later run walking into the same dead end, and the graph sums statistics rather
        than overwriting them, so recording a failure is never a loss.
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

    def _spent(self) -> Usage:
        """Everything every model behind this loop has been asked for so far.

        The LLMClient plus, when there is one, the policy - a SECOND provider with its own
        meter, and a run counting only the first would report a Jev step as free. Reading
        the meters rather than the answers is the standing rule: a call the critic or a
        policy's text helper made is charged to the run that caused it.
        """
        total = self._llm.total_usage()
        policy_usage = getattr(self._policy, "total_usage", None)
        if policy_usage is None:
            return total
        try:
            reported = policy_usage()
        except Exception:  # noqa: BLE001 - a broken meter must not fail a run
            log.warning("explore.policy.usage_unreadable", policy=type(self._policy).__name__)
            return total
        return total + reported if isinstance(reported, Usage) else total

    def _charge(self, run: _Run, *, at_least: int) -> None:
        """Charge model spend since the last charge against the run's budget.

        The difference in the clients' own running totals, not one response, so a call the
        critic made on the same client is charged here too. ``at_least`` floors the call
        count, so a client that does not report ``calls`` cannot void ``max_llm_calls``.
        """
        total = self._spent()
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

        Best effort: a desktop controller cannot navigate and a browser may already be on
        the page, and neither is a reason not to attempt the task.
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
    being asked not to and neither is a reason to throw away a usable answer.
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

    Where grounding is enforced: every action with a target names an element id in
    ``catalog``, and the point is that element's center in LOGICAL pixels. The model never
    sends coordinates, so it can never send stale ones.

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
        case "back":
            if not controller.supports("back"):
                raise _Invalid(f"{controller.describe()} has no session history to go back")
            return Back(), "back", "go back to the previous page"
    raise _Invalid(f"action kind {kind!r} is not supported here")  # pragma: no cover


def _retarget(action: Action, catalog: ElementCatalog, observation: Observation) -> str:
    """``describe_action``, with pixels named as element ids.

    The graph stores a point, because a point is what a controller takes. Quoting it back
    at a model required to act by id would show it something it may not use, so a point is
    named by the element under it ON THE CURRENT SCREEN - and stays a point when nothing
    is there, which is the useful signal that the remembered move no longer applies.
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

    It goes through exactly the same static scan and limits as a stored skill: the
    explorer must not be a way to run code a skill would not be allowed to run.
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
    """The limit named by a BudgetExceeded message."""
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
