"""``SkillContext``: the entire world a stored skill can reach.

A skill is Python source a model wrote. When it runs it gets exactly one object -
``ctx`` - and this module is that object. If something is not on ``ctx`` it does not
exist as far as skill code is concerned::

    def run(ctx, company):
        ctx.ctl.type_text(company)                       # hands
        rows = ctx.see.find_text(company, ElementKind.row)  # eyes, re-observed
        ctx.expect(bool(rows), f"no row for {company}")  # a clean failure
        ctx.ctl.click(rows[0])
        ctx.log(f"opened {company}")
        return ctx.call("confirm_payment")               # composition

Five deliberate omissions, each of which a model would otherwise reach for:

*The raw controller.* ``ctx.ctl`` is an :class:`ActionView` - eight ways to act and
nothing else. No ``capture``, no ``close``, no ``viewport``: a skill that wants to
know what is on screen asks ``ctx.see``, which is a fresh observation, not a frame a
skill has to decode itself.

*Ground truth.* :class:`~skillweaver.contracts.GroundTruthSource` reads the DOM and
is an offline teacher for labelling and scoring only. It is not merely absent from
this surface, it is unreachable from it: nothing here holds a reference to one, so
no amount of attribute walking finds it. An agent that works from pixels has to
actually work from pixels.

*Graph writing.* ``ctx.graph`` is wrapped in a :class:`ReadOnlyGraph`, so a skill can
route but cannot teach the graph things it has not verified.

*Unbounded work.* Every action goes through the :class:`RunLedger`, which charges a
step and checks the clock, so a skill cannot outrun its budget between two
observations. The ledger is shared by a whole composition, so a skill that calls
three skills is still held to one step budget.

*Silence.* ``ctx.log`` and every action land in the ledger's trace, which becomes
``SkillResult.trace`` - the thing skill synthesis feeds back to a model when a skill
needs repairing.

``sandbox.py`` compiles and executes the code; this module is what that code sees.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Box,
    Click,
    Controller,
    Drag,
    Element,
    ElementIndex,
    Fingerprint,
    GraphView,
    MouseButton,
    Move,
    Navigate,
    Observation,
    Perceiver,
    Point,
    PressKey,
    Route,
    Scroll,
    Transition,
    TypeText,
    UIState,
    Wait,
)
from skillweaver.errors import (
    ControllerError,
    ExpectationFailed,
    SkillNotFound,
    SkillWeaverError,
)
from skillweaver.logging_ import get_logger

__all__ = [
    "ActionView",
    "DepthLimitExceeded",
    "LimitExceeded",
    "NullGraph",
    "ReadOnlyGraph",
    "RunLedger",
    "SkillAPI",
    "SkillLimits",
    "StepLimitExceeded",
    "TimeLimitExceeded",
    "describe_action",
]

log = get_logger(__name__)

TRUNCATED = "... trace truncated"
"""The single line appended once a run's trace reaches ``SkillLimits.max_trace_lines``."""


# --------------------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------------------


class LimitExceeded(SkillWeaverError):
    """A skill ran past one of the sandbox's hard limits.

    Distinct from :class:`~skillweaver.errors.SandboxViolation` (the skill reached
    for something it may not touch) and from
    :class:`~skillweaver.errors.ExpectationFailed` (the screen was not what the skill
    expected). Synthesis treats the three differently: a violation needs the code
    rewritten, a limit needs it made shorter, a failed expectation may just mean the
    skill was run in the wrong place.
    """


class StepLimitExceeded(LimitExceeded):
    """The composition performed ``SkillLimits.max_steps`` controller actions."""


class TimeLimitExceeded(LimitExceeded):
    """The composition ran for ``SkillLimits.max_seconds`` wall-clock seconds."""


class DepthLimitExceeded(LimitExceeded):
    """``ctx.call`` nested deeper than ``SkillLimits.max_depth`` skills."""


@dataclass(frozen=True, slots=True)
class SkillLimits:
    """Hard limits for one top-level skill execution, nested calls included.

    These are the sandbox's own limits and are unrelated to
    :class:`~skillweaver.contracts.Budget`, which governs a whole agent run. A skill
    is meant to be a short, known procedure: if it needs more than this it is not a
    skill yet.

    Attributes:
        max_steps: Controller actions the whole composition may perform.
        max_seconds: Wall-clock seconds the whole composition may take.
        max_depth: How deep ``ctx.call`` may nest; ``1`` forbids composition.
        max_trace_lines: Trace lines kept before truncating, so a spinning skill
            cannot exhaust memory through ``ctx.log``.
    """

    max_steps: int = 40
    max_seconds: float = 20.0
    max_depth: int = 3
    max_trace_lines: int = 200


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class RunLedger:
    """Mutable bookkeeping for ONE top-level execution, shared by every nested skill.

    One ledger per ``SkillRunner.run`` at depth ``0``; ``ctx.call`` reuses it, which
    is what makes the limits apply to a composition rather than to each skill in it.
    Not thread-safe, like everything else on the action path.

    Inspect ``steps``, ``depth``, ``stack`` and ``trace`` after a run; the runner
    turns them into a :class:`~skillweaver.contracts.SkillResult`.
    """

    limits: SkillLimits = field(default_factory=SkillLimits)
    steps: int = 0
    depth: int = 0
    stack: list[tuple[str, str]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    _truncated: bool = field(default=False, repr=False)

    # -- time ---------------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since the ledger was created."""
        return time.monotonic() - self.started

    def check_time(self) -> None:
        """Raise :class:`TimeLimitExceeded` once the wall-clock limit is reached."""
        limit = self.limits.max_seconds
        if limit > 0:
            elapsed = self.elapsed_seconds
            if elapsed >= limit:
                raise TimeLimitExceeded(f"skill ran for {elapsed:.2f}s of {limit:.2f}s allowed")

    # -- steps --------------------------------------------------------------------------

    def charge_step(self, description: str) -> None:
        """Charge one controller action, after checking both the clock and the step
        budget.

        Raises:
            TimeLimitExceeded, StepLimitExceeded: when no budget remains. The action
                is NOT performed, because the check happens first.
        """
        self.check_time()
        limit = self.limits.max_steps
        if self.steps >= limit:
            raise StepLimitExceeded(f"skill performed {self.steps} actions of {limit} allowed")
        self.steps += 1
        self.note(description)

    # -- depth --------------------------------------------------------------------------

    @contextmanager
    def descend(self, name: str, domain: str) -> Iterator[None]:
        """Enter one skill, holding the composition depth for its duration.

        Raises:
            DepthLimitExceeded: naming the call chain, when entering would nest past
                ``SkillLimits.max_depth``. Mutual recursion between two skills stops
                here rather than in the interpreter.
        """
        limit = self.limits.max_depth
        if self.depth >= limit:
            chain = " -> ".join(n for n, _ in [*self.stack, (name, domain)])
            raise DepthLimitExceeded(f"skill calls nested past depth {limit}: {chain}")
        self.depth += 1
        self.stack.append((name, domain))
        try:
            yield
        finally:
            self.depth -= 1
            self.stack.pop()

    @property
    def current_domain(self) -> str:
        """The domain of the skill currently executing, or ``""`` outside a skill."""
        return self.stack[-1][1] if self.stack else ""

    # -- trace --------------------------------------------------------------------------

    def note(self, line: str) -> None:
        """Append one trace line, truncating the trace once it is long enough. Never
        raises: the trace is a record, and losing it must not lose the run."""
        if self._truncated:
            return
        if len(self.trace) >= self.limits.max_trace_lines:
            self.trace.append(TRUNCATED)
            self._truncated = True
            return
        self.trace.append(line)


# --------------------------------------------------------------------------------------
# Hands
# --------------------------------------------------------------------------------------


def describe_action(action: Action) -> str:
    """One short trace line for ``action``, e.g. ``click (420, 140)``."""
    match action:
        case Click(point=p, button=b, clicks=n):
            suffix = "" if (b, n) == ("left", 1) else f" {b} x{n}"
            return f"click ({p.x}, {p.y}){suffix}"
        case Move(point=p):
            return f"move ({p.x}, {p.y})"
        case Drag(start=a, end=b):
            return f"drag ({a.x}, {a.y}) -> ({b.x}, {b.y})"
        case TypeText(text=text):
            return f"type_text {text!r}"
        case PressKey(keys=keys):
            return "press " + "+".join(keys)
        case Scroll(point=p, dx=dx, dy=dy):
            return f"scroll ({p.x}, {p.y}) by ({dx}, {dy})"
        case Wait(ms=ms):
            return f"wait {ms}ms"
        case Navigate(url=url):
            return f"navigate {url}"
    return str(action)  # pragma: no cover - Action is a closed union


def _point_of(target: Point | Box | Element) -> Point:
    """The LOGICAL-pixel point a skill means by ``target``: a point as given, the
    center of a box, or the center of an element's box."""
    if isinstance(target, Point):
        return target
    if isinstance(target, Box):
        return target.center
    if isinstance(target, Element):
        return target.box.center
    raise TypeError(f"expected a Point, Box or Element target, got {type(target).__name__}")


class ActionView:
    """A :class:`~skillweaver.contracts.ActionSurface`: a controller narrowed to the
    eight things a skill is allowed to do to the screen.

    Every action is charged to the ledger BEFORE it is performed and invalidates the
    cached observation afterwards, whether or not it worked - a refused click can
    still have moved the UI. Unlike ``Controller.perform``, a failure raises
    :class:`~skillweaver.errors.ControllerError` so straight-line skill code does not
    have to check a result it would only ignore.

    The wrapped controller is held in a private slot and is not reachable from skill
    code: the sandbox rejects ``_``-prefixed attribute access at compile time.
    """

    __slots__ = ("_controller", "_ledger", "_on_action")

    def __init__(
        self,
        controller: Controller,
        ledger: RunLedger,
        on_action: Callable[[], None] | None = None,
    ) -> None:
        self._controller = controller
        self._ledger = ledger
        self._on_action = on_action

    def __repr__(self) -> str:
        return f"ActionView(steps={self._ledger.steps})"

    # -- ActionSurface -------------------------------------------------------------------

    def perform(self, action: Action) -> ActionResult:
        """Perform any action.

        Raises:
            ControllerError: if the action was not delivered.
            StepLimitExceeded, TimeLimitExceeded: if no budget remains, before the
                action is attempted.
        """
        self._ledger.charge_step(describe_action(action))
        try:
            result = self._controller.perform(action)
        finally:
            if self._on_action is not None:
                self._on_action()
        if not result.ok:
            raise ControllerError(f"{describe_action(action)} failed: {result.error}")
        return result

    def supports(self, action_kind: ActionKind) -> bool:
        """Whether the underlying controller can perform actions of ``action_kind``."""
        return self._controller.supports(action_kind)

    def click(
        self, target: Point | Box | Element, button: MouseButton = "left", clicks: int = 1
    ) -> ActionResult:
        """Click a point, the center of a box, or the center of an element."""
        return self.perform(Click(_point_of(target), button, clicks))

    def type_text(self, text: str) -> ActionResult:
        """Type literally into whatever has keyboard focus. Does not click first."""
        return self.perform(TypeText(text))

    def press(self, *keys: str) -> ActionResult:
        """Press a key chord: ``press("Enter")``, ``press("Meta", "a")``."""
        return self.perform(PressKey(tuple(keys)))

    def scroll(self, target: Point | Box | Element, dx: int = 0, dy: int = 0) -> ActionResult:
        """Scroll over the target; positive ``dy`` scrolls down. LOGICAL pixels."""
        return self.perform(Scroll(_point_of(target), dx, dy))

    def wait(self, ms: int) -> ActionResult:
        """Pause for ``ms`` milliseconds. Costs a step, like any other action."""
        return self.perform(Wait(ms))


# --------------------------------------------------------------------------------------
# The graph, read-only
# --------------------------------------------------------------------------------------


class ReadOnlyGraph:
    """The three query methods of a :class:`~skillweaver.contracts.GraphView`, over a
    graph that may well be a writable :class:`~skillweaver.contracts.SiteGraph`.

    Wrapping rather than passing the graph through is the point: skill code can route
    but cannot record a transition it never verified, or save one over the real graph.
    """

    __slots__ = ("_graph",)

    def __init__(self, graph: GraphView) -> None:
        self._graph = graph

    def __repr__(self) -> str:
        return f"ReadOnlyGraph({type(self._graph).__name__})"

    def route(self, src_fp: Fingerprint, dst_fp: Fingerprint) -> Route | None:
        """The lowest-cost known route, or ``None`` when no path is known."""
        return self._graph.route(src_fp, dst_fp)

    def neighbors(self, fp: Fingerprint) -> list[Transition]:
        """Outgoing edges of ``fp``, most reliable first; empty when unknown."""
        return self._graph.neighbors(fp)

    def states(self, domain: str) -> list[UIState]:
        """Every known state of ``domain``, oldest first; empty when none."""
        return self._graph.states(domain)


class NullGraph:
    """A :class:`~skillweaver.contracts.GraphView` that knows nothing.

    ``ctx.graph`` is always present, so a skill written against it does not have to
    guard for the graph being missing; with no graph configured it simply finds no
    routes. "Nothing is known" and "there is no graph" look the same from inside a
    skill, which is the honest answer to both.
    """

    __slots__ = ()

    def route(self, src_fp: Fingerprint, dst_fp: Fingerprint) -> Route | None:
        return None

    def neighbors(self, fp: Fingerprint) -> list[Transition]:
        return []

    def states(self, domain: str) -> list[UIState]:
        return []


# --------------------------------------------------------------------------------------
# The context
# --------------------------------------------------------------------------------------


Invoke = Callable[["SkillAPI", str, Mapping[str, Any]], Any]
"""What ``ctx.call`` delegates to: ``(ctx, name, kwargs) -> value``. ``SkillRunner``
supplies this, which is how a skill reaches another skill without this module knowing
anything about compiling or executing one."""


class SkillAPI:
    """The :class:`~skillweaver.contracts.SkillContext` handed to ``run(ctx, ...)``.

    Build one with ``SkillRunner.context(...)`` rather than directly: the runner has
    to wire ``invoke`` to itself for ``ctx.call`` to work, and has to share the
    ledger for the limits to cover a whole composition.

    ``ledger`` is public because the runner reads it to build a ``SkillResult``.
    Skill code cannot reach it: it is not a documented member of ``SkillContext``,
    and any attempt is one more name the sandbox's static check rejects.
    """

    __slots__ = (
        "_controller",
        "_ctl",
        "_domain",
        "_graph",
        "_invoke",
        "_observation",
        "_perceiver",
        "ledger",
    )

    def __init__(
        self,
        controller: Controller,
        perceiver: Perceiver,
        *,
        graph: GraphView | None = None,
        domain: str = "",
        ledger: RunLedger | None = None,
        invoke: Invoke | None = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else RunLedger()
        self._controller = controller
        self._perceiver = perceiver
        self._graph: GraphView = ReadOnlyGraph(graph) if graph is not None else NullGraph()
        self._domain = domain
        self._invoke = invoke
        self._observation: Observation | None = None
        self._ctl = ActionView(controller, self.ledger, self._forget_observation)

    def __repr__(self) -> str:
        return f"SkillAPI(domain={self.domain!r}, steps={self.ledger.steps})"

    # -- the published surface ------------------------------------------------------------

    @property
    def ctl(self) -> ActionView:
        """Hands: the action-only view of the controller."""
        return self._ctl

    @property
    def see(self) -> ElementIndex:
        """Eyes: an index of the screen AS IT IS NOW.

        The observation is cached until the next action through ``ctl``, so reading
        ``ctx.see`` twice in a row costs one capture, and reading it after acting
        costs a fresh one. Never hold on to it across an action.

        Raises:
            PerceptionError: if observing fails.
            TimeLimitExceeded: if the wall-clock limit is already spent.
        """
        return self.observe().index

    @property
    def graph(self) -> GraphView:
        """Read-only view of the current domain's site graph."""
        return self._graph

    def call(self, name: str, **kwargs: Any) -> Any:
        """Run another skill of this skill's domain and return its value.

        The callee shares this context: the same controller, the same ledger and the
        same trace, so its actions count against the caller's budget and its log
        lines appear in the caller's trace.

        Raises:
            SkillNotFound: if no such skill exists, or no store is configured.
            DepthLimitExceeded: if the composition is already as deep as it may go.
            ExpectationFailed, ControllerError, SandboxViolation: propagated from the
                callee, because the caller may want to handle them.
        """
        if self._invoke is None:
            raise SkillNotFound(f"cannot call {name!r}: this context has no skill runner")
        return self._invoke(self, name, kwargs)

    def expect(self, condition: bool, why: str) -> None:
        """Assert something about the screen.

        This is how a skill fails HONESTLY: a false ``condition`` means the world was
        not as the skill requires, which is a clean failure and not a bug in the
        code. The sandbox reports it as such, so synthesis does not try to repair
        code that was right.

        Raises:
            ExpectationFailed: with ``why`` as its message, when ``condition`` is false.
        """
        if condition:
            # ``why`` describes the FAILURE, so "ruled out: ..." is the only phrasing
            # that does not read, in a trace a model will later be asked to repair
            # from, as an assertion that the bad thing happened.
            self.ledger.note(f"ruled out: {why}")
            return
        self.ledger.note(f"expect FAILED: {why}")
        raise ExpectationFailed(why)

    def log(self, msg: str) -> None:
        """Append a line to the run's trace. Never raises, whatever ``msg`` is."""
        try:
            self.ledger.note(f"log: {msg}")
        except Exception:  # pragma: no cover - note() is already total
            pass

    # -- for the runner, not for skill code ------------------------------------------------

    @property
    def domain(self) -> str:
        """The domain skills are resolved in: the executing skill's, else the one
        this context was built for."""
        return self.ledger.current_domain or self._domain

    def observe(self) -> Observation:
        """The current (possibly cached) observation. ``ctx.see`` is its index; a
        planner or the runner may want the screenshot, url or fingerprint too."""
        if self._observation is None:
            self.ledger.check_time()
            self._observation = self._perceiver.observe(self._controller)
        return self._observation

    def _forget_observation(self) -> None:
        """Drop the cached observation; the next ``see`` re-observes."""
        self._observation = None
