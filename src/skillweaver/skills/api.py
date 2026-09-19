"""``SkillContext``: the entire world a stored skill can reach.

A skill is Python source a model wrote. When it runs it gets exactly one object -
``ctx`` - and this module is that object. If something is not on ``ctx`` it does not
exist as far as skill code is concerned::

    def run(ctx, company):
        ctx.ctl.type_text(company)                       # hands
        rows = ctx.see.find_text(company, ElementKind.row)  # eyes, re-observed
        ctx.expect(bool(rows), f"no row for {company}")  # a clean failure
        ctx.ctl.click(rows[0])
        ctx.expect(bool(ctx.wait_for_text("Invoice")), "the invoice never opened")
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

``ctx.wait_for_text`` is ``ctx.see.find_text`` allowed to look again while the page
answers - the one wait this surface offers, and it waits for a THING rather than for a
TIME. See :data:`AWAIT_BUDGET_MS` for the control that makes it necessary.

*Unbounded work.* Every action goes through the :class:`RunLedger`, which charges a
step and checks the clock, so a skill cannot outrun its budget between two
observations. The ledger is shared by a whole composition, so a skill that calls
three skills is still held to one step budget. The clock it checks is the skill's
OWN: time spent blocked in the controller or the detector is not charged, because a
dense real page whose OCR takes four seconds is a slow world, not a runaway skill.
That forgiveness is not a licence for the world to take forever: this clock cannot
bound a call that is executing no Python at all, so the controller and the perceiver
each bound their own - see :class:`~skillweaver.perception.ocr.OcrWorker` - and a
failure of theirs arrives here as an exception, recorded by
:meth:`RunLedger.note_perception_failure` so the skill is not blamed for it.

*Silence.* ``ctx.log`` and every action land in the ledger's trace, which becomes
``SkillResult.trace`` - the thing skill synthesis feeds back to a model when a skill
needs repairing.

``sandbox.py`` compiles and executes the code; this module is what that code sees.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from skillweaver.config import DEFAULT_SKILL_MAX_SECONDS, settings
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
    ElementKind,
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
    ConfigError,
    ControllerError,
    ExpectationFailed,
    PerceptionError,
    SkillNotFound,
    SkillWeaverError,
)
from skillweaver.logging_ import get_logger

__all__ = [
    "AWAIT_BUDGET_MS",
    "AWAIT_POLL_MS",
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
    "default_max_seconds",
    "describe_action",
]

log = get_logger(__name__)

TRUNCATED = "... trace truncated"
"""The single line appended once a run's trace reaches ``SkillLimits.max_trace_lines``."""

AWAIT_BUDGET_MS = 4000.0
"""How long :meth:`SkillAPI.wait_for_text` will keep looking, by default.

**Not a sleep, and not the reflex wait that was removed from skills.** The difference
fits in one sentence: ``ctx.ctl.wait(ms)`` spends a fixed duration whatever the page
does, while this returns the instant the text is on screen and therefore costs NOTHING
on a page that already answered - its first look is the very observation the skill was
about to make anyway. That is why one is stripped by
:func:`~skillweaver.skills.refactor.strip_reflex_waits` and this one is written in by
:meth:`~skillweaver.skills.refactor._Hardener.await_expected_reads`.

It exists because a control the page answers WITHOUT navigating is invisible to
``BrowserController._settle``, which waits for the load event. Measured on live
splitkb.com on 2026-09-19: the click on "Add to cart" returned in 130ms with the
document complete and the OLD url still showing, and the cart it redirects to did not
commit until 1170ms. A skill recorded at model speed never sees that gap - a model was
being asked what to do next between every pair of actions - and the same skill replayed
at code speed reads the page it is still standing on and concludes, correctly for the
screen in front of it, that the cart is empty.

Four seconds is the same number :data:`~skillweaver.reset_actions.SETTLE_BUDGET_MS`
carries and for the same reason: it only has to outlast one slow request, and a wait
that will never be answered still says so within one budget rather than hanging.
"""

AWAIT_POLL_MS = 120.0
"""How long :meth:`SkillAPI.wait_for_text` pauses between looks.

Each look is a real observation - detection and OCR - so this is not the thing that
paces the loop; it is there so a perceiver that answers instantly is not spun flat out.
An unchanged page is cheap to re-read by design: ``CachingTextReader`` is keyed on the
exact pixels, so the second look at a page that has not moved pays no OCR at all.
"""


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


def default_max_seconds() -> float:
    """The configured skill time limit: ``SKILLWEAVER_SKILL_MAX_SECONDS``, else
    :data:`~skillweaver.config.DEFAULT_SKILL_MAX_SECONDS`.

    Read at construction rather than at import, so a process that sets the variable
    - the command line does, for ``--skill-max-seconds`` - gets the limit it asked
    for without every caller of :class:`SkillLimits` having to thread it through. An
    unreadable environment falls back to the default instead of refusing to run a
    skill: a misspelt variable is a configuration complaint the command line already
    makes, and it is not a reason for the sandbox to have no limit at all.
    """
    try:
        return settings().skill_max_seconds
    except ConfigError:
        return DEFAULT_SKILL_MAX_SECONDS


@dataclass(frozen=True, slots=True)
class SkillLimits:
    """Hard limits for one top-level skill execution, nested calls included.

    These are the sandbox's own limits and are unrelated to
    :class:`~skillweaver.contracts.Budget`, which governs a whole agent run. A skill
    is meant to be a short, known procedure: if it needs more than this it is not a
    skill yet.

    Attributes:
        max_steps: Controller actions the whole composition may perform.
        max_seconds: Seconds of the composition's OWN running time - see
            :meth:`RunLedger.elapsed_seconds`, which does not count time spent
            blocked in the controller or the perceiver. ``0`` disables the limit
            entirely, which only a test has any business doing. Defaults to
            :func:`default_max_seconds`, so the environment and the command line can
            raise it for a slow site without this class being edited.
        max_depth: How deep ``ctx.call`` may nest; ``1`` forbids composition.
        max_trace_lines: Trace lines kept before truncating, so a spinning skill
            cannot exhaust memory through ``ctx.log``.
    """

    max_steps: int = 40
    max_seconds: float = field(default_factory=default_max_seconds)
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

    What the clock does and does not count
    --------------------------------------

    The time limit exists to interrupt a runaway loop in code a model wrote. It is
    NOT a budget for how long the world may take to answer, and the two used to be
    conflated: a Wikipedia article is dense enough that one screenshot plus OCR takes
    seconds, so a three-observation skill that did its job perfectly could spend its
    whole allowance sitting still and be killed for it. That is what
    :meth:`blocked` fixes. Time inside the controller or the perceiver is banked in
    ``blocked_seconds`` and subtracted, so :attr:`elapsed_seconds` is the time the
    skill's own Python was running.

    Nothing is lost by this. A skill that spins is spinning in its own code and the
    tracer in ``sandbox.py`` still catches it within microseconds; a skill that acts
    forever is stopped by ``max_steps``; and a skill that sleeps on purpose through
    ``ctx.ctl.wait`` IS charged, because a deliberate sleep is the skill spending its
    own time rather than the page being slow.
    """

    limits: SkillLimits = field(default_factory=SkillLimits)
    steps: int = 0
    depth: int = 0
    stack: list[tuple[str, str]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    perception_failures: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    _truncated: bool = field(default=False, repr=False)
    _blocked: float = field(default=0.0, repr=False)
    _blocked_since: float | None = field(default=None, repr=False)
    _blocked_depth: int = field(default=0, repr=False)

    # -- time ---------------------------------------------------------------------------

    @property
    def blocked_seconds(self) -> float:
        """Seconds spent waiting on the controller or the perceiver, an unfinished
        wait included - so the figure is right when read from inside one."""
        pending = 0.0 if self._blocked_since is None else time.monotonic() - self._blocked_since
        return self._blocked + pending

    @property
    def elapsed_seconds(self) -> float:
        """Seconds the skill's OWN code has been running: wall-clock since the ledger
        was created, less :attr:`blocked_seconds`.

        Never negative and never decreasing while no wait is open, so the message a
        :class:`TimeLimitExceeded` carries is a number a reader can act on.
        """
        return max(time.monotonic() - self.started - self.blocked_seconds, 0.0)

    @contextmanager
    def blocked(self) -> Iterator[None]:
        """Hold the clock for the duration of a wait on the world.

        Re-entrant, and it stops the clock from the moment it is entered rather than
        banking the time on the way out: the deadline tracer fires while a screenshot
        is still being taken, so a limit that only learned about the wait afterwards
        would trip during exactly the wait it was meant to forgive.

        Never swallows the exception a failing controller or perceiver raises; the
        time is banked on the way out either way.
        """
        if self._blocked_depth == 0:
            self._blocked_since = time.monotonic()
        self._blocked_depth += 1
        try:
            yield
        finally:
            self._blocked_depth -= 1
            if self._blocked_depth == 0 and self._blocked_since is not None:
                self._blocked += time.monotonic() - self._blocked_since
                self._blocked_since = None

    def note_perception_failure(self, why: str) -> None:
        """Record that an OBSERVATION failed during this run, and say so in the trace.

        Kept separately from the trace because it changes a verdict, not just a
        report. A skill whose eyes stopped working has not failed its task - nothing
        at all has been learned about it - and ``sandbox.py`` reads this list to
        decide not to hold the run against it. See :meth:`SkillAPI.observe`, which is
        the only place that calls this.
        """
        self.perception_failures.append(why)
        self.note(f"perception FAILED: {why}")

    def check_time(self) -> None:
        """Raise :class:`TimeLimitExceeded` once the limit is reached.

        ``max_seconds`` of ``0`` means no limit; see :class:`SkillLimits`.

        This is a PYTHON-level check and it can only fire while Python is running.
        It is reached from :meth:`charge_step`, from the runner, and from a trace hook
        that fires every few frames - all three of which a call that has wedged inside
        a native library executes none of. Bounding such a call is the job of whoever
        makes it; :class:`~skillweaver.perception.ocr.OcrWorker` is where perception
        does it, and it is the reason this ledger sees a
        :class:`~skillweaver.errors.PerceptionError` rather than never being reached
        again.
        """
        limit = self.limits.max_seconds
        if limit > 0:
            elapsed = self.elapsed_seconds
            if elapsed >= limit:
                raise TimeLimitExceeded(
                    f"skill ran for {elapsed:.2f}s of {limit:.2f}s allowed "
                    f"(not counting {self.blocked_seconds:.2f}s waiting on the page)"
                )

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

    The time an action spends inside the controller is not charged to the skill's
    clock - a browser that takes a second to settle after a click is a slow browser,
    not a runaway skill - with one exception: a :class:`~skillweaver.contracts.Wait`
    IS charged, because a skill asking to sleep is spending its own time and is
    exactly the shape a "just wait longer" repair takes.

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
        waiting = nullcontext() if isinstance(action, Wait) else self._ledger.blocked()
        try:
            with waiting:
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

        Observing costs no time against the skill's limit, however slow the page: a
        capture and an OCR pass on a dense article can take seconds, and charging
        them would kill honest skills for reading a real web page.

        Raises:
            PerceptionError: if observing fails.
            TimeLimitExceeded: if the limit is already spent.
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

    def wait_for_text(
        self,
        text: str,
        kind: ElementKind | None = None,
        *,
        budget_ms: float | None = None,
    ) -> list[Element]:
        """``ctx.see.find_text`` that is allowed to look again while the page answers.

        Returns the matching elements, best first, exactly as ``find_text`` does - and
        an empty list when the budget ran out without them ever appearing, so a caller
        checks the result rather than catching something. Pair it with ``ctx.expect``,
        which is what turns "it never arrived" into an honest failure::

            ctx.ctl.click(add_to_cart[0])
            cart = ctx.wait_for_text("Subtotal")
            ctx.expect(bool(cart), "the cart never appeared after Add to cart")

        **Wait for a THING, not for a TIME.** That one line is the whole difference
        between this and ``ctx.ctl.wait(ms)``, which
        :func:`~skillweaver.skills.refactor.strip_reflex_waits` removes on sight: a
        duration is spent whether or not it was needed, and this returns the moment the
        text is there. On a page that has already answered the first look IS the
        observation the skill was about to make, so the fast path pays nothing - which
        is the property that let this ship beside a measured 2x speedup rather than
        against it. See :data:`AWAIT_BUDGET_MS` for the measurement that makes it
        necessary at all.

        Matching is ``fuzzy=False`` - equality or containment, case-insensitive - and
        that is not a detail. A wait decides which SCREEN the rest of the skill acts
        on, and a near match answers on the screen it was supposed to wait out: asked
        fuzzily for "Your cart" while still standing on the product page, the "Add to
        cart" button answers. This is the same choice, for the same reason, that
        ``_says`` makes in :mod:`skillweaver.reset_actions`. Name text that only the
        ANSWERED screen says.

        The time spent looking is banked as blocked, not charged: a page taking a
        second to commit is a slow world, not a runaway skill, and charging it would
        make the skill that waits correctly look more expensive than the one that reads
        too early and fails. The budget is what bounds it.

        Args:
            text: What the answered screen says. Not what the current one says.
            kind: Restrict to one element kind, as ``find_text`` does.
            budget_ms: How long to keep looking. Default :data:`AWAIT_BUDGET_MS`.

        Raises:
            PerceptionError: if observing fails, as ``ctx.see`` does.
            TimeLimitExceeded: if the skill's limit is already spent.
        """
        budget = AWAIT_BUDGET_MS if budget_ms is None else max(float(budget_ms), 0.0)
        deadline = time.monotonic() + budget / 1000.0
        looks = 0
        while True:
            looks += 1
            found = self.see.find_text(text, kind, fuzzy=False)
            if found:
                if looks > 1:
                    self.ledger.note(f"waited for {text!r}: on screen after {looks} looks")
                return found
            if time.monotonic() >= deadline:
                self.ledger.note(f"waited for {text!r}: still not on screen after {budget:.0f}ms")
                return []
            with self.ledger.blocked():
                time.sleep(AWAIT_POLL_MS / 1000.0)
            self._forget_observation()

    # -- for the runner, not for skill code ------------------------------------------------

    @property
    def domain(self) -> str:
        """The domain skills are resolved in: the executing skill's, else the one
        this context was built for."""
        return self.ledger.current_domain or self._domain

    def observe(self) -> Observation:
        """The current (possibly cached) observation. ``ctx.see`` is its index; a
        planner or the runner may want the screenshot, url or fingerprint too.

        The time the perceiver takes is banked as blocked, not charged - see
        :meth:`RunLedger.blocked`.

        Raises:
            PerceptionError: whatever the perceiver raises, recorded on the ledger
                first through :meth:`RunLedger.note_perception_failure`. The record is
                what keeps the blame straight: a read that was abandoned because the
                OCR engine wedged is not this skill failing, and the run must not be
                counted against it - including in the case where skill code catches
                the exception and then fails for its own reasons.
            TimeLimitExceeded: if the limit is already spent.
        """
        if self._observation is None:
            self.ledger.check_time()
            try:
                with self.ledger.blocked():
                    self._observation = self._perceiver.observe(self._controller)
            except PerceptionError as exc:
                self.ledger.note_perception_failure(f"{type(exc).__name__}: {exc}")
                raise
        return self._observation

    def _forget_observation(self) -> None:
        """Drop the cached observation; the next ``see`` re-observes."""
        self._observation = None
