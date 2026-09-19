"""An undo that is PERFORMED in the world, rather than fetched from it.

A :data:`~skillweaver.orchestrator.WorldReset` is "how do I put this back?", and the
admission gate cannot learn a mutating task without one. The only undo the command
line could express was a single HTTP GET - ``--reset-url`` - which exists on a demo
app and on essentially nothing else. DoorDash has no endpoint that empties a cart, so
every cart task was unlearnable: not because the agent could not do it, but because
its undo could not be written down.

This module widens the expression, not the contract. A reset is an ordered list of
:class:`ResetStep` - each one an ordinary :data:`~skillweaver.contracts.Action` the
controller already performs - and :func:`world_reset_from_actions` turns the list into
the same zero-argument callable the gate has always taken.

Why a step is not just an action
--------------------------------

The gate re-runs a candidate up to three times and resets BETWEEN attempts, so an undo
that works four times in five does not produce a flaky demo - it produces a REJECTED
skill, because the fourth attempt starts on a dirty screen and disagrees with the first
three. You would have moved the failure, not fixed it. Four properties buy the
reliability, and each is a field:

* :attr:`ResetStep.until_gone` and :attr:`ResetStep.until_seen` make the step a LOOP
  with a verified exit condition. "Remove every line until the cart is empty" is not a
  number of clicks; it is a condition, re-read after every click, and it is what makes
  the reset idempotent - a clean world already satisfies it, so the step performs
  nothing and succeeds.
* :attr:`ResetStep.max_rounds` bounds that loop. A loop that cannot converge reports
  :class:`ResetDidNotConverge`, which the caller reads as a ``failed`` reset - the
  honest answer, and the one that keeps an unreliable undo from being laundered into a
  stored skill.
* :attr:`ResetStep.find` names the target by CONTENT rather than by pixel. A notice
  arriving at the top of a page - an appeal, a cookie bar, an A/B strip - pushes
  everything below it down, and a reset pinned to a grid position clicks whatever slid
  into that spot. The same lesson the fingerprinter learned (see
  :class:`~skillweaver.perception.fingerprint.StateFingerprinter`) applies to hands as
  well as to eyes: anchor on what a thing SAYS, and let :attr:`ResetStep.dx` /
  :attr:`ResetStep.dy` carry the within-row geometry that no text can express - the
  unlabelled trash icon at the end of the line the anchor sits on.
* :attr:`ResetStep.via` chooses which reading of the screen answers all three. See
  below; it is the difference between a reset that works on a real site and one that
  does not.

An anchor is LOOKED UP, not ranked
----------------------------------

``find`` resolves through ``ElementIndex.find_text(..., fuzzy=False)`` - case-
insensitive containment - and not through ``ElementIndex.best``, which is the same
distinction the planner draws between a ranking and a decision. ``best`` is a ranking,
and a ranking has a winner even when nothing fits: asked for ``"cart"`` on a screen
with no cart it returned a mail message beginning "First pass at the hero section",
and the reset clicked it, found no cart lines to remove and reported the world
restored. A lookup can answer "not here", which is the answer that makes a step fail
loudly instead of succeeding on the wrong screen. Aim at text the page literally says.

Say what the clean world LOOKS like, not only what it lacks
------------------------------------------------------------

``until_gone`` is a negative test, and a negative test passes on every screen that
does not say the thing - including the wrong screen. That is not hypothetical: a
sandbox reset whose first step missed the cart button ran its loop on the mail
inbox, found no cart lines there, and reported the world restored. ``until_seen``
is the positive test that cannot do this - "Your cart is empty" is a sentence only
the emptied cart says - and a step may carry both, in which case both must hold
before it is done. Prefer the positive one. It is also the check that survives a
site where removing a line and lowering a quantity look the same: emptiness is a
screen, not the absence of a row.

Two readings of one screen, and why the DOM is allowed here
------------------------------------------------------------

:attr:`ResetStep.via` picks ``"screen"`` - pixels, detection and OCR, the same eyes
the agent has - or ``"dom"``, the :class:`~skillweaver.contracts.GroundTruthSource`
the caller supplies. ``"screen"`` is the default and needs nothing; it is all a
desktop controller can offer.

``"dom"`` exists because a real site does not label its controls in pixels. DoorDash's
quick-add and its header cart are icon-only ``<button>`` elements whose only name is an
``aria-label`` - ``"Add item to cart"``, ``"1 items, open Order Cart"`` - and a search
for visible text finds NOTHING on that page. No amount of OCR recovers a name that was
never painted.

That source is documented as an offline teacher the agent must never touch, and this
does not breach it. A reset is not the agent: it is scaffolding, the peer of the
``curl`` behind ``--reset-url``, which does not look at the screen at all. The agent
never sees it, no skill is synthesized from it, and nothing it reads reaches a
trajectory. It is passed to :func:`world_reset_from_actions` as an explicit argument
for exactly the reason that Protocol asks for - so the dependency is visible at the
call site rather than reachable from anywhere.

Nothing here weakens the gate. An undo that cannot prove it worked raises, the gate
reports the reset as failed, and no skill is stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from skillweaver.contracts import (
    Action,
    Click,
    Controller,
    ElementIndex,
    ElementKind,
    GroundTruthSource,
    Move,
    Perceiver,
    Point,
    Scroll,
    action_from_dict,
    action_to_dict,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.perception.elements import build_index

__all__ = [
    "DEFAULT_MAX_ROUNDS",
    "RESET_ACTIONS_PARAM",
    "Oracle",
    "ResetDidNotConverge",
    "ResetStep",
    "ResetStepFailed",
    "chain_resets",
    "reset_step_from_dict",
    "reset_step_to_dict",
    "reset_steps_from",
    "world_reset_from_actions",
]

log = get_logger(__name__)


RESET_ACTIONS_PARAM = "reset_actions"
"""The task parameter holding an action reset.

It rides in ``TaskSpec.params`` beside ``reset_url`` and ``read_only`` for the reason
those do: a caller that already builds a workbench keeps working unchanged and gains
the hook by naming it. Its value is whatever :func:`reset_steps_from` accepts.
"""


Oracle = Literal["screen", "dom"]
"""Which reading of the screen a step's text is looked up in.

``"screen"`` is detection and OCR - what the agent itself sees, and all a desktop
controller can offer. ``"dom"`` is the caller-supplied
:class:`~skillweaver.contracts.GroundTruthSource`, which is the only way to reach a
control whose name lives in an ``aria-label`` and was never painted. See the module
docstring for why a reset may read it and the agent may not.
"""


DEFAULT_MAX_ROUNDS = 12
"""How many times a converging step may act before it gives up.

A bound, not a target: the whole point of :attr:`ResetStep.until_gone` is that the
step stops when the condition clears, which on a clean world is immediately. The
number only has to exceed the largest mess a task can make - a cart with more lines
than anyone orders - while staying small enough that a step which will NEVER converge
(a control that does not remove anything, a condition naming text that is always on
the page) says so in seconds instead of spinning. Raise it per step when a task really
can dirty the world more than this.
"""


class ResetStepFailed(SkillWeaverError):
    """One step of an action reset could not be performed.

    A :class:`~skillweaver.errors.SkillWeaverError` rather than an ``OSError``
    because it is this project's own failure, and both are read by
    :func:`~skillweaver.orchestrator.reset_world` as ``failed`` - "a real way back was
    tried and did not work this time" - which is exactly what it is. It is NOT a
    :class:`~skillweaver.orchestrator.ResetRefused`: that means "there is no reset
    here and never will be", and a step list is a reset whether or not it worked.
    """


class ResetDidNotConverge(ResetStepFailed):
    """A converging step ran out of rounds with its exit condition still true.

    The one failure this module exists to report honestly. The alternative - shrugging
    and returning - hands the admission gate a dirty screen and lets it blame the
    candidate skill for the mess, which is how an unreliable undo gets laundered into
    a stored skill that only works on a clean world.
    """


@dataclass(frozen=True, slots=True)
class ResetStep:
    """One step of an action reset: an action, optionally aimed and optionally looped.

    Attributes:
        action: What to do, in the controller's own vocabulary. When :attr:`find` is
            set, the action must be one that carries a point (:class:`Click`,
            :class:`Move`, :class:`Scroll`) and that point is replaced at run time.
        find: Text the ANCHOR says, looked up by ``find_text(..., fuzzy=False)``:
            case-insensitive, an exact match ranked above a containing one, and
            NOTHING when the screen does not say it. Empty means the action's own
            point is used as written, which is what a task that really does mean a
            fixed pixel wants.
        anchor_kind: Restricts the anchor to one element kind, for the screen where
            the word appears both as a label and on the control beside it. Spelled
            out rather than ``kind`` because a step dict's ``kind`` is the ACTION's -
            one key cannot mean two things, and the collision silently rewrote the
            action the first time this was tried.
        dx / dy: Logical pixels from the anchor's centre to the thing actually
            clicked. This is how an unlabelled control is reached - the trash icon at
            the end of the line whose price text is the anchor - WITHOUT naming a
            position on the page: the pair travels with the anchor when the page
            moves. Requires :attr:`find`.
        until_gone: Text whose presence means the world is still dirty. Non-empty
            turns the step into a loop: read the screen, and if nothing says this,
            stop; otherwise act and look again.
        until_seen: Text whose presence means the world is clean - "Your cart is
            empty". Also turns the step into a loop, and is the STRONGER of the two:
            a negative test passes on any screen that lacks the marker, including a
            screen an earlier step failed to leave. With both, both must hold.
        via: Which reading of the screen answers :attr:`find`, :attr:`until_gone` and
            :attr:`until_seen`. See :data:`Oracle`.
        max_rounds: The cap on actions this loop may perform. See
            :data:`DEFAULT_MAX_ROUNDS`.
        optional: Whether a :attr:`find` that matches nothing is a no-op that
            succeeds rather than a failure. For a step that opens a screen which may
            already be open. Meaningless - and rejected - on a looping step, where
            "the target is not there" either ends the loop through the exit condition
            or is the reason it cannot converge.

    Raises:
        ValueError: if the fields do not fit together.
    """

    action: Action
    find: str = ""
    anchor_kind: ElementKind | None = None
    dx: int = 0
    dy: int = 0
    until_gone: str = ""
    until_seen: str = ""
    via: Oracle = "screen"
    max_rounds: int = DEFAULT_MAX_ROUNDS
    optional: bool = False

    @property
    def loops(self) -> bool:
        """Whether this step repeats until a condition holds."""
        return bool(self.until_gone or self.until_seen)

    @property
    def reads_screen(self) -> bool:
        """Whether performing this step needs the screen read at all."""
        return bool(self.find) or self.loops

    def __post_init__(self) -> None:
        if self.find and not isinstance(self.action, Click | Move | Scroll):
            raise ValueError(
                f"'find' aims an action that has a point, so it cannot be used with "
                f"{self.action.kind!r}; drop it, or aim a click instead"
            )
        if self.anchor_kind is not None and not self.find:
            raise ValueError("'anchor_kind' narrows the anchor 'find' looks up, so it needs one")
        if (self.dx or self.dy) and not self.find:
            raise ValueError(
                "'dx'/'dy' are measured from the element 'find' locates, so they need "
                "one; without 'find', write the point into the action itself"
            )
        if self.via not in ("screen", "dom"):
            raise ValueError(f"'via' is 'screen' or 'dom', not {self.via!r}")
        if self.max_rounds < 1:
            raise ValueError(f"'max_rounds' must be at least 1, got {self.max_rounds!r}")
        if self.optional and self.loops:
            raise ValueError(
                "'optional' and a loop condition contradict each other: a looping step "
                "that cannot find its target is a step that cannot converge, and "
                "silently skipping it would spin until the bound and then lie"
            )

    def __str__(self) -> str:
        kind = f" {self.anchor_kind.value}" if self.anchor_kind is not None else ""
        where = f" at{kind} {self.find!r}" if self.find else ""
        nudge = f" {self.dx:+d},{self.dy:+d}" if self.dx or self.dy else ""
        clauses = [f"{text!r} is gone" for text in (self.until_gone,) if text]
        clauses += [f"{text!r} is on screen" for text in (self.until_seen,) if text]
        loop = f" until {' and '.join(clauses)} (<={self.max_rounds})" if clauses else ""
        seen = f" via {self.via}" if self.reads_screen else ""
        return f"{self.action.kind}{where}{nudge}{loop}{seen}"


_STEP_FIELDS = frozenset(
    {
        "find",
        "anchor_kind",
        "dx",
        "dy",
        "until_gone",
        "until_seen",
        "via",
        "max_rounds",
        "optional",
    }
)
"""Keys of a step dict that belong to the step rather than to the action it carries.

Split out before ``action_from_dict`` sees the dict, because that function turns every
mapping value into a ``Point`` and every list into a tuple - it is reading an action,
and a step is an action plus the things this module adds to it.
"""


def reset_step_from_dict(data: Mapping[str, Any]) -> ResetStep:
    """Build one :class:`ResetStep` from a JSON-shaped mapping.

    The mapping is an action dict - ``{"kind": "click", "point": {...}}``, exactly
    what :func:`~skillweaver.contracts.action_to_dict` writes - with any of
    :data:`_STEP_FIELDS` alongside it. A click that is aimed by ``find`` needs no
    ``point``; one is supplied so the action can be built and then replaced.

    Raises:
        ValueError: if the action or the step fields do not parse.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"a reset step must be an object, got {type(data).__name__}")
    extras = {k: v for k, v in data.items() if k in _STEP_FIELDS}
    action_data = {k: v for k, v in data.items() if k not in _STEP_FIELDS}
    if extras.get("find") and "point" not in action_data and action_data.get("kind") in _AIMABLE:
        action_data["point"] = {"x": 0, "y": 0}
    if "anchor_kind" in extras:
        try:
            extras["anchor_kind"] = ElementKind(extras["anchor_kind"])
        except ValueError as exc:
            raise ValueError(f"'anchor_kind': {exc}") from exc
    action = action_from_dict(action_data)
    try:
        return ResetStep(action, **extras)
    except TypeError as exc:  # pragma: no cover - _STEP_FIELDS keeps this unreachable
        raise ValueError(f"bad fields for a reset step: {exc}") from exc


_AIMABLE = frozenset({Click.kind, Move.kind, Scroll.kind})
"""Action kinds that carry a point, and so can be aimed by :attr:`ResetStep.find`."""


def reset_step_to_dict(step: ResetStep) -> dict[str, Any]:
    """The JSON-shaped mapping :func:`reset_step_from_dict` reads back.

    Only the fields that differ from their defaults are written, so a step round-trips
    to roughly what a human typed rather than to every knob this module has. It is
    what a parsed reset is stored back onto ``TaskSpec.params`` as: params are handed
    to the composer and the explorer as text, and a page of dataclass reprs there is
    tokens paid for nothing.
    """
    data = action_to_dict(step.action)
    if step.find:
        # The written point is a placeholder once an anchor names the target; keeping
        # it would suggest a fixed pixel that is never clicked.
        data.pop("point", None)
        data["find"] = step.find
        if step.anchor_kind is not None:
            data["anchor_kind"] = step.anchor_kind.value
    for name in ("dx", "dy", "until_gone", "until_seen", "optional"):
        value = getattr(step, name)
        if value:
            data[name] = value
    if step.reads_screen and step.via != "screen":
        data["via"] = step.via
    if step.loops and step.max_rounds != DEFAULT_MAX_ROUNDS:
        data["max_rounds"] = step.max_rounds
    return data


def reset_steps_from(value: Any) -> tuple[ResetStep, ...]:
    """Parse an action reset from what a command line or a task parameter holds.

    Accepts a JSON array as a string (``-p reset_actions='[{"kind": ...}]'``), an
    already-decoded sequence of mappings, or a sequence of :class:`ResetStep`. An
    empty list is a real answer - "this task configures no action reset" - and comes
    back as an empty tuple.

    Raises:
        ValueError: if the value is not one of those, or a step does not parse. The
            message names the step's position, because a list is typed by hand.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"reset steps are a JSON array, and this is not one: {exc}") from exc
    if isinstance(value, Mapping) or not isinstance(value, Sequence | Iterable):
        raise ValueError(
            f"reset steps are an ARRAY of steps, got {type(value).__name__}; wrap a "
            "single step in a list"
        )
    steps: list[ResetStep] = []
    for position, item in enumerate(value, 1):
        if isinstance(item, ResetStep):
            steps.append(item)
            continue
        try:
            steps.append(reset_step_from_dict(item))
        except ValueError as exc:
            raise ValueError(f"reset step {position}: {exc}") from exc
    return tuple(steps)


def world_reset_from_actions(
    steps: Sequence[ResetStep],
    *,
    controller: Controller,
    perceiver: Perceiver,
    truth: GroundTruthSource | None = None,
) -> Any:
    """A :data:`~skillweaver.orchestrator.WorldReset` that PERFORMS ``steps``.

    The returned callable runs the steps in order against ``controller``, reading the
    screen whenever a step has to aim or has to decide whether it is done. It returns
    normally only when every step finished and every converging step saw its condition
    hold; otherwise it raises :class:`ResetStepFailed`, which the caller reads as a
    ``failed`` reset rather than as a crash.

    Idempotent by construction, as long as every step that undoes something is a
    converging one: on an already-clean world each such step reads the screen once,
    finds its condition already satisfied, and performs nothing.

    Args:
        steps: The undo, in order. Empty is allowed and does nothing.
        controller: The hands. It is the SAME controller the gate re-runs the
            candidate with, so the reset leaves the screen where the next step of the
            gate expects to find it - the gate navigates afterwards regardless.
        perceiver: The eyes, for a step reading ``via="screen"``. Consulted only when
            a step aims or loops, because an observation is the most expensive thing
            this module can do (OCR dominates it; see :mod:`skillweaver.perception.ocr`).
        truth: The DOM, for a step reading ``via="dom"``. ``None`` means no step may
            ask for one, and a step that does fails saying so rather than quietly
            falling back to pixels - a reset that silently changed which reading it
            trusted would be the flaky undo this module exists to rule out. See the
            module docstring for why scaffolding may read this and the agent may not.
    """
    plan = tuple(steps)
    read = _Reader(controller, perceiver, truth)

    def reset() -> None:
        for position, step in enumerate(plan, 1):
            _run_step(step, position, controller, read)
        log.info("agent.world_reset.actions", steps=len(plan))

    return reset


def chain_resets(*resets: Any) -> Any:
    """Run several resets in order as one, skipping the ``None`` ones.

    Both undo kinds coexist rather than compete: a site with a seed-restoring endpoint
    AND a screen that needs tidying gets both, URL first. Returns ``None`` when every
    argument was ``None``, which is the caller's "nothing can put this back" and must
    stay distinguishable from a reset that does nothing.
    """
    live = [r for r in resets if r is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def reset() -> None:
        for one in live:
            one()

    return reset


# --------------------------------------------------------------------------------------
# Performing one step
# --------------------------------------------------------------------------------------


class _Reader:
    """Reads the screen the way a step asks to, and says so plainly when it cannot.

    One object rather than two arguments threaded everywhere, because the choice is
    per step and the failure - "this step wants the DOM and no DOM was supplied" -
    has to name the step. It holds no state: nothing here is cached, because the whole
    point of a converging loop is that each round looks at a screen that just changed.
    """

    __slots__ = ("_controller", "_perceiver", "_truth")

    def __init__(
        self, controller: Controller, perceiver: Perceiver, truth: GroundTruthSource | None
    ) -> None:
        self._controller = controller
        self._perceiver = perceiver
        self._truth = truth

    def __call__(self, step: ResetStep, position: int) -> ElementIndex:
        """The screen as ``step`` wants it read, as something ``find_text`` works on."""
        try:
            if step.via == "dom":
                if self._truth is None:
                    raise ResetStepFailed(
                        f"reset step {position} ({step}) asked to read the DOM, and this "
                        "world has no ground-truth reader - a desktop controller has "
                        "none, and a browser one is only supplied by the session that "
                        "opened it. Use via='screen' here, or run this reset in a browser."
                    )
                return build_index(self._truth.elements())
            return self._perceiver.observe(self._controller).index
        except ResetStepFailed:
            raise
        except SkillWeaverError as exc:
            raise ResetStepFailed(
                f"reset step {position} ({step}) could not read the screen, so it could "
                f"not tell whether the world was put back: {type(exc).__name__}: {exc}"
            ) from exc


def _run_step(step: ResetStep, position: int, controller: Controller, read: _Reader) -> None:
    """Perform one step, looping and verifying when it asked to."""
    if not step.loops:
        _perform(step, position, controller, read, seen=None)
        return
    for performed in range(step.max_rounds + 1):
        seen = read(step, position)
        if _settled(seen, step):
            log.info("agent.world_reset.converged", step=position, rounds=performed)
            if performed == 0 and not step.until_seen:
                # Did nothing and declared victory, on the word of a NEGATIVE test.
                # True of a world that was already clean, and equally true of a world
                # an earlier step never reached: the sandbox cart reset that missed
                # the cart button ran this loop on the mail inbox and reported the
                # world restored with three lines still in it. Say so, every time,
                # because the reader is the only thing that can tell the two apart.
                log.warning(
                    "agent.world_reset.unverified",
                    step=position,
                    until_gone=step.until_gone,
                    why="nothing was undone and only the absence of text says the world "
                    "is clean; add 'until_seen' naming something the clean screen SAYS",
                )
            return
        if performed == step.max_rounds:
            break
        _perform(step, position, controller, read, seen=seen)
    raise ResetDidNotConverge(
        f"reset step {position} ({step}) performed {step.max_rounds} actions and the "
        "world still does not look put back. Either the step does not undo what it is "
        "aimed at, or its condition names text that does not change with the state."
    )


def _settled(seen: ElementIndex, step: ResetStep) -> bool:
    """Whether ``step``'s exit condition holds on this reading of the screen.

    Both clauses must, when both are given. ``until_gone`` alone is a negative test
    and passes on any screen lacking the marker, which is why ``until_seen`` exists;
    see the module docstring for the run that proved it.
    """
    if step.until_gone and _says(seen, step.until_gone):
        return False
    return not (step.until_seen and not _says(seen, step.until_seen))


def _says(seen: ElementIndex, text: str) -> bool:
    """Whether anything on this screen reads as ``text``.

    ``fuzzy=False``: equality or containment, case-insensitive. A converging loop's
    exit condition decides whether a skill is admitted, and a fuzzy match would let an
    unrelated word keep the loop running - or, worse, stop it early on a screen that
    only nearly says the marker is gone.
    """
    return bool(seen.find_text(text, fuzzy=False))


def _perform(
    step: ResetStep,
    position: int,
    controller: Controller,
    read: _Reader,
    *,
    seen: ElementIndex | None,
) -> None:
    """Aim the step's action if it asked to, perform it, and insist that it landed."""
    action = step.action
    if step.find:
        screen = seen if seen is not None else read(step, position)
        matches = screen.find_text(step.find, kind=step.anchor_kind, fuzzy=False)
        if not matches:
            if step.optional:
                log.info("agent.world_reset.skipped", step=position, find=step.find)
                return
            raise ResetStepFailed(
                f"reset step {position} ({step}) found nothing on screen saying "
                f"{step.find!r}, so there was no way to aim it - and a reset that "
                "cannot find its target has almost certainly been left on the wrong "
                "screen by an earlier step. Name text the page actually says, or mark "
                "the step optional if it may legitimately be missing."
            )
        centre = matches[0].box.center
        action = _aimed(action, Point(centre.x + step.dx, centre.y + step.dy))
    result = controller.perform(action)
    if not result.ok:
        raise ResetStepFailed(f"reset step {position} ({step}) was not performed: {result.error}")
    log.info("agent.world_reset.step", step=position, action=action_to_dict(action))


def _aimed(action: Action, point: Point) -> Action:
    """``action`` with its point replaced. ``ResetStep`` has already checked the kind."""
    if isinstance(action, Click | Move | Scroll):
        return replace(action, point=point)
    raise ResetStepFailed(  # pragma: no cover - ResetStep.__post_init__ rejects this
        f"{action.kind!r} carries no point to aim"
    )
