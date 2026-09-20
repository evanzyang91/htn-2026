"""An undo PERFORMED on the screen, for a site with no reset endpoint.

A reset is an ordered list of ``ResetStep`` - an ordinary ``Action``, optionally aimed
by anchor text and optionally looped until a condition holds - that
``world_reset_from_actions`` turns into the zero-argument ``WorldReset`` the gate takes.

The gate resets BETWEEN its three re-runs, so an undo working four times in five
REJECTS the skill. Hence: a step aims by CONTENT (a notice at the top of a page moves
everything below it) through ``find_text(fuzzy=False)`` and never ``ElementIndex.best``,
which is a ranking with a winner even when nothing fits; a loop prefers ``until_seen``,
because ``until_gone`` is a negative test that also passes on the WRONG screen; and
``via="dom"`` exists because a real site's controls are icon-only with their name in an
``aria-label`` - see ``AGENTS.md`` for why scaffolding may read that and the agent may not.
"""

from __future__ import annotations

import json
import time
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
    "SETTLE_BUDGET_MS",
    "SETTLE_POLL_MS",
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
"""The ``TaskSpec.params`` key holding an action reset; its value is whatever
``reset_steps_from`` accepts."""


Oracle = Literal["screen", "dom"]
"""``"screen"`` is detection and OCR, all a desktop controller can offer; ``"dom"`` is the
caller-supplied ``GroundTruthSource``, the only way to reach a never-painted name."""


SETTLE_BUDGET_MS = 4000.0
"""A poll, not a sleep: it returns the instant the screen answers. An undo is often not a
navigation, so ``_settle``'s load event covers nothing - splitkb.com removes a cart line
by in-place fetch, and without this the loop re-read the unchanged cart and reported
``ResetDidNotConverge`` after twelve clicks. Four seconds only has to outlast one slow
request, since ``DEFAULT_MAX_ROUNDS`` already bounds the loop."""

SETTLE_POLL_MS = 120.0
"""A ``via="screen"`` poll costs a real observation, so this is not run flat out."""

DEFAULT_MAX_ROUNDS = 12
"""A bound, not a target: enough to exceed the largest mess a task can make, small enough
that a step which will NEVER converge says so in seconds. Raise it per step if needed."""


class ResetStepFailed(SkillWeaverError):
    """One step could not be performed; ``reset_world`` reads it as ``failed``. NOT a
    ``ResetRefused``, which means there is no reset here at all."""


class ResetDidNotConverge(ResetStepFailed):
    """A converging step ran out of rounds with its exit condition still true. Shrugging
    instead would hand the gate a dirty screen and let it blame the candidate skill."""


@dataclass(frozen=True, slots=True)
class ResetStep:
    """One step of an action reset: an action, optionally aimed and optionally looped.

    Attributes:
        find: Anchor text, via ``find_text(fuzzy=False)``; empty uses the action's own point.
        anchor_kind: Narrows the anchor to one kind. Named apart from the action dict's
            own ``kind``, which silently rewrote the action when the two collided.
        dx / dy: Pixels from the anchor's centre, so an unlabelled control (the trash icon
            on the anchored row) travels with the anchor. Requires ``find``.
        until_gone: Text meaning the world is still dirty; non-empty makes the step a loop.
        until_seen: Text meaning it is clean, and the STRONGER of the two - a negative test
            also passes on a screen an earlier step failed to leave. With both, both hold.
        optional: A ``find`` matching nothing succeeds. Rejected on a looping step, where
            a missing target is either the exit condition or the reason it cannot converge.

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
        return bool(self.until_gone or self.until_seen)

    @property
    def reads_screen(self) -> bool:
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
"""Split out before ``action_from_dict`` sees the dict: that function turns every mapping
value into a ``Point`` and every list into a tuple."""


def reset_step_from_dict(data: Mapping[str, Any]) -> ResetStep:
    """One ``ResetStep`` from an ``action_to_dict`` mapping plus any of ``_STEP_FIELDS``.
    A click aimed by ``find`` needs no ``point``; a placeholder is supplied and replaced."""
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


def reset_step_to_dict(step: ResetStep) -> dict[str, Any]:
    """The mapping ``reset_step_from_dict`` reads back. Only non-default fields are
    written: this goes back onto ``TaskSpec.params``, which the composer is shown as text."""
    data = action_to_dict(step.action)
    if step.find:
        data.pop("point", None)  # a placeholder once an anchor names the target
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
    """Parse an action reset: a JSON array as a string, a sequence of mappings, or a
    sequence of ``ResetStep``. Empty is a real answer and comes back as an empty tuple."""
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
    """A ``WorldReset`` that PERFORMS ``steps`` in order, raising ``ResetStepFailed``
    unless every step finished and every converging step saw its condition hold.

    Idempotent while every undoing step converges: on a clean world it reads once and acts
    never. ``perceiver`` is consulted only when a step aims or loops, since an observation
    is the most expensive thing here. ``truth=None`` makes a ``via="dom"`` step FAIL rather
    than fall back to pixels - silently changing which reading is trusted is the flaky undo
    this module rules out.
    """
    plan = tuple(steps)
    read = _Reader(controller, perceiver, truth)

    def reset() -> None:
        for position, step in enumerate(plan, 1):
            _run_step(step, position, controller, read)
        log.info("agent.world_reset.actions", steps=len(plan))

    return reset


def chain_resets(*resets: Any) -> Any:
    """Several resets in order as one, skipping ``None``. Returns ``None`` when all were
    ``None`` - "nothing can put this back", which must stay distinct from a reset that
    does nothing."""
    live = [r for r in resets if r is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def reset() -> None:
        for one in live:
            one()

    return reset


class _Reader:
    """Reads the screen the way a step asks to. Stateless: a converging loop's every round
    looks at a screen that just changed, so nothing may be cached."""

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
                # Did nothing, on the word of a NEGATIVE test: equally true of a clean
                # world and of a screen an earlier step never left. Only the reader can
                # tell those apart, so say so every time.
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
        _await_answer(step, position, read, seen)
    raise ResetDidNotConverge(
        f"reset step {position} ({step}) performed {step.max_rounds} actions and the "
        "world still does not look put back. Either the step does not undo what it is "
        "aimed at, or its condition names text that does not change with the state."
    )


def _await_answer(step: ResetStep, position: int, read: _Reader, before: ElementIndex) -> None:
    """Poll until the screen differs from ``before`` or the exit condition holds, else
    return at ``SETTLE_BUDGET_MS``. Timing out is not a failure - this only decides whether
    the loop looks at the answer or at the question - so a failed read is swallowed too."""
    deadline = time.monotonic() + SETTLE_BUDGET_MS / 1000.0
    start = _signature(before)
    previous = start
    changed = False
    while time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_MS / 1000.0)
        try:
            now = read(step, position)
        except SkillWeaverError:
            continue
        if _settled(now, step):
            return
        signature = _signature(now)
        # Changed AND stopped changing: a page answers in two frames (splitkb.com paints
        # "Your cart is empty" after the last line goes), and returning in between shows a
        # screen with nothing to click and nothing saying it is done.
        if changed and signature == previous:
            return
        changed = changed or signature != start
        previous = signature
    log.info("agent.world_reset.unanswered", step=position, waited_ms=SETTLE_BUDGET_MS)


def _signature(seen: ElementIndex) -> tuple[tuple[str, int, int], ...]:
    """Text and position, not a fingerprint: "did anything move or change wording", a
    deliberately lower bar. A spinner trips it, and that is fine - the exit condition
    decides whether the world is back."""
    return tuple((element.text, element.box.x, element.box.y) for element in seen.all())


def _settled(seen: ElementIndex, step: ResetStep) -> bool:
    """Whether ``step``'s exit condition holds; both clauses must, when both are given."""
    if step.until_gone and _says(seen, step.until_gone):
        return False
    return not (step.until_seen and not _says(seen, step.until_seen))


def _says(seen: ElementIndex, text: str) -> bool:
    """``fuzzy=False`` on purpose: a fuzzy match on an exit condition could stop the loop
    early on a screen that only nearly says the marker is gone."""
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
    """``action`` with its point replaced; ``ResetStep`` has already checked the kind."""
    if isinstance(action, Click | Move | Scroll):
        return replace(action, point=point)
    raise ResetStepFailed(  # pragma: no cover - ResetStep.__post_init__ rejects this
        f"{action.kind!r} carries no point to aim"
    )
