"""``describe_trajectory``: a recorded run as the text a model writes a skill from.

The only door a recording reaches the synthesizer through, so whatever it leaves out the
skill is written without, and it has to render a MESSY run honestly - a messy run is the
normal one.

A move is one DECISION, but one trajectory step is one ACTION, and the explorer writes the
move's thought and the critic's verdict onto its LAST step. Read step by step that says
the agent decided to press Enter because it wanted to search, and left the click before it
unexplained - so :func:`moves_of` regroups steps into moves, closing a group at every step
carrying a verdict, which is exactly where the explorer ends one.

A rejected move is one that fell short of what it CLAIMED (the critic judges against
``move.expect``), not one whose actions were wasted: measured on the fake app with a live
vision critic, a move that typed the search query and said it would open the invoice was
correctly rejected and its action was still required. So a rejection is MARKED as suspect
and never dropped.

Does the marking change the skills? Not on the runs tried: 2026-09-19, 26 live generations
over three recordings, half rendered each way, produced the same procedure every time - the
model was already ignoring the fumbling. The rendering is fixed because it was WRONG about
what the recording says; no quality improvement was observed and none should be reported.
What the measurement does establish is that the marking costs nothing.

Deterministic for a given trajectory - no timestamps, no dict ordering - which is what
makes a cassette replay and a repair prompt diffable.
"""

from __future__ import annotations

from collections.abc import Sequence

from skillweaver.contracts import (
    Action,
    Element,
    Observation,
    Trajectory,
    TrajectoryStep,
    Verdict,
)
from skillweaver.skills.api import describe_action

__all__ = ["Move", "describe_trajectory", "moves_of"]

MAX_ELEMENTS = 18
"""Elements described per recorded screen. Enough to write a lookup against, short
enough that a long list does not bury the ones that were acted on."""

MAX_TEXT = 80
"""Longest element text quoted, in characters; longer is truncated with an ellipsis."""

MAX_REASON = 220
"""Longest critic reason quoted per move. A vision critic's reason carries its whole
account of the screen - live verdicts run 400 to 600 characters - and the model is being
told WHETHER the move worked, not asked to re-read a screen it already has."""


def _truncate(text: str, limit: int) -> str:
    flat = text.strip().replace("\n", " ")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _describe_element(element: Element) -> str:
    text = _truncate(element.text, MAX_TEXT)
    box = element.box
    return f"  - {element.kind.value} {text!r} at ({box.x}, {box.y}) {box.w}x{box.h}"


def _describe_screen(observation: Observation, label: str) -> str:
    elements: Sequence[Element] = observation.elements[:MAX_ELEMENTS]
    url = observation.url or "none"
    lines = [f"{label} (url: {url}, screen id: {observation.fingerprint.value})"]
    lines.extend(_describe_element(e) for e in elements)
    if len(observation.elements) > MAX_ELEMENTS:
        lines.append(f"  - ... and {len(observation.elements) - MAX_ELEMENTS} more elements")
    return "\n".join(lines)


def _describe_target(observation: Observation, action: Action) -> str | None:
    """Which element the action landed on, named the way a skill can name it again.

    The recording says ``click (236, 121)`` and a skill may not write a coordinate down, so
    without this the model guesses WHICH element that was. It guessed wrong in a live run:
    it wanted a row "from Dana Whitfield", searched for that text, found none - OCR never
    reads the sender names on that page - and archived a different message. So the element
    is named twice over: by its text where there is any, and always by its kind and place
    in reading order, which a skill CAN reproduce from pixels when text fails. ``None`` for
    an action with no point.
    """
    point = getattr(action, "point", None)
    if point is None:
        return None
    hits = [e for e in observation.elements if e.box.contains(point)]
    if not hits:
        return None
    element = min(hits, key=lambda e: e.box.area)
    same_kind = [e for e in observation.elements if e.kind is element.kind]
    ordinal = same_kind.index(element) + 1
    text = _truncate(element.text, MAX_TEXT)
    kind = element.kind.value
    reads = f"reading {text!r}" if text else "with NO readable text"
    return (
        f"  this landed on the {kind} {reads}, which is {kind} number {ordinal} "
        f"of {len(same_kind)} in reading order"
    )


class Move:
    """The steps of one move, with the verdict and the reason that belong to it.

    Attributes:
        index: Position of this move in the run, counting from ``1``.
        steps: Its steps, in order. Never empty.
        reason: What the agent said it was doing, from the deciding step's note.
        asides: Notes carried by the move's other steps, deduplicated and in order.
    """

    __slots__ = ("asides", "index", "reason", "steps")

    def __init__(self, index: int, steps: Sequence[TrajectoryStep]) -> None:
        self.index = index
        self.steps = tuple(steps)
        deciding = self.steps[-1]
        self.reason = deciding.note.strip()
        seen: list[str] = []
        for step in self.steps[:-1]:
            note = step.note.strip()
            if note and note != self.reason and note not in seen:
                seen.append(note)
        self.asides = tuple(seen)

    @property
    def verdict(self) -> Verdict | None:
        """The critic's judgment of this move, or ``None`` if it was never judged."""
        return self.steps[-1].verdict

    @property
    def rejected(self) -> bool:
        """Whether a critic looked at this move and said it did not work."""
        verdict = self.verdict
        return verdict is not None and not verdict.ok

    @property
    def refused(self) -> tuple[TrajectoryStep, ...]:
        """The steps the controller could not perform at all."""
        return tuple(step for step in self.steps if not step.result.ok)


def moves_of(trajectory: Trajectory) -> tuple[Move, ...]:
    """Regroup a run's steps into the moves they were performed as.

    A step carrying a verdict is the LAST step of its move, so no note has to be parsed to
    find the boundary. Steps after the last verdict form a final unjudged move: a run cut
    off mid-move still recorded what it did.

    A recording with NO verdicts anywhere - an older file, a hand-built one - has no
    boundaries to read, so every step is its own move, which is the shape this rendering
    had before moves existed in it.
    """
    steps = trajectory.steps
    if not any(step.verdict is not None for step in steps):
        return tuple(Move(i + 1, [step]) for i, step in enumerate(steps))
    moves: list[Move] = []
    pending: list[TrajectoryStep] = []
    for step in steps:
        pending.append(step)
        if step.verdict is not None:
            moves.append(Move(len(moves) + 1, pending))
            pending = []
    if pending:
        moves.append(Move(len(moves) + 1, pending))
    return tuple(moves)


def _headline(trajectory: Trajectory, moves: Sequence[Move]) -> str:
    """The one line that says whether this run was clean, before any of it is read."""
    rejected = [move.index for move in moves if move.rejected]
    steps = f"{len(trajectory.steps)} action(s) in {len(moves)} move(s)"
    if not rejected:
        return f"STEPS: {steps}"
    listed = ", ".join(str(index) for index in rejected)
    return (
        f"STEPS: {steps}, of which the critic REJECTED {len(rejected)} "
        f"(move(s) {listed}). This run FUMBLED: it did things that did not work and "
        "recovered. Each rejected move is marked where it happened - read what the "
        "critic said about it before deciding whether it belongs in the skill."
    )


def _describe_move(move: Move) -> list[str]:
    """One move: what it was for, how it was judged, then the actions themselves."""
    verdict = move.verdict
    if verdict is None:
        head = f"MOVE {move.index} (not judged)"
    else:
        judgment = "ACCEPTED" if verdict.ok else "REJECTED"
        said = _truncate(verdict.reason, MAX_REASON)
        head = f"MOVE {move.index} - the critic {judgment} it: {said}"
    lines = [head]
    if move.reason:
        lines.append(f"  the agent's reason, given BEFORE the action(s) below: {move.reason}")
    for aside in move.asides:
        lines.append(f"  also recorded against this move: {aside}")
    if move.rejected:
        lines.append(
            "  THE CRITIC SAID NO to this move: it did not do what the agent expected, "
            "and the run carried on and recovered. Reproduce its action(s) only if the "
            "task genuinely needs them - a rejected move is where a run wasted time."
        )
    for step in move.refused:
        error = step.result.error or "no reason given"
        lines.append(f"  the controller REFUSED step {step.index}: {error}")
    return lines


def describe_trajectory(trajectory: Trajectory) -> str:
    """The recorded run as the text the model is asked to write a skill from."""
    moves = moves_of(trajectory)
    parts = [
        f"TASK: {trajectory.task}",
        f"DOMAIN: {trajectory.domain}",
        _headline(trajectory, moves),
        "",
    ]
    if trajectory.steps:
        parts.append(_describe_screen(trajectory.steps[0].before, "STARTING SCREEN"))
        parts.append("")
    for move in moves:
        parts.extend(_describe_move(move))
        for step in move.steps:
            parts.append(f"STEP {step.index}: {describe_action(step.action)}")
            target = _describe_target(step.before, step.action)
            if target:
                parts.append(target)
            parts.append(_describe_screen(step.after, "  screen after"))
        parts.append("")
    if trajectory.steps:
        label = (
            "FINAL SCREEN (the goal)"
            if trajectory.ok
            else "FINAL SCREEN (where the run STOPPED - it did not report success)"
        )
        parts.append(_describe_screen(trajectory.steps[-1].after, label))
    parts.append("")
    parts.append("Write the skill for this task as the JSON object described above.")
    return "\n".join(parts)
