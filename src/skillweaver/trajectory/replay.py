"""Replay a recorded trajectory against a controller and report where it diverged.

Replay is how a trajectory earns its keep after the run that produced it: the
admission gate replays a candidate skill's trajectory to prove it still works, and
the evaluation harness replays a library run to see whether a site has moved.

Both want the same answer, and it is not a boolean - it is *which step went wrong,
what was expected there, and what happened instead*::

    report = replay(trajectory, controller, perceiver=perceiver)
    if not report.ok:
        first = report.first_divergence
        print(first.index, first.reason, first.expected, "->", first.observed)

What counts as a divergence, in the order it is checked per step:

1. **the action was refused** - the controller reported ``ok=False`` where the
   recording reported ``ok=True`` (or the reverse);
2. **the screen went somewhere else** - with a ``Perceiver``, the fingerprint after
   the action is less than ``min_similarity`` like the recorded one;
3. **the URL went somewhere else** - the fallback when there is no perceiver and
   both the controller and the recording have a URL.

With no perceiver and no URL (a bare desktop controller) only the first check runs,
and the report says so through :attr:`ReplayReport.compared`.

A **dry run** (``dry_run=True``) never touches the controller. It checks the
trajectory itself - contiguous step indices, actions that survive a serialization
round trip, action kinds this controller supports, points inside its viewport - so a
harness can reject a trajectory it could not replay without opening a browser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from skillweaver.contracts import (
    Action,
    Box,
    Click,
    Controller,
    Drag,
    Move,
    Perceiver,
    Point,
    Scroll,
    Trajectory,
    action_from_dict,
    action_to_dict,
)
from skillweaver.errors import ControllerError
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

DivergenceReason = Literal[
    "bad_index",
    "unreadable_action",
    "unsupported_action",
    "out_of_viewport",
    "action_result",
    "fingerprint",
    "url",
    "controller_error",
]
"""Why a step did not replay. The first four are found without acting (a dry run
finds exactly these); the rest need the controller."""

Compared = Literal["fingerprint", "url", "result_only"]
"""How far replay could check each step: the strongest comparison available."""


@dataclass(frozen=True, slots=True)
class Divergence:
    """One place a replay stopped matching its recording.

    Attributes:
        index: The ``TrajectoryStep.index`` that diverged.
        reason: Which check failed.
        expected: What the recording says, as a short human-readable string.
        observed: What happened instead.
        similarity: Fingerprint similarity in ``0.0..1.0`` for a ``fingerprint``
            divergence, ``None`` for every other reason.
        action: The action being replayed when this was found.
    """

    index: int
    reason: DivergenceReason
    expected: str
    observed: str
    similarity: float | None = None
    action: Action | None = None

    def __str__(self) -> str:
        detail = "" if self.similarity is None else f" (similarity {self.similarity:.2f})"
        return (
            f"step {self.index} {self.reason}: "
            f"expected {self.expected}, observed {self.observed}{detail}"
        )


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """The structured outcome of a replay.

    Attributes:
        run_id: The trajectory that was replayed.
        ok: True when nothing diverged.
        dry_run: Whether the controller was left untouched.
        steps_replayed: How many steps were actually performed (``0`` for a dry run).
        steps_total: How many steps the trajectory has.
        divergences: Every divergence found, in step order. With the default
            ``stop_on_divergence`` there is at most one.
        compared: The strongest per-step comparison replay was able to make.
        elapsed_ms: Wall-clock milliseconds the replay took.
    """

    run_id: str
    ok: bool
    dry_run: bool
    steps_replayed: int
    steps_total: int
    divergences: tuple[Divergence, ...]
    compared: Compared
    elapsed_ms: float

    @property
    def first_divergence(self) -> Divergence | None:
        """The earliest divergence, or ``None`` when the replay was clean."""
        return self.divergences[0] if self.divergences else None

    def __str__(self) -> str:
        head = (
            f"replay {self.run_id}: "
            f"{'dry run ' if self.dry_run else ''}"
            f"{self.steps_replayed}/{self.steps_total} steps, "
            f"compared by {self.compared}"
        )
        if self.ok:
            return head + ", clean"
        return head + ", diverged - " + "; ".join(str(d) for d in self.divergences)


def _points(action: Action) -> tuple[Point, ...]:
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


def validate(trajectory: Trajectory, controller: Controller) -> list[Divergence]:
    """Everything wrong with ``trajectory`` that can be found without acting.

    Checks that step indices run ``0, 1, 2, ...``, that every action survives a
    round trip through :func:`~skillweaver.contracts.action_to_dict`, that the
    controller supports each action's kind, and that every point lies inside the
    controller's viewport. Returns them in step order; empty means replayable.

    Steps the RECORDING already shows failing (``result.ok`` is false - a refused
    ``Navigate``, a click off the edge) are exempt from the support and viewport
    checks: a controller refusing them again reproduces the recording rather than
    diverging from it.
    """
    viewport = controller.viewport()
    bounds = Box(0, 0, viewport.w, viewport.h)
    found: list[Divergence] = []
    for position, step in enumerate(trajectory.steps):
        if step.index != position:
            found.append(
                Divergence(step.index, "bad_index", f"index {position}", f"index {step.index}")
            )
        action = step.action
        try:
            if action_from_dict(action_to_dict(action)) != action:
                raise ValueError("round trip changed the action")
        except ValueError as exc:
            found.append(
                Divergence(position, "unreadable_action", repr(action), str(exc), action=action)
            )
            continue
        if not step.result.ok:
            continue
        if not controller.supports(action.kind):
            found.append(
                Divergence(
                    position,
                    "unsupported_action",
                    f"a controller supporting {action.kind!r}",
                    controller.describe(),
                    action=action,
                )
            )
        for point in _points(action):
            if not bounds.contains(point):
                found.append(
                    Divergence(
                        position,
                        "out_of_viewport",
                        f"a point inside {bounds.w}x{bounds.h}",
                        f"({point.x}, {point.y})",
                        action=action,
                    )
                )
    return found


def replay(
    trajectory: Trajectory,
    controller: Controller,
    *,
    perceiver: Perceiver | None = None,
    dry_run: bool = False,
    min_similarity: float = 1.0,
    stop_on_divergence: bool = True,
) -> ReplayReport:
    """Re-issue ``trajectory``'s actions against ``controller`` and report the result.

    Args:
        trajectory: The recording to replay.
        controller: Where to replay it. Left untouched when ``dry_run``.
        perceiver: Used to observe the screen after each action so fingerprints can
            be compared. Without it, replay falls back to comparing URLs, and
            without those, to the action results alone.
        dry_run: Only validate the trajectory (see :func:`validate`); perform
            nothing.
        min_similarity: A step's fingerprint must be at least this like the recorded
            one. ``1.0`` demands the same screen; lower it to tolerate a clock, a
            cart count or an ad.
        stop_on_divergence: Stop at the first divergence (the default - the state is
            no longer the recorded one, so later steps would compare against
            nonsense). ``False`` replays every step and collects everything, which
            is what an offline forensic pass wants.

    The controller's coordinates are LOGICAL pixels, exactly as recorded; replay
    never rescales a point.

    Returns:
        A :class:`ReplayReport`. Divergence is reported, never raised - including a
        :class:`~skillweaver.errors.ControllerError`, which becomes a
        ``controller_error`` divergence so a harness replaying a thousand runs
        survives one broken browser.
    """
    started = time.perf_counter()
    divergences = validate(trajectory, controller)
    compared: Compared = (
        "fingerprint" if perceiver is not None else "url" if controller.url() else "result_only"
    )

    def report(replayed: int) -> ReplayReport:
        elapsed = (time.perf_counter() - started) * 1000.0
        result = ReplayReport(
            run_id=trajectory.run_id,
            ok=not divergences,
            dry_run=dry_run,
            steps_replayed=replayed,
            steps_total=len(trajectory.steps),
            divergences=tuple(divergences),
            compared=compared,
            elapsed_ms=elapsed,
        )
        log.info(
            "trajectory.replay",
            run_id=result.run_id,
            ok=result.ok,
            dry_run=dry_run,
            steps=f"{replayed}/{result.steps_total}",
            diverged_at=result.first_divergence.index if result.first_divergence else None,
        )
        return result

    if dry_run:
        return report(0)
    if divergences and stop_on_divergence:
        return report(0)  # unreplayable as written; do not touch the controller

    replayed = 0
    for step in trajectory.steps:
        try:
            result = controller.perform(step.action)
        except ControllerError as exc:
            divergences.append(
                Divergence(
                    step.index,
                    "controller_error",
                    "a usable controller",
                    str(exc),
                    action=step.action,
                )
            )
            return report(replayed)
        replayed += 1
        step_diverged = False
        if result.ok != step.result.ok:
            divergences.append(
                Divergence(
                    step.index,
                    "action_result",
                    f"ok={step.result.ok}",
                    f"ok={result.ok} ({result.error or 'no error given'})",
                    action=step.action,
                )
            )
            step_diverged = True
        elif perceiver is not None:
            observed = perceiver.observe(controller)
            similarity = observed.fingerprint.similarity(step.after.fingerprint)
            if similarity < min_similarity:
                divergences.append(
                    Divergence(
                        step.index,
                        "fingerprint",
                        step.after.fingerprint.value,
                        observed.fingerprint.value,
                        similarity=similarity,
                        action=step.action,
                    )
                )
                step_diverged = True
        else:
            observed_url = controller.url()
            if observed_url is not None and step.after.url is not None:
                if observed_url != step.after.url:
                    divergences.append(
                        Divergence(
                            step.index,
                            "url",
                            step.after.url,
                            observed_url,
                            action=step.action,
                        )
                    )
                    step_diverged = True
        if step_diverged and stop_on_divergence:
            break
    return report(replayed)
