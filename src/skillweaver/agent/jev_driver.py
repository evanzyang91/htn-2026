"""Driving the explorer with a Jev policy instead of a prompted model.

:class:`JevDriver` turns Jev's vocabulary (``CLICK`` on element index 7) into the
explorer's ANSWER OBJECT naming an element id. The join is free: ``DomControl.element_id``
is the same string the Element carries as ``stable_id``, and ElementCatalog files an
element under it when it is unique, which ``perception.dom._element_id`` guarantees - so
there is no second lookup table to drift. The answer still goes through
``Explorer._ground``, so the id is checked against the CURRENT screen.

``BACK`` takes no argument and skips the catalogue, but is a step like any other,
performed and judged, because a move the graph cannot see would make a second run no
cheaper than the first. Having no target is what puts it outside :func:`_exclusions`,
hence :data:`BACK_SIGNATURE`. ``TYPE_TEXT`` is three actions - focus, clear, type - and a
move is one action OR one code block, so it becomes the block.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from skillweaver.agent.explorer import (
    Attempt,
    ElementCatalog,
    PolicyBlocked,
    signature_move,
)
from skillweaver.contracts import Observation, TaskSpec, Usage
from skillweaver.errors import SkillWeaverError
from skillweaver.llm.jev_ import BrowserPolicy, PolicyDecision, targets_of
from skillweaver.logging_ import get_logger
from skillweaver.perception.dom import DomPerceiver, DomSnapshot

log = get_logger(__name__)

__all__ = [
    "BACK_SIGNATURE",
    "DEAD_END_OPERATIONS",
    "NO_CHANGE_LIMIT",
    "SCROLL_PIXELS",
    "SELECT_ALL_CHORD",
    "WAIT_MS",
    "JevDriver",
]

DEAD_END_OPERATIONS: Mapping[str, str] = {"click": "CLICK"}
"""Which Jev operation a failed move's signature kind is evidence against.

Read this before adding to it. A ``Move.signature`` is ``<kind>:<element_id>:...`` and a
``CLICK`` is the only shape this driver produces that NAMES an element: ``TYPE_TEXT``
grounds to a CODE BLOCK, and ``SCROLL_UP``/``SCROLL_DOWN``, ``WAIT`` and ``BACK`` name
none either. Mapping ``scroll`` or ``drag`` in here would let a move the policy never
made withhold a target it never tried, and mapping a failed click onto ``TYPE_TEXT``
would take away the search box on the one screen whose task is to type in it.

A TARGETLESS operation cannot be withheld this way at all, so the one that needs
withholding carries its own constant; see :data:`BACK_SIGNATURE`.
"""

BACK_SIGNATURE = "back"
"""The whole move signature ``Explorer._resolve`` gives a ``back``.

``BACK`` is the only operation this driver produces with no target, which puts it out of
:func:`_exclusions`' reach twice over, so withholding it means withholding the
OPERATION: ``DomSnapshot.can_go_back`` turned off for that one ask.

Measured on live en.wikipedia.org, the cost of not doing it: one back the critic degraded
to "I do not know", then EIGHT more chosen at falling confidence and refused one after
another until the run gave up at ``BLOCKED`` - 9 provider calls on a move that could not
be performed.
"""

NO_CHANGE_LIMIT = 4
"""How many moves in a row one address may absorb without changing before the run stops.

Upstream's bound, and its reason for the address: a navigation that commits after the
observation window records a false "no change", so differing URLs across the streak are
proof of progress even when each step looked idle. ``WAIT`` is not counted - time passing
IS the point of one. Nothing else here bounds a single screen: all four ``Budget`` limits
are global, so without this a page that repaints slightly on every failed click can absorb
the whole run and be reported as ``budget`` rather than as the dead end it was.
"""

SCROLL_PIXELS = 560
"""Logical pixels one ``SCROLL_DOWN`` or ``SCROLL_UP`` moves, as upstream Jev scrolls.
Deliberately less than a viewport (800 here), so two consecutive scrolls overlap and a
control on the fold is fully visible on one rather than cut in half on both."""

WAIT_MS = 500
"""How long one ``WAIT`` pauses.

A real pause, not a reflex one: the policy saying the control it needs is not on screen
YET, which no load event covers. Not the kind ``strip_reflex_waits`` removes, though a
learned skill that keeps one will be told to justify it by the same gate."""

SELECT_ALL_CHORD: tuple[str, str] = ("Meta", "a") if sys.platform == "darwin" else ("Control", "a")
"""The chord that selects a field's contents before it is retyped.

Platform-dependent because the platform is, and upstream Jev makes the same split. A
CONSTANT rather than an inline literal because it is baked into skill code the
synthesizer will store: a skill learned on a Mac and replayed on Linux would otherwise
select nothing, type after what is already there, and search for ``keycapskeycaps``.
"""


class JevDriver:
    """An ActingPolicy over a :class:`~skillweaver.llm.jev_.BrowserPolicy`.

    Args:
        policy: Who chooses. Usually :class:`~skillweaver.llm.jev_.JevPolicy`.
        perceiver: The DOM perceiver driving the same run. The driver reads ``.last``
            from it, because the roles, values and states a policy needs are not on
            Element and putting them there would change the shared surface.

    Raises:
        SkillWeaverError: from :meth:`propose` if the perceiver has no snapshot for the
            observation asked about, which means the two were not driving the same run.
    """

    __slots__ = ("_perceiver", "_policy", "_steps", "_taken")

    def __init__(self, policy: BrowserPolicy, perceiver: DomPerceiver) -> None:
        self._policy = policy
        self._perceiver = perceiver
        self._steps: list[dict[str, Any]] = []
        self._taken: dict[str, set[str]] = {}

    def __repr__(self) -> str:
        return f"JevDriver({self._policy.name()})"

    def name(self) -> str:
        """The policy model identifier."""
        return self._policy.name()

    def total_usage(self) -> Usage:
        """What the policy has spent, so the explorer can charge it. See
        :meth:`skillweaver.llm.jev_.JevPolicy.total_usage` for what is NOT in it."""
        return self._policy.total_usage()

    def propose(
        self,
        task: TaskSpec,
        observation: Observation,
        catalog: ElementCatalog,
        history: Sequence[str],
        dead_ends: Sequence[Attempt],
        rejection: str | None,
    ) -> str:
        """Ask the policy, and write its choice as an explorer answer object.

        Raises:
            PolicyBlocked: on the policy's own ``BLOCKED``.
            SkillWeaverError: if the snapshot and the observation disagree, or if the
                policy chose a control the catalogue does not hold - which would mean the
                two were built from different frames.
            ProviderError: from the policy, on a provider failure.
        """
        snapshot = self._snapshot_for(observation)
        if not history and self._steps:
            # No moves yet and steps on file: the explorer has begun another run.
            self._steps, self._taken = [], {}
        state = _content_digest(snapshot)
        self._settle(snapshot, state, rejection)
        self._stop_if_inert(observation)
        exclude, restored = _exclusions(snapshot, dead_ends, self._spent(state))
        asked = snapshot
        if snapshot.can_go_back and any(a.signature == BACK_SIGNATURE for a in dead_ends):
            asked = dataclasses.replace(snapshot, can_go_back=False)
        decision = self._policy.decide(task.text, asked, self._steps, exclude)
        self._remember(decision, state)
        log.info(
            "jev.decide",
            operation=decision.operation,
            changed=self._steps[-2].get("page_changed") if len(self._steps) > 1 else None,
            confidence=round(decision.confidence, 3),
            probability=round(decision.probability, 3),
            policy_ms=round(decision.policy_ms),
            latency_ms=round(decision.latency_ms),
            offered=len(snapshot.controls),
            withheld=sum(len(ids) for ids in exclude.values()),
            restored=sorted(restored) or None,
        )
        if decision.operation == "BLOCKED":
            raise PolicyBlocked(
                f"the policy found no supported operation on {observation.url or 'this screen'}"
            )
        return json.dumps(_answer(decision, catalog, snapshot, _note(exclude, restored)))

    # -- odds and ends -----------------------------------------------------------------

    def _snapshot_for(self, observation: Observation) -> DomSnapshot:
        """The DOM snapshot behind ``observation``.

        Checked rather than assumed: the perceiver holds ONE slot, and a caller that
        observed twice between asking would hand the policy one screen's controls and the
        explorer another's. Every way to act on the wrong element, so it fails loudly.

        Raises:
            SkillWeaverError: when the two do not describe the same frame.
        """
        snapshot = self._perceiver.last
        if snapshot is None:
            raise SkillWeaverError(
                "the Jev driver has no page snapshot: it must be given the same "
                "DomPerceiver that produced the observation it is asked about"
            )
        if snapshot.url and observation.url and snapshot.url != observation.url:
            raise SkillWeaverError(
                f"the page snapshot is of {snapshot.url} and the observation is of "
                f"{observation.url}; they are not the same frame"
            )
        return snapshot

    def _settle(self, snapshot: DomSnapshot, state: str, rejection: str | None) -> None:
        """Close the previous step with what LITERALLY became of the page, before asking.

        ``page_changed`` is upstream's field and upstream's meaning: did the page's
        content differ afterwards (:func:`_content_digest`). It was the CRITIC's verdict
        here until 2026-09-20, and the two are different questions. Measured live on an
        option dialog: the policy scrolled the list twice, ticked the right add-on and
        added it to the order - the task, done - while ``state_changed`` called both
        scrolls and the tick failures (0.925, 0.767, and "the fingerprints are identical"
        for a ticked checkbox). So the policy was told three times that nothing happened,
        and under rules that make the history "the authoritative record of progress" it
        answered a finished task with ``BLOCKED`` 0.57 against ``DONE`` 0.23.
        ``AGENTS.md`` already says nothing may be keyed on that per-move verdict.

        The verdict is not lost: a move it failed is still a dead end in the explorer's
        memory and still withheld by :func:`_exclusions`.
        """
        if not self._steps:
            return
        last = self._steps[-1]
        last["page_changed"] = state != last["state"]
        last["url"] = snapshot.url
        if rejection:
            last["refused"] = rejection
        elif last["kind"] == "CLICK" and last["element_id"]:
            self._taken.setdefault(last["state"], set()).add(last["element_id"])

    def _remember(self, decision: PolicyDecision, state: str) -> None:
        """Open a step with what the policy asked for. :meth:`_settle` closes it.

        ``state``, ``element_id`` and later ``url`` are this driver's own bookkeeping;
        the request builder names the keys it sends, so they never reach the wire.
        """
        self._steps.append(
            {
                "action": decision.why or decision.operation,
                "kind": decision.operation,
                "text": decision.text,
                "state": state,
                "element_id": decision.element_id,
            }
        )

    def _spent(self, state: str) -> dict[str, set[str]]:
        """``{operation: element ids}`` this exact page state has already used up.

        Two of upstream's rules, both proofs BY IDENTITY and so both keyed on the exact
        :func:`_content_digest` rather than on fingerprint similarity. A click already
        made from this state, when the run is standing on the state again, did not
        advance the goal however much it changed the page - that is what turns
        open/close into an oscillation. And whatever has changed nothing since the page
        last changed will change nothing now; a ``WAIT`` ends that streak, because time
        passing is a reason a dead control may work.

        Exact on purpose, and it is the opposite call from
        :meth:`~skillweaver.agent.explorer.FailureMemory.near`. At the 0.26 same-state
        cut a dialog with one more box ticked IS the same screen, so similarity here
        would withhold a legitimate second ``Add``. On a page too noisy to digest
        identically twice this withholds nothing: it fails open, similarity fails closed.
        """
        spent: dict[str, set[str]] = {}
        if state in self._taken:
            spent["CLICK"] = set(self._taken[state])
        for step in reversed(self._steps):
            if step.get("page_changed") is not False or step["kind"] == "WAIT":
                break
            if step["state"] != state:
                break
            if step["kind"] in ("CLICK", "TYPE_TEXT") and step["element_id"]:
                spent.setdefault(step["kind"], set()).add(step["element_id"])
        return spent

    def _stop_if_inert(self, observation: Observation) -> None:
        """Give up on a page that has absorbed :data:`NO_CHANGE_LIMIT` moves unchanged.

        Raises:
            PolicyBlocked: which the explorer already turns into a diagnosed stop at this
                screen, after its one retry without a borrowed workflow.
        """
        recent = self._steps[-NO_CHANGE_LIMIT:]
        if len(recent) < NO_CHANGE_LIMIT:
            return
        inert = all(s.get("page_changed") is False and s["kind"] != "WAIT" for s in recent)
        if inert and len({s.get("url") for s in recent}) == 1:
            tried = "; ".join(str(s["action"]) for s in recent)
            raise PolicyBlocked(
                f"{observation.url or 'this screen'} did not change across the last "
                f"{NO_CHANGE_LIMIT} moves ({tried}), so nothing more is spent on it"
            )


# --------------------------------------------------------------------------------------
# Jev's vocabulary, as an explorer answer
# --------------------------------------------------------------------------------------


def _answer(
    decision: PolicyDecision, catalog: ElementCatalog, snapshot: DomSnapshot, note: str = ""
) -> dict[str, Any]:
    """One decision as the JSON object ``Explorer._ground`` parses.

    ``note`` is :func:`_note`'s sentence about targets this screen withheld. ``expect`` is
    filled from what the operation is FOR rather than left empty, because a critic with
    nothing to check against falls back to "did anything change", a weaker question.

    Raises:
        SkillWeaverError: if the chosen element id is not in the catalogue. That cannot
            happen while the snapshot and the observation are the same frame, so if it
            does, the frames diverged and performing the move would be performing it blind.
    """
    operation = decision.operation
    odds = _odds(decision) + note
    if operation == "DONE":
        return {
            "thought": (f"the policy reports every requirement is visibly satisfied {odds}"),
            "expect": "",
            "done": True,
        }
    if operation == "WAIT":
        return {
            "thought": f"the policy is waiting for the page to update {odds}",
            "expect": "the page finishes loading and the control it needs appears",
            "done": False,
            "action": {"kind": "wait", "ms": WAIT_MS},
        }
    if operation == "BACK":
        return {
            "thought": f"the policy is going back to the previous page {odds}",
            "expect": "the page before this one is showing again",
            "done": False,
            "action": {"kind": "back"},
        }
    if operation in ("SCROLL_DOWN", "SCROLL_UP"):
        down = operation == "SCROLL_DOWN"
        return {
            "thought": f"the policy is scrolling {'down' if down else 'up'} {odds}",
            "expect": f"content from {'below' if down else 'above'} the fold comes into view",
            "done": False,
            "action": {"kind": "scroll", "dy": SCROLL_PIXELS if down else -SCROLL_PIXELS},
        }

    element_id = decision.element_id
    if element_id is None or element_id not in catalog:
        raise SkillWeaverError(
            f"the policy chose {element_id!r} for {operation}, which is not on the screen "
            "the explorer is holding; the snapshot and the observation have diverged"
        )
    control = snapshot.by_element_id.get(element_id)
    label = control.label if control is not None else element_id

    if operation == "CLICK":
        return {
            "thought": f"the policy is clicking {label!r} {odds}",
            "expect": f"the page responds to {label!r}",
            "done": False,
            "action": {"kind": "click", "element_id": element_id},
        }
    if operation == "TYPE_TEXT":
        text = decision.text
        if not text:
            raise SkillWeaverError(
                f"the policy chose TYPE_TEXT into {label!r} but no value was written; "
                "nothing is guessed, so the move is refused"
            )
        return {
            "thought": f"the policy is typing into {label!r} {odds}",
            "expect": f"the field {label!r} now contains {text!r}",
            "done": False,
            "code": _type_into(element_id, text),
        }
    raise SkillWeaverError(
        f"the policy returned an operation this driver cannot perform: {operation!r}"
    )


def _type_into(element_id: str, text: str) -> str:
    """The three lines that replace a field's contents. See the module docstring.

    ``json.dumps`` rather than ``repr``: the value came off a web page and out of a model,
    and this string becomes source the sandbox compiles and the synthesizer may store, so
    a label containing an apostrophe must not end the string early.
    """
    chord = ", ".join(json.dumps(key) for key in SELECT_ALL_CHORD)
    return (
        f"ctx.ctl.click(el[{json.dumps(element_id)}])\n"
        f"ctx.ctl.press({chord})\n"
        f"ctx.ctl.type_text({json.dumps(text)})\n"
    )


def _content_digest(snapshot: DomSnapshot) -> str:
    """Whether the page is LITERALLY the page it was: upstream's ``fingerprint``, here.

    Address, visible text, every control with its value and state, and where things are
    scrolled to. A change DETECTOR and an exact one, which is a different job from
    identifying a screen: that stays ``StateFingerprinter`` and its calibrated similarity,
    and nothing stored is ever keyed on this. It sees what the pixel identity cannot - a
    ticked box, a list scrolled inside a dialog, a cart badge going from 0 to 1 - because
    those are the changes a policy's history has to be honest about.
    """
    scroller = snapshot.scroller
    content = [
        snapshot.url,
        snapshot.text,
        [(c.element_id, c.value, c.checked, c.selected, c.expanded) for c in snapshot.controls],
        snapshot.scroll_y,
        None if scroller is None else (scroller.can_down, scroller.can_up),
    ]
    return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()[:16]


def _exclusions(
    snapshot: DomSnapshot,
    dead_ends: Sequence[Attempt],
    spent: Mapping[str, Collection[str]] | None = None,
) -> tuple[dict[str, set[str]], set[str]]:
    """``({operation: element ids to withhold}, {operations put back})``.

    A target already known to fail ON THIS SCREEN is not offered again. The explorer would
    refuse the repeat anyway, so this changes not what can happen but what it COSTS - one
    wasted provider call per repeat, down to none. It is the one thing this policy shape
    can do and a prompted model cannot.

    Two rules that pull against each other. A failure is a MOVE at an element, not an
    element, so :data:`DEAD_END_OPERATIONS` maps only the signature kinds this driver can
    produce; a version that dropped the control outright took the search box away from a
    run whose whole task was to type in it. And an exclusion that empties an operation's
    target head is PUT BACK and said out loud by :func:`_note`: a policy offered only
    ``WAIT``, ``DONE`` and ``BLOCKED`` answers one of them, and a false ``BLOCKED`` is a
    silent failure. Measured on the sandbox's Mail screen: 17 controls, 0 offered.

    ``spent`` is :meth:`JevDriver._spent` - what this exact page state has used up by
    upstream's two identity rules - and rides the same put-back as everything else.
    """
    wanted: dict[str, set[str]] = {op: set(ids) for op, ids in (spent or {}).items() if ids}
    for attempt in dead_ends:
        move = signature_move(attempt.signature)
        if move is None:
            continue
        operation = DEAD_END_OPERATIONS.get(move[0])
        if operation is not None:
            wanted.setdefault(operation, set()).add(move[1])
    if not wanted:
        return {}, set()
    full, kept = targets_of(snapshot), targets_of(snapshot, wanted)
    restored = {op for op in wanted if op in full and op not in kept}
    return {op: ids for op, ids in wanted.items() if op not in restored}, restored


def _note(exclude: Mapping[str, Collection[str]], restored: Collection[str]) -> str:
    """What the trajectory is told about targets withheld here, or ``""``.

    It goes on the move's ``thought``, the line ``Explorer._write_down`` records against
    the step, so a run where the policy chose from less than the whole screen says so
    where anyone reading the run will see it.
    """
    if restored:
        return (
            f"; every {'/'.join(sorted(restored)).lower()} target on this screen has already "
            "failed here, so they are offered again rather than reporting blocked"
        )
    withheld = sum(len(ids) for ids in exclude.values())
    if not withheld:
        return ""
    plural = "" if withheld == 1 else "s"
    return (
        f"; {withheld} target{plural} already used up on this screen "
        f"{'was' if withheld == 1 else 'were'} withheld"
    )


def _odds(decision: PolicyDecision) -> str:
    """The confidence and probability of one decision, as a short parenthesis. Carried
    into the trajectory on purpose: a move chosen at 0.31 and one chosen at 0.99 read
    identically once they are a click."""
    return f"(confidence {decision.confidence:.2f}, p {decision.probability:.2f})"
