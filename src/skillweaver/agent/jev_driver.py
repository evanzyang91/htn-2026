"""Driving the explorer with a Jev policy instead of a prompted model.

One class, :class:`JevDriver`, which is an
:class:`~skillweaver.agent.explorer.ActingPolicy`. It sits between two things that do not
know about each other:

* :class:`~skillweaver.llm.jev_.JevPolicy`, which answers in Jev's vocabulary -
  ``CLICK`` on element index 7, ``TYPE_TEXT`` into index 3;
* :class:`~skillweaver.agent.explorer.Explorer`, which takes an ANSWER OBJECT naming an
  element id from the catalogue of the screen in front of it.

The join between them is free, and deliberately so:
:attr:`~skillweaver.perception.dom.DomControl.element_id` is the same string the
control's :class:`~skillweaver.contracts.Element` carries as ``stable_id``, and
:class:`~skillweaver.agent.explorer.ElementCatalog` files an element under its
``stable_id`` when that is unique on the screen -
:func:`~skillweaver.perception.dom._element_id` makes sure it always is. So there is no
second lookup table to drift out of step with the first.

Why the explorer still grounds it
---------------------------------

The answer this driver writes goes through
:func:`~skillweaver.agent.explorer._resolve` exactly as a model's would, which means the
id is checked against the current screen before anything is clicked. That is not
ceremony. The observation the policy was shown and the screen the click lands on are two
moments, and on a real page they are not always the same moment; upstream Jev guards that
with DOM node identity and a re-check immediately before input, and this project guards it
by re-observing after every action and refusing an id that is no longer there.

BACK is one action and has no target
------------------------------------

``BACK`` grounds to :class:`~skillweaver.contracts.Back`, which takes no argument, so it
skips the catalogue entirely - there is no element id to check against the screen. It is
still a step like any other: the explorer performs it, re-observes, records the graph
edge and asks the critic exactly as it does for a click. That is the point of grounding
it as an ACTION rather than as a code block; a move the graph cannot see would make a
second run no cheaper than the first. Having no target is also what puts it outside
:func:`_exclusions`, which is why :data:`BACK_SIGNATURE` exists.


TYPE_TEXT is three actions, so it is a code block
-------------------------------------------------

Typing into a field means focusing it, clearing what is in it and typing - and
:class:`~skillweaver.contracts.TypeText` types into whatever has focus and does not click
first, by contract. A move is one action OR one code block, so ``TYPE_TEXT`` becomes the
block, run in the same sandbox stored skills run in. That is also the shape the
synthesizer wants: a learned skill for "search for X" comes out as the same three lines.

Select-all is :data:`SELECT_ALL_CHORD` and the reason it is a constant is written there.
"""

from __future__ import annotations

import dataclasses
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
    "SCROLL_PIXELS",
    "SELECT_ALL_CHORD",
    "WAIT_MS",
    "JevDriver",
]

DEAD_END_OPERATIONS: Mapping[str, str] = {"click": "CLICK"}
"""Which Jev operation a failed move's signature kind is evidence against.

Read this before adding to it. A :attr:`~skillweaver.agent.explorer.Move.signature` is
``<kind>:<element_id>:...`` and a ``CLICK`` is the only shape this driver produces that
NAMES an element: a ``TYPE_TEXT`` grounds to a CODE BLOCK, whose signature names none
(:func:`~skillweaver.agent.explorer.signature_move` returns ``None`` for it), and
``SCROLL_UP``/``SCROLL_DOWN``, ``WAIT`` and ``BACK`` name none either.

So a click is the only dead end this path can record, and ``CLICK`` is the only thing it
is evidence against. Mapping ``scroll`` or ``drag`` in here would let a move the policy
never made withhold a target it never tried, and mapping a control's failed click onto
``TYPE_TEXT`` would take away the search box on the one screen whose task is to type in
it.

A TARGETLESS operation cannot be withheld this way at all - there is no id to leave out
of a head it does not have - so the one operation that needs withholding carries its own
constant; see :data:`BACK_SIGNATURE`.
"""

BACK_SIGNATURE = "back"
"""The whole move signature :func:`~skillweaver.agent.explorer._resolve` gives a ``back``.

``BACK`` is the only operation this driver produces with no target, which puts it out of
:func:`_exclusions`' reach twice over: ``signature_move`` answers ``None`` for a
signature that names no element, and there would be no target head to leave an id out of
anyway. Withholding it therefore means withholding the OPERATION, which is
:attr:`~skillweaver.perception.dom.DomSnapshot.can_go_back` turned off for that one ask.

Measured on live en.wikipedia.org, the cost of not doing it: one back the critic degraded
to "I do not know", then EIGHT more chosen by the policy at falling confidence and refused
by the explorer one after another until the run gave up at ``BLOCKED`` - 9 provider calls
spent on a move that could not be performed. Same rule as the exclusions above, applied
where they cannot reach.
"""

SCROLL_PIXELS = 560
"""Logical pixels one ``SCROLL_DOWN`` or ``SCROLL_UP`` moves, as upstream Jev scrolls.

Deliberately less than a viewport (800 by default here), so two consecutive scrolls
overlap and a control sitting on the fold is fully visible on one of them rather than
cut in half on both.
"""

WAIT_MS = 500
"""How long one ``WAIT`` pauses.

A real pause, not a reflex one: this is the policy saying the control it needs is not on
the screen YET, which no load event covers. It is not the kind
:func:`~skillweaver.skills.refactor.strip_reflex_waits` removes - that one removes a
sleep sitting next to an action the controller has already settled - but a learned skill
that keeps one will be told to justify it by the same gate, which is correct.
"""

SELECT_ALL_CHORD: tuple[str, str] = ("Meta", "a") if sys.platform == "darwin" else ("Control", "a")
"""The chord that selects a field's contents before it is retyped.

Platform-dependent because the platform is: macOS Chrome takes Meta+A and everything else
takes Control+A, and upstream Jev makes the same split on ``sys.platform``. It is a
CONSTANT rather than an inline literal because it is baked into skill code the synthesizer
will store - a skill learned on a Mac and replayed on Linux would otherwise select
nothing, type after what is already there, and search for ``keycapskeycaps``.
"""


class JevDriver:
    """An :class:`~skillweaver.agent.explorer.ActingPolicy` over a
    :class:`~skillweaver.llm.jev_.BrowserPolicy`.

    Args:
        policy: Who chooses. Usually :class:`~skillweaver.llm.jev_.JevPolicy`.
        perceiver: The DOM perceiver driving the same run. The driver reads
            :attr:`~skillweaver.perception.dom.DomPerceiver.last` from it, because the
            roles, values and states a policy needs are not on
            :class:`~skillweaver.contracts.Element` and putting them there would be a
            change to the shared surface.

    Raises:
        SkillWeaverError: from :meth:`propose` if the perceiver has no snapshot for the
            observation it is being asked about, which means the two were not driving
            the same run.
    """

    __slots__ = ("_perceiver", "_policy", "_steps")

    def __init__(self, policy: BrowserPolicy, perceiver: DomPerceiver) -> None:
        self._policy = policy
        self._perceiver = perceiver
        self._steps: list[dict[str, Any]] = []

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
        exclude, restored = _exclusions(snapshot, dead_ends)
        asked = snapshot
        if snapshot.can_go_back and any(a.signature == BACK_SIGNATURE for a in dead_ends):
            asked = dataclasses.replace(snapshot, can_go_back=False)
        self._settle(history, rejection)
        decision = self._policy.decide(task.text, asked, self._steps, exclude)
        self._remember(decision)
        log.info(
            "jev.decide",
            operation=decision.operation,
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
        observed twice between asking would be handing the policy one screen's controls
        and the explorer another's. There is no way to recover from that and every way
        to act on the wrong element, so it fails loudly.

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

    def _settle(self, history: Sequence[str], rejection: str | None) -> None:
        """Close the previous step with what actually became of it, before asking again.

        This is the half upstream Jev fills in from its own post-action fingerprint. Here
        the explorer owns the after-observation and the CRITIC owns the verdict, so the
        honest thing to report is the critic's: ``history``'s last line is the previous
        move and whether it was judged ``ok`` or ``FAILED``, and ``rejection`` is set when
        the previous answer was refused before it was even performed.

        Reporting it matters more than it looks. A policy that is not told its last click
        did nothing has no reason to choose differently, and the screen that prompted the
        click has not changed - which is the loop this project's failure memory exists to
        break, described in :mod:`skillweaver.agent.explorer`.
        """
        if not self._steps:
            return
        last = self._steps[-1]
        outcome = history[-1] if history else ""
        last["page_changed"] = None if not outcome else ("FAILED" not in outcome)
        if rejection:
            last["refused"] = rejection

    def _remember(self, decision: PolicyDecision) -> None:
        """Open a step with what the policy asked for. :meth:`_settle` closes it."""
        self._steps.append(
            {
                "action": decision.why or decision.operation,
                "kind": decision.operation,
                "text": decision.text,
            }
        )


# --------------------------------------------------------------------------------------
# Jev's vocabulary, as an explorer answer
# --------------------------------------------------------------------------------------


def _answer(
    decision: PolicyDecision, catalog: ElementCatalog, snapshot: DomSnapshot, note: str = ""
) -> dict[str, Any]:
    """One decision as the JSON object :meth:`Explorer._ground` parses.

    ``note`` is :func:`_note`'s sentence about targets this screen withheld, appended to
    the thought so it reaches the trajectory with the move it applied to.

    ``expect`` is filled from what the operation is FOR rather than left empty, because
    the critic is handed it as the expectation and a critic with nothing to check
    against falls back to "did anything change", which is a weaker question.

    Raises:
        SkillWeaverError: if the chosen element id is not in the catalogue. That cannot
            happen while the snapshot and the observation are the same frame, which
            :meth:`JevDriver._snapshot_for` has already established - so if it does, the
            frames diverged and performing the move would be performing it blind.
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

    ``json.dumps`` rather than ``repr`` for the literals: the value came off a web page
    and out of a language model, and this string becomes source code that the sandbox
    compiles and the synthesizer may store. JSON quoting escapes every quote, backslash
    and newline in it, so a label containing an apostrophe cannot end the string early.
    """
    chord = ", ".join(json.dumps(key) for key in SELECT_ALL_CHORD)
    return (
        f"ctx.ctl.click(el[{json.dumps(element_id)}])\n"
        f"ctx.ctl.press({chord})\n"
        f"ctx.ctl.type_text({json.dumps(text)})\n"
    )


def _exclusions(
    snapshot: DomSnapshot, dead_ends: Sequence[Attempt]
) -> tuple[dict[str, set[str]], set[str]]:
    """``({operation: element ids to withhold}, {operations put back})``.

    A target already known to fail ON THIS SCREEN is not offered again. The explorer
    would refuse the repeat anyway (:meth:`Explorer._refuse_repeat`), so this does not
    change what can happen - it changes what it COSTS, from one wasted provider call per
    repeat to none, which on a screen with one obvious-looking wrong button is the
    difference between a run that moves on and a run that spends its budget insisting.
    Not offering it at all is the one thing this policy shape can do and a prompted model
    cannot: a prompt can only ASK a model not to repeat itself.

    Two rules, and they pull against each other.

    A failure is a MOVE at an element, not an element, so
    :data:`DEAD_END_OPERATIONS` maps the signature kinds this driver can actually produce
    onto the operations they came from and nothing else is excluded. A click that
    achieved nothing says nothing about typing into the same field, and a version of this
    that dropped the control outright took the search box away from a run whose whole
    task was to type in it.

    And an exclusion that empties an operation's target head is PUT BACK, named in the
    second return value and said out loud in the trajectory by :func:`_note`. A screen
    whose every clickable control has failed is a screen this would otherwise hand the
    policy with no ``CLICK`` at all - and a policy offered only ``WAIT``, ``DONE`` and
    ``BLOCKED`` answers one of them. Measured on the sandbox's Mail screen: 17 controls,
    0 offered, and the only operations left were the two that end the run. A false
    ``BLOCKED`` is a silent failure, so the last resort is to offer the dead targets
    again and let the explorer's own repeat guard say no to them one at a time.
    """
    wanted: dict[str, set[str]] = {}
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

    It goes on the move's ``thought``, which is the line
    :meth:`~skillweaver.agent.explorer.Explorer._write_down` records against the step -
    so a run where the policy was choosing from less than the whole screen says so where
    anyone reading the run will see it, rather than only in a log line nobody keeps.
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
        f"; {withheld} target{plural} already tried and failed on this screen "
        f"{'was' if withheld == 1 else 'were'} withheld"
    )


def _odds(decision: PolicyDecision) -> str:
    """The confidence and probability of one decision, as a short parenthesis.

    Carried into the trajectory on purpose: a move chosen at 0.31 and one chosen at 0.99
    read identically once they are a click, and the trajectory is what the synthesizer
    and a person reviewing a run both read.
    """
    return f"(confidence {decision.confidence:.2f}, p {decision.probability:.2f})"
