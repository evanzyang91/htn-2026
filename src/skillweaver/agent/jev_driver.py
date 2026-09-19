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
from collections.abc import Sequence
from typing import Any

from skillweaver.agent.explorer import (
    Attempt,
    ElementCatalog,
    PolicyBlocked,
    signature_target,
)
from skillweaver.contracts import Observation, TaskSpec, Usage
from skillweaver.errors import SkillWeaverError
from skillweaver.llm.jev_ import BrowserPolicy, PolicyDecision
from skillweaver.logging_ import get_logger
from skillweaver.perception.dom import DomPerceiver, DomSnapshot

log = get_logger(__name__)

__all__ = ["SCROLL_PIXELS", "SELECT_ALL_CHORD", "WAIT_MS", "JevDriver"]

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
        dead = {
            target
            for attempt in dead_ends
            if (target := signature_target(attempt.signature)) is not None
        }
        pruned = _without(snapshot, dead)
        self._settle(history, rejection)
        decision = self._policy.decide(task.text, pruned, self._steps)
        self._remember(decision)
        log.info(
            "jev.decide",
            operation=decision.operation,
            confidence=round(decision.confidence, 3),
            probability=round(decision.probability, 3),
            latency_ms=round(decision.latency_ms),
            offered=len(pruned.controls),
            pruned=len(snapshot.controls) - len(pruned.controls),
        )
        if decision.operation == "BLOCKED":
            raise PolicyBlocked(
                f"the policy found no supported operation on {observation.url or 'this screen'}"
            )
        return json.dumps(_answer(decision, catalog, snapshot))

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
    decision: PolicyDecision, catalog: ElementCatalog, snapshot: DomSnapshot
) -> dict[str, Any]:
    """One decision as the JSON object :meth:`Explorer._ground` parses.

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
    if operation == "DONE":
        return {
            "thought": (
                f"the policy reports every requirement is visibly satisfied {_odds(decision)}"
            ),
            "expect": "",
            "done": True,
        }
    if operation == "WAIT":
        return {
            "thought": f"the policy is waiting for the page to update {_odds(decision)}",
            "expect": "the page finishes loading and the control it needs appears",
            "done": False,
            "action": {"kind": "wait", "ms": WAIT_MS},
        }
    if operation in ("SCROLL_DOWN", "SCROLL_UP"):
        down = operation == "SCROLL_DOWN"
        return {
            "thought": f"the policy is scrolling {'down' if down else 'up'} {_odds(decision)}",
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
            "thought": f"the policy is clicking {label!r} {_odds(decision)}",
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
            "thought": f"the policy is typing into {label!r} {_odds(decision)}",
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


def _without(snapshot: DomSnapshot, dead: set[str]) -> DomSnapshot:
    """``snapshot`` with the controls in ``dead`` removed, re-indexed.

    A target already known to fail ON THIS SCREEN is not offered again. The explorer
    would refuse the repeat anyway (:meth:`Explorer._refuse_repeat`), so this does not
    change what can happen - it changes what it COSTS, from one wasted provider call per
    repeat to none, which on a screen with one obvious-looking wrong button is the
    difference between a run that moves on and a run that spends its budget insisting.

    Re-indexing matters: a policy's answer is an index into the table it was shown, so
    the table it was shown has to be the one the answer is read against. Everything else
    joins on :attr:`~skillweaver.perception.dom.DomControl.element_id`, which does not
    move.
    """
    if not dead:
        return snapshot
    kept = [control for control in snapshot.controls if control.element_id not in dead]
    if len(kept) == len(snapshot.controls):
        return snapshot
    renumbered = tuple(
        dataclasses.replace(control, index=position)
        for position, control in enumerate(kept, start=1)
    )
    return dataclasses.replace(
        snapshot,
        controls=renumbered,
        by_element_id={control.element_id: control for control in renumbered},
    )


def _odds(decision: PolicyDecision) -> str:
    """The confidence and probability of one decision, as a short parenthesis.

    Carried into the trajectory on purpose: a move chosen at 0.31 and one chosen at 0.99
    read identically once they are a click, and the trajectory is what the synthesizer
    and a person reviewing a run both read.
    """
    return f"(confidence {decision.confidence:.2f}, p {decision.probability:.2f})"
