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
move is one action OR one code block, so it becomes the block. ``ENTER`` is one
``press_key`` action on whatever holds focus, the explorer's own vocabulary for a key.

Three loop rules are upstream's (``jev-ultrafast`` ``1489129``), each a PURE function here
so it can be checked without a browser: :func:`cycling`, :func:`spent_controls`, and the
one fresh look a ``DONE`` claim has to survive (:func:`fresh_look`).
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
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
    "CONTROL_MISSES",
    "CONTROL_OPERATIONS",
    "CYCLE_REPEATS",
    "CYCLE_WINDOW",
    "DEAD_END_OPERATIONS",
    "DONE_LOOK_MS",
    "NO_CHANGE_LIMIT",
    "SCROLL_PIXELS",
    "SELECT_ALL_CHORD",
    "WAIT_MS",
    "JevDriver",
    "cycling",
    "fresh_look",
    "spent_controls",
]

DEAD_END_OPERATIONS: Mapping[str, str] = {"click": "CLICK"}
"""Which Jev operation a failed move's signature kind is evidence against.

Read this before adding to it. A ``Move.signature`` is ``<kind>:<element_id>:...`` and a
``CLICK`` is the only shape this driver produces that NAMES an element: ``TYPE_TEXT``
grounds to a CODE BLOCK, and ``SCROLL_UP``/``SCROLL_DOWN``, ``WAIT`` and ``BACK`` name
none either. Mapping ``scroll`` or ``drag`` in here would let a move the policy never
made withhold a target it never tried, and mapping a failed click onto ``TYPE_TEXT``
would take away the search box on the one screen whose task is to type in it.

A TARGETLESS operation cannot be withheld this way at all: it is withheld by NAME, under
the reserved ``CONTROLS`` key, by :func:`spent_controls`, and ``BACK`` additionally by
:data:`BACK_SIGNATURE`.
"""

BACK_SIGNATURE = "back"
"""The whole move signature ``Explorer._resolve`` gives a ``back``.

A target-less operation is out of the dead-end mapping's reach twice over, so withholding
it means withholding the OPERATION: a back on this screen's dead-end list puts ``BACK`` in
the reserved ``CONTROLS`` set, and :func:`_without` turns ``DomSnapshot.can_go_back`` off
for that one ask as well, so the rule holds under a policy that does not read the key.

It is the one control withheld on the CRITIC's say-so and on a first miss, against
:data:`CONTROL_MISSES`' two literal ones, and only because it was measured. ``ENTER`` does
not get the same: the per-move verdict is wrong on pages that answer in place, which is
what a search box does, and a wrongly withheld Enter leaves a typed query with no way to
be submitted.

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

CONTROL_OPERATIONS: frozenset[str] = frozenset(
    {"SCROLL_DOWN", "SCROLL_UP", "WAIT", "BACK", "ENTER"}
)
"""The operations with no target, which the reserved ``CONTROLS`` key may name. Never
``DONE`` or ``BLOCKED``: a policy that can say neither has no honest way to stop."""

CONTROL_MISSES = 2
"""Literal no-change results, in one unchanged streak on one exact page state, after which a
target-less operation is withheld. Upstream's number and its reason: results really can
arrive during a second ``WAIT`` and a second scroll can reveal what the first did not, so
one miss is patience and two are a loop. These operations are re-offered every step and
nothing else can withhold them - a fingerprint-keyed dead end is keyed on a screen that
repaints, and ``NO_CHANGE_LIMIT`` does not count a ``WAIT`` at all."""

CYCLE_WINDOW = 8
"""How many of the latest steps :func:`cycling` judges. Upstream's ``window``."""

CYCLE_REPEATS = 6
"""Fewer steps than this is too short a run to call churn. Upstream's ``repeats``."""

DONE_LOOK_MS = 150
"""The pause in the one fresh look a first ``DONE`` claim is answered with.

Upstream's rule: ``DONE`` is judged on the frame the last action produced, often taken
before the effect lands - a cart badge, a navigation, an availability notice - so the claim
has to survive one look at the settled page. The driver holds no controller and cannot
observe, so the look is a MOVE: a wait this long (upstream's quiet window), armed through
``DomPerceiver.rest_after`` as NOT a wait, so an unchanged page gets its quiet window
before the frame is taken. What that costs, read off ``TieredCritic``: the per-move verdict
on a wait is the programmatic ``state_changed`` check, so no model call - against the
escalated one a ``DONE`` judged on an optimistic frame buys and then loses. It does cost
one step of ``max_steps``, one more policy call, and a wait in the trajectory, which
``strip_reflex_waits`` keeps out of a skill. Deliberately not :data:`WAIT_MS`: the explorer
refuses a repeat by signature, and a look that changed nothing must not make a later real
``WAIT`` on that screen a known dead end.
"""

_RESERVED: frozenset[str] = frozenset({"CONTROLS", "LABELS"})
"""The two keys of the ``exclude`` mapping that do not map an operation to element ids:
``CONTROLS`` holds target-less operation names and ``LABELS`` control labels. Anything that
counts or names withheld TARGETS steps over them."""

_ERRAND_MARKER = "The errand: "
"""How ``Explorer._aimed`` introduces the task inside a borrowed step's goal. Read, never
written, here; see :meth:`JevDriver._goal_shown` for what happens if it stops matching."""

_ACTIONS_IN: Mapping[str, int] = {"TYPE_TEXT": 3, "ENTER": 1}
"""Controller actions in the move this driver writes for an operation:
:func:`_type_into` is click, chord, type, and ``ENTER`` is one key press - said rather
than defaulted, because the frame after it is the one a submitted form arrives on. The
explorer observes after each, and ``DomPerceiver.rest_after`` needs to know which
observation is the last."""

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
        refine: ``(goal, url) -> goal``, a rewrite of the errand into the short ordered
            sentences a small policy follows best, or ``None`` - the DEFAULT - for the
            user's own words. See :meth:`_goal_shown`, which is the only place its
            result is ever used, for why that is a hard rule.

    Raises:
        SkillWeaverError: from :meth:`propose` if the perceiver has no snapshot for the
            observation asked about, which means the two were not driving the same run.
    """

    __slots__ = (
        "_decided_ms",
        "_done_seen",
        "_looking",
        "_perceiver",
        "_policy",
        "_refine",
        "_refined",
        "_steps",
        "_taken",
    )

    def __init__(
        self,
        policy: BrowserPolicy,
        perceiver: DomPerceiver,
        *,
        refine: Callable[[str, str], str] | None = None,
    ) -> None:
        self._policy = policy
        self._perceiver = perceiver
        self._refine = refine
        self._refined: dict[str, str] = {}
        self._steps: list[dict[str, Any]] = []
        self._taken: dict[str, set[str]] = {}
        self._decided_ms = 0.0
        # Whether a DONE claim has had its fresh look since the last real action, and
        # whether the move now in flight IS that look. See fresh_look.
        self._done_seen = False
        self._looking = False

    def __repr__(self) -> str:
        return f"JevDriver({self._policy.name()})"

    def name(self) -> str:
        """The policy model identifier."""
        return self._policy.name()

    @property
    def policy_ms(self) -> float:
        """Milliseconds this driver's own models have taken so far: every Jev round trip
        and every call to the text writer. The MODEL half of upstream's split, whose
        other half is ``DomPerceiver.site_ms``. It is not all of a run's model time - the
        critic's calls are the explorer's - and the report says so rather than folding
        them in."""
        return self._decided_ms

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
            self._steps, self._taken, self._refined = [], {}, {}
            self._done_seen = self._looking = False
        state = snapshot.digest
        self._settle(snapshot, state, rejection)
        self._stop_if_inert(observation)
        previous = self._steps[-1].get("page_changed") if self._steps else None
        controls = spent_controls(self._steps, state)
        if any(a.signature == BACK_SIGNATURE for a in dead_ends):
            controls.add("BACK")
        churned, labels = cycling(self._steps)
        exclude, restored = _exclusions(
            snapshot, dead_ends, self._spent(state), controls=controls | churned, labels=labels
        )
        asked = _without(snapshot, exclude.get("CONTROLS", ()))
        goal = self._goal_shown(task.text, observation.url or snapshot.url)
        decision = self._policy.decide(goal, asked, self._steps, exclude)
        looking, self._done_seen = fresh_look(decision.operation, self._done_seen)
        if looking:
            # Not a step: upstream's history holds what was PERFORMED, and a claim the
            # policy is about to be asked again is not that. It is still this policy's time.
            self._decided_ms += decision.latency_ms
            self._perceiver.rest_after(1, snapshot, waited=False)
        else:
            self._remember(decision, state, _label_of(decision, snapshot))
            if decision.operation not in ("DONE", "BLOCKED"):
                self._perceiver.rest_after(
                    _ACTIONS_IN.get(decision.operation, 1),
                    snapshot,
                    waited=decision.operation == "WAIT",
                )
        self._looking = looking
        log.info(
            "jev.decide",
            operation=decision.operation,
            changed=previous,
            confidence=round(decision.confidence, 3),
            probability=round(decision.probability, 3),
            policy_ms=round(decision.policy_ms),
            latency_ms=round(decision.latency_ms),
            offered=len(snapshot.controls),
            withheld=_withheld(exclude),
            controls=sorted(exclude.get("CONTROLS", ())) or None,
            labels=sorted(exclude.get("LABELS", ())) or None,
            restored=sorted(restored) or None,
            looking=looking or None,
        )
        if decision.operation == "BLOCKED":
            raise PolicyBlocked(
                f"the policy found no supported operation on {observation.url or 'this screen'}"
            )
        note = _note(exclude, restored)
        if looking:
            return json.dumps(_look_again(decision, note))
        return json.dumps(_answer(decision, catalog, snapshot, note))

    def _goal_shown(self, text: str, url: str) -> str:
        """The goal as THE POLICY is shown it: refined when refinement is on, and the
        refinement goes nowhere else.

        That is a hard rule, not a preference. ``task`` is what the explorer hands the
        recorder, so its text becomes ``Trajectory.task`` and then the stored skill's
        ``Precedent`` - and the warm path is decided by gates that count WORDS
        (``bind_args``, ``MIN_ACCOUNTED_FOR``, ``asks_for``). A rewrite into simplified
        technical English that reached the store would leave the user's own next request
        matched against prose they never typed. So this returns a string for one call and
        ``task`` is never touched, replaced or re-created here: the explorer's own
        ``_aimed`` draws the same line, showing a policy a goal the store never sees.

        The explorer may hand over an AIMED goal - a borrowed step, then
        ``"The errand: <the task>"`` - and the step changes every move while the errand
        does not. Only the errand is rewritten, once per run, and put back where it was;
        if that wording ever changes this rewrites the whole text instead, which costs
        calls and nothing else. A failed rewrite falls back to the user's own words, and
        says so: it is an optimization, and the run is not worth less without it.
        """
        if self._refine is None:
            return text
        head, marker, errand = text.rpartition(_ERRAND_MARKER)
        errand = errand if marker else text
        if errand not in self._refined:
            began = time.perf_counter()
            try:
                self._refined[errand] = self._refine(errand, url)
                log.info("jev.goal.refined", asked=errand, shown=self._refined[errand])
            except SkillWeaverError as exc:
                log.warning("jev.goal.unrefined", why=str(exc))
                self._refined[errand] = errand
            # The rewrite is this policy's models at work, so it is this policy's time.
            self._decided_ms += (time.perf_counter() - began) * 1000.0
        return f"{head}{marker}{self._refined[errand]}"

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
        content differ afterwards (``DomSnapshot.digest``). It was the CRITIC's verdict
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

        ``refused`` is relayed only when the page does not contradict it. The explorer's
        ``rejection`` is one string for two things - an answer refused BEFORE it ran, and
        a move that ran and that the critic then failed - and the second is that same
        per-move verdict arriving by another door. Measured on a results page whose cart
        line updates in place: the policy's second move was the right *Add to cart* at
        p=1.00, the page said so, and ``state_changed`` failed it (0.633). Told
        ``page_changed: true`` AND "your last move did not work", it added the other nine
        products one after another and never said ``DONE``. A page that literally changed
        is a move that ran, so what is relayed is what the policy can act on: nothing
        happened, and here is why.

        After a fresh look (:func:`fresh_look`) the step was already closed and the move
        just made was this driver's own wait, so only the late truth is taken: an effect
        that landed during the look turns the step's ``page_changed`` true, which is what
        the look was for. The explorer's ``rejection`` is then about the wait and is
        relayed to nobody.
        """
        if not self._steps:
            return
        last = self._steps[-1]
        if self._looking:
            if state != last["state"]:
                last["page_changed"] = True
                last.pop("refused", None)
            last["url"] = snapshot.url
            return
        last["page_changed"] = state != last["state"]
        last["url"] = snapshot.url
        # Upstream's per-step split: what the models took to choose this move, against
        # what the site took to answer it - performing, settling, observing, resting.
        log.info(
            "jev.step",
            kind=last["kind"],
            changed=last["page_changed"],
            model_ms=round(last["model_ms"]),
            site_ms=round(self._perceiver.site_ms - last["site_mark"]),
        )
        if rejection and not last["page_changed"]:
            last["refused"] = rejection
        elif last["kind"] == "CLICK" and last["element_id"]:
            self._taken.setdefault(last["state"], set()).add(last["element_id"])

    def _remember(self, decision: PolicyDecision, state: str, label: str) -> None:
        """Open a step with what the policy asked for. :meth:`_settle` closes it.

        ``state``, ``label``, ``element_id`` and later ``url`` are this driver's own
        bookkeeping; the request builder names the keys it sends, so they never reach the
        wire. ``label`` is there because ``action`` cannot be compared: it is the policy's
        ``why``, which carries an index and a probability that differ on every lap of a
        loop that is otherwise the same two controls.
        """
        self._steps.append(
            {
                "action": decision.why or decision.operation,
                "kind": decision.operation,
                "text": decision.text,
                "label": label,
                "state": state,
                "element_id": decision.element_id,
                "model_ms": decision.latency_ms,
                "site_mark": self._perceiver.site_ms,
            }
        )
        self._decided_ms += decision.latency_ms

    def _spent(self, state: str) -> dict[str, set[str]]:
        """``{operation: element ids}`` this exact page state has already used up.

        Two of upstream's rules, both proofs BY IDENTITY and so both keyed on the exact
        ``DomSnapshot.digest`` rather than on fingerprint similarity. A click already
        made from this state, when the run is standing on the state again, did not
        advance the goal however much it changed the page - that is what turns
        open/close into an oscillation. And a TARGET that has changed nothing since the
        page last changed will change nothing now - until a ``WAIT``, because time
        passing is a reason a dead control may work.

        That last clause is where this departs from upstream ``1489129``, on purpose. It
        dropped the ``WAIT`` reset from its streak because a streak that ends at every
        wait can never hold two of them, so a ``WAIT`` could never be seen to miss
        twice. Here those are two questions asked of one walk (:func:`_streak`):
        :func:`spent_controls` counts THROUGH waits, which is all upstream needed, and a
        target is spent only by the misses SINCE the last wait, as it was measured here.
        A control that hydrates after the frame is drawn changes no digest, so the wait
        is the only evidence this driver ever gets that a second click is worth offering.

        Exact on purpose, and it is the opposite call from
        :meth:`~skillweaver.agent.explorer.FailureMemory.near`. At the 0.26 same-state
        cut a dialog with one more box ticked IS the same screen, so similarity here
        would withhold a legitimate second ``Add``. On a page too noisy to digest
        identically twice this withholds nothing: it fails open, similarity fails closed.
        """
        spent: dict[str, set[str]] = {}
        if state in self._taken:
            spent["CLICK"] = set(self._taken[state])
        for step in _streak(self._steps, state):
            if step["kind"] == "WAIT":
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
        if not inert or len({s.get("url") for s in recent}) != 1:
            return
        where = observation.url or "this screen"
        if all(s["kind"] == "DONE" for s in recent):
            # Not the page's fault and must not read as if it were. Measured: a correctly
            # finished errand whose DONE the critic answered four times with an EMPTY
            # reply - the computer-tool hazard AGENTS.md records as unfixed in the critic.
            # Asking a fifth time buys another such call; the claim and its refusals are
            # the finding, so they are what is reported.
            raise PolicyBlocked(
                f"the policy reported DONE {NO_CHANGE_LIMIT} times on {where} and the claim "
                "was not accepted once, with nothing changing in between; read the critic's "
                "reasons before concluding the task was not done"
            )
        tried = "; ".join(str(s["action"]) for s in recent)
        raise PolicyBlocked(
            f"{where} did not change across the last {NO_CHANGE_LIMIT} moves ({tried}), "
            "so nothing more is spent on it"
        )


# --------------------------------------------------------------------------------------
# Upstream's loop rules, as pure functions of the step list
# --------------------------------------------------------------------------------------


def _streak(steps: Sequence[Mapping[str, Any]], state: str) -> list[Mapping[str, Any]]:
    """The latest steps, newest first, that each LITERALLY changed nothing on ``state``.

    It ends at the first step that changed the page, that has not been closed yet, or
    that was decided on another page state: a step only speaks about the exact page it
    was made on. A ``WAIT`` does not end it; see :meth:`JevDriver._spent` for who stops
    at one and why.
    """
    streak: list[Mapping[str, Any]] = []
    for step in reversed(steps):
        if step.get("page_changed") is not False or step.get("state") != state:
            break
        streak.append(step)
    return streak


def spent_controls(steps: Sequence[Mapping[str, Any]], state: str) -> set[str]:
    """Target-less operations that have missed :data:`CONTROL_MISSES` times on ``state``.

    Upstream's ``repeated``: scroll, wait, back and enter are re-offered on every step,
    so one that provably changes nothing can be chosen until the budget runs out. An
    answer the explorer refused before it ran counts as a miss, and should: it is the
    policy re-picking a move this screen cannot make, which is what ``BACK`` cost nine
    calls doing (:data:`BACK_SIGNATURE`).
    """
    misses = Counter(
        str(step["kind"]) for step in _streak(steps, state) if step["kind"] in CONTROL_OPERATIONS
    )
    return {operation for operation, count in misses.items() if count >= CONTROL_MISSES}


def cycling(
    steps: Sequence[Mapping[str, Any]],
    window: int = CYCLE_WINDOW,
    repeats: int = CYCLE_REPEATS,
) -> tuple[set[str], set[str]]:
    """``(operations, labels)`` the run is churning between, from the moves alone.

    Upstream's ``Agent.cycling``, and its reason: no memory keyed on the page can see
    these. A page that varies between laps - search suggestions, a timestamp, a
    re-rendered overlay - is a new screen every time, so two or three moves alternate
    until the budget is gone while each one "changes the page". Churn, not strict
    alternation: the Amazon loop it was measured on doubles back (open, go, open, open,
    go). And two or more, never one: a single move repeating - *Increase quantity by 1*,
    *Load more*, a scroll - is usually progress, and a dead one is caught elsewhere.

    A step is named by its control's label, or by its operation when it has no target,
    and the two come back APART because they are withheld through different reserved
    keys - so a button that happens to be labelled ``WAIT`` is still a label. ``DONE``
    and ``BLOCKED`` are not moves and are not counted, as upstream's history never holds
    them.
    """
    moves = [s for s in steps if s.get("kind") not in ("DONE", "BLOCKED")][-window:]
    if len(moves) < repeats:
        return set(), set()
    named = {
        (True, str(s["kind"]))
        if s["kind"] in CONTROL_OPERATIONS
        else (False, str(s.get("label") or ""))
        for s in moves
    }
    if not 2 <= len(named) <= 3:
        return set(), set()
    # An unlabelled control counts towards the churn and cannot be withheld by a label.
    return (
        {name for targetless, name in named if targetless},
        {name for targetless, name in named if not targetless and name},
    )


def fresh_look(operation: str, done_seen: bool) -> tuple[bool, bool]:
    """``(answer with a fresh look instead, the new done_seen)`` for one decision.

    The whole of upstream's ``done_seen``: the first ``DONE`` since a real action is
    answered with a look and the policy asked again; a ``DONE`` that follows is the claim.
    Any real action makes the next claim a claim about a new page. ``BLOCKED`` performs
    nothing and leaves it alone - the explorer may ask the same screen again without its
    borrowed workflow, and that is not a new page. A ``DONE`` the critic REFUSED leaves it
    set too, so :meth:`JevDriver._stop_if_inert` still counts four claims as four steps
    with no looks in between.
    """
    if operation == "DONE":
        return not done_seen, True
    if operation == "BLOCKED":
        return False, done_seen
    return False, False


def _without(snapshot: DomSnapshot, controls: Collection[str]) -> DomSnapshot:
    """``snapshot`` with each withheld control that a FLAG offers switched off.

    The reserved key is the contract; this is the same fact said where a policy that does
    not read the key will still see it, which is how ``BACK`` was withheld before the key
    existed. ``can_press_enter`` is another worker's field and may not be here yet.
    """
    off: dict[str, bool] = {}
    if "BACK" in controls and snapshot.can_go_back:
        off["can_go_back"] = False
    if "ENTER" in controls and getattr(snapshot, "can_press_enter", False):
        off["can_press_enter"] = False
    return dataclasses.replace(snapshot, **off) if off else snapshot


def _label_of(decision: PolicyDecision, snapshot: DomSnapshot) -> str:
    """What :func:`cycling` calls this move: the control's own label, as the reserved
    ``LABELS`` key will be matched against it, or the operation when there is no target."""
    control = snapshot.by_element_id.get(decision.element_id or "")
    return control.label if control is not None else decision.operation


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
    if operation == "ENTER":
        # The explorer's own key action, one PressKey on whatever holds focus: it names no
        # element, so there is nothing to ground, and the recorder and the synthesizer
        # already know a key press. Upstream's reason for the operation: a field that
        # submits only on Enter offers nothing to click (GitHub's search, 120 steps to 5).
        field = str(getattr(snapshot, "enter_label", "") or "")
        where = f"the field {field!r}" if field else "the focused field"
        return {
            "thought": f"the policy is pressing Enter to submit {where} {odds}",
            "expect": f"{where} is submitted: its results, or the page it leads to, show",
            "done": False,
            "action": {"kind": "press_key", "keys": ["Enter"]},
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


def _look_again(decision: PolicyDecision, note: str = "") -> dict[str, Any]:
    """The move a first ``DONE`` is answered with; see :data:`DONE_LOOK_MS`.

    ``expect`` says what is true: nothing is promised to change, so a critic that fails
    the wait on an unchanged page has said the page was already settled, not that the
    run went wrong. ``done`` is false - the claim has not been made yet.
    """
    return {
        "thought": (
            "the policy reports every requirement is visibly satisfied "
            f"{_odds(decision)}{note}; taking one fresh look at the settled page before "
            "that claim is made"
        ),
        "expect": (
            "nothing has to change: anything the last action set off - a badge, a "
            "notice, a navigation - has landed, and the page is at rest"
        ),
        "done": False,
        "action": {"kind": "wait", "ms": DONE_LOOK_MS},
    }


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


def _exclusions(
    snapshot: DomSnapshot,
    dead_ends: Sequence[Attempt],
    spent: Mapping[str, Collection[str]] | None = None,
    *,
    controls: Collection[str] = (),
    labels: Collection[str] = (),
) -> tuple[dict[str, set[str]], set[str]]:
    """``({operation: element ids to withhold, + the reserved keys}, {what was put back})``.

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

    ``controls`` and ``labels`` go out under the reserved ``CONTROLS`` and ``LABELS`` keys,
    present only when non-empty. ``labels`` rides the put-back too, worked out HERE from
    the labels themselves rather than by asking :func:`targets_of`, so it holds whether or
    not the policy on the other side reads the key yet: churn labels that would empty a
    target head are dropped whole and ``"LABELS"`` is reported as put back. ``controls``
    does not: with every target spent and every control twice inert, ``BLOCKED`` is true.
    """
    wanted: dict[str, set[str]] = {op: set(ids) for op, ids in (spent or {}).items() if ids}
    for attempt in dead_ends:
        move = signature_move(attempt.signature)
        if move is None:
            continue
        operation = DEAD_END_OPERATIONS.get(move[0])
        if operation is not None:
            wanted.setdefault(operation, set()).add(move[1])
    full = targets_of(snapshot)
    kept = targets_of(snapshot, wanted) if wanted else full
    restored = {op for op in wanted if op in full and op not in kept}
    exclude = {op: ids for op, ids in wanted.items() if op not in restored}
    if labels:
        heads = targets_of(snapshot, exclude) if exclude else full
        if any(all(c.label in labels for c in head.values()) for head in heads.values()):
            restored.add("LABELS")
        else:
            exclude["LABELS"] = set(labels)
    if controls:
        exclude["CONTROLS"] = set(controls) & CONTROL_OPERATIONS
    return {key: held for key, held in exclude.items() if held}, restored


def _note(exclude: Mapping[str, Collection[str]], restored: Collection[str]) -> str:
    """What the trajectory is told about targets withheld here, or ``""``.

    It goes on the move's ``thought``, the line ``Explorer._write_down`` records against
    the step, so a run where the policy chose from less than the whole screen says so
    where anyone reading the run will see it. The reserved keys get sentences of their
    own: what they hold are operation names and labels, and counting those as targets
    would tell the reader a number of elements that were never withheld.
    """
    said: list[str] = []
    operations = sorted(op for op in restored if op not in _RESERVED)
    if operations:
        said.append(
            f"every {'/'.join(operations).lower()} target on this screen has already "
            "failed here, so they are offered again rather than reporting blocked"
        )
    elif withheld := _withheld(exclude):
        said.append(
            f"{withheld} target{'' if withheld == 1 else 's'} already used up on this "
            f"screen {'was' if withheld == 1 else 'were'} withheld"
        )
    if controls := sorted(exclude.get("CONTROLS", ())):
        said.append(f"{', '.join(controls)} withheld as inert or churning on this screen")
    if labels := sorted(exclude.get("LABELS", ())):
        quoted = ", ".join(repr(label) for label in labels)
        said.append(f"the run is churning between {quoted}, so those are withheld")
    elif "LABELS" in restored:
        said.append(
            "the run is churning between a few controls, which are still offered because "
            "withholding them would leave nothing to choose"
        )
    return "".join(f"; {sentence}" for sentence in said)


def _withheld(exclude: Mapping[str, Collection[str]]) -> int:
    """How many TARGETS ``exclude`` withholds: element ids, so not the reserved keys."""
    return sum(len(ids) for op, ids in exclude.items() if op not in _RESERVED)


def _odds(decision: PolicyDecision) -> str:
    """The confidence and probability of one decision, as a short parenthesis. Carried
    into the trajectory on purpose: a move chosen at 0.31 and one chosen at 0.99 read
    identically once they are a click."""
    return f"(confidence {decision.confidence:.2f}, p {decision.probability:.2f})"
