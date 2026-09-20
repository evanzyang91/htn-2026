"""Skill synthesis: turning one successful fumble into a skill the library can trust.

This is the self-growing half of skillweaver, and its whole value rests on one
property: **nothing enters the library without proving it works**. A model that has
watched a run will happily write plausible code for it. Plausible code that does not
run is the known failure mode of this entire approach - a library of it is worse than
an empty one, because a planner picks from it.

So there are two doors and only one of them opens::

    synthesizer = Synthesizer(llm, store, critic)

    candidate = synthesizer.synthesize(trajectory)     # a draft. NOT stored. Ever.
    admission = synthesizer.admit(trajectory, environment)
    admission.ok, admission.skill, admission.attempts  # stored only if ok

:meth:`Synthesizer.synthesize` is the ``Synthesizer`` Protocol: it writes a candidate
and hands it back. It cannot store, because it does not store - the only ``put`` in
this module is inside :meth:`Synthesizer.admit`, after the gate, and the store is
private to the synthesizer. There is no third path.

The gate, in order, for every attempt:

1. **Structure.** :func:`~skillweaver.skills.model.validate_skill` - a name, a
   one-line summary, a docstring, params that agree with ``run``, a verifier.
2. **The sandbox's static scan.** An import, an ``open``, a dunder: refused here,
   before anything executes, and never admitted.
3. **The precondition.** The environment is put back to the recorded starting screen
   and must actually be there; a skill proved against the wrong screen proves nothing.
   "There" is :data:`MIN_PRECONDITION_SIMILARITY` alike, not identical - a live page
   never reproduces itself exactly, and demanding that it does is how a correct skill
   gets written and thrown away on every website that is not the demo.
   When the world could not be put back at all - the run archived a message, and
   re-opening the page does not un-archive it - the attempt stops at ``"reset"``
   rather than ``"precondition"``. That distinction is the whole point: a skill that
   was DISPROVED and a skill that could never be TRIED are different outcomes, and
   reporting the second as the first is how a whole class of task silently becomes
   unlearnable.
4. **Execution.** The skill is RE-RUN through
   :class:`~skillweaver.skills.sandbox.SkillRunner` against that environment with the
   model's own example arguments, verifier included.
5. **Discrimination.** The verifier that just said yes is replayed against the screen
   the skill STARTED from, held in :class:`_StartScreen`. A verifier that also says
   yes there cannot tell the finished job from the unfinished one, and is refused:
   see :data:`_INDISCRIMINATE`.
6. **The critic.** :class:`~skillweaver.contracts.Critic` judges the screen before
   against the screen after, for the task the trajectory was solving.

Only then is it stored. A failure at any stage is fed back to the model - the error
AND the sandbox's trace, which lists every action and log line in order - and the
skill is rewritten, up to ``max_repairs`` times. Exhausting them is a clean
``Admission(ok=False)`` with every attempt attached, not an exception: a synthesizer
that could not write this skill has not broken, it has simply not written it.

A reply that cannot be read at all - prose around the object, a fence, an empty
turn, a reply cut off by the token cap - is a FORMATTING failure, not a skill
defect. It is retried (``max_format_retries``, with a larger token cap when the
reply ran out of room) and does not spend a repair. Repairs are for code that was
judged and found wanting; burning them on punctuation is how a run ends with
nothing stored. Assistant prefill would be the tidier fix and is not available:
``claude-opus-5`` rejects a conversation that ends on an assistant turn outright.

Before any of that the draft goes through :mod:`skillweaver.skills.refactor`, which
replaces literal coordinates with perception lookups, lifts this run's data into
parameters, and re-anchors every element the draft reached for BY POSITION onto
something nameable. That last one is the difference between a skill that replays on a
real site and one that does not: a model that has just watched a run writes
``ctx.see.by_kind("text")[1]`` for the search box, which is true of exactly the page
it watched. Where the recording names nothing to anchor on the lookup is KEPT and
made to log that it navigates by position - a brittle skill that says so is worth
more than no skill. Hardening first, admission second: what is judged is what is
stored, and the gate's guarantee is untouched by any of it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Box,
    Controller,
    Critic,
    GraphView,
    LLMClient,
    LLMMessage,
    Observation,
    Perceiver,
    Precedent,
    Provenance,
    Screenshot,
    Skill,
    SkillResult,
    SkillStore,
    Trajectory,
    Verdict,
    utcnow,
)
from skillweaver.errors import SandboxViolation
from skillweaver.logging_ import get_logger
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD
from skillweaver.skills.api import SkillLimits
from skillweaver.skills.family import derive_signature
from skillweaver.skills.family import render as render_signature
from skillweaver.skills.model import SkillInvalid, make_skill
from skillweaver.skills.refactor import Hardening, harden
from skillweaver.skills.sandbox import SkillRunner, scan_code
from skillweaver.trajectory.render import describe_trajectory

__all__ = [
    "MIN_PRECONDITION_SIMILARITY",
    "PROMPT",
    "Admission",
    "Attempt",
    "ReplayEnvironment",
    "Stage",
    "Synthesizer",
    "describe_trajectory",
    "load_prompt",
]

log = get_logger(__name__)

PROMPT = "synthesize.md"
"""The generation prompt, next to this project's other prompts in
``skillweaver/agent/prompts/``. It states the published ``ctx`` surface exactly,
forbids imports, and demands a verifier; :func:`load_prompt` reads it."""

Stage = Literal[
    "generation",
    "structure",
    "sandbox",
    "reset",
    "precondition",
    "execution",
    "discrimination",
    "critic",
]
"""Where an attempt stopped. Everything before ``reset`` is decided without touching
the environment at all.

``"reset"`` and ``"precondition"`` both mean the environment was not on the recorded
starting screen, and they are not the same finding. ``"precondition"`` is a verdict
on the SKILL: the world was genuinely put back and the screen still does not match.
``"reset"`` is a verdict on the HARNESS: nothing put the world back, so the candidate
was neither proved nor disproved and no amount of rewriting it would help.

``"discrimination"`` is a verdict on the VERIFIER rather than on the code: the skill
ran and its verifier said yes, and the same verifier also says yes to the screen the
skill STARTED from, so its yes carries no information. See
:class:`_StartScreen`."""

_MAX_REPLY_TOKENS = 16000
"""Ceiling on the token cap a re-ask may escalate to.

Two reasons for this number and not a larger one. A reply that does not fit in it is
not a skill, it is a program. And the Anthropic SDK refuses a NON-streaming request
whose implied duration passes ten minutes - a real run died at ``max_tokens=32000``
with "Streaming is required for operations that may take longer than 10 minutes"
before this ceiling was lowered. 16000 is the contract's own default for
``LLMClient.complete`` and has been exercised live against ``claude-opus-5``."""


MIN_PRECONDITION_SIMILARITY = SAME_STATE_THRESHOLD
"""How like the recorded starting screen the environment must be before a candidate
is re-run in it.

This used to be ``1.0`` - the EXACT same screen - and that is why nothing was ever
learned on a real website. A live page does not reproduce exactly. The demo site
does, at 1.000 after ``/__reset``, so every sandbox run looked green while every
live run wrote a correct skill and threw it away.

It is deliberately not a number of its own. "Am I looking at the screen I recorded?"
is one question, and the project answers it in one place:
:data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD`, which carries the two
corpora it was calibrated on and what it refuses. Do not copy those measurements here.
An earlier version of this docstring did, as a table of whole twenty-fifths, and when
the fingerprinter was rebuilt the table became a record of a signal that no longer
existed - still stating that a page pushed down by a notice was a DIFFERENT screen,
which by then was the defect rather than the behaviour. A threshold is only meaningful
against the shape of the signal it judges, and a second copy of it drifts silently.

What this gate sees, specifically
---------------------------------

The gate walks a COLD first observation - what the recording captured - against a WARM
re-navigation in the SAME browser, which is what ``navigating_environment`` hands it.
That path is kinder than the general case: across the live ``learn`` runs on
en.wikipedia.org behind this constant the precondition scored **1.000 every time**, and
the sandbox scores 1.000 after a reset. The margin the calibration leaves is spent on
the live page that has moved between the two captures, not on this path's own noise.

Which way to be wrong
---------------------

A false REJECT is total: the skill was written correctly and is destroyed, which is the
defect this constant exists to fix. A false ACCEPT costs one sandbox execution and
nothing else - the precondition is the third of five gates, and a candidate run against
the wrong screen still has to execute, satisfy its own verifier and satisfy the critic
before anything is stored. So where the two are close, prefer admitting.

Headed and headless are NOT made comparable
-------------------------------------------

Two FRESH browsers on one page, one visible and one headless, score 0.126 (Wikipedia
Main Page) and 0.421 (``docs.python.org/3/library/json.html``) - straddling the cut, and
the Main Page decisively below it; the same comparison on the sandbox ordering app scores
0.819, above it. Two browsers of the same mode score 1.000 anywhere. The recorded
:class:`~skillweaver.contracts.Fingerprint` still carries no mode, for three reasons.

The gate never compares across modes. ``orchestrator._open_world`` builds ONE controller
and ``navigating_environment`` re-navigates in THAT controller, so the recording and the
re-run are always the same window. The gap is real and is simply not on this path.

Where it IS real - a stored skill replayed later by another process in another mode - the
precondition is the wrong place to carry the answer, and it does not: the mode is
recorded beside the skill by :mod:`skillweaver.skills.store` and read back by
``Agent._explained``, which names the crossing on a warm attempt that lost its screen.
Two strings are compared, and no stored screen identity grew a field for it -
``Fingerprint`` is shared surface besides. See :mod:`skillweaver.render_mode`, which also
says why that is an explanation rather than a refusal: 0.819 means the crossing is
survivable on a small clean page, and a gate tuned against the sandbox is the kind that
looks green until it meets a website.

And it would make the identity worse. A screen is the same screen whoever rendered it;
these scores are the fingerprinter correctly reporting that two renderers produce
measurably different pixels. The answer is to run one renderer - ``eval/wikipedia.yaml``
already says headless, for exactly this reason, and ``--headless`` is now how a run says
so - not to teach the identity to ignore a difference it was right to notice.
"""


@lru_cache(maxsize=4)
def load_prompt(name: str = PROMPT) -> str:
    """The text of a prompt shipped in ``skillweaver.agent.prompts``.

    Cached: the file does not change while a process runs, and synthesis reads it
    on every call.

    Raises:
        FileNotFoundError: if no such prompt is packaged.
    """
    return (resources.files("skillweaver.agent.prompts") / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# What the gate runs against
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayEnvironment:
    """A world to re-execute a candidate skill in, standing at the recorded start.

    ``controller`` and ``perceiver`` are what the skill acts and sees through;
    ``graph`` is the read-only site graph it may consult, if there is one.

    The gate asks for a FRESH one per attempt - see ``environment`` in
    :meth:`Synthesizer.admit` - so a repair never inherits the half-finished screen
    its predecessor left behind, which would let a broken skill pass by accident.

    Attributes:
        controller: The hands the candidate acts through.
        perceiver: The eyes it sees through.
        graph: The read-only site graph, when there is one.
        restored: Whether something ACTUALLY put this world back to its starting
            state - a seed reload, a fresh database, ``controller.reset()`` - as
            opposed to merely re-opening the screen it started on. It decides what a
            precondition mismatch is allowed to mean: with ``True`` the world was put
            back and the candidate is judged, with ``False`` the gate cannot tell a
            wrong skill from an unrestorable world and says the second, which is the
            honest answer. Defaults to ``False`` because re-navigating is the thing
            most callers can do and it is NOT a restore: nothing about re-opening
            an inbox un-archives the message the run archived.
    """

    controller: Controller
    perceiver: Perceiver
    graph: GraphView | None = None
    restored: bool = False


EnvironmentFactory = Callable[[], ReplayEnvironment]
"""Called once per admission attempt; must return the recorded environment put back to
the trajectory's first screen (``controller.reset()``, a fresh page, a new browser,
a seed-restoring endpoint), with ``ReplayEnvironment.restored`` saying whether it
managed to."""


# --------------------------------------------------------------------------------------
# What came of it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attempt:
    """One pass through the gate: what was tried and how far it got.

    Attributes:
        index: ``0`` for the first generation, ``1..n`` for repairs.
        stage: Where it stopped, or ``"critic"`` when it passed everything.
        ok: Whether this attempt was admitted.
        skill: The candidate as it was judged - hardened, validated, NOT stored.
            ``None`` when the model returned nothing usable.
        error: Why it failed, in the words the model is shown. ``None`` when ``ok``.
        trace: The sandbox trace of the run, when it got as far as running.
        result: The raw :class:`~skillweaver.contracts.SkillResult`, when it ran.
        verdict: The critic's judgment, when one was asked for.
        hardening: What the hardening pass changed before this attempt.
    """

    index: int
    stage: Stage
    ok: bool
    skill: Skill | None = None
    error: str | None = None
    trace: tuple[str, ...] = ()
    result: SkillResult | None = None
    verdict: Verdict | None = None
    hardening: Hardening | None = None

    def __str__(self) -> str:
        head = f"attempt {self.index} ({self.stage})"
        return f"{head}: admitted" if self.ok else f"{head}: {self.error}"


@dataclass(frozen=True, slots=True)
class Admission:
    """The outcome of :meth:`Synthesizer.admit`.

    Attributes:
        ok: Whether a skill entered the library.
        skill: The STORED skill (``version >= 1``) when ``ok``, else ``None``. A
            non-``None`` value here is a promise: this code ran, and a critic agreed.
        attempts: Every attempt, in order. ``attempts[0]`` is the first generation.
        reason: One line saying why, for a log or a status line.
    """

    ok: bool
    skill: Skill | None
    attempts: tuple[Attempt, ...] = ()
    reason: str = ""

    @property
    def repairs(self) -> int:
        """How many times the skill was rewritten after the first draft."""
        return max(len(self.attempts) - 1, 0)

    @property
    def unproved(self) -> bool:
        """Whether the gate never got to judge this skill because the world could not
        be put back - as opposed to judging it and finding it wanting.

        A caller that reports both as "rejected" tells its user the model wrote bad
        code, when in fact the harness has no way to undo what the task changed and
        NO skill for that task could ever be admitted. The fix is a reset hook, not a
        better model, and only this flag says so.
        """
        return bool(self.attempts) and self.attempts[-1].stage == "reset"

    def __str__(self) -> str:
        name = self.skill.name if self.skill else "no skill"
        return f"admission {'ok' if self.ok else 'rejected'} ({name}): {self.reason}"


@dataclass(frozen=True, slots=True)
class _Draft:
    """One parsed model reply."""

    raw: str
    name: str
    summary: str
    docstring: str
    code: str
    verifier_code: str | None
    params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    example_args: Mapping[str, Any] = field(default_factory=dict, hash=False)
    requires: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# Describing the run to the model
# --------------------------------------------------------------------------------------
#
# :func:`~skillweaver.trajectory.render.describe_trajectory` is the whole of it, and it
# lives with the recording rather than here: what a trajectory MEANS - which steps were
# one move, which moves the critic threw out - is a property of the recording, and this
# module only asks for it. Re-exported because that is where every caller and test
# already reaches for it.


def _repair_brief(attempt: Attempt) -> str:
    """What the model is shown after a rejection: the stage, the error, the trace.

    Rewrites the hardening pass made are normally left unsaid - the model is being
    asked about the defect, not about the polish. The two that change how the skill
    WAITS are the exception, and deliberately so: the model cannot see the code that
    was actually run. A sleep it genuinely needed and is never told was deleted is a
    repair loop with no exit - it would write the same wait again, the pass would
    remove it again, and the gate would reject it again until the attempts ran out.
    A read the pass made wait is the same debt the other way round: told nothing, the
    model reads a failure that is no longer about timing and "fixes" the wait it
    already has.
    """
    lines = [
        "Your skill was REJECTED and has not been stored.",
        f"Stage: {attempt.stage}",
        f"Error: {attempt.error}",
    ]
    if attempt.trace:
        lines.append("")
        lines.append("Trace of the run, in order:")
        lines.extend(f"  {line}" for line in attempt.trace)
    if attempt.verdict is not None and not attempt.verdict.ok:
        lines.append("")
        lines.append(f"The critic said: {attempt.verdict.reason}")
    removed = attempt.hardening.waits_removed if attempt.hardening is not None else ()
    if removed:
        lines.append("")
        lines.append(
            "Before this run, "
            + ", ".join(str(wait) for wait in removed)
            + (" was removed" if len(removed) == 1 else " were removed")
            + ": every ctx.ctl action is already followed by a settle that waits for "
            "the page to finish loading, so a fixed wait after one sleeps on top of a "
            "wait that has already happened. If this run failed because something the "
            "page load does NOT cover needed the time - an animation, a debounce, a "
            "spinner - write the wait again with a ctx.log on the line before it "
            "naming that thing, and it will be kept exactly as you wrote it."
        )
    awaited = attempt.hardening.awaits_added if attempt.hardening is not None else ()
    if awaited:
        lines.append("")
        lines.append(
            "Before this run, "
            + ", ".join(str(read) for read in awaited)
            + (" was" if len(awaited) == 1 else " were")
            + " rewritten to ctx.wait_for_text(...), which looks again for up to four "
            "seconds while the page answers and returns the instant the text is there. "
            "So that read did NOT fail because it was too early. If it still found "
            "nothing, either the click did not do what it was supposed to or the text "
            "you named is not what the answered screen says - and text that is on the "
            "screen you were LEAVING satisfies the wait at once and proves nothing."
        )
    if attempt.stage == "execution" and not removed and not awaited:
        lines.append("")
        lines.append(_SLOWER_THAN_THE_RECORDING)
    if attempt.stage == "discrimination":
        lines.append("")
        lines.append(_A_VERIFIER_MUST_BE_ABLE_TO_FAIL)
    lines.append("")
    lines.append(
        "Fix exactly this and return the same JSON object shape again. "
        "Keep what was working; change what the error names."
    )
    return "\n".join(lines)


_SLOWER_THAN_THE_RECORDING = (
    "One thing to rule out before you change the logic, because your skill runs FAR "
    "faster than the recording it was written from: a control that acts WITHOUT "
    "navigating is not covered by the settle after each action. That settle waits for "
    "the page's load event, which a control that fires a background request and then "
    "redirects has already fired long before. Measured on a live shop on 2026-09-19: "
    "the click on 'Add to cart' returned in 130ms with the document complete and the "
    "old URL still showing, and the cart page it leads to did not commit until 1170ms "
    "- so a skill that read the cart straight after the click read the page it was "
    "still standing on and correctly concluded the cart was empty. The recording never "
    "hit this because a model was being asked what to do next between every pair of "
    "actions. If that is what happened here, do NOT add a retry around the whole "
    "sequence - it fails the same way, only twice, and do not add ctx.ctl.wait(...), "
    "which spends a fixed duration whether or not it was needed and is removed before "
    "your skill is run. Use ctx.wait_for_text(<what the ANSWERED screen says>) in place "
    "of the ctx.see.find_text that read too early: it looks again while the page "
    "answers, returns the instant the text is there, and costs nothing when the page "
    "had already answered. Name text only the answered screen carries - 'Add to cart' "
    "is on the page you are leaving and would satisfy the wait immediately."
)
"""What a skill rejected at EXECUTION is told, on top of its error.

The synthesizer writes a skill from a recording made at model speed and it is then
re-run at code speed, which is one to two orders of magnitude faster. Every failure
mode that is really a race looks, in the error text, exactly like a logic bug: "the
cart is still empty" is what an empty cart and a cart read too early both say.

Without this the repair loop reliably picks the wrong fix. Measured on live
splitkb.com on 2026-09-19, a correctly-explored add-to-cart run was rejected at
execution on all three attempts, and the model's repair each time was to wrap the
whole sequence in a retry - which fails identically, only twice, because the second
pass is just as fast as the first.

It is NOT offered when the hardening pass removed a wait, nor when it already made a
read wait: both cases have their own and more specific message directly above, and two
paragraphs about waiting would make the more precise one easier to miss. The second
exclusion matters most - telling a model to add a wait to code the pass has ALREADY
made wait is how a repair loop spends three attempts on a race that is no longer there.
"""


# --------------------------------------------------------------------------------------
# Is the verifier worth storing?
# --------------------------------------------------------------------------------------
#
# A verifier is only worth storing if it can FAIL on a screen the skill might
# plausibly land on. That is the whole property, and it is not what "the verifier
# passed" measures: a check that matches SITE CHROME - the top nav, the footer, the
# logo, a category word that every page of the site carries - passes on the finished
# job and on every unfinished one alike, so its yes says nothing.
#
# Measured on live splitkb.com on 2026-09-19. A synthesized add-to-cart skill was
# given a verifier matching the word "Keycaps", which that shop's top navigation
# shows on EVERY page. It passed 8 of 8 replays whose carts were empty all 8 times
# by the site's own ``/cart.js`` (``item_count=0``), and ``SkillStats`` for that
# skill read 14 runs, 14 successes. Nothing downstream can notice this: the store is
# told ``ok`` by the sandbox, and the sandbox is told ``ok`` by the verifier. Only
# the warm critic caught it, every time, which is the system working - but a
# verifier that passes on the wrong screen is worse than no verifier at all, because
# it is what turns a skill that did nothing into a STORED one, and then into a
# statistic.
#
# The gate already re-runs a candidate SKILL from its recorded start screen before
# admitting it. This is the same move for the verifier, and it costs nothing: the
# start screen has ALREADY been observed by this attempt - it is the ``before`` the
# precondition was measured against - so replaying the verifier over it is a
# dictionary lookup and a function call. No capture, no detector, no OCR, no second
# browser, and the live page is never touched.


class _StartScreen:
    """One already-captured :class:`~skillweaver.contracts.Observation` dressed as a
    :class:`~skillweaver.contracts.Controller` and a
    :class:`~skillweaver.contracts.Perceiver`, so a verifier can be asked about the
    screen the skill started from without going back to it.

    Going back to it is the alternative, and it is the wrong one twice over: it costs
    a second reset and a second full perception pass per attempt, and on a live site
    the screen it returned to would not be the screen the precondition was measured
    against anyway. The observation this attempt already took IS that screen.

    Every action is REFUSED rather than performed. A verifier is a read-only question
    about the end state; one that clicks would be mutating the world from inside the
    gate, and here there is no world to mutate - only a frozen frame. The refusal
    surfaces to skill code as the ordinary ``ControllerError`` that
    :class:`~skillweaver.skills.api.ActionView` raises for any refused action, which
    makes the probe inconclusive, and an inconclusive probe ADMITS (see
    :meth:`Synthesizer._passes_on_the_start_screen`).
    """

    __slots__ = ("_observation",)

    def __init__(self, observation: Observation) -> None:
        self._observation = observation

    def __repr__(self) -> str:
        return f"_StartScreen(url={self._observation.url!r})"

    # -- Perceiver -------------------------------------------------------------------

    def observe(self, controller: Controller) -> Observation:
        """The captured screen, every time. ``controller`` is ignored: there is
        nothing to capture and nothing that could have changed."""
        return self._observation

    # -- Controller ------------------------------------------------------------------

    def capture(self) -> Screenshot:
        return self._observation.screenshot

    def perform(self, action: Action) -> ActionResult:
        return ActionResult(
            ok=False,
            error=(
                "a verifier may not act: it is being replayed against a screen that "
                "was captured before the skill ran"
            ),
        )

    def viewport(self) -> Box:
        shot = self._observation.screenshot
        return Box(0, 0, shot.width, shot.height)

    def supports(self, action_kind: ActionKind) -> bool:
        return False

    def url(self) -> str | None:
        return self._observation.url

    def describe(self) -> str:
        return f"the recorded starting screen ({self._observation.url or 'no url'})"

    def close(self) -> None:
        """Nothing to release: this holds a frame, not a browser."""


_PROBE_SUFFIX = "__on_the_start_screen"
"""Appended to the candidate's name for the probe run.

The probe must not land in the library's statistics. A candidate is not in the store
during admission, so ``SkillRunner._record`` finds nothing to write and says so - but
a RELEARN of a skill that is already stored would be recorded, and a probe run is not
a run of the skill. A name no store holds is the whole mechanism."""

_PROBE_CODE = "def run(ctx, result):\n    return result\n"
"""The probe's ``run``: it performs nothing and hands the real run's return value
straight to the verifier.

The verifier is ``verify(ctx, result)`` and a real one may read ``result``, so the
probe has to be able to supply it. Passing it as an argument is the only way in - a
skill's source is text, and a Python value cannot be written into it."""

_INDISCRIMINATE = (
    "the verifier for {name!r} ALSO passes on the screen the skill started from, so "
    "it cannot tell the finished task from the unstarted one. It would say yes to a "
    "run that did nothing."
)
"""The rejection, in the words the model is shown."""

_A_VERIFIER_MUST_BE_ABLE_TO_FAIL = (
    "A verifier is only worth storing if it can FAIL on a screen this skill might "
    "plausibly land on - above all the screen it STARTS on, which is exactly where "
    "yours was just re-run and passed. Site chrome is what usually does this: the "
    "top navigation, the footer, the logo, a category or section word the site shows "
    "on every one of its pages. Those words are on the end screen, so matching one "
    "looks like a check and is not one.\n"
    "\n"
    "Key on something that CHANGED because your skill ran, and prefer these in order:\n"
    "  1. A count or a quantity that moved - a cart badge, 'N items', a result "
    "count, a total or a price that is only shown once there is something to total.\n"
    "  2. A row, card or line that is NEWLY present and names the thing your "
    "parameters chose - the article title that was searched for, the product that was "
    "added. Prefer the parameter's own value over a fixed string: it is what makes "
    "the check specific to THIS run.\n"
    "  3. A URL that differs from the starting one - read it with ctx.see only if it "
    "is on screen; otherwise use 1 or 2.\n"
    "  4. Text that exists ONLY in the finished state - a confirmation heading, an "
    "'added to your cart' line, an empty-state message that has gone away.\n"
    "\n"
    "Then check your new verifier against the start screen yourself before you send "
    "it: if every string it looks for was already on screen before your skill acted, "
    "it will be rejected again for the same reason. Change the VERIFIER, not the "
    "code - the code ran and did the job."
)
"""What a skill rejected at ``"discrimination"`` is told, on top of its error.

Ordered deliberately, and the order is the fix rather than the paragraph. A longer
lecture about rigour produces a longer verifier over the same words; naming the
SIGNALS - a count, a newly present row carrying the parameter's own value, a changed
URL, text that exists only when the job is done - moves the check onto something the
starting screen does not have. Each of those is a thing the ACTION changed, which is
the only class of evidence that can distinguish the two screens at all.

It also says which half to rewrite. A model told only "rejected" rewrites the code,
because that is what a rejection has always meant here, and the code was fine.
"""


# --------------------------------------------------------------------------------------
# Reading the reply
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Unreadable:
    """Why one reply produced no draft.

    Attributes:
        why: The sentence the model is shown.
        format_only: ``True`` when nothing JSON-shaped came back AT ALL - prose, a
            fence with no object in it, an empty turn, a reply cut off mid-object.
            Such a reply says nothing about the skill, so it is re-asked for rather
            than charged to the repair budget. ``False`` is a real content defect -
            a JSON object with no verifier, say - which IS the model's mistake to
            fix and does spend a repair.
        needs_room: Whether re-asking is only worth it with a larger token cap. An
            empty reply counts: on a thinking model an empty turn usually means the
            cap was spent before any text was written.
    """

    why: str
    format_only: bool = True
    needs_room: bool = False


def _balanced_objects(text: str) -> Iterator[str]:
    """Every balanced ``{...}`` span in ``text``, outermost first, in order.

    String-aware, so a brace inside a JSON string - and Python code full of them is
    exactly what these replies carry - does not end the span. This is what makes the
    reader tolerant of the model explaining itself: prose before the object, a
    ``{placeholder}`` in that prose, a second fenced snippet afterwards. Each of
    those defeated a ``find("{")``/``rfind("}")`` pair, and each of them cost a
    generation attempt in a live run.
    """
    depth, start, in_string, escaped = 0, -1, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]


def _json_object(text: str) -> dict[str, Any] | None:
    """The skill object in a model reply, however it was wrapped, or ``None``.

    Tries the whole reply first, then every balanced object in it. An object with a
    ``code`` key is unmistakably the answer and wins immediately; any other object -
    a ``{}`` in a code sample, say - is only a fallback, so a stray brace earlier in
    the reply cannot shadow the real one.
    """
    fallback: dict[str, Any] | None = None
    for candidate in (text.strip(), *_balanced_objects(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "code" in parsed:
            return parsed
        if fallback is None:
            fallback = parsed
    return fallback


def _parse_draft(text: str, *, truncated: bool = False) -> tuple[_Draft | None, _Unreadable | None]:
    """A draft from one reply, or ``(None, why it is not one)``.

    Args:
        text: The reply.
        truncated: Whether the provider stopped at the token cap.
    """
    data = _json_object(text)
    if data is None:
        if not text.strip():
            return None, _Unreadable("the reply was empty", needs_room=True)
        if truncated:
            return None, _Unreadable(
                "the reply hit the token cap before the JSON object was closed",
                needs_room=True,
            )
        return None, _Unreadable("the reply contained no JSON object")
    missing = [key for key in ("name", "summary", "docstring", "code") if not data.get(key)]
    if missing:
        return None, _Unreadable(
            f"the JSON object is missing {', '.join(missing)}", format_only=False
        )
    verifier = data.get("verifier_code")
    if not verifier or not str(verifier).strip():
        return None, _Unreadable(
            "the skill has no verifier: return verifier_code defining "
            "def verify(ctx, result) that checks the end state with ctx.see",
            format_only=False,
        )
    requires = data.get("requires") or ()
    if isinstance(requires, str):
        requires = (requires,)
    params = data.get("params") or {}
    example = data.get("example_args") or {}
    if not isinstance(params, dict) or not isinstance(example, dict):
        return None, _Unreadable(
            "params and example_args must both be JSON objects", format_only=False
        )
    return (
        _Draft(
            raw=text,
            name=str(data["name"]).strip(),
            summary=str(data["summary"]).strip(),
            docstring=str(data["docstring"]).strip(),
            code=str(data["code"]),
            verifier_code=str(verifier),
            params=params,
            example_args=example,
            requires=tuple(str(r) for r in requires),
        ),
        None,
    )


_FORMAT_NUDGE = (
    "That reply could not be read: {why}.\n\n"
    "Nothing about your skill has been judged - this is only the shape of the reply. "
    "Send the same answer again as ONE JSON object and nothing else: no sentence "
    "before it, no sentence after it, no code fence, no tool call. Begin with {{ and "
    "end with }}."
)
"""What the model is shown after an unreadable reply. It says explicitly that the
skill was not judged, so the model does not "fix" working code it was never told was
broken."""


def _echo(reply: str) -> tuple[LLMMessage, ...]:
    """The model's own reply, as the assistant turn a follow-up hangs off.

    Empty when the reply was empty. There is nothing to quote back, and an empty
    assistant turn is worse than no turn: some providers refuse a conversation that
    ends on one, and ``claude-opus-5`` refuses an assistant turn in final position
    outright. A follow-up user turn on its own is accepted, and was confirmed against
    the live API before this was written.
    """
    return (LLMMessage(role="assistant", text=reply),) if reply.strip() else ()


REST_WINDOW_MS = 500.0
"""How long a screen must hold the SAME fingerprint before the gate believes it has
stopped painting. See :func:`_observe_at_rest`, which carries the measurements."""

REST_BUDGET_MS = 4000.0
"""The most the gate will wait for a screen to come to rest before judging whatever is
there. The slowest settle measured on walmart.com was 2s; this is
:data:`skillweaver.reset_actions.SETTLE_BUDGET_MS`, which answers the same question."""

REST_POLL_MS = 120.0
"""Between reads while waiting. A DOM read is ~50ms and a cached pixel read is cheap."""

_sleep = time.sleep


def _with_what_it_earned(skill: Skill, generation: _Generation, trajectory: Trajectory) -> Skill:
    """``skill`` with its action signature and its first precedent, IF it earned them.

    Called for a candidate the gate has just ADMITTED, and that is the whole point of
    where it sits: by this line the exact code has run to completion in a reset world,
    its own verifier has said yes to the end screen and no to the start screen, and
    the critic has agreed. What a skill DOES (:mod:`skillweaver.skills.family`) and
    the sentence-and-arguments it is proven to serve
    (:class:`~skillweaver.contracts.Precedent`) are both claims other skills and later
    requests will lean on, so neither is written from anything less.

    A candidate with NO verifier is admitted on the critic alone, as it always was,
    and gets neither: nothing checked its own account of its end state, so it joins
    no family and lends nobody a template. Arguments that are not plain values are
    left out of the precedent rather than stringified - a template is matched against
    text, and a list has no place in a sentence.
    """
    if not skill.verifier_code:
        log.info("skill.admit.no_signature", name=skill.name, why="it carries no verifier")
        return skill
    signature = derive_signature(skill.code, trajectory)
    draft = generation.draft
    example = dict(draft.example_args) if draft is not None else {}
    plain = {
        name: value
        for name, value in example.items()
        if name in skill.params and isinstance(value, str | int | float | bool)
    }
    log.info(
        "skill.admit.signature",
        name=skill.name,
        domain=skill.domain,
        signature=render_signature(signature),
        precedent=trajectory.task,
        args=plain,
    )
    return replace(
        skill,
        action_signature=signature,
        precedents=(Precedent(trajectory.task, plain),),
    )


def _observe_at_rest(env: ReplayEnvironment) -> Observation:
    """Observe the environment once it has STOPPED CHANGING, and not before.

    The gate reads the screen twice - where the candidate starts and where it ended -
    and both reads used to happen the instant the controller returned. ``_settle`` waits
    for the load event, and a page that paints after it has already fired is then
    photographed mid-paint. On live walmart.com the gate's precondition read 0.40, 1.00,
    0.238, 0.238, 1.00, 0.40 over six attempts against one recorded screen and a 0.26
    cut: not six different pages, one page caught at six moments. The filled cart moved
    through five fingerprints in its first 340ms (0.474 to its own final screen at 0s)
    and then held ONE, exactly equal, for every one of the next 125 reads.

    That equality is the signal. This waits for a THING - the fingerprint holding still
    for :data:`REST_WINDOW_MS` - and never for the score: it is not told what the screen
    is about to be compared with, so it cannot keep looking until a screen happens to
    pass. It returns ONE observation, the last one, which is then judged exactly as
    strictly as before. A screen that never rests inside :data:`REST_BUDGET_MS` - a
    carousel, a clock - is judged as it stands, which is what happened before this
    existed; the wait costs such a page time and buys it nothing.

    A window and not a single quiet poll, because a zero-window check reads quiet before
    the page has started (25-45ms, measured; AGENTS.md). This is the gate, which runs a
    handful of times per LEARNED skill - not ``_settle``, which would tax every action.
    """
    started = time.monotonic()
    seen = env.perceiver.observe(env.controller)
    still_since = time.monotonic()
    reads = 1
    while True:
        now = time.monotonic()
        rested = (now - still_since) * 1000.0 >= REST_WINDOW_MS
        if rested or (now - started) * 1000.0 >= REST_BUDGET_MS:
            log.info(
                "skill.admit.observed",
                rested=rested,
                reads=reads,
                waited_ms=round((now - started) * 1000.0),
            )
            return seen
        _sleep(REST_POLL_MS / 1000.0)
        latest = env.perceiver.observe(env.controller)
        reads += 1
        if latest.fingerprint.value != seen.fingerprint.value:
            still_since = time.monotonic()
        seen = latest


def _not_at_the_start(
    candidate: Skill, similarity: float, threshold: float, *, restored: bool
) -> str:
    """Why the candidate was not run, in the words the difference deserves.

    The same fingerprint mismatch means two opposite things. With the world genuinely
    put back it is about the skill; without, it is about the harness, and saying so
    is the difference between "your model wrote bad code" and "this task changes
    something and nothing here can change it back", which is the reason a mutating
    task could never be learned at all.

    The threshold is quoted next to the score either way. A bare "similarity 0.96" is
    what this gate reported for a year while a ``1.0`` default rejected every live
    page, and reading it needs the number it was measured against.
    """
    where = (
        f"(similarity {similarity:.2f}, below the {threshold:.2f} required, "
        f"to {candidate.precondition.value})"  # type: ignore[union-attr]
    )
    if restored:
        return (
            f"the environment is not on the recorded starting screen {where}; "
            "the skill cannot be proved here"
        )
    return (
        f"could not restore the world to the recorded starting screen {where}: this run "
        "changed something that re-opening the same screen does not undo, so the skill "
        "was neither proved nor disproved. Give the admission gate a way to put this "
        "world back to its starting state and learn the task again."
    )


@dataclass(frozen=True, slots=True)
class _Generation:
    """One draft, and the conversation that produced it.

    ``messages`` is the conversation the final reply answered - including any
    re-asks - so a repair is appended to what the model actually last saw.
    """

    draft: _Draft | None
    unreadable: _Unreadable | None
    reply: str
    messages: tuple[LLMMessage, ...]
    reasks: int = 0


# --------------------------------------------------------------------------------------
# The synthesizer
# --------------------------------------------------------------------------------------


class Synthesizer:
    """A :class:`~skillweaver.contracts.Synthesizer` with an admission gate.

    Args:
        llm: Writes and repairs the code. Its ``name()`` is recorded in provenance.
        store: Where an ADMITTED skill is put. Private: the only call to ``put`` in
            this class is the one after the gate, so there is no way to store a skill
            that has not proved itself through this object.
        critic: Judges the replayed run. Required, for the same reason.
        max_repairs: How many times a rejected skill may be rewritten. ``0`` means
            one attempt and no repairs. Bounded because a model that cannot fix its
            code in three goes will not fix it in thirty, and every go costs money.
            Only a JUDGED skill spends one - see ``max_format_retries``.
        max_format_retries: How many times an UNREADABLE reply may simply be
            re-asked for, per attempt. Prose around the object, an empty turn, a
            reply cut off at the token cap: none of those is a fact about the skill,
            so charging them to ``max_repairs`` spends the gate's whole budget on
            punctuation and stores nothing. ``0`` restores the old behaviour.
        limits: Sandbox limits for the admission run.
        min_steps: Runs shorter than this are not worth a skill; ``synthesize``
            returns ``None`` for them.
        min_similarity: How like the recorded starting screen the environment must
            be before a candidate is run in it, in ``0.0..1.0``. Defaults to
            :data:`MIN_PRECONDITION_SIMILARITY`, which is measured; ``1.0`` demands
            the byte-identical screen and no live page ever gives one.
        max_tokens, temperature: Passed to ``llm.complete``.

    Not thread-safe, and one instance may be reused across trajectories.
    """

    __slots__ = (
        "_critic",
        "_limits",
        "_llm",
        "_max_format_retries",
        "_max_repairs",
        "_max_tokens",
        "_min_similarity",
        "_min_steps",
        "_prompt",
        "_store",
        "_temperature",
    )

    def __init__(
        self,
        llm: LLMClient,
        store: SkillStore,
        critic: Critic,
        *,
        max_repairs: int = 2,
        max_format_retries: int = 2,
        limits: SkillLimits | None = None,
        min_steps: int = 1,
        min_similarity: float = MIN_PRECONDITION_SIMILARITY,
        max_tokens: int = 8000,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> None:
        if max_repairs < 0:
            raise ValueError(f"max_repairs must not be negative, got {max_repairs}")
        if max_format_retries < 0:
            raise ValueError(f"max_format_retries must not be negative, got {max_format_retries}")
        if not 0.0 <= min_similarity <= 1.0:
            # A value above 1.0 is unreachable - `Fingerprint.similarity` is capped
            # there - so it silently rejects EVERY candidate. That is how a test rig
            # can look like it is exercising the gate while proving nothing, and it
            # is worth a loud failure rather than an afternoon.
            raise ValueError(f"min_similarity must be within 0.0..1.0, got {min_similarity}")
        self._llm = llm
        self._store = store
        self._critic = critic
        self._max_repairs = max_repairs
        self._max_format_retries = max_format_retries
        self._limits = limits if limits is not None else SkillLimits()
        self._min_steps = min_steps
        self._min_similarity = min_similarity
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._prompt = prompt if prompt is not None else load_prompt()

    def __repr__(self) -> str:
        return (
            f"Synthesizer(model={self._llm.name()!r}, "
            f"store={type(self._store).__name__}, max_repairs={self._max_repairs})"
        )

    # -- the Synthesizer protocol ----------------------------------------------------------

    def synthesize(self, trajectory: Trajectory) -> Skill | None:
        """Write a candidate skill from ``trajectory``: hardened, validated,
        ``version=0``, and **not stored**.

        ``None`` when the run failed, was too short to be worth a skill, or the model
        did not return usable code. Storing happens only in :meth:`admit`.

        Raises:
            ProviderError: if the model call fails.
        """
        if not self._worth_keeping(trajectory):
            return None
        generation = self._generate([LLMMessage(role="user", text=describe_trajectory(trajectory))])
        if generation.draft is None:
            why = generation.unreadable.why if generation.unreadable else "nothing usable"
            log.info("skill.synthesize.unusable", run_id=trajectory.run_id, why=why)
            return None
        candidate, error, _ = self._build(generation.draft, trajectory)
        if candidate is None:
            log.info("skill.synthesize.invalid", run_id=trajectory.run_id, error=error)
            return None
        return candidate

    # -- the gate --------------------------------------------------------------------------

    def admit(self, trajectory: Trajectory, environment: EnvironmentFactory) -> Admission:
        """Synthesize a skill and store it ONLY if it proves itself.

        The candidate is hardened, structurally validated, statically scanned,
        re-executed through the sandbox against a fresh ``environment`` standing at
        the recorded starting screen, asked to prove its VERIFIER discriminates - see
        :meth:`_passes_on_the_start_screen` - and judged by the critic. A rejection is
        fed back to the model with its error and trace and the skill is rewritten, up
        to ``max_repairs`` times.

        An attempt that stops at ``"reset"`` ends the loop at once: the world could
        not be put back, so nothing was learned about the candidate and rewriting it
        would only spend money to fail the same way. :attr:`Admission.unproved` marks
        that outcome so a caller can say "give me a way to restore this world"
        instead of "the model wrote bad code".

        Args:
            trajectory: The successful run to learn from.
            environment: Called once per attempt; returns the recorded environment
                put back to the trajectory's first screen, saying through
                :attr:`ReplayEnvironment.restored` whether it managed to.

        Returns:
            An :class:`Admission`. ``ok`` means - and only means - that this exact
            code ran to completion in the replayed environment, its verifier agreed
            and the critic agreed, and ``skill`` is what the store now holds.
            Otherwise nothing was stored and ``attempts`` says why.

        Raises:
            ProviderError: if a model call fails.
            BudgetExceeded: if the sandbox run exhausts an agent-level budget.
        """
        if not self._worth_keeping(trajectory):
            reason = (
                "the run did not succeed"
                if not trajectory.ok
                else f"the run is {len(trajectory.steps)} step(s) long; too short to be a skill"
            )
            log.info("skill.admit.skipped", run_id=trajectory.run_id, reason=reason)
            return Admission(ok=False, skill=None, reason=reason)

        messages: list[LLMMessage] = [LLMMessage(role="user", text=describe_trajectory(trajectory))]
        attempts: list[Attempt] = []
        for index in range(self._max_repairs + 1):
            generation = self._generate(messages)
            messages = list(generation.messages)
            attempt = self._attempt(index, generation, trajectory, environment)
            attempts.append(attempt)
            if attempt.ok and attempt.skill is not None:
                stored = self._store.put(
                    _with_what_it_earned(attempt.skill, generation, trajectory)
                )
                log.info(
                    "skill.admit.stored",
                    run_id=trajectory.run_id,
                    name=stored.name,
                    domain=stored.domain,
                    version=stored.version,
                    repairs=index,
                )
                return Admission(
                    ok=True,
                    skill=stored,
                    attempts=tuple(attempts),
                    reason=f"admitted after {index} repair(s)",
                )
            log.info(
                "skill.admit.rejected",
                run_id=trajectory.run_id,
                attempt=index,
                stage=attempt.stage,
                error=attempt.error,
            )
            if attempt.stage == "reset":
                # Nothing was judged, so there is nothing for the model to repair.
                log.warning("skill.admit.unrestorable", run_id=trajectory.run_id)
                return Admission(
                    ok=False,
                    skill=None,
                    attempts=tuple(attempts),
                    reason=f"nothing was stored: {attempt.error}",
                )
            if index < self._max_repairs:
                messages = [
                    *messages,
                    *_echo(generation.reply),
                    LLMMessage(role="user", text=_repair_brief(attempt)),
                ]
        last = attempts[-1]
        return Admission(
            ok=False,
            skill=None,
            attempts=tuple(attempts),
            reason=(
                f"rejected at {last.stage} after {len(attempts)} attempt(s) "
                f"({self._max_repairs} repair(s) allowed): {last.error}"
            ),
        )

    # -- getting a readable reply ----------------------------------------------------------

    def _generate(self, messages: Sequence[LLMMessage]) -> _Generation:
        """One draft, re-asking for the JSON object when the reply is unreadable.

        A re-ask is not a repair. It costs a model call and says so in the log, but
        it does not advance the repair counter, because the skill has not been judged
        - the reply merely could not be read. A reply that ran out of room gets a
        bigger cap on the way round; a reply that was simply chatty does not need one.

        Raises:
            ProviderError: if a model call fails.
        """
        conversation = list(messages)
        tokens = self._max_tokens
        for reask in range(self._max_format_retries + 1):
            response = self._llm.complete(
                conversation,
                system=self._prompt,
                max_tokens=tokens,
                temperature=self._temperature,
            )
            draft, unreadable = _parse_draft(
                response.text, truncated=response.stop_reason == "max_tokens"
            )
            readable = draft is not None or (unreadable is not None and not unreadable.format_only)
            if readable or reask == self._max_format_retries:
                return _Generation(draft, unreadable, response.text, tuple(conversation), reask)
            assert unreadable is not None  # `readable` is false only when it is set
            if unreadable.needs_room:
                tokens = min(tokens * 2, _MAX_REPLY_TOKENS)
            log.info(
                "skill.generate.reask",
                why=unreadable.why,
                reask=reask + 1,
                of=self._max_format_retries,
                max_tokens=tokens,
            )
            conversation = [
                *conversation,
                *_echo(response.text),
                LLMMessage(role="user", text=_FORMAT_NUDGE.format(why=unreadable.why)),
            ]
        raise AssertionError("unreachable: the loop returns on its last iteration")

    # -- one attempt -----------------------------------------------------------------------

    def _attempt(
        self,
        index: int,
        generation: _Generation,
        trajectory: Trajectory,
        environment: EnvironmentFactory,
    ) -> Attempt:
        """Everything that has to be true before a skill may be stored."""
        draft = generation.draft
        if draft is None:
            why = generation.unreadable.why if generation.unreadable else "nothing usable"
            return Attempt(index, "generation", False, error=why)

        candidate, error, hardening = self._build(draft, trajectory)
        if candidate is None:
            return Attempt(index, "structure", False, error=error, hardening=hardening)

        try:
            scan_code(candidate.code, what=f"skill {candidate.name!r}")
            if candidate.verifier_code:
                scan_code(candidate.verifier_code, what=f"the verifier for {candidate.name!r}")
        except SandboxViolation as exc:
            return Attempt(
                index, "sandbox", False, skill=candidate, error=str(exc), hardening=hardening
            )

        env = environment()
        before = _observe_at_rest(env)
        if candidate.precondition is not None:
            similarity = before.fingerprint.similarity(candidate.precondition)
            # Logged whether it passes or fails. A gate that only speaks when it
            # refuses cannot be calibrated: the number that mattered here was the one
            # from the runs that were REJECTED at 0.96, and nobody could see the ones
            # that passed to know how much room was left.
            log.info(
                "skill.admit.precondition",
                name=candidate.name,
                similarity=round(similarity, 3),
                required=self._min_similarity,
                ok=similarity >= self._min_similarity,
            )
            if similarity < self._min_similarity:
                return Attempt(
                    index,
                    "precondition" if env.restored else "reset",
                    False,
                    skill=candidate,
                    error=_not_at_the_start(
                        candidate, similarity, self._min_similarity, restored=env.restored
                    ),
                    hardening=hardening,
                )

        runner = SkillRunner(self._store, limits=self._limits)
        ctx = runner.context(
            env.controller,
            env.perceiver,
            graph=env.graph,
            domain=candidate.domain,
            limits=self._limits,
        )
        result = runner.run(candidate, dict(draft.example_args), ctx)
        if not result.ok:
            return Attempt(
                index,
                "execution",
                False,
                skill=candidate,
                error=result.error,
                trace=result.trace,
                result=result,
                hardening=hardening,
            )

        if self._passes_on_the_start_screen(runner, candidate, before, result.value, env.graph):
            return Attempt(
                index,
                "discrimination",
                False,
                skill=candidate,
                error=_INDISCRIMINATE.format(name=candidate.name),
                trace=result.trace,
                result=result,
                hardening=hardening,
            )

        after = _observe_at_rest(env)
        verdict = self._critic.judge(trajectory.task, before, after, candidate.docstring)
        if not verdict.ok:
            return Attempt(
                index,
                "critic",
                False,
                skill=candidate,
                error=f"the critic rejected the result: {verdict.reason}",
                trace=result.trace,
                result=result,
                verdict=verdict,
                hardening=hardening,
            )
        return Attempt(
            index,
            "critic",
            True,
            skill=candidate,
            trace=result.trace,
            result=result,
            verdict=verdict,
            hardening=hardening,
        )

    # -- is the verifier worth storing? ----------------------------------------------------

    def _passes_on_the_start_screen(
        self,
        runner: SkillRunner,
        candidate: Skill,
        before: Observation,
        value: Any,
        graph: GraphView | None,
    ) -> bool:
        """Whether ``candidate``'s verifier ALSO says yes to the screen the skill
        started from - in which case its yes about the end screen means nothing.

        The verifier is replayed exactly as the runner replays it, through the same
        ``SkillRunner.run``, so what is probed is what will be stored. Only two things
        differ: the world is :class:`_StartScreen`, the observation this attempt
        already took, and ``run`` is :data:`_PROBE_CODE`, which performs nothing and
        hands ``value`` - the real run's return value - to ``verify``.

        Args:
            runner: The runner the candidate was just executed through.
            candidate: The skill whose verifier is in question.
            before: The screen the skill started on, already observed by the gate.
            value: What ``run`` returned on the real screen, so a verifier that reads
                ``result`` is asked the same question it was asked for real.
            graph: The environment's site graph, so a verifier that consults it can.

        Returns:
            ``True`` only when the verifier ran to completion on the starting screen
            and returned a truthy value. Anything else - it returned false, it raised,
            it tried to ACT and the frozen controller refused, the probe blew a limit -
            is INCONCLUSIVE and answers ``False``, which admits.

            That asymmetry is the whole design. A false accept here costs what it
            always cost: the critic still has to agree before anything is stored. A
            false REJECT would destroy a correct skill over a verifier this probe
            could not run, which is the more expensive mistake and the one this
            project has already made once, with ``min_similarity`` at ``1.0``.
        """
        if not candidate.verifier_code:
            return False
        screen = _StartScreen(before)
        probe = replace(
            candidate,
            name=f"{candidate.name}{_PROBE_SUFFIX}",
            code=_PROBE_CODE,
            params={},
            requires=(),
        )
        try:
            result = runner.run(
                probe,
                {"result": value},
                runner.context(
                    screen,
                    screen,
                    graph=graph,
                    domain=candidate.domain,
                    limits=self._limits,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - an unrunnable probe proves nothing
            # Including BudgetExceeded, which SkillRunner.run re-raises. The probe has
            # its own fresh ledger, so one here is about the probe and not about the
            # agent's remaining budget; aborting a whole run over it would be wrong.
            log.info("skill.admit.discrimination.unrunnable", name=candidate.name, why=str(exc))
            return False
        log.info(
            "skill.admit.discrimination",
            name=candidate.name,
            passes_on_start=result.ok,
            why=result.error,
        )
        return result.ok

    # -- building a candidate --------------------------------------------------------------

    def _build(
        self, draft: _Draft, trajectory: Trajectory
    ) -> tuple[Skill | None, str | None, Hardening | None]:
        """Harden the draft and validate it into a :class:`Skill`.

        The precondition is the trajectory's FIRST screen: where the recording
        started is where the skill may be used, and the planner routes there.
        """
        hardening = harden(draft.code, trajectory, params=draft.params)
        params = {**dict(draft.params), **dict(hardening.added_params)}
        if hardening.changed:
            log.debug(
                "skill.harden.applied",
                name=draft.name,
                changes=len(hardening.changes),
                anchored=hardening.positions_anchored,
            )
        if hardening.positions_announced:
            # Not a rejection: the recording offered nothing to anchor these on, so
            # the skill keeps them and says so in its own trace. Worth a line in the
            # run log, because it is the honest measure of how thin a skill is.
            log.info(
                "skill.harden.positional",
                name=draft.name,
                lookups=hardening.positions_announced,
            )
        try:
            candidate = make_skill(
                name=draft.name,
                domain=trajectory.domain,
                summary=draft.summary,
                docstring=draft.docstring,
                code=hardening.code,
                params=params,
                requires=draft.requires,
                precondition=trajectory.steps[0].before.fingerprint if trajectory.steps else None,
                verifier_code=draft.verifier_code,
                provenance=Provenance(
                    trajectory_id=trajectory.run_id,
                    task_text=trajectory.task,
                    model=self._llm.name(),
                    created_at=utcnow(),
                ),
            )
        except SkillInvalid as exc:
            return None, f"{type(exc).__name__}: {exc}", hardening
        return candidate, None, hardening

    def _worth_keeping(self, trajectory: Trajectory) -> bool:
        """A failed run teaches nothing a skill can repeat, and a one-action run is
        not a procedure worth a name."""
        return trajectory.ok and len(trajectory.steps) >= self._min_steps
