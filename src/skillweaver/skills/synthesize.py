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
5. **The critic.** :class:`~skillweaver.contracts.Critic` judges the screen before
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from skillweaver.contracts import (
    Action,
    Controller,
    Critic,
    Element,
    GraphView,
    LLMClient,
    LLMMessage,
    Observation,
    Perceiver,
    Provenance,
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
from skillweaver.skills.api import SkillLimits, describe_action
from skillweaver.skills.model import SkillInvalid, make_skill
from skillweaver.skills.refactor import Hardening, harden
from skillweaver.skills.sandbox import SkillRunner, scan_code

__all__ = [
    "MIN_PRECONDITION_SIMILARITY",
    "PROMPT",
    "Admission",
    "Attempt",
    "ReplayEnvironment",
    "Stage",
    "Synthesizer",
    "load_prompt",
]

log = get_logger(__name__)

PROMPT = "synthesize.md"
"""The generation prompt, next to this project's other prompts in
``skillweaver/agent/prompts/``. It states the published ``ctx`` surface exactly,
forbids imports, and demands a verifier; :func:`load_prompt` reads it."""

Stage = Literal[
    "generation", "structure", "sandbox", "reset", "precondition", "execution", "critic"
]
"""Where an attempt stopped. Everything before ``reset`` is decided without touching
the environment at all.

``"reset"`` and ``"precondition"`` both mean the environment was not on the recorded
starting screen, and they are not the same finding. ``"precondition"`` is a verdict
on the SKILL: the world was genuinely put back and the screen still does not match.
``"reset"`` is a verdict on the HARNESS: nothing put the world back, so the candidate
was neither proved nor disproved and no amount of rewriting it would help."""

_MAX_ELEMENTS = 18
"""Elements described per recorded screen. Enough to write a lookup against, short
enough that a long list does not bury the ones that were acted on."""

_MAX_TEXT = 80

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

What was measured, and where
----------------------------

All of it on 2026-09-19, Chromium at 1280x800 - headless except where a paragraph
says otherwise - through the SHIPPED pipeline (``YoloDetector`` + ``RapidOcrReader``
+ ``StateFingerprinter``), which emits about 25 parts for a live page, so every score
below is an exact ``n/25``.

The numbers are taken on **the path the gate actually walks**: a COLD first
observation, which is what the recording captures, against a WARM re-navigation in
the SAME browser, which is what ``navigating_environment`` hands the gate. Three
trials a page, same rendering mode on both sides:

===================================  ==========================  ======================
page                                 cold record -> warm re-run  what moved
===================================  ==========================  ======================
en.wikipedia.org/wiki/Main_Page      0.880, 1.000, 1.000         the right rail hydrating
en.wikipedia.org/wiki/Ada_Lovelace   1.000, 0.920, 0.040         a reflow; then a banner
docs.python.org/3/library/json.html  1.000, 1.000, 1.000         nothing
the sandbox site at ``/``            1.000, 1.000, 1.000         nothing
===================================  ==========================  ======================

The 0.040 is not noise and not a near miss: a Wikimedia fundraising notice arrived
between the two captures and pushed the whole article down the viewport. That is a
genuinely different screen and rejecting it is the right answer, which is the point
of keeping a threshold at all.

Then the same question asked of three trajectories this gate had actually recorded on
live Wikipedia, re-navigated three times each - nine re-runs, no model calls. Six
scored **0.840**, every one of which the old ``1.0`` threw away and every one of which
now runs. The other three were the fundraising banner again (0.040, 0.040, 0.200) and
are still refused. Those recordings were made through the CLI, which opens a visible
window, and the re-runs were headless, so 0.840 is a cross-mode score as well; it is
the lowest same-page number anywhere in this calibration and it sets the floor.

And the rejections that have to keep working - different screens built from the same
template, which is the only kind worth testing:

==========================================  =====
pair                                        score
==========================================  =====
two Wikipedia revision-history pages        0.280
two Wikipedia category listings             0.120
two Wikipedia search-result pages           0.080
two Wikipedia stub articles                 0.000
two docs.python.org stdlib pages            0.000
the Main Page against an article            0.040
------------------------------------------  -----
``same_layout_different_content``           0.500
==========================================  =====

The last row is the contrived worst case committed under
``tests/fixtures/shots/pairs``: one invoice list for two different accounts,
identical URL, identical chrome, identical layout, every row different. Nothing
live came near it, and it is the row that sets the floor - admitting it would let
the gate prove a skill against the wrong account's data and store the result.

Why 0.62
--------

Same-page bottoms out at **0.840** and different-screen tops out at **0.280** live
and **0.500** contrived, so the cut belongs in ``(0.500, 0.840)``. 0.62 sits there
with 0.12 of margin below and 0.22 above.

It is deliberately nearer the bottom of that window, because the two mistakes do not
cost the same. A false REJECT is total: the skill was written correctly and is
destroyed, which is the defect this constant exists to fix. A false ACCEPT costs one
sandbox execution and nothing else - the precondition is the third of five gates, and
a candidate run against the wrong screen still has to execute, satisfy its own
verifier and satisfy the critic before anything is stored.

That it lands on :data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD` is
worth saying out loud rather than leaving as a coincidence. That constant was
calibrated independently, on a different corpus, for the graph's question; this is
the same question - "am I looking at the screen I recorded?" - so the project holds
ONE number for it, and the measurements above are this gate's own evidence for it
rather than a borrowing.

Headed and headless are NOT made comparable
-------------------------------------------

Two FRESH browsers on one page, one visible and one headless, score 0.440 (Main Page)
and 0.680 (``json.html``). The recorded precondition still does not record which mode
produced it, for three reasons.

The gate never compares across modes. ``orchestrator._open_world`` builds ONE
controller and ``navigating_environment`` re-navigates in THAT controller, so the
recording and the re-run are always the same window, and all three live ``learn``
runs behind this constant scored their precondition at 1.000 in-process. The gap is
real and is simply not on this path.

Nor does it need a field to survive being off that path. Re-navigating those same
recordings from the other mode scored 0.840 - inside the window, six times out of
six - because a re-navigation is not a cold first paint. The 0.440 is two races
compounding, a different renderer AND a rail that had not hydrated; separate them and
the mode alone does not move a page out of the window.

Where it IS real - a stored skill replayed later by another process in another mode -
the precondition is the wrong place to carry the answer. ``Controller.describe()``
already reports ``headed``, so a caller that wants to refuse a cross-mode replay can
compare two strings without every stored skill in the library growing a field, and
``Fingerprint`` is shared surface besides.

And it would make the identity worse. A screen is the same screen whoever rendered
it; 0.440 is the fingerprinter correctly reporting that these two renderers produce
measurably different pixels. The answer is to run one renderer - ``eval/wikipedia.yaml``
already says headless, for exactly this reason - not to teach the identity to ignore
a difference it was right to notice.
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


def _describe_element(element: Element) -> str:
    text = element.text.strip().replace("\n", " ")
    if len(text) > _MAX_TEXT:
        text = text[: _MAX_TEXT - 1] + "…"
    box = element.box
    return f"  - {element.kind.value} {text!r} at ({box.x}, {box.y}) {box.w}x{box.h}"


def _describe_screen(observation: Observation, label: str) -> str:
    elements: Sequence[Element] = observation.elements[:_MAX_ELEMENTS]
    url = observation.url or "none"
    lines = [f"{label} (url: {url}, screen id: {observation.fingerprint.value})"]
    lines.extend(_describe_element(e) for e in elements)
    if len(observation.elements) > _MAX_ELEMENTS:
        lines.append(f"  - ... and {len(observation.elements) - _MAX_ELEMENTS} more elements")
    return "\n".join(lines)


def _describe_target(observation: Observation, action: Action) -> str | None:
    """Which element the action landed on, named the way a skill can name it again.

    The recording says ``click (236, 121)`` and the skill may not write that down -
    rule 6 of the prompt, and rightly, because a coordinate is a screenshot. But
    without this line the model is left guessing WHICH element that coordinate was,
    and a wrong guess is admitted or rejected by luck.

    It was rejected. In a live run the model wanted a row "from Dana Whitfield",
    searched for that text, found none - OCR never reads the sender names on this
    page - and archived a different message instead. So the element is named twice
    over: by its text where there is any, and always by its kind and its place in
    reading order, which is a handle a skill CAN reproduce from pixels when text
    fails. ``None`` for an action with no point (typing, a key press).
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
    text = element.text.strip().replace("\n", " ")
    if len(text) > _MAX_TEXT:
        text = text[: _MAX_TEXT - 1] + "\u2026"
    kind = element.kind.value
    reads = f"reading {text!r}" if text else "with NO readable text"
    return (
        f"  this landed on the {kind} {reads}, which is {kind} number {ordinal} "
        f"of {len(same_kind)} in reading order"
    )


def describe_trajectory(trajectory: Trajectory) -> str:
    """The recorded run as the text the model is asked to write a skill from.

    Deterministic for a given trajectory - no timestamps, no ordering by dict - so
    the same run produces the same request, which is what makes an LLM cassette
    replay and a repair prompt diffable.
    """
    parts = [
        f"TASK: {trajectory.task}",
        f"DOMAIN: {trajectory.domain}",
        f"STEPS: {len(trajectory.steps)}",
        "",
    ]
    if trajectory.steps:
        parts.append(_describe_screen(trajectory.steps[0].before, "STARTING SCREEN"))
        parts.append("")
    for step in trajectory.steps:
        parts.append(f"STEP {step.index}: {describe_action(step.action)}")
        target = _describe_target(step.before, step.action)
        if target:
            parts.append(target)
        if step.note:
            parts.append(f"  reason given at the time: {step.note}")
        parts.append(_describe_screen(step.after, "  screen after"))
        parts.append("")
    if trajectory.steps:
        parts.append(_describe_screen(trajectory.steps[-1].after, "FINAL SCREEN (the goal)"))
    parts.append("")
    parts.append("Write the skill for this task as the JSON object described above.")
    return "\n".join(parts)


def _repair_brief(attempt: Attempt) -> str:
    """What the model is shown after a rejection: the stage, the error, the trace."""
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
    lines.append("")
    lines.append(
        "Fix exactly this and return the same JSON object shape again. "
        "Keep what was working; change what the error names."
    )
    return "\n".join(lines)


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
        the recorded starting screen, and judged by the critic. A rejection is fed
        back to the model with its error and trace and the skill is rewritten, up to
        ``max_repairs`` times.

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
                stored = self._store.put(attempt.skill)
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
        before = env.perceiver.observe(env.controller)
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

        after = env.perceiver.observe(env.controller)
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
