"""A deterministic operator that stands where the model stands.

The point of this module is to make the whole agent measurable without a model.

``skillweaver`` calls a model in four places - to propose the next move while
exploring, to judge a move, to write a skill from a finished run, and to bind a
stored skill's parameters. :class:`ScriptedOperator` is a
:class:`~skillweaver.contracts.LLMClient` that answers all four, so the browser, the
perception stack, the sandbox, the site graph, the admission gate and the whole
cold-versus-warm decision run for real while the one non-deterministic component is
replaced by something repeatable.

What it can and cannot tell you
-------------------------------

It measures **the system**: how long a screen takes to read, whether a synthesized
skill really replays, whether the warm path finds and binds the right skill, whether
ground truth agrees at the end. It does NOT measure **the model**: how well a real
computer-use model explores an unfamiliar screen, or whether it writes good skill
code.

The consequence for a timing comparison is one-directional and worth stating plainly:
a real cold run additionally waits on the model - seconds per move, several moves per
task - while a warm run consults no model at all. Replacing the model with something
instant removes time from the COLD side only, so a cold-versus-warm speedup measured
this way is a **lower bound** on the speedup with a real model, never an inflated one.

How the operator decides
------------------------

It reads the same prompt the model would, and nothing else. It has no handle on the
controller, the browser, the sandbox server or the referee: the element ids it acts by
are parsed out of the prompt's own element listing, which is exactly the grounding the
real model is held to. Two consequences that matter:

* if perception fails to find a control, the operator cannot click it either, so a
  detection or OCR failure shows up as a failed run rather than being papered over;
* the run history quoted back in the prompt is how it knows which step it is on, so a
  move the critic rejected is retried rather than skipped.

A playbook is a list of steps for one task. Each step is a function of the elements
currently on screen and of how many times it has already failed, and returns the move
to make - which lets a step scroll first and click second when the control it wants is
below the fold, without any of the surrounding machinery knowing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillweaver.contracts import LLMMessage, LLMResponse, Usage

__all__ = [
    "Element",
    "Playbook",
    "ScriptedOperator",
    "Step",
    "click",
    "find",
    "squash",
    "key",
    "scroll",
    "type_text",
]


# --------------------------------------------------------------------------------------
# What the operator can see: the prompt's own element listing, parsed back
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """One line of the prompt's element listing, as the operator reads it.

    The id is the only thing it may act by - the same rule the real model is given -
    and the geometry is here so a step can prefer, say, the leftmost checkbox on the
    row whose text matched.
    """

    id: str
    kind: str
    text: str
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


_ELEMENT_LINE = re.compile(
    r"^\s*\[(?P<id>[^\]]+)\]\s+(?P<kind>\w+)\s*(?P<text>'.*?'|\"(?:[^\"]*)\")?\s*"
    r"at\s+\((?P<x>-?\d+),(?P<y>-?\d+)\)\s+(?P<w>\d+)x(?P<h>\d+)\s*$"
)


def parse_elements(prompt: str) -> list[Element]:
    """Every element the prompt listed, in the order it listed them."""
    found: list[Element] = []
    for line in prompt.splitlines():
        match = _ELEMENT_LINE.match(line)
        if match is None:
            continue
        raw = match.group("text") or ""
        text = raw[1:-1] if len(raw) >= 2 else ""
        found.append(
            Element(
                id=match.group("id"),
                kind=match.group("kind"),
                text=text,
                x=int(match.group("x")),
                y=int(match.group("y")),
                w=int(match.group("w")),
                h=int(match.group("h")),
            )
        )
    return found


def squash(text: str) -> str:
    """Text with case, spaces and punctuation-ish noise removed.

    Real OCR of a rendered page runs words together and picks up stray glyphs -
    ``Invoice 4471isready``, ``Export csV``, ``QSearch mail`` - so a comparison that
    respects spacing finds nothing. Matching on the letters and digits alone is what
    the element index's fuzzy lookup does for a skill, and the operator needs the same
    tolerance for the same reason.
    """
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def find(
    elements: Sequence[Element],
    wanted: str,
    *,
    kind: str | None = None,
    nth: int = 0,
    within: tuple[int, int, int, int] | None = None,
    exclude: str | None = None,
) -> Element | None:
    """The ``nth`` element whose text contains ``wanted``, or ``None``.

    Args:
        elements: What is on screen.
        wanted: Text to look for, compared with spacing and case ignored.
        kind: Restrict to one element kind.
        nth: Which match to take, in reading order.
        within: ``(x0, y0, x1, y1)`` bounding region the element's centre must be in -
            how a step says "the nav bar one, not the heading with the same word".
        exclude: Skip matches whose text also contains this.
    """
    needle = squash(wanted)
    skip = squash(exclude) if exclude else None
    hits = []
    for element in elements:
        if kind is not None and element.kind != kind:
            continue
        squashed = squash(element.text)
        if needle not in squashed:
            continue
        if skip and skip in squashed:
            continue
        if within is not None:
            x0, y0, x1, y1 = within
            if not (x0 <= element.cx <= x1 and y0 <= element.cy <= y1):
                continue
        hits.append(element)
    return hits[nth] if len(hits) > nth else None


# --------------------------------------------------------------------------------------
# What a step returns
# --------------------------------------------------------------------------------------


def click(element: Element, **extra: Any) -> dict[str, Any]:
    """Click one element, by the id the prompt gave it."""
    return {"action": {"kind": "click", "element_id": element.id}, **extra}


def type_text(text: str, **extra: Any) -> dict[str, Any]:
    """Type into whatever the last click focused."""
    return {"action": {"kind": "type_text", "text": text}, **extra}


def key(*keys: str, **extra: Any) -> dict[str, Any]:
    """Press a key or a chord."""
    return {"action": {"kind": "press_key", "keys": list(keys)}, **extra}


def scroll(element: Element | None, dy: int, **extra: Any) -> dict[str, Any]:
    """Scroll over an element, or over the middle of the screen when given ``None``."""
    action: dict[str, Any] = {"kind": "scroll", "dy": dy}
    if element is not None:
        action["element_id"] = element.id
    return {"action": action, **extra}


Step = Callable[[Sequence[Element], int], dict[str, Any] | None]
"""One move of a playbook.

Takes what is on screen and how many times this step has already been rejected, and
returns the move - or ``None`` when it cannot find what it needs, which the operator
reports as an honest failure rather than clicking something else.
"""


@dataclass(frozen=True, slots=True)
class Playbook:
    """How one task is done, and what is learned from having done it.

    Attributes:
        task: The task sentence, matched against the prompt's ``TASK:`` line.
        steps: The moves, in order. The last one carries ``done``.
        skill: The JSON object the synthesizer is asked for - the skill this run
            teaches. ``None`` means this task teaches nothing, and the report will
            show its warm runs exploring again.
        bind: The composer's reply for the plain-English warm path: which stored
            skill to run and with what arguments read out of the sentence.
    """

    task: str
    steps: tuple[Step, ...]
    skill: dict[str, Any] | None = None
    bind: dict[str, Any] | None = None
    notes: str = ""


# --------------------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------------------

_ROLES = (
    ("explore", "acting half of a computer-use agent"),
    ("critic", "You are the critic"),
    ("synthesize", "Write a reusable skill"),
    ("compose", "You are the composer"),
)
"""How each of the four questions announces itself.

Matched against the system prompt AND the user turn, because they do not all arrive
the same way: the composer has no system prompt at all and puts its instructions in
the message. Missing that made every composer call look like a judging call, so the
composer got a verdict where it wanted a plan, declined, and no warm run with
parameters to bind ever used the library.
"""

_HISTORY = re.compile(r"^\s*\d+\.\s+(?P<body>.*?)\s*->\s*(?P<verdict>ok|FAILED|performed)", re.M)


@dataclass
class ScriptedOperator:
    """An :class:`~skillweaver.contracts.LLMClient` that plays every model role.

    Args:
        playbooks: One per task, keyed by the task sentence.
        model: The name reported to anything that records which model ran.

    Attributes:
        calls: How many replies it has given, by role. A benchmark asserts against
            this: a warm run that consulted the "model" shows up here.
    """

    playbooks: dict[str, Playbook] = field(default_factory=dict)
    model: str = "scripted-operator"
    calls: dict[str, int] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    debug_to: Path | None = None
    """Write the next prompt here and then stop. For working out why a playbook did
    not fire, which is otherwise invisible: the operator only sees a string."""

    def name(self) -> str:
        return self.model

    def total_usage(self) -> Usage:
        """No tokens and no dollars, because nothing was sent anywhere.

        Reported as a real zero rather than an estimate: the benchmark's own summary
        says the model was simulated, and inventing a plausible token count here would
        put a number in the report that nothing measured.
        """
        return Usage(input_tokens=0, output_tokens=0, calls=sum(self.calls.values()), cost_usd=0.0)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Any = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Answer whichever of the four questions this prompt is asking."""
        prompt = "\n\n".join(m.text for m in messages if m.text)
        role = self._role(system or "", prompt)
        self.calls[role] = self.calls.get(role, 0) + 1
        if self.debug_to is not None:
            with self.debug_to.open("a", encoding="utf-8", errors="replace") as sink:
                sink.write(f"\n=== call {sum(self.calls.values())} role={role} ===\n{prompt}\n")
        answer = {
            "explore": self._explore,
            "critic": self._judge,
            "synthesize": self._write_skill,
            "compose": self._bind,
        }[role](prompt)
        return LLMResponse(text=answer, usage=Usage(calls=1))

    @staticmethod
    def _role(system: str, prompt: str) -> str:
        for role, marker in _ROLES:
            if marker in system or marker in prompt:
                return role
        return "critic"  # the safest default: judging is the only role with no side effect

    # -- exploring ---------------------------------------------------------------------

    def _explore(self, prompt: str) -> str:
        """The next move, from the playbook for whatever task this prompt names."""
        playbook = self._playbook_for(prompt)
        if playbook is None:
            return json.dumps(
                {
                    "thought": "no playbook covers this task",
                    "expect": "nothing",
                    "done": False,
                }
            )

        elements = parse_elements(prompt)
        index, failures = _progress(prompt)
        if index >= len(playbook.steps):
            return json.dumps(
                {
                    "thought": "every step of the playbook has been performed",
                    "expect": "the task is complete",
                    "done": True,
                }
            )

        move = playbook.steps[index](elements, failures)
        if move is None:
            return json.dumps(
                {
                    "thought": (
                        f"step {index + 1} needs a control that is not on this screen; "
                        "perception did not find it"
                    ),
                    "expect": "nothing, because the control could not be located",
                    "done": False,
                }
            )
        move.setdefault("thought", f"playbook step {index + 1} of {len(playbook.steps)}")
        move.setdefault("expect", "the screen advances to the next step")
        move.setdefault("done", index == len(playbook.steps) - 1)
        return json.dumps(move)

    def _playbook_for(self, prompt: str) -> Playbook | None:
        match = re.search(r"^TASK:\s*(?P<task>.+)$", prompt, re.M)
        if match is None:
            return None
        asked = squash(match.group("task"))
        for sentence, playbook in self.playbooks.items():
            if squash(sentence) == asked:
                return playbook
        self.unmatched.append(match.group("task"))
        return None

    # -- judging -----------------------------------------------------------------------

    @staticmethod
    def _judge(prompt: str) -> str:
        """Always "yes", and that is deliberate rather than lazy.

        A critic is only asked at all once the deterministic checks could not decide,
        and the operator has no way to look at the screen that is any better than those
        checks. Agreeing lets the run proceed to the point where the thing that CAN
        decide - the harness's referee, reading the application's own state, which the
        agent never sees - scores it. A benchmark run therefore reports the referee's
        verdict as the truth and any disagreement with the agent's own claim by name,
        which is exactly the arrangement the harness was built for.
        """
        return json.dumps(
            {
                "ok": True,
                "evidence": "the scripted operator does not second-guess the deterministic checks",
                "reason": "deferring to ground truth, which the harness checks independently",
                "confidence": 0.5,
            }
        )

    # -- writing a skill ----------------------------------------------------------------

    def _write_skill(self, prompt: str) -> str:
        playbook = self._playbook_for_run(prompt)
        if playbook is None or playbook.skill is None:
            return "this run taught nothing worth storing"
        return "```json\n" + json.dumps(playbook.skill) + "\n```"

    def _bind(self, prompt: str) -> str:
        # The composer states the task under a heading rather than after a "TASK:"
        # label, so it is recognised the tolerant way - by finding a playbook's own
        # sentence somewhere in the prompt.
        playbook = self._playbook_for_run(prompt)
        if playbook is None or playbook.bind is None:
            return json.dumps({"steps": [], "why": "no stored skill covers this task"})
        return json.dumps(playbook.bind)

    def _playbook_for_run(self, prompt: str) -> Playbook | None:
        """The playbook whose sentence appears in a prompt that does not label it.

        The longest match wins, so a task sentence that contains another one cannot be
        answered with the shorter task's plan.
        """
        found = squash(prompt)
        best: Playbook | None = None
        for sentence, playbook in self.playbooks.items():
            if squash(sentence) in found and (best is None or len(sentence) > len(best.task)):
                best = playbook
        return best


def _progress(prompt: str) -> tuple[int, int]:
    """``(step to perform, consecutive failures at it)`` read from the run history.

    The prompt quotes what this run has done, each line ending in how it was judged.
    Counting the accepted moves gives the step to perform next, and counting the
    rejected ones since the last accepted move tells the step how hard it is finding
    this screen - which is how a step knows to scroll before it clicks again.

    Only the last :data:`RECENT_MOVES` lines are quoted, so a run longer than that
    would undercount. Every playbook here is far shorter, and a benchmark whose
    playbook outgrows the window would show up as a task that never finishes rather
    than as a wrong number.
    """
    done = 0
    failures = 0
    for match in _HISTORY.finditer(prompt):
        if match.group("verdict") == "ok":
            done += 1
            failures = 0
        else:
            failures += 1
    # A refused answer is counted too: it never reached the screen, but it is still a
    # sign this step is not finding what it wants.
    failures += len(re.findall(r"^\s*\d+\.\s+\(answer refused\)", prompt, re.M))
    return done, failures
