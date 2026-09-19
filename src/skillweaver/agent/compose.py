"""Decomposing a task into skills the library already holds - one model call, checked.

The planner's warm path answers a task it has seen before with **no model call at
all**. This module is the next rung down: the task is new, but its *parts* are not.
"Pay the Acme invoice and archive it" is unknown; ``search_invoice``, ``pay_invoice``
and ``archive_invoice`` are all known. One cheap text call proposes the ordering, and
from then on the run is model-free again.

That single call is the only place in the fast path where a model speaks, which is
why everything here is built around not trusting it:

**The proposal is validated against the store before anything is performed.** A
model asked to pick from a list will, sooner or later, return a name that is not on
it - a skill it remembers from another site, a plausible-sounding invention, a
misspelling. :meth:`Composer.compose` resolves every name against
:class:`~skillweaver.contracts.SkillStore` and checks every argument against the
skill's declared parameters *before* the planner is given a plan. A proposal with
one bad name is rejected **whole**: the agent explores instead, which is slower and
correct, rather than executing the good half of a plan on a real screen and
discovering the rest mid-run.

**Declining is a valid answer.** ``{"steps": []}`` comes back as a
:class:`Decomposition` with no steps and a reason, and the planner falls through to
exploration. The library grows by exploring things it could not compose; a composer
that bluffs to look useful poisons that.

**The sequence is capped.** :data:`MAX_STEPS` bounds how much a single unverified
model answer may set in motion.

Routing between the chosen skills is deliberately *not* decided here. A stored skill
declares the screen it starts on but not the screen it ends on, so where step *k*
leaves the agent is a fact about the live world, not about the library. The planner
re-reads the screen before each step and routes to that step's precondition from
wherever it actually is - see :mod:`skillweaver.agent.planner`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from importlib import resources
from typing import Any

from skillweaver.contracts import (
    Fingerprint,
    GraphView,
    LLMClient,
    LLMMessage,
    Observation,
    Skill,
    SkillCall,
    SkillStore,
    TaskSpec,
    Usage,
)
from skillweaver.errors import ProviderError
from skillweaver.logging_ import get_logger
from skillweaver.skills.model import signature

__all__ = [
    "MAX_STEPS",
    "Composer",
    "Decomposition",
    "ScreenNames",
    "describe_screen",
    "load_template",
    "parse_proposal",
    "render_skills",
]

log = get_logger(__name__)

MAX_STEPS = 4
"""How many stored skills one unverified model answer may chain by default.

Four is enough for the errands a demo actually strings together ("search, open, pay,
confirm") and short enough that a wrong proposal is cheap to abandon. A task that
needs a longer chain is a task the library does not really know, so it belongs to the
explorer.
"""

_MAX_SCREEN_ELEMENTS = 12
"""Element texts shown to the composer. It is choosing between stored skills, not
clicking anything, so it needs the gist of the screen, not an accessibility tree."""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decomposition:
    """What the composer decided, and what it cost.

    Exactly one of two shapes:

    * ``steps`` non-empty and ``rejected is None`` - a validated chain. Every name
      exists in the store, every argument is declared by its skill, and the length is
      within the cap. It is safe to execute in this order.
    * ``steps`` empty and ``rejected`` a short reason - either the model declined or
      its proposal failed validation. Either way NOTHING has been performed, and the
      caller should explore.

    ``usage`` is the cost of the one model call (zero when no call was made, as when
    the library is empty). ``why`` is the model's own one-line justification, kept
    for the log and the dashboard. ``raw`` is the unparsed reply, kept because a
    rejected proposal is the most useful thing there is for improving the prompt.
    """

    steps: tuple[SkillCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    why: str = ""
    rejected: str | None = None
    raw: str = ""

    def __bool__(self) -> bool:
        """True when there is a chain to run."""
        return bool(self.steps)

    @property
    def names(self) -> tuple[str, ...]:
        """The skill names of the chain, in order."""
        return tuple(step.name for step in self.steps)


# --------------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------------


@cache
def load_template() -> str:
    """The decomposition prompt as shipped in ``prompts/compose.md``.

    Cached: it is a file that cannot change inside one run, and the composer is on
    the latency path.
    """
    return resources.files("skillweaver.agent.prompts").joinpath("compose.md").read_text("utf-8")


class ScreenNames:
    """Human names for screens, so the prompt can talk about them.

    A :class:`~skillweaver.contracts.Fingerprint` is a hash. Putting one in a prompt
    tells the model nothing it can reason about and invites it to treat the hash as
    meaningful, so this turns the fingerprints the site graph has labelled into the
    names a person uses - "the invoice list", "an open invoice" - and says plainly
    that the rest are screens the agent will route to by itself.

    Built from whatever :class:`~skillweaver.contracts.GraphView` the composer was
    given; with no graph, every screen is simply unnamed, which reads fine.
    """

    __slots__ = ("_labels",)

    def __init__(self, graph: GraphView | None = None, domain: str = "") -> None:
        self._labels: dict[str, str] = {}
        if graph is not None:
            for state in graph.states(domain):
                if state.label:
                    self._labels[state.fingerprint.value] = state.label

    def describe(self, fingerprint: Fingerprint | None) -> str:
        """The clause a skill line or a screen line uses for ``fingerprint``."""
        if fingerprint is None:
            return "Can start on any screen."
        label = self._labels.get(fingerprint.value)
        if label:
            return f"Starts on the `{label}` screen."
        return "Starts on one particular screen, which the agent routes to itself."

    def name(self, fingerprint: Fingerprint) -> str:
        """A short name for the screen the agent is on, for the "where am I" block."""
        label = self._labels.get(fingerprint.value)
        return f"the `{label}` screen" if label else "a screen with no name yet"


def render_skills(skills: Sequence[Skill], screens: ScreenNames | None = None) -> str:
    """The available skills as the prompt shows them: real signatures, nothing else.

    The model is told to answer only with names from this block, so this block must
    be the truth about the store - rendered from
    :func:`~skillweaver.skills.model.signature`, the same function that would build
    the call site, rather than from anything written by hand.
    """
    known = screens if screens is not None else ScreenNames()
    lines: list[str] = []
    for skill in skills:
        proven = (
            f" {skill.stats.successes}/{skill.stats.runs} runs succeeded."
            if skill.stats.runs
            else ""
        )
        lines.append(
            f"- `{signature(skill)}` - {skill.summary} {known.describe(skill.precondition)}{proven}"
        )
    return "\n".join(lines) if lines else "- (the library is empty for this site)"


def describe_screen(observation: Observation | None, screens: ScreenNames | None = None) -> str:
    """A few lines about where the agent is, or a note that it has not looked yet."""
    if observation is None:
        return "The agent has not looked at the screen yet."
    known = screens if screens is not None else ScreenNames()
    texts = [e.text.strip() for e in observation.elements if e.text.strip()]
    head = ", ".join(f"`{t}`" for t in texts[:_MAX_SCREEN_ELEMENTS])
    parts = [f"On {known.name(observation.fingerprint)}."]
    if observation.url:
        parts.append(f"URL `{observation.url}`.")
    parts.append(f"Visible: {head}." if head else "Nothing legible on it.")
    return " ".join(parts)


def _render(template: str, **values: str) -> str:
    """Fill ``{{NAME}}`` placeholders. Deliberately not a template engine: the prompt
    is a document a human edits, and the only thing it needs is substitution."""
    text = template
    for name, value in values.items():
        text = text.replace("{{" + name + "}}", value)
    return text


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def parse_proposal(reply: str) -> tuple[list[dict[str, Any]], str, str | None]:
    """``(raw steps, why, rejection reason)`` from a model reply.

    Tolerant about the wrapping a chat model adds - a code fence, a sentence before
    the JSON - and strict about the shape inside it. Nothing here touches the store:
    this only establishes that the model said something of the right form.
    """
    text = reply.strip()
    if not text:
        return [], "", "the model returned nothing"
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        found = _OBJECT.search(text)
        if found is None:
            return [], "", "the reply contained no JSON object"
        text = found.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], "", f"the reply was not valid JSON: {exc}"
    if not isinstance(data, Mapping):
        return [], "", f"expected a JSON object, got {type(data).__name__}"

    why = str(data.get("why", "")).strip()
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        return [], why, f"'steps' must be a list, got {type(steps).__name__}"
    for position, step in enumerate(steps):
        if not isinstance(step, Mapping):
            return [], why, f"step {position + 1} is not an object"
        if not isinstance(step.get("skill"), str) or not step["skill"].strip():
            return [], why, f"step {position + 1} has no 'skill' name"
        args = step.get("args", {})
        if not isinstance(args, Mapping):
            return [], why, f"step {position + 1} has non-object 'args'"
    return [dict(s) for s in steps], why, None


# --------------------------------------------------------------------------------------
# The composer
# --------------------------------------------------------------------------------------


class Composer:
    """Turns a task no single skill covers into a chain of skills that exist.

    Args:
        llm: The model that proposes the ordering. Called at most ONCE per
            :meth:`compose`, and not at all when the library holds nothing for the
            task's domain.
        store: The library the proposal is checked against. This is the authority:
            what the model says a skill is called does not matter, only what is in
            here does.
        graph: Optional site graph, used only to give screens their human names in
            the prompt. It is never routed over here; the planner does that.
        max_steps: Cap on the chain length (see :data:`MAX_STEPS`).
        max_tokens: Reply cap. A decomposition is a few dozen tokens of JSON; a
            model rambling past this has not answered the question.
        template: Prompt override, for tests and experiments. ``None`` loads
            ``prompts/compose.md``.
    """

    __slots__ = ("_graph", "_llm", "_max_steps", "_max_tokens", "_store", "_template")

    def __init__(
        self,
        llm: LLMClient,
        store: SkillStore,
        *,
        graph: GraphView | None = None,
        max_steps: int = MAX_STEPS,
        max_tokens: int = 1024,
        template: str | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        self._llm = llm
        self._store = store
        self._graph = graph
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._template = template

    def __repr__(self) -> str:
        return f"Composer(llm={self._llm.name()!r}, max_steps={self._max_steps})"

    @property
    def max_steps(self) -> int:
        """The chain-length cap this composer enforces."""
        return self._max_steps

    def compose(self, task: TaskSpec, observation: Observation | None = None) -> Decomposition:
        """Propose and validate a chain of stored skills for ``task``.

        Costs exactly one model call when the library holds skills for
        ``task.domain``, and zero when it does not. Performs nothing, touches no
        controller, and returns only chains that passed every check in
        :meth:`_validate`.

        Raises:
            ProviderError: if the model call fails. Nothing has been performed.
        """
        available = self._store.list(domain=task.domain)
        if not available:
            return Decomposition(rejected=f"no skills stored for domain {task.domain!r}")

        prompt = self.render(task, available, observation)
        try:
            reply = self._llm.complete(
                [LLMMessage(role="user", text=prompt)],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
        except ProviderError:
            raise
        except Exception as exc:  # a third-party client can raise anything
            raise ProviderError(f"decomposition call failed: {exc}") from exc

        usage = _charge_one_call(reply.usage)
        proposed, why, bad = parse_proposal(reply.text)
        if bad is not None:
            return self._reject(bad, usage, why, reply.text, task)

        index = {skill.name: skill for skill in available}
        steps, bad = self._validate(proposed, index, task.domain)
        if bad is not None:
            return self._reject(bad, usage, why, reply.text, task)

        log.info(
            "compose.ok",
            task=task.text,
            domain=task.domain,
            steps=" -> ".join(s.name for s in steps),
            calls=usage.calls,
        )
        return Decomposition(steps=steps, usage=usage, why=why, raw=reply.text)

    def render(
        self,
        task: TaskSpec,
        available: Sequence[Skill],
        observation: Observation | None = None,
    ) -> str:
        """The exact prompt :meth:`compose` would send. Public so a human can read
        what the model was actually asked, which is most of debugging a bad chain."""
        screens = ScreenNames(self._graph, task.domain)
        return _render(
            self._template if self._template is not None else load_template(),
            TASK=task.text,
            DOMAIN=task.domain,
            PARAMS=json.dumps(dict(task.params), indent=2, default=str, sort_keys=True),
            SCREEN=describe_screen(observation, screens),
            SKILLS=render_skills(available, screens),
            MAX_STEPS=str(self._max_steps),
        )

    # -- validation ----------------------------------------------------------------

    def _validate(
        self, proposed: Sequence[Mapping[str, Any]], index: Mapping[str, Skill], domain: str
    ) -> tuple[tuple[SkillCall, ...], str | None]:
        """Check a parsed proposal against the library.

        Returns the calls to make, or ``(), reason`` for a proposal that must not be
        executed. The checks are all-or-nothing on purpose: a chain is a sequence of
        real actions on a real screen, and stopping halfway through one leaves the
        world somewhere nobody planned for.
        """
        if not proposed:
            return (), "the model declined: no chain of stored skills covers this task"
        if len(proposed) > self._max_steps:
            return (), f"{len(proposed)} steps proposed, at most {self._max_steps} allowed"

        calls: list[SkillCall] = []
        for position, step in enumerate(proposed, start=1):
            name = str(step["skill"]).strip()
            skill = index.get(name)
            if skill is None:
                known = ", ".join(sorted(index)) or "(none)"
                return (), (
                    f"step {position} names skill {name!r}, which is not in the library "
                    f"for {domain!r}; known skills are: {known}"
                )
            args = dict(step.get("args", {}))
            bad = _check_args(skill, args, position)
            if bad is not None:
                return (), bad
            calls.append(SkillCall(name=skill.name, domain=skill.domain, args=args))
        return tuple(calls), None

    def _reject(
        self, reason: str, usage: Usage, why: str, raw: str, task: TaskSpec
    ) -> Decomposition:
        log.info("compose.rejected", task=task.text, domain=task.domain, reason=reason)
        return Decomposition(usage=usage, why=why, rejected=reason, raw=raw)


def _charge_one_call(reported: Usage) -> Usage:
    """The usage of the one ``complete`` that :meth:`Composer.compose` just made.

    The planner's headline number is a model-CALL count, so it is counted here, where
    the call demonstrably happened, rather than read back out of whatever the provider
    chose to report. A client that does not fill in ``calls`` - a cassette replaying a
    recording, a stub, a fake - would otherwise make a composed run indistinguishable
    from a warm one, which is the single measurement this project cannot afford to get
    wrong. Token counts and cost stay exactly as reported.
    """
    return reported if reported.calls >= 1 else replace(reported, calls=1)


def _check_args(skill: Skill, args: Mapping[str, Any], position: int) -> str | None:
    """``None`` when ``args`` fit ``skill``'s declared parameters, else why not.

    An undeclared argument is as fatal as a missing one: ``run(ctx, **args)`` would
    raise ``TypeError`` inside the sandbox, which is a real failure recorded against
    a skill that did nothing wrong.
    """
    declared = dict(skill.params)
    unknown = sorted(set(args) - set(declared))
    if unknown:
        return (
            f"step {position} passes {', '.join(repr(u) for u in unknown)} to "
            f"{skill.name!r}, which takes {signature(skill)}"
        )
    missing = sorted(
        name
        for name, schema in declared.items()
        if name not in args and not (isinstance(schema, Mapping) and "default" in schema)
    )
    if missing:
        return (
            f"step {position} omits required argument(s) "
            f"{', '.join(repr(m) for m in missing)} of {signature(skill)}"
        )
    return None
