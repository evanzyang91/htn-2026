"""The warm fast path: doing a known task with no model in the action loop.

This is the module the project's claim rests on. Everything else - the controllers,
the perception stack, the sandbox, the site graph, the synthesizer - exists so that
the *second* time the agent is asked to do something, it can do it like this:

    retrieve a stored skill -> route to the screen it starts on -> run it -> verify

and consult **no model for a single click**. Not a cheaper model, not a smaller
prompt: none. A warm run's cost is a screenshot, some graph arithmetic and the
skill's own Python.

That is a claim, so it is measured rather than asserted. Every warm run returns a
:class:`~skillweaver.contracts.RunOutcome` whose ``spend.llm_calls`` is the number of
model calls the planner caused, and ``tests/agent/test_planner.py`` asserts it is
exactly zero. The structure backs the number up: a :class:`Planner` has no
:class:`~skillweaver.contracts.LLMClient`. The only object here that can reach a
model is the optional :class:`~skillweaver.agent.compose.Composer`, and it is touched
only after single-skill retrieval has already failed.

What "warm" costs
-----------------

One :class:`~skillweaver.contracts.Perceiver` observation before each skill and one
after the last, because the critic needs a *before* and an *after* and the router
needs to know where the agent is. Route actions are replayed straight from the site
graph. Retrieval may embed the task text if an embedder is configured - that is a
retrieval cost, paid once, before the loop, and it is not a call in the action loop.

Plain English, still model-free
-------------------------------

A caller who supplies ``TaskSpec.params`` hands the planner its arguments and nothing
has to be worked out. A caller who just says what they want - "Search Wikipedia for
machine learning" - used to cost one model call, because the composer was the only
thing here that could read a sentence.

It no longer does, for the one case that is not a guess. A stored skill remembers the
sentence it was learned from, so when the new task is that sentence with one thing
changed, the change is the argument (:func:`_bind_from_text`). That path declines
loudly more often than it binds - see its guards - and every binding it does make is
logged as ``planner.bound_from_text`` with both sentences and the values derived, so
an argument this code invented is findable in one look. Everything downstream is
unchanged: a skill bound this way is routed to, run and verified exactly like one the
caller supplied arguments for.

Failing usefully
----------------

The planner's job includes knowing when it does not know. It returns ``None`` rather
than improvising, and leaves :attr:`Planner.last_failure` describing why, so the
caller can hand that context to an :class:`~skillweaver.contracts.Explorer` instead
of starting the task cold::

    outcome = planner.attempt(task, observation)
    if outcome is None:
        outcome = explorer.explore(task, controller, budget)  # with planner.last_failure

What demotion means
-------------------

A skill that *ran and did not work* - it raised, failed a ``ctx.expect``, or its own
``verifier_code`` rejected its result - is demoted with the reason, so retrieval
stops offering it and the explorer gets a chance to learn a replacement.

Nothing else demotes, and the line is deliberate. A skill's verifier answers "did
THIS SKILL do its job"; the critic answers "is THE TASK done", which is a different
question with other reasons to be false. The commonest is a task no single skill
covers: ``search_invoice`` does exactly what it promises and the errand is still
unfinished. Demoting on a critic's no would retire correct skills for being
incomplete, and a library that deletes its working parts whenever it is asked
something bigger does not grow. A rejected verdict is reported instead, and the
explorer takes over. A route that could not be walked, a missing argument or a broken
controller are not evidence against a skill either.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from skillweaver.agent.compose import Composer, Decomposition
from skillweaver.contracts import (
    Action,
    Budget,
    Candidate,
    Controller,
    Critic,
    Fingerprint,
    GraphView,
    Observation,
    Perceiver,
    Plan,
    Route,
    RunOutcome,
    Skill,
    SkillCall,
    SkillRetriever,
    SkillRunner,
    SkillStore,
    Spend,
    TaskSpec,
    Trajectory,
    Transition,
    Usage,
    Verdict,
    utcnow,
)
from skillweaver.errors import ControllerError, PerceptionError, SkillNotFound
from skillweaver.graph.route import VERIFIED_ONLY, RoutingPolicy, find_route
from skillweaver.logging_ import get_logger
from skillweaver.skills.api import LimitExceeded

__all__ = ["FRAME_WORDS", "FailureStage", "PlanFailure", "Planner", "Rejection"]

log = get_logger(__name__)

FailureStage = Literal[
    "no_candidates",
    "unbindable_args",
    "no_route",
    "no_decomposition",
    "vanished",
    "route_failed",
    "skill_failed",
    "rejected",
]
"""Where the fast path gave up. The first four happen before anything is performed."""

_PERFORMED_NOTHING = frozenset({"no_candidates", "unbindable_args", "no_route", "no_decomposition"})
"""Stages that are reached while planning, so the screen is untouched."""


@dataclass(frozen=True, slots=True)
class Rejection:
    """One retrieved skill the planner looked at and turned down.

    A warm run that explores is only diagnosable if the offer that was NOT taken is
    recorded: ``score`` says how well retrieval thought the skill fit, and ``stage``
    names the rule that discarded it. Without these two together, "the library was
    not used" cannot be told apart from "the library had nothing".
    """

    skill: str
    score: float
    stage: FailureStage
    reason: str

    def __str__(self) -> str:
        return f"{self.skill} (score {self.score:.3f}) -> {self.stage}"


@dataclass(frozen=True, slots=True)
class PlanFailure:
    """Why the fast path declined, in a form an explorer can use.

    ``stage`` says how far it got, ``reason`` is one human-readable line, ``skill``
    names the skill involved when there was one, ``demoted`` records whether that
    skill was retired, ``trace`` carries the failed skill's trace - the most valuable
    thing an explorer can be handed, since it is a play-by-play of what the library
    *thought* would work - and ``rejected`` holds every candidate that was offered
    and discarded before anything ran.
    """

    stage: FailureStage
    reason: str
    skill: str | None = None
    demoted: bool = False
    trace: tuple[str, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    @property
    def performed_nothing(self) -> bool:
        """Whether the screen is untouched, so an explorer may start from it as-is."""
        return self.stage in _PERFORMED_NOTHING

    def __str__(self) -> str:
        who = f" [{self.skill}]" if self.skill else ""
        return f"{self.stage}{who}: {self.reason}"


class Planner:
    """A :class:`~skillweaver.contracts.Planner`: do it from memory, or say you cannot.

    Args:
        store: The library. The planner reads skills from it and demotes through it.
        retriever: Finds candidate skills for the task text.
        graph: The site graph, for routing to a skill's precondition.
        runner: The sandbox that executes skill code.
        critic: Judges the run. A programmatic critic keeps a warm run model-free;
            a model-backed one will charge ``spend`` for its call, which is why the
            zero-call claim is about the action loop.
        controller: The hands.
        perceiver: The eyes.
        composer: Optional. Enables the composite path - one model call to chain
            known skills for a task no single skill covers. ``None`` means the
            planner is strictly model-free and simply declines such tasks.
        policy: How much routing risk to take (see
            :class:`~skillweaver.graph.route.RoutingPolicy`). The default uses only
            proven edges, which is what makes a replayed route trustworthy.
        budget: Limits for one :meth:`attempt`.
        top_k: How many candidates to retrieve.
        max_candidates: How many of them to try to build a plan from, best first.
            A candidate is skipped when its arguments cannot be bound from the task
            or when no route to its precondition is known.
    """

    __slots__ = (
        "_budget",
        "_composer",
        "_controller",
        "_critic",
        "_graph",
        "_last_failure",
        "_max_candidates",
        "_perceiver",
        "_policy",
        "_retriever",
        "_runner",
        "_store",
        "_top_k",
    )

    def __init__(
        self,
        *,
        store: SkillStore,
        retriever: SkillRetriever,
        graph: GraphView,
        runner: SkillRunner,
        critic: Critic,
        controller: Controller,
        perceiver: Perceiver,
        composer: Composer | None = None,
        policy: RoutingPolicy = VERIFIED_ONLY,
        budget: Budget | None = None,
        top_k: int = 5,
        max_candidates: int = 3,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._graph = graph
        self._runner = runner
        self._critic = critic
        self._controller = controller
        self._perceiver = perceiver
        self._composer = composer
        self._policy = policy
        self._budget = budget if budget is not None else Budget()
        self._top_k = top_k
        self._max_candidates = max_candidates
        self._last_failure: PlanFailure | None = None

    def __repr__(self) -> str:
        mode = "single+composite" if self._composer is not None else "single-skill"
        return f"Planner({mode}, policy={self._policy})"

    @property
    def last_failure(self) -> PlanFailure | None:
        """Why the most recent :meth:`plan` or :meth:`attempt` gave up, or ``None``
        after one that produced a plan. Hand this to the explorer as context."""
        return self._last_failure

    # -- the Planner protocol ---------------------------------------------------------

    def plan(self, task: TaskSpec, observation: Observation) -> Plan | None:
        """A plan for ``task`` from the current screen, or ``None``.

        Nothing is performed here: this only reads the library and the graph. The
        single-skill path is entirely model-free. Only when no single stored skill
        covers the task does a configured :class:`~skillweaver.agent.compose.Composer`
        spend ONE model call proposing a chain.

        Raises:
            ProviderError: if retrieval's embedder or the composer's model fails.
        """
        plan, _ = self._build(task, observation)
        return plan

    # -- planning and doing -----------------------------------------------------------

    def attempt(self, task: TaskSpec, observation: Observation) -> RunOutcome | None:
        """Plan, perform and verify - or return ``None`` so the caller explores.

        On success the outcome's ``spend.llm_calls`` is the number of model calls
        this attempt caused: **zero for a warm single-skill hit**, one when the
        composer was needed, plus whatever a model-backed critic spent.

        On failure :attr:`last_failure` explains what happened and ``None`` comes
        back, because a half-worked fast path is not an outcome - it is a reason to
        explore. A skill that ran and failed is demoted first; a clean run the critic
        merely judges incomplete is not (see the module docstring).

        Raises:
            BudgetExceeded: if ``budget`` runs out mid-run. Deliberately not caught:
                an exhausted run must stop, not fall through to a slower path.
            ProviderError: if retrieval or the composer fails.
        """
        spend = Spend(self._budget).start()
        plan, usage = self._build(task, observation)
        spend.add_usage(usage)
        if plan is None:
            return None

        started = utcnow()
        ctx = self._runner.context(
            self._controller, self._perceiver, graph=self._graph, domain=task.domain
        )
        used, failure = self._perform(plan, task, ctx, spend, observation)
        if failure is not None:
            self._last_failure = failure
            log.info("planner.failed", task=task.text, stage=failure.stage, why=failure.reason)
            return None

        after = self._observe()
        verdict = self._critic.judge(task.text, observation, after)
        if not verdict.ok:
            self._last_failure = self._reject(task, used, verdict)
            return None

        self._last_failure = None
        note = (
            f"warm path: {' -> '.join(used)} in {spend.steps} actions, "
            f"{spend.llm_calls} model call(s)"
        )
        log.info(
            "planner.ok",
            task=task.text,
            domain=task.domain,
            skills=" -> ".join(used),
            steps=spend.steps,
            llm_calls=spend.llm_calls,
        )
        return RunOutcome(
            ok=True,
            trajectory=self._trajectory(task, started, True, note),
            verdict=verdict,
            spend=spend,
            skill_used="+".join(used) if used else None,
            note=note,
        )

    # -- building a plan ---------------------------------------------------------------

    def _build(self, task: TaskSpec, observation: Observation) -> tuple[Plan | None, Usage]:
        """``(plan, model usage)``. The usage is zero on every warm single-skill hit,
        which is the whole point of separating this from :meth:`_perform`."""
        self._last_failure = None
        candidates = self._retriever.search(task.text, domain=task.domain, k=self._top_k)

        reasons: list[str] = []
        looked_at: list[Rejection] = []
        stage: FailureStage = "no_candidates"
        for candidate in candidates[: self._max_candidates]:
            skill = candidate.skill
            args = _bind_args(skill, task)
            if args is None:
                args = _bind_from_text(skill, task)
            if args is None:
                stage = "unbindable_args"
                reason = (
                    f"{skill.name}: task supplies no value for a required parameter, and its "
                    f"wording does not line up with {skill.provenance.task_text!r}"
                )
                reasons.append(reason)
                looked_at.append(Rejection(skill.name, candidate.score, "unbindable_args", reason))
                continue
            route = self._route_to(observation.fingerprint, skill.precondition)
            if route is None:
                stage = "no_route"
                reason = (
                    f"{skill.name}: no known route from {observation.fingerprint.value!r} "
                    f"to its start screen {skill.precondition.value!r}"  # type: ignore[union-attr]
                )
                reasons.append(reason)
                looked_at.append(Rejection(skill.name, candidate.score, "no_route", reason))
                continue
            log.info(
                "planner.hit",
                task=task.text,
                skill=skill.name,
                score=round(candidate.score, 3),
                args=args,
                route_steps=len(route.steps),
            )
            return (
                Plan(
                    steps=(*route.steps, SkillCall(skill.name, skill.domain, args)),
                    skills_used=(skill.name,),
                    estimated_ms=route.cost + skill.stats.mean_ms,
                ),
                Usage(),
            )

        self._report_miss(task, candidates, looked_at)
        return self._compose(task, observation, stage, reasons, looked_at)

    def _report_miss(
        self, task: TaskSpec, candidates: Sequence[Candidate], looked_at: Sequence[Rejection]
    ) -> None:
        """One log line naming everything needed to explain a model-free path that
        found nothing. It fires whenever no single stored skill was usable, whether
        the composer then rescues the task for one model call or the run explores.

        Four of six warm runs on the live Wikipedia suite reported ``skill_used=None``
        and left no record of WHY, so the first question - was the skill never
        offered, or offered and discarded? - could not be answered from a report. It
        can now: this names what the library held for the domain, what every candidate
        scored, and the rule that turned each one down.
        """
        shelf = [s.name for s in self._store.list(domain=task.domain)]
        log.info(
            "planner.miss",
            task=task.text,
            domain=task.domain,
            library=", ".join(shelf) or "(empty)",
            offered=", ".join(f"{c.skill.name}={c.score:.3f}" for c in candidates) or "(none)",
            rejected=" | ".join(str(r) for r in looked_at) or "(none reached binding)",
            why=" | ".join(f"{c.skill.name}: {c.why}" for c in candidates),
        )

    def _compose(
        self,
        task: TaskSpec,
        observation: Observation,
        stage: FailureStage,
        reasons: list[str],
        looked_at: Sequence[Rejection] = (),
    ) -> tuple[Plan | None, Usage]:
        """The composite path: one model call, then the same model-free execution.

        Only the leading route - from where the agent is now to the FIRST skill's
        start screen - can be computed here. Where a skill leaves the agent is not
        something the library records, so the routes between later steps are resolved
        against the live screen in :meth:`_perform`.
        """
        if self._composer is None:
            self._last_failure = PlanFailure(
                stage=stage,
                reason=_joined(reasons) or f"no stored skill matches {task.text!r}",
                rejected=tuple(looked_at),
            )
            return None, Usage()

        result: Decomposition = self._composer.compose(task, observation)
        if not result.steps:
            self._last_failure = PlanFailure(
                stage="no_decomposition",
                reason=result.rejected or "the composer proposed no chain",
                rejected=tuple(looked_at),
            )
            return None, result.usage

        first = self._lookup(result.steps[0])
        if first is None:
            self._last_failure = PlanFailure(
                stage="vanished",
                reason=f"skill {result.steps[0].name!r} left the library while planning",
                skill=result.steps[0].name,
            )
            return None, result.usage
        route = self._route_to(observation.fingerprint, first.precondition)
        if route is None:
            self._last_failure = PlanFailure(
                stage="no_route",
                reason=(
                    f"{first.name}: no known route from {observation.fingerprint.value!r} "
                    f"to its start screen {first.precondition.value!r}"  # type: ignore[union-attr]
                ),
                skill=first.name,
            )
            return None, result.usage

        log.info(
            "planner.composed",
            task=task.text,
            chain=" -> ".join(result.names),
            calls=result.usage.calls,
        )
        return (
            Plan(
                steps=(*route.steps, *result.steps),
                skills_used=result.names,
                estimated_ms=route.cost + sum(self._mean_ms(call) for call in result.steps),
            ),
            result.usage,
        )

    # -- performing a plan -------------------------------------------------------------

    def _perform(
        self,
        plan: Plan,
        task: TaskSpec,
        ctx: Any,
        spend: Spend,
        observation: Observation,
    ) -> tuple[list[str], PlanFailure | None]:
        """Walk the plan. ``(skills that ran, failure or None)``.

        Every skill call re-reads the screen and routes to that skill's precondition
        from where the agent actually is. For the first skill that route is normally
        empty - the plan's leading actions just walked it - which makes the check
        free and makes a plan that drifted fail cleanly instead of running a skill on
        the wrong screen.
        """
        used: list[str] = []
        current: Observation | None = observation

        for step in plan.steps:
            spend.check()
            if not isinstance(step, SkillCall):
                failure = self._act(step, ctx, spend)
                if failure is not None:
                    return used, failure
                current = None
                continue

            skill = self._lookup(step)
            if skill is None:
                return used, PlanFailure(
                    stage="vanished",
                    reason=f"skill {step.name!r} is no longer in the library",
                    skill=step.name,
                )
            if current is None:
                try:
                    current = self._observe()
                except (ControllerError, PerceptionError) as exc:
                    return used, PlanFailure("route_failed", f"could not read the screen: {exc}")

            route = self._route_to(current.fingerprint, skill.precondition)
            if route is None:
                return used, PlanFailure(
                    stage="no_route",
                    reason=(
                        f"{skill.name}: no known route from {current.fingerprint.value!r} "
                        f"to its start screen {skill.precondition.value!r}"  # type: ignore[union-attr]
                    ),
                    skill=skill.name,
                )
            for action in route.steps:
                failure = self._act(action, ctx, spend)
                if failure is not None:
                    return used, failure
                current = None

            result = self._runner.run(skill, step.args, ctx)
            spend.add_step(result.steps)
            current = None
            if not result.ok:
                reason = result.error or "the skill failed without saying why"
                self._demote(skill, f"failed on the warm path: {reason}")
                return used, PlanFailure(
                    stage="skill_failed",
                    reason=reason,
                    skill=skill.name,
                    demoted=True,
                    trace=result.trace,
                )
            used.append(skill.name)

        return used, None

    def _act(self, action: Action, ctx: Any, spend: Spend) -> PlanFailure | None:
        """Perform one replayed route action. Failing to walk a remembered route is
        not evidence against any skill, so nothing is demoted for it."""
        try:
            ctx.ctl.perform(action)
        except (ControllerError, LimitExceeded) as exc:
            return PlanFailure(stage="route_failed", reason=f"replaying the route failed: {exc}")
        spend.add_step()
        return None

    # -- verdict and bookkeeping --------------------------------------------------------

    def _reject(self, task: TaskSpec, used: list[str], verdict: Verdict) -> PlanFailure:
        """Report a critic that says the task is not done.

        Nothing is demoted here. Every skill in ``used`` ran cleanly - each one that
        carries a ``verifier_code`` also passed it - so the evidence says the plan was
        incomplete or the screen is somewhere unexpected, not that any of these skills
        is broken. The explorer is told what ran and what the critic said, and finishes
        the job from wherever the plan left off.
        """
        reason = verdict.reason or "the critic rejected the result"
        log.info(
            "planner.rejected",
            task=task.text,
            chain=" -> ".join(used) or "(nothing)",
            reason=reason,
            confidence=round(verdict.confidence, 3),
        )
        return PlanFailure("rejected", reason, skill=" + ".join(used) or None)

    def _demote(self, skill: Skill, reason: str) -> None:
        """Retire a skill from retrieval, tolerating one that has already gone."""
        try:
            self._store.demote(skill.name, skill.domain, reason)
        except SkillNotFound:
            log.debug("planner.demote.missing", skill=skill.name, domain=skill.domain)
            return
        log.info("planner.demoted", skill=skill.name, domain=skill.domain, reason=reason)

    def _trajectory(self, task: TaskSpec, started: Any, ok: bool, note: str) -> Trajectory:
        """The record of a warm run.

        It has no steps, and that is the finding rather than an omission: the fast
        path does not observe between actions, which is most of why it is fast. The
        run is fully described by the plan, the verdict and the spend, and there is
        nothing here for a synthesizer to learn - the skill it would write already
        exists.
        """
        return Trajectory(
            run_id=f"warm-{uuid.uuid4().hex[:12]}",
            task=task.text,
            domain=task.domain,
            steps=(),
            ok=ok,
            started_at=started,
            finished_at=utcnow(),
            note=note,
        )

    # -- small helpers -------------------------------------------------------------------

    def _route_to(self, src: Fingerprint, precondition: Fingerprint | None) -> Route | None:
        """The route to a skill's start screen, or ``None`` when none is known.

        A skill with no precondition can start anywhere, so it gets the empty route
        rather than a search.
        """
        if precondition is None:
            return Route((), 0.0, ())
        return find_route(src, precondition, self._outgoing, self._policy)

    def _outgoing(self, value: str) -> Iterable[Transition]:
        return self._graph.neighbors(Fingerprint(value))

    def _observe(self) -> Observation:
        return self._perceiver.observe(self._controller)

    def _lookup(self, call: SkillCall) -> Skill | None:
        return self._find(call.name, call.domain)

    def _find(self, name: str, domain: str) -> Skill | None:
        try:
            return self._store.get(name, domain)
        except SkillNotFound:
            return None

    def _mean_ms(self, call: SkillCall) -> float:
        skill = self._lookup(call)
        return skill.stats.mean_ms if skill is not None else 0.0


def _bind_args(skill: Skill, task: TaskSpec) -> dict[str, Any] | None:
    """The skill's arguments taken from ``task.params``, or ``None`` if it cannot be
    called from them.

    This is the model-free half of "what do I pass?". A ``TaskSpec`` carries the
    concrete values for the errand (``{"company": "Acme Corp"}``) and a stored skill
    declares the names it wants; when they line up there is nothing to reason about.
    A required parameter the task has no value for means the task did not supply its
    arguments; :func:`_bind_from_text` gets a chance to read them out of the task's
    wording before the candidate is given up on.

    Names that line up EXACTLY are taken first, and only then is what is left over
    matched by :func:`_alias_for` - because the caller names a parameter and the
    model that wrote the skill names it again, independently, and they routinely
    disagree about one word. Measured on the live Wikipedia suite: the caller passes
    ``link`` where ``open_linked_article`` declares ``link_title``, which cost that
    task its whole warm path.
    """
    args: dict[str, Any] = {}
    spare = {k: v for k, v in task.params.items() if k not in skill.params}
    for name, schema in skill.params.items():
        if name in task.params:
            args[name] = task.params[name]
            continue
        alias = _alias_for(name, spare, skill.params)
        if alias is not None:
            args[name] = spare.pop(alias)
            log.info("planner.param_alias", skill=skill.name, declared=name, supplied=alias)
        elif not _has_default(schema):
            return None
    return args


def _alias_for(name: str, spare: Mapping[str, Any], declared: Mapping[str, Any]) -> str | None:
    """The one key of ``spare`` that plainly means the parameter ``name``, or ``None``.

    Two names mean the same thing here when, split on underscores, one's words are a
    subset of the other's: ``link`` and ``link_title``, ``query`` and ``search_query``.
    Nothing is bound unless the reading is unambiguous in BOTH directions - exactly
    one spare key fits this parameter, and that key fits no other declared parameter -
    so a skill taking ``section`` and ``parent_section`` is never fed a ``section``
    value twice, and a task carrying ``link`` and ``link_text`` is not guessed at.
    """
    wanted = _name_words(name)
    fits = [key for key in spare if _shares_words(_name_words(key), wanted)]
    if len(fits) != 1:
        return None
    key = fits[0]
    words = _name_words(key)
    rivals = [d for d in declared if d != name and _shares_words(words, _name_words(d))]
    return None if rivals else key


def _shares_words(left: frozenset[str], right: frozenset[str]) -> bool:
    """Whether two parameter names plainly mean the same thing: one's words contain
    the other's, and there is at least one word to contain. An empty name matches
    nothing, so a parameter called ``_`` is never fed somebody else's value."""
    return bool(left & right) and (left <= right or right <= left)


def _name_words(name: str) -> frozenset[str]:
    return frozenset(part for part in name.casefold().split("_") if part)


# -- binding a plain-English repeat ------------------------------------------------------


FRAME_WORDS = frozenset(
    """
    a about after against all an and any around as at be been before being between both but by
    called did do does entitled every for from had has have how in into is it its me my named
    named of off on onto or our out over please should the their then this through titled to up
    upon us via was were what when where which while who whose why will with within without
    would you your
    """.split()
)
"""Words a value may sit next to without the alignment being in doubt.

A parameter's value is only read out of a sentence when the words that FRAME it are
words like these - or punctuation, or the edge of the sentence. The rule exists
because a maximal common prefix happily eats a word that belongs to the value:
``"search for computer vision"`` against ``"search for computer graphics"`` agrees on
``"search for computer"``, and the span that is left is ``"graphics"`` - which is not
the argument. ``"computer"`` is not a framing word, so that alignment is refused and
the composer, which can actually read the sentence, gets the job instead.
"""

_TOKEN = re.compile(r"\w+|[^\w\s]")
"""One word or one punctuation mark. Whitespace is not a token, so a value's own
spacing survives being sliced back out of the original text."""

_MAX_VALUE_CHARS = 120
"""A bound argument longer than this is a sentence, not a value."""


@dataclass(frozen=True, slots=True)
class _Token:
    key: str
    start: int
    end: int
    word: bool


def _tokens(text: str) -> list[_Token]:
    return [
        _Token(
            m.group(0).casefold(),
            m.start(),
            m.end(),
            m.group(0)[0].isalnum() or m.group(0)[0] == "_",
        )
        for m in _TOKEN.finditer(text)
    ]


def _bind_from_text(skill: Skill, task: TaskSpec) -> dict[str, Any] | None:
    """The skill's arguments read out of the task TEXT, or ``None`` to decline.

    This is what makes a plain-English repeat of a known task cost nothing. A stored
    skill remembers the exact sentence the run it was synthesized from was solving
    (:attr:`~skillweaver.contracts.Provenance.task_text`). When the new task is the
    same sentence with one thing changed, that change IS the argument::

        learned: "Search Wikipedia for computer vision"
        asked:   "Search Wikipedia for machine learning"
                                      ^^^^^^^^^^^^^^^^  -> query="machine learning"

    Without this the candidate is rejected as ``unbindable_args`` and the planner
    falls through to the composer, which spends one model call to read a sentence it
    has already seen the shape of.

    When the diff says nothing, :func:`_from_template` gets a turn. The diff is blind
    in exactly the two cases the live Wikipedia suite is full of: the sentence is
    repeated WORD FOR WORD (no difference at all to attribute), or it changed in two
    places and only one of them is the argument. See that function for how a slot is
    located in the learned sentence instead.

    It declines far more often than it binds, deliberately: a wrong argument is much
    worse than a model call, because it runs real actions on a real screen behind the
    same preconditions and verifier as a correct one. It refuses unless

    * exactly ONE required parameter is missing (several missing means several spans
      to attribute, which is a guess). A parameter with a declared default is never
      bound here - omitting it already works;
    * the two sentences agree everywhere except one contiguous span, both sides of
      which are non-empty - so a sentence that merely ADDS words is not an argument;
    * at least two tokens are shared, and the span is framed by :data:`FRAME_WORDS`,
      punctuation or the sentence edge on both sides;
    * the two spans share no word, which is how two separate differences pretending
      to be one are caught (``"cats on Monday"`` vs ``"dogs on Tuesday"``);
    * the new span is not much longer than the learned one, and can actually be the
      parameter's declared type.

    Returns:
        The full argument dict, or ``None`` to leave the candidate unbindable.
    """
    missing = [
        name
        for name, schema in skill.params.items()
        if name not in task.params and not _has_default(schema)
    ]
    if len(missing) != 1:
        return None

    name = missing[0]
    learned = skill.provenance.task_text or ""
    span = _differing_span(learned, task.text)
    if span is None:
        span = _from_template(skill, name, learned, task.text)
    if span is None:
        return None

    value = _as_declared(span, skill.params[name])
    if value is _REFUSED:
        return None

    args: dict[str, Any] = {k: task.params[k] for k in skill.params if k in task.params}
    args[name] = value
    log.info(
        "planner.bound_from_text",
        skill=skill.name,
        domain=skill.domain,
        param=name,
        value=value,
        learned=learned,
        task=task.text,
        args=args,
    )
    return args


def _differing_span(learned: str, asked: str) -> str | None:
    """The one span of ``asked`` that ``learned`` does not have, or ``None``.

    The two sentences are aligned from both ends; what is left in the middle is the
    difference. Every guard described in :func:`_bind_from_text` is applied here, and
    the returned text is sliced out of ``asked`` verbatim, so the value keeps its own
    capitalisation, spacing and internal punctuation.
    """
    lhs, rhs = _tokens(learned), _tokens(asked)
    if not lhs or not rhs:
        return None

    limit = min(len(lhs), len(rhs))
    head = 0
    while head < limit and lhs[head].key == rhs[head].key:
        head += 1
    tail = 0
    while tail < limit - head and lhs[-1 - tail].key == rhs[-1 - tail].key:
        tail += 1

    learned_mid = lhs[head : len(lhs) - tail]
    asked_mid = rhs[head : len(rhs) - tail]
    if not _is_a_value(learned_mid) or not _is_a_value(asked_mid):
        return None
    if head + tail < 2:
        return None  # two sentences that barely agree are not the same sentence
    if _words(learned_mid) & _words(asked_mid):
        return None  # a shared word inside the span means this is two differences
    if not _framed(rhs, head, tail):
        return None
    if len(_words(asked_mid)) > max(3, len(_words(learned_mid)) + 2):
        return None  # the new span grew into something bigger than an argument

    value = asked[asked_mid[0].start : asked_mid[-1].end]
    return value if 0 < len(value) <= _MAX_VALUE_CHARS else None


# -- binding from the slot the learned sentence left ------------------------------------


_DOUBLE_QUOTED = re.compile(r"[\"“]([^\"“”]{1,120})[\"”]")
"""A span in double quotes, straight or curly. People quote the thing they mean."""

_SINGLE_QUOTED = re.compile(r"(?<![\w'’])['‘]([^'‘’]{1,120})['’](?![\w])")
"""A span in single quotes, with the apostrophe of ``Wikipedia's`` excluded by the
look-around on both sides - otherwise every possessive opens a quotation."""

_MIN_ANCHOR_TOKENS = 2
"""How many tokens either side of a slot must line up before it is believed."""

_MAX_TAIL_DRIFT = 2
"""How many more (or fewer) tokens may follow the value than followed it when the
skill was learned. A sentence that grew past this is a bigger errand, not a rephrasing."""


@dataclass(frozen=True, slots=True)
class _Slot:
    """Where a parameter's value sat inside the sentence a skill was learned from."""

    start: int
    end: int
    value: str
    quoted: bool


def _from_template(skill: Skill, name: str, learned: str, asked: str) -> str | None:
    """``asked``'s value for ``name``, read through the learned sentence as a template.

    :func:`_differing_span` asks "what changed?", which has no answer in the two
    cases that dominated the live Wikipedia misses of 2026-09-19::

        learned: 'Search Wikipedia for "Ada Lovelace" and open her article.'
        asked:   'Search Wikipedia for "Ada Lovelace" and open her article.'
          -> nothing changed, so nothing is the argument, so the skill is not used

        learned: 'Search Wikipedia for "Ada Lovelace" and open her article.'
        asked:   'Search Wikipedia for "Photosynthesis" and open the article.'
          -> TWO things changed ("her" -> "the"), so neither is trusted

    This asks the other question: where in the learned sentence did the value sit?
    :func:`_learned_slot` answers it from evidence already in the skill, and then

    * a word-for-word repeat replays the learned value. It is not a guess: this skill
      exists BECAUSE a run of this exact sentence succeeded with that value;
    * a sentence whose value is quoted takes the asked sentence's one quoted span,
      but only when the words anchoring the slot still line up (:func:`_anchored`).

    Anything else returns ``None`` and the composer, which can read a sentence, is
    left to do it.
    """
    slot = _learned_slot(skill, name, learned)
    if slot is None:
        return None
    if _same_sentence(learned, asked):
        return slot.value
    if not slot.quoted:
        return None
    quoted = _quoted_spans(asked)
    if len(quoted) != 1 or not _anchored(learned, slot, asked, quoted[0]):
        return None
    return quoted[0].value


def _learned_slot(skill: Skill, name: str, learned: str) -> _Slot | None:
    """Where ``name``'s value sat in ``learned``, or ``None`` when it cannot be shown.

    Two independent sources, and they check each other:

    *The one quoted span in the learned sentence.* A person quoting exactly one thing
    in a one-parameter errand is quoting the parameter.

    *A quoted example in the parameter's own schema description* - the synthesizer
    writes ``e.g. 'Ada Lovelace'`` from the run it just watched - accepted only when
    that example occurs exactly once in the learned sentence. That occurrence is the
    proof: an example the sentence does not contain says nothing about where the
    value was, and is ignored rather than trusted.

    When both exist they must agree, so a skill learned from *Open the "Settings" page
    and search for widgets* does not hand ``"Settings"`` to a ``query`` parameter whose
    description says ``e.g. 'widgets'``.
    """
    quoted = _quoted_spans(learned)
    examples = _quoted_spans(_description(skill.params.get(name)))
    shown = [
        found
        for example in examples
        if (found := _sole_occurrence(learned, example.value)) is not None
    ]

    if len(quoted) == 1:
        if examples and not any(s.value.casefold() == quoted[0].value.casefold() for s in shown):
            return None  # the description names a different value; do not guess
        return quoted[0]
    return shown[0] if len(shown) == 1 else None


def _quoted_spans(text: str) -> list[_Slot]:
    """Every quoted span of ``text``, in order, straight or curly, double or single."""
    found = [
        _Slot(m.start(1), m.end(1), m.group(1).strip(), True)
        for pattern in (_DOUBLE_QUOTED, _SINGLE_QUOTED)
        for m in pattern.finditer(text)
    ]
    found.sort(key=lambda s: s.start)
    return [s for s in found if s.value]


def _description(schema: Any) -> str:
    text = schema.get("description") if isinstance(schema, Mapping) else None
    return text if isinstance(text, str) else ""


def _sole_occurrence(text: str, value: str) -> _Slot | None:
    """``value``'s position in ``text`` when it appears exactly once, else ``None``."""
    haystack, needle = text.casefold(), value.casefold()
    first = haystack.find(needle)
    if first < 0 or haystack.find(needle, first + 1) >= 0:
        return None
    return _Slot(first, first + len(needle), text[first : first + len(needle)], False)


def _same_sentence(learned: str, asked: str) -> bool:
    """Whether the two are the same sentence down to punctuation, ignoring case and
    whitespace - the only reading under which the learned value is certainly right."""
    return [t.key for t in _tokens(learned)] == [t.key for t in _tokens(asked)]


def _anchored(learned: str, slot: _Slot, asked: str, found: _Slot) -> bool:
    """Whether ``found`` sits where ``slot`` sat, judged by the words around it.

    The tokens immediately BEFORE the value must be the same on both sides - that is
    what makes this the same sentence shape rather than a different errand that also
    quotes something - and the sentence must not have grown a new clause after the
    value, which is how *search for "Alan Turing", open his article, and from there
    open Bletchley Park* is refused: it is two errands, and the composer's job.
    """
    head_l, head_a = _tokens(learned[: slot.start]), _tokens(asked[: found.start])
    tail_l, tail_a = _tokens(learned[slot.end :]), _tokens(asked[found.end :])

    anchor = min(len(head_l), len(head_a), 4)
    if anchor < _MIN_ANCHOR_TOKENS:
        return False
    if [t.key for t in head_l[-anchor:]] != [t.key for t in head_a[-anchor:]]:
        return False
    if [t.key for t in tail_l[:1]] != [t.key for t in tail_a[:1]]:
        return False
    return abs(len(tail_a) - len(tail_l)) <= _MAX_TAIL_DRIFT


def _framed(tokens: Sequence[_Token], head: int, tail: int) -> bool:
    """Whether the span sits between words that cannot themselves belong to it."""
    before = tokens[head - 1] if head else None
    after = tokens[len(tokens) - tail] if tail else None
    return all(edge is None or not edge.word or edge.key in FRAME_WORDS for edge in (before, after))


def _is_a_value(span: Sequence[_Token]) -> bool:
    """A span is a candidate value only if it has at least one word in it."""
    return any(token.word for token in span)


def _words(span: Sequence[_Token]) -> set[str]:
    return {token.key for token in span if token.word}


_REFUSED = object()
"""Sentinel: this span cannot be the parameter's declared type."""

_INTEGER = re.compile(r"[+-]?\d+$")
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _as_declared(span: str, schema: Any) -> Any:
    """``span`` as the parameter's declared type, or :data:`_REFUSED`.

    A span of prose is a string and nothing else unless it plainly reads as the
    declared type. Anything with structure - an array, an object, a boolean - is
    refused rather than parsed out of English, because there is no reading of a
    phrase as ``True`` that is safer than asking the model.
    """
    text = span.strip()
    if not text:
        return _REFUSED
    if not isinstance(schema, Mapping):
        return text

    choices = schema.get("enum")
    if isinstance(choices, Sequence) and not isinstance(choices, str):
        matches = [c for c in choices if isinstance(c, str) and c.casefold() == text.casefold()]
        return matches[0] if len(matches) == 1 else _REFUSED

    declared = schema.get("type")
    if declared in (None, "string"):
        return text
    if declared == "integer" and _INTEGER.match(text):
        return int(text)
    if declared == "number" and _NUMBER.match(text):
        return float(text)
    return _REFUSED


def _has_default(schema: Any) -> bool:
    return isinstance(schema, Mapping) and "default" in schema


def _joined(reasons: list[str]) -> str:
    return "; ".join(reasons)
