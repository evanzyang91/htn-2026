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

import uuid
from collections.abc import Callable, Iterable, Mapping
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
from skillweaver.errors import (
    ControllerError,
    PerceptionError,
    SkillNotFound,
    SkillWeaverError,
)
from skillweaver.graph.route import VERIFIED_ONLY, RoutingPolicy, find_route
from skillweaver.logging_ import get_logger
from skillweaver.skills.api import LimitExceeded

__all__ = ["FailureStage", "PlanFailure", "Planner"]

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
class PlanFailure:
    """Why the fast path declined, in a form an explorer can use.

    ``stage`` says how far it got, ``reason`` is one human-readable line, ``skill``
    names the skill involved when there was one, ``demoted`` records whether that
    skill was retired, and ``trace`` carries the failed skill's trace - the most
    valuable thing an explorer can be handed, since it is a play-by-play of what the
    library *thought* would work.
    """

    stage: FailureStage
    reason: str
    skill: str | None = None
    demoted: bool = False
    trace: tuple[str, ...] = ()

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
        end_states: Looks up, by skill name, the screen the run that TAUGHT that skill
            ended on. Used to judge a finished plan against the right screen rather
            than against whichever skill retrieval happened to rank first; see
            :meth:`_judge`. ``None`` leaves the critic as it was built.
    """

    __slots__ = (
        "_budget",
        "_composer",
        "_controller",
        "_critic",
        "_end_states",
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
        end_states: Callable[[str], Fingerprint | None] | None = None,
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
        self._end_states = end_states
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
        verdict = self._judge(task, used, observation, after)
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
        which is the whole point of separating this from :meth:`_perform`.

        Candidates are considered best-first, and a candidate is only RUN when nothing
        that retrieval liked better is merely waiting to be told its arguments. That
        rule is the difference between a fast path and a wrong one: a general skill
        that takes no parameters - ``open_records_screen`` - binds trivially against
        every task, so the first version of this loop skipped the skill the task was
        actually about (because the sentence had not been read for its arguments yet)
        and went off and opened the Records screen instead. It then had to be judged,
        rejected and rescued by exploration, which is slower than never having run it
        and leaves the user looking at a screen they did not ask for.

        Reading the arguments out of the sentence is exactly what the composer is for,
        so a better candidate that cannot be bound sends the plan there rather than
        letting a worse one act.
        """
        self._last_failure = None
        candidates = self._retriever.search(task.text, domain=task.domain, k=self._top_k)

        reasons: list[str] = []
        stage: FailureStage = "no_candidates"
        ready: tuple[Candidate, dict[str, Any], Route] | None = None
        outranked = False
        for candidate in candidates[: self._max_candidates]:
            skill = candidate.skill
            args = _bind_args(skill, task)
            if args is None:
                stage = "unbindable_args"
                reasons.append(f"{skill.name}: task supplies no value for a required parameter")
                if ready is None:
                    # Retrieval put this one ahead of anything runnable so far, and the
                    # only thing it lacks is its arguments.
                    outranked = True
                continue
            route = self._route_to(observation.fingerprint, skill.precondition)
            if route is None:
                stage = "no_route"
                reasons.append(
                    f"{skill.name}: no known route from {observation.fingerprint.value!r} "
                    f"to its start screen {skill.precondition.value!r}"  # type: ignore[union-attr]
                )
                continue
            if ready is None:
                ready = (candidate, args, route)
            if not outranked:
                break

        if ready is not None and not outranked:
            return self._single(task, *ready), Usage()

        # Outranked: the composer is the only thing that can turn the better candidate
        # into a call, so it gets the chance. If it cannot, the answer is to decline -
        # NOT to run the worse skill anyway. Declining hands a clean screen to the
        # explorer, which can read the sentence; running the wrong errand performs
        # something nobody asked for and still needs rescuing afterwards.
        return self._compose(task, observation, stage, reasons)

    def _single(
        self, task: TaskSpec, candidate: Candidate, args: dict[str, Any], route: Route
    ) -> Plan:
        """The plan that walks a route and runs one stored skill."""
        skill = candidate.skill
        log.info(
            "planner.hit",
            task=task.text,
            skill=skill.name,
            score=round(candidate.score, 3),
            route_steps=len(route.steps),
        )
        return Plan(
            steps=(*route.steps, SkillCall(skill.name, skill.domain, args)),
            skills_used=(skill.name,),
            estimated_ms=route.cost + skill.stats.mean_ms,
        )

    def _compose(
        self,
        task: TaskSpec,
        observation: Observation,
        stage: FailureStage,
        reasons: list[str],
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
            )
            return None, Usage()

        result: Decomposition = self._composer.compose(task, observation)
        if not result.steps:
            self._last_failure = PlanFailure(
                stage="no_decomposition",
                reason=result.rejected or "the composer proposed no chain",
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
                retired = self._worth_demoting(skill)
                if retired:
                    self._demote(skill, f"failed on the warm path: {reason}")
                else:
                    log.info(
                        "planner.spared",
                        skill=skill.name,
                        successes=skill.stats.successes,
                        runs=skill.stats.runs,
                        reason=reason,
                    )
                return used, PlanFailure(
                    stage="skill_failed",
                    reason=reason,
                    skill=skill.name,
                    demoted=retired,
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

    def _judge(
        self, task: TaskSpec, used: list[str], before: Observation, after: Observation
    ) -> Verdict:
        """Judge the finished plan, against where the LAST skill that ran should end.

        The critic this planner was built with carries an expected screen guessed
        before anything ran - from whichever skill retrieval liked best, because at
        that point nothing better is known. Once a plan has actually run, which skill
        it was is no longer a guess, and using the guess instead is how a warm run that
        did its job exactly gets rejected: with a library of one they are the same
        skill, and with a library of fourteen the critic is holding some other task's
        finishing screen.

        Falls back to the critic as built whenever the end state cannot be looked up or
        the critic cannot be specialised, so this is an improvement on the evidence
        rather than a dependency.
        """
        expected = self._end_state_for(used)
        rebind = getattr(self._critic, "for_end_state", None)
        critic = self._critic if expected is None or rebind is None else rebind(expected)
        return critic.judge(task.text, before, after)

    def _end_state_for(self, used: list[str]) -> Fingerprint | None:
        """Where the run that taught the last skill in the chain ended."""
        if self._end_states is None or not used:
            return None
        try:
            return self._end_states(used[-1])
        except SkillWeaverError:
            return None

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

    @staticmethod
    def _worth_demoting(skill: Skill) -> bool:
        """Whether one failure is evidence the skill is broken, or only that it was
        asked the wrong thing.

        A failure means one of two things and they look identical from here: the skill
        has stopped working, or it was called for a task it was never about. The second
        really happens - retrieval ranks a skill that opens a message top for a task
        that opens a message AND sends one, the arguments bind, and the skill goes
        looking for a subject belonging to the other half of the sentence. Retiring it
        for that loses a skill that works, and the next run of the task it IS about has
        to learn it again.

        So a skill that has a record of working is spared its first failures and the
        failure is reported instead; one that fails more often than it works is not.
        A skill that has never succeeded has nothing to weigh against the evidence.
        """
        stats = skill.stats
        if stats.successes == 0:
            return True
        return stats.successes * 2 <= stats.runs

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
    A required parameter the task has no value for means this candidate cannot be
    invoked warm, so the planner tries the next one - and eventually the composer,
    which can read the task text.
    """
    args: dict[str, Any] = {}
    for name, schema in skill.params.items():
        if name in task.params:
            args[name] = task.params[name]
        elif isinstance(schema, Mapping) and "default" in schema:
            continue
        else:
            return None
    return args


def _joined(reasons: list[str]) -> str:
    return "; ".join(reasons)
