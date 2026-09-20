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

Ranking picks a winner; this decides whether to run one
-------------------------------------------------------

Retrieval answers "which stored skill is closest to this request?", and that question
has an answer even when the honest answer is "none of them". On 2026-09-19 the
ordering suite asked for *order two Vegetable Rolls from Sakura Counter, then open the
Orders tab and confirm the order is listed there* and the closest skill in the library
was a navigation skill that opens a tab. It bound, it routed, it ran in four seconds,
it consulted no model, and it got the task wrong every time - the cheapest possible
way to be wrong, and the one a wall-clock table rewards.

So a candidate now has to pass one more test before any action is spent on it: every
word of the request must be accounted for, either by the skill's own text or by the
arguments it is about to be handed (:data:`MIN_ACCOUNTED_FOR`). A skill whose
declared end state has no account of *Vegetable Rolls*, *Sakura Counter* or
*confirm* is not what was asked for, whatever it ranked. Rejected candidates are
recorded as rejections and the task falls through to the composer and then
to exploration, which is what happens for any other task the library cannot do.

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
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    Precedent,
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
from skillweaver.skills.family import (
    earned,
    head_verb,
    intent_of,
    relatives,
    render,
    same_intent,
)
from skillweaver.skills.retrieve import accounted_for, unaddressed

__all__ = [
    "FRAME_WORDS",
    "MEASURE_WORDS",
    "MIN_ACCOUNTED_FOR",
    "FailureStage",
    "FamilyFit",
    "PlanFailure",
    "Planner",
    "Rejection",
    "account_of",
    "bind_args",
    "fit_through_family",
]

log = get_logger(__name__)

FailureStage = Literal[
    "no_candidates",
    "unbindable_args",
    "unaccounted",
    "other_intent",
    "no_route",
    "no_decomposition",
    "vanished",
    "route_failed",
    "skill_failed",
    "rejected",
]
"""Where the fast path gave up. The first six happen before anything is performed.

``other_intent`` is the one family reuse added: the request's words line up with a
stored skill and its VERB does not, which is *remove ... from my cart* meeting a skill
learned as *add ... to my cart*. See :func:`~skillweaver.skills.family.same_intent`."""

_PERFORMED_NOTHING = frozenset(
    {
        "no_candidates",
        "unbindable_args",
        "unaccounted",
        "other_intent",
        "no_route",
        "no_decomposition",
    }
)
"""Stages that are reached while planning, so the screen is untouched."""

MIN_ACCOUNTED_FOR = 0.75
"""How much of the request a single skill must have an account of to be run.

Retrieval ranks, and a ranking always has a winner. This is the planner's own
question, and it is a different one: *does the winner have any account of the whole
errand?* :func:`~skillweaver.skills.retrieve.accounted_for` answers it without a
model - every word of the task must appear either in the skill's own text or in the
arguments it is about to be handed - and a candidate below this line is passed over
instead of performed.

Calibrated 2026-09-19 against the ordering suite's own library, over all twelve tasks
in both binding shapes. Only candidates that BIND are measured, because the others
never reach this check. The number sits in a gap rather than on top of the one case
that prompted it:

    the right skill, where the library had one     0.90 to 1.00 (0.90 exactly once,
                                                   on `place_an_order`, whose skill
                                                   really does leave "place" undone)
    the best WRONG skill that also bound           0.60 at its worst, and 0.42 on
                                                   `order_appears_in_history`, 0.24
                                                   on `order_with_address_and_tip`

Both failing cases are the same trivial navigation skill, which binds because its one
parameter has a default, and therefore falls out as the first usable candidate
whenever nothing better binds. 0.75 leaves 0.15 of margin on each side. The table is
pinned by ``test_the_calibration_gap_this_line_sits_in`` in
``tests/agent/test_planner.py``.

Which way to be wrong is not symmetric. A skill declined here falls through to the
composer and then to exploration, and the task still gets done a little slower; a
skill wrongly run reports a fast, free success that was not one, and every efficiency
figure in the project improves because of it."""


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


@dataclass(frozen=True, slots=True)
class _Reuse:
    """How the single skill of the current plan came to be chosen.

    ``how`` is ``"params"`` (the caller supplied the arguments), ``"diff"`` or
    ``"template"`` (the two strict sentence readers), ``"slot"`` (read through a
    proven sentence) or ``"family"`` (bound through a relative on another site). It
    decides two things after the run: what a FAILURE is evidence against
    (:data:`_WIDENED`), and whether a SUCCESS is worth writing down as a precedent.
    """

    skill: str
    args: Mapping[str, Any]
    how: str
    vouchers: tuple[Skill, ...] = ()


_WIDENED = frozenset({"slot", "family"})
"""Bindings a failed run is evidence against, INSTEAD of the skill.

A skill that fails after the strict readers bound it has failed at its own job and is
demoted, as it always was. A skill that fails after one of these bound it may simply
have been handed the wrong argument - the reader is newer and looser than the skill -
so the run is reported as failed, nothing is recorded for it, and the skill keeps its
place. Wrongly demoting a working skill costs every later run a cold start; wrongly
sparing a broken one costs one more failed warm attempt, which demotes it the next time
a strict reader reaches it."""


@dataclass(frozen=True, slots=True)
class FamilyFit:
    """One way a request can be run through a family: ``skill`` is the member filed
    under the request's own domain and is what will RUN; ``via`` is the relative whose
    proven sentence the request bound against; ``vouchers`` are the relatives of the
    same intent whose words counted towards ``share``."""

    skill: Skill
    args: dict[str, Any]
    via: Skill
    vouchers: tuple[Skill, ...]
    share: float


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
        "_reuse",
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
        self._reuse: _Reuse | None = None

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
        self._remember(task, used)
        note = (
            f"warm path: {' -> '.join(used)} in {spend.steps} actions, "
            f"{spend.llm_calls} model call(s)"
        )
        if self._reuse is not None and self._reuse.how in _WIDENED:
            note += f" (bound by the {self._reuse.how} reader)"
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
        self._reuse = None
        for candidate in candidates[: self._max_candidates]:
            skill = candidate.skill
            args = bind_args(skill, task)
            if args is None:
                stage, reason = _why_unbound(skill, task)
                reasons.append(reason)
                looked_at.append(Rejection(skill.name, candidate.score, stage, reason))
                continue
            share, vouchers = account_of(task, skill, args, self._library)
            if share < MIN_ACCOUNTED_FOR:
                # Only now is it worth naming the words: the happy path pays for the
                # ratio and nothing else.
                missing = unaddressed(task.text, skill, args, family=vouchers)
                stage = "unaccounted"
                reason = (
                    f"{skill.name}: nothing in it or its arguments accounts for "
                    + ", ".join(f"{word!r}" for word in missing[:6])
                    + f", so its end state cannot be what {task.text!r} asks for"
                )
                reasons.append(reason)
                looked_at.append(Rejection(skill.name, candidate.score, "unaccounted", reason))
                log.info(
                    "planner.unaccounted",
                    task=task.text,
                    skill=skill.name,
                    score=round(candidate.score, 3),
                    unaccounted=", ".join(missing),
                )
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
                vouched_by=", ".join(f"{v.name}@{v.domain}" for v in vouchers) or None,
            )
            self._reuse = _Reuse(skill.name, args, _how_bound(skill, task), vouchers)
            return self._single(skill, args, route), Usage()

        fit = self._through_family(task, observation, reasons, looked_at)
        if fit is not None:
            return fit, Usage()

        self._report_miss(task, candidates, looked_at)
        return self._compose(task, observation, stage, reasons, looked_at)

    @staticmethod
    def _single(skill: Skill, args: dict[str, Any], route: Route) -> Plan:
        return Plan(
            steps=(*route.steps, SkillCall(skill.name, skill.domain, args)),
            skills_used=(skill.name,),
            estimated_ms=route.cost + skill.stats.mean_ms,
        )

    def _library(self) -> list[Skill]:
        """Every healthy skill on every site: where relatives are looked for."""
        return self._store.list()

    def _through_family(
        self,
        task: TaskSpec,
        observation: Observation,
        reasons: list[str],
        looked_at: list[Rejection],
    ) -> Plan | None:
        """The family path: run THIS site's member of a workflow the request binds
        against on any site.

        Reached only when no skill here could be run on its own words. Still entirely
        model-free - :func:`fit_through_family` reads the library and nothing else -
        and everything after it is the ordinary warm path: the member is routed to,
        run behind its own precondition and verifier, and judged by the same critic.
        """
        shelf = self._store.list(domain=task.domain)
        for fit in fit_through_family(task, shelf, self._library()):
            skill = fit.skill
            route = self._route_to(observation.fingerprint, skill.precondition)
            if route is None:
                reason = (
                    f"{skill.name}: in the family of {fit.via.name}@{fit.via.domain}, which "
                    f"binds this request, but no known route reaches its start screen"
                )
                reasons.append(reason)
                looked_at.append(Rejection(skill.name, fit.share, "no_route", reason))
                continue
            log.info(
                "planner.family_hit",
                task=task.text,
                skill=skill.name,
                domain=skill.domain,
                signature=render(skill.action_signature),
                bound_through=f"{fit.via.name}@{fit.via.domain}",
                learned=fit.via.provenance.task_text,
                args=fit.args,
                accounted_for=round(fit.share, 3),
                vouched_by=", ".join(f"{v.name}@{v.domain}" for v in fit.vouchers),
            )
            self._reuse = _Reuse(skill.name, fit.args, "family", fit.vouchers)
            return self._single(skill, fit.args, route)
        return None

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
                widened = self._reuse is not None and self._reuse.how in _WIDENED
                if widened:
                    reason = (
                        f"{reason} [bound by the {self._reuse.how} reader, so this is "  # type: ignore[union-attr]
                        "evidence against the binding and the skill is NOT demoted]"
                    )
                else:
                    self._demote(skill, f"failed on the warm path: {reason}")
                return used, PlanFailure(
                    stage="skill_failed",
                    reason=reason,
                    skill=skill.name,
                    demoted=not widened,
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

    def _remember(self, task: TaskSpec, used: Sequence[str]) -> None:
        """Write a proven request down as a :class:`~skillweaver.contracts.Precedent`.

        Reached only after the skill ran, its OWN verifier passed and the critic said
        the task is done - and the first of those is checked again here rather than
        assumed, because a skill with no verifier has had nothing proved about it and
        a precedent is a template every later request is bound against. A verdict the
        critic could not reach comes back ``ok=False`` and never gets this far.

        Only a single-skill run whose argument was read out of the TEXT is recorded:
        a caller who supplied ``params`` taught the binder nothing about wording, and
        a composed chain is the composer's reading, not a sentence this skill served.
        """
        reuse = self._reuse
        if reuse is None or reuse.how == "params" or list(used) != [reuse.skill]:
            return
        skill = self._find(reuse.skill, task.domain)
        record = getattr(self._store, "record_precedent", None)
        if skill is None or record is None or not skill.verifier_code:
            return
        if not all(isinstance(v, str | int | float | bool) for v in reuse.args.values()):
            return
        try:
            record(skill.name, skill.domain, Precedent(task.text, dict(reuse.args)))
        except SkillWeaverError as exc:  # a note is not worth failing a good run for
            log.warning("planner.precedent.unwritten", skill=skill.name, error=str(exc))

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


def _how_bound(skill: Skill, task: TaskSpec) -> str:
    """Which reader :func:`bind_args` used for a candidate it DID bind."""
    if _bind_args(skill, task) is not None:
        return "params"
    name = _text_bound(skill, task)
    found = _span_for(skill, name, task.text) if name is not None else None
    return found[1] if found is not None else "params"


def _vouchers(
    task: TaskSpec, skill: Skill, args: Mapping[str, Any], library: Sequence[Skill]
) -> tuple[Skill, ...]:
    """The relatives of ``skill`` allowed to speak for ``task`` in their own words.

    Shape AND intent, both: :func:`~skillweaver.skills.family.relatives` answers the
    first from the action signatures, and
    :func:`~skillweaver.skills.family.same_intent` the second, against each relative's
    own learned sentence and against ``skill``'s. A skill with no earned signature has
    no relatives, so for every library stored before families existed this is ``()``
    and the gate is exactly what it was.
    """
    if not earned(skill):
        return ()
    outside = task.text
    for value in args.values():
        outside = _without(outside, str(value))
    if not same_intent(task.text, skill, outside=outside):
        return ()
    return tuple(
        other
        for other in relatives(skill, library)
        if same_intent(task.text, other, outside=outside)
    )


def account_of(
    task: TaskSpec,
    skill: Skill,
    args: Mapping[str, Any],
    library: Callable[[], Sequence[Skill]],
) -> tuple[float, tuple[Skill, ...]]:
    """``(how much of task is accounted for, the relatives that had to vouch)``.

    The skill's own words are asked first and, when they clear
    :data:`MIN_ACCOUNTED_FOR`, nobody else is consulted and ``library`` is never
    called - so every run that was warm before families existed reads exactly the
    libraries it read then. Only a candidate that falls short on its own is given its
    family's words (:func:`_vouchers`), and the line it then has to clear is the same
    one. WHAT COUNTS as an account was widened; how much of one is needed was not.
    """
    share = accounted_for(task.text, skill, args)
    if share >= MIN_ACCOUNTED_FOR:
        return share, ()
    vouchers = _vouchers(task, skill, args, library())
    if not vouchers:
        return share, ()
    return accounted_for(task.text, skill, args, family=vouchers), vouchers


def fit_through_family(
    task: TaskSpec, shelf: Sequence[Skill], library: Sequence[Skill]
) -> list[FamilyFit]:
    """Every way ``task`` can be run by a member of ``shelf`` through its family.

    This is the captain's case: *adding to a cart is almost the same workflow
    everywhere, so those runs should aid each other*. A request that does not line up
    with the sentence THIS site's skill was learned from may line up with the sentence
    another site's skill was learned from; when the two skills perform the same
    workflow, the argument read through one is what the other needs.

    A fit needs all of:

    * the member and the relative are one family by SHAPE
      (:func:`~skillweaver.skills.family.relatives`, which also requires that both
      earned a signature from a verifier-passed run and still carry the verifier);
    * the request binds against the relative's proven sentence, from its TEXT - the
      whole of :func:`_bind_from_text`, guards and intent check included;
    * each side has exactly one parameter for the text to fill, so which value goes
      where is not a guess, and the value can be the member's declared type;
    * the request is the same INTENT as the member's own learned sentence too;
    * member, relatives and arguments together account for the request to the same
      :data:`MIN_ACCOUNTED_FOR` as any other candidate.

    Public and pure - it reads two lists - because
    :func:`~skillweaver.orchestrator.resolve_domain` asks the same question before a
    browser is open, and two implementations of "would the planner run this?" is the
    defect that function's docstring is about.

    Returns:
        Fits, best accounted-for first; empty when there is none.
    """
    fits: list[FamilyFit] = []
    for member in shelf:
        if not earned(member):
            continue
        slot = _text_bound(member, task)
        if slot is None:
            continue
        for relative in relatives(member, library):
            bound = _bind_from_text(relative, task)
            theirs = _text_bound(relative, task)
            if bound is None or theirs is None:
                continue
            value = _as_declared(str(bound[theirs]), member.params[slot])
            if value is _REFUSED:
                continue
            args: dict[str, Any] = {k: task.params[k] for k in member.params if k in task.params}
            args[slot] = value
            vouchers = _vouchers(task, member, args, library)
            if relative not in vouchers:
                continue  # the member's own learned sentence is a different errand
            share = accounted_for(task.text, member, args, family=vouchers)
            if share < MIN_ACCOUNTED_FOR:
                continue
            fits.append(FamilyFit(member, args, relative, vouchers, share))
            break
    fits.sort(key=lambda fit: (-fit.share, fit.skill.domain, fit.skill.name))
    return fits


def bind_args(skill: Skill, task: TaskSpec) -> dict[str, Any] | None:
    """The arguments ``skill`` would be called with for ``task``, or ``None`` to decline.

    The model-free half of the warm path, in one name: the values the caller supplied
    (:func:`_bind_args`), and failing that the one value the task's own wording
    differs from the sentence the skill was learned in (:func:`_bind_from_text`).

    Public because binding is half of "would the planner run this skill for this
    task?", and that whole question is asked in a second place - resolving which
    DOMAIN a bare ``run`` means, in
    :func:`~skillweaver.orchestrator.resolve_domain`. Asking it there with a
    reimplemented binder would be asking a different question in the same words: a
    parameterized repeat binds its argument from the task's wording, and
    :func:`~skillweaver.skills.retrieve.accounted_for` only clears
    :data:`MIN_ACCOUNTED_FOR` once that argument is in hand. One definition, so the
    two callers cannot drift apart.
    """
    args = _bind_args(skill, task)
    return args if args is not None else _bind_from_text(skill, task)


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

_CLAUSE_WORDS = frozenset({"and", "then", "also", "but", "after", "before"})
"""Words that start a second clause. A value read out of UNQUOTED text may not
introduce one: what follows it is a second errand, which is the composer's to read."""

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

    found = _span_for(skill, missing[0], task.text)
    if found is None:
        return None
    span, how = found
    name = missing[0]

    value = _as_declared(span, skill.params[name])
    if value is _REFUSED:
        return None
    if not same_intent(task.text, skill, outside=_without(task.text, span)):
        log.info(
            "planner.other_intent",
            skill=skill.name,
            domain=skill.domain,
            task=task.text,
            learned=skill.provenance.task_text,
            asked_verb=head_verb(task.text),
            learned_verb=head_verb(skill.provenance.task_text),
        )
        return None

    args: dict[str, Any] = {k: task.params[k] for k in skill.params if k in task.params}
    args[name] = value
    log.info(
        "planner.bound_from_text",
        skill=skill.name,
        domain=skill.domain,
        param=name,
        value=value,
        how=how,
        learned=skill.provenance.task_text,
        task=task.text,
        args=args,
    )
    return args


def _span_for(skill: Skill, name: str, asked: str) -> tuple[str, str] | None:
    """``(the text of asked that is name's value, how it was found)``, or ``None``.

    Three readers, strictest first, and the first to answer wins: the one-span diff
    (:func:`_differing_span`), the quoted template (:func:`_from_template`) and the
    slot (:func:`_through_slot`). ``how`` is ``"diff"``, ``"template"`` or ``"slot"``
    and travels into the log and into what a failed run is allowed to demote.
    """
    learned = skill.provenance.task_text or ""
    span = _differing_span(learned, asked)
    if span is not None:
        return span, "diff"
    span = _from_template(skill, name, learned, asked)
    if span is not None:
        return span, "template"
    span = _through_slot(skill, name, asked)
    return (span, "slot") if span is not None else None


def _without(text: str, span: str) -> str:
    """``text`` with the bound value cut out, so a product called *Clear Glass Set* is
    not read as the verb *clear* by the intent check."""
    at = text.casefold().find(span.casefold())
    return text if at < 0 else f"{text[:at]} {text[at + len(span) :]}"


def _text_bound(skill: Skill, task: TaskSpec) -> str | None:
    """The one required parameter ``task.params`` leaves for the task TEXT to supply,
    or ``None`` when there is not exactly one."""
    missing = [
        name
        for name, schema in skill.params.items()
        if name not in task.params and not _has_default(schema)
    ]
    return missing[0] if len(missing) == 1 else None


def _why_unbound(skill: Skill, task: TaskSpec) -> tuple[FailureStage, str]:
    """The stage and the sentence for a candidate :func:`bind_args` turned down.

    Asked only after the fact, so the happy path pays nothing for it. It separates the
    two refusals a reader must never confuse: the sentence did not line up at all
    (``unbindable_args``), and the sentence lined up but asks for a DIFFERENT errand
    (``other_intent``) - *remove X from my cart* against a skill learned as *add X to
    my cart*. The second is the one a family-widened library has to be seen to make.
    """
    name = _text_bound(skill, task)
    found = _span_for(skill, name, task.text) if name is not None else None
    if found is not None and _as_declared(found[0], skill.params[name]) is not _REFUSED:
        asked, known = head_verb(task.text), head_verb(skill.provenance.task_text)
        return (
            "other_intent",
            f"{skill.name}: the request lines up with {skill.provenance.task_text!r} but "
            f"asks for a different errand - it leads with {asked!r} "
            f"({intent_of(asked or '') or 'no known intent'}) where the skill was learned "
            f"under {known!r} ({intent_of(known or '') or 'no known intent'}), or names an "
            "opposing verb elsewhere - so it is not run",
        )
    asked, known = head_verb(task.text), head_verb(skill.provenance.task_text)
    mine, theirs = intent_of(known or ""), intent_of(asked or "")
    if mine is not None and theirs is not None and mine != theirs:
        # Refused on alignment first - *from my cart* is not *to my cart* - but the
        # sentence to show a reader is the one about the verb.
        return (
            "other_intent",
            f"{skill.name}: the request leads with {asked!r} ({theirs}) and the skill was "
            f"learned under {known!r} ({mine}); the same controls in the opposite "
            "direction are a different errand, so it is not run",
        )
    return (
        "unbindable_args",
        f"{skill.name}: task supplies no value for a required parameter, and its "
        f"wording does not line up with {skill.provenance.task_text!r}",
    )


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
    if (_words(asked_mid) & _CLAUSE_WORDS) - _words(learned_mid):
        # *add a box of Tide and then check out* is one span against *add a box of
        # Folgers ... to my cart*, framed and short enough - and it bound, as a product
        # called "Tide and then check out", accounted for at 1.00 by its own argument.
        return None

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


# -- binding through the slot a PROVEN request left ---------------------------------------


MEASURE_WORDS = frozenset(
    """
    bag bags bottle bottles box boxes bunch can cans carton cartons case cases copy copies
    couple dozen jar jars item items pack packs package packet pair pairs piece pieces pound
    pounds roll rolls set sets unit units
    """.split()
)
"""Words that say HOW MUCH of a thing is wanted and nothing about which errand it is.

*A box of* and *a bag of* frame the same slot in the same sentence, so they may differ
between the learned wording and the asked one. A closed list, like
:data:`FRAME_WORDS`: what is not on it is a content word, and a content word that
changed outside the slot is a different request. They are neutral only for LINING THE
SENTENCES UP - :func:`~skillweaver.skills.retrieve.accounted_for` still counts a
measure word nobody has an account of against the candidate."""

_CLAUSE_BREAKS = _CLAUSE_WORDS | frozenset({",", ";", ":", "."})
"""Tokens an UNQUOTED value may not contain when it is read through a slot. A value
that runs over one of these has swallowed a second clause - *Tide and then check out* -
and a second clause is a bigger errand, which is the composer's job."""

_NEUTRAL = (
    FRAME_WORDS
    | MEASURE_WORDS
    | frozenset(
        "please kindly can could would i we like want need help just now ok okay hey".split()
    )
)
"""What is ignored when two sentence HEADS or TAILS are compared for being the same
request: framing, measure and politeness. A word that is here is never the difference
between two errands; every word that is not here must agree."""


_LEAD_INS = frozenset({"a", "an", "the", "some", "me", "us", "my", "our"})
"""What may stand between a bare verb and the value it governs."""


def _content(tokens: Sequence[_Token]) -> list[str]:
    """The words of a sentence fragment that say which errand it is, in order."""
    return [t.key for t in tokens if t.word and t.key not in _NEUTRAL]


def _same_head(learned: Sequence[_Token], asked: Sequence[_Token]) -> bool:
    """Whether two sentence heads - everything before the value - ask for one thing.

    Every content word must agree, in order, with one licence: the FIRST, which is
    the verb in the imperative sentences tasks are written in, may differ. Whether the
    two verbs mean the same errand is NOT decided here -
    :func:`~skillweaver.skills.family.same_intent` decides it, over the whole request,
    for every text binding - and keeping the two questions apart is what lets a
    refusal say which one it was: *buy me a box of* lines up and is the same errand,
    *remove a box of* lines up and is not (``other_intent``), and *add a review of*
    does not line up at all.
    """
    mine, theirs = _content(learned), _content(asked)
    return len(mine) == len(theirs) and mine[1:] == theirs[1:]


_DIRECTIONS = frozenset({"to", "into", "onto", "from", "off", "out", "in", "on"})
"""Framing words that carry the DIRECTION of an errand, so :func:`_tail_fits` compares
them where every other comparison skips them. *To my cart* and *from my cart* differ in
nothing else."""


def _tail_fits(learned: Sequence[_Token], asked: Sequence[_Token]) -> bool:
    """Whether what FOLLOWS the value is the learned sentence's own tail, or the start
    of it, or nothing.

    A request may stop early - *add a box of Tide* for a skill learned as *add a box
    of ... to my cart* - because the verb already says the rest. It may never go on
    LONGER or go somewhere else: *from my cart* is not a prefix of *to my cart* (the
    direction of the errand lives in exactly that word, and ``from``/``to`` are
    compared here although they are framing everywhere else), and a new clause is a
    bigger errand.
    """
    mine = [t.key for t in learned if t.word and t.key not in (_NEUTRAL - _DIRECTIONS)]
    theirs = [t.key for t in asked if t.word and t.key not in (_NEUTRAL - _DIRECTIONS)]
    return theirs == mine[: len(theirs)]


def _templates(skill: Skill, name: str) -> list[tuple[str, _Slot]]:
    """Every sentence ``skill`` is proven to have served, with where ``name`` sat in it.

    A :class:`~skillweaver.contracts.Precedent` records the sentence AND the arguments
    of a verifier-passed run, so the slot is not inferred: it is wherever that run's
    own value occurs, exactly once, in that run's own sentence. Newest first, because
    the most recent wording is the likeliest to be repeated. A skill admitted before
    precedents existed falls back to :func:`_learned_slot`, which reads the slot out
    of a quotation or the parameter's described example.
    """
    found: list[tuple[str, _Slot]] = []
    for precedent in reversed(skill.precedents):
        value = precedent.args.get(name)
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            continue
        slot = _sole_occurrence(precedent.task_text, str(value))
        if slot is not None:
            found.append((precedent.task_text, slot))
    learned = skill.provenance.task_text or ""
    if not any(sentence == learned for sentence, _ in found):
        slot = _learned_slot(skill, name, learned)
        if slot is not None:
            found.append((learned, slot))
    return found


def _through_slot(skill: Skill, name: str, asked: str) -> str | None:
    """``asked``'s value for ``name``, read through a sentence that is KNOWN to work.

    The two readers before this one compare whole sentences, so they answer only when
    the request is the learned sentence with one framed span changed. Measured on
    2026-09-19 against a skill learned from *add a box of Folgers classic roast ground
    coffee to my cart*::

        add a box of Starbucks classic roast ground coffee    does not bind
        add a bag of Starbucks Pike Place coffee to my cart   does not bind
        buy me a box of Tide laundry detergent                does not bind

    The first fails because the diff is *Folgers* -> *Starbucks* and the word after it,
    *classic*, is not a framing word - the reader cannot know the value runs on for
    four more words. But the SKILL knows: the run that proved it recorded the value it
    was given (:func:`_templates`), so where the value sat is a fact. With the slot in
    hand a request is read as ``head + value + tail``, and it binds when the head asks
    for the same thing (:func:`_same_head`) and the tail is the learned one or stops
    short of it (:func:`_tail_fits`).

    It still declines more than it binds. A value that is not quoted has to end
    somewhere, and the only places it may end are where the learned tail begins or at
    the end of the request - and then only if it swallowed no clause break and none of
    the learned tail's own words, so *add a box of Tide to the basket* is not read as
    a product called *Tide to the basket*.
    """
    rhs = _tokens(asked)
    quoted = _quoted_spans(asked)
    for sentence, slot in _templates(skill, name):
        head = _tokens(sentence[: slot.start])
        tail = _tokens(sentence[slot.end :])
        if len(quoted) == 1:
            found = quoted[0]
            if _same_head(head, _tokens(asked[: found.start])) and _tail_fits(
                tail, _tokens(asked[found.end :])
            ):
                return found.value
            continue
        if quoted:
            continue  # several quotations: which one is the value is a guess
        value = _unquoted_value(head, tail, rhs, asked, len(_tokens(slot.value)))
        if value is not None:
            return value
    return None


def _unquoted_value(
    head: Sequence[_Token],
    tail: Sequence[_Token],
    rhs: Sequence[_Token],
    asked: str,
    learned_words: int,
) -> str | None:
    """The span of ``asked`` between a head that matches and a tail that fits."""
    head_words = [t for t in head if t.word]
    if not head_words:
        return None  # a sentence that OPENS with its value has no head to line up
    anchor = head_words[-1].key
    # The value starts after the token that framed it in the learned sentence - or,
    # when that token was the verb itself, after the asked sentence's own verb.
    starts = [i + 1 for i, t in enumerate(rhs) if t.key == anchor]
    if not starts and anchor == head_verb(" ".join(t.key for t in head_words)):
        verb = head_verb(asked)
        starts = [i + 1 for i, t in enumerate(rhs) if t.key == verb][:1]
        # *buy me the Tide pods*: what follows a bare verb may open with an article
        # or a pronoun that belongs to the sentence and not to the product.
        while starts and starts[0] < len(rhs) - 1 and rhs[starts[0]].key in _LEAD_INS:
            starts[0] += 1
    tail_words = [t.key for t in tail if t.word]
    for start in starts:
        if not _same_head(head, rhs[:start]):
            continue
        ends = [i for i in range(start + 1, len(rhs)) if tail_words and rhs[i].key == tail_words[0]]
        for end in [*ends, len(rhs)]:
            span = rhs[start:end]
            if not _is_a_value(span) or not _tail_fits(tail, rhs[end:]):
                continue
            keys = {t.key for t in span}
            if keys & _CLAUSE_BREAKS:
                continue
            if end == len(rhs) and keys & set(tail_words):
                continue  # it ran to the end THROUGH the learned tail's own words
            if len(_words(span)) > max(3, learned_words + 2):
                continue
            value = asked[span[0].start : span[-1].end]
            if 0 < len(value) <= _MAX_VALUE_CHARS:
                return value
    return None


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
