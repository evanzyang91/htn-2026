"""The cold-versus-warm decision, and the wiring that makes it runnable.

Eighteen modules in this project each do one thing well. This one decides which of
them gets to answer a task, and that decision is the product::

    retrieve  ->  WARM: plan from stored skills and run it
                    |
                    +-- could not plan, or the plan ran and did not work
                    v
                  COLD: explore by trial and error
                    |
                    +-- succeeded -> synthesize, gate, store -> next time is WARM

Why the fall-through is reported rather than hidden
---------------------------------------------------

The tempting shape for this module is "try the fast thing, and if it does not work,
quietly do the slow thing" - a caller then only ever sees success, and the library
looks flawless while it silently rots. So every :class:`RunReport` carries an
:class:`AttemptRecord` **per path attempted**, including the warm attempt that failed,
what stage it failed at, and whether a skill was demoted for it. A warm run that was
rescued by exploration reports ``rescued`` and both records; it is the most
informative thing that can happen to this system and it is never flattened into a
plain success. :meth:`RunReport.explain` is the paragraph a human reads.

What "zero model calls" needs from this module
-----------------------------------------------

The warm path is model-free in its action loop only if the critic that judges it is
too, and :class:`~skillweaver.agent.critic.TieredCritic` is programmatic only when it
has something free to run. What this module supplies is :func:`recall_end_state`: the
screen the recorded run that TAUGHT the skill ended on, read back from the trajectory
store. A warm run that lands there is a decisive, free yes. It is handed over as
*corroboration* and not as evidence, because a mismatch means "this replay was given
a different argument" at least as often as it means "this replay failed" - see
:func:`recall_end_state`. When it does not match, or cannot be recalled at all, the
critic escalates to a model and pays for one call.

And a cost that is not counted is a cost that gets claimed as a saving, so
:class:`Agent` charges each attempt every model call made while it ran - read from
the client's own ``total_usage`` across the attempt - rather than only the calls that
survived into a returned outcome. A composer call spent on a plan that was then
thrown away at routing used to vanish from the report, and a warm path that is
cheaper on paper than in the bill is the one bug this project cannot ship.

Budgets
-------

:class:`~skillweaver.contracts.Budget` bounds a run four ways. The planner
deliberately lets ``BudgetExceeded`` escape so an exhausted run stops rather than
falling through to a slower path; :class:`Agent` honours that by recording the
exhaustion as the warm attempt's failure and **not** exploring afterwards. Falling
back to the expensive path after running out of money is the one fallback that is
always wrong.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from skillweaver.agent.compose import Composer
from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import Explorer
from skillweaver.agent.planner import MIN_ACCOUNTED_FOR, PlanFailure, Planner, bind_args
from skillweaver.config import Settings, settings
from skillweaver.contracts import (
    Budget,
    Candidate,
    Controller,
    Detector,
    Element,
    Fingerprint,
    Fingerprinter,
    LLMClient,
    Navigate,
    Observation,
    Perceiver,
    RunOutcome,
    Screenshot,
    SiteGraph,
    Skill,
    SkillStore,
    TaskSpec,
    TextReader,
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStore,
    Usage,
    utcnow,
)
from skillweaver.errors import BudgetExceeded, SkillWeaverError
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.graph.store import JSONGraphStore
from skillweaver.logging_ import get_logger
from skillweaver.perception.elements import build_index, merge_elements
from skillweaver.perception.fingerprint import StateFingerprinter
from skillweaver.perception.ocr import (
    DEFAULT_CACHE_SIZE,
    CachingTextReader,
    PerceptionCounters,
    PerceptionCounts,
)
from skillweaver.skills.retrieve import SkillRetriever, accounted_for
from skillweaver.skills.sandbox import SkillRunner
from skillweaver.skills.store import FileSkillStore
from skillweaver.skills.synthesize import (
    Admission,
    EnvironmentFactory,
    ReplayEnvironment,
    Synthesizer,
)
from skillweaver.trajectory.record import Recorder
from skillweaver.trajectory.store import TrajectoryFileStore

__all__ = [
    "READ_ONLY_PARAM",
    "RESET_URL_PARAM",
    "Agent",
    "AttemptRecord",
    "ComposedPerceiver",
    "DomainChoice",
    "EnvironmentFor",
    "NavigatingEnvironment",
    "PerceptionCounts",
    "Recollection",
    "ResetOutcome",
    "ResetRefused",
    "ResetReport",
    "RunReport",
    "SynthesisFactory",
    "Workbench",
    "WorldReset",
    "build_agent",
    "budget_from",
    "build_workbench",
    "navigating_environment",
    "perception_counts",
    "recall",
    "recall_end_state",
    "reset_world",
    "resolve_domain",
    "task_spec",
    "world_reset_from_url",
]

log = get_logger(__name__)

Path_ = Literal["warm", "cold"]

SynthesisFactory = Callable[[Trajectory], Synthesizer]
"""Builds the synthesizer for one trajectory.

A factory rather than an instance because the admission gate's critic wants to know
where the recorded run ENDED, and that is only knowable once the run has happened.
"""

EnvironmentFor = Callable[[Trajectory], EnvironmentFactory | None]
"""Builds the admission gate's replay environment for one trajectory, or declines.

The gate re-runs a candidate skill from the screen the recording STARTED on, and where
that is depends on the run: an exploration that rescued a failed warm attempt began
wherever the warm attempt left the screen, not at the task's start URL. So the reset is
resolved per trajectory rather than per task.

``None`` means this world cannot be put back there - a desktop that cannot navigate, a
run whose first screen has no URL - and the caller then admits nothing and says why.
"""

WorldReset = Callable[[], None]
"""Put the world back the way it was before the run: "how do I undo this?".

Re-opening a screen is not this. Most tasks worth learning CHANGE something - a
message is archived, an invoice is paid, a row is deleted - and the screen the run
started on no longer exists once they have. Without a way back, the admission gate
can never stand a candidate where the recording stood, so no skill for such a task
can ever be admitted. That is not a corner case; it is most of the product.

What counts as one is whatever that world actually offers: a seed-restoring endpoint
(:func:`world_reset_from_url`), a database snapshot rolled back, a container
replaced, a fresh profile. skillweaver does not care which - it only needs the call.

Raises (by contract): anything, and the caller treats a failure as "not restored"
rather than as a crash; a gate that cannot reset still has a report to write.
:class:`ResetRefused` is the one distinguished failure - "this endpoint is not a reset
hook and never will be" as opposed to "it did not work this time" - and
:class:`ResetReport` is how the difference reaches a human.
"""


# --------------------------------------------------------------------------------------
# What a run reports about itself
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One path the agent tried, and what came of it.

    Attributes:
        path: ``"warm"`` (compose stored skills) or ``"cold"`` (explore).
        ok: Whether this attempt achieved the task.
        reason: One human-readable line. On a warm failure this is the planner's
            :class:`~skillweaver.agent.planner.PlanFailure` reason; on a cold failure
            it is the explorer's rendered diagnosis.
        stage: Where it stopped. For warm, a
            :data:`~skillweaver.agent.planner.FailureStage` or ``"empty_library"`` /
            ``"budget"``; for cold, the diagnosis' ``stopped_by``. Empty on success.
        skills_used: The stored skills this attempt ran, in order.
        steps: Controller actions performed.
        llm_calls: Model calls this attempt was charged for. **Zero is the claim the
            warm path exists to make**, which is exactly why it is read from the
            model client across the attempt rather than from the plan the attempt
            happened to return: a call spent on a plan that was discarded is still a
            call, and leaving it out flatters every efficiency figure downstream.
        usd: Model spend charged to this attempt, counted the same way.
        seconds: Wall-clock seconds the attempt was charged.
        perception: What the eyes did during this attempt - observations, captures,
            detections and, above all, OCR reads against cache hits. It sits beside
            ``llm_calls`` and ``usd`` because it is the same kind of fact: the cost
            of the attempt, stated as a count so it means the same thing on a busy
            machine as on an idle one.
        demoted: The skill retired because it ran and failed, or ``None``.
        performed_nothing: Whether the screen is untouched, so the next path may
            start from it as it stands.
        run_id: The trajectory this attempt recorded, when it recorded one.
        trace: The failed skill's sandbox trace - a play-by-play of what the library
            thought would work, and the most useful thing a failed warm run leaves.
    """

    path: Path_
    ok: bool
    reason: str
    stage: str = ""
    skills_used: tuple[str, ...] = ()
    steps: int = 0
    llm_calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    perception: PerceptionCounts = PerceptionCounts()
    demoted: str | None = None
    performed_nothing: bool = True
    run_id: str | None = None
    trace: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.path} path: {'ok' if self.ok else 'failed'}"
        where = f" at {self.stage}" if self.stage else ""
        chain = f" [{' -> '.join(self.skills_used)}]" if self.skills_used else ""
        eyes = f", {self.perception}" if self.perception else ""
        return f"{head}{where}{chain} ({self.llm_calls} model call(s){eyes}) - {self.reason}"


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything one :meth:`Agent.run` did, and why it did it that way.

    This type exists so that "which path ran?" is never a guess. ``decision`` names
    the path that produced the answer; ``attempts`` holds every path that was tried,
    failures included, in the order they were tried.

    Attributes:
        ok: Whether the task was achieved by any path.
        task: What was asked.
        decision: The path that produced the result, or ``"none"`` when none did.
        attempts: One :class:`AttemptRecord` per path tried, in order.
        candidates: The skills retrieval offered before anything was attempted.
        outcome: The winning path's :class:`~skillweaver.contracts.RunOutcome`, or
            the last failed one when nothing worked. ``None`` if nothing ran.
        admission: The admission gate's verdict, when a cold success was fed to it.
        learned: The skill that entered the library because of this run, or ``None``.
        learning_note: Why nothing was learned, when ``learned`` is ``None`` and the
            run succeeded cold.
    """

    ok: bool
    task: TaskSpec
    decision: Literal["warm", "cold", "none"]
    attempts: tuple[AttemptRecord, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    outcome: RunOutcome | None = None
    admission: Admission | None = None
    learned: Skill | None = None
    learning_note: str = ""

    @property
    def warm(self) -> AttemptRecord | None:
        """The warm attempt, or ``None`` when the warm path was not tried."""
        return next((a for a in self.attempts if a.path == "warm"), None)

    @property
    def cold(self) -> AttemptRecord | None:
        """The cold attempt, or ``None`` when exploration was not reached."""
        return next((a for a in self.attempts if a.path == "cold"), None)

    @property
    def rescued(self) -> bool:
        """Whether the library RAN something, it did not work, and exploration saved it.

        The single most important thing this report can say: the library believed it
        knew this task, acted on that belief, and was wrong. Treating that as a plain
        success is how a skill library quietly stops being true.

        A warm path that declined before touching the screen - an empty library, no
        candidate, no route - is not a rescue. It is an ordinary cold start, and
        calling it a rescue would cry wolf on every first run.
        """
        warm, cold = self.warm, self.cold
        return (
            warm is not None
            and not warm.ok
            and not warm.performed_nothing
            and cold is not None
            and cold.ok
        )

    @property
    def warm_missed(self) -> bool:
        """Whether the library was consulted, could not plan the task, and exploring paid.

        :attr:`rescued` is the louder cousin of this: there the library RAN something
        and was wrong. Here it declined before touching the screen - an empty
        namespace, no candidate, no route - and the run then explored.

        That is a legitimate cold start on a task nobody has taught yet, and it is
        also exactly what a lookup in the WRONG namespace looks like. The two are
        indistinguishable from the outside, which is the whole problem: ``run``
        without a ``--url`` used to resolve its domain to the literal string
        ``"browser"``, miss a library it was standing next to, explore from a blank
        page and print ``SOLVED by the cold path``. True, and useless. So every
        fall-through says that the library was consulted and what it said, and the
        verdict line says the answer cost full price - see :meth:`explain`. A cold
        path that is reported as a plain success is how a defect like that survives
        for months.
        """
        warm, cold = self.warm, self.cold
        return (
            warm is not None
            and not warm.ok
            and warm.performed_nothing
            and cold is not None
            and cold.ok
        )

    @property
    def llm_calls(self) -> int:
        """Model calls across every attempt of this run."""
        return sum(a.llm_calls for a in self.attempts)

    @property
    def steps(self) -> int:
        """Controller actions performed across every attempt of this run."""
        return sum(a.steps for a in self.attempts)

    @property
    def perception(self) -> PerceptionCounts:
        """What the eyes did across every attempt of this run.

        The headline this whole optimization is judged by is
        ``report.perception.ocr_reads``: a task that used to need one OCR read per
        observation and now needs one per CHANGED screen says so here, in a number that
        does not move when the machine is loaded.
        """
        total = PerceptionCounts()
        for attempt in self.attempts:
            total = total + attempt.perception
        return total

    @property
    def run_id(self) -> str | None:
        """The trajectory id of the path that produced the result."""
        return self.outcome.trajectory.run_id if self.outcome is not None else None

    def explain(self) -> str:
        """The paragraph a human reads: which path ran, why, and what it cost."""
        lines = [f"task: {self.task.text}", f"domain: {self.task.domain}"]
        if self.candidates:
            offered = ", ".join(f"{c.skill.name} ({c.score:.2f})" for c in self.candidates)
            lines.append(f"retrieved: {offered}")
        else:
            lines.append(
                f"retrieved: nothing - the library holds no candidate for this task "
                f"under domain {self.task.domain!r}"
            )
        lines += [f"  {attempt}" for attempt in self.attempts]
        if self.rescued:
            lines.append(
                "NOTE: a stored skill ran, did not work, and exploration rescued the "
                "run. The library was WRONG about this task, not merely slow."
            )
        elif self.warm_missed:
            warm = self.warm
            assert warm is not None  # warm_missed says so
            lines.append(
                f"NOTE: the library was consulted first and missed at {warm.stage} - "
                f"{warm.reason}. Exploration then paid the full cold-path price for "
                f"this answer."
            )
        if self.learned is not None:
            lines.append(
                f"learned: {self.learned.name} v{self.learned.version} "
                f"({self.task.domain}) - the next run of this task can be warm"
            )
        elif self.learning_note:
            lines.append(f"learned: nothing - {self.learning_note}")
        eyes = self.perception
        if eyes:
            lines.append(
                f"perception: {eyes.ocr_reads} OCR read(s) for {eyes.observations} "
                f"observation(s) - {eyes.ocr_hits} served from cache "
                f"({eyes.hit_rate:.0%}), {eyes.detections} detection(s)"
            )
        verdict = "SOLVED" if self.ok else "NOT SOLVED"
        # "SOLVED by the cold path" on its own reads as a plain success, and after a
        # warm attempt that was tried and missed it is the single most misleading
        # sentence this report can end on: the fast path did not apply, and the
        # headline is where that has to be said.
        after = " AFTER A WARM MISS" if (self.rescued or self.warm_missed) else ""
        lines.append(
            f"{verdict} by the {self.decision} path{after} "
            f"in {self.steps} action(s) and {self.llm_calls} model call(s)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------------------


class Agent:
    """Runs one task by deciding between the warm and the cold path.

    Args:
        controller: The hands, already open and pointed at the right world.
        perceiver: The eyes.
        store: The skill library. Read for the warm path, written by admission.
        retriever: Finds candidates. Consulted once here for the report, and again
            inside the planner - both are lexical-and-embedding lookups outside the
            action loop, and the duplicate buys a report that names what was offered.
        planner: The warm path.
        explorer: The cold path.
        graph: The site graph. Saved after a run so what was learned survives.
        trajectories: Where a cold run's trajectory is persisted.
        synthesis: Builds the synthesizer for a finished trajectory. ``None``
            disables learning, and the report says so rather than pretending.
        environment: Builds the admission gate's world for a finished trajectory,
            reset to the screen that run STARTED on. ``None`` - or a call that
            returns ``None`` - disables learning for this run, because a skill that
            has not been re-run has not proved anything.
        llm: The model client the planner, the explorer and the critics were wired
            with. Never called here: it is read, through ``total_usage``, to charge
            each attempt every call made while it ran. ``None`` falls back to what
            each attempt reports about itself, which undercounts a plan that was
            discarded after the composer had already paid for it.
        budget: Limits for one :meth:`run`.
        top_k: How many candidates to retrieve for the report.
    """

    __slots__ = (
        "_budget",
        "_controller",
        "_environment",
        "_explorer",
        "_graph",
        "_llm",
        "_perceiver",
        "_planner",
        "_retriever",
        "_store",
        "_synthesis",
        "_top_k",
        "_trajectories",
    )

    def __init__(
        self,
        *,
        controller: Controller,
        perceiver: Perceiver,
        store: SkillStore,
        retriever: SkillRetriever,
        planner: Planner,
        explorer: Explorer,
        graph: SiteGraph | None = None,
        trajectories: TrajectoryStore | None = None,
        synthesis: SynthesisFactory | None = None,
        environment: EnvironmentFor | None = None,
        llm: LLMClient | None = None,
        budget: Budget | None = None,
        top_k: int = 5,
    ) -> None:
        self._controller = controller
        self._perceiver = perceiver
        self._store = store
        self._retriever = retriever
        self._planner = planner
        self._explorer = explorer
        self._graph = graph
        self._trajectories = trajectories
        self._synthesis = synthesis
        self._environment = environment
        self._llm = llm
        self._budget = budget if budget is not None else Budget()
        self._top_k = top_k

    def __repr__(self) -> str:
        return f"Agent(learns={self._synthesis is not None and self._environment is not None})"

    @property
    def controller(self) -> Controller:
        """The open world. Exposed so a command can replay a recorded run against the
        same browser this session opened, rather than opening a second one."""
        return self._controller

    @property
    def perceiver(self) -> Perceiver:
        """The eyes, for the same reason."""
        return self._perceiver

    @property
    def budget(self) -> Budget:
        """The limits this agent will run one task under."""
        return self._budget

    # -- the decision ------------------------------------------------------------------

    def run(
        self, task: TaskSpec, *, learn: bool = True, warm: bool = True, cold: bool = True
    ) -> RunReport:
        """Do ``task`` and report which path did it.

        Args:
            task: What to do.
            learn: Whether a cold success should be offered to the admission gate.
            warm: Whether the warm path may be tried at all. ``False`` forces
                exploration, which is what ``learn`` on the command line wants.
            cold: Whether exploration may be tried. ``False`` makes this a
                library-only run that reports honestly when the library falls short.

        Returns:
            A :class:`RunReport`. A failure is a report, not an exception.

        Raises:
            ControllerError: if the controller breaks mid-run.
            ProviderError: if a model call fails outright.
        """
        candidates = self._retrieve(task)
        attempts: list[AttemptRecord] = []

        if warm:
            mark, spent = perception_counts(self._perceiver), self._usage()
            record, outcome = self._try_warm(task)
            record = self._charge_model(self._charge_eyes(record, mark), spent)
            attempts.append(record)
            if record.ok and outcome is not None:
                self._persist()
                return RunReport(
                    ok=True,
                    task=task,
                    decision="warm",
                    attempts=tuple(attempts),
                    candidates=candidates,
                    outcome=outcome,
                    learning_note="the skill was already in the library",
                )
            if record.stage == "budget":
                # The planner lets BudgetExceeded escape so an exhausted run STOPS.
                # Exploring now would spend money the run has already been told it
                # does not have.
                self._persist()
                return RunReport(
                    ok=False,
                    task=task,
                    decision="none",
                    attempts=tuple(attempts),
                    candidates=candidates,
                )

        if not cold:
            return RunReport(
                ok=False,
                task=task,
                decision="none",
                attempts=tuple(attempts),
                candidates=candidates,
            )

        mark, spent = perception_counts(self._perceiver), self._usage()
        record, outcome = self._try_cold(task)
        record = self._charge_model(self._charge_eyes(record, mark), spent)
        attempts.append(record)
        learned, admission, note = self._learn(task, outcome, enabled=learn)
        self._persist()
        return RunReport(
            ok=record.ok,
            task=task,
            decision="cold" if record.ok else "none",
            attempts=tuple(attempts),
            candidates=candidates,
            outcome=outcome,
            admission=admission,
            learned=learned,
            learning_note=note,
        )

    def _charge_eyes(self, record: AttemptRecord, mark: PerceptionCounts) -> AttemptRecord:
        """Attribute the perception work done since ``mark`` to ``record``.

        Counted here rather than inside each attempt because the perceiver is shared by
        the planner, the explorer and every skill they run, so the only honest boundary
        for "what this attempt made the eyes do" is the attempt itself.
        """
        return replace(record, perception=perception_counts(self._perceiver) - mark)

    def _charge_model(self, record: AttemptRecord, mark: Usage) -> AttemptRecord:
        """Charge ``record`` every model call made since ``mark``.

        The attempt's own number is kept when it is the larger one, so a path that
        counts a call this meter cannot see - a second client, a model reached
        through something other than ``llm`` - is never talked DOWN by this. The
        meter's job is the opposite direction: a call that was made and then lost,
        because the plan it bought was discarded at routing and its outcome thrown
        away with it, is found again here. Undercounting is the only failure mode
        that makes this project look better than it is.
        """
        spent = self._usage()
        calls = max(record.llm_calls, spent.calls - mark.calls)
        usd = max(record.usd, spent.cost_usd - mark.cost_usd)
        return replace(record, llm_calls=calls, usd=usd)

    def _usage(self) -> Usage:
        """What the model client has been asked for so far, or an empty tally.

        :class:`~skillweaver.contracts.LLMClient` does promise ``total_usage``, but a
        stub wired in by a test need not, and a report is not worth an exception.
        """
        total = getattr(self._llm, "total_usage", None)
        if total is None:
            return Usage()
        try:
            result = total()
        except Exception:  # noqa: BLE001 - a broken meter must not fail a run
            log.warning("agent.usage.unreadable", llm=type(self._llm).__name__)
            return Usage()
        return result if isinstance(result, Usage) else Usage()

    # -- the warm path -----------------------------------------------------------------

    def _try_warm(self, task: TaskSpec) -> tuple[AttemptRecord, RunOutcome | None]:
        """Plan from the library and run the plan, or say why not.

        Nothing is performed when the library holds nothing for the domain, so that
        case is answered without even reading the screen.
        """
        if not self._store.list(domain=task.domain):
            return (
                AttemptRecord(
                    path="warm",
                    ok=False,
                    stage="empty_library",
                    reason=(
                        f"the library holds no skill for domain {task.domain!r}"
                        f"{self._held_elsewhere(task.domain)}"
                    ),
                ),
                None,
            )

        observation = self._perceiver.observe(self._controller)
        try:
            outcome = self._planner.attempt(task, observation)
        except BudgetExceeded as exc:
            return (
                AttemptRecord(
                    path="warm",
                    ok=False,
                    stage="budget",
                    reason=f"the warm attempt ran out of budget: {exc}",
                    performed_nothing=False,
                ),
                None,
            )

        if outcome is not None:
            used = tuple(outcome.skill_used.split("+")) if outcome.skill_used else ()
            log.info("agent.warm", task=task.text, skills=outcome.skill_used, llm=0)
            return (
                AttemptRecord(
                    path="warm",
                    ok=True,
                    reason=outcome.note or "the stored skill did the task",
                    skills_used=used,
                    steps=outcome.spend.steps,
                    llm_calls=outcome.spend.llm_calls,
                    usd=outcome.spend.usd,
                    seconds=outcome.spend.elapsed_seconds(),
                    performed_nothing=False,
                    run_id=outcome.trajectory.run_id,
                ),
                outcome,
            )

        failure = self._planner.last_failure
        return self._warm_failure(task, failure), None

    def _held_elsewhere(self, domain: str) -> str:
        """What the library holds under OTHER domains, as a clause to append.

        "The library holds no skill for domain 'browser'" is true and stops one
        question short of the answer. Naming the namespaces that DO hold something is
        what turns it into a diagnosis: a reader who sees ``'browser'`` empty while
        ``en.wikipedia.org`` holds four skills has been handed the bug rather than a
        shrug. Empty is worth saying too - it is the ordinary first run.
        """
        try:
            others = sorted({s.domain for s in self._store.list() if s.domain != domain})
        except SkillWeaverError:  # a report is not worth failing a run for
            return ""
        if not others:
            return "; the library holds nothing under any other domain either"
        named = ", ".join(others[:4]) + (", ..." if len(others) > 4 else "")
        return f"; it DOES hold skills under {len(others)} other domain(s): {named}"

    @staticmethod
    def _warm_failure(task: TaskSpec, failure: PlanFailure | None) -> AttemptRecord:
        """A warm attempt that produced no outcome, reported in full."""
        if failure is None:  # the planner always leaves one; defend the report anyway
            return AttemptRecord(
                path="warm",
                ok=False,
                stage="unknown",
                reason="the planner declined without saying why",
            )
        log.info("agent.warm.failed", task=task.text, stage=failure.stage, why=failure.reason)
        return AttemptRecord(
            path="warm",
            ok=False,
            stage=failure.stage,
            reason=failure.reason,
            skills_used=(failure.skill,) if failure.skill else (),
            demoted=failure.skill if failure.demoted else None,
            performed_nothing=failure.performed_nothing,
            trace=failure.trace,
        )

    # -- the cold path -----------------------------------------------------------------

    def _try_cold(self, task: TaskSpec) -> tuple[AttemptRecord, RunOutcome]:
        """Explore, and persist whatever the run recorded - success or not.

        A failed exploration is saved too: its trajectory names the screen the agent
        was stuck on and everything it had tried there, which is the raw material for
        the next attempt and for the dashboard.
        """
        outcome = self._explorer.explore(task, self._controller, self._budget)
        self._save(outcome.trajectory)
        log.info(
            "agent.cold",
            task=task.text,
            ok=outcome.ok,
            steps=outcome.spend.steps,
            llm_calls=outcome.spend.llm_calls,
        )
        # ``diagnosis`` belongs to ExplorationOutcome, not to the Explorer Protocol,
        # so an explorer that only promises the Protocol still reports a stage.
        diagnosis = getattr(outcome, "diagnosis", None)
        stage = "" if outcome.ok else (getattr(diagnosis, "stopped_by", "") or "gave-up")
        return (
            AttemptRecord(
                path="cold",
                ok=outcome.ok,
                stage=stage,
                reason=outcome.note or ("explored to the goal" if outcome.ok else "failed"),
                steps=outcome.spend.steps,
                llm_calls=outcome.spend.llm_calls,
                usd=outcome.spend.usd,
                seconds=outcome.spend.elapsed_seconds(),
                performed_nothing=False,
                run_id=outcome.trajectory.run_id,
            ),
            outcome,
        )

    # -- learning ----------------------------------------------------------------------

    def _learn(
        self, task: TaskSpec, outcome: RunOutcome, *, enabled: bool
    ) -> tuple[Skill | None, Admission | None, str]:
        """Offer a successful cold run to the admission gate.

        Returns ``(stored skill, admission, note)``. The note says why nothing was
        learned; an empty note with no skill means learning was not attempted.
        """
        if not outcome.ok:
            return None, None, "the run did not succeed, so there is nothing to learn from"
        if not enabled:
            return None, None, "learning was switched off for this run"
        if self._synthesis is None:
            return None, None, "this agent was built without a synthesizer"
        factory = self._environment(outcome.trajectory) if self._environment is not None else None
        if factory is None:
            return (
                None,
                None,
                "this run cannot be replayed from the screen it started on, and a "
                "skill that has not been re-run has proved nothing, so none was admitted",
            )
        admission = self._synthesis(outcome.trajectory).admit(outcome.trajectory, factory)
        if admission.ok and admission.skill is not None:
            log.info(
                "agent.learned",
                task=task.text,
                skill=admission.skill.name,
                version=admission.skill.version,
            )
            return admission.skill, admission, ""
        return None, admission, self._with_reset_cause(admission.reason)

    def _with_reset_cause(self, reason: str) -> str:
        """Name what the reset actually did, when the environment kept a record.

        The gate is handed one bool and so can only say "restored" or "not restored".
        A reader chasing a failed learning step needs the other half: an endpoint that
        refused the job is a wrong flag, a hook that timed out is a world having a bad
        day, and nothing configured at all is a mutating task nobody gave a way back.
        Those have three different fixes and the bool spells them the same way.
        """
        report = getattr(self._environment, "last_reset", None)
        if not isinstance(report, ResetReport) or report.restored:
            return reason
        return f"{reason} [{report}]" if reason else str(report)

    # -- bookkeeping -------------------------------------------------------------------

    def _retrieve(self, task: TaskSpec) -> tuple[Candidate, ...]:
        """What the library offers for this task. A retrieval failure is not a run
        failure: it means the warm path starts blind, not that the task is impossible."""
        try:
            return tuple(self._retriever.search(task.text, domain=task.domain, k=self._top_k))
        except SkillWeaverError as exc:
            log.warning("agent.retrieve.failed", task=task.text, error=str(exc))
            return ()

    def _save(self, trajectory: Trajectory) -> None:
        """Persist a trajectory, tolerating a store that cannot take it."""
        if self._trajectories is None:
            return
        try:
            self._trajectories.save(trajectory)
        except SkillWeaverError as exc:
            log.warning("agent.trajectory.save_failed", run_id=trajectory.run_id, error=str(exc))

    def _persist(self) -> None:
        """Write the site graph out. What a run learned about the map is worth
        keeping even when the run itself failed."""
        if self._graph is None:
            return
        try:
            self._graph.save()
        except SkillWeaverError as exc:
            log.warning("agent.graph.save_failed", error=str(exc))


# --------------------------------------------------------------------------------------
# Recalling where a task ends
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Recollection:
    """Where this task ended last time, and which stored skill remembered it.

    The second half is what decides how much authority the first half gets. A skill
    that carries a ``verifier_code`` has already answered "did I do my job?" in the
    sandbox, before the critic is asked anything; its recalled end screen is a
    shortcut to a free yes and nothing more. A skill with no verifier has answered
    nothing, and then the recalled screen is the only evidence there is - so it keeps
    its veto, and a stored skill that runs cleanly while finishing half the errand is
    still caught for free.

    Attributes:
        state: The fingerprint of the screen the recorded run ended on.
        source: The stored skill whose trajectory it was read back from.
    """

    state: Fingerprint | None = None
    source: Skill | None = None

    @property
    def self_checking(self) -> bool:
        """Whether the remembering skill can check its own work."""
        return self.source is not None and bool(self.source.verifier_code)


def recall(
    store: SkillStore,
    trajectories: TrajectoryStore | None,
    task: TaskSpec,
    candidates: Sequence[Candidate] = (),
) -> Recollection:
    """:func:`recall_end_state`, with the skill it came from kept.

    Read that one first; this exists because the answer is only half useful without
    knowing who remembered it. See :class:`Recollection`.
    """
    if trajectories is None:
        return Recollection()
    skills = [c.skill for c in candidates] or store.list(domain=task.domain)
    for skill in skills:
        run_id = skill.provenance.trajectory_id
        if not run_id:
            continue
        try:
            trajectory = _load_light(trajectories, run_id)
        except SkillWeaverError:
            continue
        if trajectory.ok and trajectory.steps:
            return Recollection(trajectory.steps[-1].after.fingerprint, skill)
    return Recollection()


def recall_end_state(
    store: SkillStore,
    trajectories: TrajectoryStore | None,
    task: TaskSpec,
    candidates: Sequence[Candidate] = (),
) -> Fingerprint | None:
    """The screen a successful run of this task ended on, or ``None``.

    This is what lets the warm path be judged for free. A stored skill records the
    trajectory it was synthesized from; that trajectory's last screen is where the
    task finished when it demonstrably worked. Handing it to a
    :class:`~skillweaver.agent.critic.TieredCritic` as ``corroborating_state`` turns
    "did the task get done?" into a fingerprint comparison - and a match is decisive,
    and costs nothing.

    A MISMATCH is not decisive, and that distinction is the whole of it. This screen
    is where ONE run of this task ended, with the arguments that run was given. A
    skill learned from "Search Wikipedia for computer vision" and replayed for
    "machine learning" correctly ends on a different article; handing this fingerprint
    over as ``expected_state`` made that correct replay a failure, at similarity
    0.120, twice, measured on live Wikipedia. So it is offered as corroboration: a
    shortcut to a free yes with no power to say no.

    Without it the critic has no evidence to run and escalates to a model, so a warm
    run would still work but would no longer be model-free. That is a real cost and
    it is reported in ``spend`` rather than hidden - see :class:`Agent`, which charges
    an attempt every model call made while it ran, not only the ones a plan survived
    to report.

    Args:
        store: The library, searched when ``candidates`` is empty.
        trajectories: Where recorded runs live. ``None`` gives ``None``.
        task: The task being run; only its domain is used.
        candidates: Retrieval's offer, best first. The first one whose trajectory
            can still be read wins.

    Returns:
        The recalled fingerprint, or ``None`` when nothing could be read back.
    """
    return recall(store, trajectories, task, candidates).state


def _load_light(trajectories: TrajectoryStore, run_id: str) -> Trajectory:
    """Load a trajectory without its screenshots when the store can do that.

    ``TrajectoryStore`` does not promise the keyword - the file-backed store offers
    it and the in-memory one does not - and reading a run's pixels to look at one
    fingerprint is pure waste, so it is asked for and not insisted on.
    """
    try:
        return trajectories.load(run_id, screenshots=False)  # type: ignore[call-arg]
    except TypeError:
        return trajectories.load(run_id)


# --------------------------------------------------------------------------------------
# Eyes
# --------------------------------------------------------------------------------------


class ComposedPerceiver:
    """A :class:`~skillweaver.contracts.Perceiver` over a detector, a reader and a
    fingerprinter.

    Capture once, detect and read that one frame, fuse the two element lists, index
    them and fingerprint the result. Boxes are LOGICAL pixels throughout, because
    both producers are required to convert before they return.

    Reading is 84% to 97% of that, measured on every live page tried, so the reader is
    wrapped in a :class:`~skillweaver.perception.ocr.CachingTextReader` by default: an
    observation of a screen nothing has touched costs a capture and a detection, and no
    OCR at all. Everything the perceiver does is charged to :attr:`counters`, which is
    what a run reports so the saving can be stated as a count rather than as seconds
    measured on whatever else the machine happened to be doing.

    Args:
        detector: Finds controls. Required.
        reader: Reads text. ``None`` runs detection alone, which is what a machine
            without the OCR models can still do.
        fingerprinter: Identifies the screen. Defaults to
            :class:`~skillweaver.perception.fingerprint.StateFingerprinter`.
        cache_size: Frames the text cache remembers. ``0`` reads every frame afresh,
            which is how a caller turns the optimization off to measure against it.
        counters: The tally to charge work to. A fresh one is made when not given.

    Why the text is not read LAZILY
    -------------------------------

    The obvious next step is to defer the read until something actually asks for text,
    so the steps that never touch it never pay. It is implementable -
    :class:`~skillweaver.contracts.Observation` is a frozen slots dataclass, and a
    subclass whose ``elements``, ``index`` and ``fingerprint`` are properties over one
    memoized thunk passes ``isinstance``, equality and ``repr`` untouched - and it is
    worth nothing, because of a dependency that is easy to miss:

        ``fingerprint`` is computed FROM ``elements``, and ``elements`` includes the OCR
        ones. :class:`~skillweaver.perception.fingerprint.StateFingerprinter`'s
        structural hash bins every element by kind and position, text elements included,
        so asking a screen what state it is in already requires the read.

    Since every consumer in this project compares fingerprints - the explorer to
    remember a state, the critic to judge a move, the planner to check a precondition -
    a lazy field would materialize almost immediately. Measured rather than assumed: on
    two live Wikipedia exploration runs, **12 of 12 and 9 of 9 observations had their
    elements and fingerprint read**, so the laziness would have saved exactly zero reads
    in both. The saving is real only for a fingerprinter that does not consume OCR
    elements, and changing what ``StateFingerprinter`` hashes changes every stored graph
    node id and every stored skill precondition - a coordination decision, not a local
    one.

    This lives here rather than in :mod:`skillweaver.perception` only because that
    package has not grown a composing perceiver yet; it is wiring, and wiring is this
    module's job until it has a better home.
    """

    __slots__ = ("_counters", "_detector", "_fingerprinter", "_reader")

    def __init__(
        self,
        detector: Detector,
        reader: TextReader | None = None,
        fingerprinter: Fingerprinter | None = None,
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
        counters: PerceptionCounters | None = None,
    ) -> None:
        self._detector = detector
        self._reader: CachingTextReader | None
        if isinstance(reader, CachingTextReader):
            # A reader that already caches is reused rather than wrapped again, so two
            # perceivers sharing one warm cache share its tally instead of each counting
            # half the frames. It charges its reads where it was told to at construction,
            # and a tally that only some of the work reaches is worse than none.
            if counters is not None and counters is not reader.counters:
                raise ValueError(
                    "this reader already charges its reads to another PerceptionCounters; "
                    "pass that one, or pass an unwrapped reader"
                )
            self._reader = reader
            self._counters = reader.counters
        else:
            self._counters = counters if counters is not None else PerceptionCounters()
            self._reader = (
                None
                if reader is None
                else CachingTextReader(reader, capacity=cache_size, counters=self._counters)
            )
        self._fingerprinter = fingerprinter if fingerprinter is not None else StateFingerprinter()

    def __repr__(self) -> str:
        return (
            f"ComposedPerceiver(reader={self._reader is not None}, "
            f"counts={self._counters.snapshot()})"
        )

    @property
    def counters(self) -> PerceptionCounters:
        """The running tally of everything this perceiver has done."""
        return self._counters

    @property
    def reader(self) -> CachingTextReader | None:
        """The caching reader, or ``None`` when this perceiver reads no text."""
        return self._reader

    def observe(self, controller: Controller) -> Observation:
        """One frame, fully understood.

        Raises:
            ControllerError: if the capture fails.
            PerceptionError: if detection, reading or fingerprinting fails.
        """
        shot: Screenshot = controller.capture()
        self._counters.captures += 1
        found: list[Element] = self._detector.detect(shot)
        self._counters.detections += 1
        text: list[Element] = self._reader.read(shot) if self._reader is not None else []
        elements = tuple(merge_elements(found, text))
        url = controller.url()
        observation = Observation(
            screenshot=shot,
            elements=elements,
            index=build_index(elements),
            fingerprint=self._fingerprinter.fingerprint(shot, elements, url),
            url=url,
            taken_at=utcnow(),
        )
        self._counters.observations += 1
        return observation


def perception_counts(perceiver: Perceiver | None) -> PerceptionCounts:
    """What ``perceiver`` has done so far, or an empty tally when it does not count.

    :class:`~skillweaver.contracts.Perceiver` says nothing about counters - a fake in a
    test, or a perceiver another worker writes, is under no obligation to keep them - so
    a report asks politely and reports nothing rather than failing when the answer is no.
    """
    counters = getattr(perceiver, "counters", None)
    snapshot = getattr(counters, "snapshot", None)
    if snapshot is None:
        return PerceptionCounts()
    result = snapshot()
    return result if isinstance(result, PerceptionCounts) else PerceptionCounts()


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


RESET_URL_PARAM = "reset_url"
"""The task parameter holding a :data:`WorldReset` endpoint.

It rides in ``TaskSpec.params`` beside ``start_url`` rather than in a new argument
to the session factory, so a caller that already builds a workbench - the evaluation
harness, a test - keeps working unchanged and gains the hook by naming it.
"""


READ_ONLY_PARAM = "read_only"
"""The task parameter declaring that this task changes nothing.

It rides in ``TaskSpec.params`` beside ``start_url`` for the same reason
:data:`RESET_URL_PARAM` does, and because that makes it reachable from the command
line that already exists: ``-p read_only=true``.

A task that only reads and navigates needs no way back - re-opening the page IS the
way back - and saying so is what stops the admission gate reporting a precondition it
missed for some other reason as a world it could not restore. Live sites are the
whole case: Wikipedia has no reset endpoint, and pointing ``--reset-url`` at one
answers ``403``.
"""


def task_spec(
    text: str,
    *,
    domain: str | None = None,
    target: Literal["browser", "desktop"] = "browser",
    url: str | None = None,
    reset_url: str | None = None,
    read_only: bool = False,
    params: dict[str, Any] | None = None,
) -> TaskSpec:
    """Build a :class:`~skillweaver.contracts.TaskSpec` the way the command line does.

    The domain defaults to the host of ``url`` for a browser task, so
    ``--url https://example.com/invoices`` files its skills under ``example.com``
    without anyone having to say so twice. A desktop task with no domain gets
    ``"desktop"``.

    ``reset_url`` is how this world is put back before the admission gate re-runs a
    candidate; see :data:`WorldReset` for why a task that changes anything cannot be
    learned without one. ``read_only`` is the other answer to the same question - this
    task changes nothing, so there is nothing to put back - and see
    :data:`READ_ONLY_PARAM` for when that is the true one.
    """
    merged: dict[str, Any] = dict(params or {})
    if url:
        merged.setdefault("start_url", url)
    if reset_url:
        merged.setdefault(RESET_URL_PARAM, reset_url)
    if read_only:
        merged.setdefault(READ_ONLY_PARAM, True)
    resolved = domain or (_host(url) if target == "browser" else None) or target
    return TaskSpec(text=text, domain=resolved, target=target, params=merged)


def _host(url: str | None) -> str | None:
    """The host of a URL, or ``None`` when there is not one to take."""
    if not url:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(url if "//" in url else f"//{url}")
    return parsed.hostname or None


# --------------------------------------------------------------------------------------
# Which namespace a task means
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DomainChoice:
    """Which library namespace a task belongs to, and on whose authority.

    Attributes:
        domain: The namespace to file under and to look in.
        start_url: Where to open the world, when something knows. For a looked-up
            domain this is the page the winning skill's start screen was last seen
            at, which is what makes a bare repeat runnable at all.
        source: Who decided. ``"named"`` a ``--domain``; ``"url"`` the host of a
            ``--url``; ``"library"`` a stored skill that answered for this task;
            ``"target"`` nothing did, and the target's own name is the fallback.
        skill: The skill that answered, for ``"library"``. Empty otherwise.
        score: That skill's retrieval score.
        why: One line for a human: what was consulted and what it said.
    """

    domain: str
    start_url: str | None = None
    source: Literal["named", "url", "library", "target"] = "target"
    skill: str = ""
    score: float = 0.0
    why: str = ""

    @property
    def looked_up(self) -> bool:
        """Whether the library, rather than the caller, named this domain."""
        return self.source == "library"

    def __str__(self) -> str:
        head = f"{self.domain}"
        if self.skill:
            head += f" (from {self.skill}, {self.score:.2f})"
        return f"{head} - {self.why}" if self.why else head


def resolve_domain(
    text: str,
    *,
    target: Literal["browser", "desktop"] = "browser",
    url: str | None = None,
    domain: str | None = None,
    params: Mapping[str, Any] | None = None,
    retriever: SkillRetriever | None = None,
    graph: SiteGraph | None = None,
    k: int = 10,
) -> DomainChoice:
    """Which library namespace ``text`` means, when the caller did not say.

    A skill is filed under a domain, and a lookup happens in one. ``learn`` is given
    a ``--url``, so what it stores is filed under that host. A repeat has no reason to
    pass one - the whole point is that the agent already knows how - and resolving
    that silence to the literal target (``"browser"``) files the lookup in a namespace
    nothing is ever stored in. The miss is then guaranteed, and the run explores from
    a blank page and reports a plain success at full cold-path price. That is not a
    corner case: it is the two commands this project's own README tells a new user to
    type, and it made the project's entire claim untestable from the command line.

    So the silence is resolved by ASKING THE LIBRARY instead of guessing: every
    domain is searched, and the best candidate that the planner would actually run
    names its own domain. Where the caller did say - a ``--domain``, or a ``--url``
    to take the host of - nothing is looked up and what they said stands.

    Ranking is not the authority here, and cannot be
    ------------------------------------------------

    Retrieval RANKS, so it has a winner whenever the library is non-empty, and a
    library with one skill in it ranks that skill first for every sentence in the
    world. Letting the top of a cross-domain ranking name the domain would therefore
    send a Wikipedia task to a grocery site for no better reason than that the
    grocery site was the only thing stored - the same mistake, in a new place, that
    :data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR` was calibrated to stop
    (short generic names rank perfectly and account for nothing; see
    :mod:`skillweaver.skills.retrieve`).

    So a candidate only gets to name the domain when it passes the planner's own
    admission question: does it account for the whole request
    (:func:`~skillweaver.skills.retrieve.accounted_for` against
    :data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR`)? That is the same test, run
    with the same code, that decides whether a skill is worth performing at all. When
    nothing passes it, nothing is resolved and ``source`` is ``"target"``, which the
    report then states rather than dressing up as a cold start that was always going
    to be cold.

    Arguments count towards that account exactly as they do in the planner, and for
    the same reason - a skill learned on *Ada Lovelace* has no text about
    *photosynthesis*, and ordering it to search for one is what it is FOR. They are
    taken from :func:`~skillweaver.agent.planner.bind_args` when it binds them for
    free, and from the caller's own ``-p`` values otherwise. What is deliberately NOT
    reachable from here is the composer: it binds a sentence the cheap binder cannot,
    and it costs a model call to do it, which is not a price a question about
    NAMESPACES may pay. So a candidate the composer would have rescued is judged on
    its bare text here, which is the strict direction.

    Measured against the live Wikipedia library on 2026-09-19, for *Search Wikipedia
    for computer vision and open the article*: the skill learned from that exact
    sentence accounts for 1.000 of it, the one learned from *Ada Lovelace* for 0.667
    and a link-following skill for 0.500, while a grocery errand scores 0.000 against
    all three. The cut sits in the gap rather than on top of a case.

    Args:
        text: The task, in the words it was asked in.
        target: Which world it drives.
        url: ``--url``, if given. Its host wins when there is one.
        domain: ``--domain``, if given. Always wins.
        params: The values the caller supplied (``-p``), which count towards a
            candidate's account of the request exactly as they do in the planner.
        retriever: Retrieval over the whole library. ``None`` disables the lookup,
            which leaves the old fallback and is what a caller with no library wants.
        graph: The site graph, consulted only to find where the winning skill's start
            screen lives. ``None`` means the caller gets a domain and no URL.
        k: How many candidates to consider.

    Returns:
        A :class:`DomainChoice`. Never raises: a retrieval or graph failure leaves the
        domain unresolved, because failing to look something up is not a reason to
        refuse to run.
    """
    if domain:
        return DomainChoice(domain=domain, start_url=url, source="named", why="named with --domain")
    host = _host(url) if target == "browser" else None
    if host:
        return DomainChoice(domain=host, start_url=url, source="url", why="the host of --url")
    if target != "browser" or retriever is None:
        return DomainChoice(
            domain=target,
            start_url=url,
            source="target",
            why="nothing named a domain, so the target's own name is used",
        )

    probe = TaskSpec(text=text, domain="", target=target, params=dict(params or {}))
    try:
        candidates = retriever.search(text, domain=None, k=k)
    except SkillWeaverError as exc:
        log.warning("domain.lookup.failed", task=text, error=str(exc))
        return DomainChoice(
            domain=target,
            start_url=url,
            source="target",
            why=f"the library could not be searched ({exc})",
        )

    passed_over: list[str] = []
    for candidate in candidates:
        skill = candidate.skill
        # The planner's own question, never asked more loosely than the planner asks
        # it: the arguments when they bind for free, and the skill's bare text when
        # they do not - because what binds them there is the composer, which costs a
        # model call and cannot run before the browser is even open.
        args = bind_args(skill, probe) or probe.params
        share = accounted_for(text, skill, args)
        if share < MIN_ACCOUNTED_FOR:
            passed_over.append(f"{skill.name}@{skill.domain} (accounts for {share:.0%})")
            continue
        log.info(
            "domain.resolved",
            task=text,
            domain=skill.domain,
            skill=skill.name,
            score=round(candidate.score, 3),
            accounted_for=round(share, 3),
            passed_over=", ".join(passed_over) or "(none)",
        )
        return DomainChoice(
            domain=skill.domain,
            start_url=_where_it_starts(graph, skill) or url,
            source="library",
            skill=skill.name,
            score=candidate.score,
            why=(
                f"the library answered: {skill.name} accounts for "
                f"{share:.0%} of this request and is filed under {skill.domain}"
            ),
        )

    log.info(
        "domain.unresolved",
        task=text,
        considered=len(candidates),
        passed_over=", ".join(passed_over) or "(none)",
    )
    return DomainChoice(
        domain=target,
        start_url=url,
        source="target",
        why=(
            f"the whole library was searched and no stored skill accounts for this "
            f"request ({_or_nothing(passed_over)}), so the target's own name is used"
        ),
    )


def _or_nothing(passed_over: Sequence[str]) -> str:
    """``passed_over`` as one clause, naming what was looked at and turned down."""
    if not passed_over:
        return "it holds nothing"
    return "passed over " + "; ".join(passed_over[:4])


def _where_it_starts(graph: SiteGraph | None, skill: Skill) -> str | None:
    """The URL ``skill``'s start screen was last seen at, or ``None``.

    Resolving the domain is only half of a bare repeat: the browser still has to open
    somewhere, and a warm attempt against ``about:blank`` fails at ``no_route`` having
    consulted the right library. The site graph already records where each screen was
    seen (:attr:`~skillweaver.contracts.UIState.url_pattern`), and the skill declares
    which screen it starts on, so the two together answer it without a new memory and
    without a guess.

    A graph that has never seen that screen answers ``None`` and the caller keeps
    whatever URL it had, which is the honest outcome: the domain is still right, and
    the warm attempt will report what it could not route to.
    """
    if graph is None or skill.precondition is None:
        return None
    try:
        graph.load(skill.domain)
        states = graph.states(skill.domain)
    except SkillWeaverError as exc:
        log.warning("domain.graph.unreadable", domain=skill.domain, error=str(exc))
        return None
    for state in states:
        if state.fingerprint == skill.precondition and state.url_pattern:
            return state.url_pattern
    return None


class ResetRefused(OSError):
    """The endpoint answered, and its answer was "I am not a reset hook".

    An ``OSError`` so that every existing caller of a :data:`WorldReset` - the
    evaluation harness translates one into its own error, the admission gate treats
    one as "not restored" - keeps working unchanged, and a subclass so that the one
    caller who wants to tell the two apart can.

    The distinction is not academic. Live Wikipedia answers ``403`` to the URL a
    ``--reset-url`` pointed at it, and reporting that as "the world was not restored"
    sent a day of debugging after a mutating task that did not exist. ``403`` means
    the flag was wrong; a timeout means the endpoint was down.
    """


_REFUSALS = frozenset({401, 403, 404, 405, 410, 451, 501})
"""HTTP statuses that mean this URL will never act as a reset hook.

Authentication, absence and method refusals are all permanent for a GET that asks an
application to restore itself: retrying cannot change any of them, and neither can
running the task again. Everything else - a timeout, a connection refused, a ``5xx`` -
is the endpoint having a bad day and is reported as a plain failure.
"""


class ResetOutcome(enum.StrEnum):
    """What became of the attempt to put the world back.

    ``restored`` and ``unnecessary`` both mean the world is fit to judge a candidate
    in; the other three mean it is not, and they differ in whose problem that is.
    """

    restored = "restored"
    """Something actually put the world back."""

    unnecessary = "unnecessary"
    """Nothing needed putting back: the task only reads and navigates."""

    absent = "absent"
    """No way back was configured, and the task did not say it needs none."""

    refused = "refused"
    """The endpoint answered and refused the job - it is not a reset hook."""

    failed = "failed"
    """A real way back was tried and did not work this time."""


@dataclass(frozen=True, slots=True)
class ResetReport:
    """The outcome of one reset attempt, and the sentence a human should read.

    :attr:`restored` is what :class:`~skillweaver.skills.synthesize.ReplayEnvironment`
    takes, a bool because that is what it takes; :attr:`outcome` is what a report
    should quote, because "this endpoint refuses to be a reset hook" and "this task
    changed something nothing can undo" are opposite problems that the bool spells
    the same way.
    """

    outcome: ResetOutcome
    detail: str = ""

    @property
    def restored(self) -> bool:
        """Whether the world is standing where the recording started."""
        return self.outcome in (ResetOutcome.restored, ResetOutcome.unnecessary)

    def __str__(self) -> str:
        return (
            f"reset {self.outcome.value}: {self.detail}"
            if self.detail
            else (f"reset {self.outcome.value}")
        )


def world_reset_from_url(url: str, *, timeout: float = 10.0) -> WorldReset:
    """A :data:`WorldReset` that restores the world by asking it to.

    One GET to ``url``, and the application puts itself back to its seed state. The
    sandbox site's ``/__reset`` is the instance this project ships, and any
    application with a "restore the demo data" endpoint fits the same shape - which
    is the point of taking a URL rather than knowing about the sandbox.

    Args:
        url: The endpoint to call. Its response body is read and discarded.
        timeout: Seconds to wait before giving up.

    Returns:
        The callable. It raises ``OSError`` (``URLError`` and ``HTTPError`` are both
        that) when the endpoint cannot be reached, which the caller reads as "the
        world was not restored" - except for the statuses in :data:`_REFUSALS`, which
        raise :class:`ResetRefused` and mean "there is no reset here to fail".
    """

    def reset() -> None:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in _REFUSALS:
                raise ResetRefused(
                    f"{url} answered HTTP {exc.code}: it is not a reset endpoint, so "
                    "there is nothing here that could have put the world back. Point "
                    "--reset-url at an endpoint that restores this site, or leave it "
                    "off for a task that changes nothing."
                ) from exc
            raise
        log.info("agent.world_reset", url=url)

    return reset


def reset_world(restore: WorldReset | None, *, read_only: bool = False) -> ResetReport:
    """Call ``restore`` if there is one, and say honestly what happened.

    Never raises: the run whose learning this precedes has already succeeded, and
    losing it to a traceback out of a reset hook would throw away the expensive half
    of the work.

    ``read_only`` is the task saying it changed nothing, which makes the answer
    ``unnecessary`` whatever the hook did - re-opening a page you only read IS the
    way back, and reporting that as a failed reset is how a read-only navigation task
    gets blamed for a mutation it never made.
    """
    detail = ""
    if restore is not None:
        try:
            restore()
        except ResetRefused as exc:
            log.warning("agent.world_reset.refused", error=str(exc))
            if not read_only:
                return ResetReport(ResetOutcome.refused, str(exc))
            detail = f"the reset endpoint refused the job ({exc}), and nothing needed it"
        except (OSError, SkillWeaverError) as exc:
            log.warning("agent.world_reset.failed", error=f"{type(exc).__name__}: {exc}")
            if not read_only:
                return ResetReport(
                    ResetOutcome.failed,
                    f"the reset hook did not work this time: {type(exc).__name__}: {exc}",
                )
            detail = (
                f"the reset hook did not work ({type(exc).__name__}: {exc}), and nothing needed it"
            )
        else:
            if not read_only:
                return ResetReport(ResetOutcome.restored, "the world was put back")
            detail = "the world was put back, though this task changed nothing"
    if read_only:
        return ResetReport(
            ResetOutcome.unnecessary,
            detail
            or (
                "this task only reads and navigates, so re-opening the screen it "
                "started on is a complete way back"
            ),
        )
    return ResetReport(
        ResetOutcome.absent,
        "no way to put this world back was configured, so a task that changed "
        "anything cannot be proved here",
    )


class NavigatingEnvironment:
    """Put the world back, then open the screen the recorded run started on.

    An :data:`EnvironmentFor` - it is called with a trajectory and hands back a
    factory, exactly as the plain function it replaced did - that additionally
    remembers what the last reset attempt came to, in :attr:`last_reset`. The gate
    only gets a bool, and a bool cannot tell "this endpoint is not a reset hook"
    from "this task changed something nothing can undo"; those are opposite problems
    with opposite fixes, and a run that reports the wrong one sends whoever reads it
    after the wrong cause. That happened, on live Wikipedia, and it is the reason
    this is a class.

    Two separate jobs, and only the second one is navigation. ``restore`` undoes what
    the run CHANGED; navigating then returns to where it started. A browser can
    always do the second and can never do the first, which is why a mutating task -
    archive this message, pay this invoice - was unlearnable until ``restore`` was
    passed: the gate arrived at a screen that no longer held what the recording held,
    failed its precondition, and did so forever.

    The trajectory's own first screen is the navigation target, not the task's start
    URL. They differ exactly when it matters: a run that rescued a failed warm attempt
    began part-way through the site, and sending the gate back to the front page would
    have it judge the candidate on the wrong screen.

    A ``restore`` that raises is reported, not propagated: the environment comes back
    with ``restored=False`` and the gate says the world could not be put back, which
    is a truer answer than a traceback out of a learning step.

    The call declines - ``None`` - for a controller that cannot navigate (every
    desktop one) or a run whose first screen had no URL. Nothing is then admitted,
    and the report says so.

    Args:
        controller / perceiver: The world the candidate is re-run in.
        graph: The site graph the candidate may read.
        restore: How to put this world back. ``None`` means nothing can, and a
            mutating task will be reported as unproved rather than as rejected.
        read_only: The task saying it changes nothing, so re-opening the screen it
            started on IS the way back. The gate then judges the candidate instead of
            reporting a reset that was never needed - which is what a search-and-read
            task on a live site needs, since no such site has a reset endpoint and
            pointing ``--reset-url`` at one only earns a ``403``.
    """

    __slots__ = ("_controller", "_graph", "_last_reset", "_perceiver", "_read_only", "_restore")

    def __init__(
        self,
        controller: Controller,
        perceiver: Perceiver,
        *,
        graph: SiteGraph | None = None,
        restore: WorldReset | None = None,
        read_only: bool = False,
    ) -> None:
        self._controller = controller
        self._perceiver = perceiver
        self._graph = graph
        self._restore = restore
        self._read_only = read_only
        self._last_reset: ResetReport | None = None

    def __repr__(self) -> str:
        how = "read-only" if self._read_only else ("reset" if self._restore else "navigate only")
        return f"NavigatingEnvironment({how}, last={self._last_reset})"

    @property
    def last_reset(self) -> ResetReport | None:
        """What the most recent attempt to put the world back came to, or ``None``
        when no candidate has been stood up yet."""
        return self._last_reset

    def __call__(self, trajectory: Trajectory) -> EnvironmentFactory | None:
        if not trajectory.steps or not self._controller.supports("navigate"):
            return None
        url = trajectory.steps[0].before.url
        if not url:
            return None

        def factory() -> ReplayEnvironment:
            report = reset_world(self._restore, read_only=self._read_only)
            self._last_reset = report
            self._controller.perform(Navigate(url))
            return ReplayEnvironment(
                self._controller, self._perceiver, self._graph, restored=report.restored
            )

        return factory


def navigating_environment(
    controller: Controller,
    perceiver: Perceiver,
    *,
    graph: SiteGraph | None = None,
    restore: WorldReset | None = None,
    read_only: bool = False,
) -> NavigatingEnvironment:
    """Build a :class:`NavigatingEnvironment`. See it for what the arguments mean."""
    return NavigatingEnvironment(
        controller, perceiver, graph=graph, restore=restore, read_only=read_only
    )


def build_agent(
    task: TaskSpec,
    *,
    controller: Controller,
    perceiver: Perceiver,
    llm: LLMClient,
    store: SkillStore,
    retriever: SkillRetriever | None = None,
    graph: SiteGraph | None = None,
    trajectories: TrajectoryStore | None = None,
    recorder: TrajectoryRecorder | None = None,
    environment: EnvironmentFor | None = None,
    budget: Budget | None = None,
    compose: bool = True,
    learn: bool = True,
    max_repairs: int = 2,
    top_k: int = 5,
) -> Agent:
    """Assemble the planner, the explorer and the admission gate around one task.

    This is the ONE place the real agent is wired, so a test that injects fakes at
    the leaves exercises the same wiring the command line runs. The two critics
    differ on purpose:

    * The **warm** critic gets :func:`recall` - the recorded end screen AND the skill
      that remembered it - and :func:`_warm_critic` decides from the second how much
      the first is allowed to say. Either way a warm run that lands where the recorded
      run landed is judged programmatically and the action loop stays model-free.
    * The **cold** critic is a plain :class:`~skillweaver.agent.critic.TieredCritic`
      over ``llm``: exploration has no recorded end state to compare against, and
      paying for a verdict is the cheapest part of a run that is already paying for
      every move.
    * The **admission** critic, built per trajectory, demands as decisive evidence
      that the replayed candidate reach the same screen the recording reached. That
      is right HERE and wrong on the warm path, and the difference is the arguments:
      the gate re-runs the recorded run with the recorded values, so one end screen
      is the only correct answer. Also free.

    Args:
        task: What the agent will be asked to do; used to recall the end state.
        controller / perceiver: The world.
        llm: The model. Used by exploration, synthesis, composition and by any
            critic that has to escalate - never by a warm action loop.
        store: The library.
        retriever: Defaults to a :class:`~skillweaver.skills.retrieve.SkillRetriever`
            over ``store``.
        graph: The site graph. Defaults to a fresh in-memory one.
        trajectories: Where runs are saved and end states are recalled from.
        recorder: Where exploration writes. ``None`` lets the explorer choose, which
            means the configured data directory.
        environment: Resets the world for the admission gate, per trajectory.
            ``None`` means this agent cannot learn, and it reports that rather than
            skipping quietly.
        budget: Limits for one run.
        compose: Whether the planner may spend one model call chaining known skills
            for a task no single skill covers. ``False`` keeps it strictly
            model-free.
        learn: Whether to build a synthesizer at all.
        max_repairs: How many times the gate may ask for the code to be rewritten.
        top_k: Retrieval breadth.
    """
    retriever = retriever if retriever is not None else SkillRetriever(store)
    graph = graph if graph is not None else InMemorySiteGraph()
    runner = SkillRunner(store)

    candidates = _safe_search(retriever, task, top_k)
    recalled = recall(store, trajectories, task, candidates)
    if recalled.state is None:
        log.info("agent.warm.no_end_state", task=task.text, domain=task.domain)
    else:
        log.info(
            "agent.warm.end_state",
            task=task.text,
            domain=task.domain,
            skill=recalled.source.name if recalled.source else "",
            role="corroboration" if recalled.self_checking else "evidence",
        )

    planner = Planner(
        store=store,
        retriever=retriever,
        graph=graph,
        runner=runner,
        critic=_warm_critic(llm, recalled),
        controller=controller,
        perceiver=perceiver,
        composer=Composer(llm, store, graph=graph) if compose else None,
        budget=budget,
        top_k=top_k,
    )
    explorer = Explorer(
        llm,
        perceiver,
        critic=TieredCritic(llm),
        graph=graph,
        recorder=recorder,
        retriever=retriever,
        runner=runner,
    )

    synthesis: SynthesisFactory | None = None
    if learn:

        def synthesis(trajectory: Trajectory) -> Synthesizer:
            end = trajectory.steps[-1].after.fingerprint if trajectory.steps else None
            return Synthesizer(
                llm,
                store,
                TieredCritic(llm, expected_state=end),
                max_repairs=max_repairs,
            )

    return Agent(
        controller=controller,
        perceiver=perceiver,
        store=store,
        retriever=retriever,
        planner=planner,
        explorer=explorer,
        graph=graph,
        trajectories=trajectories,
        synthesis=synthesis,
        environment=environment,
        llm=llm,
        budget=budget,
        top_k=top_k,
    )


def _warm_critic(llm: LLMClient, recalled: Recollection) -> TieredCritic:
    """The critic that judges a warm replay, and how much the recalled screen may say.

    One decision, and it is the difference between a library that improves and one
    that eats itself.

    * A skill that **carries a verifier** has already proved it did its own job before
      the critic is consulted: the sandbox ran that verifier and a failure there would
      have failed the run. The recalled screen is then :data:`corroboration` - a free
      yes when it matches, and silent when it does not. It has to be, because it is
      the screen ONE run ended on with ONE set of arguments: a skill learned from
      "Search Wikipedia for computer vision" and replayed for "machine learning"
      correctly ends somewhere else, and holding it to the recorded screen rejected
      that correct replay at similarity 0.120, twice, on live Wikipedia.
    * A skill with **no verifier** has proved nothing, and the recalled screen is the
      only free evidence there is, so it keeps its veto. That is what catches the
      stored skill which runs without error and finishes half the errand.

    What can still fail a warm run either way: any veto -
    :func:`~skillweaver.agent.checks.state_changed` on a replay that did nothing and
    :func:`~skillweaver.agent.checks.no_error_state` on one that ended on an error
    page - and, when the corroboration misses, the model that is then asked and paid
    for. A skill's own say-so is never enough on its own.
    """
    if recalled.self_checking:
        return TieredCritic(llm, corroborating_state=recalled.state)
    return TieredCritic(llm, expected_state=recalled.state)


def _safe_search(retriever: SkillRetriever, task: TaskSpec, top_k: int) -> tuple[Candidate, ...]:
    try:
        return tuple(retriever.search(task.text, domain=task.domain, k=top_k))
    except SkillWeaverError:
        return ()


# --------------------------------------------------------------------------------------
# What the command line holds
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Workbench:
    """Everything a command needs, built once per invocation.

    The cheap memories - the skill library, the site graph, the trajectory store -
    are built eagerly because half the subcommands need nothing else. Anything
    expensive (a browser, a model) is behind :attr:`session`, so ``skills ls`` never
    launches Chromium.

    Attributes:
        settings: Resolved configuration. The only source of paths and defaults.
        store: The skill library.
        retriever: Retrieval over ``store``.
        graph: The site graph, backed by the configured graph directory.
        trajectories: Recorded runs.
        session: Opens a live agent for one task, and closes the world afterwards.
            Tests replace this with one wired to fakes.
    """

    settings: Settings
    store: SkillStore
    retriever: SkillRetriever
    graph: SiteGraph
    trajectories: TrajectoryStore
    session: Callable[[TaskSpec, Budget], Any]

    @property
    def data_dir(self) -> Path:
        """The data directory every memory in this workbench lives under."""
        return self.settings.data_dir


def build_workbench(config: Settings | None = None) -> Workbench:
    """The real workbench: file-backed memories and a live browser session.

    Nothing here touches a browser, a model or a network; the session factory does
    that, and only when a command opens one.

    Args:
        config: Resolved settings. ``None`` uses :func:`skillweaver.config.settings`.
    """
    resolved = config if config is not None else settings()
    store = FileSkillStore(resolved.skills_dir)
    graph = InMemorySiteGraph(store=JSONGraphStore(resolved.graphs_dir))
    trajectories = TrajectoryFileStore(resolved.trajectories_dir)

    @contextmanager
    def session(task: TaskSpec, budget: Budget) -> Any:
        controller, perceiver = _open_world(resolved, task)
        try:
            graph.load(task.domain)
            yield build_agent(
                task,
                controller=controller,
                perceiver=perceiver,
                llm=_open_model(resolved),
                store=store,
                retriever=SkillRetriever(store),
                graph=graph,
                trajectories=trajectories,
                recorder=Recorder(resolved.trajectories_dir),
                environment=navigating_environment(
                    controller,
                    perceiver,
                    graph=graph,
                    restore=_reset_for(task),
                    read_only=_is_read_only(task),
                ),
                budget=budget,
            )
        finally:
            controller.close()

    return Workbench(
        settings=resolved,
        store=store,
        retriever=SkillRetriever(store),
        graph=graph,
        trajectories=trajectories,
        session=session,
    )


def _reset_for(task: TaskSpec) -> WorldReset | None:
    """How to put this task's world back, if the task said.

    ``None`` is a real answer and not a failure: the gate then reports that a
    mutating task could not be proved, which is what is true.
    """
    url = task.params.get(RESET_URL_PARAM)
    return world_reset_from_url(str(url)) if url else None


def _is_read_only(task: TaskSpec) -> bool:
    """Whether the task declared that it changes nothing. See :data:`READ_ONLY_PARAM`.

    Anything truthy counts, and ``-p read_only=false`` parses to the JSON ``False``
    the command line intends, so the flag reads the way it is written.
    """
    return bool(task.params.get(READ_ONLY_PARAM, False))


def _open_world(config: Settings, task: TaskSpec) -> tuple[Controller, Perceiver]:
    """Open the controller the task asks for, and eyes to go with it.

    Imported here rather than at module scope: Playwright, ultralytics and RapidOCR
    are all slow to import, and ``skillweaver skills ls`` has no business paying for
    any of them.
    """
    from skillweaver.perception.detect_yolo import DEFAULT_WEIGHTS_NAME, YoloDetector
    from skillweaver.perception.ocr import RapidOcrReader

    controller: Controller
    if task.target == "desktop":
        from skillweaver.controllers.desktop import DesktopController

        controller = DesktopController()
    else:
        from skillweaver.controllers.browser import BrowserController

        controller = BrowserController(headless=False, start_url=task.params.get("start_url"))
    # Not ``default_weights_path()``: that reads the process-wide settings, which a
    # --data-dir on this invocation has already overridden.
    detector = YoloDetector(config.models_dir / DEFAULT_WEIGHTS_NAME)
    return controller, ComposedPerceiver(detector, RapidOcrReader())


def _open_model(config: Settings) -> LLMClient:
    """The computer-use model. Claude is primary; Gemini is the alternative."""
    from skillweaver.llm.anthropic_ import AnthropicClient

    return AnthropicClient(model=config.claude_model, computer_use=True)


def budget_from(
    config: Settings,
    *,
    max_steps: int | None = None,
    max_seconds: float | None = None,
    max_usd: float | None = None,
    max_llm_calls: int | None = None,
) -> Budget:
    """The configured budget with any explicitly given flag overriding it.

    Configuration is the floor, flags are the override, and a flag that was not
    given never silently resets a configured limit to a default.
    """
    base = config.default_budget
    return Budget(
        max_steps=base.max_steps if max_steps is None else max_steps,
        max_seconds=base.max_seconds if max_seconds is None else max_seconds,
        max_usd=base.max_usd if max_usd is None else max_usd,
        max_llm_calls=base.max_llm_calls if max_llm_calls is None else max_llm_calls,
    )
