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
has evidence to run. The evidence this module supplies is
:func:`recall_end_state`: the screen the recorded run that TAUGHT the skill ended on,
read back from the trajectory store. A warm run that lands there is a decisive,
free yes. When no end state can be recalled the critic escalates to a model and pays
for one call - that is visible in ``spend``, not swept under the rug.

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

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from skillweaver.agent.compose import Composer
from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import Explorer
from skillweaver.agent.planner import PlanFailure, Planner
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
    utcnow,
)
from skillweaver.errors import BudgetExceeded, SkillWeaverError
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.graph.store import JSONGraphStore
from skillweaver.logging_ import get_logger
from skillweaver.perception.elements import build_index, merge_elements
from skillweaver.perception.fingerprint import StateFingerprinter
from skillweaver.skills.retrieve import SkillRetriever
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
    "Agent",
    "AttemptRecord",
    "ComposedPerceiver",
    "EnvironmentFor",
    "RunReport",
    "SynthesisFactory",
    "Workbench",
    "build_agent",
    "budget_from",
    "build_workbench",
    "navigating_environment",
    "recall_end_state",
    "task_spec",
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
            warm path exists to make.**
        usd: Model spend charged to this attempt.
        seconds: Wall-clock seconds the attempt was charged.
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
    demoted: str | None = None
    performed_nothing: bool = True
    run_id: str | None = None
    trace: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.path} path: {'ok' if self.ok else 'failed'}"
        where = f" at {self.stage}" if self.stage else ""
        chain = f" [{' -> '.join(self.skills_used)}]" if self.skills_used else ""
        return f"{head}{where}{chain} ({self.llm_calls} model call(s)) - {self.reason}"


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
    def llm_calls(self) -> int:
        """Model calls across every attempt of this run."""
        return sum(a.llm_calls for a in self.attempts)

    @property
    def steps(self) -> int:
        """Controller actions performed across every attempt of this run."""
        return sum(a.steps for a in self.attempts)

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
            lines.append("retrieved: nothing - the library has no candidate for this task")
        lines += [f"  {attempt}" for attempt in self.attempts]
        if self.rescued:
            lines.append(
                "NOTE: a stored skill ran, did not work, and exploration rescued the "
                "run. The library was WRONG about this task, not merely slow."
            )
        if self.learned is not None:
            lines.append(
                f"learned: {self.learned.name} v{self.learned.version} "
                f"({self.task.domain}) - the next run of this task can be warm"
            )
        elif self.learning_note:
            lines.append(f"learned: nothing - {self.learning_note}")
        verdict = "SOLVED" if self.ok else "NOT SOLVED"
        lines.append(
            f"{verdict} by the {self.decision} path "
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
        budget: Limits for one :meth:`run`.
        top_k: How many candidates to retrieve for the report.
    """

    __slots__ = (
        "_budget",
        "_controller",
        "_environment",
        "_explorer",
        "_graph",
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
            record, outcome = self._try_warm(task)
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

        record, outcome = self._try_cold(task)
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
                    reason=f"the library holds no skill for domain {task.domain!r}",
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
        return None, admission, admission.reason

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
    :class:`~skillweaver.agent.critic.TieredCritic` as ``expected_state`` turns "did
    the task get done?" into a fingerprint comparison - decisive, and costing nothing.

    Without it the critic has no evidence to run and escalates to a model, so a warm
    run would still work but would no longer be model-free. That is a real cost and
    it is reported in ``spend`` rather than hidden.

    Args:
        store: The library, searched when ``candidates`` is empty.
        trajectories: Where recorded runs live. ``None`` gives ``None``.
        task: The task being run; only its domain is used.
        candidates: Retrieval's offer, best first. The first one whose trajectory
            can still be read wins.

    Returns:
        The recalled fingerprint, or ``None`` when nothing could be read back.
    """
    if trajectories is None:
        return None
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
            return trajectory.steps[-1].after.fingerprint
    return None


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

    Args:
        detector: Finds controls. Required.
        reader: Reads text. ``None`` runs detection alone, which is what a machine
            without the OCR models can still do.
        fingerprinter: Identifies the screen. Defaults to
            :class:`~skillweaver.perception.fingerprint.StateFingerprinter`.

    This lives here rather than in :mod:`skillweaver.perception` only because that
    package has not grown a composing perceiver yet; it is wiring, and wiring is this
    module's job until it has a better home.
    """

    __slots__ = ("_detector", "_fingerprinter", "_reader")

    def __init__(
        self,
        detector: Detector,
        reader: TextReader | None = None,
        fingerprinter: Fingerprinter | None = None,
    ) -> None:
        self._detector = detector
        self._reader = reader
        self._fingerprinter = fingerprinter if fingerprinter is not None else StateFingerprinter()

    def __repr__(self) -> str:
        return f"ComposedPerceiver(reader={self._reader is not None})"

    def observe(self, controller: Controller) -> Observation:
        """One frame, fully understood.

        Raises:
            ControllerError: if the capture fails.
            PerceptionError: if detection, reading or fingerprinting fails.
        """
        shot: Screenshot = controller.capture()
        found: list[Element] = self._detector.detect(shot)
        text: list[Element] = self._reader.read(shot) if self._reader is not None else []
        elements = tuple(merge_elements(found, text))
        url = controller.url()
        return Observation(
            screenshot=shot,
            elements=elements,
            index=build_index(elements),
            fingerprint=self._fingerprinter.fingerprint(shot, elements, url),
            url=url,
            taken_at=utcnow(),
        )


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def task_spec(
    text: str,
    *,
    domain: str | None = None,
    target: Literal["browser", "desktop"] = "browser",
    url: str | None = None,
    params: dict[str, Any] | None = None,
) -> TaskSpec:
    """Build a :class:`~skillweaver.contracts.TaskSpec` the way the command line does.

    The domain defaults to the host of ``url`` for a browser task, so
    ``--url https://example.com/invoices`` files its skills under ``example.com``
    without anyone having to say so twice. A desktop task with no domain gets
    ``"desktop"``.
    """
    merged: dict[str, Any] = dict(params or {})
    if url:
        merged.setdefault("start_url", url)
    resolved = domain or (_host(url) if target == "browser" else None) or target
    return TaskSpec(text=text, domain=resolved, target=target, params=merged)


def _host(url: str | None) -> str | None:
    """The host of a URL, or ``None`` when there is not one to take."""
    if not url:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(url if "//" in url else f"//{url}")
    return parsed.hostname or None


def navigating_environment(
    controller: Controller,
    perceiver: Perceiver,
    *,
    graph: SiteGraph | None = None,
) -> EnvironmentFor:
    """Reset a browser by navigating to the URL the recorded run started on.

    The trajectory's own first screen is the target, not the task's start URL. They
    differ exactly when it matters: a run that rescued a failed warm attempt began
    part-way through the site, and sending the gate back to the front page would have
    it judge the candidate on the wrong screen.

    The returned callable declines - ``None`` - for a controller that cannot navigate
    (every desktop one) or a run whose first screen had no URL. Nothing is then
    admitted, and the report says so.
    """

    def environment_for(trajectory: Trajectory) -> EnvironmentFactory | None:
        if not trajectory.steps or not controller.supports("navigate"):
            return None
        url = trajectory.steps[0].before.url
        if not url:
            return None

        def factory() -> ReplayEnvironment:
            controller.perform(Navigate(url))
            return ReplayEnvironment(controller, perceiver, graph)

        return factory

    return environment_for


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

    * The **warm** critic gets :func:`recall_end_state` as evidence, so a warm run
      that lands where the recorded run landed is judged programmatically and the
      action loop stays model-free.
    * The **cold** critic is a plain :class:`~skillweaver.agent.critic.TieredCritic`
      over ``llm``: exploration has no recorded end state to compare against, and
      paying for a verdict is the cheapest part of a run that is already paying for
      every move.
    * The **admission** critic, built per trajectory, demands that the replayed
      candidate reach the same screen the recording reached. Also free.

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
    expected = recall_end_state(store, trajectories, task, candidates)
    if expected is None:
        log.info("agent.warm.no_end_state", task=task.text, domain=task.domain)

    planner = Planner(
        store=store,
        retriever=retriever,
        graph=graph,
        runner=runner,
        critic=TieredCritic(llm, expected_state=expected),
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
        budget=budget,
        top_k=top_k,
    )


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
                environment=navigating_environment(controller, perceiver, graph=graph),
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
