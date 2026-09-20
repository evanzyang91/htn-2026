"""The cold-versus-warm decision, and the wiring that makes it runnable::

    retrieve  ->  WARM: plan from stored skills and run it
                    |
                    +-- could not plan, or the plan ran and did not work
                    v
                  COLD: explore by trial and error
                    |
                    +-- succeeded -> synthesize, gate, store -> next time is WARM

A fall-through is REPORTED, never hidden: ``RunReport`` carries an ``AttemptRecord`` per
path tried, and ``rescued`` / ``warm_missed`` keep a warm miss out of the headline
"SOLVED by the cold path", which is the sentence that has hidden a defect for months.

Two costs are counted here for the same reason. ``Agent._charge_model`` reads each
attempt's spend from the client's own ``total_usage`` across it, so a composer call
whose plan was discarded is not lost; and a ``BudgetExceeded`` from the planner is
recorded as the warm failure and NOT explored after - falling back to the expensive path
having run out of money is the one fallback that is always wrong.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from skillweaver.agent.compose import Composer
from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import ActingPolicy, Explorer
from skillweaver.agent.planner import (
    MIN_ACCOUNTED_FOR,
    PlanFailure,
    Planner,
    account_of,
    asks_for,
    bind_args,
    fit_through_family,
)
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
from skillweaver.errors import BudgetExceeded, ConfigError, SkillWeaverError
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
from skillweaver.perception_mode import DOM, PIXELS, namespace, path_of
from skillweaver.render_mode import crossing, mode_name, mode_of
from skillweaver.reset_actions import (
    RESET_ACTIONS_PARAM,
    ResetStep,
    chain_resets,
    reset_step_to_dict,
    reset_steps_from,
    world_reset_from_actions,
)
from skillweaver.skills.embed import embedder_for, load_embedder
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
    "READ_ONLY_PARAM",
    "RESET_ACTIONS_PARAM",
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
    "ResetStep",
    "RunReport",
    "SynthesisFactory",
    "Workbench",
    "WorldReset",
    "build_agent",
    "budget_from",
    "build_retriever",
    "build_workbench",
    "navigating_environment",
    "perception_counts",
    "recall",
    "recall_end_state",
    "reset_steps_from",
    "reset_world",
    "resolve_domain",
    "task_spec",
    "world_reset_from_actions",
    "world_reset_from_url",
]

log = get_logger(__name__)

Path_ = Literal["warm", "cold"]

SynthesisFactory = Callable[[Trajectory], Synthesizer]
"""A factory and not an instance because the gate's critic wants to know where the
recorded run ENDED, which is only knowable once it has happened."""

EnvironmentFor = Callable[[Trajectory], EnvironmentFactory | None]
"""The gate's replay world for one trajectory, or ``None`` when this world cannot be put
back there. Per trajectory and not per task: an exploration that rescued a failed warm
attempt began wherever that attempt left the screen, not at the task's start URL."""

WorldReset = Callable[[], None]
""""How do I undo this?" - re-opening a screen is NOT it. A task worth learning changes
something, so without a way back the gate can never stand a candidate where the recording
stood and no such skill is ever admitted.

Whatever that world offers counts: a seed endpoint (``world_reset_from_url``), a snapshot,
a fresh profile, steps performed on the screen (``reset_actions``). It may raise anything
and the caller reads that as "not restored"; ``ResetRefused`` is the one distinguished
failure, meaning "this is not a reset hook and never will be".
"""


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One path the agent tried, and what came of it.

    Attributes:
        reason: The planner's ``PlanFailure`` reason (warm) or the explorer's rendered
            diagnosis (cold).
        stage: A ``FailureStage``, ``"empty_library"`` or ``"budget"`` for warm; the
            diagnosis' ``stopped_by`` for cold. Empty on success.
        llm_calls: ZERO is the claim the warm path exists to make, which is why this is
            read from the model client across the attempt and not from the plan it
            returned - a call spent on a discarded plan is still a call.
        usd: Model spend, counted the same way.
        wall_ms: Wall-clock milliseconds of the attempt, taken over the SAME boundary as
            the two clocks below so they can never add up to more than it. NOT
            ``seconds``, which is the budget's clock and starts later: measured on a warm
            Wikipedia replay, ``site_ms`` came to 104% of ``seconds``, because the first
            observation happens before the budget starts counting.
        policy_ms: Milliseconds an acting policy's OWN models took - Jev and its text
            writer - and ``0.0`` without one. Upstream's model half. Not the critic's
            calls: those are inside ``wall_ms`` and outside both halves, so
            ``wall_ms - policy_ms - site_ms`` is judging, recording and overhead -
            which on a cold Wikipedia run was 73% of the wall, against 15% and 12%.
        site_ms: Milliseconds spent waiting on the SITE - performing, settling, observing
            and resting - when the eyes keep that clock (``DomPerceiver.site_ms``), and
            ``0.0`` when they do not. Meaningful on the warm path too, where it is nearly
            all of the run. Like every timing here it is for a run whose ground truth
            passed; ``eval.metrics._measured`` is the only door to a comparison.
        perception: What the eyes did, as COUNTS, so it means the same on a busy machine
            as on an idle one.
        cross_mode: The render-mode crossing that most likely lost the screen. Its own
            field, not a ``stage``, because ``stage`` must keep saying where the attempt
            stopped (``no_route``) while this says what put it there.
        demoted: The skill retired because it ran and failed.
        performed_nothing: Whether the screen is untouched, so the next path may start
            from it as it stands.
        trace: The failed skill's sandbox trace - the most useful thing a warm miss leaves.
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
    wall_ms: float = 0.0
    policy_ms: float = 0.0
    site_ms: float = 0.0
    perception: PerceptionCounts = PerceptionCounts()
    cross_mode: str | None = None
    demoted: str | None = None
    performed_nothing: bool = True
    run_id: str | None = None
    trace: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.path} path: {'ok' if self.ok else 'failed'}"
        where = f" at {self.stage}" if self.stage else ""
        chain = f" [{' -> '.join(self.skills_used)}]" if self.skills_used else ""
        eyes = f", {self.perception}" if self.perception else ""
        crossed = f" [{self.cross_mode}]" if self.cross_mode else ""
        return (
            f"{head}{where}{chain} ({self.llm_calls} model call(s){eyes}) - {self.reason}{crossed}"
        )


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything one ``Agent.run`` did, so "which path ran?" is never a guess.

    ``decision`` names the path that produced the answer and ``attempts`` holds every path
    tried, failures included, in order. ``outcome`` is the winner's, or the last failure
    when nothing worked; ``learning_note`` says why nothing was learned.
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
        """The warm attempt, or ``None`` when it was not tried."""
        return next((a for a in self.attempts if a.path == "warm"), None)

    @property
    def cold(self) -> AttemptRecord | None:
        """The cold attempt, or ``None`` when exploration was not reached."""
        return next((a for a in self.attempts if a.path == "cold"), None)

    @property
    def rescued(self) -> bool:
        """The library RAN something, it did not work, and exploration saved it - the most
        important thing this report can say. A warm path that declined before touching the
        screen is an ordinary cold start, not a rescue; see ``warm_missed``."""
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
        """The library was consulted, declined before touching the screen, and exploring
        paid full price.

        A legitimate cold start looks EXACTLY like a lookup in the wrong namespace - ``run``
        without a ``--url`` once resolved its domain to the literal ``"browser"``, missed a
        library it was standing next to, and printed ``SOLVED by the cold path``. So every
        fall-through says so in the headline.
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
        """Model calls across every attempt."""
        return sum(a.llm_calls for a in self.attempts)

    @property
    def steps(self) -> int:
        """Controller actions across every attempt."""
        return sum(a.steps for a in self.attempts)

    @property
    def perception(self) -> PerceptionCounts:
        """What the eyes did across every attempt; ``ocr_reads`` is the number the OCR
        cache is judged by, and it does not move with machine load."""
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
        # "SOLVED by the cold path" alone reads as a plain success, and after a warm miss
        # it is the most misleading sentence this report can end on.
        after = " AFTER A WARM MISS" if (self.rescued or self.warm_missed) else ""
        lines.append(
            f"{verdict} by the {self.decision} path{after} "
            f"in {self.steps} action(s) and {self.llm_calls} model call(s)"
        )
        return "\n".join(lines)


class Agent:
    """Runs one task by deciding between the warm and the cold path.

    Args:
        retriever: Consulted here for the report and again inside the planner; both are
            outside the action loop, and the duplicate buys a report naming what was offered.
        synthesis: ``None`` disables learning, and the report says so rather than pretending.
        environment: The gate's world, reset to the screen the run STARTED on. ``None``, or a
            call returning ``None``, disables learning: an un-re-run skill has proved nothing.
        llm: Never called here - only READ, through ``total_usage``, to charge each attempt
            every call made while it ran. ``None`` falls back to what each attempt says about
            itself, which undercounts a plan discarded after the composer paid for it.
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
        "_policy",
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
        policy: Any | None = None,
    ) -> None:
        self._policy = policy
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
        """The open world, exposed so a command can replay against the browser this session
        opened rather than a second one."""
        return self._controller

    @property
    def perceiver(self) -> Perceiver:
        """The eyes, for the same reason."""
        return self._perceiver

    @property
    def budget(self) -> Budget:
        """The limits one task runs under."""
        return self._budget

    def run(
        self, task: TaskSpec, *, learn: bool = True, warm: bool = True, cold: bool = True
    ) -> RunReport:
        """Do ``task`` and report which path did it; a failure is a report, not an exception.

        ``warm=False`` forces exploration, which is what the ``learn`` command wants;
        ``cold=False`` makes this library-only and reports honestly when the library falls
        short.
        """
        candidates = self._retrieve(task)
        attempts: list[AttemptRecord] = []

        if warm:
            mark, spent, clock = perception_counts(self._perceiver), self._usage(), self._clock()
            record, outcome = self._try_warm(task)
            record = self._charge_model(self._charge_eyes(record, mark), spent)
            record = self._charge_clock(record, clock)
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
                # The planner lets BudgetExceeded escape so an exhausted run STOPS:
                # exploring now spends money the run was told it does not have.
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

        mark, spent, clock = perception_counts(self._perceiver), self._usage(), self._clock()
        record, outcome = self._try_cold(task)
        record = self._charge_model(self._charge_eyes(record, mark), spent)
        record = self._charge_clock(record, clock)
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
        # Counted here, not inside each attempt: the perceiver is shared by the planner,
        # the explorer and every skill they run, so the attempt is the only honest boundary.
        return replace(record, perception=perception_counts(self._perceiver) - mark)

    def _clock(self) -> tuple[float, float, float]:
        """``(now_ms, policy_ms, site_ms)`` so far, from whoever keeps them - duck-typed,
        because neither clock is on a shared Protocol and most policies and eyes keep
        neither. ``now_ms`` rides along so all three are read at one moment."""
        policy = getattr(self._policy, "policy_ms", 0.0)
        site = getattr(self._perceiver, "site_ms", 0.0)
        return (
            time.monotonic() * 1000.0,
            float(policy) if isinstance(policy, (int, float)) else 0.0,
            float(site) if isinstance(site, (int, float)) else 0.0,
        )

    def _charge_clock(
        self, record: AttemptRecord, mark: tuple[float, float, float]
    ) -> AttemptRecord:
        # Same boundary as _charge_eyes and for the same reason: both clocks are shared by
        # every path, so the attempt is the only honest place to take the difference.
        now_ms, policy_ms, site_ms = self._clock()
        return replace(
            record,
            wall_ms=now_ms - mark[0],
            policy_ms=policy_ms - mark[1],
            site_ms=site_ms - mark[2],
        )

    def _charge_model(self, record: AttemptRecord, mark: Usage) -> AttemptRecord:
        """Charge ``record`` every model call since ``mark``, keeping the attempt's own
        number when it is larger - a path counting a call this meter cannot see is never
        talked DOWN. This meter only ever finds calls that were LOST, such as a plan
        discarded at routing after the composer paid for it."""
        spent = self._usage()
        calls = max(record.llm_calls, spent.calls - mark.calls)
        usd = max(record.usd, spent.cost_usd - mark.cost_usd)
        return replace(record, llm_calls=calls, usd=usd)

    def _usage(self) -> Usage:
        """The client's spend so far, or an empty tally: a stub need not implement
        ``total_usage``, and a report is not worth an exception."""
        total = getattr(self._llm, "total_usage", None)
        if total is None:
            return Usage()
        try:
            result = total()
        except Exception:  # noqa: BLE001 - a broken meter must not fail a run
            log.warning("agent.usage.unreadable", llm=type(self._llm).__name__)
            return Usage()
        return result if isinstance(result, Usage) else Usage()

    def _try_warm(self, task: TaskSpec) -> tuple[AttemptRecord, RunOutcome | None]:
        """Plan from the library and run it, or say why not. An empty domain is answered
        without even reading the screen."""
        stored = self._store.list(domain=task.domain)
        if not stored:
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
        return self._explained(self._warm_failure(task, failure), stored, task.domain), None

    def _held_elsewhere(self, domain: str) -> str:
        """What the library holds under OTHER domains, as a clause to append: an empty
        ``'browser'`` beside four skills under ``en.wikipedia.org`` hands the reader the
        bug rather than a shrug."""
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

    _LOST_THE_SCREEN = frozenset({"no_route"})
    """The only warm stage a render-mode crossing can explain, deliberately: every other
    is about something else (words, the library, a broken controller, or screens that
    matched well enough to run), and offering the mode there would be a confident
    falsehood where the user is already confused."""

    def _explained(
        self, record: AttemptRecord, stored: Sequence[Skill], domain: str
    ) -> AttemptRecord:
        """``record`` with a render-mode crossing attached, when one explains it - never
        refused, only explained, since the gap is not a constant.

        Requires that EVERY skill for ``domain`` recorded its start screen in the other
        mode: one skill in this mode, or one that never said, means the library is not
        uniformly unreadable and the crossing is not the explanation.
        """
        if record.stage not in self._LOST_THE_SCREEN:
            return record
        current = mode_of(self._controller)
        if current is None:  # a desktop, or a controller with no opinion - no claim
            return record
        recorded = getattr(self._store, "recorded_render_modes", None)
        if recorded is None:  # a store that does not keep the answer
            return record
        claimed = recorded(domain)
        names = sorted(skill.name for skill in stored)
        if not all(claimed.get(name, current) != current for name in names):
            return record
        sentence = crossing(claimed[names[0]], current)
        if sentence is None:  # unreachable while there are two modes; stay honest
            return record
        log.info("agent.warm.cross_mode", domain=domain, stage=record.stage, why=sentence)
        return replace(record, cross_mode=sentence)

    def _try_cold(self, task: TaskSpec) -> tuple[AttemptRecord, RunOutcome]:
        """Explore, and persist whatever the run recorded. A failed exploration is saved
        too: its trajectory names the screen the agent was stuck on and what it tried."""
        outcome = self._explorer.explore(task, self._controller, self._budget)
        self._save(outcome.trajectory)
        log.info(
            "agent.cold",
            task=task.text,
            ok=outcome.ok,
            steps=outcome.spend.steps,
            llm_calls=outcome.spend.llm_calls,
        )
        # ``diagnosis`` belongs to ExplorationOutcome, not the Explorer Protocol, so an
        # explorer promising only the Protocol still reports a stage.
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

    def _learn(
        self, task: TaskSpec, outcome: RunOutcome, *, enabled: bool
    ) -> tuple[Skill | None, Admission | None, str]:
        """Offer a successful cold run to the admission gate, as ``(skill, admission, note)``.
        An empty note with no skill means learning was not attempted."""
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
        """Name what the reset actually did. The gate has one bool and spells three
        different fixes the same way: a refusing endpoint is a wrong flag, a timeout is a
        bad day, and nothing configured is a mutating task with no way back."""
        report = getattr(self._environment, "last_reset", None)
        if not isinstance(report, ResetReport) or report.restored:
            return reason
        return f"{reason} [{report}]" if reason else str(report)

    def _retrieve(self, task: TaskSpec) -> tuple[Candidate, ...]:
        """What the library offers. A retrieval failure starts the warm path blind; it does
        not fail the run."""
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
        """Write the site graph out - worth keeping even when the run failed."""
        if self._graph is None:
            return
        try:
            self._graph.save()
        except SkillWeaverError as exc:
            log.warning("agent.graph.save_failed", error=str(exc))


@dataclass(frozen=True, slots=True)
class Recollection:
    """Where this task ended last time, and which stored skill remembered it.

    The second half decides the first half's authority (``_warm_critic``): a skill with a
    ``verifier_code`` already answered "did I do my job?" in the sandbox, so its recalled
    screen is a shortcut to a free yes and nothing more. One without a verifier has
    answered nothing, so the recalled screen is the only evidence there is and keeps its veto.
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
    """``recall_end_state``, keeping the skill it came from - the answer is only half
    useful without knowing who remembered it."""
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
    """The screen a successful run of this task ended on, or ``None`` - what lets the warm
    path be judged for free, as a ``TieredCritic``'s ``corroborating_state``.

    A match is decisive; a MISMATCH is not, and that is the whole point. This is where ONE
    run ended with the arguments IT was given: a skill learned on "computer vision" and
    correctly replayed for "machine learning" ends elsewhere, and passing this as
    ``expected_state`` failed that correct replay at 0.120 similarity, twice, on live
    Wikipedia. So it can only ever say yes.

    ``candidates`` is retrieval's offer, best first, and the first whose trajectory still
    reads wins; the library is searched when it is empty.
    """
    return recall(store, trajectories, task, candidates).state


def _load_light(trajectories: TrajectoryStore, run_id: str) -> Trajectory:
    """Ask for a trajectory without screenshots; ``TrajectoryStore`` does not promise the
    keyword, and reading a run's pixels for one fingerprint is pure waste."""
    try:
        return trajectories.load(run_id, screenshots=False)  # type: ignore[call-arg]
    except TypeError:
        return trajectories.load(run_id)


class ComposedPerceiver:
    """A ``Perceiver`` over a detector, a reader and a fingerprinter: capture once, detect
    and read that frame, fuse, index and fingerprint. Boxes are LOGICAL throughout.

    Reading is 84-97% of that on every live page tried, so the reader is wrapped in a
    ``CachingTextReader`` by default and an observation of an untouched screen costs no OCR
    at all. Everything is charged to ``counters``, so the saving is a COUNT and not seconds
    measured against whatever else the machine was doing.

    Args:
        reader: ``None`` runs detection alone, which a machine without the OCR models can
            still do.
        cache_size: ``0`` reads every frame afresh, which is how the optimization is
            turned off to measure against it.

    The text is NOT read lazily, and that is measured, not an omission: ``fingerprint`` is
    computed FROM ``elements``, which include the OCR ones, so asking a screen what state
    it is in already requires the read - and every consumer here compares fingerprints. On
    two live Wikipedia runs, 12 of 12 and 9 of 9 observations read both, so laziness would
    have saved zero. It would pay only for a fingerprinter that does not consume OCR
    elements, and changing what ``StateFingerprinter`` hashes invalidates every stored
    graph node id and skill precondition.
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
            # Reused rather than wrapped again, so two perceivers over one warm cache share
            # its tally instead of each counting half the frames.
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
        """One frame, fully understood. CAPTURES FIRST and reads after, so a post-click
        read judges the frame taken while the click was being answered."""
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
    """What ``perceiver`` has done, or an empty tally: ``Perceiver`` says nothing about
    counters, so this asks politely rather than failing when the answer is no."""
    counters = getattr(perceiver, "counters", None)
    snapshot = getattr(counters, "snapshot", None)
    if snapshot is None:
        return PerceptionCounts()
    result = snapshot()
    return result if isinstance(result, PerceptionCounts) else PerceptionCounts()


RESET_URL_PARAM = "reset_url"
"""The ``TaskSpec.params`` key holding a ``WorldReset`` endpoint: one GET is the whole
undo, which is a demo app's shape and almost nothing else's. ``RESET_ACTIONS_PARAM`` is
the other shape, and a task naming both gets the endpoint first."""


READ_ONLY_PARAM = "read_only"
"""``-p read_only=true``: this task changes nothing, so re-opening the page IS the way
back. Saying so stops the gate reporting a precondition it missed for another reason as a
world it could not restore - and a live site (Wikipedia answers ``403``) has no endpoint
to point ``--reset-url`` at."""


def task_spec(
    text: str,
    *,
    domain: str | None = None,
    target: Literal["browser", "desktop"] = "browser",
    url: str | None = None,
    reset_url: str | None = None,
    reset_steps: Any = None,
    read_only: bool = False,
    params: dict[str, Any] | None = None,
) -> TaskSpec:
    """Build a ``TaskSpec`` the way the command line does; ``domain`` defaults to the host
    of ``url`` for a browser task, else the target's name.

    ``reset_url``, ``reset_steps`` and ``read_only`` are the three answers to "how is this
    world put back?" - see ``WorldReset``. ``reset_steps`` is normalized HERE so a malformed
    list is rejected while still a command-line argument, not halfway through a paid run.
    """
    merged: dict[str, Any] = dict(params or {})
    if url:
        merged.setdefault("start_url", url)
    if reset_url:
        merged.setdefault(RESET_URL_PARAM, reset_url)
    given = reset_steps if reset_steps is not None else merged.get(RESET_ACTIONS_PARAM)
    if given is not None:
        parsed = reset_steps_from(given)
        if parsed:
            merged[RESET_ACTIONS_PARAM] = [reset_step_to_dict(step) for step in parsed]
        else:
            merged.pop(RESET_ACTIONS_PARAM, None)
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


@dataclass(frozen=True, slots=True)
class DomainChoice:
    """Which library namespace a task belongs to, and on whose authority.

    Attributes:
        start_url: Where to open the world. For a looked-up domain this is where the
            winning skill's start screen was last seen, which is what makes a bare
            repeat runnable at all.
        source: ``"named"`` a ``--domain``, ``"url"`` the host of a ``--url``,
            ``"library"`` a stored skill that answered, ``"target"`` nothing did.
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
    path: str = PIXELS,
    k: int = 10,
) -> DomainChoice:
    """Which library namespace ``text`` means, when the caller did not say.

    ``learn`` is given a ``--url`` and files under that host; a repeat has no reason to
    pass one, and resolving that silence to the literal target (``"browser"``) looks up a
    namespace nothing is stored in - a guaranteed miss reported as a plain success at full
    cold price. So the silence is resolved by ASKING THE LIBRARY: every domain is searched,
    and the best candidate the planner would actually run names its own.

    Ranking cannot be the authority, since a library with one skill ranks it first for
    every sentence in the world. A candidate must pass the planner's OWN question - does
    it account for the whole request (``MIN_ACCOUNTED_FOR``) - run with the same code.
    Arguments count towards that as they do there, taken from ``bind_args`` when it binds
    them free and from ``-p`` otherwise; the composer is deliberately unreachable, since
    binding a sentence the cheap binder cannot costs a model call, which a question about
    NAMESPACES may not pay. So the strict direction wins.

    Measured on the live Wikipedia library, 2026-09-19, for *Search Wikipedia for computer
    vision and open the article*: 1.000 for the skill learned from that sentence, 0.667 for
    the *Ada Lovelace* one, 0.500 for a link-follower, 0.000 for a grocery errand. The cut
    sits in the gap, not on top of a case.

    Args:
        domain: ``--domain``. Always wins; ``url``'s host wins next.
        retriever: ``None`` disables the lookup and leaves the old fallback.
        graph: Consulted only for where the winning skill's start screen lives; ``None``
            gives a domain and no URL.
        path: The perception path; every namespace returned belongs to it, and only
            candidates filed under it are considered - the fingerprint will NOT catch a
            crossing, see ``perception_mode``.

    Never raises: failing to look something up is not a reason to refuse to run.
    """
    if domain:
        return DomainChoice(
            domain=namespace(domain, path),
            start_url=url,
            source="named",
            why="named with --domain",
        )
    host = _host(url) if target == "browser" else None
    if host:
        return DomainChoice(
            domain=namespace(host, path),
            start_url=url,
            source="url",
            why="the host of --url",
        )
    if target != "browser" or retriever is None:
        return DomainChoice(
            domain=namespace(target, path),
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
            domain=namespace(target, path),
            start_url=url,
            source="target",
            why=f"the library could not be searched ({exc})",
        )

    passed_over: list[str] = []
    for candidate in candidates:
        skill = candidate.skill
        # A skill from the OTHER path is WRONG, not weaker: its code was written against
        # element text a different reader produced. Skipped before it is scored.
        if path_of(skill.domain) != path:
            continue
        # The planner's own question, never asked more loosely: bound arguments when they
        # come free, the bare text otherwise - what binds them there is the composer.
        args = bind_args(skill, probe) or probe.params
        if not asks_for(probe, skill, args):
            passed_over.append(f"{skill.name}@{skill.domain} (a different errand: other intent)")
            continue
        share, _ = account_of(probe, skill, args, _whole_library(retriever))
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

    # Nothing answers in its own words, but a request may still bind against a RELATIVE's
    # proven sentence - the same workflow learned on another site. The planner's own
    # function, so this cannot say yes to something the planner declines.
    everything = _whole_library(retriever)()
    shelf = [skill for skill in everything if path_of(skill.domain) == path]
    for fit in fit_through_family(probe, shelf, everything):
        log.info(
            "domain.resolved",
            task=text,
            domain=fit.skill.domain,
            skill=fit.skill.name,
            through=f"{fit.via.name}@{fit.via.domain}",
            accounted_for=round(fit.share, 3),
        )
        return DomainChoice(
            domain=fit.skill.domain,
            start_url=_where_it_starts(graph, fit.skill) or url,
            source="library",
            skill=fit.skill.name,
            score=fit.share,
            why=(
                f"the library answered through a family: this request binds against "
                f"{fit.via.name} ({fit.via.domain}), which does what {fit.skill.name} does, "
                f"and {fit.skill.name} is filed under {fit.skill.domain}"
            ),
        )

    log.info(
        "domain.unresolved",
        task=text,
        considered=len(candidates),
        passed_over=", ".join(passed_over) or "(none)",
    )
    return DomainChoice(
        domain=namespace(target, path),
        start_url=url,
        source="target",
        why=(
            f"the whole library was searched and no stored skill accounts for this "
            f"request ({_or_nothing(passed_over)}), so the target's own name is used"
        ),
    )


def _whole_library(retriever: SkillRetriever) -> Callable[[], list[Skill]]:
    """Every healthy skill behind ``retriever``, read only if somebody asks. A retriever
    that does not expose its store has no families to offer."""

    def read() -> list[Skill]:
        store = getattr(retriever, "store", None)
        try:
            return list(store.list()) if store is not None else []
        except SkillWeaverError:
            return []

    return read


def _or_nothing(passed_over: Sequence[str]) -> str:
    """``passed_over`` as one clause, naming what was looked at and turned down."""
    if not passed_over:
        return "it holds nothing"
    return "passed over " + "; ".join(passed_over[:4])


def _where_it_starts(graph: SiteGraph | None, skill: Skill) -> str | None:
    """The URL ``skill``'s start screen was last seen at, or ``None``.

    The other half of a bare repeat: a warm attempt against ``about:blank`` fails at
    ``no_route`` having consulted the right library. The graph's ``url_pattern`` and the
    skill's own precondition answer it together, with no new memory and no guess.
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

    An ``OSError`` so every existing ``WorldReset`` caller keeps working, and a subclass so
    the one that cares can tell the two apart: a ``403`` (live Wikipedia's answer) means
    the flag was wrong, while a timeout means the endpoint was down.
    """


_REFUSALS = frozenset({401, 403, 404, 405, 410, 451, 501})
"""Statuses no retry and no re-run can change, for a GET asking an app to restore itself.
Everything else - a timeout, a refused connection, a ``5xx`` - is a bad day, not a refusal."""


class ResetOutcome(enum.StrEnum):
    """``restored`` and ``unnecessary`` both mean the world is fit to judge a candidate in;
    the other three mean it is not, and differ in whose problem that is."""

    restored = "restored"
    """Something actually put the world back."""

    unnecessary = "unnecessary"
    """The task only reads and navigates."""

    absent = "absent"
    """No way back was configured, and the task did not say it needs none."""

    refused = "refused"
    """The endpoint answered and refused: it is not a reset hook."""

    failed = "failed"
    """A real way back was tried and did not work this time."""


@dataclass(frozen=True, slots=True)
class ResetReport:
    """The outcome of one reset attempt. ``restored`` is the bool ``ReplayEnvironment``
    takes; a REPORT should quote ``outcome``, since the bool spells two opposite problems
    the same way."""

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
    """A ``WorldReset`` that restores the world with one GET - a demo app's "restore seed
    data" endpoint, and almost nothing else. The callable raises ``OSError`` when the
    endpoint cannot be reached, and ``ResetRefused`` for a ``_REFUSALS`` status."""

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
    """Call ``restore`` if there is one and say honestly what happened. Never raises: the
    run this precedes already succeeded, and a traceback out of a reset hook would throw
    away the expensive half.

    ``read_only`` makes the answer ``unnecessary`` whatever the hook did, so a task that
    only read is never blamed for a mutation it never made.
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

    An ``EnvironmentFor`` that also remembers what the last reset came to, in
    ``last_reset``: the gate gets only a bool, which spells "this endpoint is not a reset
    hook" and "nothing can undo this task" the same way. That sent a day of debugging after
    the wrong cause on live Wikipedia, and is why this is a class.

    Two jobs, and only the second is navigation: ``restore`` undoes what the run CHANGED,
    then navigating returns to where it started. A browser can always do the second and
    never the first, which is why a mutating task was unlearnable before ``restore``.

    The navigation target is the TRAJECTORY's first screen, not the task's start URL: a run
    that rescued a failed warm attempt began part-way through the site.

    Declines with ``None`` for a controller that cannot navigate or a run whose first screen
    had no URL; a ``restore`` that raises comes back ``restored=False`` rather than
    propagating out of a learning step.

    Args:
        restore: ``None`` means nothing can, and a mutating task is reported unproved
            rather than rejected.
        read_only: Re-opening the start screen IS the way back, so the gate judges the
            candidate instead of reporting a reset that was never needed.
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
        """What the most recent reset came to, or ``None`` before any candidate stood up."""
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
    """Build a ``NavigatingEnvironment``; see it for what the arguments mean."""
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
    policy: ActingPolicy | None = None,
    budget: Budget | None = None,
    compose: bool = True,
    learn: bool = True,
    max_repairs: int = 2,
    top_k: int = 5,
) -> Agent:
    """Assemble the planner, the explorer and the admission gate around one task - the ONE
    place the real agent is wired. The three critics differ on purpose:

    * **warm**: gets ``recall`` - the recorded end screen AND the skill that remembered it
      - and ``_warm_critic`` decides from the second how much the first may say. Either
      way a warm run landing where the recording landed is judged free.
    * **cold**: a plain ``TieredCritic``; exploration has no recorded end state, and a
      verdict is the cheapest part of a run already paying for every move.
    * **admission**: per trajectory, and DEMANDS the recorded end screen. Right here and
      wrong on the warm path, and the difference is the arguments: the gate re-runs the
      recording with the recorded values, so one end screen is the only correct answer.

    Args:
        llm: Used by exploration, synthesis, composition and any critic that escalates -
            never by a warm action loop.
        recorder: ``None`` lets the explorer choose the configured data directory.
        environment: ``None`` means this agent cannot learn, and it says so rather than
            skipping quietly.
        policy: ``None`` (the default) is ``llm`` through the acting prompt. A policy
            replaces that ONE step; the planner, critics, gate and store are the same
            objects either way.
        compose: Whether the planner may spend one model call chaining known skills for a
            task no single skill covers. ``False`` keeps it strictly model-free.
        max_repairs: How many times the gate may ask for the code to be rewritten.
    """
    retriever = retriever if retriever is not None else build_retriever(store)
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
        policy=policy,
        library=store.list,
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
        policy=policy,
    )


def _warm_critic(llm: LLMClient, recalled: Recollection) -> TieredCritic:
    """How much authority the recalled end screen gets - the difference between a library
    that improves and one that eats itself.

    A skill WITH a verifier already proved it did its own job in the sandbox, so the
    recalled screen is CORROBORATION: a free yes, silent on a miss. It must be, because it
    is where one run ended with ONE set of arguments - holding a correct "machine learning"
    replay of a "computer vision" skill to it rejected it at 0.120 similarity, twice, on
    live Wikipedia. A skill WITHOUT one has proved nothing, so that screen is the only free
    evidence there is and keeps its veto.

    Either way the vetoes still run (``state_changed``, ``no_error_state``) and a missed
    corroboration escalates to a paid model call. A skill's own say-so is never enough.
    """
    if recalled.self_checking:
        return TieredCritic(llm, corroborating_state=recalled.state)
    return TieredCritic(llm, expected_state=recalled.state)


def build_retriever(store: SkillStore, config: Settings | None = None) -> SkillRetriever:
    """The retriever every shipped command ranks with, and the one place an ``Embedder`` is
    constructed, so ``learn``, ``run`` and ``eval run`` rank alike.

    ``load_embedder`` answers in two ``exists`` calls and loads no model, so a command that
    never searches pays nothing. When there is none the REASON travels into every
    candidate's ``why``, and a backend breaking mid-run is dropped rather than taking the
    library down (``degrade_on_error``).
    """
    embedder, reason = load_embedder() if config is None else embedder_for(config)
    if reason:
        log.info("skills.retriever.keywords_only", reason=reason)
    return SkillRetriever(store, embedder, unavailable_reason=reason, degrade_on_error=True)


def _safe_search(retriever: SkillRetriever, task: TaskSpec, top_k: int) -> tuple[Candidate, ...]:
    try:
        return tuple(retriever.search(task.text, domain=task.domain, k=top_k))
    except SkillWeaverError:
        return ()


@dataclass(frozen=True, slots=True)
class Workbench:
    """Everything a command needs, built once per invocation. The cheap memories are eager
    because half the subcommands need nothing else; anything expensive (a browser, a model)
    is behind ``session``, so ``skills ls`` never launches Chromium."""

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
    """The real workbench: file-backed memories and a live browser session. Nothing here
    touches a browser, a model or a network - the session factory does, and only when a
    command opens one."""
    resolved = config if config is not None else settings()
    # The mode is stamped on everything this workbench STORES: a stored screen outlives
    # the window that rendered it (see render_mode). A command that never opens a world
    # writes no skill either, so this cannot mislabel one.
    store = FileSkillStore(resolved.skills_dir, render_mode=mode_name(resolved.headless))
    graph = InMemorySiteGraph(store=JSONGraphStore(resolved.graphs_dir))
    trajectories = TrajectoryFileStore(resolved.trajectories_dir)

    @contextmanager
    def session(task: TaskSpec, budget: Budget) -> Any:
        controller, perceiver = _open_world(resolved, task)
        try:
            graph.load(task.domain)
            llm = _open_model(resolved)
            yield build_agent(
                task,
                controller=controller,
                perceiver=perceiver,
                llm=llm,
                policy=_open_policy(resolved, perceiver, llm),
                store=store,
                retriever=build_retriever(store),
                graph=graph,
                trajectories=trajectories,
                recorder=Recorder(resolved.trajectories_dir),
                environment=navigating_environment(
                    controller,
                    perceiver,
                    graph=graph,
                    restore=_reset_for(task, controller, perceiver),
                    read_only=_is_read_only(task),
                ),
                budget=budget,
            )
        finally:
            controller.close()

    return Workbench(
        settings=resolved,
        store=store,
        retriever=build_retriever(store),
        graph=graph,
        trajectories=trajectories,
        session=session,
    )


def _reset_for(task: TaskSpec, controller: Controller, perceiver: Perceiver) -> WorldReset | None:
    """How to put this task's world back, if it said. A task may name both an endpoint and
    on-screen steps, and gets them in that order so the coarse restore lands before the fine
    tidying reads the screen it left. ``None`` is a real answer, not a failure."""
    url = task.params.get(RESET_URL_PARAM)
    steps = reset_steps_from(task.params.get(RESET_ACTIONS_PARAM) or ())
    return chain_resets(
        world_reset_from_url(str(url)) if url else None,
        world_reset_from_actions(
            steps, controller=controller, perceiver=perceiver, truth=_dom_of(controller)
        )
        if steps
        else None,
    )


def _dom_of(controller: Controller) -> Any:
    """A DOM reader for a reset step that asks for one, or ``None`` on a desktop.

    ``BrowserGroundTruth`` is an offline teacher the AGENT must never see; a world reset is
    scaffolding, the peer of the GET behind ``--reset-url``, and this ONE call site is where
    the dependency is made visible. It exists because a real site names controls where no
    camera can read them - DoorDash's quick-add and header cart are icon-only, with only an
    ``aria-label``, and a search of that page's visible text finds nothing.
    """
    from skillweaver.controllers.browser import BrowserController, BrowserGroundTruth

    return BrowserGroundTruth(controller) if isinstance(controller, BrowserController) else None


def _is_read_only(task: TaskSpec) -> bool:
    """Whether the task declared it changes nothing; ``-p read_only=false`` parses to JSON
    ``False``, so the flag reads the way it is written."""
    return bool(task.params.get(READ_ONLY_PARAM, False))


def _open_world(config: Settings, task: TaskSpec) -> tuple[Controller, Perceiver]:
    """Open the controller the task asks for and the eyes it was configured with.

    Imported here, not at module scope: Playwright, ultralytics and RapidOCR are all slow,
    and ``skills ls`` must not pay for them. ``--perception dom`` on a desktop target raises
    ``ConfigError`` here, before anything opens.
    """
    controller: Controller
    if task.target == "desktop":
        from skillweaver.controllers.desktop import DesktopController

        if config.perception == DOM:
            raise ConfigError(
                "--perception dom is browser-only: a desktop target has no page to ask. "
                "Use --perception pixels, which is the default."
            )
        controller = DesktopController()
    else:
        from skillweaver.controllers.browser import BrowserController

        controller = BrowserController(
            headless=config.headless,
            start_url=task.params.get("start_url"),
            user_data_dir=config.chrome_profile,
            attach=config.chrome_attach,
        )
    return controller, _open_eyes(config)


def _open_eyes(config: Settings) -> Perceiver:
    """The configured perceiver; see ``perception_mode`` for why the two keep separate
    libraries."""
    if config.perception == DOM:
        from skillweaver.perception.dom import DomPerceiver

        return DomPerceiver()
    from skillweaver.perception.detect_yolo import DEFAULT_WEIGHTS_NAME, YoloDetector
    from skillweaver.perception.ocr import RapidOcrReader

    # Not ``default_weights_path()``: it reads the process-wide settings, which a
    # --data-dir on this invocation has already overridden.
    detector = YoloDetector(config.models_dir / DEFAULT_WEIGHTS_NAME)
    return ComposedPerceiver(detector, RapidOcrReader())


def _open_model(config: Settings) -> LLMClient:
    """The computer-use model, whichever acting policy is selected: a policy replaces only
    the MOVE decision, while the critic, synthesizer, composer and the Jev path's own text
    helper are all this client."""
    from skillweaver.llm.anthropic_ import AnthropicClient

    return AnthropicClient(model=config.claude_model, computer_use=True)


def _open_policy(config: Settings, perceiver: Perceiver, llm: LLMClient) -> Any | None:
    """The acting policy, or ``None`` for the default - Claude through the prompt.

    ``--policy jev`` without ``--perception dom`` raises: Jev answers with an index into a
    table of named controls, and OCR gives a box some text was near, not a control with a
    role and a value. A missing Jev credential also raises, before a browser opens - and
    so does a missing ``OPENAI_API_KEY``, because what the policy types is written by an
    OpenAI-compatible text model and there is no fall-back writer
    (:class:`~skillweaver.llm.openai_.OpenAITextWriter` says why the role left ``llm``,
    which is kept in the signature for the callers that pass it and is no longer used).
    """
    if config.policy != "jev":
        return None
    from skillweaver.agent.jev_driver import JevDriver
    from skillweaver.llm.jev_ import JevPolicy
    from skillweaver.llm.openai_ import OpenAITextWriter
    from skillweaver.perception.dom import DomPerceiver

    if not isinstance(perceiver, DomPerceiver):
        raise ConfigError(
            "--policy jev needs --perception dom: the policy chooses an index into the "
            "page's own list of named controls, which only the DOM path produces."
        )
    writer = OpenAITextWriter(
        api_key=config.openai_api_key,
        model=config.text_model,
        base_url=config.text_base_url,
        effort=config.text_effort,
    )
    return JevDriver(JevPolicy(writer, api_key=config.typesafe_api_key), perceiver)


def budget_from(
    config: Settings,
    *,
    max_steps: int | None = None,
    max_seconds: float | None = None,
    max_usd: float | None = None,
    max_llm_calls: int | None = None,
) -> Budget:
    """The configured budget with any explicitly given flag overriding it; a flag NOT given
    never silently resets a configured limit to a default."""
    base = config.default_budget
    return Budget(
        max_steps=base.max_steps if max_steps is None else max_steps,
        max_seconds=base.max_seconds if max_seconds is None else max_seconds,
        max_usd=base.max_usd if max_usd is None else max_usd,
        max_llm_calls=base.max_llm_calls if max_llm_calls is None else max_llm_calls,
    )
