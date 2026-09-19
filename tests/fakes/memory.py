"""In-memory doubles for the memory Protocols: skill store, site graph, trajectory
recorder and trajectory store. Nothing is written to disk."""

from __future__ import annotations

import heapq
import itertools
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from skillweaver.contracts import (
    Action,
    ActionResult,
    Fingerprint,
    Observation,
    Route,
    Skill,
    SkillStats,
    Trajectory,
    TrajectoryStep,
    Transition,
    UIState,
    Verdict,
    utcnow,
)
from skillweaver.errors import SkillNotFound, SkillWeaverError


def _fold_mean(mean: float, count_before: int, value: float) -> float:
    return (mean * count_before + value) / (count_before + 1)


class InMemorySkillStore:
    """A ``contracts.SkillStore`` holding every version of every skill in a dict."""

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], list[Skill]] = {}

    def put(self, skill: Skill) -> Skill:
        versions = self._versions.setdefault((skill.name, skill.domain), [])
        stored = replace(skill, version=len(versions) + 1)
        versions.append(stored)
        return stored

    def get(self, name: str, domain: str, version: int | None = None) -> Skill:
        versions = self._versions.get((name, domain))
        if not versions:
            raise SkillNotFound(f"no skill {name!r} in domain {domain!r}")
        if version is None:
            return versions[-1]
        if not 1 <= version <= len(versions):
            raise SkillNotFound(f"skill {name!r} in domain {domain!r} has no version {version}")
        return versions[version - 1]

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[Skill]:
        latest = (versions[-1] for versions in self._versions.values())
        return sorted(
            (
                s
                for s in latest
                if (domain is None or s.domain == domain)
                and (include_demoted or s.demoted_reason is None)
            ),
            key=lambda s: (s.domain, s.name),
        )

    def record_run(self, name: str, domain: str, ok: bool, ms: float) -> Skill:
        skill = self.get(name, domain)
        old = skill.stats
        stats = SkillStats(
            runs=old.runs + 1,
            successes=old.successes + (1 if ok else 0),
            mean_ms=_fold_mean(old.mean_ms, old.successes, ms) if ok else old.mean_ms,
            last_ok_at=utcnow() if ok else old.last_ok_at,
        )
        return self._replace_latest(replace(skill, stats=stats))

    def demote(self, name: str, domain: str, reason: str) -> Skill:
        return self._replace_latest(replace(self.get(name, domain), demoted_reason=reason))

    def _replace_latest(self, skill: Skill) -> Skill:
        self._versions[(skill.name, skill.domain)][-1] = skill
        return skill


_EdgeKey = tuple[str, str, tuple[Action, ...]]


class InMemorySiteGraph:
    """A ``contracts.SiteGraph`` in dicts. ``route`` is Dijkstra over edges with at
    least one success, each costing ``mean_ms / success_rate``. ``save`` and ``load``
    do nothing."""

    def __init__(self) -> None:
        self._states: dict[str, UIState] = {}
        self._edges: dict[_EdgeKey, Transition] = {}

    def upsert_state(self, state: UIState) -> UIState:
        known = self._states.get(state.fingerprint.value)
        if known is not None:
            state = replace(state, first_seen=known.first_seen)
        self._states[state.fingerprint.value] = state
        return state

    def observe_transition(
        self, src: Fingerprint, actions: Sequence[Action], dst: Fingerprint, ok: bool, ms: float
    ) -> Transition:
        domain = self._states[src.value].domain if src.value in self._states else ""
        for fp in (src, dst):
            if fp.value not in self._states:
                self._states[fp.value] = UIState(fingerprint=fp, domain=domain)
        key: _EdgeKey = (src.value, dst.value, tuple(actions))
        old = self._edges.get(key) or Transition(src, dst, tuple(actions))
        edge = replace(
            old,
            attempts=old.attempts + 1,
            successes=old.successes + (1 if ok else 0),
            mean_ms=_fold_mean(old.mean_ms, old.successes, ms) if ok else old.mean_ms,
            last_verified=utcnow() if ok else old.last_verified,
        )
        self._edges[key] = edge
        return edge

    def route(self, src_fp: Fingerprint, dst_fp: Fingerprint) -> Route | None:
        if src_fp.value == dst_fp.value:
            return Route((), 0.0, ())
        tiebreak = itertools.count()
        queue: list[tuple[float, int, str, tuple[Transition, ...]]] = [
            (0.0, next(tiebreak), src_fp.value, ())
        ]
        settled: set[str] = set()
        while queue:
            cost, _, node, path = heapq.heappop(queue)
            if node == dst_fp.value:
                steps = tuple(a for edge in path for a in edge.actions)
                return Route(steps, cost, path)
            if node in settled:
                continue
            settled.add(node)
            for edge in self.neighbors(Fingerprint(node)):
                if edge.successes > 0 and edge.dst.value not in settled:
                    edge_cost = edge.mean_ms / (edge.successes / edge.attempts)
                    heapq.heappush(
                        queue, (cost + edge_cost, next(tiebreak), edge.dst.value, (*path, edge))
                    )
        return None

    def neighbors(self, fp: Fingerprint) -> list[Transition]:
        outgoing = [e for e in self._edges.values() if e.src.value == fp.value]
        return sorted(outgoing, key=lambda e: -(e.successes / e.attempts if e.attempts else 0.0))

    def states(self, domain: str) -> list[UIState]:
        return sorted(
            (s for s in self._states.values() if s.domain == domain), key=lambda s: s.first_seen
        )

    def save(self) -> None:
        return None

    def load(self, domain: str) -> None:
        return None


class InMemoryTrajectoryRecorder:
    """A ``contracts.TrajectoryRecorder`` building one trajectory at a time in a list."""

    def __init__(self) -> None:
        self._run_id: str | None = None
        self._task = ""
        self._domain = ""
        self._started_at: datetime | None = None
        self._steps: list[TrajectoryStep] = []

    def start(self, task: str, domain: str) -> str:
        if self._run_id is not None:
            raise SkillWeaverError(f"run {self._run_id} is still in progress")
        self._run_id = uuid.uuid4().hex[:12]
        self._task, self._domain = task, domain
        self._started_at = utcnow()
        self._steps = []
        return self._run_id

    def step(
        self,
        action: Action,
        before: Observation,
        after: Observation,
        result: ActionResult,
        verdict: Verdict | None = None,
        note: str = "",
    ) -> TrajectoryStep:
        if self._run_id is None:
            raise SkillWeaverError("no run in progress: call start() first")
        recorded = TrajectoryStep(len(self._steps), action, before, after, result, verdict, note)
        self._steps.append(recorded)
        return recorded

    def finish(self, ok: bool, note: str = "") -> Trajectory:
        if self._run_id is None or self._started_at is None:
            raise SkillWeaverError("no run in progress: call start() first")
        trajectory = Trajectory(
            run_id=self._run_id,
            task=self._task,
            domain=self._domain,
            steps=tuple(self._steps),
            ok=ok,
            started_at=self._started_at,
            finished_at=utcnow(),
            note=note,
        )
        self._run_id = None
        return trajectory


class InMemoryTrajectoryStore:
    """A ``contracts.TrajectoryStore`` in an insertion-ordered dict."""

    def __init__(self) -> None:
        self._trajectories: dict[str, Trajectory] = {}

    def save(self, trajectory: Trajectory) -> None:
        self._trajectories[trajectory.run_id] = trajectory

    def load(self, run_id: str) -> Trajectory:
        try:
            return self._trajectories[run_id]
        except KeyError:
            raise SkillWeaverError(f"no trajectory with run_id {run_id!r}") from None

    def list(self) -> list[str]:
        return list(self._trajectories)
