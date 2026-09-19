"""What a run says its EYES cost, and the perceiver that makes that number small.

``AttemptRecord`` already reports the two costs a run is judged on - model calls and
dollars. Perception is the third, and until now it was invisible: the agent could
double the work its eyes did and no report would change. These tests hold the
orchestrator to reporting it, per attempt, in counts that mean the same thing on a
loaded machine as on an idle one.

The wiring here is deliberately shallow: stub a planner and an explorer that OBSERVE a
known number of times, and the report has to say so. Nothing loads a model, a browser
or a network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from skillweaver.contracts import (
    Budget,
    Controller,
    Observation,
    Perceiver,
    RunOutcome,
    Spend,
    TaskSpec,
    Trajectory,
    Verdict,
)
from skillweaver.orchestrator import (
    Agent,
    AttemptRecord,
    ComposedPerceiver,
    PerceptionCounts,
    RunReport,
    perception_counts,
    task_spec,
)
from skillweaver.perception.ocr import PerceptionCounters
from skillweaver.skills.retrieve import SkillRetriever
from tests.fakes import (
    FakeDetector,
    FakeTextReader,
    InMemorySkillStore,
    Scenario,
)


@pytest.fixture
def eyes(scenario: Scenario) -> ComposedPerceiver:
    """The real composing perceiver over the fake scenario."""
    return ComposedPerceiver(
        FakeDetector(scenario.controller),
        FakeTextReader(scenario.controller),
        scenario.perceiver.fingerprinter,
    )


# --------------------------------------------------------------------------------------
# What one attempt is charged
# --------------------------------------------------------------------------------------


class _Explorer:
    """An explorer that observes ``looks`` times and then reports what it was told to."""

    def __init__(self, perceiver: Perceiver, task: TaskSpec, *, looks: int, ok: bool) -> None:
        self._perceiver = perceiver
        self._task = task
        self._looks = looks
        self._ok = ok

    def explore(self, task: TaskSpec, controller: Controller, budget: Budget | None) -> RunOutcome:
        for _ in range(self._looks):
            self._perceiver.observe(controller)
        now = datetime(2026, 9, 19, tzinfo=UTC)
        trajectory = Trajectory(
            run_id="run-eyes",
            task=task.text,
            domain=task.domain,
            steps=(),
            ok=self._ok,
            started_at=now,
            finished_at=now,
        )
        return RunOutcome(
            ok=self._ok,
            trajectory=trajectory,
            verdict=Verdict(ok=self._ok, reason="stubbed"),
            spend=Spend(),
            note="stubbed exploration",
        )


def _agent(scenario: Scenario, eyes: ComposedPerceiver, *, looks: int, ok: bool = True) -> Agent:
    store = InMemorySkillStore()
    return Agent(
        controller=scenario.controller,
        perceiver=eyes,
        store=store,
        retriever=SkillRetriever(store),
        planner=None,  # type: ignore[arg-type]
        explorer=_Explorer(eyes, scenario.task, looks=looks, ok=ok),  # type: ignore[arg-type]
    )


def test_an_attempt_reports_what_it_made_the_eyes_do(
    scenario: Scenario, eyes: ComposedPerceiver
) -> None:
    report = _agent(scenario, eyes, looks=4).run(scenario.task, warm=False, learn=False)

    cold = report.cold
    assert cold is not None
    assert cold.perception.observations == 4
    assert cold.perception.detections == 4
    # Four looks at one unchanged screen: one read, three hits. That is the saving.
    assert (cold.perception.ocr_reads, cold.perception.ocr_hits) == (1, 3)
    assert report.perception == cold.perception


def test_each_attempt_is_charged_only_its_own_share(
    scenario: Scenario, eyes: ComposedPerceiver
) -> None:
    """A mark is taken per attempt, so the cold record excludes the warm one's frames."""
    eyes.observe(scenario.controller)  # work done before the run is nobody's attempt
    report = _agent(scenario, eyes, looks=3).run(scenario.task, warm=False, learn=False)

    cold = report.cold
    assert cold is not None
    assert cold.perception.observations == 3
    assert eyes.counters.observations == 4, "the perceiver's own tally keeps climbing"


def test_a_report_with_no_attempts_reports_no_perception() -> None:
    assert RunReport(ok=False, task=task_spec("nothing"), decision="none").perception == (
        PerceptionCounts()
    )


def test_run_perception_sums_every_attempt() -> None:
    warm = AttemptRecord(
        path="warm", ok=False, reason="no", perception=PerceptionCounts(observations=2, ocr_reads=2)
    )
    cold = AttemptRecord(
        path="cold", ok=True, reason="yes", perception=PerceptionCounts(observations=5, ocr_hits=4)
    )
    report = RunReport(ok=True, task=task_spec("t"), decision="cold", attempts=(warm, cold))
    assert report.perception == PerceptionCounts(observations=7, ocr_reads=2, ocr_hits=4)


# --------------------------------------------------------------------------------------
# What a human reads
# --------------------------------------------------------------------------------------


def test_explain_states_the_ocr_count_beside_the_model_count(
    scenario: Scenario, eyes: ComposedPerceiver
) -> None:
    report = _agent(scenario, eyes, looks=5).run(scenario.task, warm=False, learn=False)
    prose = report.explain()
    assert "perception: 1 OCR read(s) for 5 observation(s) - 4 served from cache (80%)" in prose


def test_explain_says_nothing_about_eyes_that_did_nothing() -> None:
    report = RunReport(
        ok=False,
        task=task_spec("t"),
        decision="none",
        attempts=(AttemptRecord(path="warm", ok=False, reason="empty library"),),
    )
    assert "perception:" not in report.explain()


def test_an_attempt_line_carries_the_counts_without_them_being_hunted_for() -> None:
    record = AttemptRecord(
        path="cold",
        ok=True,
        reason="explored to the goal",
        llm_calls=7,
        perception=PerceptionCounts(observations=9, detections=9, ocr_reads=3, ocr_hits=6),
    )
    assert "7 model call(s), 9 observation(s), 3 OCR read(s) + 6 cached" in str(record)


# --------------------------------------------------------------------------------------
# Asking a perceiver that does not count
# --------------------------------------------------------------------------------------


def test_a_run_through_a_perceiver_that_does_not_count_still_reports(scenario: Scenario) -> None:
    """``Perceiver`` promises no counters; a run must report zero, not explode."""
    fake = scenario.perceiver
    assert perception_counts(fake) == PerceptionCounts()
    store = InMemorySkillStore()
    agent = Agent(
        controller=scenario.controller,
        perceiver=fake,
        store=store,
        retriever=SkillRetriever(store),
        planner=None,  # type: ignore[arg-type]
        explorer=_Explorer(fake, scenario.task, looks=2, ok=True),  # type: ignore[arg-type]
    )
    report = agent.run(scenario.task, warm=False, learn=False)
    assert report.ok
    assert report.perception == PerceptionCounts()
    assert "perception:" not in report.explain()


# --------------------------------------------------------------------------------------
# The perceiver itself
# --------------------------------------------------------------------------------------


def test_the_composed_perceiver_satisfies_the_perceiver_protocol(
    eyes: ComposedPerceiver,
) -> None:
    from skillweaver import contracts

    assert isinstance(eyes, contracts.Perceiver)


def test_an_observation_is_complete_whether_or_not_its_text_came_from_the_cache(
    scenario: Scenario, eyes: ComposedPerceiver
) -> None:
    """A cheaper observation is not a smaller one: every field is fully populated."""
    first = eyes.observe(scenario.controller)
    cached = eyes.observe(scenario.controller)

    assert eyes.counters.ocr_hits == 1
    assert cached.elements == first.elements
    assert cached.fingerprint == first.fingerprint
    assert cached.index.all() == first.index.all()
    assert cached.url == first.url
    assert isinstance(cached, Observation) and type(cached) is Observation


def test_a_shared_counter_lets_a_caller_measure_one_stretch_in_isolation(
    scenario: Scenario,
) -> None:
    counters = PerceptionCounters()
    eyes = ComposedPerceiver(
        FakeDetector(scenario.controller),
        FakeTextReader(scenario.controller),
        counters=counters,
    )
    eyes.observe(scenario.controller)
    counters.reset()
    eyes.observe(scenario.controller)
    assert counters.snapshot() == PerceptionCounts(
        observations=1, captures=1, detections=1, ocr_hits=1
    )
