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
from typing import Any

import pytest

from skillweaver.agent.planner import PlanFailure
from skillweaver.contracts import (
    Budget,
    Controller,
    LLMMessage,
    LLMResponse,
    Observation,
    Perceiver,
    Provenance,
    RunOutcome,
    Skill,
    Spend,
    TaskSpec,
    Trajectory,
    Usage,
    Verdict,
)
from skillweaver.orchestrator import (
    READ_ONLY_PARAM,
    Agent,
    AttemptRecord,
    ComposedPerceiver,
    PerceptionCounts,
    Recollection,
    ResetOutcome,
    ResetRefused,
    RunReport,
    navigating_environment,
    perception_counts,
    recall,
    reset_world,
    task_spec,
)
from skillweaver.perception.ocr import PerceptionCounters
from skillweaver.skills.retrieve import SkillRetriever
from tests.fakes import (
    FakeDetector,
    FakeLLM,
    FakeTextReader,
    InMemorySkillStore,
    InMemoryTrajectoryStore,
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


# --------------------------------------------------------------------------------------
# What a run says its MODEL calls cost
# --------------------------------------------------------------------------------------


class _Planner:
    """A planner that spends ``calls`` on the model and then reports what it was told.

    It stands in for the real one at the only place that matters here: the composer
    pays for a chain, the plan is then discarded at routing, and the outcome that
    would have carried the bill is thrown away with it.
    """

    def __init__(
        self, llm: FakeLLM, *, calls: int, outcome: RunOutcome | None, failure: PlanFailure | None
    ) -> None:
        self._llm = llm
        self._calls = calls
        self._outcome = outcome
        self.last_failure = failure

    def plan(self, task: TaskSpec, observation: Observation) -> None:  # pragma: no cover
        raise AssertionError("the agent plans through attempt()")

    def attempt(self, task: TaskSpec, observation: Observation) -> RunOutcome | None:
        for _ in range(self._calls):
            self._llm.complete([LLMMessage(role="user", text="chain these skills")])
        return self._outcome


def _warm_agent(
    scenario: Scenario,
    *,
    llm: FakeLLM | None,
    planner: Any,
    store: Any = None,
) -> Agent:
    """An agent whose library is not empty, so the warm path is actually tried."""
    store = store if store is not None else _store_with(_skill())
    return Agent(
        controller=scenario.controller,
        perceiver=scenario.perceiver,
        store=store,
        retriever=SkillRetriever(store),
        planner=planner,
        explorer=_Explorer(scenario.perceiver, scenario.task, looks=0, ok=False),  # type: ignore[arg-type]
        llm=llm,
    )


def _skill(*, verifier: str | None = "def verify(ctx, result):\n    return True\n") -> Skill:
    return Skill(
        name="search_wikipedia",
        domain=DOMAIN,
        summary="Search Wikipedia and open the article.",
        docstring="Searches for `query` and opens the article.",
        params={"query": {"type": "string"}},
        code="def run(ctx, query):\n    return True\n",
        requires=(),
        precondition=None,
        verifier_code=verifier,
        provenance=Provenance("run-taught", "Search Wikipedia for computer vision", "m", _NOW),
    )


def _store_with(*skills: Skill) -> InMemorySkillStore:
    store = InMemorySkillStore()
    for skill in skills:
        store.put(skill)
    return store


DOMAIN = "fake.test"  # the scenario's own domain, so the warm path is tried
_NOW = datetime(2026, 9, 19, tzinfo=UTC)
_REPLY = LLMResponse(text="{}", usage=Usage(calls=1, cost_usd=0.02))


def test_a_call_the_composer_spent_is_charged_even_when_the_plan_is_discarded(
    scenario: Scenario,
) -> None:
    """THE undercount. A warm attempt that fails at routing used to report zero.

    The composer really does spend a model call to read a plain-English task; when the
    plan it proposes cannot be routed to, the outcome carrying that cost is thrown
    away. Reporting zero there makes every efficiency figure downstream flatter than
    the bill, which is the one direction this project cannot round in.
    """
    llm = FakeLLM([_REPLY])
    planner = _Planner(
        llm,
        calls=1,
        outcome=None,
        failure=PlanFailure(stage="no_route", reason="no known route to its start screen"),
    )

    report = _warm_agent(scenario, llm=llm, planner=planner).run(
        scenario.task, warm=True, cold=False, learn=False
    )

    warm = report.warm
    assert warm is not None and warm.ok is False and warm.stage == "no_route"
    assert warm.llm_calls == 1, "the composer's call vanished with the plan it bought"
    assert warm.usd == pytest.approx(0.02)
    assert report.llm_calls == 1


def test_a_critic_that_had_to_escalate_is_charged_to_the_warm_attempt(
    scenario: Scenario,
) -> None:
    """A warm run judged by the model is a warm run that cost one call, even though
    the planner's own spend only ever counts what the planner itself spent."""
    llm = FakeLLM([_REPLY])
    outcome = RunOutcome(
        ok=True,
        trajectory=Trajectory("r", "t", DOMAIN, (), True, _NOW, _NOW),
        verdict=Verdict(ok=True, reason="the model said so", source="model"),
        spend=Spend(),  # the planner charges itself nothing for the critic
        skill_used="search_wikipedia",
        note="warm path",
    )
    planner = _Planner(llm, calls=1, outcome=outcome, failure=None)

    report = _warm_agent(scenario, llm=llm, planner=planner).run(
        scenario.task, warm=True, cold=False, learn=False
    )

    assert report.ok and report.decision == "warm"
    assert report.warm is not None and report.warm.llm_calls == 1


def test_an_attempt_is_never_talked_down_by_the_meter(scenario: Scenario) -> None:
    """The meter only ever finds calls; it never argues one away.

    A path that reached a model this agent cannot see - a second client, a tool the
    planner owns - would otherwise be silently zeroed by a meter that read nothing.
    """
    spend = Spend()
    spend.add_usage(Usage(calls=3, cost_usd=0.5))
    outcome = RunOutcome(
        ok=True,
        trajectory=Trajectory("r", "t", DOMAIN, (), True, _NOW, _NOW),
        verdict=Verdict(ok=True),
        spend=spend,
        note="warm path",
    )
    llm = FakeLLM()
    planner = _Planner(llm, calls=0, outcome=outcome, failure=None)

    report = _warm_agent(scenario, llm=llm, planner=planner).run(
        scenario.task, warm=True, cold=False, learn=False
    )

    assert report.warm is not None
    assert report.warm.llm_calls == 3 and report.warm.usd == pytest.approx(0.5)


def test_an_agent_with_no_model_client_reports_what_each_attempt_says(
    scenario: Scenario,
) -> None:
    """``llm=None`` is the old behaviour exactly, not a crash and not a zero."""
    spend = Spend()
    spend.add_usage(Usage(calls=2))
    outcome = RunOutcome(
        ok=True,
        trajectory=Trajectory("r", "t", DOMAIN, (), True, _NOW, _NOW),
        verdict=Verdict(ok=True),
        spend=spend,
        note="warm path",
    )
    planner = _Planner(FakeLLM(), calls=0, outcome=outcome, failure=None)

    report = _warm_agent(scenario, llm=None, planner=planner).run(
        scenario.task, warm=True, cold=False, learn=False
    )

    assert report.warm is not None and report.warm.llm_calls == 2


def test_a_model_client_that_cannot_be_metered_does_not_fail_the_run(
    scenario: Scenario,
) -> None:
    class _Unmetered:
        def total_usage(self) -> Usage:
            raise RuntimeError("no idea")

    outcome = RunOutcome(
        ok=True,
        trajectory=Trajectory("r", "t", DOMAIN, (), True, _NOW, _NOW),
        verdict=Verdict(ok=True),
        spend=Spend(),
        note="warm path",
    )
    planner = _Planner(FakeLLM(), calls=0, outcome=outcome, failure=None)
    report = _warm_agent(scenario, llm=_Unmetered(), planner=planner).run(  # type: ignore[arg-type]
        scenario.task, warm=True, cold=False, learn=False
    )
    assert report.ok and report.warm is not None and report.warm.llm_calls == 0


def test_each_attempt_is_charged_only_the_calls_it_made(scenario: Scenario) -> None:
    """A warm attempt's calls must not reappear on the cold record after it."""
    llm = FakeLLM([_REPLY, _REPLY])
    planner = _Planner(
        llm, calls=1, outcome=None, failure=PlanFailure(stage="no_route", reason="no route")
    )
    store = _store_with(_skill())
    agent = Agent(
        controller=scenario.controller,
        perceiver=scenario.perceiver,
        store=store,
        retriever=SkillRetriever(store),
        planner=planner,  # type: ignore[arg-type]
        explorer=_Explorer(scenario.perceiver, scenario.task, looks=1, ok=True),  # type: ignore[arg-type]
        llm=llm,
    )

    report = agent.run(scenario.task, learn=False)

    warm, cold = report.warm, report.cold
    assert warm is not None and cold is not None
    assert warm.llm_calls == 1
    assert cold.llm_calls == 0, "the warm attempt's bill landed on the cold record"


# --------------------------------------------------------------------------------------
# Putting the world back, and saying honestly what happened
# --------------------------------------------------------------------------------------


def _refuses() -> None:
    raise ResetRefused("https://en.wikipedia.org/__reset answered HTTP 403")


def _breaks() -> None:
    raise OSError("connection refused")


class TestResetHonesty:
    """Three different problems that a single ``restored=False`` spells the same way.

    "This endpoint is not a reset hook" is a wrong flag, "the hook did not answer" is
    an endpoint having a bad day, and "nothing was configured" is a mutating task with
    no way back. Reporting all three as a failed reset sent a day of live debugging
    after a mutation that never happened.
    """

    def test_a_working_hook_restores(self) -> None:
        called: list[int] = []
        report = reset_world(lambda: called.append(1))
        assert called == [1]
        assert report.outcome is ResetOutcome.restored and report.restored is True

    def test_no_hook_at_all_is_absent_not_failed(self) -> None:
        report = reset_world(None)
        assert report.outcome is ResetOutcome.absent
        assert report.restored is False
        assert "cannot be proved" in report.detail

    def test_a_refusal_is_not_a_failure(self) -> None:
        report = reset_world(_refuses)
        assert report.outcome is ResetOutcome.refused
        assert report.restored is False
        assert "403" in report.detail

    def test_a_hook_that_broke_is_a_failure_and_says_which(self) -> None:
        report = reset_world(_breaks)
        assert report.outcome is ResetOutcome.failed
        assert report.restored is False
        assert "connection refused" in report.detail
        assert report.outcome is not ResetOutcome.refused

    def test_a_read_only_task_needs_no_reset_and_is_not_a_failure(self) -> None:
        """The live case: Wikipedia has no reset endpoint and needs none."""
        report = reset_world(None, read_only=True)
        assert report.outcome is ResetOutcome.unnecessary
        assert report.restored is True, "a task that changed nothing is already back"

    def test_a_read_only_task_survives_a_refusing_endpoint(self) -> None:
        """A 403 from a site that was never going to reset is not this run's problem,
        and the detail still records what the endpoint said."""
        report = reset_world(_refuses, read_only=True)
        assert report.outcome is ResetOutcome.unnecessary and report.restored is True
        assert "403" in report.detail

    def test_a_hook_that_raises_never_escapes(self) -> None:
        """The expensive half of the work is already done; a traceback here loses it."""
        for hook in (_refuses, _breaks):
            assert reset_world(hook).restored is False

    def test_the_report_reads_as_one_line(self) -> None:
        assert str(reset_world(_refuses)).startswith("reset refused: ")


class TestResetFromUrl:
    """``world_reset_from_url`` keeps raising ``OSError``; a refusal is a subclass."""

    @staticmethod
    def _answering(code: int) -> Any:
        import urllib.error

        def fake_urlopen(url: str, timeout: float = 0.0) -> Any:
            raise urllib.error.HTTPError(url, code, "no", None, None)  # type: ignore[arg-type]

        return fake_urlopen

    def _call(self, code: int) -> None:
        import urllib.request

        from skillweaver.orchestrator import world_reset_from_url

        original = urllib.request.urlopen
        urllib.request.urlopen = self._answering(code)  # type: ignore[assignment]
        try:
            world_reset_from_url("https://en.wikipedia.org/__reset")()
        finally:
            urllib.request.urlopen = original  # type: ignore[assignment]

    def test_a_403_is_a_refusal_with_a_sentence_a_stranger_can_act_on(self) -> None:
        with pytest.raises(ResetRefused, match="not a reset endpoint"):
            self._call(403)

    @pytest.mark.parametrize("code", [401, 404, 405, 501])
    def test_the_other_permanent_answers_are_refusals_too(self, code: int) -> None:
        with pytest.raises(ResetRefused):
            self._call(code)

    def test_a_server_having_a_bad_day_is_still_a_plain_failure(self) -> None:
        with pytest.raises(OSError) as caught:
            self._call(503)
        assert not isinstance(caught.value, ResetRefused)

    def test_a_refusal_is_an_oserror_so_every_existing_caller_still_catches_it(self) -> None:
        assert issubclass(ResetRefused, OSError)


class _Navigable:
    """The one thing :class:`NavigatingEnvironment` needs of a controller: a way back
    to a URL. The fake invoicing app deliberately cannot navigate, so this stands in."""

    def __init__(self) -> None:
        self.went: list[str] = []

    def supports(self, capability: str) -> bool:
        return capability == "navigate"

    def perform(self, action: Any) -> None:
        self.went.append(action.url)


def _navigated_trajectory(scenario: Scenario) -> Trajectory:
    """A one-step trajectory whose first screen has a URL to go back to."""
    from dataclasses import replace as _replace

    from skillweaver.contracts import ActionResult, Click, Point, TrajectoryStep

    seen = scenario.perceiver.observe(scenario.controller)
    before = _replace(seen, url="https://en.wikipedia.org/wiki/Main_Page")
    action = Click(Point(10, 10))
    return Trajectory(
        run_id="run-1",
        task=scenario.task.text,
        domain=scenario.task.domain,
        steps=(TrajectoryStep(0, action, before, seen, ActionResult(ok=True)),),
        ok=True,
        started_at=_NOW,
        finished_at=_NOW,
    )


class TestNavigatingEnvironment:
    def test_a_read_only_task_is_stood_up_as_restored_with_no_hook_at_all(
        self, scenario: Scenario
    ) -> None:
        trajectory = _navigated_trajectory(scenario)
        world = navigating_environment(_Navigable(), scenario.perceiver, read_only=True)

        factory = world(trajectory)
        assert factory is not None
        assert factory().restored is True
        assert world.last_reset is not None
        assert world.last_reset.outcome is ResetOutcome.unnecessary

    def test_without_that_declaration_nothing_changes(self, scenario: Scenario) -> None:
        """Re-navigating is not a reset, and never became one."""
        trajectory = _navigated_trajectory(scenario)
        world = navigating_environment(_Navigable(), scenario.perceiver)

        factory = world(trajectory)
        assert factory is not None
        assert factory().restored is False
        assert world.last_reset is not None
        assert world.last_reset.outcome is ResetOutcome.absent

    def test_the_last_reset_names_a_refusal_rather_than_a_mutation(
        self, scenario: Scenario
    ) -> None:
        trajectory = _navigated_trajectory(scenario)
        world = navigating_environment(_Navigable(), scenario.perceiver, restore=_refuses)

        factory = world(trajectory)
        assert factory is not None
        assert factory().restored is False
        assert world.last_reset is not None
        assert world.last_reset.outcome is ResetOutcome.refused

    def test_nothing_is_remembered_before_a_candidate_is_stood_up(self, scenario: Scenario) -> None:
        world = navigating_environment(_Navigable(), scenario.perceiver)
        assert world.last_reset is None


def test_a_task_can_declare_that_it_changes_nothing() -> None:
    spec = task_spec(
        "read the article", url="https://en.wikipedia.org/wiki/Main_Page", read_only=True
    )
    assert spec.params[READ_ONLY_PARAM] is True
    assert READ_ONLY_PARAM not in task_spec("archive it", url="https://mail.test").params


# --------------------------------------------------------------------------------------
# How much authority a recalled end screen gets
# --------------------------------------------------------------------------------------


def _taught_by(scenario: Scenario, skill: Skill) -> InMemoryTrajectoryStore:
    """A trajectory store holding the run that taught ``skill``."""
    from skillweaver.contracts import ActionResult, Click, Point, TrajectoryStep

    before = scenario.perceiver.observe(scenario.controller)
    scenario.controller.state = "done"
    after = scenario.perceiver.observe(scenario.controller)
    store = InMemoryTrajectoryStore()
    store.save(
        Trajectory(
            run_id=skill.provenance.trajectory_id,
            task=skill.provenance.task_text,
            domain=skill.domain,
            steps=(TrajectoryStep(0, Click(Point(10, 10)), before, after, ActionResult(ok=True)),),
            ok=True,
            started_at=_NOW,
            finished_at=_NOW,
        )
    )
    scenario.controller.state = "list"
    return store


class TestRecall:
    """A recalled end screen without the skill it came from is only half an answer."""

    def test_it_names_the_skill_that_remembered_the_screen(self, scenario: Scenario) -> None:
        skill = _skill()
        store = _store_with(skill)
        recalled = recall(store, _taught_by(scenario, skill), scenario.task)

        assert recalled.state is not None
        assert recalled.source is not None and recalled.source.name == skill.name
        assert recalled.self_checking is True

    def test_a_skill_that_cannot_check_itself_says_so(self, scenario: Scenario) -> None:
        skill = _skill(verifier=None)
        store = _store_with(skill)
        recalled = recall(store, _taught_by(scenario, skill), scenario.task)

        assert recalled.state is not None
        assert recalled.self_checking is False

    def test_nothing_recorded_recalls_nothing(self, scenario: Scenario) -> None:
        assert recall(_store_with(_skill()), None, scenario.task) == Recollection()
        assert Recollection().self_checking is False

    def test_the_old_entry_point_still_answers_the_old_question(self, scenario: Scenario) -> None:
        skill = _skill()
        store = _store_with(skill)
        trajectories = _taught_by(scenario, skill)
        from skillweaver.orchestrator import recall_end_state

        assert recall_end_state(store, trajectories, scenario.task) == (
            recall(store, trajectories, scenario.task).state
        )


class TestWarmCriticAuthority:
    """Which role the recalled screen gets, decided by who remembered it.

    This is the fix for the defect that demoted a skill for succeeding, and the guard
    against overcorrecting into trusting a skill that cannot check itself.
    """

    def _critic(self, scenario: Scenario, skill: Skill) -> Any:
        from skillweaver.orchestrator import build_agent

        store = _store_with(skill)
        agent = build_agent(
            scenario.task,
            controller=scenario.controller,
            perceiver=scenario.perceiver,
            llm=FakeLLM(),
            store=store,
            trajectories=_taught_by(scenario, skill),
        )
        # Reaching past one private name is the point: this asserts the real wiring
        # rather than re-deciding the same thing beside it.
        return agent._planner._critic  # noqa: SLF001

    def test_a_self_checking_skill_gets_corroboration(self, scenario: Scenario) -> None:
        critic = self._critic(scenario, _skill())
        assert [c.name for c in critic.corroboration_checks] == ["matches_state"]
        assert critic.evidence_checks == (), "the recalled screen must not be able to veto"

    def test_a_skill_with_no_verifier_keeps_the_veto(self, scenario: Scenario) -> None:
        """It has proved nothing on its own, so the recalled screen is all there is."""
        critic = self._critic(scenario, _skill(verifier=None))
        assert [c.name for c in critic.evidence_checks] == ["matches_state"]
        assert critic.corroboration_checks == ()

    def test_the_vetoes_are_there_either_way(self, scenario: Scenario) -> None:
        for skill in (_skill(), _skill(verifier=None)):
            names = [c.name for c in self._critic(scenario, skill).veto_checks]
            assert names == ["state_changed", "no_error_state"]


# --------------------------------------------------------------------------------------
# What a failed learning step says went wrong
# --------------------------------------------------------------------------------------


class TestLearningNote:
    """A reader chasing a failed admission needs the real cause, not a bool."""

    def _report(self, scenario: Scenario, world: Any) -> RunReport:
        from skillweaver.skills.synthesize import Admission

        class _Gate:
            def admit(self, trajectory: Trajectory, environment: Any) -> Admission:
                environment()  # standing the candidate up is what attempts the reset
                return Admission(ok=False, skill=None, reason="the candidate was not proved")

        store = InMemorySkillStore()
        agent = Agent(
            controller=_Navigable(),  # type: ignore[arg-type]
            perceiver=scenario.perceiver,
            store=store,
            retriever=SkillRetriever(store),
            planner=None,  # type: ignore[arg-type]
            explorer=_Trajectorying(_navigated_trajectory(scenario)),  # type: ignore[arg-type]
            synthesis=lambda trajectory: _Gate(),  # type: ignore[arg-type,return-value]
            environment=world,
            llm=None,
        )
        return agent.run(scenario.task, warm=False, cold=True, learn=True)

    def test_a_refused_endpoint_is_named_as_one(self, scenario: Scenario) -> None:
        world = navigating_environment(_Navigable(), scenario.perceiver, restore=_refuses)
        note = self._report(scenario, world).learning_note
        assert "the candidate was not proved" in note
        assert "reset refused" in note and "403" in note

    def test_a_read_only_task_adds_nothing_about_a_reset_it_did_not_need(
        self, scenario: Scenario
    ) -> None:
        world = navigating_environment(_Navigable(), scenario.perceiver, read_only=True)
        note = self._report(scenario, world).learning_note
        assert note == "the candidate was not proved", "a reset that was not needed is not news"


class _Trajectorying:
    """An explorer that succeeds and hands back one prepared trajectory."""

    def __init__(self, trajectory: Trajectory) -> None:
        self._trajectory = trajectory

    def explore(self, task: TaskSpec, controller: Controller, budget: Budget | None) -> RunOutcome:
        return RunOutcome(
            ok=True,
            trajectory=self._trajectory,
            verdict=Verdict(ok=True),
            spend=Spend(),
            note="explored",
        )
