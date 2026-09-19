"""The contracts hold: fakes satisfy their Protocols, values are immutable, the
geometry and similarity maths match hand-computed numbers, actions serialize."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import typing
from datetime import UTC, datetime

import pytest

from skillweaver import contracts as c
from skillweaver import errors
from skillweaver.config import Settings, load_settings
from skillweaver.logging_ import format_event, get_logger
from tests.fakes import (
    FakeController,
    FakeCritic,
    FakeDetector,
    FakeEmbedder,
    FakeFingerprinter,
    FakeGroundTruth,
    FakeLLM,
    FakePerceiver,
    FakeTextReader,
    InMemorySiteGraph,
    InMemorySkillStore,
    InMemoryTrajectoryRecorder,
    InMemoryTrajectoryStore,
    Scenario,
    ScriptExhausted,
    SimpleElementIndex,
    render_png,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

# -- fakes satisfy the Protocols they claim ---------------------------------------------


def test_every_fake_satisfies_its_protocol(scenario: Scenario) -> None:
    ctl = scenario.controller
    pairs: list[tuple[object, type]] = [
        (ctl, c.Controller),
        (FakeGroundTruth(ctl), c.GroundTruthSource),
        (FakeDetector(ctl), c.Detector),
        (FakeTextReader(ctl), c.TextReader),
        (FakeFingerprinter(), c.Fingerprinter),
        (SimpleElementIndex([]), c.ElementIndex),
        (scenario.perceiver, c.Perceiver),
        (FakeLLM(), c.LLMClient),
        (FakeCritic(), c.Critic),
        (FakeEmbedder(), c.Embedder),
        (InMemorySkillStore(), c.SkillStore),
        (InMemorySiteGraph(), c.SiteGraph),
        (InMemorySiteGraph(), c.GraphView),
        (InMemoryTrajectoryRecorder(), c.TrajectoryRecorder),
        (InMemoryTrajectoryStore(), c.TrajectoryStore),
    ]
    for fake, protocol in pairs:
        assert isinstance(fake, protocol), f"{type(fake).__name__} is not a {protocol.__name__}"


def test_fake_method_signatures_match_the_protocols(scenario: Scenario) -> None:
    """isinstance() only checks method names; this checks the parameter names too."""
    pairs: list[tuple[type, type]] = [
        (FakeController, c.Controller),
        (FakeGroundTruth, c.GroundTruthSource),
        (FakeDetector, c.Detector),
        (FakeTextReader, c.TextReader),
        (FakeFingerprinter, c.Fingerprinter),
        (SimpleElementIndex, c.ElementIndex),
        (FakePerceiver, c.Perceiver),
        (FakeLLM, c.LLMClient),
        (FakeCritic, c.Critic),
        (FakeEmbedder, c.Embedder),
        (InMemorySkillStore, c.SkillStore),
        (InMemorySiteGraph, c.SiteGraph),
        (InMemoryTrajectoryRecorder, c.TrajectoryRecorder),
        (InMemoryTrajectoryStore, c.TrajectoryStore),
    ]
    for fake, protocol in pairs:
        for name in sorted(protocol.__protocol_attrs__):  # type: ignore[attr-defined]
            want = inspect.signature(getattr(protocol, name))
            got = inspect.signature(getattr(fake, name))
            assert list(got.parameters) == list(want.parameters), f"{fake.__name__}.{name}"
            for pname, param in want.parameters.items():
                assert got.parameters[pname].default == param.default, (
                    f"{fake.__name__}.{name}({pname}) default differs from the protocol"
                )


def test_an_object_missing_a_method_is_rejected() -> None:
    class HalfController:
        def capture(self) -> None: ...

    assert not isinstance(HalfController(), c.Controller)


# -- immutability ------------------------------------------------------------------------


def _samples(scenario: Scenario) -> dict[type, object]:
    obs = scenario.perceiver.observe(scenario.controller)
    fp = c.Fingerprint("abc", {"url": "u"})
    prov = c.Provenance("run-1", "task", "fake-llm", NOW)
    skill = c.Skill("s", "d", "sum", "doc", {}, "def run(ctx): ...", (), None, None, prov)
    click = c.Click(c.Point(1, 2))
    result = c.ActionResult(True)
    verdict = c.Verdict(True)
    step = c.TrajectoryStep(0, click, obs, obs, result)
    trajectory = c.Trajectory("run-1", "task", "d", (step,), True, NOW, NOW)
    edge = c.Transition(fp, fp, (click,))
    values: list[object] = [
        c.Point(1, 2),
        c.Box(0, 0, 1, 1),
        obs.screenshot,
        obs.elements[0],
        fp,
        obs,
        click,
        c.Move(c.Point(1, 2)),
        c.Drag(c.Point(1, 2), c.Point(3, 4)),
        c.TypeText("hi"),
        c.PressKey(("Enter",)),
        c.Scroll(c.Point(1, 2), 0, 10),
        c.Wait(5),
        c.Navigate("https://fake.test"),
        result,
        prov,
        c.SkillStats(),
        skill,
        c.Candidate(skill, 0.5),
        c.UIState(fp, "d"),
        edge,
        c.Route((click,), 1.0, (edge,)),
        c.Usage(),
        c.ToolSpec("t", "desc", {}),
        c.ToolCall("t", {}),
        c.LLMMessage("user", "hi"),
        c.LLMResponse("hi"),
        verdict,
        c.Budget(),
        step,
        trajectory,
        c.TaskSpec("task", "d"),
        c.RunOutcome(True, trajectory, verdict, c.Spend()),
        c.SkillResult(True),
        c.SkillCall("s", "d"),
        c.Plan((click,)),
    ]
    return {type(v): v for v in values}


def _contract_dataclasses() -> set[type]:
    return {
        obj
        for _, obj in inspect.getmembers(c, inspect.isclass)
        if obj.__module__ == c.__name__ and dataclasses.is_dataclass(obj)
    }


def test_every_value_type_is_frozen_with_slots_except_spend() -> None:
    for cls in _contract_dataclasses() - {c.Spend}:
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"
        assert "__slots__" in vars(cls), f"{cls.__name__} must use slots"
    assert not c.Spend.__dataclass_params__.frozen


def test_every_frozen_dataclass_rejects_mutation(scenario: Scenario) -> None:
    samples = _samples(scenario)
    assert set(samples) == _contract_dataclasses() - {c.Spend}, "add a sample for the new type"
    for cls, value in samples.items():
        first = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, first, getattr(value, first))
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            value.brand_new_attribute = 1  # type: ignore[attr-defined]
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(value, first)


# -- geometry ----------------------------------------------------------------------------


def test_box_center_area_contains() -> None:
    box = c.Box(10, 20, 30, 41)
    assert box.center == c.Point(25, 40)
    assert box.area == 1230
    assert box.contains(c.Point(10, 20))
    assert box.contains(c.Point(39, 60))
    assert not box.contains(c.Point(40, 60))
    assert not box.contains(c.Point(39, 61))


def test_box_iou_matches_hand_computed_values() -> None:
    a = c.Box(0, 0, 10, 10)
    # identical
    assert a.iou(a) == 1.0
    # overlap 5x5=25; union 100+100-25=175
    assert a.iou(c.Box(5, 5, 10, 10)) == pytest.approx(25 / 175)
    # contained 5x5 inside 10x10: 25 / 100
    assert a.iou(c.Box(2, 2, 5, 5)) == pytest.approx(0.25)
    # touching edges share no area
    assert a.iou(c.Box(10, 0, 10, 10)) == 0.0
    # disjoint
    assert a.iou(c.Box(50, 50, 5, 5)) == 0.0
    # offset origin: intersection x 4..8 (4), y 6..9 (3) = 12; union 6*6+4*3-12 = 36
    assert c.Box(2, 3, 6, 6).iou(c.Box(4, 6, 4, 3)) == pytest.approx(12 / 36)
    # symmetric, and degenerate boxes never divide by zero
    assert c.Box(5, 5, 10, 10).iou(a) == pytest.approx(a.iou(c.Box(5, 5, 10, 10)))
    assert c.Box(0, 0, 0, 0).iou(c.Box(0, 0, 0, 0)) == 0.0


def test_fingerprint_similarity_matches_hand_computed_values() -> None:
    a = c.Fingerprint("A", {"url": "1", "layout": "2", "text": "3"})
    # equal value -> 1.0 regardless of parts
    assert a.similarity(c.Fingerprint("A")) == 1.0
    # union of names {url, layout, text, extra} = 4; only url agrees -> 1/4
    b = c.Fingerprint("B", {"url": "1", "layout": "9", "extra": "4"})
    assert a.similarity(b) == pytest.approx(0.25)
    assert b.similarity(a) == pytest.approx(0.25)
    # two of three agree -> 2/3
    assert a.similarity(c.Fingerprint("C", {"url": "1", "layout": "2", "text": "x"})) == (
        pytest.approx(2 / 3)
    )
    # nothing to compare -> 0.0
    assert c.Fingerprint("X").similarity(c.Fingerprint("Y")) == 0.0
    assert a.similarity(c.Fingerprint("Y")) == 0.0


def test_fingerprint_identity_is_its_value() -> None:
    assert c.Fingerprint("A", {"k": "1"}) == c.Fingerprint("A", {"k": "2"})
    assert c.Fingerprint("A") != c.Fingerprint("B")
    assert len({c.Fingerprint("A", {"k": "1"}), c.Fingerprint("A")}) == 1


def test_screenshot_to_array_is_logical_by_default() -> None:
    png = render_png([], width=40, height=30, scale=2.0)
    shot = c.Screenshot(png, width=40, height=30, scale=2.0, captured_at=NOW)
    assert shot.to_array().shape == (30, 40, 3)
    assert shot.to_array(logical=False).shape == (60, 80, 3)
    assert shot.to_array().dtype.name == "uint8"
    with pytest.raises(errors.PerceptionError):
        c.Screenshot(b"not a png", 1, 1, 1.0, NOW).to_array()


# -- actions -----------------------------------------------------------------------------

ALL_ACTIONS: list[c.Action] = [
    c.Click(c.Point(10, 20)),
    c.Click(c.Point(10, 20), button="right", clicks=2),
    c.Move(c.Point(1, 2)),
    c.Drag(c.Point(1, 2), c.Point(30, 40)),
    c.TypeText('hello "world"\n'),
    c.PressKey(("Meta", "a")),
    c.Scroll(c.Point(5, 5), dx=-3, dy=120),
    c.Wait(250),
    c.Navigate("https://fake.test/invoices?q=acme"),
]


@pytest.mark.parametrize("action", ALL_ACTIONS, ids=lambda a: a.kind)
def test_every_action_round_trips_through_json(action: c.Action) -> None:
    wire = json.dumps(c.action_to_dict(action))
    restored = c.action_from_dict(json.loads(wire))
    assert restored == action
    assert type(restored) is type(action)
    assert json.loads(wire)["kind"] == action.kind


def test_action_union_is_closed_and_fully_registered() -> None:
    members = set(typing.get_args(c.Action))
    assert {type(a) for a in ALL_ACTIONS} == members
    assert set(c.ACTION_TYPES.values()) == members
    assert set(c.ACTION_TYPES) == set(typing.get_args(c.ActionKind))


def test_action_from_dict_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown action kind"):
        c.action_from_dict({"kind": "teleport"})
    with pytest.raises(ValueError, match="bad fields"):
        c.action_from_dict({"kind": "wait", "seconds": 1})


# -- accounting --------------------------------------------------------------------------


def test_usage_adds() -> None:
    total = c.Usage(10, 5, 1, 0.01) + c.Usage(1, 2, 3, 0.5)
    assert total == c.Usage(11, 7, 4, pytest.approx(0.51))
    assert sum([c.Usage(1, 1, 1, 1.0)] * 3, c.Usage()) == c.Usage(3, 3, 3, 3.0)


def test_spend_check_raises_when_a_limit_is_reached() -> None:
    spend = c.Spend(c.Budget(max_steps=2, max_seconds=10, max_usd=1.0, max_llm_calls=3))
    spend.check()
    spend.add_step()
    spend.check()
    spend.add_step()
    with pytest.raises(errors.BudgetExceeded, match="max_steps"):
        spend.check()

    spend = c.Spend(c.Budget(max_usd=1.0, max_llm_calls=3))
    spend.add_usage(c.Usage(calls=2, cost_usd=0.4))
    spend.check()
    spend.add_usage(c.Usage(calls=1, cost_usd=0.1))
    with pytest.raises(errors.BudgetExceeded, match="max_llm_calls"):
        spend.check()

    spend = c.Spend(c.Budget(max_seconds=5), seconds=5.0)
    with pytest.raises(errors.BudgetExceeded, match="max_seconds"):
        spend.check()
    assert c.Spend().start().elapsed_seconds() >= 0.0


def test_error_hierarchy() -> None:
    for name in (
        "ControllerError PerceptionError BudgetExceeded SandboxViolation ExpectationFailed "
        "AdmissionRejected SkillNotFound RouteNotFound ProviderError ConfigError"
    ).split():
        assert issubclass(getattr(errors, name), errors.SkillWeaverError)


# -- fakes behave as documented ----------------------------------------------------------


def test_fake_llm_counts_calls_exactly_and_fails_loudly_when_exhausted() -> None:
    llm = FakeLLM(["one", c.LLMResponse("two", usage=c.Usage(7, 3, 1, 0.02))])
    assert llm.calls == 0
    assert llm.complete([c.LLMMessage("user", "hi")], system="sys").text == "one"
    assert llm.complete([c.LLMMessage("user", "again")]).text == "two"
    assert llm.calls == 2 and llm.remaining == 0
    assert llm.requests[0].system == "sys"
    assert llm.total_usage() == c.Usage(7, 3, 2, 0.02)
    with pytest.raises(ScriptExhausted, match="call #3"):
        llm.complete([])
    assert llm.calls == 3
    assert not isinstance(ScriptExhausted(), errors.SkillWeaverError)


def test_fake_critic_modes(scenario: Scenario) -> None:
    obs = scenario.perceiver.observe(scenario.controller)
    scripted = FakeCritic([c.Verdict(False, "nope", source="model")])
    assert scripted.judge("g", obs, obs).reason == "nope"
    with pytest.raises(ScriptExhausted):
        scripted.judge("g", obs, obs)
    assert FakeCritic(goal=obs.fingerprint).judge("g", obs, obs).ok
    assert not FakeCritic(goal=c.Fingerprint("elsewhere")).judge("g", obs, obs).ok


def test_fake_embedder_is_deterministic_normalized_and_word_sensitive() -> None:
    e = FakeEmbedder()
    a, b, far = e.embed(["search the invoice list", "search invoice", "open the settings menu"])
    assert e.embed(["search the invoice list"])[0] == a
    assert sum(v * v for v in a) == pytest.approx(1.0)

    def dot(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert dot(a, b) > dot(a, far)
    assert e.embed([]) == [] and len(e.embed([""])[0]) == e.dim


def test_skill_store_versions_stats_and_demotion(
    skill_store: InMemorySkillStore, sample_skill: c.Skill
) -> None:
    v1 = skill_store.put(sample_skill)
    v2 = skill_store.put(dataclasses.replace(sample_skill, summary="better"))
    assert (v1.version, v2.version) == (1, 2)
    assert skill_store.get("search_invoice", "fake.test").summary == "better"
    assert skill_store.get("search_invoice", "fake.test", version=1) == v1
    with pytest.raises(errors.SkillNotFound):
        skill_store.get("nope", "fake.test")
    with pytest.raises(errors.SkillNotFound):
        skill_store.get("search_invoice", "fake.test", version=9)

    skill_store.record_run("search_invoice", "fake.test", ok=True, ms=100)
    skill_store.record_run("search_invoice", "fake.test", ok=False, ms=9000)
    stats = skill_store.record_run("search_invoice", "fake.test", ok=True, ms=300).stats
    assert (stats.runs, stats.successes, stats.mean_ms) == (3, 2, pytest.approx(200.0))

    assert [s.name for s in skill_store.list("fake.test")] == ["search_invoice"]
    skill_store.demote("search_invoice", "fake.test", "stopped working")
    assert skill_store.list() == []
    assert skill_store.list(include_demoted=True)[0].demoted_reason == "stopped working"


def test_site_graph_routes_by_expected_cost(site_graph: InMemorySiteGraph) -> None:
    a, b, d = c.Fingerprint("a"), c.Fingerprint("b"), c.Fingerprint("d")
    site_graph.upsert_state(c.UIState(a, "fake.test", label="A"))
    slow, hop1, hop2 = c.Wait(1), c.Wait(2), c.Wait(3)
    site_graph.observe_transition(a, [slow], d, ok=True, ms=1000)
    site_graph.observe_transition(a, [hop1], b, ok=True, ms=100)
    site_graph.observe_transition(b, [hop2], d, ok=True, ms=100)
    route = site_graph.route(a, d)
    assert route is not None and route.steps == (hop1, hop2) and route.cost == pytest.approx(200)
    # halving the first hop's success rate doubles its expected cost: 200 + 100 = 300
    site_graph.observe_transition(a, [hop1], b, ok=False, ms=50)
    assert site_graph.route(a, d).cost == pytest.approx(300)  # type: ignore[union-attr]
    assert site_graph.route(d, a) is None
    assert site_graph.route(a, a) == c.Route((), 0.0, ())
    assert site_graph.route(c.Fingerprint("unknown"), a) is None
    assert [t.dst for t in site_graph.neighbors(a)] == [d, b]
    assert {s.fingerprint for s in site_graph.states("fake.test")} == {a, b, d}


def test_controller_refuses_what_it_should(fake_controller: FakeController) -> None:
    assert not fake_controller.supports("navigate")
    assert not fake_controller.perform(c.Navigate("https://fake.test")).ok
    outside = fake_controller.perform(c.Click(c.Point(5000, 5)))
    assert not outside.ok and "outside the viewport" in (outside.error or "")
    fake_controller.fail_next("boom")
    assert fake_controller.perform(c.Wait(1)).error == "boom"
    assert fake_controller.perform(c.Wait(1)).ok
    assert fake_controller.state == "list" and len(fake_controller.actions) == 4
    fake_controller.close()
    fake_controller.close()
    with pytest.raises(errors.ControllerError):
        fake_controller.capture()


def test_element_index_return_contract(scenario: Scenario) -> None:
    index = scenario.perceiver.observe(scenario.controller).index
    assert [e.text for e in index.find_text("globex")] == ["Globex INV-2041 $310.00"]
    assert index.find_text("Globx", fuzzy=True)[0].text.startswith("Globex")
    assert index.find_text("Globx", fuzzy=False) == []
    assert index.find_text("zzzz") == [] and index.best("zzzz") == []
    assert index.best("search field")[0].kind == c.ElementKind.text_field
    assert index.nearest(c.Point(790, 70))[0].text == "Archive"
    assert index.containing(c.Point(1, 1)) == []
    assert [e.kind for e in index.by_kind(c.ElementKind.link)] == [c.ElementKind.link]
    ys = [e.box.y for e in index.all()]
    assert ys == sorted(ys)


# -- config and logging ------------------------------------------------------------------


def test_settings_defaults_env_and_dotenv(tmp_path: typing.Any) -> None:
    default = load_settings(env={}, env_file=None)
    assert default == Settings()
    assert default.default_target == "browser" and default.anthropic_api_key is None
    assert default.skills_dir.as_posix() == "data/skills"
    assert {p.name for p in (default.graphs_dir, default.trajectories_dir)} == {
        "graphs",
        "trajectories",
    }
    assert {default.models_dir.name, default.eval_dir.name} == {"models", "eval"}

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        '# comment\nexport ANTHROPIC_API_KEY="sk-from-file"\nSKILLWEAVER_MAX_STEPS=7\n'
    )
    s = load_settings(
        env={"SKILLWEAVER_MAX_STEPS": "9", "SKILLWEAVER_TARGET": "desktop"}, env_file=dotenv
    )
    assert s.anthropic_api_key == "sk-from-file"
    assert s.default_budget.max_steps == 9, "the environment wins over .env"
    assert s.default_target == "desktop"
    assert "sk-from-file" not in repr(s)

    for bad in (
        {"SKILLWEAVER_MAX_STEPS": "many"},
        {"SKILLWEAVER_TARGET": "phone"},
        {"SKILLWEAVER_MAX_USD": "-1"},
        {"SKILLWEAVER_LOG_LEVEL": "LOUD"},
    ):
        with pytest.raises(errors.ConfigError):
            load_settings(env=bad, env_file=None)


def test_logging_is_structured_and_leaves_the_root_logger_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert format_event("skill.run", {"name": "a b", "ok": True}) == 'skill.run name="a b" ok=True'
    root_handlers = list(logging.getLogger().handlers)
    log = get_logger("tests.contracts")
    assert log.stdlib.name == "skillweaver.tests.contracts"
    assert list(logging.getLogger().handlers) == root_handlers
    with caplog.at_level(logging.INFO, logger="skillweaver"):
        log.info("graph.route", src="a", cost=1.5)
    assert "graph.route src=a cost=1.5" in caplog.text
