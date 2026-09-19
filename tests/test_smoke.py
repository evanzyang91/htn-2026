"""The interfaces compose: drive the fake app from start to done through nothing but
contract types, the way a real agent loop will."""

from __future__ import annotations

from skillweaver.contracts import (
    Click,
    Controller,
    ElementKind,
    Observation,
    Perceiver,
    Spend,
    TypeText,
)
from tests.fakes import (
    FakeCritic,
    FakeGroundTruth,
    FakeLLM,
    InMemorySiteGraph,
    InMemoryTrajectoryRecorder,
    InMemoryTrajectoryStore,
    Scenario,
)


def test_drive_the_scenario_from_start_to_done(scenario: Scenario, fake_llm: FakeLLM) -> None:
    # Typed as the Protocols on purpose: only the contract surface is used below.
    controller: Controller = scenario.controller
    perceiver: Perceiver = scenario.perceiver
    graph = InMemorySiteGraph()
    recorder = InMemoryTrajectoryRecorder()
    store = InMemoryTrajectoryStore()
    spend = Spend()

    run_id = recorder.start(scenario.task.text, scenario.task.domain)
    seen: list[Observation] = [perceiver.observe(controller)]
    assert seen[0].url == "https://fake.test/invoices"
    assert seen[0].screenshot.to_array().shape == (600, 800, 3)

    def act(action: Click | TypeText) -> Observation:
        spend.check()
        before = seen[-1]
        result = controller.perform(action)
        after = perceiver.observe(controller)
        assert result.ok
        spend.add_step()
        recorder.step(action, before, after, result)
        graph.observe_transition(
            before.fingerprint, [action], after.fingerprint, ok=True, ms=result.elapsed_ms
        )
        seen.append(after)
        return after

    # 1. Acme is not on the unfiltered list, so search for it.
    assert seen[0].index.find_text("Acme") == []
    searched = act(TypeText("acme"))

    # 2. Find the row by what it says, click where it is.
    rows = searched.index.find_text("Acme Corp", kind=ElementKind.row)
    assert len(rows) == 1
    selected = act(Click(rows[0].box.center))

    # 3. Confirm.
    confirm = selected.index.best("Confirm payment button")[0]
    assert confirm.kind == ElementKind.button
    done = act(Click(confirm.box.center))

    assert scenario.solved
    assert [e.text for e in done.elements] == ["Payment confirmed", "INV-1042 paid to Acme Corp"]

    # The fingerprint changed at every step and every state is distinct.
    fingerprints = [o.fingerprint for o in seen]
    assert all(a != b for a, b in zip(fingerprints, fingerprints[1:], strict=False))
    assert len(set(fingerprints)) == 4
    # Neighbouring screens that share a layout are more alike than unrelated ones.
    assert fingerprints[0].similarity(fingerprints[1]) < 1.0
    # Observing the same screen twice gives the same fingerprint.
    assert perceiver.observe(controller).fingerprint == fingerprints[-1]

    # The run was recorded, stored, and its actions are exactly what was performed.
    trajectory = recorder.finish(ok=True, note="confirmed")
    store.save(trajectory)
    assert store.list() == [run_id]
    assert [s.index for s in store.load(run_id).steps] == [0, 1, 2]
    assert tuple(s.action for s in trajectory.steps) == scenario.solution
    assert scenario.controller.actions == list(scenario.solution)
    assert spend.steps == 3

    # The graph learned the path and can replay it without perception or a model.
    route = graph.route(fingerprints[0], fingerprints[-1])
    assert route is not None and route.steps == scenario.solution
    scenario.controller.reset()
    for step in route.steps:
        assert controller.perform(step).ok
    assert scenario.solved

    # A critic keyed on the goal fingerprint agrees, and no model was ever consulted.
    assert FakeCritic(goal=fingerprints[-1]).judge(scenario.task.text, seen[0], done).ok
    assert fake_llm.calls == 0


def test_the_dead_end_is_really_dead(scenario: Scenario) -> None:
    controller, perceiver = scenario.controller, scenario.perceiver
    start = perceiver.observe(controller)
    for action in scenario.trap:
        assert controller.perform(action).ok
    assert scenario.stuck

    trapped = perceiver.observe(controller)
    assert trapped.fingerprint != start.fingerprint
    # Nothing on screen leads anywhere: click every element, type, and stay stuck.
    for element in trapped.elements:
        controller.perform(Click(element.box.center))
    controller.perform(TypeText("acme"))
    assert scenario.stuck and not scenario.solved
    assert perceiver.observe(controller).fingerprint == trapped.fingerprint

    controller.reset()
    assert perceiver.observe(controller).fingerprint == start.fingerprint


def test_wrong_moves_do_not_shortcut_the_path(scenario: Scenario) -> None:
    controller, perceiver = scenario.controller, scenario.perceiver
    start = perceiver.observe(controller)
    # Clicking the visible rows does nothing (one even sits where Acme's row will be).
    for row in start.index.by_kind(ElementKind.row):
        controller.perform(Click(row.box.center))
    assert controller.state == "list"
    # Back and Clear make real loops.
    for action in scenario.solution[:2]:
        controller.perform(action)
    back = perceiver.observe(controller).index.find_text("Back")[0]
    controller.perform(Click(back.box.center))
    assert controller.state == "searched"
    clear = perceiver.observe(controller).index.find_text("Clear")[0]
    controller.perform(Click(clear.box.center))
    assert controller.state == "list"


def test_ground_truth_agrees_with_perception_but_is_a_separate_object(
    scenario: Scenario,
) -> None:
    truth = FakeGroundTruth(scenario.controller)
    observed = scenario.perceiver.observe(scenario.controller)
    assert {e.box for e in truth.elements()} == {e.box for e in observed.elements}
    assert truth.url() == observed.url
    assert all(e.source == "dom" for e in truth.elements())
