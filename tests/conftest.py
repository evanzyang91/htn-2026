"""Shared fixtures: every fake in ``tests/fakes`` is available by name.

All fixtures are function-scoped, so each test gets fresh, independent state. The
``fake_controller`` / ``fake_detector`` / ``fake_text_reader`` / ``fake_perceiver`` /
``fake_ground_truth`` fixtures all belong to the SAME ``scenario`` instance, so they
can be mixed freely within one test.
"""

from __future__ import annotations

import pytest

from skillweaver.contracts import Skill
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
    make_scenario,
)


@pytest.fixture
def scenario() -> Scenario:
    """The four-state fake invoicing app (see ``tests/fakes/scenario.py``)."""
    return make_scenario()


@pytest.fixture
def fake_controller(scenario: Scenario) -> FakeController:
    """The scenario's controller, in its ``list`` start state."""
    return scenario.controller


@pytest.fixture
def fake_perceiver(scenario: Scenario) -> FakePerceiver:
    """The scenario's perceiver, wired to ``fake_controller``."""
    return scenario.perceiver


@pytest.fixture
def fake_detector(fake_perceiver: FakePerceiver) -> FakeDetector:
    return fake_perceiver.detector  # type: ignore[return-value]


@pytest.fixture
def fake_text_reader(fake_perceiver: FakePerceiver) -> FakeTextReader:
    return fake_perceiver.reader  # type: ignore[return-value]


@pytest.fixture
def fake_fingerprinter(fake_perceiver: FakePerceiver) -> FakeFingerprinter:
    return fake_perceiver.fingerprinter  # type: ignore[return-value]


@pytest.fixture
def fake_ground_truth(fake_controller: FakeController) -> FakeGroundTruth:
    """Offline teacher over ``fake_controller``; never give it to an agent."""
    return FakeGroundTruth(fake_controller)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A model with an EMPTY script: any call fails, and ``fake_llm.calls == 0``
    proves none was made. Build ``FakeLLM([...])`` yourself to script replies."""
    return FakeLLM()


@pytest.fixture
def fake_critic() -> FakeCritic:
    """A critic with an empty script. Build ``FakeCritic([...])`` to script verdicts."""
    return FakeCritic()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def skill_store() -> InMemorySkillStore:
    return InMemorySkillStore()


@pytest.fixture
def site_graph() -> InMemorySiteGraph:
    return InMemorySiteGraph()


@pytest.fixture
def trajectory_recorder() -> InMemoryTrajectoryRecorder:
    return InMemoryTrajectoryRecorder()


@pytest.fixture
def trajectory_store() -> InMemoryTrajectoryStore:
    return InMemoryTrajectoryStore()


@pytest.fixture
def sample_skill() -> Skill:
    """A minimal valid, not-yet-stored skill (``version=0``) for store and retrieval
    tests."""
    from datetime import UTC, datetime

    from skillweaver.contracts import Provenance

    return Skill(
        name="search_invoice",
        domain="fake.test",
        summary="Search the invoice list for a company.",
        docstring="Types the company name into the search field and waits for results.",
        params={"company": {"type": "string"}},
        code="def run(ctx, company):\n    ctx.ctl.type_text(company)\n",
        requires=(),
        precondition=None,
        verifier_code=None,
        provenance=Provenance(
            "run-0", "find an invoice", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC)
        ),
    )
