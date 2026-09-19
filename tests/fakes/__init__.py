"""Test doubles for every ``skillweaver.contracts`` Protocol.

Import from here (``from tests.fakes import FakeController, FakeLLM``) or use the
pytest fixtures of the same purpose declared in ``tests/conftest.py``. With these you
can test any part of the project without a browser, a model or a network.
"""

from tests.fakes.controller import (
    ActionMatcher,
    FakeController,
    FakeGroundTruth,
    FakeState,
    any_action,
    clicks,
    kind_is,
    navigates,
    presses,
    render_png,
    types,
)
from tests.fakes.llm import (
    CriticCall,
    FakeCritic,
    FakeEmbedder,
    FakeLLM,
    LLMRequest,
    ScriptExhausted,
)
from tests.fakes.memory import (
    InMemorySiteGraph,
    InMemorySkillStore,
    InMemoryTrajectoryRecorder,
    InMemoryTrajectoryStore,
)
from tests.fakes.perception import (
    FakeDetector,
    FakeFingerprinter,
    FakePerceiver,
    FakeTextReader,
    SimpleElementIndex,
)
from tests.fakes.scenario import Scenario, make_controller, make_scenario

__all__ = [
    "ActionMatcher",
    "CriticCall",
    "FakeController",
    "FakeCritic",
    "FakeDetector",
    "FakeEmbedder",
    "FakeFingerprinter",
    "FakeGroundTruth",
    "FakeLLM",
    "FakePerceiver",
    "FakeState",
    "FakeTextReader",
    "InMemorySiteGraph",
    "InMemorySkillStore",
    "InMemoryTrajectoryRecorder",
    "InMemoryTrajectoryStore",
    "LLMRequest",
    "Scenario",
    "ScriptExhausted",
    "SimpleElementIndex",
    "any_action",
    "clicks",
    "kind_is",
    "make_controller",
    "make_scenario",
    "navigates",
    "presses",
    "render_png",
    "types",
]
