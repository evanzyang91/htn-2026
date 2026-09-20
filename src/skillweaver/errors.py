"""Exception hierarchy. Every deliberate failure derives from ``SkillWeaverError``
so a boundary (CLI, eval harness, dashboard) can catch one type."""

from __future__ import annotations


class SkillWeaverError(Exception):
    """Base class for every deliberate skillweaver failure."""


class ControllerError(SkillWeaverError):
    """The controller itself is broken; a single failed action is an ``ActionResult(ok=False)``."""


class PerceptionError(SkillWeaverError):
    """Detection, OCR, indexing or fingerprinting failed on a screenshot."""


class BudgetExceeded(SkillWeaverError):
    """A run exhausted its ``Budget``; the message names the limit hit."""


class SandboxViolation(SkillWeaverError):
    """Skill code tried to reach something outside its ``SkillContext`` surface."""


class ExpectationFailed(SkillWeaverError):
    """A ``ctx.expect(condition, why)`` in skill code was false; the message is ``why``."""


class AdmissionRejected(SkillWeaverError):
    """A synthesized skill failed the checks required to enter the skill library."""


class SkillNotFound(SkillWeaverError):
    """No skill with the requested name, domain and version exists."""


class RouteNotFound(SkillWeaverError):
    """No path between two UI states is known; ``SiteGraph.route`` returns ``None`` instead."""


class ProviderError(SkillWeaverError):
    """An LLM provider call failed (network, auth, rate limit, malformed reply)."""


class ConfigError(SkillWeaverError):
    """Settings are missing or invalid (bad value in the environment or ``.env``)."""
