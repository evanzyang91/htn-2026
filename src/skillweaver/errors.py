"""Exception hierarchy for skillweaver.

Every error the project raises on purpose derives from :class:`SkillWeaverError`,
so callers at a boundary (CLI, eval harness, dashboard) can catch one type.
Raise the most specific subclass; put the human-readable cause in the message.
"""

from __future__ import annotations


class SkillWeaverError(Exception):
    """Base class for every deliberate skillweaver failure."""


class ControllerError(SkillWeaverError):
    """A controller could not capture the screen or is unusable (closed, crashed).

    Note that a single action failing is normally reported through
    ``ActionResult(ok=False, ...)`` rather than raised; this is for a broken controller.
    """


class PerceptionError(SkillWeaverError):
    """Detection, OCR, indexing or fingerprinting failed on a screenshot."""


class BudgetExceeded(SkillWeaverError):
    """A run exhausted its :class:`~skillweaver.contracts.Budget`.

    Raised by ``Spend.check()``. The message names the limit that was hit.
    """


class SandboxViolation(SkillWeaverError):
    """Skill code tried to reach something outside its ``SkillContext`` surface."""


class ExpectationFailed(SkillWeaverError):
    """A ``ctx.expect(condition, why)`` in skill code was false; the message is ``why``."""


class AdmissionRejected(SkillWeaverError):
    """A synthesized skill failed the checks required to enter the skill library."""


class SkillNotFound(SkillWeaverError):
    """No skill with the requested name, domain and version exists."""


class RouteNotFound(SkillWeaverError):
    """No path between two UI states is known to the site graph.

    ``SiteGraph.route`` returns ``None`` instead of raising; this is for callers
    that require a route and want to fail loudly.
    """


class ProviderError(SkillWeaverError):
    """An LLM provider call failed (network, auth, rate limit, malformed reply)."""


class ConfigError(SkillWeaverError):
    """Settings are missing or invalid (bad value in the environment or ``.env``)."""
