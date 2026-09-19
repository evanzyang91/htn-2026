"""Runtime settings, read from the process environment and an optional ``.env`` file.

The process environment wins over ``.env``; both win over the defaults below. No
secret has a default, and ``.env`` is git-ignored. Use :func:`settings` everywhere;
use :func:`load_settings` in tests that need a specific environment.

Recognized variables::

    SKILLWEAVER_DATA_DIR        data directory                  (default: data)
    SKILLWEAVER_TARGET          "browser" or "desktop"          (default: browser)
    SKILLWEAVER_LOG_LEVEL       DEBUG/INFO/WARNING/ERROR        (default: INFO)
    SKILLWEAVER_CLAUDE_MODEL    Claude model id                 (default: claude-opus-5)
    SKILLWEAVER_GEMINI_MODEL    Gemini computer-use model id
    SKILLWEAVER_MAX_STEPS       default Budget.max_steps        (default: 40)
    SKILLWEAVER_MAX_SECONDS     default Budget.max_seconds      (default: 300)
    SKILLWEAVER_MAX_USD         default Budget.max_usd          (default: 2.0)
    SKILLWEAVER_MAX_LLM_CALLS   default Budget.max_llm_calls    (default: 60)
    SKILLWEAVER_SKILL_MAX_SECONDS
                                default SkillLimits.max_seconds (default: 45)
    ANTHROPIC_API_KEY           Claude credentials (optional: the SDK also resolves
                                its own credentials when this is unset)
    GEMINI_API_KEY              Gemini credentials (GOOGLE_API_KEY also accepted)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from skillweaver.contracts import Budget
from skillweaver.errors import ConfigError

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-computer-use-preview-10-2025"

DEFAULT_SKILL_MAX_SECONDS = 45.0
"""Wall-clock seconds one stored skill may spend running its OWN code.

Not a guess. A skill is a short procedure and the code between two observations is
milliseconds of it, so this number is a runaway-loop tripwire rather than a work
allowance - see :class:`~skillweaver.skills.api.SkillLimits`, which does not charge a
skill for the time it sits blocked on a screenshot or on OCR. Forty-five seconds is
long enough that no honest procedure on a real, slow page reaches it and short enough
that ``while True`` is a pause rather than a hang."""

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TARGETS = ("browser", "desktop")


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration. API keys are excluded from ``repr`` so they cannot
    leak into logs."""

    data_dir: Path = Path("data")
    default_target: Literal["browser", "desktop"] = "browser"
    log_level: str = "INFO"
    claude_model: str = DEFAULT_CLAUDE_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    skill_max_seconds: float = DEFAULT_SKILL_MAX_SECONDS
    anthropic_api_key: str | None = field(default=None, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)
    default_budget: Budget = field(default_factory=Budget)

    @property
    def skills_dir(self) -> Path:
        """Where the skill library is stored: ``<data_dir>/skills``."""
        return self.data_dir / "skills"

    @property
    def graphs_dir(self) -> Path:
        """Where per-domain site graphs are stored: ``<data_dir>/graphs``."""
        return self.data_dir / "graphs"

    @property
    def trajectories_dir(self) -> Path:
        """Where recorded runs are stored: ``<data_dir>/trajectories``."""
        return self.data_dir / "trajectories"

    @property
    def models_dir(self) -> Path:
        """Where detector weights and datasets live: ``<data_dir>/models``."""
        return self.data_dir / "models"

    @property
    def eval_dir(self) -> Path:
        """Where evaluation results are written: ``<data_dir>/eval``."""
        return self.data_dir / "eval"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file: ``KEY=value`` lines, ``#`` comments, optional
    ``export`` prefix and optional matching quotes. A missing file gives ``{}``."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _number[T: (int, float)](env: Mapping[str, str], key: str, default: T, cast: type[T]) -> T:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid {cast.__name__}") from exc
    if value <= 0:
        raise ConfigError(f"{key}={raw!r} must be greater than zero")
    return value


def load_settings(
    env: Mapping[str, str] | None = None, env_file: Path | str | None = Path(".env")
) -> Settings:
    """Build :class:`Settings` without caching.

    Args:
        env: The environment to read; ``None`` means ``os.environ``.
        env_file: A ``.env`` path whose values fill in anything ``env`` lacks;
            ``None`` disables it. A missing file is ignored.

    Raises:
        ConfigError: on a malformed number, an unknown target or log level.
    """
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(parse_env_file(Path(env_file)))
    merged.update(os.environ if env is None else env)

    target = merged.get("SKILLWEAVER_TARGET") or "browser"
    if target not in _TARGETS:
        raise ConfigError(f"SKILLWEAVER_TARGET={target!r} must be one of {_TARGETS}")
    log_level = (merged.get("SKILLWEAVER_LOG_LEVEL") or "INFO").upper()
    if log_level not in _LOG_LEVELS:
        raise ConfigError(f"SKILLWEAVER_LOG_LEVEL={log_level!r} must be one of {_LOG_LEVELS}")

    defaults = Budget()
    budget = Budget(
        max_steps=_number(merged, "SKILLWEAVER_MAX_STEPS", defaults.max_steps, int),
        max_seconds=_number(merged, "SKILLWEAVER_MAX_SECONDS", defaults.max_seconds, float),
        max_usd=_number(merged, "SKILLWEAVER_MAX_USD", defaults.max_usd, float),
        max_llm_calls=_number(merged, "SKILLWEAVER_MAX_LLM_CALLS", defaults.max_llm_calls, int),
    )
    return Settings(
        data_dir=Path(merged.get("SKILLWEAVER_DATA_DIR") or "data"),
        default_target=target,  # type: ignore[arg-type]
        log_level=log_level,
        claude_model=merged.get("SKILLWEAVER_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
        gemini_model=merged.get("SKILLWEAVER_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        skill_max_seconds=_number(
            merged, "SKILLWEAVER_SKILL_MAX_SECONDS", DEFAULT_SKILL_MAX_SECONDS, float
        ),
        anthropic_api_key=merged.get("ANTHROPIC_API_KEY") or None,
        gemini_api_key=merged.get("GEMINI_API_KEY") or merged.get("GOOGLE_API_KEY") or None,
        default_budget=budget,
    )


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process-wide settings, loaded once from ``os.environ`` and ``./.env``.
    Tests that change the environment call ``settings.cache_clear()`` afterwards.

    Raises:
        ConfigError: if the environment holds an invalid value.
    """
    return load_settings()
