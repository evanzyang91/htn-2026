"""Runtime settings, read from the process environment and an optional ``.env`` file.

The process environment wins over ``.env``; both win over the defaults below. No
secret has a default, and ``.env`` is git-ignored. Use :func:`settings` everywhere;
use :func:`load_settings` in tests that need a specific environment.

Recognized variables::

    SKILLWEAVER_DATA_DIR        data directory                  (default: data)
    SKILLWEAVER_TARGET          "browser" or "desktop"          (default: browser)
    SKILLWEAVER_HEADLESS        run the browser without a window (default: false)
    SKILLWEAVER_CHROME_PROFILE  persistent real-Chrome profile dir (default: unset)
    SKILLWEAVER_PERCEPTION      "pixels" or "dom"               (default: pixels)
    SKILLWEAVER_POLICY          "claude" or "jev"               (default: claude)
    SKILLWEAVER_LOG_LEVEL       DEBUG/INFO/WARNING/ERROR        (default: INFO)
    SKILLWEAVER_CLAUDE_MODEL    Claude model id                 (default: claude-opus-5)
    SKILLWEAVER_GEMINI_MODEL    Gemini computer-use model id
    SKILLWEAVER_MAX_STEPS       default Budget.max_steps        (default: 40)
    SKILLWEAVER_MAX_SECONDS     default Budget.max_seconds      (default: 300)
    SKILLWEAVER_MAX_USD         default Budget.max_usd          (default: 2.0)
    SKILLWEAVER_MAX_LLM_CALLS   default Budget.max_llm_calls    (default: 60)
    SKILLWEAVER_SKILL_MAX_SECONDS
                                default SkillLimits.max_seconds (default: 45)
    SKILLWEAVER_EMBEDDER        rank skills with the local embedding model when its
                                weights are present                (default: true)
    SKILLWEAVER_EMBEDDER_DIR    where those weights live
                                (default: <data_dir>/models/text_embedder)
    ANTHROPIC_API_KEY           Claude credentials (optional: the SDK also resolves
                                its own credentials when this is unset)
    GEMINI_API_KEY              Gemini credentials (GOOGLE_API_KEY also accepted)
    TYPESAFE_API_KEY            Jev policy credentials, read by skillweaver.llm.jev_
                                itself rather than through Settings, so no credential
                                is copied into a value this module puts in a repr
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
from skillweaver.perception_mode import PATHS, PIXELS

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-computer-use-preview-10-2025"

EMBEDDER_DIRNAME = "text_embedder"
"""The directory under ``models_dir`` that holds the retrieval embedding model."""

DEFAULT_SKILL_MAX_SECONDS = 45.0
"""Wall-clock seconds one stored skill may spend running its OWN code.

Not a guess. A skill is a short procedure and the code between two observations is
milliseconds of it, so this number is a runaway-loop tripwire rather than a work
allowance - see :class:`~skillweaver.skills.api.SkillLimits`, which does not charge a
skill for the time it sits blocked on a screenshot or on OCR. Forty-five seconds is
long enough that no honest procedure on a real, slow page reaches it and short enough
that ``while True`` is a pause rather than a hang."""

DEFAULT_EMBEDDER_ENABLED = False
"""Whether retrieval ranks with the local embedding model when its weights are there.

OFF, and that is a MEASURED decision rather than caution. Ranking by meaning does what
it was built to do - it finds a stored skill asked for in other words, lifting top-1
recall from 18/24 to 21/24 over the two libraries in this repository - and not one of
those extra hits turns into a warm run, because what stops them is downstream of the
ranking and is counted in words:
:func:`~skillweaver.agent.planner.bind_args` first, then
:data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR`. Runnable candidates: 3/24 both
ways when a person types the request, 9/24 both ways when a suite supplies its values.
Meanwhile retrieval's precision gets WORSE - a cosine is almost never zero, so five of
six irrelevant requests come back with a candidate instead of one - and no cut-off
separates the two populations, so there is nothing to tune either.

Turn it on with ``SKILLWEAVER_EMBEDDER=true`` to re-measure (``make embedder``, then
``scripts/bench_retrieval.py``), and flip this line when the number that matters moves.
"""

DEFAULT_HEADLESS = False
"""Whether the browser this project opens runs without a visible window.

Headed by DEFAULT, and that is a product decision rather than an oversight. A visible
window is what makes the agent legible - a person watching it work is the whole reason
a computer-use agent is convincing - so the mode a plain command opens is the one a
person can see.

Headless is what MEASUREMENT wants, and it is one flag away: a suite of tasks run four
times each has no audience, and a browser window stealing focus on a laptop, in CI or
over SSH is a cost with no benefit. See ``--headless`` in :mod:`skillweaver.cli`.

The two modes are not interchangeable, which is why this is recorded rather than merely
chosen: two fresh browsers of opposite modes on ONE page fingerprint 0.126 and 0.421
apart, both at or below :data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD`,
so a skill learned in one mode can never match a screen rendered in the other. See
:mod:`skillweaver.render_mode`, which names that mismatch instead of letting it show up
as a mysteriously low similarity."""

DEFAULT_CHROME_PROFILE: Path | None = None
"""The persistent Chrome profile a browser run drives, or ``None`` for none.

``None`` by default, so a plain run opens the bundled Chromium with a throwaway profile
exactly as it always has. Set it - ``SKILLWEAVER_CHROME_PROFILE``, or ``--chrome-profile``
on one invocation - to drive the real Google Chrome on this machine out of a directory
that survives the run, which is what a site refusing an automated browser requires. See
``REAL_CHROME_CHANNEL`` in :mod:`skillweaver.controllers.browser` for what that is
measured to fix and what it deliberately does not do.

One directory per run. A profile is EXCLUSIVE - a Chrome window already open on it makes
the launch fail rather than share it - and two runs pointed at one directory fight over
the lock and spoil the state that made the setting worth having."""

DEFAULT_CHROME_ATTACH = False
"""Whether a browser run starts that real Chrome ITSELF and attaches to it, rather than
letting Playwright launch it. ``False`` by default, so nothing changes for a run that
says nothing.

Set it - ``SKILLWEAVER_CHROME_ATTACH``, or ``--chrome-attach`` on one invocation, both
of which need ``chrome_profile`` as well - for a site that refuses even the real Chrome
when the automation framework is what started it. That is measured, three configurations
against live doordash.com, at ``PLAINLY_LAUNCHED`` in
:mod:`skillweaver.controllers.chrome_launch`, which also carries what this mode must
never be extended into: it does not defeat, mask or retry past a human-verification page,
and a challenge fails the run for a person to clear by hand."""
DEFAULT_PERCEPTION = PIXELS
"""Which eyes a run opens with. PIXELS, and that is the point of the default.

The pixel path is what every stored skill was learned against, what both landed
speedups were measured on, and what the project's claim to be a computer-use agent
rests on. The DOM path (``--perception dom``) is opt-in for browser use only; see
:mod:`skillweaver.perception_mode` and ``AGENTS.md`` for the invariant it relaxes.
"""

DEFAULT_POLICY = "claude"
"""Who chooses each move. Claude through the acting prompt, unchanged.

``jev`` swaps in :class:`~skillweaver.llm.jev_.JevPolicy` and requires
``--perception dom``, because a policy that acts on an indexed control table needs a
perceiver that produces one.
"""

POLICIES = ("claude", "jev")
"""Every acting policy, for validating a flag or a setting."""

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TARGETS = ("browser", "desktop")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration. API keys are excluded from ``repr`` so they cannot
    leak into logs."""

    data_dir: Path = Path("data")
    default_target: Literal["browser", "desktop"] = "browser"
    log_level: str = "INFO"
    headless: bool = DEFAULT_HEADLESS
    chrome_profile: Path | None = DEFAULT_CHROME_PROFILE
    chrome_attach: bool = DEFAULT_CHROME_ATTACH
    perception: str = DEFAULT_PERCEPTION
    policy: str = DEFAULT_POLICY
    claude_model: str = DEFAULT_CLAUDE_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    skill_max_seconds: float = DEFAULT_SKILL_MAX_SECONDS
    embedder_enabled: bool = DEFAULT_EMBEDDER_ENABLED
    embedder_dir_override: Path | None = None
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
    def embedder_dir(self) -> Path:
        """Where the retrieval embedding model's weights live.

        ``<models_dir>/text_embedder`` unless ``SKILLWEAVER_EMBEDDER_DIR`` names
        somewhere else, so several worktrees can share one 90 MB download.
        """
        if self.embedder_dir_override is not None:
            return self.embedder_dir_override
        return self.models_dir / EMBEDDER_DIRNAME

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


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    """A boolean setting, spelled any of the ways a shell or a CI file spells one.

    ``1/true/yes/on`` and ``0/false/no/off``, in any case. An unset or empty value
    means ``default``, so a variable that is merely present-but-blank does not flip
    a mode nobody asked to flip.

    Raises:
        ConfigError: on anything else - a mode silently misread is worse than a
            command that refuses to start.
    """
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{key}={raw!r} is not a boolean: use one of {_TRUE} or one of {_FALSE}")


def _path(env: Mapping[str, str], key: str, default: Path | None) -> Path | None:
    """An optional filesystem path. Blank and unset both mean the default, so
    ``SKILLWEAVER_CHROME_PROFILE=`` turns a configured profile off rather than
    resolving to the current directory."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw.strip()).expanduser()


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

    perception = (merged.get("SKILLWEAVER_PERCEPTION") or DEFAULT_PERCEPTION).lower()
    if perception not in PATHS:
        raise ConfigError(f"SKILLWEAVER_PERCEPTION={perception!r} must be one of {PATHS}")
    policy = (merged.get("SKILLWEAVER_POLICY") or DEFAULT_POLICY).lower()
    if policy not in POLICIES:
        raise ConfigError(f"SKILLWEAVER_POLICY={policy!r} must be one of {POLICIES}")

    defaults = Budget()
    budget = Budget(
        max_steps=_number(merged, "SKILLWEAVER_MAX_STEPS", defaults.max_steps, int),
        max_seconds=_number(merged, "SKILLWEAVER_MAX_SECONDS", defaults.max_seconds, float),
        max_usd=_number(merged, "SKILLWEAVER_MAX_USD", defaults.max_usd, float),
        max_llm_calls=_number(merged, "SKILLWEAVER_MAX_LLM_CALLS", defaults.max_llm_calls, int),
    )
    settings = Settings(
        data_dir=Path(merged.get("SKILLWEAVER_DATA_DIR") or "data"),
        default_target=target,  # type: ignore[arg-type]
        log_level=log_level,
        headless=_flag(merged, "SKILLWEAVER_HEADLESS", DEFAULT_HEADLESS),
        chrome_profile=_path(merged, "SKILLWEAVER_CHROME_PROFILE", DEFAULT_CHROME_PROFILE),
        chrome_attach=_flag(merged, "SKILLWEAVER_CHROME_ATTACH", DEFAULT_CHROME_ATTACH),
        perception=perception,
        policy=policy,
        claude_model=merged.get("SKILLWEAVER_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
        gemini_model=merged.get("SKILLWEAVER_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        skill_max_seconds=_number(
            merged, "SKILLWEAVER_SKILL_MAX_SECONDS", DEFAULT_SKILL_MAX_SECONDS, float
        ),
        embedder_enabled=_flag(merged, "SKILLWEAVER_EMBEDDER", DEFAULT_EMBEDDER_ENABLED),
        embedder_dir_override=(
            Path(merged["SKILLWEAVER_EMBEDDER_DIR"])
            if merged.get("SKILLWEAVER_EMBEDDER_DIR")
            else None
        ),
        anthropic_api_key=merged.get("ANTHROPIC_API_KEY") or None,
        gemini_api_key=merged.get("GEMINI_API_KEY") or merged.get("GOOGLE_API_KEY") or None,
        default_budget=budget,
    )
    return check_settings(settings)


def check_settings(settings: Settings) -> Settings:
    """The settings back, or a refusal naming what does not go together.

    Applied to the environment AND to the flags laid over it, because the two halves of
    a combination can arrive from different places: ``SKILLWEAVER_CHROME_ATTACH=1`` in a
    shell profile and the directory it needs on the command line, or the other way
    about. The alternative is discovering it when the browser fails to open, several
    seconds and one confusing message later.

    Raises:
        ConfigError: if two settings contradict each other.
    """
    if settings.chrome_attach and settings.chrome_profile is None:
        raise ConfigError(
            "chrome_attach starts a real Chrome of its own and needs a profile "
            "directory to start it on: set SKILLWEAVER_CHROME_PROFILE or pass "
            "--chrome-profile, and give every run its own directory"
        )
    return settings


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process-wide settings, loaded once from ``os.environ`` and ``./.env``.
    Tests that change the environment call ``settings.cache_clear()`` afterwards.

    Raises:
        ConfigError: if the environment holds an invalid value.
    """
    return load_settings()
