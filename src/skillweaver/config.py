"""Runtime settings from the process environment and an optional ``.env``.

Environment wins over ``.env``, both win over the defaults below; ``load_settings``
names every ``SKILLWEAVER_*`` variable it reads. No secret has a default.
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

DEFAULT_SKILL_MAX_SECONDS = 45.0
"""Seconds a skill may spend in its OWN code - a runaway-loop tripwire, not a work
allowance: ``SkillLimits`` does not charge it for time blocked on a screenshot or OCR."""

DEFAULT_EMBEDDER_ENABLED = False
"""Off because it was MEASURED and did not pay: top-1 recall 18/24 -> 21/24, but runnable
candidates 3/24 and 9/24 BOTH ways (the word-counting gates downstream stop them) and
precision worse, 1/6 -> 5/6 irrelevant requests answered. Re-measure with
``scripts/bench_retrieval.py`` before flipping this."""

DEFAULT_HEADLESS = False
"""Headed by default because the demo is watched. The two modes are not interchangeable
- 0.126 and 0.421 cross-mode similarity on two real pages, under the same-state cut - so
the mode is recorded beside a skill and a crossing is named by ``render_mode``."""

DEFAULT_CHROME_PROFILE: Path | None = None
"""``None`` opens bundled Chromium on a throwaway profile; set it to drive the real Chrome
(``REAL_CHROME_CHANNEL`` in ``controllers.browser``). One directory per run - a profile is
exclusive, and two runs sharing one fight over the lock."""

DEFAULT_CHROME_ATTACH = False
"""Start real Chrome ourselves and attach, for a site that refuses even real Chrome when
the framework launched it (measured at ``PLAINLY_LAUNCHED`` in ``controllers.chrome_launch``,
which also states what this must never become). Needs ``chrome_profile``."""
DEFAULT_BROWSER = "harness"
"""``harness`` drives the Chrome the person already has open, through Browser Harness and
raw DevTools-protocol calls, the way ``jev_ultrafast`` does (``controllers.harness``).
``playwright`` is the framework-driven browser every earlier run used, and the only one
of the two that can run headless or without a Chrome already open. The default moved
because the framework-driven browser is refused by most real shops, fresh profile and all,
and a browser a site will not serve has no other property worth having."""

BROWSERS = ("harness", "playwright")

DEFAULT_PERCEPTION = PIXELS
"""Pixels: what every stored skill was learned against. ``--perception dom`` is opt-in
and browser-only; see ``perception_mode`` and ``AGENTS.md``."""

DEFAULT_POLICY = "claude"
"""``jev`` swaps in ``JevPolicy`` and requires ``--perception dom``: it acts on an
indexed control table and needs a perceiver that produces one."""

POLICIES = ("claude", "jev")

DEFAULT_TEXT_MODEL = "gpt-5.4-mini"
"""The model that writes what ``--policy jev`` types (``SKILLWEAVER_TEXT_MODEL``), over an
OpenAI-compatible endpoint. A SETTING because the model belongs in ``.env`` and not in
code, and the id was read from the account's own model listing rather than from memory.

Chosen by measurement through ``OpenAITextWriter``, 2026-09-20. The case that decides it
is an errand's SECOND item, the first already in the cart: a wrong query there is a wrong
product. Right answers of 15, and median latency:

==============  =====  ======  ==================================================
``gpt-5.4-mini``  15/15  650ms   (also 5/5 declining a newsletter field)
``gpt-4.1-nano``  13/15  412ms   both misses typed the field's own LABEL into it
``gpt-5.4-nano``  10/15  526ms   five replies with no usable value
``gpt-4.1-mini``   5/5   432ms   but declined the newsletter field only 2 times in 5
``gpt-5-nano``     4/5  4288ms   reasons first: 9613 output tokens for a few words
==============  =====  ======  ==================================================

The 240ms over the fastest is not the cost that matters; a value nobody asked for,
typed into a real form, is."""

DEFAULT_REFINE_GOAL = False
"""``SKILLWEAVER_REFINE_GOAL``: rewrite the errand, once, into the short ordered sentences
a small policy follows best, and show THE POLICY that instead. Off by default because it
is one more model call before the first move and it changes what every decision sees, so
it should be measurable against a run without it rather than switched on invisibly.

MEASURED, and on the one kind of task it could be measured on it did not pay. Live
Wikipedia, a deliberately vague single-item request ("the article about how plants make
food from light"), cold, n=2 per arm, 2026-09-20: solved 2 of 2 BOTH ways by the same
``TYPE_TEXT > CLICK > DONE`` in 4 actions, and refinement cost 2-3 more model calls and
1.6-8.6s of wall. What it is FOR is a vague MULTI-item errand on a store, which needs a
cart that can be put back between the gate's re-runs and was not run. Turn it on for that
and measure it there; do not quote it as a saving before then.

The rewrite never reaches the recorder or a stored precedent - verified on a learn run:
the refined wording in 0 stored files, and the trajectory, the provenance and the
``Precedent`` all carrying the user's own sentence. ``JevDriver._goal_shown`` holds that
line and says why. ``SKILLWEAVER_REFINE_MODEL`` names a slower, more careful model
for it, and defaults to the text model."""

DEFAULT_FAST_MOVES = True
"""``SKILLWEAVER_FAST_MOVES``: on a ``--policy jev`` run, judge each MOVE by literal page
change alone (``agent/move_critic.py``) instead of escalating to the vision model. It
reaches NO other path - ``orchestrator._open_move_critic`` - and never the ``done`` claim,
which keeps the full critic; set it to ``0`` to get the per-move model verdict back.

On by default because it was measured and learning survived it. Live splitkb.com, cold
from the home page, *Add the "LPF Glow ... Keycaps" to the cart*, private headless Chrome
through the harness, interleaved A B A B, 2026-09-20, n=2 per arm:

  per-move verdict | wall to solved | model calls | cost          | solved | admitted
  model (off)      | 33.5s, 32.1s   | 17, 14      | $0.243, $0.198 | 2/2    | 2/2
  literal (on)     | 14.3s, 13.6s   | 11, 11      | $0.048, $0.049 | 2/2    | 2/2 (one repair)

The same six moves both ways. A move the model judged took 5.8-8.4s against 0.5-2.4s for
the same move judged literally, and the model arm was WRONG twice per run in the
direction ``AGENTS.md`` warns of: it failed the typed search field and the AJAX *Add to
cart*, both of which had worked. The skill the literal arm stored replayed warm in 4.4s,
0 model calls. Read-only control on live Wikipedia search, n=1 per arm: 15.3s / 9 calls
against 10.8s / 8 (log timestamps, start to solved), both admitted. ``wall to solved`` is
``AttemptRecord.wall_ms``.
"""

DEFAULT_TEXT_BASE_URL = "https://api.openai.com/v1"
"""Any OpenAI-compatible endpoint works; ``SKILLWEAVER_TEXT_BASE_URL`` names another."""

TEXT_EFFORTS = ("low", "medium", "high")
"""What ``SKILLWEAVER_TEXT_EFFORT`` may say. Sent only to a model that reasons, and left
unset by default: it is a latency knob and the default model does not need it."""

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TARGETS = ("browser", "desktop")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved configuration; API keys are out of ``repr`` so they cannot leak into logs."""

    data_dir: Path = Path("data")
    default_target: Literal["browser", "desktop"] = "browser"
    log_level: str = "INFO"
    headless: bool = DEFAULT_HEADLESS
    chrome_profile: Path | None = DEFAULT_CHROME_PROFILE
    chrome_attach: bool = DEFAULT_CHROME_ATTACH
    browser: str = DEFAULT_BROWSER
    perception: str = DEFAULT_PERCEPTION
    policy: str = DEFAULT_POLICY
    claude_model: str = DEFAULT_CLAUDE_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    skill_max_seconds: float = DEFAULT_SKILL_MAX_SECONDS
    embedder_enabled: bool = DEFAULT_EMBEDDER_ENABLED
    embedder_dir_override: Path | None = None
    anthropic_api_key: str | None = field(default=None, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)
    typesafe_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    text_model: str = DEFAULT_TEXT_MODEL
    text_base_url: str = DEFAULT_TEXT_BASE_URL
    text_effort: str | None = None
    refine_goal: bool = DEFAULT_REFINE_GOAL
    refine_model: str | None = None
    fast_moves: bool = DEFAULT_FAST_MOVES
    default_budget: Budget = field(default_factory=Budget)

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def graphs_dir(self) -> Path:
        return self.data_dir / "graphs"

    @property
    def trajectories_dir(self) -> Path:
        return self.data_dir / "trajectories"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def embedder_dir(self) -> Path:
        """``SKILLWEAVER_EMBEDDER_DIR`` overrides, so worktrees can share one 90 MB download."""
        if self.embedder_dir_override is not None:
            return self.embedder_dir_override
        return self.models_dir / EMBEDDER_DIRNAME

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


def parse_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines, ``#`` comments, optional ``export`` prefix and matching
    quotes; a missing file gives ``{}``."""
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
    """``1/true/yes/on`` or ``0/false/no/off``, any case; unset or blank means ``default``.
    Anything else raises rather than silently misreading a mode."""
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
    """Blank and unset both mean ``default``, so ``SKILLWEAVER_CHROME_PROFILE=`` turns a
    profile off rather than resolving to the current directory."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw.strip()).expanduser()


def load_settings(
    env: Mapping[str, str] | None = None, env_file: Path | str | None = Path(".env")
) -> Settings:
    """Build ``Settings`` without caching; ``env_file`` fills in anything ``env`` lacks,
    and ``None`` for either means ``os.environ`` / no file."""
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

    browser = (merged.get("SKILLWEAVER_BROWSER") or DEFAULT_BROWSER).lower()
    if browser not in BROWSERS:
        raise ConfigError(f"SKILLWEAVER_BROWSER={browser!r} must be one of {BROWSERS}")

    text_effort = (merged.get("SKILLWEAVER_TEXT_EFFORT") or "").strip().lower() or None
    if text_effort is not None and text_effort not in TEXT_EFFORTS:
        raise ConfigError(
            f"SKILLWEAVER_TEXT_EFFORT={text_effort!r} must be one of {TEXT_EFFORTS}, or unset"
        )

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
        browser=browser,
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
        typesafe_api_key=merged.get("TYPESAFE_API_KEY") or None,
        openai_api_key=merged.get("OPENAI_API_KEY") or None,
        text_model=merged.get("SKILLWEAVER_TEXT_MODEL") or DEFAULT_TEXT_MODEL,
        text_base_url=merged.get("SKILLWEAVER_TEXT_BASE_URL") or DEFAULT_TEXT_BASE_URL,
        text_effort=text_effort,
        refine_goal=_flag(merged, "SKILLWEAVER_REFINE_GOAL", DEFAULT_REFINE_GOAL),
        refine_model=merged.get("SKILLWEAVER_REFINE_MODEL") or None,
        fast_moves=_flag(merged, "SKILLWEAVER_FAST_MOVES", DEFAULT_FAST_MOVES),
        default_budget=budget,
    )
    return check_settings(settings)


def check_settings(settings: Settings) -> Settings:
    """The settings back, or ``ConfigError`` naming what does not go together. Applied to
    the environment AND to the flags over it, since the halves of a combination can arrive
    from different places."""
    if settings.chrome_attach and settings.chrome_profile is None:
        raise ConfigError(
            "chrome_attach starts a real Chrome of its own and needs a profile "
            "directory to start it on: set SKILLWEAVER_CHROME_PROFILE or pass "
            "--chrome-profile, and give every run its own directory"
        )
    return settings


def browser_backend(settings: Settings) -> str:
    """Which of :data:`BROWSERS` a run opens.

    ``--headless`` and ``--chrome-profile`` each describe a browser THIS PROJECT starts,
    and the person's own Chrome is neither headless nor on a directory of our choosing,
    so either one selects ``playwright`` rather than being silently ignored. That keeps
    every command written before the default moved meaning what it meant.
    """
    if settings.browser == "harness" and (settings.headless or settings.chrome_profile):
        return "playwright"
    return settings.browser


@lru_cache(maxsize=1)
def settings() -> Settings:
    """The process-wide settings, loaded once from ``os.environ`` and ``./.env``. Anything
    changing the environment must call ``settings.cache_clear()`` afterwards."""
    return load_settings()
