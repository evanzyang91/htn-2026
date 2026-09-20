"""Which renderer produced a screen - headed or headless - and one sentence naming a
crossing when a warm attempt loses its screen.

Two fresh browsers of opposite modes fingerprint 0.126 apart (Wikipedia Main Page) and
0.421 (``json.html``), both under ``SAME_STATE_THRESHOLD``, but 0.819 on the sandbox app
- so a crossing is explained, never forbidden. Nothing here demotes a skill: a crossing
is lost at ``no_route`` while planning, and only a skill that ran and failed is retired.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HEADED", "HEADLESS", "MODES", "crossing", "mode_name", "mode_of"]

HEADED = "headed"
"""The default: a visible window is what makes the agent legible to somebody watching."""

HEADLESS = "headless"

MODES = (HEADED, HEADLESS)


def mode_name(headless: bool) -> str:
    """``HEADLESS`` when ``headless``, else ``HEADED``."""
    return HEADLESS if headless else HEADED


def mode_of(controller: Any) -> str | None:
    """The mode ``controller`` renders in, or ``None`` for no claim - which callers must
    not read as "headed". Duck-typed on a ``headless`` attribute, so ``Controller`` stays small."""
    found = getattr(controller, "headless", None)
    return mode_name(found) if isinstance(found, bool) else None


def crossing(recorded: str | None, current: str | None) -> str | None:
    """One sentence when ``recorded`` and ``current`` are two known, unequal modes, else
    ``None``. Offered as the likely reason a screen went unrecognised, not a verdict."""
    if recorded is None or current is None or recorded == current:
        return None
    fix = "--headless" if recorded == HEADLESS else "--headed"
    return (
        f"the library was recorded {recorded} and this run is {current}, which is the "
        f"likely reason: the two renderers do not draw one page alike, and how far apart "
        f"depends on the page - measured 0.126 and 0.421 similarity across modes on two "
        f"real pages against a same-state cut of 0.26, and 0.819 on the sandbox app. The "
        f"library is intact; re-run with {fix}, or learn the task again in this mode."
    )
