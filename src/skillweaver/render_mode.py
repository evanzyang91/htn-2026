"""Which renderer produced a screen - headed or headless - and how a crossing is named.

Two fresh browsers on ONE page do not agree about what that page looks like, and HOW
MUCH they disagree is the page's business rather than a constant. Measured 2026-09-19,
the same URL in two fresh Chromium browsers of opposite modes fingerprints **0.126**
apart (Wikipedia's Main Page) and **0.421** apart (``docs.python.org/3/library/json.html``),
both at or below :data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD`; the
same measurement against a stored precondition of the sandbox ordering app scores
**0.819**, comfortably above it. Two browsers of the SAME mode score **1.000** anywhere.

That spread is why this module explains rather than forbids. A small, clean,
purpose-built page survives the crossing and a real one does not, so refusing every
cross-mode comparison up front would break the sandbox to protect Wikipedia - and a
gate tuned against the sandbox is exactly the kind that looks green until it meets a
website. The comparison is attempted; when it comes back saying "I do not know this
screen", :func:`crossing` supplies the reason it most likely did.

None of this is the fingerprinter being wrong. Two renderers really do produce
measurably different pixels, and a screen identity taught to shrug at that would be a
worse identity - see the "Headed and headless are NOT made comparable" section of
:mod:`skillweaver.skills.synthesize`, which declines to launder the difference and
points here for what to do instead.

What is left is a reporting duty. A skill learned with a visible window is stored
CORRECTLY and may simply be unrecognisable from a headless one, so the failure a user
meets is a warm attempt that quietly never fires: retrieval ranks the right skill, the
precondition scores 0.13, and the run falls through to a model as though the library had
never learned anything. The cost is real money and the report says nothing about why. So
the mode that produced a stored screen is RECORDED -
:mod:`skillweaver.skills.store` keeps it beside the skill it belongs to - and a warm
attempt that lost the screen across modes says which two.

Nothing here demotes anything, and nothing had to be added to stop it: a crossing is
lost at ``no_route``, which is reached while planning, and only a skill that RAN and
failed is ever retired (see "What demotion means" in
:mod:`skillweaver.agent.planner`). A cross-mode run cannot retire a working skill.

The mode itself is chosen once per invocation, by
:attr:`~skillweaver.config.Settings.headless`; ``skillweaver --headless`` is the flag
that overrides it. Headed stays the default, because a visible browser is what makes the
agent legible to somebody watching.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HEADED", "HEADLESS", "MODES", "crossing", "mode_name", "mode_of"]

HEADED = "headed"
"""A browser with a visible window - the default, and what a demo needs."""

HEADLESS = "headless"
"""A browser with no window - what measurement, CI and a run over SSH need."""

MODES = (HEADED, HEADLESS)
"""Every mode a screen can have been rendered in, for validating a recorded value."""


def mode_name(headless: bool) -> str:
    """:data:`HEADLESS` when ``headless``, else :data:`HEADED`."""
    return HEADLESS if headless else HEADED


def mode_of(controller: Any) -> str | None:
    """The mode ``controller`` renders in, or ``None`` when the question does not apply.

    ``None`` is a real answer and the common one outside a browser: a desktop
    controller drives a screen that is always visible and has no second mode to be
    confused with, and a fake or a remote controller need not have an opinion. Callers
    treat ``None`` as "no claim", never as "headed" - inventing a mode for a controller
    that did not state one would put a wrong sentence in a report.

    Duck-typed on a ``headless`` attribute rather than on a class, so a controller this
    module has never heard of can answer simply by having one, and
    :class:`~skillweaver.contracts.Controller` stays as small as it is.
    """
    found = getattr(controller, "headless", None)
    return mode_name(found) if isinstance(found, bool) else None


def crossing(recorded: str | None, current: str | None) -> str | None:
    """One plain sentence when ``recorded`` and ``current`` are different modes.

    ``None`` means there is nothing to say: the modes agree, or one of them is unknown
    and no honest comparison can be claimed either way. Only two known, unequal modes
    produce a sentence - silence is the right answer to an unanswered question.

    It is offered as the LIKELY reason a screen went unrecognised, not as a verdict,
    because the gap is not a constant: the numbers it quotes are the measurements this
    module opens with, and on one of those pages the crossing was survivable. A run
    that lost the screen for some other reason is not helped by being told a falsehood
    confidently.

    The sentence names both modes and the flag that closes the gap, because the person
    reading it is looking at a warm attempt that did nothing and needs to know the
    library itself is intact.
    """
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
