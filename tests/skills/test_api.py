"""The skill clock: what it charges a skill for, and what it does not.

``tests/skills/test_sandbox.py`` covers the sandbox as a whole - the static scan, the
namespace, the three limits in their simple form. This file is about the one thing
that measurement got wrong on a real web page, and how it was fixed.

The bug it pins down, from a live run against en.wikipedia.org on 2026-09-19: a
stored skill that did its job perfectly was killed with ``skill ran for 20.89s of
20.00s allowed``. It had performed three actions and observed three times. The
actions and the code were milliseconds; the twenty seconds were a screenshot and an
OCR pass over a dense article, three times over. The limit exists to interrupt a
runaway loop in generated code, and it had instead become a bet on how fast somebody
else's web page renders.

So the ledger now separates the two clocks, and the two tests that matter are a pair:

* a skill that genuinely spins STILL hits the limit - the protection is intact;
* a skill that merely observes a slow page does NOT - the conflation is gone.

Nothing here touches a network, a model or a browser. "A slow page" is
:class:`SlowPerceiver` below, which sleeps and counts.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest

from skillweaver.config import DEFAULT_SKILL_MAX_SECONDS, settings
from skillweaver.contracts import (
    Click,
    Observation,
    Point,
    Provenance,
    Skill,
    TypeText,
    Wait,
    utcnow,
)
from skillweaver.errors import ControllerError
from skillweaver.skills.api import (
    RunLedger,
    SkillAPI,
    SkillLimits,
    TimeLimitExceeded,
    default_max_seconds,
)
from skillweaver.skills.sandbox import SkillRunner
from tests.fakes import Scenario

DOMAIN = "acme.test"
PROVENANCE = Provenance("run-0", "watch a slow page", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))


def make(name: str, code: str) -> Skill:
    """A structurally complete, unstored skill around ``code``, as
    ``tests/skills/test_sandbox.py`` builds one."""
    return Skill(
        name=name,
        domain=DOMAIN,
        summary=f"{name} for tests",
        docstring=f"{name} for tests",
        params={},
        code=code,
        requires=(),
        precondition=None,
        verifier_code=None,
        provenance=PROVENANCE,
    )


class SlowPerceiver:
    """A ``contracts.Perceiver`` that takes ``delay`` seconds to answer.

    A stand-in for a real page: a capture plus OCR over a dense article is seconds of
    wall clock during which the skill's own code is doing nothing at all. ``calls``
    counts how many times it was asked, so a test can show the cost was really paid
    rather than optimized away by the observation cache.
    """

    def __init__(self, inner: Any, delay: float) -> None:
        self._inner = inner
        self._delay = delay
        self.calls = 0

    def observe(self, controller: Any) -> Observation:
        self.calls += 1
        time.sleep(self._delay)
        return self._inner.observe(controller)


class SlowController:
    """A controller that takes ``delay`` seconds to deliver every action."""

    def __init__(self, inner: Any, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def perform(self, action: Any) -> Any:
        time.sleep(self._delay)
        return self._inner.perform(action)


# --------------------------------------------------------------------------------------
# The pair: the protection stays, the conflation goes
# --------------------------------------------------------------------------------------


def test_a_skill_that_genuinely_loops_still_hits_the_time_limit(scenario: Scenario) -> None:
    runner = SkillRunner()
    ctx = runner.context(
        scenario.controller, scenario.perceiver, limits=SkillLimits(max_seconds=0.25)
    )
    started = time.monotonic()

    result = runner.run(
        make("spin", "def run(ctx):\n    n = 0\n    while True:\n        n += 1\n"), {}, ctx
    )

    elapsed = time.monotonic() - started
    assert not result.ok
    assert result.error is not None and "TimeLimitExceeded" in result.error
    assert elapsed < 5.0, "the runaway-loop tripwire must still fire, and quickly"


def test_a_skill_that_only_watches_a_slow_page_is_not_charged_for_it(
    scenario: Scenario,
) -> None:
    """Four observations of a page that takes 0.2s each, against a 0.5s limit.

    Under the old accounting this is 0.8s of "skill time" and a dead skill. It is the
    shape of the Wikipedia replay that failed: real work, real result, and the whole
    allowance spent waiting for pixels.
    """
    slow = SlowPerceiver(scenario.perceiver, delay=0.2)
    runner = SkillRunner()
    ctx = runner.context(scenario.controller, slow, limits=SkillLimits(max_seconds=0.5))

    result = runner.run(
        make(
            "watch",
            "def run(ctx):\n"
            "    seen = 0\n"
            "    for _i in range(4):\n"
            "        seen = len(ctx.see.all())\n"
            "        ctx.ctl.wait(1)\n"
            "    return seen\n",
        ),
        {},
        ctx,
    )

    assert result.ok, result.error
    assert slow.calls == 4, "the waiting really happened; it just was not charged"
    assert ctx.ledger.blocked_seconds >= 0.8
    assert ctx.ledger.elapsed_seconds < 0.5


def test_a_slow_controller_is_not_charged_either(scenario: Scenario) -> None:
    """The browser taking a second to settle after a click is a slow browser."""
    slow = SlowController(scenario.controller, delay=0.15)
    runner = SkillRunner()
    ctx = runner.context(slow, scenario.perceiver, limits=SkillLimits(max_seconds=0.4))

    result = runner.run(
        make(
            "poke",
            "def run(ctx):\n    for _i in range(4):\n        ctx.ctl.press('Escape')\n",
        ),
        {},
        ctx,
    )

    assert result.ok, result.error
    assert ctx.ledger.blocked_seconds >= 0.6


def test_a_skill_that_sleeps_on_purpose_is_charged_for_it() -> None:
    """``ctx.ctl.wait`` is the skill spending its own time, so it counts.

    The distinction is the whole rule: waiting on the page is the world being slow,
    asking to sleep is the skill choosing to be. Without this a "just wait longer"
    repair - the most common shape a model reaches for - would be unbounded.
    """
    ledger = RunLedger(SkillLimits(max_seconds=100))
    ledger.charge_step("wait 1ms")
    before = ledger.elapsed_seconds

    time.sleep(0.05)  # a Wait is delivered outside any ledger.blocked() region

    assert ledger.elapsed_seconds - before >= 0.05
    assert ledger.blocked_seconds == 0.0


# --------------------------------------------------------------------------------------
# The ledger's own bookkeeping
# --------------------------------------------------------------------------------------


def test_the_clock_is_held_from_the_moment_the_wait_starts() -> None:
    """Read from INSIDE an unfinished wait, the elapsed time must not be growing.

    This is not a detail. ``sandbox.py``'s deadline tracer fires on trace events
    raised by the controller's and perceiver's own Python, which is to say DURING the
    wait. A ledger that only banked the time on the way out would trip during exactly
    the wait it was meant to forgive.
    """
    ledger = RunLedger(SkillLimits(max_seconds=0.2))
    with ledger.blocked():
        time.sleep(0.3)
        inside = ledger.elapsed_seconds
        ledger.check_time()  # must not raise
    assert inside < 0.05
    assert ledger.blocked_seconds >= 0.3


def test_waits_nest_without_double_counting() -> None:
    ledger = RunLedger(SkillLimits(max_seconds=10))
    with ledger.blocked():
        time.sleep(0.1)
        with ledger.blocked():
            time.sleep(0.1)
    assert 0.2 <= ledger.blocked_seconds < 0.35, "0.2s of wall clock, banked once"


def test_a_wait_that_raises_still_banks_its_time() -> None:
    ledger = RunLedger(SkillLimits(max_seconds=10))
    with pytest.raises(ControllerError):
        with ledger.blocked():
            time.sleep(0.05)
            raise ControllerError("the page went away")
    assert ledger.blocked_seconds >= 0.05
    assert ledger.elapsed_seconds < 0.05


def test_the_message_names_what_was_not_counted() -> None:
    """A reader who sees the limit trip needs to know the wait was already forgiven,
    otherwise the obvious next move is to raise the limit for no reason."""
    ledger = RunLedger(SkillLimits(max_seconds=0.01))
    with ledger.blocked():
        time.sleep(0.05)
    time.sleep(0.02)
    with pytest.raises(TimeLimitExceeded, match=r"of 0\.01s allowed \(not counting 0\.0\ds"):
        ledger.check_time()


def test_elapsed_time_is_never_negative() -> None:
    ledger = RunLedger(SkillLimits(max_seconds=10))
    with ledger.blocked():
        time.sleep(0.05)
    assert ledger.elapsed_seconds >= 0.0


def test_a_zero_limit_still_means_no_limit() -> None:
    ledger = RunLedger(SkillLimits(max_seconds=0))
    time.sleep(0.02)
    ledger.check_time()  # must not raise


# --------------------------------------------------------------------------------------
# Where the limit comes from
# --------------------------------------------------------------------------------------


def test_the_default_limit_is_the_configured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLWEAVER_SKILL_MAX_SECONDS", "90")
    settings.cache_clear()
    try:
        assert default_max_seconds() == 90.0
        assert SkillLimits().max_seconds == 90.0
    finally:
        settings.cache_clear()


def test_the_shipped_default_is_honest_about_a_real_page() -> None:
    settings.cache_clear()
    try:
        assert SkillLimits().max_seconds == DEFAULT_SKILL_MAX_SECONDS
        assert DEFAULT_SKILL_MAX_SECONDS > 20.0, (
            "20s is the value that killed a working Wikipedia skill"
        )
    finally:
        settings.cache_clear()


def test_a_broken_setting_falls_back_rather_than_removing_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misspelt variable must not leave the sandbox with no tripwire at all."""
    monkeypatch.setenv("SKILLWEAVER_SKILL_MAX_SECONDS", "not-a-number")
    settings.cache_clear()
    try:
        assert default_max_seconds() == DEFAULT_SKILL_MAX_SECONDS
    finally:
        settings.cache_clear()


def test_an_explicit_limit_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLWEAVER_SKILL_MAX_SECONDS", "90")
    settings.cache_clear()
    try:
        assert SkillLimits(max_seconds=2.0).max_seconds == 2.0
    finally:
        settings.cache_clear()


# --------------------------------------------------------------------------------------
# What a Wait is, at the seam
# --------------------------------------------------------------------------------------


def test_only_a_wait_escapes_the_blocked_region(scenario: Scenario) -> None:
    """Every action but ``Wait`` is delivered inside a held clock.

    Checked at the seam rather than by timing, so the rule is pinned down rather than
    inferred from a sleep that might be scheduled generously on a loaded machine.
    """
    ledger = RunLedger(SkillLimits(max_seconds=100))
    ctx = SkillAPI(scenario.controller, scenario.perceiver, ledger=ledger)

    ctx.ctl.perform(Wait(1))
    after_wait = ledger.blocked_seconds
    ctx.ctl.perform(Click(Point(10, 10)))
    ctx.ctl.perform(TypeText("hello"))

    assert after_wait == 0.0, "a deliberate sleep is the skill's own time"
    assert ledger.blocked_seconds > 0.0, "a click is the world's"


def test_observing_is_banked_as_blocked(scenario: Scenario) -> None:
    ledger = RunLedger(SkillLimits(max_seconds=100))
    slow = SlowPerceiver(scenario.perceiver, delay=0.1)
    ctx = SkillAPI(scenario.controller, slow, ledger=ledger)

    ctx.observe()
    assert ledger.blocked_seconds >= 0.1

    banked = ledger.blocked_seconds
    ctx.observe()
    assert ledger.blocked_seconds == banked, "the cached observation costs nothing"
    assert slow.calls == 1


def test_an_observation_is_still_an_observation(scenario: Scenario) -> None:
    """The blocked region must not change what observing returns."""
    ctx = SkillAPI(scenario.controller, scenario.perceiver, ledger=RunLedger())
    observation = ctx.observe()
    assert isinstance(observation, Observation)
    assert observation.taken_at <= utcnow()
    assert observation.elements
