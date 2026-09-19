"""The boundary tests: what a skill may do, what it may not, and what it is told.

Two things are being pinned down here. The first is that the published surface is
the WHOLE surface - every escape a careless generation would reach for is refused by
name. The second is that the three ways a skill can fail stay distinguishable:
skill synthesis reads ``SkillResult.error`` to decide whether to rewrite the code
(a violation or an exception), shorten it (a limit) or leave it alone and run it
somewhere else (a failed expectation).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest

from skillweaver.contracts import (
    ActionSurface,
    Provenance,
    Skill,
    SkillContext,
    utcnow,
)
from skillweaver.errors import (
    ControllerError,
    ExpectationFailed,
    PerceptionError,
    SandboxViolation,
    SkillNotFound,
)
from skillweaver.skills.api import (
    ActionView,
    DepthLimitExceeded,
    NullGraph,
    ReadOnlyGraph,
    RunLedger,
    SkillLimits,
    StepLimitExceeded,
    TimeLimitExceeded,
)
from skillweaver.skills.sandbox import SAFE_BUILTINS, SkillRunner, scan_code
from tests.fakes import (
    FakeController,
    FakeGroundTruth,
    InMemorySiteGraph,
    InMemorySkillStore,
    Scenario,
)
from tests.fakes.controller import navigates

DOMAIN = "fake.test"
PROVENANCE = Provenance("run-0", "confirm a payment", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))


def make(name: str, code: str, *, verifier: str | None = None, domain: str = DOMAIN) -> Skill:
    """A structurally complete, unstored skill around ``code``.

    Built directly rather than through ``make_skill`` so a test can hand the sandbox
    code that ``model.validate_code`` would refuse - refusing it is this module's job
    too, and the two layers have to be tested apart.
    """
    return Skill(
        name=name,
        domain=domain,
        summary=f"{name} for tests",
        docstring=f"{name} for tests",
        params={},
        code=code,
        requires=(),
        precondition=None,
        verifier_code=verifier,
        provenance=PROVENANCE,
    )


@pytest.fixture
def runner(skill_store: InMemorySkillStore) -> SkillRunner:
    return SkillRunner(skill_store)


@pytest.fixture
def ctx(runner: SkillRunner, scenario: Scenario) -> Any:
    return runner.context(scenario.controller, scenario.perceiver, domain=DOMAIN)


# --------------------------------------------------------------------------------------
# A skill that works
# --------------------------------------------------------------------------------------

PAY_ACME = '''
def run(ctx, company):
    """Search for the company, open its invoice and confirm the payment."""
    ctx.ctl.type_text("acme")
    rows = ctx.see.find_text(company)
    ctx.expect(bool(rows), "no row for " + company)
    ctx.ctl.click(rows[0])
    ctx.log("opened the invoice")
    ctx.ctl.click(ctx.see.find_text("Confirm payment")[0])
    return ctx.see.find_text("Payment confirmed")[0].text
'''


def test_a_straightforward_skill_runs_and_returns_its_value(
    runner: SkillRunner, ctx: Any, scenario: Scenario
) -> None:
    result = runner.run(make("pay_acme", PAY_ACME), {"company": "Acme Corp"}, ctx)

    assert result.ok, result.error
    assert result.error is None
    assert result.value == "Payment confirmed"
    assert scenario.solved, "the skill should have driven the app to its goal state"
    assert result.steps == 3, "three controller actions: type, click row, click confirm"
    assert result.ms > 0.0


def test_the_trace_is_the_log_lines_and_the_actions_in_order(runner: SkillRunner, ctx: Any) -> None:
    result = runner.run(make("pay_acme", PAY_ACME), {"company": "Acme Corp"}, ctx)

    assert result.trace[0] == "call pay_acme(company='Acme Corp')"
    assert "type_text 'acme'" in result.trace
    assert "log: opened the invoice" in result.trace
    assert any(line.startswith("ruled out:") for line in result.trace), (
        "a held expectation records the failure it ruled out, never asserts it happened"
    )


def test_see_is_re_observed_after_acting_and_cached_between_actions(
    runner: SkillRunner, ctx: Any, scenario: Scenario
) -> None:
    code = (
        "def run(ctx):\n"
        "    before = len(ctx.see.all()) + len(ctx.see.all())\n"
        "    ctx.ctl.type_text('acme')\n"
        "    return ctx.see.find_text('Clear')[0].text\n"
    )
    result = runner.run(make("look", code), {}, ctx)

    assert result.ok, result.error
    assert result.value == "Clear", "the second look must see the state the action produced"
    assert scenario.perceiver.calls == 2, "twice: once cached, once after the action"


def test_a_verifier_that_rejects_the_result_is_a_clean_failure(
    runner: SkillRunner, ctx: Any
) -> None:
    skill = make(
        "always_seven",
        "def run(ctx):\n    return 7\n",
        verifier="def verify(ctx, result):\n    return result == 8\n",
    )
    result = runner.run(skill, {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert "ExpectationFailed" in result.error
    assert "always_seven" in result.error


# --------------------------------------------------------------------------------------
# The namespace: what skill code may not reach
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "code", "named"),
    [
        ("import", "def run(ctx):\n    import os\n    return os.getcwd()\n", "import os"),
        (
            "from import",
            "def run(ctx):\n    from os import getcwd\n    return getcwd()\n",
            "from os import",
        ),
        ("open", "def run(ctx):\n    return open('/etc/passwd').read()\n", "'open'"),
        ("eval", "def run(ctx):\n    return eval('1 + 1')\n", "'eval'"),
        ("exec", "def run(ctx):\n    exec('x = 1')\n", "'exec'"),
        ("dunder import", "def run(ctx):\n    return __import__('os')\n", "'__import__'"),
        ("module global", "def run(ctx):\n    return SAFE_BUILTINS\n", "'SAFE_BUILTINS'"),
        ("module global", "def run(ctx):\n    return log.info('hi')\n", "'log'"),
        ("globals", "def run(ctx):\n    return globals()\n", "'globals'"),
        ("private attribute", "def run(ctx):\n    return ctx.ctl._controller\n", "'_controller'"),
        ("dunder attribute", "def run(ctx):\n    return ctx.__class__\n", "'__class__'"),
        ("getattr", "def run(ctx):\n    return getattr(ctx, 'ctl')\n", "'getattr'"),
        ("global statement", "def run(ctx):\n    global x\n    x = 1\n", "'global x'"),
        ("async", "async def run(ctx):\n    return 1\n", "asynchronous"),
    ],
)
def test_the_scan_rejects_the_escape_and_names_it(label: str, code: str, named: str) -> None:
    with pytest.raises(SandboxViolation) as caught:
        scan_code(code)

    message = str(caught.value)
    assert named in message, f"{label}: the violation must name what was attempted"
    assert "line" in message, f"{label}: the violation must name the line, for repair"


def test_a_violation_is_reported_by_run_rather_than_raised(runner: SkillRunner, ctx: Any) -> None:
    result = runner.run(make("sneaky", "def run(ctx):\n    import os\n    return os\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("SandboxViolation:")
    assert "import os" in result.error


def test_only_the_allowlisted_builtins_are_in_the_namespace(runner: SkillRunner, ctx: Any) -> None:
    code = "def run(ctx):\n    return sorted([3, 1, 2]) + [len('ab'), max(1, 2), int('7')]\n"
    assert runner.run(make("arithmetic", code), {}, ctx).value == [1, 2, 3, 2, 2, 7]
    assert "print" not in SAFE_BUILTINS, "skills log through ctx.log, which lands in the trace"
    assert "type" not in SAFE_BUILTINS, "type() is one step from the class hierarchy"


def test_a_helper_defined_by_the_skill_itself_is_reachable(runner: SkillRunner, ctx: Any) -> None:
    code = "def half(n):\n    return n // 2\n\n\ndef run(ctx):\n    return half(10)\n"
    assert runner.run(make("with_helper", code), {}, ctx).value == 5


def test_a_skill_cannot_leave_state_behind_for_its_next_run(runner: SkillRunner, ctx: Any) -> None:
    code = "seen = []\n\n\ndef run(ctx):\n    seen.append(1)\n    return len(seen)\n"
    skill = make("counter", code)
    assert runner.run(skill, {}, ctx).value == 1
    assert runner.run(skill, {}, ctx).value == 1, "each run gets a fresh namespace"


# --------------------------------------------------------------------------------------
# The limits, one at a time
# --------------------------------------------------------------------------------------


def test_the_step_budget_trips_before_the_action_is_performed(
    runner: SkillRunner, scenario: Scenario
) -> None:
    ctx = runner.context(
        scenario.controller, scenario.perceiver, limits=SkillLimits(max_steps=3, max_seconds=10)
    )
    result = runner.run(
        make("spin", "def run(ctx):\n    while True:\n        ctx.ctl.wait(1)\n"), {}, ctx
    )

    assert not result.ok
    assert result.error is not None
    assert "StepLimitExceeded" in result.error
    assert result.steps == 3
    assert len(scenario.controller.actions) == 3, "the refused action is never delivered"


def test_the_wall_clock_timeout_interrupts_a_loop_that_never_acts(
    runner: SkillRunner, scenario: Scenario
) -> None:
    ctx = runner.context(
        scenario.controller, scenario.perceiver, limits=SkillLimits(max_seconds=0.25)
    )
    started = time.monotonic()
    result = runner.run(
        make("burn", "def run(ctx):\n    n = 0\n    while True:\n        n += 1\n"), {}, ctx
    )
    elapsed = time.monotonic() - started

    assert not result.ok
    assert result.error is not None
    assert "TimeLimitExceeded" in result.error
    assert elapsed < 5.0, "a pure-Python spin must be interrupted, not waited out"
    assert scenario.controller.actions == [], "it never got as far as acting"


def test_the_tracer_is_removed_again_after_the_run(runner: SkillRunner, scenario: Scenario) -> None:
    import sys

    ctx = runner.context(
        scenario.controller, scenario.perceiver, limits=SkillLimits(max_seconds=0.1)
    )
    before = sys.gettrace()
    assert not runner.run(
        make("burn", "def run(ctx):\n    while True:\n        pass\n"), {}, ctx
    ).ok
    assert sys.gettrace() is before, "settrace must not leak out of a run"


def test_the_composition_depth_cap_trips(runner: SkillRunner, scenario: Scenario) -> None:
    store = InMemorySkillStore()
    for name, target in (("a", "b"), ("b", "c")):
        store.put(make(name, f"def run(ctx):\n    return ctx.call('{target}')\n"))
    store.put(make("c", "def run(ctx):\n    return 'bottom'\n"))
    runner = SkillRunner(store, limits=SkillLimits(max_depth=2))
    ctx = runner.context(scenario.controller, scenario.perceiver, domain=DOMAIN)

    result = runner.run(store.get("a", DOMAIN), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert "DepthLimitExceeded" in result.error
    assert "a -> b -> c" in result.error, "the chain names who called whom"


def test_a_limit_is_not_a_sandbox_violation() -> None:
    assert not issubclass(StepLimitExceeded, SandboxViolation)
    assert not issubclass(TimeLimitExceeded, SandboxViolation)
    assert not issubclass(DepthLimitExceeded, SandboxViolation)


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def test_a_skill_calling_a_sub_skill_succeeds_and_the_sub_skill_run_is_recorded(
    runner: SkillRunner, ctx: Any, skill_store: InMemorySkillStore, scenario: Scenario
) -> None:
    skill_store.put(
        make("open_acme", "def run(ctx):\n    ctx.ctl.type_text('acme')\n    return 'searched'\n")
    )
    caller = skill_store.put(
        make(
            "search_then_report",
            "def run(ctx):\n"
            "    where = ctx.call('open_acme')\n"
            "    ctx.log('sub-skill said ' + where)\n"
            "    return where.upper()\n",
        )
    )

    result = runner.run(caller, {}, ctx)

    assert result.ok, result.error
    assert result.value == "SEARCHED"
    assert result.steps == 1, "the caller's steps include the callee's actions"
    assert "call open_acme()" in result.trace, "the callee's trace is part of the caller's"

    callee_stats = skill_store.get("open_acme", DOMAIN).stats
    assert callee_stats.runs == 1 and callee_stats.successes == 1
    assert callee_stats.mean_ms > 0.0
    assert skill_store.get("search_then_report", DOMAIN).stats.runs == 1


def test_a_failed_run_is_recorded_too(
    runner: SkillRunner, ctx: Any, skill_store: InMemorySkillStore
) -> None:
    skill_store.put(make("flop", "def run(ctx):\n    ctx.expect(False, 'nope')\n"))

    assert not runner.run(skill_store.get("flop", DOMAIN), {}, ctx).ok

    stats = skill_store.get("flop", DOMAIN).stats
    assert stats.runs == 1 and stats.successes == 0


def test_running_a_skill_that_is_not_in_the_library_yet_still_works(
    runner: SkillRunner, ctx: Any
) -> None:
    result = runner.run(make("unstored", "def run(ctx):\n    return 1\n"), {}, ctx)
    assert result.ok and result.value == 1


def test_mutual_recursion_is_stopped_by_the_depth_cap(
    runner: SkillRunner, ctx: Any, skill_store: InMemorySkillStore
) -> None:
    skill_store.put(make("ping", "def run(ctx):\n    return ctx.call('pong')\n"))
    skill_store.put(make("pong", "def run(ctx):\n    return ctx.call('ping')\n"))

    result = runner.run(skill_store.get("ping", DOMAIN), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert "DepthLimitExceeded" in result.error
    assert skill_store.get("ping", DOMAIN).stats.runs >= 1, "the attempts still count"


def test_calling_a_skill_that_does_not_exist_names_it(runner: SkillRunner, ctx: Any) -> None:
    result = runner.run(make("caller", "def run(ctx):\n    return ctx.call('nope')\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert "SkillNotFound" in result.error and "nope" in result.error


def test_a_runner_without_a_store_cannot_compose(scenario: Scenario) -> None:
    runner = SkillRunner()
    ctx = runner.context(scenario.controller, scenario.perceiver, domain=DOMAIN)

    result = runner.run(make("caller", "def run(ctx):\n    return ctx.call('anything')\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None and "SkillNotFound" in result.error


# --------------------------------------------------------------------------------------
# Failure reporting
# --------------------------------------------------------------------------------------


def test_an_exception_reports_the_failing_line_with_its_source(
    runner: SkillRunner, ctx: Any
) -> None:
    code = "def run(ctx):\n    rows = []\n    return rows[3]\n"

    result = runner.run(make("boom", code), {}, ctx)

    assert not result.ok
    assert result.value is None
    assert result.error is not None
    assert result.error.startswith("IndexError:")
    assert "(line 3)" in result.error
    assert any("line 3: return rows[3]" in line for line in result.trace)


def test_the_failing_line_is_the_innermost_skill_line(runner: SkillRunner, ctx: Any) -> None:
    code = "def half(n):\n    return n // 0\n\n\ndef run(ctx):\n    return half(4)\n"

    result = runner.run(make("divider", code), {}, ctx)

    assert result.error is not None
    assert "(line 2)" in result.error, "the division, not the call to it"
    assert any("run() line 6" in line for line in result.trace), "the whole skill chain is shown"


def test_an_expect_failure_is_a_clean_failure_not_a_sandbox_violation(
    runner: SkillRunner, ctx: Any
) -> None:
    code = "def run(ctx):\n    ctx.expect(False, 'no confirm button on this screen')\n"

    result = runner.run(make("checker", code), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("ExpectationFailed:")
    assert "no confirm button on this screen" in result.error
    assert "SandboxViolation" not in result.error
    assert "expect FAILED: no confirm button on this screen" in result.trace


def test_a_refused_action_becomes_a_controller_error(
    runner: SkillRunner, ctx: Any, scenario: Scenario
) -> None:
    scenario.controller.fail_next("the page was busy")

    result = runner.run(make("clicker", "def run(ctx):\n    ctx.ctl.wait(1)\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("ControllerError:")
    assert "the page was busy" in result.error


def test_wrong_arguments_are_reported_rather_than_raised(runner: SkillRunner, ctx: Any) -> None:
    result = runner.run(make("needs_arg", "def run(ctx, company):\n    return company\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None and "TypeError" in result.error


def test_the_trace_is_truncated_rather_than_growing_without_bound(
    runner: SkillRunner, scenario: Scenario
) -> None:
    ctx = runner.context(
        scenario.controller, scenario.perceiver, limits=SkillLimits(max_trace_lines=10)
    )
    code = "def run(ctx):\n    for i in range(500):\n        ctx.log('line')\n    return 'done'\n"

    result = runner.run(make("chatty", code), {}, ctx)

    assert result.ok, result.error
    assert len(result.trace) == 11
    assert result.trace[-1] == "... trace truncated"


# --------------------------------------------------------------------------------------
# The surface itself
# --------------------------------------------------------------------------------------


def test_the_context_and_its_action_view_satisfy_the_contracts(ctx: Any) -> None:
    assert isinstance(ctx, SkillContext)
    assert isinstance(ctx.ctl, ActionSurface)


def test_a_skill_can_navigate_because_the_explorer_can(
    runner: SkillRunner, ctx: Any, scenario: Scenario
) -> None:
    """A recording that navigated has to be reproducible by the skill written from it.

    The explorer may go straight to a URL, and it does: reaching a product listing at
    ``/collections/all`` is one ``navigate`` step. Without this the resulting
    trajectory was unlearnable - the gate spent every repair watching the model
    imitate a URL jump with menu clicks, then rejected all three. An agent that can
    solve a task in a way it cannot remember has a hole in it.
    """
    # The shared Scenario denies `navigate` on purpose, so this brings its own
    # controller rather than weakening a fake every other test depends on.
    source = scenario.controller
    controller = FakeController(
        source.states, source.transitions, start=source.state, viewport=source.viewport()
    )
    context = runner.context(controller, scenario.perceiver, domain=DOMAIN)
    url = "https://shop.test/collections/all"
    code = f"def run(ctx):\n    ctx.ctl.navigate({url!r})\n    return 'gone'\n"

    result = runner.run(make("go", code), {}, context)

    assert result.ok, result.error
    assert [a.kind for a in controller.actions] == ["navigate"]
    assert navigates(url)(controller.actions[0])
    assert result.steps == 1, "navigating costs a step like every other action"


def test_ctl_cannot_reach_the_raw_controller_or_ground_truth(
    runner: SkillRunner, ctx: Any, scenario: Scenario
) -> None:
    public = {name: getattr(ctx.ctl, name) for name in dir(ctx.ctl) if not name.startswith("_")}

    assert set(public) == {
        "click",
        "navigate",
        "perform",
        "press",
        "scroll",
        "supports",
        "type_text",
        "wait",
    }
    assert not any(isinstance(value, FakeController | FakeGroundTruth) for value in public.values())
    assert not hasattr(ctx.ctl, "capture"), "a skill looks with ctx.see, never at raw pixels"
    assert not hasattr(ctx.ctl, "close")

    held = [getattr(ctx.ctl, slot, None) for slot in ActionView.__slots__]
    assert not any(isinstance(value, FakeGroundTruth) for value in held)
    assert not any(
        isinstance(value, FakeGroundTruth) for value in vars(scenario.perceiver).values()
    )

    for code in (
        "def run(ctx):\n    return ctx.ctl._controller\n",
        "def run(ctx):\n    return ctx.ctl.__dict__\n",
    ):
        with pytest.raises(SandboxViolation):
            scan_code(code)

    # ``capture`` is not private, so the scan has nothing to object to - it simply is
    # not there, which is the stronger statement and reaches the model as a plain
    # AttributeError naming the member it invented.
    reaching = runner.run(make("peek", "def run(ctx):\n    return ctx.ctl.capture()\n"), {}, ctx)
    assert not reaching.ok
    assert reaching.error is not None and "AttributeError" in reaching.error

    allowed = runner.run(
        make("ok", "def run(ctx):\n    return ctx.ctl.supports('click')\n"), {}, ctx
    )
    assert allowed.ok and allowed.value is True


def test_graph_is_read_only_even_when_the_real_graph_can_be_written(
    runner: SkillRunner, scenario: Scenario, site_graph: InMemorySiteGraph
) -> None:
    ctx = runner.context(scenario.controller, scenario.perceiver, graph=site_graph, domain=DOMAIN)

    assert isinstance(ctx.graph, ReadOnlyGraph)
    assert not hasattr(ctx.graph, "observe_transition")
    assert not hasattr(ctx.graph, "upsert_state")
    assert not hasattr(ctx.graph, "save")
    assert ctx.graph.states(DOMAIN) == []


def test_a_context_with_no_graph_finds_no_routes(ctx: Any) -> None:
    from skillweaver.contracts import Fingerprint

    assert isinstance(ctx.graph, NullGraph), "no double wrapping: a null graph is already read-only"
    here = Fingerprint("a", {})
    assert ctx.graph.route(here, Fingerprint("b", {})) is None
    assert ctx.graph.neighbors(here) == []


def test_log_never_raises(ctx: Any) -> None:
    ctx.log("")
    ctx.log("x" * 10_000)
    assert len(ctx.ledger.trace) == 2


def test_expect_passing_is_silent_and_returns_none(ctx: Any) -> None:
    assert ctx.expect(True, "the screen is the invoice list") is None
    with pytest.raises(ExpectationFailed, match="the confirm button is gone"):
        ctx.expect(False, "the confirm button is gone")


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------


def test_the_ledger_charges_the_clock_before_the_step() -> None:
    ledger = RunLedger(SkillLimits(max_steps=1, max_seconds=100))
    ledger.charge_step("click (1, 1)")
    assert ledger.steps == 1
    with pytest.raises(StepLimitExceeded, match="1 actions of 1 allowed"):
        ledger.charge_step("click (2, 2)")
    assert ledger.steps == 1, "a refused step is not charged"


def test_the_ledger_reports_elapsed_time_against_its_limit() -> None:
    ledger = RunLedger(SkillLimits(max_seconds=0.01))
    ledger.check_time()
    time.sleep(0.02)
    with pytest.raises(TimeLimitExceeded, match="of 0.01s allowed"):
        ledger.check_time()


def test_descend_names_the_chain_and_unwinds() -> None:
    ledger = RunLedger(SkillLimits(max_depth=1))
    with ledger.descend("outer", DOMAIN):
        assert ledger.current_domain == DOMAIN
        with pytest.raises(DepthLimitExceeded, match="outer -> inner"):
            with ledger.descend("inner", DOMAIN):
                pass  # pragma: no cover - the descend above raises
    assert ledger.depth == 0 and ledger.stack == []


def test_perception_failures_propagate_as_themselves(
    runner: SkillRunner, scenario: Scenario
) -> None:
    class BrokenPerceiver:
        def observe(self, controller: Any) -> Any:
            raise PerceptionError("the detector fell over")

    ctx = runner.context(scenario.controller, BrokenPerceiver(), domain=DOMAIN)
    result = runner.run(make("looker", "def run(ctx):\n    return ctx.see.all()\n"), {}, ctx)

    assert not result.ok
    assert result.error is not None and "PerceptionError" in result.error


def test_the_errors_stay_distinguishable_for_synthesis() -> None:
    for error in (SandboxViolation, ExpectationFailed, ControllerError, SkillNotFound):
        assert not issubclass(error, StepLimitExceeded | TimeLimitExceeded | DepthLimitExceeded)
    assert not issubclass(ExpectationFailed, SandboxViolation)
    assert utcnow().tzinfo is not None
