"""The warm fast path, and the one number this project is about.

``test_a_warm_run_makes_no_model_calls`` is the headline: a task the agent has done
before is done again with the model's call count asserted at exactly zero. Every
other test here defends that claim from the ways a fast path usually cheats - by
improvising when it does not know, by trusting a model's list of skill names, by
running a plan it has no route for, or by keeping a skill that has stopped working.

The model in every test is a :class:`FakeLLM` whose script is EMPTY unless the test
scripts it, so an unexpected model call is not a subtle accounting error: it raises
``ScriptExhausted`` and fails the test loudly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from skillweaver.agent.compose import Composer, Decomposition
from skillweaver.agent.critic import CriticVerdict, TieredCritic
from skillweaver.agent.planner import MIN_ACCOUNTED_FOR, PlanFailure, Planner
from skillweaver.contracts import (
    Action,
    Click,
    Fingerprint,
    Observation,
    Provenance,
    Skill,
    SkillCall,
    TaskSpec,
    TypeText,
    UIState,
    Verdict,
)
from skillweaver.contracts import (
    Planner as PlannerProtocol,
)
from skillweaver.skills.retrieve import SkillRetriever
from skillweaver.skills.sandbox import SkillRunner
from tests.fakes import (
    FakeCritic,
    FakeLLM,
    InMemorySiteGraph,
    InMemorySkillStore,
    Scenario,
    make_scenario,
)
from tests.fakes.scenario import CONFIRM_BUTTON, DOMAIN, ROW_ACME

PROVENANCE = Provenance("run-1", "pay an invoice", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))

TYPE_ACME = TypeText("Acme Corp")
CLICK_ROW = Click(ROW_ACME.box.center)
CLICK_CONFIRM = Click(CONFIRM_BUTTON.box.center)


# -- the fake app's screens, as fingerprints ------------------------------------------


def _screen_fingerprints() -> dict[str, Fingerprint]:
    """The fingerprint of every state of the fake app.

    Taken from a throwaway copy so the scenario under test is never driven, captured
    from or otherwise disturbed by test setup: ``FakeFingerprinter`` is deterministic
    from elements and URL, so these are the same values the scenario under test will
    produce.
    """
    probe = make_scenario()
    found: dict[str, Fingerprint] = {}
    for name in probe.controller.states:
        probe.controller.state = name
        found[name] = probe.perceiver.observe(probe.controller).fingerprint
    return found


FP = _screen_fingerprints()
LIST, SEARCHED, SELECTED, DONE = FP["list"], FP["searched"], FP["selected"], FP["done"]


# -- skills the library is planted with -------------------------------------------------


def plant(
    name: str,
    summary: str,
    code: str,
    *,
    precondition: Fingerprint | None,
    params: dict[str, Any] | None = None,
    docstring: str = "",
) -> Skill:
    return Skill(
        name=name,
        domain=DOMAIN,
        summary=summary,
        docstring=docstring or summary,
        params=params or {},
        code=code,
        requires=(),
        precondition=precondition,
        verifier_code=None,
        provenance=PROVENANCE,
    )


PAY_INVOICE = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company):
    ctx.ctl.type_text(company)
    rows = ctx.see.find_text(company, fuzzy=False)
    ctx.expect(bool(rows), "the search returned no row for " + company)
    ctx.ctl.click(rows[0])
    buttons = ctx.see.find_text("Confirm payment", fuzzy=False)
    ctx.expect(bool(buttons), "no Confirm payment button on the invoice")
    ctx.ctl.click(buttons[0])
    return "confirmed"
""",
    precondition=LIST,
    params={"company": {"type": "string"}},
    docstring="Searches the invoice list for the company, opens its invoice and confirms payment.",
)
"""One skill that does the whole errand from the list screen: the warm path's subject."""

CONFIRM_PAYMENT = plant(
    "confirm_payment",
    "Confirm the payment on an invoice that is already open.",
    """
def run(ctx):
    buttons = ctx.see.find_text("Confirm payment", fuzzy=False)
    ctx.expect(bool(buttons), "no Confirm payment button on this screen")
    ctx.ctl.click(buttons[0])
    return "confirmed"
""",
    precondition=SELECTED,
    docstring="Clicks Confirm payment on the open invoice and leaves the confirmation page up.",
)
"""Starts two screens away from where a run begins, so it can only run after routing."""

SEARCH_INVOICE = plant(
    "search_invoice",
    "Filter the invoice list down to one company.",
    """
def run(ctx, company):
    ctx.ctl.type_text(company)
    return company
""",
    precondition=LIST,
    params={"company": {"type": "string"}},
    docstring="Types a company name into the search field so only its rows remain.",
)

BROKEN_PAY = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company):
    ctx.ctl.type_text(company)
    ctx.expect(False, "the confirmation dialog never appeared")
""",
    precondition=LIST,
    params={"company": {"type": "string"}},
    docstring="Claims to pay an invoice; the site changed underneath it and it no longer works.",
)
"""``pay_invoice`` as it behaves after the site changed: it runs, and it fails."""


# -- wiring -------------------------------------------------------------------------------


LABELS = {
    LIST: "invoice list",
    SEARCHED: "filtered invoice list",
    SELECTED: "open invoice",
    DONE: "payment confirmed",
}


@pytest.fixture
def graph() -> InMemorySiteGraph:
    """The site graph the agent has already learned: the four screens and the three
    edges along the path.

    The edges are planted as observed successes, which is what makes them usable by
    the default verified-only routing policy.
    """
    g = InMemorySiteGraph()
    for fingerprint, label in LABELS.items():
        g.upsert_state(UIState(fingerprint=fingerprint, domain=DOMAIN, label=label))
    g.observe_transition(LIST, (TYPE_ACME,), SEARCHED, ok=True, ms=40.0)
    g.observe_transition(SEARCHED, (CLICK_ROW,), SELECTED, ok=True, ms=30.0)
    g.observe_transition(SELECTED, (CLICK_CONFIRM,), DONE, ok=True, ms=25.0)
    return g


@pytest.fixture
def store() -> InMemorySkillStore:
    return InMemorySkillStore()


def build(
    scenario: Scenario,
    store: InMemorySkillStore,
    graph: InMemorySiteGraph,
    llm: FakeLLM,
    *,
    critic: FakeCritic | None = None,
    compose: bool = False,
) -> Planner:
    """A planner wired to the fake app.

    ``compose=False`` gives it no :class:`Composer` at all, so it cannot reach a model
    even in principle. ``compose=True`` wires one to ``llm`` - whose script is empty
    unless the test filled it, so any call still fails loudly.
    """
    runner = SkillRunner(store)
    return Planner(
        store=store,
        retriever=SkillRetriever(store),
        graph=graph,
        runner=runner,
        critic=critic if critic is not None else FakeCritic(goal=DONE),
        controller=scenario.controller,
        perceiver=scenario.perceiver,
        composer=Composer(llm, store, graph=graph) if compose else None,
    )


def look(scenario: Scenario) -> Observation:
    return scenario.perceiver.observe(scenario.controller)


def performed(scenario: Scenario) -> list[Action]:
    return list(scenario.controller.actions)


# -- the central claim ---------------------------------------------------------------------


def test_a_warm_run_makes_no_model_calls(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THIS IS THE PROJECT'S CENTRAL CLAIM.

    A task the agent has already learned is carried out end to end - the invoice is
    searched for, opened and paid - and the model's call count is asserted at EXACTLY
    ZERO. Not "few", not "cheap": none. The whole point of the skill library is that
    the second time is model-free in the action loop, and this assertion is what makes
    that a fact about the code rather than a claim in a README.

    The planner here is even wired WITH a composer over ``fake_llm``, whose script is
    empty, so a single call would raise ``ScriptExhausted`` rather than quietly pass.
    """
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    outcome = planner.attempt(scenario.task, look(scenario))

    assert outcome is not None and outcome.ok
    assert scenario.solved, "the fake app did not actually reach the confirmation page"
    # The claim, stated twice: at the model, and in the run's own accounting.
    assert fake_llm.calls == 0
    assert outcome.spend.llm_calls == 0
    assert outcome.skill_used == "pay_invoice"
    assert outcome.verdict.ok and outcome.verdict.source == "programmatic"
    assert planner.last_failure is None


def test_the_warm_path_is_model_free_end_to_end_with_the_real_critic(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The same claim again, with no fake in the judging seat.

    The other tests use ``FakeCritic`` to keep the subject narrow. This one wires the
    real :class:`~skillweaver.agent.critic.TieredCritic` - and gives it the same
    empty-scripted model - so the assertion covers the whole warm path as it actually
    ships: retrieve, route, run, judge. ``escalated`` is the critic's own record of
    whether it paid for a model, and it is ``False`` because a fingerprint match
    already answered the question.
    """
    store.put(PAY_INVOICE)
    critic = TieredCritic(fake_llm, expected_state=DONE)
    runner = SkillRunner(store)
    planner = Planner(
        store=store,
        retriever=SkillRetriever(store),
        graph=graph,
        runner=runner,
        critic=critic,
        controller=scenario.controller,
        perceiver=scenario.perceiver,
        composer=Composer(fake_llm, store, graph=graph),
    )

    outcome = planner.attempt(scenario.task, look(scenario))

    assert outcome is not None and outcome.ok
    assert scenario.solved
    assert fake_llm.calls == 0
    assert outcome.spend.llm_calls == 0
    assert isinstance(outcome.verdict, CriticVerdict)
    assert not outcome.verdict.escalated, "the critic answered without a model too"
    assert outcome.verdict.source == "programmatic"


def test_a_warm_plan_is_built_without_touching_the_model_or_the_screen(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """``plan`` is pure: it reads the library and the graph, and performs nothing."""
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    plan = planner.plan(scenario.task, look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Acme Corp"}),)
    assert plan.skills_used == ("pay_invoice",)
    assert fake_llm.calls == 0
    assert performed(scenario) == []
    assert scenario.controller.state == "list"


def test_the_planner_satisfies_the_protocol(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    assert isinstance(build(scenario, store, graph, fake_llm), PlannerProtocol)


# -- declining rather than improvising -------------------------------------------------------


def test_a_task_with_no_matching_skill_returns_none(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Nothing in the library speaks to the task, so the planner declines.

    It does not run the nearest thing it has and hope. The screen is untouched, which
    is what lets the caller hand the very same observation to an explorer.
    """
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Export the address book as a CSV file.", domain=DOMAIN)

    assert planner.attempt(task, look(scenario)) is None
    assert performed(scenario) == []
    assert scenario.controller.state == "list"
    assert fake_llm.calls == 0

    failure = planner.last_failure
    assert isinstance(failure, PlanFailure)
    assert failure.stage == "no_candidates"
    assert failure.performed_nothing
    assert not failure.demoted


def test_a_skill_whose_arguments_the_task_cannot_supply_is_not_run(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A required parameter with no value in the task is a reason to decline, not to
    guess one - guessing is what the explorer is for."""
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Confirm payment of the invoice.", domain=DOMAIN, params={})

    assert planner.plan(task, look(scenario)) is None
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "unbindable_args"
    assert performed(scenario) == []


# -- a candidate that has no account of the task ---------------------------------------------
#
# The ordering suite of 2026-09-19 asked for "order two Vegetable Rolls from Sakura
# Counter, then open the Orders tab and confirm the order is listed there" and the
# library answered with a skill that opens a navigation tab. It bound, it routed, it
# ran in four seconds with no model call, and it was wrong all four times - the
# cheapest possible way to be wrong, and the one a wall-clock table rewards. These
# tests are about the line that now stops it, and about the two things that line must
# NOT do: reject a skill being reused with new arguments, or start accepting a skill
# for a task nobody taught it.


OPEN_TAB = plant(
    "open_top_nav_tab",
    "Open a tab in the top navigation bar.",
    """
def run(ctx, tab):
    hits = ctx.see.find_text(tab, fuzzy=True)
    ctx.expect(bool(hits), "no tab called " + tab)
    ctx.ctl.click(hits[0])
    return tab
""",
    precondition=LIST,
    params={"tab": {"type": "string", "default": "Invoices"}},
    docstring="Clicks a tab in the top navigation bar and leaves that tab's screen up.",
)
"""The trivial navigation skill, in the shape that caused the finding: a short generic
name, a default for its only parameter, and no account of any errand at all."""


def test_a_skill_with_no_account_of_the_task_is_not_run(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The finding, reproduced in miniature and then refused.

    ``open_top_nav_tab`` binds - its one parameter has a default - and it has a route,
    so before this check every gate the planner had said yes. What it does not have is
    any account of paying, of Acme Corp or of an invoice, which is the whole request.
    """
    store.put(OPEN_TAB)
    planner = build(scenario, store, graph, fake_llm)

    assert planner.attempt(scenario.task, look(scenario)) is None
    assert performed(scenario) == [], "nothing may be performed on a candidate this weak"
    assert scenario.controller.state == "list"
    assert fake_llm.calls == 0

    failure = planner.last_failure
    assert isinstance(failure, PlanFailure)
    assert failure.stage == "unaccounted"
    assert failure.performed_nothing
    assert not failure.demoted, "the skill is not broken; it is simply not this task's skill"


def test_the_words_it_could_not_account_for_are_named_in_the_rejection(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A run that declined is only diagnosable if it says what was missing."""
    store.put(OPEN_TAB)
    planner = build(scenario, store, graph, fake_llm)

    assert planner.plan(scenario.task, look(scenario)) is None
    failure = planner.last_failure
    assert failure is not None
    assert "payment" in failure.reason and "acme" in failure.reason
    rejected = {r.skill: r for r in failure.rejected}
    assert rejected["open_top_nav_tab"].stage == "unaccounted"
    assert rejected["open_top_nav_tab"].score > 0.0, "it really was ranked, and really was wrong"


def test_the_skill_that_does_the_task_is_still_run(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The guard against fixing this by refusing everything: with both skills on the
    shelf the right one is still retrieved, still run, and still model-free."""
    store.put(OPEN_TAB)
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm)

    outcome = planner.attempt(scenario.task, look(scenario))

    assert outcome is not None and outcome.ok
    assert outcome.skill_used == "pay_invoice"
    assert outcome.spend.llm_calls == 0


def test_a_skill_reused_with_unfamiliar_arguments_is_not_rejected_for_them(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The false positive this check would be worthless without.

    ``pay_invoice`` was learned on Acme Corp and says nothing anywhere about Initech.
    Paying Initech's invoice is precisely what the skill is FOR, and the words it
    cannot be expected to know arrive as its argument - so they are accounted for.
    """
    store.put(PAY_INVOICE)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(
        text="Confirm payment of the Initech invoice.",
        domain=DOMAIN,
        params={"company": "Initech"},
    )

    plan = planner.plan(task, look(scenario))

    assert plan is not None, "a skill reused with new arguments must still be usable"
    assert plan.skills_used == ("pay_invoice",)


def test_a_task_the_library_cannot_do_still_declines_rather_than_reaching(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """With a full shelf and an errand on none of it, the answer is still no.

    This is the other half of "do not raise a threshold until one case passes": the
    check has to keep declining the tasks that were already declined, for the same
    reason, rather than becoming a number that only this one case trips.
    """
    store.put(OPEN_TAB)
    store.put(PAY_INVOICE)
    store.put(SEARCH_INVOICE)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Export the address book as a CSV file.", domain=DOMAIN)

    assert planner.attempt(task, look(scenario)) is None
    assert performed(scenario) == []
    assert planner.last_failure is not None
    assert planner.last_failure.performed_nothing


def test_the_composer_is_still_consulted_after_an_unaccounted_candidate(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Rejecting a single skill must not close the composite path.

    A chain is built exactly out of skills that each do PART of an errand, so the
    candidate this check turns down as a whole answer may still be a piece of one.
    The planner passes it over and goes on to ask, which is what the model call below
    proves; the composer here declines, and that decline - not the rejection - is what
    is reported.
    """
    store.put(OPEN_TAB)
    llm = FakeLLM(['{"steps": [], "why": "opening a tab does not pay anything"}'])
    planner = build(scenario, store, graph, llm, compose=True)

    assert planner.plan(scenario.task, look(scenario)) is None
    assert llm.calls == 1, "the composite path must still be reached"
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "no_decomposition"
    assert performed(scenario) == []


# -- routing ------------------------------------------------------------------------------------


def test_the_route_to_a_precondition_is_walked_before_the_skill_runs(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The skill starts two screens away, so the graph has to carry the agent there.

    This is the site graph paying for itself: the actions that get from the invoice
    list to an open invoice are replayed from memory, in front of the skill, with no
    model deciding any of them.
    """
    store.put(CONFIRM_PAYMENT)
    planner = build(scenario, store, graph, fake_llm, compose=True)
    task = TaskSpec(text="Confirm the payment on this invoice.", domain=DOMAIN)

    plan = planner.plan(task, look(scenario))
    assert plan is not None
    assert plan.steps == (TYPE_ACME, CLICK_ROW, SkillCall("confirm_payment", DOMAIN, {}))
    assert plan.estimated_ms == pytest.approx(70.0), "route cost came from the recorded edges"

    outcome = planner.attempt(task, look(scenario))

    assert outcome is not None and outcome.ok
    assert performed(scenario) == [TYPE_ACME, CLICK_ROW, CLICK_CONFIRM]
    assert [before for before, _, _ in scenario.controller.history] == [
        "list",
        "searched",
        "selected",
    ]
    assert scenario.solved
    assert fake_llm.calls == 0


def test_a_plan_is_not_executed_at_all_when_no_route_to_the_precondition_exists(
    scenario: Scenario, store: InMemorySkillStore, fake_llm: FakeLLM
) -> None:
    """With an empty graph the agent has no way to reach the skill's start screen.

    Nothing is attempted. A fast path that "tries anyway" is how an agent ends up in
    the archive dead end, and there is no cheap way back from a real screen.

    No composer here on purpose: this is about routing alone, so the planner has no
    second path to fall through to and ``None`` can only mean the route was refused.
    """
    store.put(CONFIRM_PAYMENT)
    planner = build(scenario, store, InMemorySiteGraph(), fake_llm, compose=False)
    task = TaskSpec(text="Confirm the payment on this invoice.", domain=DOMAIN)

    assert planner.plan(task, look(scenario)) is None
    assert planner.attempt(task, look(scenario)) is None

    assert performed(scenario) == []
    assert scenario.controller.state == "list"
    assert fake_llm.calls == 0
    failure = planner.last_failure
    assert failure is not None and failure.stage == "no_route"
    assert failure.performed_nothing
    assert "no known route" in failure.reason


def test_an_unreliable_edge_is_refused_rather_than_replayed(
    scenario: Scenario, store: InMemorySkillStore, fake_llm: FakeLLM
) -> None:
    """A route is only as good as its evidence: an edge that mostly fails is no route."""
    flaky = InMemorySiteGraph()
    for _ in range(9):
        flaky.observe_transition(LIST, (TYPE_ACME,), SEARCHED, ok=False, ms=40.0)
    flaky.observe_transition(LIST, (TYPE_ACME,), SEARCHED, ok=True, ms=40.0)
    flaky.observe_transition(SEARCHED, (CLICK_ROW,), SELECTED, ok=True, ms=30.0)

    store.put(CONFIRM_PAYMENT)
    planner = build(scenario, store, flaky, fake_llm)
    task = TaskSpec(text="Confirm the payment on this invoice.", domain=DOMAIN)

    assert planner.plan(task, look(scenario)) is None
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "no_route"


# -- composition --------------------------------------------------------------------------------


def _chain(*steps: tuple[str, dict[str, Any]]) -> str:
    """The reply a model would give: a JSON chain of skill names and arguments."""
    return json.dumps(
        {"steps": [{"skill": name, "args": args} for name, args in steps], "why": "two known parts"}
    )


COMPOSITE_TASK = TaskSpec(
    text="Take care of the Acme Corp bill, end to end.",
    domain=DOMAIN,
    params={"company": "Acme Corp"},
)
"""Phrased as a goal, not as the steps - so no single stored skill is retrieved for it."""


def test_a_composite_task_is_solved_from_two_skills_with_exactly_one_model_call(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Neither planted skill covers the task; together, in the right order, they do.

    The model is consulted ONCE, to name the two skills. Everything after that is the
    warm path again: the graph supplies the click between them, and the skills supply
    the rest. One call for the whole task, not one per action.
    """
    store.put(SEARCH_INVOICE)
    store.put(CONFIRM_PAYMENT)
    llm = FakeLLM([_chain(("search_invoice", {"company": "Acme Corp"}), ("confirm_payment", {}))])
    planner = build(scenario, store, graph, llm, compose=True)

    assert SkillRetriever(store).search(COMPOSITE_TASK.text, domain=DOMAIN) == [], (
        "the premise: no single stored skill is retrieved for this task"
    )

    outcome = planner.attempt(COMPOSITE_TASK, look(scenario))

    assert outcome is not None and outcome.ok
    assert scenario.solved
    assert llm.calls == 1, "one call for the decomposition, and not one more"
    assert outcome.spend.llm_calls == 1
    assert outcome.skill_used == "search_invoice+confirm_payment"
    # The click between the two skills is the graph's, not the model's.
    assert performed(scenario) == [TYPE_ACME, CLICK_ROW, CLICK_CONFIRM]


def test_the_composer_is_given_the_real_signatures_of_skills_that_exist(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """The prompt names every stored skill with its true call signature, so a model
    that follows instructions cannot help but answer with names that resolve."""
    store.put(SEARCH_INVOICE)
    store.put(CONFIRM_PAYMENT)
    llm = FakeLLM([_chain(("search_invoice", {"company": "Acme Corp"}), ("confirm_payment", {}))])
    build(scenario, store, graph, llm, compose=True).attempt(COMPOSITE_TASK, look(scenario))

    prompt = llm.requests[0].messages[0].text
    assert "`search_invoice(company: str)`" in prompt
    assert "`confirm_payment()`" in prompt
    assert "Acme Corp" in prompt
    # Screens are named, not hashed: a fingerprint in a prompt is noise a model
    # cannot reason about, and invites it to treat the hash as meaningful.
    assert "Starts on the `open invoice` screen." in prompt
    assert "On the `invoice list` screen." in prompt
    assert SELECTED.value not in prompt and LIST.value not in prompt


def test_a_decomposition_naming_a_skill_that_does_not_exist_is_rejected_before_execution(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """The model proposes a real skill followed by an invented one.

    The whole proposal is refused and NOTHING is performed - not even the first step,
    which happens to be valid. Half a plan executed on a real screen leaves the world
    somewhere nobody planned for, and the agent is better off exploring from a screen
    it still recognizes.
    """
    store.put(SEARCH_INVOICE)
    store.put(CONFIRM_PAYMENT)
    llm = FakeLLM(
        [_chain(("search_invoice", {"company": "Acme Corp"}), ("archive_invoice", {"id": "1042"}))]
    )
    planner = build(scenario, store, graph, llm, compose=True)

    assert planner.attempt(COMPOSITE_TASK, look(scenario)) is None

    assert performed(scenario) == [], "the valid first step must not run either"
    assert scenario.controller.state == "list"
    assert llm.calls == 1

    failure = planner.last_failure
    assert failure is not None and failure.stage == "no_decomposition"
    assert failure.performed_nothing
    assert "archive_invoice" in failure.reason
    assert "search_invoice" in failure.reason, "the reason lists what does exist"


def test_a_decomposition_with_arguments_a_skill_does_not_take_is_rejected(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """An undeclared argument would be a ``TypeError`` inside the sandbox, recorded
    against a skill that did nothing wrong. It is caught before anything runs."""
    store.put(SEARCH_INVOICE)
    llm = FakeLLM([_chain(("search_invoice", {"customer": "Acme Corp"}))])
    planner = build(scenario, store, graph, llm, compose=True)

    assert planner.attempt(COMPOSITE_TASK, look(scenario)) is None
    assert performed(scenario) == []
    assert planner.last_failure is not None
    assert "'customer'" in planner.last_failure.reason


def test_a_chain_longer_than_the_cap_is_refused(
    scenario: Scenario, store: InMemorySkillStore
) -> None:
    store.put(SEARCH_INVOICE)
    llm = FakeLLM([_chain(*(("search_invoice", {"company": "Acme Corp"}),) * 3)])
    composer = Composer(llm, store, max_steps=2)

    result = composer.compose(COMPOSITE_TASK, look(scenario))

    assert isinstance(result, Decomposition) and not result
    assert result.rejected is not None and "at most 2" in result.rejected


def test_a_composer_that_declines_leaves_the_planner_with_nothing(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """An empty answer is a correct answer, and it must not be dressed up as a plan."""
    store.put(SEARCH_INVOICE)
    llm = FakeLLM(['{"steps": [], "why": "nothing here archives an invoice"}'])
    planner = build(scenario, store, graph, llm, compose=True)

    assert planner.plan(COMPOSITE_TASK, look(scenario)) is None
    assert performed(scenario) == []
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "no_decomposition"


def test_without_a_composer_the_planner_cannot_reach_a_model_at_all(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    store.put(SEARCH_INVOICE)
    store.put(CONFIRM_PAYMENT)
    planner = build(scenario, store, graph, fake_llm, compose=False)

    assert planner.attempt(COMPOSITE_TASK, look(scenario)) is None
    assert fake_llm.calls == 0
    assert performed(scenario) == []


# -- a skill that stopped working ------------------------------------------------------------


def test_a_skill_that_fails_on_the_warm_path_is_demoted_with_a_reason(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The site changed and the stored skill no longer works.

    It is retired from retrieval with the reason it gave, and the planner returns
    ``None`` so the caller explores - which is how the library heals: the explorer
    solves it the slow way and the synthesizer stores a replacement version.
    """
    store.put(BROKEN_PAY)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    assert planner.attempt(scenario.task, look(scenario)) is None

    demoted = store.get("pay_invoice", DOMAIN)
    assert demoted.demoted_reason is not None
    assert "the confirmation dialog never appeared" in demoted.demoted_reason
    assert store.list(domain=DOMAIN) == [], "a demoted skill is no longer offered"
    assert SkillRetriever(store).search(scenario.task.text, domain=DOMAIN) == []

    failure = planner.last_failure
    assert failure is not None and failure.stage == "skill_failed"
    assert failure.skill == "pay_invoice" and failure.demoted
    assert not failure.performed_nothing
    assert failure.trace, "the explorer is handed the failed skill's trace"
    assert fake_llm.calls == 0


def test_a_clean_run_the_critic_judges_incomplete_is_reported_but_not_demoted(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """``search_invoice`` does exactly what it promises; the errand is still unfinished.

    The critic judges the TASK, not the skill, so its "no" is not evidence the skill
    is broken. Demoting here would empty a working library one composite task at a
    time.
    """
    store.put(SEARCH_INVOICE)
    critic = FakeCritic([Verdict(False, "the invoice is not paid yet", 0.9)])
    planner = build(scenario, store, graph, fake_llm, critic=critic)
    task = TaskSpec(
        text="Filter the invoice list to one company.", domain=DOMAIN, params={"company": "Acme"}
    )

    assert planner.attempt(task, look(scenario)) is None

    assert store.get("search_invoice", DOMAIN).demoted_reason is None
    assert store.list(domain=DOMAIN) != []
    failure = planner.last_failure
    assert failure is not None and failure.stage == "rejected"
    assert not failure.demoted
    assert failure.reason == "the invoice is not paid yet"


def test_a_repaired_version_makes_the_warm_path_work_again(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The whole point of demoting rather than deleting: the library heals.

    A broken skill is retired, the explorer solves the task the slow way, the
    synthesizer stores a fixed version - and the next run is warm again, at zero model
    calls. This test plays the last step of that cycle.
    """
    store.put(BROKEN_PAY)
    planner = build(scenario, store, graph, fake_llm)

    assert planner.attempt(scenario.task, look(scenario)) is None
    assert store.get("pay_invoice", DOMAIN).demoted_reason is not None

    store.put(PAY_INVOICE)  # what the synthesizer would write after exploring
    scenario.controller.reset()
    outcome = planner.attempt(scenario.task, look(scenario))

    assert outcome is not None and outcome.ok
    assert outcome.skill_used == "pay_invoice"
    assert store.get("pay_invoice", DOMAIN).demoted_reason is None
    assert scenario.solved
    assert fake_llm.calls == 0


# -- a plain-English repeat, still at zero model calls -------------------------------------------
#
# The warm path costs nothing when the caller supplies ``TaskSpec.params``. Asked in
# plain English it used to cost one model call, because reading a sentence was the
# composer's job. These tests cover the narrow case where the sentence does not have
# to be reasoned about: the skill remembers the sentence it was LEARNED from, and the
# new task is that sentence with one thing changed.
#
# Most of them are about declining. A wrong argument is far worse than a model call -
# it runs real clicks on a real screen - so every test below that asserts ``None`` is
# asserting that the planner preferred to pay rather than guess.


def learned_from(skill: Skill, sentence: str) -> Skill:
    """``skill`` as if it had been synthesized from the run that solved ``sentence``."""
    return replace(skill, provenance=replace(PROVENANCE, task_text=sentence))


PAY_GLOBEX = learned_from(PAY_INVOICE, "Pay the invoice for Globex Industries")
"""``pay_invoice`` as the library would hold it after learning it on another company."""

PAY_ACME_IN_ENGLISH = TaskSpec(text="Pay the invoice for Acme Corp", domain=DOMAIN)
"""The same errand, asked in words, for a DIFFERENT company, with no params at all."""


def test_a_plain_english_repeat_with_a_new_value_costs_zero_model_calls(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE RESULT THIS PATH EXISTS FOR.

    The library learned this errand for Globex; it is now asked for Acme, in a
    sentence, with no parameters supplied. The two sentences differ in exactly one
    place and that place is the argument, so the whole run happens with the composer
    wired in and never called: the model's count is EXACTLY ZERO, the same as if the
    caller had passed ``company="Acme Corp"`` themselves.
    """
    store.put(PAY_GLOBEX)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    outcome = planner.attempt(PAY_ACME_IN_ENGLISH, look(scenario))

    assert outcome is not None and outcome.ok
    assert scenario.solved, "the fake app did not actually reach the confirmation page"
    assert fake_llm.calls == 0
    assert outcome.spend.llm_calls == 0
    assert outcome.skill_used == "pay_invoice"
    # The value really came from the sentence: Acme was typed, not the learned Globex.
    assert performed(scenario) == [TYPE_ACME, CLICK_ROW, CLICK_CONFIRM]


def test_the_argument_is_the_span_the_two_sentences_disagree_on(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The plan names the bound value, and building it performs nothing."""
    store.put(PAY_GLOBEX)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    plan = planner.plan(PAY_ACME_IN_ENGLISH, look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Acme Corp"}),)
    assert fake_llm.calls == 0
    assert performed(scenario) == []


def test_every_binding_taken_from_a_sentence_is_logged_with_both_sentences(
    scenario: Scenario,
    store: InMemorySkillStore,
    graph: InMemorySiteGraph,
    fake_llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An argument this code invented has to be findable in one look.

    The log line carries the sentence the skill was learned from, the sentence that
    was asked, and the value derived - which is everything needed to see that a
    binding was wrong, without reproducing the run.
    """
    store.put(PAY_GLOBEX)
    planner = build(scenario, store, graph, fake_llm)

    with caplog.at_level(logging.INFO, logger="skillweaver.agent.planner"):
        planner.plan(PAY_ACME_IN_ENGLISH, look(scenario))

    lines = [r.getMessage() for r in caplog.records if "planner.bound_from_text" in r.getMessage()]
    assert len(lines) == 1
    assert 'learned="Pay the invoice for Globex Industries"' in lines[0]
    assert 'task="Pay the invoice for Acme Corp"' in lines[0]
    assert 'value="Acme Corp"' in lines[0]
    assert "param=company" in lines[0]


def test_a_text_bound_skill_runs_behind_the_same_checks_as_any_other(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Binding from a sentence removes a model call, not a guarantee.

    The skill is the broken one, reached by a plain-English task whose argument had to
    be read out of the wording. It is still routed to its precondition, still run in
    the sandbox, still fails its own ``ctx.expect`` - and is still demoted with the
    reason, exactly as when the caller supplied the arguments.
    """
    store.put(learned_from(BROKEN_PAY, "Pay the invoice for Globex Industries"))
    planner = build(scenario, store, graph, fake_llm)

    assert planner.attempt(PAY_ACME_IN_ENGLISH, look(scenario)) is None

    demoted = store.get("pay_invoice", DOMAIN)
    assert demoted.demoted_reason is not None
    assert "the confirmation dialog never appeared" in demoted.demoted_reason
    failure = planner.last_failure
    assert failure is not None and failure.stage == "skill_failed" and failure.demoted
    assert fake_llm.calls == 0


# -- the cases it refuses ---------------------------------------------------------------------


def declines(
    scenario: Scenario,
    store: InMemorySkillStore,
    graph: InMemorySiteGraph,
    fake_llm: FakeLLM,
    skill: Skill,
    text: str,
) -> PlanFailure:
    """Plan ``text`` against ``skill`` with NO composer, and assert it was declined.

    With no composer the planner cannot reach a model even in principle, so ``None``
    here can only mean the sentence alignment refused to bind.
    """
    store.put(skill)
    planner = build(scenario, store, graph, fake_llm, compose=False)

    assert planner.plan(TaskSpec(text=text, domain=DOMAIN), look(scenario)) is None
    assert performed(scenario) == []
    assert fake_llm.calls == 0
    failure = planner.last_failure
    assert failure is not None and failure.performed_nothing
    return failure


def test_a_span_whose_neighbour_is_an_ordinary_word_is_refused(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE CASE THAT MAKES A NAIVE ALIGNMENT WRONG.

    "Pay the Globex Corp invoice" against the learned "Pay the Acme Corp invoice"
    agrees on the trailing "Corp invoice", so the span left over is "Globex" - and
    the company is "Globex Corp". The alignment is refused because "Corp" is an
    ordinary word that could just as well belong to the value, and a sentence this
    code cannot take apart unambiguously is a sentence for the composer to read.
    """
    failure = declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay the Acme Corp invoice"),
        "Pay the Globex Corp invoice",
    )
    assert failure.stage == "unbindable_args"
    assert "Pay the Acme Corp invoice" in failure.reason, "the reason names the learned sentence"


def test_two_differing_spans_are_not_one_argument(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The sentences differ in two places, and the alignment sees one contiguous
    span holding both. The shared word inside it gives the pretence away."""
    failure = declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay the invoice for Acme on Monday"),
        "Pay the invoice for Globex on Tuesday",
    )
    assert failure.stage == "unbindable_args"


def test_a_sentence_that_only_adds_words_supplies_no_argument(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Nothing was substituted - the new task asks for MORE. That is a different
    errand, not a new value, and the empty span on the learned side says so."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay the invoice for Acme Corp"),
        "Pay the invoice for Acme Corp and archive it",
    )


def test_the_identical_sentence_supplies_no_argument_when_nothing_marks_the_value(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The value is in there somewhere and nothing says which words it is.

    Neither the learned sentence nor the parameter's description marks a slot, so the
    template path has nothing to work from either and the planner still declines. The
    two tests below are the same repeat with a slot to read.
    """
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay the invoice for Acme Corp"),
        "Pay the invoice for Acme Corp",
    )


def test_a_sentence_of_a_different_shape_is_refused(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Two sentences that barely agree are not the same sentence with one edit -
    whether this skill fits at all is retrieval's question and the composer's, and
    answering it here would be this code deciding something it cannot see."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay the invoice for Acme Corp"),
        "Settle up with Globex Industries",
    )


TWO_PARAMS = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company, reference):
    ctx.ctl.type_text(company)
    return reference
""",
    precondition=LIST,
    params={"company": {"type": "string"}, "reference": {"type": "string"}},
    docstring="Searches the invoice list for the company and confirms the referenced invoice.",
)


def test_two_missing_parameters_are_never_split_out_of_one_span(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """One difference cannot be two arguments, and deciding which part is which is
    exactly the reasoning this path exists to avoid."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(TWO_PARAMS, "Pay the invoice for Acme Corp"),
        "Pay the invoice for Globex Industries",
    )


WITH_DEFAULT = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company, note="paid"):
    ctx.ctl.type_text(company)
    return note
""",
    precondition=LIST,
    params={"company": {"type": "string"}, "note": {"type": "string", "default": "paid"}},
    docstring="Searches the invoice list for the company, opens its invoice and confirms payment.",
)


def test_a_parameter_with_a_default_is_left_alone(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A defaulted parameter is not missing - omitting it already works - so it is
    neither counted as a second unbound parameter nor filled in from the sentence."""
    store.put(learned_from(WITH_DEFAULT, "Pay the invoice for Globex Industries"))
    planner = build(scenario, store, graph, fake_llm)

    plan = planner.plan(PAY_ACME_IN_ENGLISH, look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Acme Corp"}),)


SET_QUANTITY = plant(
    "set_quantity",
    "Set the quantity on the invoice list.",
    """
def run(ctx, quantity):
    return quantity
""",
    precondition=LIST,
    params={"quantity": {"type": "integer"}},
    docstring="Sets the quantity field on the invoice list to the given whole number.",
)


def test_a_span_is_bound_as_the_type_the_skill_declares(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A declared ``integer`` arrives as an ``int``, not as the text ``"12"``."""
    store.put(learned_from(SET_QUANTITY, "Set the quantity to 3"))
    planner = build(scenario, store, graph, fake_llm)

    plan = planner.plan(TaskSpec(text="Set the quantity to 12", domain=DOMAIN), look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("set_quantity", DOMAIN, {"quantity": 12}),)


def test_a_span_that_cannot_be_the_declared_type_is_not_a_binding(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The sentences line up perfectly and the span is still not an integer."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(SET_QUANTITY, "Set the quantity to 3"),
        "Set the quantity to twelve",
    )


def test_the_composer_still_gets_the_sentences_this_path_declines(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Declining is a fall-through, not a failure: the composer reads the sentence
    and the run still happens, at the one model call it always cost."""
    store.put(learned_from(PAY_INVOICE, "Pay the Acme Corp invoice"))
    llm = FakeLLM([_chain(("pay_invoice", {"company": "Acme Corp"}))])
    planner = build(scenario, store, graph, llm, compose=True)

    outcome = planner.attempt(
        TaskSpec(text="Pay the Globex Corp invoice", domain=DOMAIN), look(scenario)
    )

    assert outcome is not None and outcome.ok
    assert llm.calls == 1
    assert outcome.spend.llm_calls == 1


# -- the slot the learned sentence left ----------------------------------------------------------
#
# Measured on the live Wikipedia suite, 2026-09-19: four of six warm runs explored
# although the right skill was stored, and every one of them was retrieval's TOP hit
# discarded at ``unbindable_args``. The diff above is blind in exactly two shapes,
# and between them they are all four runs:
#
#   * the sentence is repeated word for word, so nothing changed to attribute;
#   * two things changed and only one is the argument ("her article" -> "the article").
#
# So the question is turned round: not "what changed?" but "where did the value sit?".
# These tests are that path, and - as above - most of them are about declining.


QUOTED_PAY = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company):
    ctx.ctl.type_text(company)
    rows = ctx.see.find_text(company, fuzzy=False)
    ctx.expect(bool(rows), "the search returned no row for " + company)
    ctx.ctl.click(rows[0])
    buttons = ctx.see.find_text("Confirm payment", fuzzy=False)
    ctx.expect(bool(buttons), "no Confirm payment button on the invoice")
    ctx.ctl.click(buttons[0])
    return "confirmed"
""",
    precondition=LIST,
    params={"company": {"type": "string"}},
    docstring="Searches the invoice list for the company, opens its invoice and confirms payment.",
)
"""``pay_invoice`` again; the tests below give it a learned sentence that quotes."""

DESCRIBED_PAY = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    QUOTED_PAY.code,
    precondition=LIST,
    params={
        "company": {
            "type": "string",
            "description": "The company whose invoice to pay, e.g. 'Globex Industries'.",
        }
    },
    docstring="Searches the invoice list for the company, opens its invoice and confirms payment.",
)
"""A skill whose parameter description carries the example the synthesizer saw."""


def test_a_word_for_word_repeat_of_a_known_task_costs_zero_model_calls(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE SINGLE BIGGEST CAUSE OF THE LIVE MISSES.

    Asking for a stored task in the very words it was learned from is the strongest
    evidence a library can get, and it used to be the one thing the binder could not
    read: no difference between the sentences meant no span, so the best possible
    match was declined. Here the learned sentence quotes its value, so the repeat
    replays it - and the composer, wired in and scripted empty, is never called.
    """
    store.put(learned_from(QUOTED_PAY, 'Pay the invoice for "Acme Corp"'))
    planner = build(scenario, store, graph, fake_llm, compose=True)
    task = TaskSpec(text='Pay the invoice for "Acme Corp"', domain=DOMAIN)

    outcome = planner.attempt(task, look(scenario))

    assert outcome is not None and outcome.ok
    assert scenario.solved
    assert fake_llm.calls == 0
    assert outcome.spend.llm_calls == 0
    assert outcome.skill_used == "pay_invoice"
    assert performed(scenario) == [TYPE_ACME, CLICK_ROW, CLICK_CONFIRM]


def test_a_repeat_is_the_same_sentence_through_case_and_punctuation(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A person retyping a task does not reproduce its capitalisation or its full
    stop, and neither difference changes what was asked for."""
    store.put(learned_from(QUOTED_PAY, 'Pay the invoice for "Acme Corp".'))
    planner = build(scenario, store, graph, fake_llm)

    plan = planner.plan(
        TaskSpec(text='pay the invoice for "Acme Corp".', domain=DOMAIN), look(scenario)
    )

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Acme Corp"}),)


def test_a_quoted_value_binds_although_the_rest_of_the_sentence_also_moved(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE OTHER HALF OF THE LIVE MISSES, in miniature.

    The live sentence pair was *Search Wikipedia for "Ada Lovelace" and open HER
    article* against *... for "Photosynthesis" and open THE article*: two differences,
    of which one is the argument and the other is English. The diff refuses both,
    correctly, because from its side they are indistinguishable. The quotes say which
    is which, and the words around the slot still line up, so this binds.
    """
    store.put(learned_from(QUOTED_PAY, 'Pay "Globex Industries" and close her invoice'))
    planner = build(scenario, store, graph, fake_llm)

    plan = planner.plan(
        TaskSpec(text='Pay "Acme Corp" and close the invoice', domain=DOMAIN), look(scenario)
    )

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Acme Corp"}),)
    assert fake_llm.calls == 0


def test_an_example_in_the_parameter_description_locates_an_unquoted_value(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Nobody quotes every value, so there is a second way to find the slot.

    The synthesizer writes the parameter's description from the run it just watched,
    and the example it puts there is the value that run used. That is only believed
    when the example actually occurs in the learned sentence - which is the proof it
    WAS the value - and then a word-for-word repeat can replay it.
    """
    store.put(learned_from(DESCRIBED_PAY, "Pay the invoice for Globex Industries, please"))
    planner = build(scenario, store, graph, fake_llm)

    plan = planner.plan(
        TaskSpec(text="Pay the invoice for Globex Industries, please", domain=DOMAIN),
        look(scenario),
    )

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company": "Globex Industries"}),)
    assert fake_llm.calls == 0


def test_an_example_the_learned_sentence_never_contained_is_ignored(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The example is evidence only because the sentence contains it.

    Here the skill was learned from a sentence about Acme while its description still
    says ``e.g. 'Globex Industries'``. Nothing in that pair shows where the value sat,
    so no slot is found and the repeat is declined rather than answered with the
    description's example.
    """
    failure = declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(DESCRIBED_PAY, "Pay the invoice for Acme Corp"),
        "Pay the invoice for Acme Corp",
    )
    assert failure.stage == "unbindable_args"


def test_a_quoted_span_the_description_contradicts_is_refused(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Quotes around something that is NOT the argument must not become the argument.

    The learned sentence quotes a screen name while the parameter's description names
    a different value, so the two sources disagree about the slot and the planner
    declines rather than pick one.
    """
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(DESCRIBED_PAY, 'Open the "Unpaid" tab and pay Globex Industries'),
        'Open the "Unpaid" tab and pay Globex Industries',
    )


def test_a_sentence_that_grew_a_second_errand_is_not_one_skill(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE GUARD THAT KEEPS THIS FROM BEING THE OBVIOUS WRONG FIX.

    *Pay "Acme Corp", then archive it and email the receipt* quotes exactly one thing
    and starts exactly like the learned sentence, so a rule that only looked at the
    quotes would happily run a one-skill plan for a three-part errand. The sentence
    grew a clause after the value, so this is refused and left to the composer, which
    is what the live composite task needs.
    """
    failure = declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(QUOTED_PAY, 'Pay "Globex Industries" and close her invoice'),
        'Pay "Acme Corp" and close the invoice, then archive it and email the receipt',
    )
    assert failure.stage == "unbindable_args"


def test_a_differently_shaped_sentence_that_happens_to_quote_is_refused(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The words leading up to the value have to be the same words. A quoted name in
    a sentence about something else is a different errand, not a new argument."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(QUOTED_PAY, 'Pay the invoice for "Globex Industries"'),
        'Export the address book for "Acme Corp"',
    )


def test_two_quoted_spans_do_not_say_which_one_is_the_argument(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """One quoted thing in a one-parameter errand is the parameter. Two is a guess."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(QUOTED_PAY, 'Pay the invoice for "Globex Industries"'),
        'Pay the invoice for "Acme Corp" in the "Unpaid" tab',
    )


def test_a_possessive_apostrophe_does_not_open_a_quotation(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """``Wikipedia's`` is not a quote, and a rule that thought it was would find a
    slot in half the sentences a person writes."""
    declines(
        scenario,
        store,
        graph,
        fake_llm,
        learned_from(PAY_INVOICE, "Pay Acme's outstanding invoice"),
        "Pay Acme's outstanding invoice",
    )


# -- the caller's name for a parameter and the skill's ----------------------------------------


ALIASED_PAY = plant(
    "pay_invoice",
    "Confirm payment of a company's invoice from the invoice list.",
    """
def run(ctx, company_name):
    ctx.ctl.type_text(company_name)
    return company_name
""",
    precondition=LIST,
    params={"company_name": {"type": "string"}},
    docstring="Searches the invoice list for the company and confirms its payment.",
)


def test_a_caller_who_named_the_parameter_differently_is_still_understood(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The caller names a parameter and the model that wrote the skill names it again,
    independently, and they routinely disagree about one word.

    On the live suite the caller passed ``link`` where the stored skill declared
    ``link_title``, and that alone cost the task its warm path. One name's words being
    a subset of the other's is enough to line them up.
    """
    store.put(ALIASED_PAY)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Pay the invoice.", domain=DOMAIN, params={"company": "Acme Corp"})

    plan = planner.plan(task, look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("pay_invoice", DOMAIN, {"company_name": "Acme Corp"}),)
    assert fake_llm.calls == 0


def test_an_exactly_named_parameter_is_never_stolen_by_an_alias(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """``section`` and ``parent_section`` both look like a supplied ``section``.

    Exact names are bound first and the key is then spent, so the defaulted parameter
    falls back to its default instead of being handed the same value a second time.
    """
    two_sections = plant(
        "jump_to_section",
        "Jump to a section of the open invoice.",
        """
def run(ctx, section, parent_section=""):
    return parent_section + "/" + section
""",
        precondition=LIST,
        params={"section": {"type": "string"}, "parent_section": {"type": "string", "default": ""}},
        docstring="Jumps to a named section, optionally nested under a parent section.",
    )
    store.put(two_sections)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Jump to a section.", domain=DOMAIN, params={"section": "Childhood"})

    plan = planner.plan(task, look(scenario))

    assert plan is not None
    assert plan.steps == (SkillCall("jump_to_section", DOMAIN, {"section": "Childhood"}),)


def test_an_ambiguous_parameter_name_is_refused(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Two supplied keys that both read as the declared parameter are a guess, and a
    guessed argument runs real clicks on a real screen."""
    store.put(ALIASED_PAY)
    planner = build(scenario, store, graph, fake_llm, compose=False)
    task = TaskSpec(
        text="Pay the invoice.",
        domain=DOMAIN,
        params={"company": "Acme Corp", "name": "Globex Industries"},
    )

    assert planner.plan(task, look(scenario)) is None
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "unbindable_args"


# -- still saying no -------------------------------------------------------------------------


def test_an_unrelated_task_is_still_declined_with_every_binder_in_play(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """THE PRICE OF THE FIX, CHECKED.

    Making the library easier to reach is only worth anything if it did not become a
    library that answers everything. The store holds three skills, two of them with a
    quoted learned sentence and a described parameter - every new route to an argument
    - and a task none of them does is still declined with the screen untouched.
    """
    store.put(learned_from(QUOTED_PAY, 'Pay the invoice for "Globex Industries"'))
    store.put(learned_from(DESCRIBED_PAY, "Pay the invoice for Globex Industries"))
    store.put(CONFIRM_PAYMENT)
    planner = build(scenario, store, graph, fake_llm, compose=False)
    task = TaskSpec(text='Export the address book as a "CSV" file.', domain=DOMAIN)

    assert planner.attempt(task, look(scenario)) is None
    assert performed(scenario) == []
    assert scenario.controller.state == "list"
    assert fake_llm.calls == 0
    failure = planner.last_failure
    assert failure is not None and failure.performed_nothing


def test_a_warm_attempt_that_explores_records_what_it_was_offered(
    scenario: Scenario,
    store: InMemorySkillStore,
    graph: InMemorySiteGraph,
    fake_llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WHAT MADE THE LIVE MISSES TAKE A DAY TO EXPLAIN.

    A run that reported ``skill_used=None`` said nothing about whether the library had
    been consulted, so "never offered" and "offered and discarded" looked identical in
    the report. Both questions are now answered in one line and on the failure itself:
    what the domain held, what each candidate scored, and the rule that turned it down.
    """
    store.put(TWO_PARAMS)
    planner = build(scenario, store, graph, fake_llm, compose=False)
    task = TaskSpec(text="Pay the invoice for Acme Corp", domain=DOMAIN)

    with caplog.at_level(logging.INFO, logger="skillweaver.agent.planner"):
        assert planner.attempt(task, look(scenario)) is None

    failure = planner.last_failure
    assert failure is not None and failure.stage == "unbindable_args"
    assert [r.skill for r in failure.rejected] == ["pay_invoice"]
    assert failure.rejected[0].stage == "unbindable_args"
    assert failure.rejected[0].score > 0.0

    line = next(r.getMessage() for r in caplog.records if "planner.miss" in r.getMessage())
    assert "library=pay_invoice" in line
    assert "pay_invoice=0.700" in line, "the score retrieval gave it is in the line"
    assert "pay_invoice (score 0.700) -> unbindable_args" in line


def test_the_calibration_gap_this_line_sits_in() -> None:
    """The measurement :data:`MIN_ACCOUNTED_FOR` was chosen from, pinned.

    Taken 2026-09-19 from the ordering suite's own library, built by running all
    twelve tasks cold against the sandbox app, then scoring every task against every
    stored skill offline. Only candidates that BIND are listed: the rest are rejected
    a step earlier and never reach this check.

    The point of pinning it is that the threshold must keep separating these two
    populations, not merely keep the one failing case out. A change that pushes a
    right-hand number below the line, or a left-hand number above it, has broken
    something whatever the case that prompted it does.
    """
    wrong_but_bindable = [
        0.42,  # order_appears_in_history  <- the finding: open_order_screen, ok=False
        0.24,  # order_with_address_and_tip <- the same skill, the same shape
        0.60,  # open_order_screen against filter_by_cuisine and search_for_a_dish
    ]
    right_and_bindable = [
        1.00,  # 6 tasks whose own skill was in the library and took its arguments
        0.90,  # place_an_order: its skill adds the dish and leaves "place" undone
    ]
    assert max(wrong_but_bindable) < MIN_ACCOUNTED_FOR < min(right_and_bindable)
    assert MIN_ACCOUNTED_FOR - max(wrong_but_bindable) >= 0.1, "too close to the wrong answers"
    assert min(right_and_bindable) - MIN_ACCOUNTED_FOR >= 0.1, "too close to the right ones"
