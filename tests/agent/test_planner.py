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

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from skillweaver.agent.compose import Composer, Decomposition
from skillweaver.agent.critic import CriticVerdict, TieredCritic
from skillweaver.agent.planner import PlanFailure, Planner
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
from tests.fakes.controller import FakeController
from tests.fakes.perception import FakePerceiver
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
    end_states: Any = None,
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
        end_states=end_states,
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


OPEN_ANYTHING = plant(
    "open_the_invoice_list",
    "Open the invoice list. Takes no arguments.",
    "def run(ctx):\n    return True\n",
    precondition=LIST,
    docstring="Opens the invoice list from anywhere.",
)
"""A general, argument-free skill - the kind that quietly matches every task.

It binds against anything, because there is nothing to bind, so a planner that ran
the first candidate it COULD invoke would run this one on its way past whatever the
task was really about.
"""


def test_a_better_skill_waiting_for_its_arguments_is_not_preempted_by_a_worse_one(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The failure this rule exists for, measured on the live sandbox.

    ``pay_invoice`` is what the task is about but its ``company`` is still inside the
    sentence; ``open_the_invoice_list`` needs no arguments and so binds instantly.
    Running the one that binds means performing the wrong errand on a real screen and
    then needing a critic to notice - slower than never having run it, and the user is
    left looking at something they did not ask for.

    With no composer wired, the right answer is to decline and say the arguments are
    the problem, so the explorer can take over.
    """
    store.put(PAY_INVOICE)
    store.put(OPEN_ANYTHING)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Confirm payment of the Acme Corp invoice.", domain=DOMAIN, params={})

    assert planner.attempt(task, look(scenario)) is None
    assert performed(scenario) == [], "nothing was performed, least of all the wrong skill"
    assert planner.last_failure is not None
    assert planner.last_failure.stage == "unbindable_args"
    assert fake_llm.calls == 0


def test_the_argument_free_skill_still_runs_when_it_is_the_best_match(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """The rule is about rank, not about arguments: a skill that needs none is run
    whenever nothing retrieval liked better is merely waiting to be told its own."""
    store.put(PAY_INVOICE)
    store.put(OPEN_ANYTHING)
    planner = build(scenario, store, graph, fake_llm)
    task = TaskSpec(text="Open the invoice list.", domain=DOMAIN, params={})

    plan = planner.plan(task, look(scenario))
    assert plan is not None
    assert plan.skills_used == ("open_the_invoice_list",)
    assert fake_llm.calls == 0


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


def test_the_start_screen_is_reached_by_its_url_when_the_graph_knows_no_way(
    scenario: Scenario, store: InMemorySkillStore, fake_llm: FakeLLM
) -> None:
    """A URL is an edge from anywhere, and the graph is sparse on a real site.

    The graph only holds edges somebody walked, so it is routinely silent about the
    commonest move there is - going back to the front page. That silence broke CHAINS
    rather than single skills: on a live shop the composer read "add these two things
    and show me the cart", planned add -> add -> open_cart correctly in ONE model call,
    and then the second skill could not start because nothing had recorded a way back
    from a product page to the shop's home screen. The whole chain fell through to
    exploration and spent 26 calls redoing what the library already knew.
    """
    source = scenario.controller
    controller = FakeController(
        source.states, source.transitions, start=source.state, viewport=source.viewport()
    )
    can_navigate = dataclasses.replace(
        scenario, controller=controller, perceiver=FakePerceiver.for_controller(controller)
    )
    store.put(SEARCH_INVOICE)
    store.put(CONFIRM_PAYMENT)
    # Both screens are KNOWN, with addresses - but the graph holds no edge between
    # them, exactly as a real site's graph is silent about the way back to a page it
    # has only ever arrived at once.
    graph = InMemorySiteGraph()
    start = look(can_navigate)
    for fingerprint, url in (
        (start.fingerprint, "https://fake.test/invoices"),
        (CONFIRM_PAYMENT.precondition, "https://fake.test/invoices/1042"),
    ):
        graph.upsert_state(UIState(fingerprint=fingerprint, domain=DOMAIN, url_pattern=url))
    assert start.fingerprint != CONFIRM_PAYMENT.precondition
    llm = FakeLLM([_chain(("search_invoice", {"company": "Acme Corp"}), ("confirm_payment", {}))])
    planner = build(can_navigate, store, graph, llm, compose=True)

    planner.attempt(COMPOSITE_TASK, start)

    # The fake app has no navigate transition, so the errand does not finish here.
    # What is being proved is that the step between the two skills was ATTEMPTED at
    # all: before this, the chain stopped dead at "no known route".
    kinds = [a.kind for a in performed(can_navigate)]
    assert "navigate" in kinds, f"expected the chain to route by URL, performed {kinds}"


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


def test_a_skill_with_a_record_of_working_is_not_retired_for_one_failure(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """A failure means the skill broke OR that it was asked the wrong thing.

    The second really happens: retrieval ranks a skill that opens a message top for a
    task that opens a message AND sends one, its arguments bind, and it goes looking
    for the other half of the sentence. Retiring it then throws away something that
    works, and the task it IS about has to learn it again from scratch. A skill with a
    record of succeeding is spared, and the failure is still reported in full.
    """
    store.put(BROKEN_PAY)
    for _ in range(4):  # a history of working, before today
        store.record_run("pay_invoice", DOMAIN, ok=True, ms=30.0)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    assert planner.attempt(scenario.task, look(scenario)) is None

    assert store.get("pay_invoice", DOMAIN).demoted_reason is None
    assert [s.name for s in store.list(domain=DOMAIN)] == ["pay_invoice"]
    failure = planner.last_failure
    assert failure is not None and failure.stage == "skill_failed"
    assert not failure.demoted, "reported, not retired"
    assert failure.trace, "and the explorer still gets the trace"


def test_a_skill_that_fails_more_often_than_it_works_is_retired(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Sparing a skill is about weighing evidence, not about never retiring one: a
    site that changed makes a skill fail every time, and the record says so."""
    store.put(BROKEN_PAY)
    store.record_run("pay_invoice", DOMAIN, ok=True, ms=30.0)
    for _ in range(3):
        store.record_run("pay_invoice", DOMAIN, ok=False, ms=30.0)
    planner = build(scenario, store, graph, fake_llm, compose=True)

    assert planner.attempt(scenario.task, look(scenario)) is None

    assert store.get("pay_invoice", DOMAIN).demoted_reason is not None
    assert store.list(domain=DOMAIN) == []


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


def test_the_verdict_is_measured_against_the_skill_that_actually_ran(
    scenario: Scenario, store: InMemorySkillStore, graph: InMemorySiteGraph, fake_llm: FakeLLM
) -> None:
    """Which screen a warm run should finish on depends on which skill ran.

    The critic is built before anything runs, so its expected screen is a guess taken
    from whichever skill retrieval ranked first. With one skill in the library that
    guess is always right; with fourteen it is usually some other task's finishing
    screen, and a warm run that did its job exactly is rejected for landing in the
    wrong place. Once a plan has run, the skill is known, and the known thing wins.
    """
    store.put(PAY_INVOICE)
    asked: list[str] = []

    def end_states(name: str) -> Fingerprint:
        asked.append(name)
        return DONE  # where the run that taught pay_invoice really ended

    planner = build(
        scenario,
        store,
        graph,
        fake_llm,
        # Built expecting the WRONG screen, as it would be when another skill outranks
        # this one. Nothing else tells the planner otherwise.
        critic=TieredCritic(None, expected_state=LIST),
        end_states=end_states,
    )
    task = TaskSpec(
        text="Confirm payment of the Acme Corp invoice.",
        domain=DOMAIN,
        params={"company": "Acme Corp"},
    )

    outcome = planner.attempt(task, look(scenario))

    assert outcome is not None and outcome.ok, "judged against where THIS skill ends"
    assert asked == ["pay_invoice"]
    assert outcome.spend.llm_calls == 0, "and still without a model"
