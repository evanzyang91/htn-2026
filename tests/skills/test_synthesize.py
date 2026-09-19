"""The admission gate, which is the only reason this library is worth anything.

Every test here is a variation on one question: can code that does not work get into
the store? The answer has to be no - through a broken generation, through a repair
loop that gives up, through an import the sandbox refuses, through a critic that says
no. ``store.list() == []`` after a rejection is the assertion that matters.

Nothing here touches a browser, a model or a network: the fake invoicing app of
``tests/fakes/scenario.py`` is the environment, ``FakeLLM`` writes the skills and
``FakeCritic`` judges them. The critic is reached only through the ``Critic``
Protocol in ``contracts.py``, so the gate never needs to know which implementation
it has: scripting the verdicts here is what makes "the critic said no" a case the
tests can put on the table, and the real ``agent/critic.py`` was driven through the
same gate by hand (it admits this skill programmatically, with no model call).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from skillweaver.contracts import Fingerprint, LLMResponse, Trajectory, Verdict
from skillweaver.errors import SkillNotFound
from skillweaver.llm.cassette import CassetteClient
from skillweaver.skills.refactor import harden
from skillweaver.skills.synthesize import (
    ReplayEnvironment,
    Synthesizer,
    describe_trajectory,
    load_prompt,
)
from tests.fakes import (
    FakeCritic,
    FakeLLM,
    InMemorySkillStore,
    InMemoryTrajectoryRecorder,
    Scenario,
)

DOMAIN = "fake.test"

# The draft a model writes from the recording: correct in substance, and exactly as
# brittle as a first draft is - a URL it starts by typing, two raw coordinates, and
# the search term of this one run baked in.
GOOD_CODE = (
    "def run(ctx, company):\n"
    '    ctx.ctl.type_text("https://fake.test/invoices")\n'
    '    ctx.ctl.type_text("acme")\n'
    "    ctx.ctl.click(Point(400, 140))\n"
    "    ctx.ctl.click(Point(100, 140))\n"
    "    return True\n"
)

# Plausible, and wrong: it looks for a row that is not there, so it fails cleanly on
# a ctx.expect - which is precisely the kind of skill that must never be stored.
BROKEN_CODE = (
    "def run(ctx, company):\n"
    '    ctx.ctl.type_text("acme")\n'
    '    rows = ctx.see.find_text("Globex INV-9999", "row")\n'
    '    ctx.expect(bool(rows), "no Globex row on the filtered list")\n'
    "    ctx.ctl.click(rows[0])\n"
    "    return True\n"
)

VERIFIER = 'def verify(ctx, result):\n    return bool(ctx.see.find_text("Payment confirmed"))\n'


def reply(code: str = GOOD_CODE, **overrides: Any) -> str:
    """One model reply: the JSON object the prompt asks for, in a fenced block."""
    draft: dict[str, Any] = {
        "name": "confirm_invoice_payment",
        "summary": "Confirm payment of a company's invoice from the invoice list.",
        "docstring": (
            "Searches the invoice list for the company and confirms payment.\n\n"
            "Assumes: the invoice list is on screen with its search field focused.\n"
            "Ends on: the payment-confirmed page."
        ),
        "params": {"company": {"type": "string", "description": "Company whose invoice to pay."}},
        "example_args": {"company": "Acme Corp"},
        "requires": [],
        "code": code,
        "verifier_code": VERIFIER,
    }
    draft.update(overrides)
    return "Here is the skill:\n```json\n" + json.dumps(draft) + "\n```\n"


@pytest.fixture
def trajectory(scenario: Scenario) -> Trajectory:
    """The recorded solution of the fake app, replayed from its start state."""
    recorder = InMemoryTrajectoryRecorder()
    recorder.start(scenario.task.text, DOMAIN)
    for action in scenario.solution:
        before = scenario.perceiver.observe(scenario.controller)
        result = scenario.controller.perform(action)
        after = scenario.perceiver.observe(scenario.controller)
        recorder.step(action, before, after, result)
    assert scenario.solved
    finished = recorder.finish(ok=True, note="solved by exploration")
    scenario.controller.reset()
    return finished


@pytest.fixture
def environment(scenario: Scenario) -> Callable[[], ReplayEnvironment]:
    """A factory handing the gate the recorded app, put back to its first screen.

    ``restored=True`` because ``controller.reset()`` genuinely is a restore: this app
    keeps no state outside the state machine, so returning to ``list`` is returning
    to the seed. That flag is what entitles a precondition mismatch here to be read
    as a fact about the SKILL rather than about the harness.
    """

    def build() -> ReplayEnvironment:
        scenario.controller.reset()
        return ReplayEnvironment(scenario.controller, scenario.perceiver, restored=True)

    return build


@pytest.fixture
def critic(trajectory: Trajectory) -> FakeCritic:
    """Programmatic: the run is good exactly when it ends on the recorded goal screen."""
    return FakeCritic(goal=trajectory.steps[-1].after.fingerprint)


def synthesizer(llm: FakeLLM, store: InMemorySkillStore, critic: FakeCritic, **kwargs: Any):
    return Synthesizer(llm, store, critic, **kwargs)


# --------------------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------------------


def test_a_trajectory_and_a_cassette_yield_an_admitted_skill_that_replays_green(
    tmp_path, trajectory, environment, critic, skill_store
):
    """The whole path, with the model behind a cassette: record once, then replay
    with no model at all and still get the same admitted skill."""
    path = tmp_path / "synthesize_confirm_invoice.json"
    recorder = CassetteClient(path, mode="record", inner=FakeLLM([reply()]))
    first = synthesizer(recorder, skill_store, critic, max_repairs=0).admit(trajectory, environment)
    assert first.ok, first.reason
    assert path.is_file()

    replayed_store = InMemorySkillStore()
    replaying = CassetteClient(path, mode="replay")
    admission = synthesizer(
        replaying, replayed_store, FakeCritic(goal=trajectory.steps[-1].after.fingerprint)
    ).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert admission.skill is not None
    assert admission.attempts[-1].result is not None
    assert admission.attempts[-1].result.ok
    assert admission.attempts[-1].verdict == Verdict(True, "reached goal state", 1.0)

    stored = replayed_store.get("confirm_invoice_payment", DOMAIN)
    assert stored.version == 1
    assert stored == admission.skill
    assert [s.name for s in replayed_store.list()] == ["confirm_invoice_payment"]


def test_a_deliberately_broken_synthesis_is_rejected_and_is_absent_from_the_store(
    trajectory, environment, critic, skill_store
):
    """The test the whole approach rests on: code that does not work stays out."""
    llm = FakeLLM([reply(BROKEN_CODE)])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert not admission.ok
    assert admission.skill is None
    assert admission.attempts[-1].stage == "execution"
    assert "no Globex row on the filtered list" in (admission.attempts[-1].error or "")

    assert skill_store.list() == []
    assert skill_store.list(include_demoted=True) == []
    with pytest.raises(SkillNotFound):
        skill_store.get("confirm_invoice_payment", DOMAIN)


def test_a_skill_the_critic_rejects_is_not_stored(trajectory, environment, skill_store):
    """Running to completion is not enough: the critic has the last word."""
    llm = FakeLLM([reply()])
    refusing = FakeCritic([Verdict(False, "the invoice is still unpaid", 0.9, "model")])
    admission = synthesizer(llm, skill_store, refusing, max_repairs=0).admit(
        trajectory, environment
    )

    assert not admission.ok
    assert admission.attempts[-1].stage == "critic"
    assert admission.attempts[-1].result is not None and admission.attempts[-1].result.ok
    assert "the invoice is still unpaid" in (admission.attempts[-1].error or "")
    assert skill_store.list() == []


def test_generated_code_that_imports_is_refused_by_the_sandbox_and_never_admitted(
    trajectory, environment, critic, skill_store
):
    importing = 'def run(ctx, company):\n    import time\n    ctx.ctl.type_text("acme")\n'
    llm = FakeLLM([reply(importing)])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert not admission.ok
    assert admission.attempts[-1].stage == "sandbox"
    assert "may not import modules" in (admission.attempts[-1].error or "")
    assert admission.attempts[-1].result is None  # it never ran
    assert skill_store.list() == []


def test_a_skill_without_a_verifier_is_refused(trajectory, environment, critic, skill_store):
    """The prompt demands a verifier; the gate is what makes that a demand."""
    llm = FakeLLM([reply(verifier_code=None)])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert not admission.ok
    assert admission.attempts[-1].stage == "generation"
    assert "verifier" in (admission.attempts[-1].error or "")
    assert skill_store.list() == []


def test_a_candidate_is_refused_when_the_environment_is_not_on_the_recorded_screen(
    scenario, trajectory, critic, skill_store
):
    """A skill proved on the wrong screen proves nothing, so the gate will not try."""

    def wrong_screen() -> ReplayEnvironment:
        scenario.controller.reset()
        scenario.controller.perform(scenario.solution[0])  # one state further on
        return ReplayEnvironment(scenario.controller, scenario.perceiver, restored=True)

    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, wrong_screen)

    assert not admission.ok
    assert admission.attempts[-1].stage == "precondition"
    assert "not on the recorded starting screen" in (admission.attempts[-1].error or "")
    assert skill_store.list() == []


# --------------------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------------------


def test_a_failed_attempt_is_repaired_with_the_sandboxs_own_error_trace(
    trajectory, environment, critic, skill_store
):
    """One bad attempt, one good one - and the model is shown the trace of the first."""
    llm = FakeLLM([reply(BROKEN_CODE), reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=2).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert admission.repairs == 1
    assert [a.ok for a in admission.attempts] == [False, True]
    assert llm.calls == 2

    repair = llm.requests[1].messages[-1]
    assert repair.role == "user"
    assert "REJECTED" in repair.text
    assert "no Globex row on the filtered list" in repair.text
    # The sandbox's trace, verbatim: the actions it managed and the expectation it failed.
    assert "expect FAILED: no Globex row on the filtered list" in repair.text
    assert "type_text 'acme'" in repair.text
    # And the rejected draft is in the conversation, so the model repairs its own code.
    assert llm.requests[1].messages[-2].role == "assistant"
    assert "Globex INV-9999" in llm.requests[1].messages[-2].text

    assert skill_store.get("confirm_invoice_payment", DOMAIN).version == 1


def test_repairs_are_bounded_and_exhausting_them_is_a_clean_failure(
    trajectory, environment, critic, skill_store
):
    llm = FakeLLM([reply(BROKEN_CODE), reply(BROKEN_CODE)])
    admission = synthesizer(llm, skill_store, critic, max_repairs=1).admit(trajectory, environment)

    assert not admission.ok
    assert admission.skill is None
    assert len(admission.attempts) == 2  # the draft and one repair, and no more
    assert admission.repairs == 1
    assert llm.calls == 2 and llm.remaining == 0  # the bound held; nothing over-ran
    assert "1 repair(s) allowed" in admission.reason
    assert skill_store.list() == []


def test_max_repairs_zero_means_one_attempt(trajectory, environment, critic, skill_store):
    llm = FakeLLM([reply(BROKEN_CODE)])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)
    assert not admission.ok
    assert len(admission.attempts) == 1
    assert llm.calls == 1


def test_a_negative_repair_bound_is_refused(fake_llm, skill_store, fake_critic):
    with pytest.raises(ValueError, match="max_repairs"):
        Synthesizer(fake_llm, skill_store, fake_critic, max_repairs=-1)


# --------------------------------------------------------------------------------------
# A reply that could not be read
# --------------------------------------------------------------------------------------
#
# Every case here was found by running the real command line against the live API. Two
# of three admission attempts died reading the reply, the repair budget was spent on
# punctuation, and the run finished with nothing stored - so the project's whole claim,
# "learn it once and then repeat it for free", could not be demonstrated at all.


def test_a_skill_wrapped_in_prose_is_read_rather_than_thrown_away(
    trajectory, environment, critic, skill_store
):
    """The model explains itself on either side of the object; the object still counts.

    The braces in that prose are the sharp bit: reading from the first `{` to the last
    `}` swallows them and yields something that is not JSON, which is exactly how a
    perfectly good skill used to be reported as "the reply was not a JSON object".
    """
    chatty = (
        "Looking at the recording, the {company} value is what varied, so I lifted it "
        "into a parameter.\n\n"
        + json.dumps(json.loads(reply().split("```json\n")[1].split("\n```")[0]))
        + "\n\nNote the closing } above - that is the whole object."
    )
    llm = FakeLLM([chatty])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert llm.calls == 1, "a readable reply was re-asked for"
    assert skill_store.get("confirm_invoice_payment", DOMAIN).version == 1


def test_an_empty_reply_is_re_asked_for_and_does_not_spend_a_repair(
    trajectory, environment, critic, skill_store
):
    """THE ONE THAT COST A LIVE RUN ITS SKILL.

    The model returns nothing - on a thinking model, usually because the token cap went
    on thinking. That says nothing whatever about the skill, so charging it to the
    repair budget spends the gate's whole allowance on a reply that was never judged.
    Here the budget is ``max_repairs=0``: under the old behaviour the empty turn WAS
    the one and only attempt and the run ended with nothing. It is now a re-ask, the
    skill that follows is admitted, and ``repairs`` is still zero.
    """
    llm = FakeLLM(["", reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert admission.repairs == 0, "a reply that was never judged was charged as a repair"
    assert len(admission.attempts) == 1
    assert llm.calls == 2
    assert skill_store.get("confirm_invoice_payment", DOMAIN).version == 1


def test_an_empty_reply_is_re_asked_for_with_more_room_and_nothing_to_repair(
    trajectory, environment, critic, skill_store
):
    """The re-ask raises the token cap and does not accuse the model of anything.

    An empty turn means the reply ran out of room, so asking again with the same cap
    invites the same silence. And the follow-up must not say the skill was rejected:
    a model told to fix working code will change it.
    """
    llm = FakeLLM(["", reply()])
    synthesizer(llm, skill_store, critic, max_repairs=0, max_tokens=4000).admit(
        trajectory, environment
    )

    first, second = llm.requests
    assert first.max_tokens == 4000
    assert second.max_tokens == 8000, "the re-ask did not give the reply more room"
    # Nothing to quote back, so there is no assistant turn - and no claim of rejection.
    assert [m.role for m in second.messages] == ["user", "user"]
    assert "could not be read" in second.messages[-1].text
    assert "REJECTED" not in second.messages[-1].text


def test_a_reply_cut_off_at_the_token_cap_is_re_asked_for_with_more_room(
    trajectory, environment, critic, skill_store
):
    """A half-written object is a cap that was too small, not a skill that is wrong."""
    cut_off = LLMResponse(text=reply()[:120], stop_reason="max_tokens")
    llm = FakeLLM([cut_off, reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0, max_tokens=4000).admit(
        trajectory, environment
    )

    assert admission.ok, admission.reason
    assert admission.repairs == 0
    assert [r.max_tokens for r in llm.requests] == [4000, 8000]


def test_re_asks_are_bounded_and_the_last_one_is_reported_as_the_failure(
    trajectory, environment, critic, skill_store
):
    """Tolerance is not patience: a model that will not return an object is given up on."""
    llm = FakeLLM(["nope", "still nope"])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0, max_format_retries=1).admit(
        trajectory, environment
    )

    assert not admission.ok
    assert llm.calls == 2 and llm.remaining == 0
    assert admission.attempts[-1].stage == "generation"
    assert "no JSON object" in (admission.attempts[-1].error or "")
    assert skill_store.list() == []


def test_a_real_content_defect_still_spends_a_repair(trajectory, environment, critic, skill_store):
    """The line between the two failures, stated as a test.

    A JSON object with no verifier IS the model's mistake and IS worth a repair - it
    is told what is missing and asked again. Only a reply that could not be read at
    all escapes the budget. Getting this backwards would gut the gate: every rejection
    would become a free retry.
    """
    llm = FakeLLM([reply(verifier_code=None), reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=1, max_format_retries=2).admit(
        trajectory, environment
    )

    assert admission.ok, admission.reason
    assert admission.repairs == 1, "a content defect was excused as a formatting problem"
    assert llm.calls == 2
    assert "verifier" in (admission.attempts[0].error or "")
    assert "REJECTED" in llm.requests[1].messages[-1].text


def test_a_negative_re_ask_bound_is_refused(fake_llm, skill_store, fake_critic):
    with pytest.raises(ValueError, match="max_format_retries"):
        Synthesizer(fake_llm, skill_store, fake_critic, max_format_retries=-1)


# --------------------------------------------------------------------------------------
# A world that cannot be put back
# --------------------------------------------------------------------------------------


@pytest.fixture
def unrestorable(scenario: Scenario) -> Callable[[], ReplayEnvironment]:
    """The gate's world as a browser alone can offer it: re-opened, not put back.

    This is what ``navigating_environment`` does without a reset hook. The app is left
    standing where the recorded run left it - the payment is confirmed, and no amount
    of re-opening the page un-confirms it - and ``restored=False`` says so. Replaying
    the solution is idempotent: ``done`` is terminal, so a second call changes nothing.
    """

    def build() -> ReplayEnvironment:
        for action in scenario.solution:
            scenario.controller.perform(action)
        return ReplayEnvironment(scenario.controller, scenario.perceiver, restored=False)

    return build


def test_a_mutating_task_cannot_be_proved_without_a_way_to_put_the_world_back(
    trajectory, unrestorable, critic, skill_store
):
    """THE SECOND DEFECT, AT ITS SOURCE.

    The recorded run ended on the confirmation page; that is where the world now
    stands and re-opening a screen does not un-confirm a payment. The gate must not
    call this a bad skill - it never ran it - and it must not quietly repair its way
    through the budget trying to fix code that was never judged.
    """
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=2).admit(trajectory, unrestorable)

    assert not admission.ok
    assert admission.unproved, "a world that could not be restored was reported as a bad skill"
    assert admission.attempts[-1].stage == "reset"
    assert "could not restore the world" in (admission.attempts[-1].error or "")
    assert "could not restore the world" in admission.reason
    # No repair was attempted: there is nothing for the model to fix.
    assert len(admission.attempts) == 1
    assert llm.calls == 1 and llm.remaining == 0
    assert skill_store.list() == []


def test_the_same_task_is_admitted_once_the_world_can_be_put_back(
    trajectory, environment, critic, skill_store
):
    """The other half of the pair: one reset hook is the whole difference.

    Same trajectory, same model reply, same gate - and the skill is stored, because
    ``environment`` really does restore the app. Without this half the fix could be a
    weakened gate; with it, the gate is intact and the harness grew a hand.
    """
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert not admission.unproved
    assert skill_store.get("confirm_invoice_payment", DOMAIN).version == 1


def test_a_restored_world_on_the_wrong_screen_is_still_the_skills_problem(
    scenario, trajectory, critic, skill_store
):
    """``restored=True`` and a mismatch means what it always meant: wrong screen.

    The distinction only buys something if it stays narrow. A harness that really did
    put the world back and still finds the wrong screen has said something about the
    candidate, and that must keep reading ``precondition``.
    """

    def restored_but_elsewhere() -> ReplayEnvironment:
        scenario.controller.reset()
        scenario.controller.perform(scenario.solution[0])
        return ReplayEnvironment(scenario.controller, scenario.perceiver, restored=True)

    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(
        trajectory, restored_but_elsewhere
    )

    assert not admission.ok
    assert not admission.unproved
    assert admission.attempts[-1].stage == "precondition"
    assert "could not restore the world" not in (admission.attempts[-1].error or "")


# --------------------------------------------------------------------------------------
# The candidate itself
# --------------------------------------------------------------------------------------


def test_the_precondition_is_the_trajectorys_first_screen(
    trajectory, environment, critic, skill_store
):
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert admission.skill is not None
    assert admission.skill.precondition == trajectory.steps[0].before.fingerprint
    assert admission.skill.precondition != trajectory.steps[-1].after.fingerprint


def test_provenance_records_the_run_the_task_and_the_model(
    trajectory, environment, critic, skill_store
):
    llm = FakeLLM([reply()], model="fake-opus")
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.skill is not None
    provenance = admission.skill.provenance
    assert provenance.trajectory_id == trajectory.run_id
    assert provenance.task_text == trajectory.task
    assert provenance.model == "fake-opus"


def test_synthesize_writes_a_candidate_and_stores_nothing(trajectory, critic, skill_store):
    """The Protocol method is a draft shop, not a door into the library."""
    llm = FakeLLM([reply()])
    candidate = synthesizer(llm, skill_store, critic).synthesize(trajectory)

    assert candidate is not None
    assert candidate.version == 0
    assert candidate.name == "confirm_invoice_payment"
    assert skill_store.list() == []


def test_a_failed_or_trivial_run_is_not_worth_a_skill(trajectory, environment, skill_store):
    """No model is even consulted: ``fake_llm.calls == 0`` proves it."""
    llm = FakeLLM()
    from dataclasses import replace

    failed = replace(trajectory, ok=False)
    synth = synthesizer(llm, skill_store, FakeCritic(), max_repairs=0)
    assert synth.synthesize(failed) is None
    rejected = synth.admit(failed, environment)
    assert not rejected.ok and "did not succeed" in rejected.reason

    short = replace(trajectory, steps=trajectory.steps[:1])
    picky = synthesizer(llm, skill_store, FakeCritic(), min_steps=2)
    assert picky.synthesize(short) is None
    assert not picky.admit(short, environment).ok

    assert llm.calls == 0
    assert skill_store.list() == []


def test_an_unusable_reply_is_not_a_skill(trajectory, critic, skill_store):
    """A model that never returns an object is re-asked, and then given up on."""
    prose = "I could not work out what that run was doing."
    llm = FakeLLM([prose, prose, prose])
    synth = synthesizer(llm, skill_store, critic, max_format_retries=2)
    assert synth.synthesize(trajectory) is None
    assert llm.calls == 3 and llm.remaining == 0  # the first ask and two re-asks
    assert skill_store.list() == []


# --------------------------------------------------------------------------------------
# Hardening
# --------------------------------------------------------------------------------------


def test_hardening_removes_literal_coordinates_and_introduces_a_perception_lookup(
    trajectory, environment, critic, skill_store
):
    """Asserted on the source that was STORED, not on an intermediate value."""
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)
    assert admission.ok, admission.reason
    assert admission.skill is not None
    code = admission.skill.code

    # gone: every literal coordinate the model wrote
    assert "Point(" not in code
    assert "400" not in code and "140" not in code
    # arrived: a lookup for the element the recording shows at each of those points,
    # checked before it is used
    assert "ctx.see.find_text('Acme Corp INV-1042 $1,200.00', 'row')" in code
    assert "ctx.see.find_text('Confirm payment', 'button')" in code
    assert code.count("ctx.expect(bool(") == 2
    assert "ctx.ctl.click(confirm_payment_button[0])" in code

    hardening = admission.attempts[-1].hardening
    assert hardening is not None
    assert hardening.coordinates_replaced == 2
    assert hardening.lookups_added == 2


def test_hardening_lifts_navigation_out_of_the_body_into_the_precondition(
    trajectory, environment, critic, skill_store
):
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)
    assert admission.skill is not None

    assert "https://fake.test/invoices" not in admission.skill.code
    assert admission.attempts[-1].hardening is not None
    assert admission.attempts[-1].hardening.navigation_lifted == ("https://fake.test/invoices",)
    # Where the skill starts is stated once, as the precondition, not re-navigated to.
    assert admission.skill.precondition == trajectory.steps[0].before.fingerprint


def test_hardening_lifts_a_typed_literal_into_a_parameter_that_still_replays(
    trajectory, environment, critic, skill_store
):
    llm = FakeLLM([reply()])
    admission = synthesizer(llm, skill_store, critic, max_repairs=0).admit(trajectory, environment)
    assert admission.skill is not None
    skill = admission.skill

    lifted = [name for name in skill.params if name != "company"]
    assert lifted == ["search_invoices_text"]
    assert skill.params["search_invoices_text"]["default"] == "acme"
    assert '"acme"' not in skill.code and "'acme'" not in skill.code.split("\n")[1]
    assert "ctx.ctl.type_text(search_invoices_text)" in skill.code


def test_hardening_renames_opaque_locals_bound_to_a_lookup(trajectory):
    draft = (
        "def run(ctx):\n"
        '    e = ctx.see.find_text("Confirm payment", "button")\n'
        '    ctx.expect(bool(e), "no confirm button")\n'
        "    ctx.ctl.click(e[0])\n"
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.renamed == {"e": "confirm_payment"}
    assert "\n    e = " not in hardened.code
    assert "confirm_payment = ctx.see.find_text('Confirm payment', 'button')" in hardened.code
    assert "ctx.ctl.click(confirm_payment[0])" in hardened.code


def test_hardening_recognizes_every_shape_of_a_written_coordinate(trajectory):
    draft = (
        "def run(ctx):\n"
        "    ctx.ctl.click(Point(100, 140))\n"
        "    ctx.ctl.move((100, 140))\n"
        "    ctx.ctl.scroll([100, 140], dy=120)\n"
        "    ctx.ctl.click(Box(20, 120, 160, 40))\n"
        "    return True\n"
    )
    hardened = harden(draft, trajectory)

    assert hardened.coordinates_replaced == 4
    assert "Point(" not in hardened.code and "Box(" not in hardened.code
    assert "(100, 140)" not in hardened.code and "[100, 140]" not in hardened.code
    assert hardened.code.count("ctx.see.find_text('Confirm payment', 'button')") == 4


def test_hardening_leaves_a_coordinate_it_cannot_ground_alone(trajectory):
    """Honest about what it does not know: an unrecognizable point stays, and the
    admission gate is what then refuses the skill."""
    draft = "def run(ctx):\n    ctx.ctl.click(Point(5, 7))\n    return True\n"
    hardened = harden(draft, trajectory)

    assert hardened.coordinates_replaced == 0
    assert "Point(5, 7)" in hardened.code
    assert not hardened.changed


def test_hardening_passes_through_source_it_cannot_read(trajectory):
    assert harden("def run(ctx:\n", trajectory).code == "def run(ctx:\n"
    assert harden("x = 1\n", trajectory).code == "x = 1\n"
    assert not harden("x = 1\n", trajectory).changed


def test_hardening_keeps_an_already_hard_skill_working(trajectory, environment, critic):
    """A draft with nothing to fix is still admitted - hardening is not a tax."""
    hard = (
        "def run(ctx, company):\n"
        "    ctx.ctl.type_text(company)\n"
        '    rows = ctx.see.find_text("Acme Corp", "row")\n'
        '    ctx.expect(bool(rows), "no row for the company")\n'
        "    ctx.ctl.click(rows[0])\n"
        '    buttons = ctx.see.find_text("Confirm payment", "button")\n'
        '    ctx.expect(bool(buttons), "no confirm button")\n'
        "    ctx.ctl.click(buttons[0])\n"
        "    return True\n"
    )
    store = InMemorySkillStore()
    llm = FakeLLM([reply(hard, example_args={"company": "acme"})])
    admission = synthesizer(llm, store, critic, max_repairs=0).admit(trajectory, environment)

    assert admission.ok, admission.reason
    assert admission.attempts[-1].hardening is not None
    assert admission.attempts[-1].hardening.coordinates_replaced == 0
    assert store.get("confirm_invoice_payment", DOMAIN).version == 1


# --------------------------------------------------------------------------------------
# The prompt and the brief
# --------------------------------------------------------------------------------------


def test_the_prompt_states_the_api_forbids_imports_and_demands_a_verifier():
    prompt = load_prompt()
    for member in ("ctx.ctl.click", "ctx.ctl.type_text", "ctx.see.find_text", "ctx.expect"):
        assert member in prompt
    assert "No imports" in prompt
    assert "def verify(ctx, result)" in prompt
    assert "Never write coordinates" in prompt


def test_the_brief_describes_the_run_deterministically(trajectory):
    brief = describe_trajectory(trajectory)
    assert brief == describe_trajectory(trajectory)
    assert trajectory.task in brief
    assert "Confirm payment" in brief  # the element it has to end up clicking
    assert "STEP 0: type_text 'acme'" in brief
    assert trajectory.steps[0].before.fingerprint.value in brief


def test_the_prompt_is_what_the_model_is_given(trajectory, critic, skill_store):
    llm = FakeLLM([reply()])
    synthesizer(llm, skill_store, critic).synthesize(trajectory)
    assert llm.requests[0].system == load_prompt()


def test_a_fingerprint_mismatch_is_reported_not_raised(trajectory, critic, skill_store, scenario):
    """A precondition that matches nothing on screen fails the attempt cleanly."""

    def elsewhere() -> ReplayEnvironment:
        return ReplayEnvironment(scenario.controller, scenario.perceiver, restored=True)

    llm = FakeLLM([reply()])
    synth = synthesizer(llm, skill_store, critic, max_repairs=0, min_similarity=2.0)
    admission = synth.admit(trajectory, elsewhere)
    assert not admission.ok
    assert admission.attempts[-1].stage == "precondition"
    assert isinstance(trajectory.steps[0].before.fingerprint, Fingerprint)
