"""The critic and its programmatic checks.

Two things are being proved here.

**Correctness of "I do not know".** Every check is exercised alone for all three of its
outcomes, and the third one - ``unknown`` - is the case each test file section leads
with, because collapsing it into "no" is the failure mode that makes a critic lie.

**The cost policy.** ``assert fake_llm.calls == 0`` on a decisive positive and on a
decisive negative is the headline assertion of this piece: the common paths are free.
The escalated path is driven through a recorded cassette, so the "exactly one model call"
claim is measured against a recording rather than asserted about a mock.

Nothing here touches a network or a real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillweaver.agent import checks as C
from skillweaver.agent.critic import (
    MIN_EVIDENCE_CHARS,
    MODEL_CONFIDENCE_CAP,
    PROMPT_PATH,
    CriticVerdict,
    TieredCritic,
    load_prompt,
)
from skillweaver.contracts import (
    Box,
    Critic,
    Element,
    ElementKind,
    ElementSource,
    Fingerprint,
    LLMResponse,
    Observation,
    Screenshot,
    Usage,
    Verdict,
    utcnow,
)
from skillweaver.errors import ProviderError
from skillweaver.llm.cassette import CassetteClient
from tests.fakes import FakeLLM, Scenario
from tests.fakes.controller import render_png
from tests.fakes.perception import SimpleElementIndex

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def observe(scenario: Scenario, state: str) -> Observation:
    """A real :class:`Observation` of one state of the fake invoicing app."""
    scenario.controller.state = state
    return scenario.perceiver.observe(scenario.controller)


def element(text: str, y: int = 100, kind: ElementKind = ElementKind.text) -> Element:
    return Element(Box(20, y, 300, 24), kind, text, 0.9, None, ElementSource.merged)


def make_obs(
    elements: tuple[Element, ...] = (),
    *,
    fingerprint: Fingerprint | None = None,
    url: str | None = "https://fake.test/x",
) -> Observation:
    """An observation built by hand, for the cases the fake app cannot produce -
    an empty element index, or a chosen fingerprint similarity."""
    png = render_png(elements)
    shot = Screenshot(png, 800, 600, 1.0, utcnow())
    return Observation(
        screenshot=shot,
        elements=elements,
        index=SimpleElementIndex(elements),
        fingerprint=fingerprint or Fingerprint("hand-built", {"layout": "L", "text": "T"}),
        url=url,
        taken_at=utcnow(),
    )


def fingerprints(
    matching: int, total: int = 10, *, differing: int | None = None
) -> tuple[Fingerprint, Fingerprint]:
    """Two distinct fingerprints whose ``similarity`` is exactly ``matching / total``.

    Lets a test land a score precisely inside or outside the ambiguity band around
    ``SAME_STATE_THRESHOLD`` without depending on what any real screen happens to hash to.
    """
    if differing is not None:
        total, matching = matching, matching - differing
    left = {f"p{i}": "same" for i in range(total)}
    right = {f"p{i}": ("same" if i < matching else "other") for i in range(total)}
    return Fingerprint("left", left), Fingerprint("right", right)


def band_pair() -> tuple[Observation, Observation]:
    """Two observations whose similarity (``0.6``) sits inside the ambiguity band."""
    left, right = fingerprints(6)
    assert left.similarity(right) == pytest.approx(0.6)
    return make_obs((element("a"),), fingerprint=left), make_obs((element("b"),), fingerprint=right)


def incomparable_pair() -> tuple[Observation, Observation]:
    """Two different fingerprints that share no parts, so how alike they are cannot be
    measured - the one case a change check honestly cannot answer."""
    return (
        make_obs((element("a"),), fingerprint=Fingerprint("a")),
        make_obs((element("b"),), fingerprint=Fingerprint("b")),
    )


ERROR_BANNER = element("Error: the payment was declined. Please try again.", y=200)


# --------------------------------------------------------------------------------------
# CheckVerdict: the three-valued contract itself
# --------------------------------------------------------------------------------------


class TestCheckVerdict:
    def test_a_check_verdict_is_a_verdict(self, scenario: Scenario) -> None:
        result = C.state_changed()(observe(scenario, "list"), observe(scenario, "done"))
        assert isinstance(result, Verdict)
        assert result.source == "programmatic"
        assert len(result.reason) > 10, "a reason a human can read"

    def test_unknown_is_distinct_from_no(self) -> None:
        before, after = incomparable_pair()
        unknown = C.state_changed()(before, after)
        no = C.state_changed()(before, before)
        assert unknown.outcome is C.Outcome.unknown
        assert no.outcome is C.Outcome.failed
        # Both are falsy, which is exactly why the confidence has to carry the difference.
        assert unknown.ok is False and no.ok is False
        assert unknown.confidence == 0.0 and no.confidence > 0.0
        assert unknown.unknown and not no.unknown
        assert no.decisive and not unknown.decisive


# --------------------------------------------------------------------------------------
# Fingerprint checks
# --------------------------------------------------------------------------------------


class TestStateChanged:
    def test_a_screen_that_is_mostly_the_same_still_counts_as_changed(self) -> None:
        """ "Did anything happen?" is not "is this still the same screen?".

        Typing a name into a form leaves you on the same page with nearly the same
        pixels - a same-state comparison says yes, it is still the settings screen -
        and a step that reported that as "nothing changed" called a successful form
        fill a failure and refused to do the next step. Only an indistinguishable
        pair means nothing happened.
        """
        before, after = band_pair()
        assert before.fingerprint.similarity(after.fingerprint) == pytest.approx(0.6)
        result = C.state_changed()(before, after)
        assert result.outcome is C.Outcome.passed
        assert "the screen changed" in result.reason

    def test_a_single_differing_chunk_is_not_a_change(self) -> None:
        """Slack for a frame that is the same picture measured slightly differently -
        a caret, an anti-aliased edge - without which the veto would never fire."""
        left, right = fingerprints(60, differing=1)
        before = make_obs(fingerprint=left)
        after = make_obs(fingerprint=right)
        assert left.similarity(right) >= C.UNCHANGED_SIMILARITY
        assert C.state_changed()(before, after).outcome is C.Outcome.failed

    def test_unknown_when_the_fingerprints_have_no_comparable_parts(self) -> None:
        before = make_obs(fingerprint=Fingerprint("a"))
        after = make_obs(fingerprint=Fingerprint("b"))
        result = C.state_changed()(before, after)
        assert result.outcome is C.Outcome.unknown
        assert "no comparable parts" in result.reason

    def test_passes_when_the_screen_really_changed(self, scenario: Scenario) -> None:
        result = C.state_changed()(observe(scenario, "list"), observe(scenario, "done"))
        assert result.outcome is C.Outcome.passed
        assert result.confidence >= 0.8

    def test_fails_when_nothing_moved(self, scenario: Scenario) -> None:
        same = observe(scenario, "list")
        result = C.state_changed()(same, observe(scenario, "list"))
        assert result.outcome is C.Outcome.failed
        assert result.confidence == 1.0
        assert "did not change" in result.reason
        assert same.fingerprint == observe(scenario, "list").fingerprint


class TestStateUnchanged:
    def test_it_is_the_exact_mirror_of_state_changed(self) -> None:
        """Whatever one calls a change, the other calls the absence of one - including
        the one case neither can answer."""
        for pair in (band_pair(), incomparable_pair(), (make_obs(), make_obs())):
            changed = C.state_changed()(*pair)
            unchanged = C.state_unchanged()(*pair)
            if changed.outcome is C.Outcome.unknown:
                assert unchanged.outcome is C.Outcome.unknown
            else:
                assert changed.ok is not unchanged.ok

    def test_passes_when_nothing_moved(self, scenario: Scenario) -> None:
        result = C.state_unchanged()(observe(scenario, "list"), observe(scenario, "list"))
        assert result.outcome is C.Outcome.passed

    def test_fails_when_the_screen_changed(self, scenario: Scenario) -> None:
        result = C.state_unchanged()(observe(scenario, "list"), observe(scenario, "done"))
        assert result.outcome is C.Outcome.failed


class TestMatchesState:
    def test_unknown_inside_the_ambiguity_band(self) -> None:
        left, right = fingerprints(6)
        result = C.matches_state(left)(make_obs(), make_obs(fingerprint=right))
        assert result.outcome is C.Outcome.unknown

    def test_passes_on_the_expected_state(self, scenario: Scenario) -> None:
        goal = observe(scenario, "done").fingerprint
        result = C.matches_state(goal)(observe(scenario, "list"), observe(scenario, "done"))
        assert result.outcome is C.Outcome.passed
        assert result.confidence == 1.0
        assert goal.value[:12] in result.reason

    def test_fails_on_any_other_state(self, scenario: Scenario) -> None:
        goal = observe(scenario, "done").fingerprint
        result = C.matches_state(goal)(observe(scenario, "list"), observe(scenario, "archive"))
        assert result.outcome is C.Outcome.failed


# --------------------------------------------------------------------------------------
# Element checks
# --------------------------------------------------------------------------------------


class TestElementPresent:
    def test_unknown_when_perception_returned_nothing(self, scenario: Scenario) -> None:
        result = C.element_present(text="Confirm payment")(observe(scenario, "list"), make_obs())
        assert result.outcome is C.Outcome.unknown
        assert "no elements at all" in result.reason

    def test_passes_on_an_exact_match(self, scenario: Scenario) -> None:
        result = C.element_present(text="Confirm payment", kind=ElementKind.button)(
            observe(scenario, "list"), observe(scenario, "selected")
        )
        assert result.outcome is C.Outcome.passed
        assert result.confidence == 1.0

    def test_a_fuzzy_only_match_passes_with_less_confidence(self, scenario: Scenario) -> None:
        result = C.element_present(text="Confirm paymnt")(
            observe(scenario, "list"), observe(scenario, "selected")
        )
        assert result.outcome is C.Outcome.passed
        assert result.confidence == C.FUZZY_ONLY_CONFIDENCE
        assert "fuzzy" in result.reason

    def test_fails_when_the_element_is_not_there(self, scenario: Scenario) -> None:
        result = C.element_present(text="Confirm payment")(
            observe(scenario, "selected"), observe(scenario, "list")
        )
        assert result.outcome is C.Outcome.failed

    def test_finds_by_kind_alone(self, scenario: Scenario) -> None:
        before, after = observe(scenario, "done"), observe(scenario, "selected")
        assert C.element_present(kind=ElementKind.button)(before, after).outcome is C.Outcome.passed
        assert C.element_present(kind=ElementKind.button)(after, before).outcome is C.Outcome.failed

    def test_rejects_an_empty_query(self) -> None:
        with pytest.raises(ValueError, match="needs a text, a kind, or both"):
            C.element_present()
        with pytest.raises(ValueError, match="blank text"):
            C.element_present(text="   ")


class TestElementAbsent:
    def test_unknown_when_perception_returned_nothing(self, scenario: Scenario) -> None:
        result = C.element_absent(text="Confirm payment")(observe(scenario, "selected"), make_obs())
        assert result.outcome is C.Outcome.unknown

    def test_unknown_on_a_fuzzy_only_hit(self, scenario: Scenario) -> None:
        # A near-miss is the one case where neither "gone" nor "there" is honest.
        result = C.element_absent(text="Confirm paymnt")(
            observe(scenario, "list"), observe(scenario, "selected")
        )
        assert result.outcome is C.Outcome.unknown
        assert "near match" in result.reason

    def test_passes_when_the_element_is_gone(self, scenario: Scenario) -> None:
        # The kind matters here: the confirmation page says "Payment confirmed", which
        # is close enough to "Confirm payment" for the fuzzy pass to flag it. Naming the
        # kind restricts the search to buttons, of which the page has none.
        result = C.element_absent(text="Confirm payment", kind=ElementKind.button)(
            observe(scenario, "selected"), observe(scenario, "done")
        )
        assert result.outcome is C.Outcome.passed

    def test_a_confusable_label_alone_is_only_unknown(self, scenario: Scenario) -> None:
        result = C.element_absent(text="Confirm payment")(
            observe(scenario, "selected"), observe(scenario, "done")
        )
        assert result.outcome is C.Outcome.unknown

    def test_fails_when_the_element_is_still_there(self, scenario: Scenario) -> None:
        result = C.element_absent(text="Confirm payment")(
            observe(scenario, "list"), observe(scenario, "selected")
        )
        assert result.outcome is C.Outcome.failed

    def test_rejects_an_empty_query(self) -> None:
        with pytest.raises(ValueError):
            C.element_absent()


# --------------------------------------------------------------------------------------
# Text checks
# --------------------------------------------------------------------------------------


class TestTextAppeared:
    def test_unknown_when_nothing_on_screen_carries_text(self, scenario: Scenario) -> None:
        blank = make_obs((Element(Box(0, 0, 10, 10), ElementKind.icon, ""),))
        result = C.text_appeared("Payment confirmed")(observe(scenario, "list"), blank)
        assert result.outcome is C.Outcome.unknown
        assert "carries any text" in result.reason

    def test_passes_when_the_text_is_on_screen(self, scenario: Scenario) -> None:
        result = C.text_appeared("payment CONFIRMED")(
            observe(scenario, "selected"), observe(scenario, "done")
        )
        assert result.outcome is C.Outcome.passed

    def test_fails_when_the_text_is_nowhere(self, scenario: Scenario) -> None:
        result = C.text_appeared("Payment confirmed")(
            observe(scenario, "selected"), observe(scenario, "list")
        )
        assert result.outcome is C.Outcome.failed

    def test_require_new_rejects_text_that_was_already_there(self, scenario: Scenario) -> None:
        both = observe(scenario, "list")
        assert C.text_appeared("Invoices")(both, both).outcome is C.Outcome.passed
        strict = C.text_appeared("Invoices", require_new=True)(both, both)
        assert strict.outcome is C.Outcome.failed
        assert "already there" in strict.reason

    def test_rejects_an_empty_query(self) -> None:
        with pytest.raises(ValueError):
            C.text_appeared("  ")


# --------------------------------------------------------------------------------------
# Error checks
# --------------------------------------------------------------------------------------


class TestNoErrorState:
    def test_unknown_when_nothing_on_screen_carries_text(self, scenario: Scenario) -> None:
        blank = make_obs((Element(Box(0, 0, 10, 10), ElementKind.icon, ""),))
        result = C.no_error_state()(observe(scenario, "list"), blank)
        assert result.outcome is C.Outcome.unknown

    def test_passes_on_a_clean_screen(self, scenario: Scenario) -> None:
        result = C.no_error_state()(observe(scenario, "selected"), observe(scenario, "done"))
        assert result.outcome is C.Outcome.passed

    def test_fails_when_an_error_appears(self, scenario: Scenario) -> None:
        result = C.no_error_state()(observe(scenario, "list"), make_obs((ERROR_BANNER,)))
        assert result.outcome is C.Outcome.failed
        assert "error" in result.reason

    def test_a_preexisting_error_is_not_this_steps_fault(self) -> None:
        before = make_obs((ERROR_BANNER,))
        after = make_obs((ERROR_BANNER, element("Invoices", y=20)))
        assert C.no_error_state()(before, after).outcome is C.Outcome.passed
        strict = C.no_error_state(ignore_preexisting=False)(before, after)
        assert strict.outcome is C.Outcome.failed

    def test_ordinary_ui_wording_does_not_trip_it(self, scenario: Scenario) -> None:
        # The phrase list is deliberately narrow; the fake app's chrome must stay clean.
        for state in ("list", "searched", "selected", "done", "archive"):
            result = C.no_error_state()(observe(scenario, "list"), observe(scenario, state))
            assert result.outcome is C.Outcome.passed, state


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def yes() -> C.Check:
    return C.text_appeared("Payment confirmed")


def no() -> C.Check:
    return C.text_appeared("No such words anywhere")


def dunno() -> C.Check:
    """Unknown on the standard pair: "Payment confirmed" is a fuzzy-only hit for this
    query, and :class:`element_absent` refuses to call a near-miss either way."""
    return C.element_absent(text="Confirm paymnt")


class TestComposition:
    """Kleene three-valued logic: ignorance must propagate, never round to a "no"."""

    @pytest.fixture
    def pair(self, scenario: Scenario) -> tuple[Observation, Observation]:
        return observe(scenario, "selected"), observe(scenario, "done")

    def test_the_three_helpers_really_are_yes_no_and_unknown(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        assert yes()(*pair).outcome is C.Outcome.passed
        assert no()(*pair).outcome is C.Outcome.failed
        assert dunno()(*pair).outcome is C.Outcome.unknown

    def test_all_of_is_unknown_when_a_part_is_unknown(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        result = C.all_of(yes(), dunno())(*pair)
        assert result.outcome is C.Outcome.unknown

    def test_all_of_fails_when_any_part_fails(self, pair: tuple[Observation, Observation]) -> None:
        result = C.all_of(yes(), no())(*pair)
        assert result.outcome is C.Outcome.failed
        assert "No such words" in result.reason

    def test_all_of_passes_only_when_every_part_passes(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        result = C.all_of(yes(), C.text_appeared("INV-1042"))(*pair)
        assert result.outcome is C.Outcome.passed

    def test_any_of_passes_when_one_part_passes(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        assert C.any_of(no(), yes())(*pair).outcome is C.Outcome.passed

    def test_any_of_fails_only_when_every_part_fails(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        assert C.any_of(no(), no())(*pair).outcome is C.Outcome.failed

    def test_any_of_is_unknown_when_no_part_passes_but_one_is_unknown(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        assert C.any_of(no(), dunno())(*pair).outcome is C.Outcome.unknown

    def test_empty_composites_are_unknown(self, pair: tuple[Observation, Observation]) -> None:
        assert C.all_of()(*pair).outcome is C.Outcome.unknown
        assert C.any_of()(*pair).outcome is C.Outcome.unknown

    def test_not_swaps_yes_and_no_but_never_touches_unknown(
        self, pair: tuple[Observation, Observation]
    ) -> None:
        assert C.not_(yes())(*pair).outcome is C.Outcome.failed
        assert C.not_(no())(*pair).outcome is C.Outcome.passed
        assert C.not_(dunno())(*pair).outcome is C.Outcome.unknown


# --------------------------------------------------------------------------------------
# The critic: the cost policy
# --------------------------------------------------------------------------------------


class TestCostPolicy:
    """The headline of this piece: a decisive answer costs exactly zero model calls."""

    def test_the_critic_satisfies_the_protocol(self) -> None:
        assert isinstance(TieredCritic(), Critic)

    def test_a_decisive_positive_makes_no_model_call(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        before, after = observe(scenario, "selected"), observe(scenario, "done")
        critic = TieredCritic(fake_llm, expected_state=after.fingerprint)

        verdict = critic.judge("confirm the Acme payment", before, after)

        assert fake_llm.calls == 0, "a fingerprint match must never cost a model call"
        assert verdict.ok is True
        assert verdict.source == "programmatic"
        assert verdict.escalated is False
        assert verdict.policy == "evidence-passed"
        assert verdict.confidence == 1.0

    def test_a_corroborating_match_is_decisive_and_free(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        """A recalled ending that MATCHES still settles it without paying anyone."""
        before, after = observe(scenario, "selected"), observe(scenario, "done")
        critic = TieredCritic(fake_llm, corroborating_state=after.fingerprint)

        verdict = critic.judge("confirm the Acme payment", before, after)

        assert fake_llm.calls == 0, "a fingerprint match must never cost a model call"
        assert verdict.ok is True
        assert verdict.escalated is False
        assert verdict.policy == "corroborated"

    def test_a_corroborating_mismatch_asks_rather_than_refuses(self, scenario: Scenario) -> None:
        """The whole point of the third role: a recalled ending may never say no.

        Where the LAST run of this task ended is not where this one must end - it is
        where one run ended with the arguments it was given. Measured twice live: a
        Wikipedia skill replayed for a new query landed on the right article and was
        refused at similarity 0.120, and a shop listing recorded mid-lazy-load was
        refused at 0.400 against the page the replay had genuinely reached. As
        evidence this screen rejects correct work; as corroboration it buys a look.
        """
        before, after = observe(scenario, "selected"), observe(scenario, "done")
        elsewhere = observe(scenario, "list").fingerprint
        fake_llm = FakeLLM(
            ['{"ok": true, "evidence": "the receipt is on screen", "confidence": 1.0}']
        )
        critic = TieredCritic(fake_llm, corroborating_state=elsewhere)

        verdict = critic.judge("confirm the Acme payment", before, after)

        assert verdict.escalated is True, "a mismatch must ask, never refuse outright"
        assert fake_llm.calls == 1
        assert verdict.source == "model"

    def test_a_decisive_negative_makes_no_model_call(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        stuck = observe(scenario, "list")
        critic = TieredCritic(fake_llm, expected_state=observe(scenario, "done").fingerprint)

        verdict = critic.judge("confirm the Acme payment", stuck, observe(scenario, "list"))

        assert fake_llm.calls == 0, "a screen that did not move must never cost a model call"
        assert verdict.ok is False
        assert verdict.source == "programmatic"
        assert verdict.escalated is False
        assert verdict.policy == "check-failed"
        assert "did not change" in verdict.reason

    def test_an_error_on_screen_is_a_free_negative(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        verdict = TieredCritic(fake_llm).judge(
            "confirm the Acme payment", observe(scenario, "selected"), make_obs((ERROR_BANNER,))
        )
        assert fake_llm.calls == 0
        assert verdict.ok is False and verdict.policy == "check-failed"
        assert "error text" in verdict.reason

    def test_a_failing_evidence_check_beats_a_changed_screen(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        # The screen moved and there is no error, but it moved to the wrong place.
        critic = TieredCritic(fake_llm, expected_state=observe(scenario, "done").fingerprint)
        verdict = critic.judge(
            "confirm the Acme payment", observe(scenario, "list"), observe(scenario, "archive")
        )
        assert fake_llm.calls == 0
        assert verdict.ok is False and verdict.policy == "check-failed"

    def test_a_changed_screen_alone_is_never_a_yes(
        self, scenario: Scenario, fake_llm: FakeLLM
    ) -> None:
        # Vetoes passing proves nothing, so with no evidence check this must escalate -
        # and with no model configured it must say so rather than guess.
        verdict = TieredCritic().judge(
            "confirm the Acme payment", observe(scenario, "list"), observe(scenario, "archive")
        )
        assert verdict.ok is False
        assert verdict.policy == "inconclusive-no-model"
        assert verdict.escalated is False
        assert verdict.confidence == 0.0
        assert fake_llm.calls == 0

    def test_require_change_can_be_turned_off(self, scenario: Scenario) -> None:
        same = observe(scenario, "list")
        critic = TieredCritic(require_change=False, evidence=[C.text_appeared("Invoices")])
        verdict = critic.judge("stay put", same, observe(scenario, "list"))
        assert verdict.ok is True and verdict.policy == "evidence-passed"

    def test_the_verdict_carries_the_whole_check_trail(self, scenario: Scenario) -> None:
        after = observe(scenario, "done")
        critic = TieredCritic(expected_state=after.fingerprint)
        verdict = critic.judge("confirm", observe(scenario, "selected"), after)
        names = [c.name for c in verdict.checks]
        assert names == ["matches_state", "state_changed", "no_error_state"]
        assert all(isinstance(c, Verdict) for c in verdict.checks)
        assert "no model call" in verdict.summary()

    def test_expecting_derives_a_critic_for_one_step(self, scenario: Scenario) -> None:
        base = TieredCritic()
        step = base.expecting(C.text_appeared("Payment confirmed"))
        assert len(base.evidence_checks) == 0 and len(step.evidence_checks) == 1
        verdict = step.judge("confirm", observe(scenario, "selected"), observe(scenario, "done"))
        assert verdict.ok is True and verdict.escalated is False


# --------------------------------------------------------------------------------------
# The critic: the escalated path, driven through a cassette
# --------------------------------------------------------------------------------------

GOOD_JSON = (
    '{"ok": true, "evidence": "the heading reads \'Payment confirmed\' and the Confirm '
    'payment button is gone", "reason": "The invoice page was replaced by the '
    'confirmation page for INV-1042.", "confidence": 0.95}'
)

GOOD_REPLY = LLMResponse(
    text=GOOD_JSON,
    usage=Usage(input_tokens=1180, output_tokens=64, calls=1, cost_usd=0.0043),
    stop_reason="end",
)
"""A realistic escalated reply, priced, so the cassette meter has something to count."""


def inconclusive_pair(scenario: Scenario) -> tuple[Observation, Observation]:
    """A before/after the checks genuinely cannot decide: the screen moved, no error
    is showing, and nothing was configured that could establish success."""
    return observe(scenario, "selected"), observe(scenario, "done")


def record_then_replay(
    tmp_path: Path, scenario: Scenario, reply: str | LLMResponse
) -> CassetteClient:
    """Record one critic escalation into a fresh cassette, then hand back a player.

    Recording through the real :class:`CassetteClient` rather than writing a fixture by
    hand means the request digest under test is the one the critic actually builds.
    """
    path = tmp_path / "critic.json"
    recorder = CassetteClient(path, mode="record", inner=FakeLLM([reply]))
    before, after = inconclusive_pair(scenario)
    TieredCritic(recorder).judge("confirm the Acme payment", before, after)
    assert len(recorder.cassette.interactions) == 1
    return CassetteClient(path, mode="replay")


class TestEscalation:
    def test_an_inconclusive_case_consumes_exactly_one_cassette_verdict(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        player = record_then_replay(tmp_path, scenario, GOOD_REPLY)
        before, after = inconclusive_pair(scenario)

        verdict = TieredCritic(player).judge("confirm the Acme payment", before, after)

        assert player.total_usage().calls == 1, "exactly one model call, replayed"
        assert len(player.cassette.interactions) == 1
        assert verdict.ok is True
        assert verdict.source == "model"
        assert "Payment confirmed" in verdict.reason

    def test_the_cassette_holds_that_one_request_and_no_other(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        # Proof that the replay above matched a real recording rather than anything
        # going: a different goal builds a different request and finds nothing.
        player = record_then_replay(tmp_path, scenario, GOOD_REPLY)
        with pytest.raises(ProviderError, match="no recorded interaction"):
            TieredCritic(player).judge("something else entirely", *inconclusive_pair(scenario))

    def test_the_escalation_decision_is_visible_in_the_verdict(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        player = record_then_replay(tmp_path, scenario, GOOD_REPLY)
        before, after = inconclusive_pair(scenario)

        verdict = TieredCritic(player).judge("confirm the Acme payment", before, after)

        assert isinstance(verdict, CriticVerdict)
        assert verdict.escalated is True
        assert verdict.policy == "model"
        # The reader can see WHY: the vetoes passed and nothing could prove success.
        assert [c.name for c in verdict.checks] == ["state_changed", "no_error_state"]
        assert all(c.outcome is C.Outcome.passed for c in verdict.checks)
        assert "1 model call" in verdict.summary()

    def test_the_model_is_sent_both_frames_the_goal_and_the_check_trail(
        self, scenario: Scenario
    ) -> None:
        llm = FakeLLM([GOOD_REPLY])
        before, after = inconclusive_pair(scenario)
        TieredCritic(llm).judge("confirm the Acme payment", before, after, "the receipt page")

        request = llm.requests[0]
        assert request.system == load_prompt()
        (message,) = request.messages
        assert message.images == (before.screenshot.png, after.screenshot.png)
        assert "confirm the Acme payment" in message.text
        assert "the receipt page" in message.text
        assert "state_changed [passed]" in message.text

    def test_the_model_confidence_is_capped_below_a_measurement(self, scenario: Scenario) -> None:
        llm = FakeLLM(['{"ok": true, "evidence": "the receipt is on screen", "confidence": 1.0}'])
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))
        assert verdict.ok is True
        assert verdict.confidence == MODEL_CONFIDENCE_CAP

    def test_a_model_no_is_returned_as_a_no(self, scenario: Scenario) -> None:
        llm = FakeLLM(
            [
                '{"ok": false, "evidence": "the Confirm button is still there",'
                ' "reason": "Nothing was submitted.", "confidence": 0.8}'
            ]
        )
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))
        assert verdict.ok is False
        assert verdict.escalated is True
        assert verdict.confidence == pytest.approx(0.8)
        assert "Nothing was submitted" in verdict.reason

    def test_prose_around_the_json_is_tolerated(self, scenario: Scenario) -> None:
        llm = FakeLLM(
            ["Sure, here is my judgment:\n```json\n" + GOOD_JSON + "\n```\nHope that helps."]
        )
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))
        assert verdict.ok is True and verdict.policy == "model"


# --------------------------------------------------------------------------------------
# The critic: degrading, not lying
# --------------------------------------------------------------------------------------


class TestDegradation:
    @pytest.mark.parametrize(
        "reply",
        [
            pytest.param("", id="empty"),
            pytest.param("Yes, that worked!", id="prose"),
            pytest.param("{not json at all", id="broken-json"),
            pytest.param('["ok"]', id="json-but-not-an-object"),
            pytest.param('{"verdict": "success"}', id="object-without-ok"),
        ],
    )
    def test_malformed_output_degrades_to_an_honest_do_not_know(
        self, scenario: Scenario, reply: str
    ) -> None:
        llm = FakeLLM([reply])
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))

        assert verdict.ok is False, "an unreadable reply must never become a success"
        assert verdict.confidence == 0.0, "and it must not pretend to be a confident failure"
        assert verdict.source == "model"
        assert verdict.escalated is True
        assert verdict.policy == "model-unparseable"
        assert "could not be parsed" in verdict.reason

    def test_a_success_claimed_without_evidence_degrades(self, scenario: Scenario) -> None:
        llm = FakeLLM(['{"ok": true, "evidence": "yes", "confidence": 0.99}'])
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))

        assert len("yes") < MIN_EVIDENCE_CHARS
        assert verdict.ok is False
        assert verdict.confidence == 0.0
        assert verdict.policy == "model-unevidenced"
        assert "without naming anything it could see" in verdict.reason

    def test_a_nonsense_confidence_lands_in_the_middle(self, scenario: Scenario) -> None:
        llm = FakeLLM(['{"ok": false, "reason": "the page never loaded", "confidence": "high"}'])
        verdict = TieredCritic(llm).judge("confirm", *inconclusive_pair(scenario))
        assert verdict.ok is False and verdict.confidence == 0.5

    def test_a_provider_failure_is_raised_not_degraded(self, scenario: Scenario) -> None:
        """Documented policy: a broken provider is the caller's problem to see.

        Degrading it would quietly turn an outage into a stream of "I do not know"
        verdicts and the run would burn its whole budget learning nothing.
        """

        class Broken:
            def complete(self, *a: object, **k: object) -> LLMResponse:
                raise ProviderError("rate limited after 5 retries")

            def total_usage(self) -> object:  # pragma: no cover - never reached
                raise AssertionError

            def name(self) -> str:
                return "broken"

        with pytest.raises(ProviderError, match="rate limited"):
            TieredCritic(Broken()).judge("confirm", *inconclusive_pair(scenario))  # type: ignore[arg-type]

    def test_an_exhausted_script_is_not_swallowed(self, scenario: Scenario) -> None:
        # ScriptExhausted is an AssertionError; catching it would hide a test's own bug.
        with pytest.raises(AssertionError, match="script exhausted"):
            TieredCritic(FakeLLM()).judge("confirm", *inconclusive_pair(scenario))


# --------------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------------


class TestPrompt:
    def test_the_prompt_ships_with_the_package(self) -> None:
        assert PROMPT_PATH.is_file()
        assert load_prompt() == PROMPT_PATH.read_text(encoding="utf-8")

    def test_it_demands_evidence_rather_than_a_bare_answer(self) -> None:
        prompt = load_prompt().lower()
        assert "evidence" in prompt
        assert "confidence" in prompt
        assert '"ok"' in prompt
        assert "never a bare" in prompt

    def test_it_pushes_back_against_agreeing(self) -> None:
        prompt = load_prompt().lower()
        assert "would prefer the answer to be yes" in prompt
        assert "intent is not completion" in prompt
        assert "unsure" in prompt
