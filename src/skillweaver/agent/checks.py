"""Programmatic checks: the free half of the critic.

Deterministic, over two already-captured Observations, never calling a model, a network
or a browser - microseconds where a vision model costs cents and seconds. So they are
written to be decisive as often as honestly possible, and to admit it loudly when
they are not.

**Three outcomes, not two.** ``Verdict.ok`` is a bool, so an unknown check reports
``ok=False`` - but pairs it with ``confidence == 0.0`` and ``outcome is
Outcome.unknown``, so nothing downstream can mistake "I did not see it" for "it did not
happen". :class:`~skillweaver.agent.critic.TieredCritic` escalates on exactly that
distinction: collapse it and the critic becomes either expensive or wrong. Every check
names its own ignorance - an empty element index, a screen with no text, a fingerprint
score inside :data:`AMBIGUITY_MARGIN`. :func:`all_of`, :func:`any_of` and :func:`not_`
combine them with three-valued logic so ignorance propagates.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from skillweaver.contracts import ElementKind, Fingerprint, Observation, Verdict
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD

__all__ = [
    "AMBIGUITY_MARGIN",
    "ERROR_PHRASES",
    "Check",
    "CheckVerdict",
    "Outcome",
    "all_of",
    "any_of",
    "element_absent",
    "element_present",
    "matches_state",
    "no_error_state",
    "not_",
    "state_changed",
    "state_unchanged",
    "text_appeared",
]


class Outcome(enum.StrEnum):
    """The three things a programmatic check can conclude."""

    passed = "passed"
    failed = "failed"
    unknown = "unknown"


@dataclass(frozen=True, slots=True)
class CheckVerdict(Verdict):
    """A Verdict that can also say "I do not know".

    ``ok`` is ``True`` exactly when ``outcome`` is ``Outcome.passed``, and an
    ``Outcome.unknown`` verdict always carries ``confidence == 0.0``. ``name`` is the
    check that produced it, for the critic's readable trail.
    """

    name: str = ""
    outcome: Outcome = Outcome.unknown

    @property
    def unknown(self) -> bool:
        """Whether this check declined to decide."""
        return self.outcome is Outcome.unknown

    @property
    def decisive(self) -> bool:
        """Whether this check reached a conclusion (passed or failed)."""
        return self.outcome is not Outcome.unknown


def _passed(name: str, reason: str, confidence: float = 1.0) -> CheckVerdict:
    return CheckVerdict(True, reason, _clamp(confidence), "programmatic", name, Outcome.passed)


def _failed(name: str, reason: str, confidence: float = 1.0) -> CheckVerdict:
    return CheckVerdict(False, reason, _clamp(confidence), "programmatic", name, Outcome.failed)


def _unknown(name: str, reason: str) -> CheckVerdict:
    return CheckVerdict(False, reason, 0.0, "programmatic", name, Outcome.unknown)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@runtime_checkable
class Check(Protocol):
    """A deterministic judgment on a before/after pair. Implementations are cheap, side
    effect free and safe to run in any order; ``name`` is stable and appears in the
    critic's trail."""

    name: str

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        """Judge the pair. Never raises for "no match"; returns ``Outcome.unknown``."""
        ...


# --------------------------------------------------------------------------------------
# Fingerprint checks
# --------------------------------------------------------------------------------------

AMBIGUITY_MARGIN = 0.023
"""Half-width of the "I do not know" band around the same-state threshold.

Derived from the separation ``SAME_STATE_THRESHOLD`` was fitted to, and it has to be
re-derived whenever that is. Same-state scores bottom out at ``0.305`` and
different-state scores top out at ``0.213``, so the ``0.26`` cut has ``0.047`` of
clearance below and ``0.045`` above; this is HALF that gap, leaving every measured pair
half its clearance. A band WIDER than the gap makes every check abstain at both extremes
and every abstention is a model call - even ``0.04`` clears the extremes by ``0.005``,
finer than the thing being measured."""


def _similarity_confidence(distance: float) -> float:
    """Confidence for a fingerprint decision made ``distance`` beyond the band edge:
    ``0.8`` at the edge, rising to ``1.0`` a further ``0.1`` away. Never below ``0.8``,
    because inside the band we return ``unknown``."""
    return _clamp(0.8 + 2.0 * max(distance - AMBIGUITY_MARGIN, 0.0))


def _comparable(a: Fingerprint, b: Fingerprint) -> bool:
    """Whether ``similarity`` has anything to work with beyond raw equality."""
    return bool(set(a.parts) | set(b.parts))


@dataclass(frozen=True, slots=True)
class _SameState:
    """Shared machinery: is ``left`` the same UI state as ``right``?

    ``(outcome, similarity, confidence, detail)``, ``unknown`` inside the band or when
    the two fingerprints carry no comparable parts.
    """

    threshold: float = SAME_STATE_THRESHOLD
    margin: float = AMBIGUITY_MARGIN

    def compare(self, left: Fingerprint, right: Fingerprint) -> tuple[Outcome, float, float, str]:
        if left == right:
            return Outcome.passed, 1.0, 1.0, "the fingerprints are identical"
        similarity = left.similarity(right)
        if not _comparable(left, right):
            return (
                Outcome.unknown,
                similarity,
                0.0,
                "the fingerprints differ but carry no comparable parts, so how alike "
                "the two screens are cannot be measured",
            )
        distance = abs(similarity - self.threshold)
        detail = (
            f"similarity {similarity:.3f} against a same-state threshold of {self.threshold:.2f}"
        )
        if similarity >= self.threshold + self.margin:
            return Outcome.passed, similarity, _similarity_confidence(distance), detail
        if similarity <= self.threshold - self.margin:
            return Outcome.failed, similarity, _similarity_confidence(distance), detail
        return (
            Outcome.unknown,
            similarity,
            0.0,
            f"{detail}, inside the +/-{self.margin:.2f} ambiguity band",
        )


@dataclass(frozen=True, slots=True)
class state_changed:  # noqa: N801 - a check reads as a verb at the call site
    """Passes when ``after`` is a different UI state from ``before``. The workhorse
    negative: a step that left the screen exactly as it found it did not do anything,
    whatever the model would like to say about it."""

    threshold: float = SAME_STATE_THRESHOLD
    margin: float = AMBIGUITY_MARGIN
    name: str = field(default="state_changed", init=False)

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        outcome, _, confidence, detail = _SameState(self.threshold, self.margin).compare(
            before.fingerprint, after.fingerprint
        )
        if outcome is Outcome.unknown:
            return _unknown(self.name, f"cannot tell whether the screen changed: {detail}")
        if outcome is Outcome.passed:  # same state -> the screen did NOT change
            return _failed(self.name, f"the screen did not change: {detail}", confidence)
        return _passed(self.name, f"the screen changed: {detail}", confidence)


@dataclass(frozen=True, slots=True)
class state_unchanged:  # noqa: N801
    """Passes when ``after`` is the same UI state as ``before`` - the mirror of
    :class:`state_changed`, for a step whose whole point is that nothing moved."""

    threshold: float = SAME_STATE_THRESHOLD
    margin: float = AMBIGUITY_MARGIN
    name: str = field(default="state_unchanged", init=False)

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        outcome, _, confidence, detail = _SameState(self.threshold, self.margin).compare(
            before.fingerprint, after.fingerprint
        )
        if outcome is Outcome.unknown:
            return _unknown(self.name, f"cannot tell whether the screen changed: {detail}")
        if outcome is Outcome.passed:
            return _passed(self.name, f"the screen did not change: {detail}", confidence)
        return _failed(self.name, f"the screen changed: {detail}", confidence)


@dataclass(frozen=True, slots=True)
class matches_state:  # noqa: N801
    """Passes when ``after`` is the UI state identified by ``expected``. The cheapest
    decisive POSITIVE there is, and the reason a learned skill is worth recording with
    the fingerprint of the screen it ends on."""

    expected: Fingerprint
    threshold: float = SAME_STATE_THRESHOLD
    margin: float = AMBIGUITY_MARGIN
    name: str = field(default="matches_state", init=False)

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        outcome, _, confidence, detail = _SameState(self.threshold, self.margin).compare(
            after.fingerprint, self.expected
        )
        expected_id = self.expected.value[:12]
        if outcome is Outcome.unknown:
            return _unknown(
                self.name, f"cannot tell whether the screen is state {expected_id}: {detail}"
            )
        if outcome is Outcome.passed:
            return _passed(
                self.name, f"the screen is the expected state {expected_id}: {detail}", confidence
            )
        return _failed(
            self.name,
            f"the screen is not the expected state {expected_id} "
            f"(it is {after.fingerprint.value[:12]}): {detail}",
            confidence,
        )


# --------------------------------------------------------------------------------------
# Element checks
# --------------------------------------------------------------------------------------

FUZZY_ONLY_CONFIDENCE = 0.75
"""Confidence for a hit that only a fuzzy text match found. An exact or substring match
is the element; a fuzzy match is a guess about what OCR meant - and for
:class:`element_absent` it is worth nothing, because a near-miss is precisely where "it
is not there" cannot be trusted."""


def _locate(
    observation: Observation, text: str | None, kind: ElementKind | None
) -> tuple[int, int]:
    """``(exact_hits, fuzzy_hits)`` for a text and/or kind query against the index."""
    index = observation.index
    if text is None:
        pool = index.all() if kind is None else index.by_kind(kind)
        return len(pool), len(pool)
    exact = index.find_text(text, kind, fuzzy=False)
    fuzzy = index.find_text(text, kind, fuzzy=True)
    return len(exact), len(fuzzy)


def _describe_query(text: str | None, kind: ElementKind | None) -> str:
    if text is not None and kind is not None:
        return f"a {kind.value} reading {text!r}"
    if text is not None:
        return f"an element reading {text!r}"
    if kind is not None:
        return f"an element of kind {kind.value}"
    return "any element"


@dataclass(frozen=True, slots=True)
class element_present:  # noqa: N801
    """Passes when ``after`` shows an element matching ``text`` and/or ``kind``.

    At least one of ``text`` and ``kind`` must be given. Matching goes through
    ElementIndex, so it is case-insensitive and tolerant of OCR noise; a fuzzy-only hit
    is reported at :data:`FUZZY_ONLY_CONFIDENCE`. Unknown when ``after`` has no elements
    at all: an empty index is not evidence about this element.
    """

    text: str | None = None
    kind: ElementKind | None = None
    name: str = field(default="element_present", init=False)

    def __post_init__(self) -> None:
        if self.text is None and self.kind is None:
            raise ValueError("element_present needs a text, a kind, or both")
        if self.text is not None and not self.text.strip():
            raise ValueError("element_present was given a blank text")

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        query = _describe_query(self.text, self.kind)
        if not after.elements:
            return _unknown(
                self.name,
                f"cannot tell whether {query} is present: perception returned no elements at all",
            )
        exact, fuzzy = _locate(after, self.text, self.kind)
        if exact:
            return _passed(self.name, f"{query} is on screen ({exact} match(es))")
        if fuzzy:
            return _passed(
                self.name,
                f"{query} is probably on screen: only a fuzzy text match found it "
                f"({fuzzy} near match(es))",
                FUZZY_ONLY_CONFIDENCE,
            )
        return _failed(
            self.name, f"{query} is not on screen, among {len(after.elements)} element(s) seen"
        )


@dataclass(frozen=True, slots=True)
class element_absent:  # noqa: N801
    """Passes when ``after`` shows NO element matching ``text`` and/or ``kind``.

    Stricter about its own ignorance than :class:`element_present`, because absence is
    the weaker claim: unknown on an empty index, and unknown when only a FUZZY match was
    found, since a near-miss is the one case where neither answer can be said honestly.
    """

    text: str | None = None
    kind: ElementKind | None = None
    name: str = field(default="element_absent", init=False)

    def __post_init__(self) -> None:
        if self.text is None and self.kind is None:
            raise ValueError("element_absent needs a text, a kind, or both")
        if self.text is not None and not self.text.strip():
            raise ValueError("element_absent was given a blank text")

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        query = _describe_query(self.text, self.kind)
        if not after.elements:
            return _unknown(
                self.name,
                f"cannot tell whether {query} is gone: perception returned no elements at all",
            )
        exact, fuzzy = _locate(after, self.text, self.kind)
        if exact:
            return _failed(self.name, f"{query} is still on screen ({exact} match(es))")
        if fuzzy:
            return _unknown(
                self.name,
                f"cannot tell whether {query} is gone: no exact match, but {fuzzy} near "
                f"match(es) that could be the same element misread",
            )
        return _passed(
            self.name, f"{query} is not on screen, among {len(after.elements)} element(s) seen"
        )


# --------------------------------------------------------------------------------------
# Text checks
# --------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _screen_text(observation: Observation) -> str:
    """All visible text of an observation, normalized and joined by ``|``."""
    return " | ".join(_normalize(e.text) for e in observation.elements if e.text.strip())


@dataclass(frozen=True, slots=True)
class text_appeared:  # noqa: N801
    """Passes when ``text`` is legible on ``after``.

    Case- and whitespace-insensitive over the concatenated element text, falling back to
    the index's fuzzy search so OCR noise does not turn a real hit into a false negative.
    With ``require_new=True`` the text must additionally NOT have been on ``before``,
    which is what judging a step wants: a confirmation banner that was already there
    proves nothing. Unknown when nothing on ``after`` carries any text.
    """

    text: str
    require_new: bool = False
    name: str = field(default="text_appeared", init=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text_appeared was given a blank text")

    def _seen(self, observation: Observation) -> tuple[bool, bool]:
        """``(exact, fuzzy)`` - whether the text is on this screen, and how surely."""
        needle = _normalize(self.text)
        if needle in _screen_text(observation):
            return True, True
        return False, bool(observation.index.find_text(self.text, None, fuzzy=True))

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        if not _screen_text(after):
            return _unknown(
                self.name,
                f"cannot tell whether {self.text!r} appeared: no element on the screen "
                "carries any text",
            )
        exact, fuzzy = self._seen(after)
        if not exact and not fuzzy:
            return _failed(self.name, f"{self.text!r} is not anywhere on the screen")
        if self.require_new and any(self._seen(before)):
            return _failed(
                self.name,
                f"{self.text!r} is on the screen but it was already there before the step, "
                "so it is not new evidence",
            )
        if exact:
            return _passed(self.name, f"{self.text!r} is on the screen")
        return _passed(
            self.name,
            f"{self.text!r} is probably on the screen: only a fuzzy text match found it",
            FUZZY_ONLY_CONFIDENCE,
        )


# --------------------------------------------------------------------------------------
# Error checks
# --------------------------------------------------------------------------------------

ERROR_PHRASES: tuple[str, ...] = (
    r"error",
    r"failed",
    r"failure",
    r"invalid",
    r"is required",
    r"required field",
    r"not allowed",
    r"not permitted",
    r"denied",
    r"try again",
    r"went wrong",
    r"unable to",
    r"could ?n[o']?t",
    r"must be",
    r"please (?:correct|fix)",
    r"warning",
    r"exception",
    r"timed out",
    r"unauthori[sz]ed",
    r"forbidden",
    r"does ?n[o']?t match",
    r"not found",
    r"not available",
)
"""Word-boundary patterns that mark a dialog, alert or validation message.

Deliberately a PHRASE list rather than single alarming words: ``"required"`` alone fires
on the label "Required fields are marked". **A call to action is not a failure**, and
this list once could not tell them apart - it carried ``please (?:enter|select|...)``,
and Wikipedia's fundraising appeal says "Please select an amount (CAD)", so a run that
had just finished correctly was told its end screen was an error state.

Measured 2026-09-19 over 47 captured frames read by the shipped OCR, 42 with no error
and 5 with one: the original patterns fired on 3 of 42 ordinary frames and caught 1 of 5
errors; narrowing the ``please`` verbs dropped that to 1 of 42; adding ``not found`` and
``not available`` caught a second error frame while still firing on 1 of 42. Those two
are the only candidates of two dozen tried that caught a real error and fired on NO
ordinary frame. The one false positive left is a forum thread whose TITLE quotes an
error: a page discussing an error reads exactly like a page showing one.

**Recall is low for a reason that is not vocabulary.** Three of the four missed error
frames are missed because OCR runs words together - Wikipedia's permission page reads
``donothavepermissionto edit thispage``. Matching with all spacing removed recovers that
frame and costs one more false positive, which this corpus does not justify.
"""

_ERROR_RE = re.compile(r"\b(?:" + "|".join(ERROR_PHRASES) + r")\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class no_error_state:  # noqa: N801
    """Passes when ``after`` shows no error dialog, alert or validation message.

    Detection is by TEXT, because nothing upstream labels an element as a dialog, so a
    silent modal with no words in it is invisible here. A ``passed`` means "no error
    text", not "no error". With ``ignore_preexisting=True`` (the default) a phrase
    already on ``before`` does not fail the step. Unknown when ``after`` carries no text.
    """

    ignore_preexisting: bool = True
    name: str = field(default="no_error_state", init=False)

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        if not _screen_text(after):
            return _unknown(
                self.name,
                "cannot tell whether an error is showing: no element on the screen carries "
                "any text",
            )
        found = sorted({m.group(0).lower() for m in _ERROR_RE.finditer(_screen_text(after))})
        if not found:
            return _passed(self.name, "no error, alert or validation text is on the screen")
        if self.ignore_preexisting:
            was = {m.group(0).lower() for m in _ERROR_RE.finditer(_screen_text(before))}
            new = [phrase for phrase in found if phrase not in was]
            if not new:
                return _passed(
                    self.name,
                    f"error text is on the screen ({', '.join(found)}) but all of it was "
                    "already there before the step",
                )
            found = new
        quoted = ", ".join(repr(phrase) for phrase in found)
        return _failed(self.name, f"error text is on the screen: {quoted}")


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def _combine(name: str, results: Sequence[CheckVerdict], outcome: Outcome) -> CheckVerdict:
    """Build the composite verdict, quoting the parts that decided it."""
    deciding = [r for r in results if r.outcome is outcome]
    quoted = "; ".join(f"{r.name}: {r.reason}" for r in (deciding or results))
    if outcome is Outcome.unknown:
        return _unknown(name, quoted)
    confidence = min((r.confidence for r in deciding), default=1.0)
    build = _passed if outcome is Outcome.passed else _failed
    return build(name, quoted, confidence)


@dataclass(frozen=True, slots=True)
class all_of:  # noqa: N801
    """Kleene conjunction: fails if any part fails, unknown if any part is unknown.
    Ignorance propagates rather than being rounded down to a "no", so a merely incomplete
    set of checks escalates to the model instead of failing the step."""

    checks: tuple[Check, ...]
    name: str = field(default="all_of", init=False)

    def __init__(self, *checks: Check) -> None:
        object.__setattr__(self, "checks", tuple(checks))
        object.__setattr__(self, "name", "all_of")

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        if not self.checks:
            return _unknown(self.name, "no checks to run")
        results = [check(before, after) for check in self.checks]
        if any(r.outcome is Outcome.failed for r in results):
            return _combine(self.name, results, Outcome.failed)
        if any(r.unknown for r in results):
            return _combine(self.name, results, Outcome.unknown)
        return _combine(self.name, results, Outcome.passed)


@dataclass(frozen=True, slots=True)
class any_of:  # noqa: N801
    """Kleene disjunction: passes if any part passes, unknown if any part is unknown."""

    checks: tuple[Check, ...]
    name: str = field(default="any_of", init=False)

    def __init__(self, *checks: Check) -> None:
        object.__setattr__(self, "checks", tuple(checks))
        object.__setattr__(self, "name", "any_of")

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        if not self.checks:
            return _unknown(self.name, "no checks to run")
        results = [check(before, after) for check in self.checks]
        if any(r.outcome is Outcome.passed for r in results):
            return _combine(self.name, results, Outcome.passed)
        if any(r.unknown for r in results):
            return _combine(self.name, results, Outcome.unknown)
        return _combine(self.name, results, Outcome.failed)


@dataclass(frozen=True, slots=True)
class not_:  # noqa: N801
    """Kleene negation: swaps passed and failed, and leaves unknown untouched. Negating
    ignorance into a decision is the one thing this must never do."""

    check: Check
    name: str = field(default="not_", init=False)

    def __call__(self, before: Observation, after: Observation) -> CheckVerdict:
        result = self.check(before, after)
        if result.unknown:
            return _unknown(self.name, f"not {result.name}: {result.reason}")
        build = _failed if result.ok else _passed
        return build(self.name, f"not {result.name}: {result.reason}", result.confidence)


def run_all(checks: Iterable[Check], before: Observation, after: Observation) -> list[CheckVerdict]:
    """Run every check and return the verdicts in order. Never short-circuits: the
    critic wants the whole trail, not just the first thing that decided."""
    return [check(before, after) for check in checks]
