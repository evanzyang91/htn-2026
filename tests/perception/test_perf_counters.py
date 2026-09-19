"""Counting what perception did, so an optimization can be believed.

Every claim this project makes about perception getting cheaper is a claim about
COUNTS, not seconds. Seconds were tried first and thrown away: the same frame read on
a quiet machine and on one running a training job differ by more than any change in
this package could buy, and the profiling run that started this work was taken at load
average 9.8 and had to be discarded for exactly that reason. A count does not move.

So these tests hold the counters to being right, and hold the perceiver to charging
its work to them honestly - including the one number the whole exercise is judged by,
``ocr_reads``, which must fall when a screen has not changed and must NOT fall when it
has.

Nothing here loads a model, opens a browser or touches a network: the doubles in
``tests/fakes`` are the whole world.
"""

from __future__ import annotations

import dataclasses
import threading

import pytest

from skillweaver.contracts import Element, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.orchestrator import ComposedPerceiver, perception_counts
from skillweaver.perception.ocr import (
    DEFAULT_CACHE_SIZE,
    CachingTextReader,
    PerceptionCounters,
    PerceptionCounts,
)
from tests.fakes import FakeController, FakeDetector, FakeTextReader, Scenario

# --------------------------------------------------------------------------------------
# The value type a run reports
# --------------------------------------------------------------------------------------


def test_counts_add_and_subtract_so_one_attempt_can_be_taken_out_of_a_run() -> None:
    before = PerceptionCounts(observations=3, captures=3, detections=3, ocr_reads=2, ocr_hits=1)
    after = PerceptionCounts(observations=8, captures=8, detections=8, ocr_reads=4, ocr_hits=4)
    assert after - before == PerceptionCounts(
        observations=5, captures=5, detections=5, ocr_reads=2, ocr_hits=3
    )
    assert before + (after - before) == after


def test_subtracting_a_later_mark_reports_no_work_rather_than_negative_work() -> None:
    """Counters only climb, so a mark from another counter is nonsense, not a deficit."""
    assert PerceptionCounts(ocr_reads=1) - PerceptionCounts(ocr_reads=9) == PerceptionCounts()


def test_hit_rate_is_the_fraction_of_reads_the_cache_answered() -> None:
    assert PerceptionCounts(ocr_reads=3, ocr_hits=9).hit_rate == 0.75
    assert PerceptionCounts(ocr_reads=3, ocr_hits=9).text_reads == 12
    assert PerceptionCounts().hit_rate == 0.0, "no reads is 0.0, not a division by zero"


def test_an_empty_tally_is_falsy_so_a_report_can_stay_silent() -> None:
    assert not PerceptionCounts()
    assert PerceptionCounts(observations=1)
    assert PerceptionCounts(ocr_hits=1)


def test_counters_snapshot_and_measure_since_a_mark() -> None:
    counters = PerceptionCounters()
    counters.observations = 2
    counters.ocr_reads = 2
    mark = counters.snapshot()
    counters.observations += 3
    counters.ocr_hits += 3
    assert counters.since(mark) == PerceptionCounts(observations=3, ocr_hits=3)
    assert mark == PerceptionCounts(observations=2, ocr_reads=2), "a snapshot is a copy"
    counters.reset()
    assert counters.snapshot() == PerceptionCounts()


def test_a_snapshot_is_immutable_and_a_counter_is_not() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PerceptionCounts().observations = 1  # type: ignore[misc]
    counters = PerceptionCounters()
    counters.observations = 1
    assert counters.observations == 1


# --------------------------------------------------------------------------------------
# What the perceiver charges
# --------------------------------------------------------------------------------------


@pytest.fixture
def perceiver(scenario: Scenario) -> ComposedPerceiver:
    """The real ``ComposedPerceiver`` over the fake scenario's detector and reader."""
    return ComposedPerceiver(
        FakeDetector(scenario.controller),
        FakeTextReader(scenario.controller),
        scenario.perceiver.fingerprinter,
    )


def test_every_observation_charges_a_capture_and_a_detection(
    perceiver: ComposedPerceiver, fake_controller: FakeController
) -> None:
    for _ in range(3):
        perceiver.observe(fake_controller)
    counts = perceiver.counters.snapshot()
    assert (counts.observations, counts.captures, counts.detections) == (3, 3, 3)


def test_observing_an_unchanged_screen_costs_one_ocr_read_not_three(
    perceiver: ComposedPerceiver, fake_controller: FakeController
) -> None:
    """The point of the whole exercise, stated as the number that proves it."""
    seen = [perceiver.observe(fake_controller) for _ in range(3)]
    counts = perceiver.counters.snapshot()

    assert (counts.ocr_reads, counts.ocr_hits) == (1, 2)
    assert counts.hit_rate == pytest.approx(2 / 3)
    assert all(o.elements == seen[0].elements for o in seen), (
        "a cheaper observation is not a smaller one"
    )
    assert all(o.fingerprint == seen[0].fingerprint for o in seen)


def test_a_screen_that_changed_is_read_again(
    perceiver: ComposedPerceiver, scenario: Scenario
) -> None:
    """The correctness bar: acting on the screen must invalidate what was remembered."""
    controller = scenario.controller
    before = perceiver.observe(controller)
    controller.perform(scenario.solution[0])
    after = perceiver.observe(controller)

    counts = perceiver.counters.snapshot()
    assert (counts.ocr_reads, counts.ocr_hits) == (2, 0), "the new screen paid for itself"
    assert after.elements != before.elements
    assert after.fingerprint != before.fingerprint


def test_going_back_to_a_screen_still_in_the_cache_is_free(
    perceiver: ComposedPerceiver, scenario: Scenario
) -> None:
    controller = scenario.controller
    first = perceiver.observe(controller)
    controller.perform(scenario.solution[0])
    perceiver.observe(controller)
    controller.reset()
    again = perceiver.observe(controller)

    counts = perceiver.counters.snapshot()
    assert (counts.ocr_reads, counts.ocr_hits) == (2, 1)
    assert again.elements == first.elements


def test_a_perceiver_with_no_reader_never_charges_a_read(
    scenario: Scenario, fake_controller: FakeController
) -> None:
    """Detection alone is what a machine without the OCR models can still do."""
    perceiver = ComposedPerceiver(FakeDetector(scenario.controller), None)
    perceiver.observe(fake_controller)
    counts = perceiver.counters.snapshot()
    assert (counts.detections, counts.ocr_reads, counts.ocr_hits) == (1, 0, 0)
    assert perceiver.reader is None


def test_cache_size_zero_reads_every_frame_which_is_how_the_before_arm_is_measured(
    scenario: Scenario, fake_controller: FakeController
) -> None:
    perceiver = ComposedPerceiver(
        FakeDetector(scenario.controller),
        FakeTextReader(scenario.controller),
        cache_size=0,
    )
    for _ in range(4):
        perceiver.observe(fake_controller)
    counts = perceiver.counters.snapshot()
    assert (counts.ocr_reads, counts.ocr_hits) == (4, 0)


def test_an_already_caching_reader_is_reused_rather_than_wrapped_twice(
    scenario: Scenario, fake_controller: FakeController
) -> None:
    """Two perceivers sharing one warm cache report one tally, not two half-tallies."""
    shared = CachingTextReader(FakeTextReader(scenario.controller))
    one = ComposedPerceiver(FakeDetector(scenario.controller), shared)
    two = ComposedPerceiver(FakeDetector(scenario.controller), shared)

    one.observe(fake_controller)
    two.observe(fake_controller)

    assert one.reader is shared and two.reader is shared
    assert one.counters is two.counters
    counts = one.counters.snapshot()
    assert (counts.observations, counts.ocr_reads, counts.ocr_hits) == (2, 1, 1)


def test_a_broken_reader_still_raises_through_the_cache(
    scenario: Scenario, fake_controller: FakeController
) -> None:
    """Failure stays loud: OCR must never look like "this screen has no text"."""

    class Broken:
        def read(self, _shot: Screenshot) -> list[Element]:
            raise PerceptionError("the engine is gone")

    perceiver = ComposedPerceiver(FakeDetector(scenario.controller), Broken())
    with pytest.raises(PerceptionError, match="the engine is gone"):
        perceiver.observe(fake_controller)


def test_the_default_cache_is_small_and_bounded() -> None:
    assert 0 < DEFAULT_CACHE_SIZE <= 64


def test_concurrent_readers_of_one_cache_all_get_the_true_read(
    scenario: Scenario, fake_controller: FakeController
) -> None:
    """``Observation`` is shared across threads in this project; the cache must be too."""
    caching = CachingTextReader(FakeTextReader(scenario.controller))
    shot = fake_controller.capture()
    results: list[list[Element]] = []
    barrier = threading.Barrier(8)

    def read() -> None:
        barrier.wait()
        results.append(caching.read(shot))

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r == results[0] for r in results)


# --------------------------------------------------------------------------------------
# Asking a perceiver that does not count
# --------------------------------------------------------------------------------------


def test_a_perceiver_without_counters_reports_nothing_rather_than_failing(
    fake_perceiver: object,
) -> None:
    """``Perceiver`` promises no counters, so a report asks politely."""
    assert perception_counts(fake_perceiver) == PerceptionCounts()  # type: ignore[arg-type]
    assert perception_counts(None) == PerceptionCounts()


def test_a_perceiver_whose_counters_are_nonsense_is_treated_as_not_counting() -> None:
    class Odd:
        counters = "not a tally"

    assert perception_counts(Odd()) == PerceptionCounts()  # type: ignore[arg-type]


def test_a_shared_cache_cannot_be_pointed_at_two_tallies_at_once(scenario: Scenario) -> None:
    """A tally only some of the work reaches is worse than no tally at all."""
    shared = CachingTextReader(FakeTextReader(scenario.controller))
    with pytest.raises(ValueError, match="already charges its reads"):
        ComposedPerceiver(FakeDetector(scenario.controller), shared, counters=PerceptionCounters())
    # Passing the reader's own tally is the way to say "count these together".
    twin = ComposedPerceiver(FakeDetector(scenario.controller), shared, counters=shared.counters)
    assert twin.counters is shared.counters
