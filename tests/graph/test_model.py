"""The in-memory graph: node identity under drifting fingerprints, and the
incremental statistics that routing then depends on."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from skillweaver.contracts import Route, SiteGraph, Transition, UIState
from skillweaver.graph.model import (
    DEFAULT_MATCH_THRESHOLD,
    GraphSnapshot,
    InMemorySiteGraph,
    merge_transitions,
)
from skillweaver.graph.route import EXPLORATORY
from tests.graph.builders import DOMAIN, click, edge, fp, screen, state

LIST, DETAIL, EDIT = screen("list"), screen("detail"), screen("edit")


@pytest.fixture
def graph() -> InMemorySiteGraph:
    return InMemorySiteGraph()


def drifted(base, **changes: str):
    """``base`` with some ``parts`` replaced and a different ``value``, as a
    re-observation of the same screen produces."""
    parts = dict(base.parts) | changes
    return fp(base.value + ":again", **parts)


# -- the Protocol --------------------------------------------------------------------


def test_the_graph_satisfies_the_sitegraph_protocol(graph):
    assert isinstance(graph, SiteGraph)


# -- node identity -------------------------------------------------------------------


def test_a_near_match_fingerprint_resolves_to_the_existing_node(graph):
    graph.upsert_state(state(LIST, label="invoice list"))

    # One of four parts changed: similarity 3/4 = 0.75, at the default threshold.
    same_screen = drifted(LIST, text="t:list-but-the-badge-ticked")
    assert same_screen.similarity(LIST) == pytest.approx(0.75)

    resolved = graph.resolve(same_screen)

    assert resolved is not None
    assert resolved.fingerprint == LIST
    assert resolved.label == "invoice list"
    assert graph.canonical(same_screen) == LIST


def test_a_genuinely_different_fingerprint_creates_a_new_node(graph):
    graph.upsert_state(state(LIST))

    assert graph.resolve(DETAIL) is None

    graph.upsert_state(state(DETAIL))

    assert {s.fingerprint.value for s in graph.states(DOMAIN)} == {"list", "detail"}


def test_upserting_a_near_match_enriches_the_existing_node_instead_of_forking_it(graph):
    graph.upsert_state(state(LIST, label="invoice list", day=1))

    graph.upsert_state(state(drifted(LIST, text="drift"), label="", day=9))

    states = graph.states(DOMAIN)
    assert len(states) == 1
    assert states[0].fingerprint == LIST  # the first fingerprint stays canonical
    assert states[0].label == "invoice list"
    assert states[0].first_seen == datetime(2026, 1, 1, tzinfo=UTC)


def test_upserting_a_known_state_fills_in_missing_details_and_keeps_first_seen(graph):
    graph.upsert_state(state(LIST, day=1))

    stored = graph.upsert_state(
        UIState(
            fingerprint=LIST,
            domain=DOMAIN,
            label="inbox",
            url_pattern="/inbox",
            first_seen=datetime(2026, 6, 1, tzinfo=UTC),
            thumbnail=b"png",
        )
    )

    assert stored.label == "inbox"
    assert stored.url_pattern == "/inbox"
    assert stored.thumbnail == b"png"
    assert stored.first_seen == datetime(2026, 1, 1, tzinfo=UTC)


def test_the_match_threshold_is_configurable(graph):
    strict = InMemorySiteGraph(match_threshold=0.9)
    loose = InMemorySiteGraph(match_threshold=0.5)
    for g in (strict, loose):
        g.upsert_state(state(LIST))

    drift = drifted(LIST, text="drift")  # similarity 0.75

    assert strict.resolve(drift) is None
    assert loose.resolve(drift) is not None


def test_a_threshold_of_one_disables_near_matching(graph):
    exact = InMemorySiteGraph(match_threshold=1.0)
    exact.upsert_state(state(LIST))

    assert exact.resolve(drifted(LIST, text="drift")) is None
    assert exact.resolve(LIST) is not None


def test_the_nearest_node_wins_when_several_are_above_the_threshold():
    graph = InMemorySiteGraph(match_threshold=0.5)
    graph.upsert_state(state(LIST))
    graph.upsert_state(state(fp("half", url="u:list", layout="l:list", text="x", chrome="y")))

    # Differs from LIST in one part (0.75) and from "half" in two (0.5).
    probe = drifted(LIST, text="drift")

    assert graph.canonical(probe) == LIST


def test_resolve_never_raises_for_a_fingerprint_without_parts(graph):
    graph.upsert_state(state(LIST))

    assert graph.resolve(fp("bare")) is None


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_a_nonsensical_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="match_threshold"):
        InMemorySiteGraph(match_threshold=bad)


def test_the_default_threshold_suits_a_three_part_fingerprint():
    """The fingerprinters in this project emit ``url``, ``layout`` and ``text``, so
    the default has to sit between one part drifting and two. A default above
    0.667 rejects every drift a three-part fingerprint can express."""
    assert 1 / 3 < DEFAULT_MATCH_THRESHOLD <= 2 / 3


def test_a_real_three_part_fingerprint_resolves_when_one_part_drifts(graph):
    settled = fp(value="v1", url="u", layout="l", text="t1")
    graph.upsert_state(state(settled))

    ticked = fp(value="v2", url="u", layout="l", text="t2")
    assert ticked.similarity(settled) == pytest.approx(1 / 3 * 2)

    assert graph.canonical(ticked) == settled


def test_a_real_three_part_fingerprint_forks_when_two_parts_drift(graph):
    settled = fp(value="v1", url="u", layout="l", text="t1")
    graph.upsert_state(state(settled))

    elsewhere = fp(value="v2", url="u", layout="l2", text="t2")
    assert elsewhere.similarity(settled) == pytest.approx(1 / 3)

    assert graph.resolve(elsewhere) is None


# -- observe_transition --------------------------------------------------------------


def test_observing_a_transition_creates_both_states(graph):
    graph.upsert_state(state(LIST))

    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=120.0)

    assert {s.fingerprint.value for s in graph.states(DOMAIN)} == {"list", "detail"}


def test_an_implicitly_created_state_inherits_the_source_domain(graph):
    graph.upsert_state(state(LIST))

    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=120.0)

    assert graph.states(DOMAIN)[-1].domain == DOMAIN


def test_an_implicitly_created_state_with_no_known_source_has_no_domain(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=120.0)

    assert graph.states("") == sorted(graph.states(""), key=lambda s: s.first_seen)
    assert {s.fingerprint.value for s in graph.states("")} == {"list", "detail"}


def test_statistics_fold_incrementally(graph):
    actions = (click(1, 1),)

    first = graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=100.0)
    assert (first.attempts, first.successes, first.mean_ms) == (1, 1, 100.0)

    second = graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=200.0)
    assert (second.attempts, second.successes, second.mean_ms) == (2, 2, 150.0)

    third = graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=300.0)
    assert (third.attempts, third.successes) == (3, 3)
    assert third.mean_ms == pytest.approx(200.0)  # (100 + 200 + 300) / 3


def test_a_failure_counts_as_an_attempt_but_does_not_move_the_mean(graph):
    actions = (click(1, 1),)
    graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=100.0)

    after = graph.observe_transition(LIST, actions, DETAIL, ok=False, ms=9999.0)

    assert (after.attempts, after.successes) == (2, 1)
    assert after.mean_ms == pytest.approx(100.0)


def test_a_failure_does_not_move_last_verified(graph):
    actions = (click(1, 1),)
    verified = graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=100.0).last_verified

    after = graph.observe_transition(LIST, actions, DETAIL, ok=False, ms=50.0)

    assert after.last_verified == verified


def test_an_edge_that_has_only_ever_failed_has_no_verification_stamp(graph):
    after = graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=False, ms=50.0)

    assert after.successes == 0
    assert after.last_verified is None
    assert after.mean_ms == 0.0


def test_different_actions_between_the_same_states_are_different_edges(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    graph.observe_transition(LIST, (click(2, 2),), DETAIL, ok=True, ms=100.0)

    assert len(graph.neighbors(LIST)) == 2


def test_a_drifted_fingerprint_reinforces_the_existing_edge(graph):
    """The whole point of near matching: a second run must strengthen what the
    first run learned, not lay a parallel edge beside it."""
    actions = (click(1, 1),)
    graph.observe_transition(LIST, actions, DETAIL, ok=True, ms=100.0)

    again = graph.observe_transition(
        drifted(LIST, text="drift"), actions, drifted(DETAIL, text="drift"), ok=True, ms=200.0
    )

    assert len(graph.transitions()) == 1
    assert (again.attempts, again.successes, again.mean_ms) == (2, 2, 150.0)
    assert (again.src, again.dst) == (LIST, DETAIL)


# -- reads ---------------------------------------------------------------------------


def test_neighbors_are_most_reliable_first(graph):
    actions_good, actions_bad = (click(1, 1),), (click(2, 2),)
    graph.observe_transition(LIST, actions_bad, EDIT, ok=False, ms=10.0)
    graph.observe_transition(LIST, actions_bad, EDIT, ok=True, ms=10.0)
    graph.observe_transition(LIST, actions_good, DETAIL, ok=True, ms=10.0)

    assert [e.dst.value for e in graph.neighbors(LIST)] == ["detail", "edit"]


def test_neighbors_of_an_unknown_state_is_empty(graph):
    assert graph.neighbors(screen("nowhere")) == []


def test_neighbors_can_resolve_a_drifted_fingerprint_on_request(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    drift = drifted(LIST, text="drift")

    assert graph.neighbors(drift) == []
    assert len(graph.neighbors(drift, approximate=True)) == 1


def test_states_are_oldest_first(graph):
    graph.upsert_state(state(EDIT, day=3))
    graph.upsert_state(state(LIST, day=1))
    graph.upsert_state(state(DETAIL, day=2))

    assert [s.fingerprint.value for s in graph.states(DOMAIN)] == ["list", "detail", "edit"]


def test_states_of_an_unknown_domain_is_empty(graph):
    graph.upsert_state(state(LIST))

    assert graph.states("other.test") == []


def test_domains_lists_every_domain_alphabetically(graph):
    graph.upsert_state(state(LIST, domain="b.test"))
    graph.upsert_state(state(DETAIL, domain="a.test"))

    assert graph.domains() == ["a.test", "b.test"]


def test_transitions_can_be_filtered_by_domain(graph):
    graph.upsert_state(state(LIST, domain="a.test"))
    graph.upsert_state(state(EDIT, domain="b.test"))
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=10.0)
    graph.observe_transition(EDIT, (click(2, 2),), screen("other"), ok=True, ms=10.0)

    assert len(graph.transitions()) == 2
    assert [e.src.value for e in graph.transitions("a.test")] == ["list"]


# -- routing through the graph -------------------------------------------------------


def test_route_finds_a_learned_path(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    graph.observe_transition(DETAIL, (click(2, 2),), EDIT, ok=True, ms=200.0)

    route = graph.route(LIST, EDIT)

    assert route is not None
    assert route.steps == (click(1, 1), click(2, 2))
    assert route.cost == pytest.approx(300.0)


def test_route_to_an_unreachable_target_returns_none(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)

    assert graph.route(LIST, EDIT) is None


def test_route_from_an_unknown_state_returns_none_rather_than_raising(graph):
    assert graph.route(screen("nowhere"), LIST) is None


def test_route_to_the_same_state_is_free(graph):
    assert graph.route(LIST, LIST) == Route((), 0.0, ())


def test_route_ignores_an_edge_that_has_never_succeeded(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=False, ms=100.0)

    assert graph.route(LIST, DETAIL) is None


def test_route_matches_exactly_by_default_and_approximately_on_request(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    drift = drifted(LIST, text="drift")

    assert graph.route(drift, DETAIL) is None
    assert graph.route(drift, DETAIL, approximate=True) is not None


def test_a_graph_can_be_built_with_a_different_routing_policy():
    graph = InMemorySiteGraph(policy=EXPLORATORY)
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=False, ms=100.0)

    assert graph.route(LIST, DETAIL) is not None


# -- merge_transitions ---------------------------------------------------------------


def test_merging_an_edge_sums_attempts_and_successes():
    left = edge(LIST, DETAIL, attempts=4, successes=3, mean_ms=100.0, day=1)
    right = edge(LIST, DETAIL, attempts=6, successes=5, mean_ms=200.0, day=5)

    merged = merge_transitions(left, right)

    assert (merged.attempts, merged.successes) == (10, 8)
    # Success-weighted: (100 * 3 + 200 * 5) / 8 = 1300 / 8 = 162.5
    assert merged.mean_ms == pytest.approx(162.5)
    assert merged.last_verified == datetime(2026, 1, 5, tzinfo=UTC)


def test_merging_two_never_verified_edges_keeps_a_zero_mean():
    left = edge(LIST, DETAIL, attempts=2, successes=0, mean_ms=0.0, day=None)
    right = edge(LIST, DETAIL, attempts=3, successes=0, mean_ms=0.0, day=None)

    merged = merge_transitions(left, right)

    assert (merged.attempts, merged.successes, merged.mean_ms) == (5, 0, 0.0)
    assert merged.last_verified is None


def test_merging_different_edges_is_refused():
    with pytest.raises(ValueError, match="same edge"):
        merge_transitions(edge(LIST, DETAIL), edge(LIST, EDIT))


def test_merging_edges_with_different_actions_is_refused():
    with pytest.raises(ValueError, match="same edge"):
        merge_transitions(
            edge(LIST, DETAIL, actions=(click(1, 1),)),
            edge(LIST, DETAIL, actions=(click(2, 2),)),
        )


# -- snapshots and absorb ------------------------------------------------------------


def test_a_snapshot_holds_one_domain(graph):
    graph.upsert_state(state(LIST, domain="a.test"))
    graph.upsert_state(state(EDIT, domain="b.test"))

    snap = graph.snapshot("a.test")

    assert snap.domain == "a.test"
    assert [s.fingerprint.value for s in snap.states] == ["list"]


def test_absorbing_a_snapshot_is_verbatim_by_default(graph):
    """Two near-matching screens stored side by side must survive a reload;
    collapsing them here would make a round-trip lossy."""
    close = drifted(LIST, text="drift")
    snap = GraphSnapshot(DOMAIN, states=(state(LIST), state(close, day=2)))

    graph.absorb(snap)

    assert len(graph.states(DOMAIN)) == 2


def test_absorbing_with_resolve_consolidates_near_matches(graph):
    graph.upsert_state(state(LIST, label="invoice list", day=1))
    snap = GraphSnapshot(DOMAIN, states=(state(drifted(LIST, text="drift"), label="relabelled"),))

    graph.absorb(snap, resolve=True)

    merged = graph.states(DOMAIN)
    assert len(merged) == 1
    assert merged[0].fingerprint == LIST  # the node keeps its canonical id
    assert merged[0].label == "relabelled"  # but an incoming label is an update
    assert merged[0].first_seen == datetime(2026, 1, 1, tzinfo=UTC)


def test_absorbing_an_edge_twice_sums_its_statistics(graph):
    snap = GraphSnapshot(
        DOMAIN,
        states=(state(LIST), state(DETAIL, day=2)),
        transitions=(edge(LIST, DETAIL, attempts=2, successes=2, mean_ms=100.0),),
    )

    graph.absorb(snap)
    graph.absorb(snap)

    assert len(graph.transitions()) == 1
    assert graph.transitions()[0].attempts == 4


def test_absorbing_an_edge_creates_its_endpoint_states(graph):
    snap = GraphSnapshot(DOMAIN, transitions=(edge(LIST, DETAIL),))

    graph.absorb(snap)

    assert len(graph.states(DOMAIN)) == 2


# -- forget --------------------------------------------------------------------------


def test_forget_drops_a_domains_states_and_edges(graph):
    graph.upsert_state(state(LIST, domain="a.test"))
    graph.upsert_state(state(DETAIL, domain="a.test", day=2))
    graph.upsert_state(state(EDIT, domain="b.test"))
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=10.0)

    graph.forget("a.test")

    assert graph.states("a.test") == []
    assert graph.transitions() == []
    assert len(graph.states("b.test")) == 1


def test_forgetting_an_unknown_domain_is_harmless(graph):
    graph.upsert_state(state(LIST))

    graph.forget("nothing.test")

    assert len(graph.states(DOMAIN)) == 1


# -- storeless graph -----------------------------------------------------------------


def test_save_without_a_store_is_a_no_op(graph):
    graph.upsert_state(state(LIST))

    graph.save()

    assert len(graph.states(DOMAIN)) == 1


def test_load_without_a_store_empties_the_domain(graph):
    graph.upsert_state(state(LIST))

    graph.load(DOMAIN)

    assert graph.states(DOMAIN) == []


def test_repr_reports_the_size(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=10.0)

    assert "states=2" in repr(graph)
    assert "edges=1" in repr(graph)


def test_transitions_are_plain_contract_transitions(graph):
    graph.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=10.0)

    assert isinstance(graph.transitions()[0], Transition)
