"""Routing: the cost model, the unverified penalty and the hopelessness cap.

Every expected cost here is written as the arithmetic that produces it, so a
change to the cost model fails with a number that says what it should have been.
"""

from __future__ import annotations

import pytest

from skillweaver.contracts import Fingerprint, Route
from skillweaver.errors import RouteNotFound
from skillweaver.graph.route import (
    EXPLORATORY,
    VERIFIED_ONLY,
    RoutingPolicy,
    edge_cost,
    find_route,
    is_verified,
    require_route,
    same_state,
    success_rate,
)
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD
from tests.graph.builders import click, edge, live_pair, screen

A, B, C, D = screen("a"), screen("b"), screen("c"), screen("d")


def outgoing_from(*edges):
    """A routing edge lookup over a fixed list of transitions."""
    return lambda value: [e for e in edges if e.src.value == value]


# -- the cost model ----------------------------------------------------------------


def test_cost_is_latency_divided_by_success_rate():
    flaky = edge(A, B, attempts=10, successes=3, mean_ms=300.0)
    assert success_rate(flaky) == pytest.approx(0.3)
    # 300ms per success, but it only works 3 times in 10, so ~3.33 walks per arrival.
    assert edge_cost(flaky) == pytest.approx(300.0 / 0.3)
    assert edge_cost(flaky) == pytest.approx(1000.0)


def test_perfect_edge_costs_exactly_its_latency():
    assert edge_cost(edge(A, B, attempts=8, successes=8, mean_ms=420.0)) == pytest.approx(420.0)


def test_routing_prefers_a_slower_reliable_edge_over_a_faster_flaky_one():
    reliable = edge(A, B, actions=(click(1, 1),), attempts=10, successes=10, mean_ms=900.0)
    flaky = edge(A, B, actions=(click(2, 2),), attempts=10, successes=3, mean_ms=300.0)

    # By hand: reliable costs 900 / 1.0 = 900.0; flaky costs 300 / 0.3 = 1000.0.
    assert edge_cost(reliable) == pytest.approx(900.0)
    assert edge_cost(flaky) == pytest.approx(1000.0)

    route = find_route(A, B, outgoing_from(flaky, reliable))

    assert route is not None
    assert route.cost == pytest.approx(900.0)
    assert route.edges == (reliable,)
    assert route.steps == (click(1, 1),)


def test_a_fast_flaky_edge_still_wins_when_it_is_cheap_enough():
    """The preference is arithmetic, not a bias against flakiness."""
    reliable = edge(A, B, actions=(click(1, 1),), attempts=10, successes=10, mean_ms=900.0)
    flaky = edge(A, B, actions=(click(2, 2),), attempts=10, successes=5, mean_ms=400.0)

    route = find_route(A, B, outgoing_from(reliable, flaky))

    assert route is not None
    assert route.cost == pytest.approx(400.0 / 0.5)  # 800.0, under the reliable 900.0
    assert route.edges == (flaky,)


def test_cost_accumulates_over_a_multi_hop_route():
    first = edge(A, B, attempts=2, successes=1, mean_ms=100.0)  # 100 / 0.5 = 200
    second = edge(B, C, attempts=4, successes=4, mean_ms=50.0)  # 50 / 1.0  =  50

    route = find_route(A, C, outgoing_from(first, second))

    assert route is not None
    assert route.cost == pytest.approx(250.0)
    assert route.edges == (first, second)


def test_shorter_hop_count_loses_to_cheaper_total_cost():
    direct = edge(A, C, actions=(click(9, 9),), attempts=10, successes=1, mean_ms=100.0)
    hop_one = edge(A, B, actions=(click(1, 1),), attempts=10, successes=10, mean_ms=60.0)
    hop_two = edge(B, C, actions=(click(2, 2),), attempts=10, successes=10, mean_ms=60.0)

    route = find_route(A, C, outgoing_from(direct, hop_one, hop_two))

    assert route is not None
    # Direct costs 100 / 0.1 = 1000.0; two reliable hops cost 60 + 60 = 120.0.
    assert route.cost == pytest.approx(120.0)
    assert route.steps == (click(1, 1), click(2, 2))


# -- never-verified edges ------------------------------------------------------------


def test_an_unverified_edge_is_not_used_by_default():
    unverified = edge(A, B, attempts=1, successes=0, mean_ms=0.0)

    assert not is_verified(unverified)
    assert edge_cost(unverified) is None
    assert find_route(A, B, outgoing_from(unverified)) is None


def test_an_unverified_edge_loses_to_a_proven_one_of_equal_nominal_cost():
    policy = RoutingPolicy(allow_unverified=True, unverified_ms=1000.0, unverified_penalty=3.0)
    proven = edge(A, B, actions=(click(1, 1),), attempts=5, successes=5, mean_ms=1000.0)
    unverified = edge(A, B, actions=(click(2, 2),), attempts=1, successes=0, mean_ms=0.0)

    # Equal nominal cost: the proven edge takes 1000ms, the unverified edge is
    # assumed to take the same 1000ms. Only the penalty separates them.
    assert edge_cost(proven, policy) == pytest.approx(1000.0)
    assert edge_cost(unverified, policy) == pytest.approx(3000.0)

    route = find_route(A, B, outgoing_from(unverified, proven), policy)

    assert route is not None
    assert route.cost == pytest.approx(1000.0)
    assert route.edges == (proven,)


def test_an_unverified_edge_is_used_when_it_is_the_only_way_through():
    unverified = edge(A, B, attempts=1, successes=0, mean_ms=0.0)

    assert find_route(A, B, outgoing_from(unverified), EXPLORATORY) is not None


def test_an_unverified_edge_is_priced_from_the_policy_not_its_mean_ms():
    """``mean_ms`` averages successful traversals only, so an unverified edge has
    none and must not be priced at its stored 0.0 - that would make it free."""
    unverified = edge(A, B, attempts=1, successes=0, mean_ms=0.0)

    cheap = RoutingPolicy(allow_unverified=True, unverified_ms=10.0, unverified_penalty=2.0)
    dear = RoutingPolicy(allow_unverified=True, unverified_ms=5000.0, unverified_penalty=2.0)

    assert edge_cost(unverified, cheap) == pytest.approx(20.0)
    assert edge_cost(unverified, dear) == pytest.approx(10000.0)


# -- the hopelessness cap ------------------------------------------------------------


def test_a_hopeless_edge_is_refused_even_though_it_has_successes():
    hopeless = edge(A, B, attempts=100, successes=1, mean_ms=10.0)

    assert success_rate(hopeless) == pytest.approx(0.01)
    assert edge_cost(hopeless) is None
    assert find_route(A, B, outgoing_from(hopeless)) is None


def test_the_cap_is_configurable():
    marginal = edge(A, B, attempts=10, successes=3, mean_ms=300.0)

    assert edge_cost(marginal, RoutingPolicy(min_success_rate=0.2)) == pytest.approx(1000.0)
    assert edge_cost(marginal, RoutingPolicy(min_success_rate=0.5)) is None
    assert edge_cost(marginal, RoutingPolicy(min_success_rate=0.0)) == pytest.approx(1000.0)


def test_the_cap_waits_for_enough_attempts_before_condemning_an_edge():
    """One unlucky attempt is not evidence; three failures in a row are."""
    unlucky = edge(A, B, attempts=1, successes=0, mean_ms=0.0)
    condemned = edge(A, B, attempts=3, successes=0, mean_ms=0.0)
    policy = RoutingPolicy(allow_unverified=True, min_attempts_for_cap=3)

    assert edge_cost(unlucky, policy) is not None
    assert edge_cost(condemned, policy) is None


def test_routing_detours_around_a_hopeless_edge():
    hopeless = edge(A, C, actions=(click(9, 9),), attempts=50, successes=1, mean_ms=10.0)
    hop_one = edge(A, B, actions=(click(1, 1),), attempts=5, successes=5, mean_ms=800.0)
    hop_two = edge(B, C, actions=(click(2, 2),), attempts=5, successes=5, mean_ms=800.0)

    route = find_route(A, C, outgoing_from(hopeless, hop_one, hop_two))

    assert route is not None
    assert route.cost == pytest.approx(1600.0)
    assert route.edges == (hop_one, hop_two)


# -- no route ------------------------------------------------------------------------


def test_an_unreachable_target_returns_none():
    assert find_route(A, D, outgoing_from(edge(A, B), edge(B, C))) is None


def test_an_unknown_source_returns_none_rather_than_raising():
    assert find_route(screen("nowhere"), A, outgoing_from(edge(A, B))) is None


def test_a_backwards_edge_does_not_make_a_route():
    assert find_route(B, A, outgoing_from(edge(A, B))) is None


def test_a_route_from_a_state_to_itself_is_empty_and_free():
    assert find_route(A, A, outgoing_from()) == Route((), 0.0, ())


def test_a_route_to_itself_is_free_even_for_an_unknown_state():
    assert find_route(screen("unseen"), screen("unseen"), outgoing_from()) == Route((), 0.0, ())


# -- already there -------------------------------------------------------------------


def test_the_screen_the_agent_is_standing_on_needs_no_route():
    """THE defect. A skill records the screen it starts from; on the next run the agent
    is looking at that screen and fingerprints it afresh, so the two ids differ. Asked
    for a route between them, exact matching answers ``None`` - and the planner reports
    ``no_route`` for a journey of nought steps, which is how the first skill ever learned
    on live Wikipedia became unusable the moment it was stored."""
    recorded, seen_again = live_pair(agreeing=120)
    assert recorded.value != seen_again.value
    assert seen_again.similarity(recorded) > SAME_STATE_THRESHOLD

    assert find_route(seen_again, recorded, outgoing_from()) == Route((), 0.0, ())


def test_two_screens_that_merely_resemble_each_other_are_not_one_screen():
    """The tempting wrong fix, refused: loosen this and a skill runs on a screen it was
    never written for, which on a real site means clicking real things."""
    recorded, elsewhere = live_pair(agreeing=60)
    assert elsewhere.similarity(recorded) < SAME_STATE_THRESHOLD

    assert find_route(elsewhere, recorded, outgoing_from()) is None


def test_exactness_can_still_be_demanded():
    """``InMemorySiteGraph.route`` needs it: ``contracts.GraphView.route`` specifies
    matching on ``Fingerprint.value``, and that promise is kept."""
    recorded, seen_again = live_pair(agreeing=120)

    assert find_route(seen_again, recorded, outgoing_from(), same_state_threshold=1.0) is None
    assert find_route(recorded, recorded, outgoing_from(), same_state_threshold=1.0) == Route(
        (), 0.0, ()
    )


def test_a_fingerprint_without_parts_never_matches_by_similarity():
    """Half this project reconstructs ``Fingerprint(value)`` to use as a node key. Such a
    fingerprint carries no evidence at all, and no evidence must never read as agreement."""
    bare_a, bare_b = Fingerprint("a"), Fingerprint("b")
    assert same_state(bare_a, bare_b) is False
    assert find_route(bare_a, bare_b, outgoing_from()) is None


def test_endpoints_can_be_resolved_onto_the_nodes_they_belong_to():
    """Only the ENDPOINTS are fuzzy; the hops between them are node ids and are matched
    exactly. A caller holding live fingerprints hands over the mapping."""
    drifted, _ = live_pair(agreeing=120)
    edges = outgoing_from(edge(A, B))

    assert find_route(drifted, B, edges) is None
    route = find_route(drifted, B, edges, resolve=lambda f: A if f is drifted else f)
    assert route is not None
    assert route.edges == (edge(A, B),)


def test_a_cycle_does_not_hang_the_search():
    edges = outgoing_from(edge(A, B), edge(B, A), edge(B, C))

    route = find_route(A, C, edges)

    assert route is not None
    assert route.edges[0].dst.value == "b"


def test_a_self_loop_is_ignored():
    route = find_route(A, B, outgoing_from(edge(A, A, actions=(click(0, 0),)), edge(A, B)))

    assert route is not None
    assert len(route.edges) == 1


# -- require_route -------------------------------------------------------------------


def test_require_route_returns_the_same_route_as_find_route():
    edges = outgoing_from(edge(A, B))

    assert require_route(A, B, edges) == find_route(A, B, edges)


def test_require_route_raises_when_there_is_no_route():
    with pytest.raises(RouteNotFound, match="no known route"):
        require_route(A, D, outgoing_from(edge(A, B)))


def test_the_error_names_both_endpoints():
    with pytest.raises(RouteNotFound) as caught:
        require_route(A, D, outgoing_from())

    assert "'a'" in str(caught.value)
    assert "'d'" in str(caught.value)


# -- policy validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_success_rate": 1.5}, "min_success_rate"),
        ({"min_success_rate": -0.1}, "min_success_rate"),
        ({"min_attempts_for_cap": 0}, "min_attempts_for_cap"),
        ({"unverified_ms": 0.0}, "unverified_ms"),
        ({"unverified_penalty": 0.5}, "unverified_penalty"),
    ],
)
def test_a_nonsensical_policy_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RoutingPolicy(**kwargs)


def test_the_default_policy_is_the_contract_behavior():
    assert VERIFIED_ONLY.allow_unverified is False
