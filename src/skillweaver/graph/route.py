"""Shortest-path routing over the transitions of a site graph: the "without asking a
model" half of the site graph.

An edge costs EXPECTED milliseconds, ``mean_ms / success_rate``. ``mean_ms`` averages only
SUCCESSFUL traversals, so dividing by the rate is what charges an edge for its failures -
an edge that works half the time costs twice its happy path, because on average it is
walked twice. That one formula is why slow-but-certain beats fast-and-flaky with no
special-casing. :class:`RoutingPolicy` adds two adjustments: a never-verified edge has no
happy path to average and is priced at a prior times a penalty (or refused outright, which
is the :data:`VERIFIED_ONLY` default and what ``contracts.GraphView.route`` specifies), and
an edge with enough attempts to judge and a success rate under the floor is DROPPED -
otherwise Dijkstra routes through a 1%-reliable edge whenever nothing else connects, and
"no route" plus a fresh exploration serves the agent better.

"Am I already there?" is NOT an exact-match question. A route's endpoints are usually a
LIVE screen and a RECORDED one, and a live page never fingerprints identically twice, so
comparing ``value`` answers "no" for a screen the agent is standing on and reports
``no_route`` for a journey of zero steps - which is how the first skill learned on live
Wikipedia became unusable the moment it was stored. :func:`find_route` settles that case
with ``SAME_STATE_THRESHOLD``; ``same_state_threshold=1.0`` demands equality, which
``InMemorySiteGraph.route`` passes. Tolerance stops at the endpoints: intermediate hops
match exactly, because those values ARE node ids.

A stored route replays fast, free and confident whether or not its numbers mean anything -
the graph once carried 262144 successes on an edge walked a few dozen times - so
:func:`explain_edge` and :func:`explain_route` give the pricing with its counts, through
the same :func:`_price` the router uses.

:func:`find_route` returns ``None`` when no route exists; :func:`require_route` raises.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from skillweaver.contracts import Fingerprint, Route, Transition
from skillweaver.errors import RouteNotFound
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD

Outgoing = Callable[[str], Iterable[Transition]]
"""Outgoing edges of a node, keyed by ``Fingerprint.value``. Routing needs no node table,
which is what keeps this module independent of how a graph stores itself."""


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """How much the router is willing to gamble.

    Attributes:
        min_success_rate: An edge below this is refused, provided it has at least
            ``min_attempts_for_cap`` attempts. ``0.0`` disables the cap.
        min_attempts_for_cap: Attempts needed before ``min_success_rate`` may condemn an
            edge - one unlucky first attempt is not evidence.
        allow_unverified: Whether edges with no successes may be routed through.
            ``False`` reproduces ``contracts.GraphView.route`` exactly.
        unverified_ms: Assumed latency of a never-verified edge, having no measured one.
        unverified_penalty: Multiplier on ``unverified_ms``; at least ``1.0``, and
            anything higher makes an unverified edge lose to a proven edge of equal
            nominal latency.

    Raises:
        ValueError: any value is out of range.
    """

    min_success_rate: float = 0.2
    min_attempts_for_cap: int = 3
    allow_unverified: bool = False
    unverified_ms: float = 1000.0
    unverified_penalty: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError(f"min_success_rate must be in 0.0..1.0, got {self.min_success_rate}")
        if self.min_attempts_for_cap < 1:
            raise ValueError(
                f"min_attempts_for_cap must be at least 1, got {self.min_attempts_for_cap}"
            )
        if self.unverified_ms <= 0.0:
            raise ValueError(f"unverified_ms must be positive, got {self.unverified_ms}")
        if self.unverified_penalty < 1.0:
            raise ValueError(
                f"unverified_penalty must be at least 1.0, got {self.unverified_penalty}"
            )


VERIFIED_ONLY = RoutingPolicy()
"""Proven edges only, exactly as ``contracts.GraphView.route`` specifies."""

EXPLORATORY = RoutingPolicy(allow_unverified=True)
"""Same cost model, but a never-verified edge is usable as a last resort, for an agent
that would rather try a remembered-but-unproven path than fall back to the model."""


def success_rate(edge: Transition) -> float:
    """``successes / attempts`` in ``0.0..1.0``; ``0.0`` for an edge never attempted."""
    return edge.successes / edge.attempts if edge.attempts else 0.0


def is_verified(edge: Transition) -> bool:
    """Whether ``edge`` has ever been seen to work."""
    return edge.successes > 0


def edge_cost(edge: Transition, policy: RoutingPolicy = VERIFIED_ONLY) -> float | None:
    """The expected milliseconds of traversing ``edge``, or ``None`` to refuse it.

    ``None`` means hopeless under the cap, or unverified under a policy that forbids it.
    :func:`explain_edge` is the same decision with its reasons kept.
    """
    return _price(edge, policy)[0]


def _price(edge: Transition, policy: RoutingPolicy) -> tuple[float | None, str]:
    """The cost decision and the sentence justifying it, from ONE place, so the number
    the router uses and the reason a person is shown cannot drift apart."""
    rate = success_rate(edge)
    if edge.attempts >= policy.min_attempts_for_cap and rate < policy.min_success_rate:
        return None, (
            f"refused: {edge.successes}/{edge.attempts} is a {rate:.0%} success rate, "
            f"under the {policy.min_success_rate:.0%} floor, over at least "
            f"{policy.min_attempts_for_cap} attempts"
        )
    if not is_verified(edge):
        if not policy.allow_unverified:
            return None, f"refused: never verified in {edge.attempts} attempts"
        cost = policy.unverified_ms * policy.unverified_penalty
        return cost, (
            f"{cost:.0f}ms assumed: never verified in {edge.attempts} attempts, priced at "
            f"{policy.unverified_ms:.0f}ms x{policy.unverified_penalty:g}"
        )
    return edge.mean_ms / rate, (
        f"{edge.mean_ms / rate:.0f}ms expected: {edge.mean_ms:.0f}ms over "
        f"{edge.successes} successful of {edge.attempts} attempts ({rate:.0%})"
    )


@dataclass(frozen=True, slots=True)
class EdgeVerdict:
    """Why the router priced one edge the way it did: its cost, or ``None`` when refused,
    and one line naming the statistics that decided it."""

    edge: Transition
    cost: float | None
    reason: str

    def __str__(self) -> str:
        return f"{self.edge.src.value[:12]} -> {self.edge.dst.value[:12]}: {self.reason}"


def explain_edge(edge: Transition, policy: RoutingPolicy = VERIFIED_ONLY) -> EdgeVerdict:
    """:func:`edge_cost` with the counts it rests on, so a route that looks wrong is
    answerable with the numbers that chose it."""
    cost, reason = _price(edge, policy)
    return EdgeVerdict(edge=edge, cost=cost, reason=reason)


def explain_route(route: Route, policy: RoutingPolicy = VERIFIED_ONLY) -> tuple[EdgeVerdict, ...]:
    """One :class:`EdgeVerdict` per hop, in walk order; empty for a zero-hop route."""
    return tuple(explain_edge(edge, policy) for edge in route.edges)


def same_state(
    left: Fingerprint, right: Fingerprint, threshold: float = SAME_STATE_THRESHOLD
) -> bool:
    """Whether these two fingerprints name the same screen.

    Equal ``value`` always says yes; otherwise ``similarity`` against ``threshold``, which
    is how a live screen is recognised although it never reproduces its id.
    ``threshold >= 1.0`` demands exact equality.
    """
    if left.value == right.value:
        return True
    return threshold < 1.0 and left.similarity(right) >= threshold


def find_route(
    src: Fingerprint,
    dst: Fingerprint,
    outgoing: Outgoing,
    policy: RoutingPolicy = VERIFIED_ONLY,
    *,
    same_state_threshold: float = SAME_STATE_THRESHOLD,
    resolve: Callable[[Fingerprint], Fingerprint] | None = None,
) -> Route | None:
    """The lowest-cost known route from ``src`` to ``dst``, or ``None``.

    Dijkstra over :func:`edge_cost`, which is non-negative. An unknown fingerprint simply
    has no outgoing edges. A route to the SAME state is ``Route((), 0.0, ())``, and
    sameness is :func:`same_state`, not equality; intermediate hops match exactly. Ties
    break by the order ``outgoing`` yields edges, so one graph always gives one route.

    Args:
        same_state_threshold: How alike the endpoints must be to count as one screen.
            ``1.0`` restores exact matching.
        resolve: Maps each endpoint onto the node it belongs to first, for fingerprints
            taken from live observations.
    """
    if resolve is not None:
        src, dst = resolve(src), resolve(dst)
    if same_state(src, dst, same_state_threshold):
        return Route((), 0.0, ())

    order = itertools.count()
    best: dict[str, float] = {src.value: 0.0}
    came_from: dict[str, Transition] = {}
    settled: set[str] = set()
    queue: list[tuple[float, int, str]] = [(0.0, next(order), src.value)]

    while queue:
        cost, _, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == dst.value:
            return _assemble(src, dst, came_from, cost)
        for edge in outgoing(node):
            step = edge_cost(edge, policy)
            if step is None:
                continue
            successor = edge.dst.value
            if successor in settled:
                continue
            through = cost + step
            if through < best.get(successor, math.inf):
                best[successor] = through
                came_from[successor] = edge
                heapq.heappush(queue, (through, next(order), successor))
    return None


def require_route(
    src: Fingerprint,
    dst: Fingerprint,
    outgoing: Outgoing,
    policy: RoutingPolicy = VERIFIED_ONLY,
    **kwargs: object,
) -> Route:
    """:func:`find_route`, insisting on an answer.

    Raises:
        RouteNotFound: when :func:`find_route` would return ``None``.
    """
    route = find_route(src, dst, outgoing, policy, **kwargs)  # type: ignore[arg-type]
    if route is None:
        raise RouteNotFound(f"no known route from {src.value!r} to {dst.value!r}")
    return route


def _assemble(
    src: Fingerprint, dst: Fingerprint, came_from: dict[str, Transition], cost: float
) -> Route:
    edges: list[Transition] = []
    node = dst.value
    while node != src.value:
        edge = came_from[node]
        edges.append(edge)
        node = edge.src.value
    edges.reverse()
    steps = tuple(action for edge in edges for action in edge.actions)
    return Route(steps=steps, cost=cost, edges=tuple(edges))
