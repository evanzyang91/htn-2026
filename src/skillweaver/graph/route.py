"""Shortest-path routing over the transitions of a site graph.

The point of the site graph is that a screen the agent has reached before can be
reached again without asking a model. This module is the "without asking a model"
part: given the edges, it picks the cheapest believable way from one screen to
another and hands back the concrete actions to replay.

Cost model
----------

The cost of an edge is **expected milliseconds**, not measured milliseconds::

    cost = mean_ms / success_rate

``mean_ms`` averages only SUCCESSFUL traversals (see :class:`~skillweaver.contracts.Transition`),
so dividing by the success rate is what charges an edge for its failures: an edge
that works half the time costs twice its happy-path latency, because on average it
has to be walked twice. That single formula is why a slow-but-certain path beats a
fast-and-flaky one without any extra special-casing.

Two adjustments sit on top of it, both controlled by :class:`RoutingPolicy`:

**Never-verified edges are penalized, not trusted.** An edge with no successes has
no ``mean_ms`` at all - there is no happy path to average - so there is nothing
honest to divide. It is priced at a configured prior
(:attr:`RoutingPolicy.unverified_ms`) multiplied by
:attr:`RoutingPolicy.unverified_penalty`, which keeps it usable as a last resort
while making it lose to any proven edge of the same nominal latency. By default
(:data:`VERIFIED_ONLY`) such edges are not used at all, which is what
``contracts.GraphView.route`` specifies.

**Hopeless edges are refused outright.** An edge with enough attempts on record to
be judged (:attr:`RoutingPolicy.min_attempts_for_cap`) and a success rate below
:attr:`RoutingPolicy.min_success_rate` is dropped from the search. Without the cap
a 1%-reliable edge is merely expensive, and Dijkstra will still route through it
when nothing else connects; the agent is better served by "no route" and a fresh
exploration than by replaying something that reliably does not work.

Failure behavior
----------------

:func:`find_route` returns ``None`` when no route exists, matching
``contracts.GraphView.route``. :func:`require_route` is the only function here that
raises :class:`~skillweaver.errors.RouteNotFound`, for callers that want to fail
loudly rather than branch on ``None``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from skillweaver.contracts import Fingerprint, Route, Transition
from skillweaver.errors import RouteNotFound

Outgoing = Callable[[str], Iterable[Transition]]
"""Supplies the outgoing edges of a node, keyed by ``Fingerprint.value``.

Routing never needs the node table, only this lookup, which is what keeps this
module independent of how a graph stores itself.
"""


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """How much the router is willing to gamble.

    Attributes:
        min_success_rate: An edge whose success rate is strictly below this is
            refused, provided it has at least ``min_attempts_for_cap`` attempts on
            record. ``0.0`` disables the cap.
        min_attempts_for_cap: How many attempts an edge needs before
            ``min_success_rate`` may condemn it. One unlucky first attempt is not
            evidence that an edge is hopeless, so the cap holds its fire until
            there is a sample worth believing.
        allow_unverified: Whether edges with no successes may be routed through at
            all. ``False`` reproduces ``contracts.GraphView.route`` exactly.
        unverified_ms: The assumed latency of a never-verified edge, in
            milliseconds, since it has no measured one.
        unverified_penalty: Multiplier applied to ``unverified_ms``. Must be at
            least ``1.0``; anything higher makes an unverified edge lose to a
            proven edge of equal nominal latency.

    Raises:
        ValueError: if any value is out of range.
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
"""The default policy: proven edges only, exactly as ``contracts.GraphView.route``
specifies. Under it every edge costs ``mean_ms / success_rate`` and a
:class:`~skillweaver.contracts.Route`'s ``cost`` is the contract's expected
milliseconds."""

EXPLORATORY = RoutingPolicy(allow_unverified=True)
"""Same cost model, but a never-verified edge may be used as a last resort at
``unverified_ms * unverified_penalty``. For an agent that would rather try a
remembered-but-unproven path than fall back to the model."""


def success_rate(edge: Transition) -> float:
    """``successes / attempts`` in ``0.0..1.0``; ``0.0`` for an edge never attempted."""
    return edge.successes / edge.attempts if edge.attempts else 0.0


def is_verified(edge: Transition) -> bool:
    """Whether ``edge`` has ever been seen to work.

    Equivalent to ``edge.last_verified is not None`` for any edge built through
    ``SiteGraph.observe_transition``, which stamps both together.
    """
    return edge.successes > 0


def edge_cost(edge: Transition, policy: RoutingPolicy = VERIFIED_ONLY) -> float | None:
    """The expected milliseconds of traversing ``edge``, or ``None`` to refuse it.

    ``None`` means the router must not use this edge at all: either it is hopeless
    under the policy's cap, or it is unverified and the policy does not allow that.
    Never raises.
    """
    rate = success_rate(edge)
    if edge.attempts >= policy.min_attempts_for_cap and rate < policy.min_success_rate:
        return None
    if not is_verified(edge):
        if not policy.allow_unverified:
            return None
        return policy.unverified_ms * policy.unverified_penalty
    return edge.mean_ms / rate


def find_route(
    src: Fingerprint,
    dst: Fingerprint,
    outgoing: Outgoing,
    policy: RoutingPolicy = VERIFIED_ONLY,
) -> Route | None:
    """The lowest-cost known route from ``src`` to ``dst``, or ``None``.

    Dijkstra over :func:`edge_cost`, which is non-negative, so the first time a node
    is settled it is settled with its best cost. Matching is exact on
    ``Fingerprint.value``; an unknown fingerprint simply has no outgoing edges and
    yields ``None`` rather than an error. A route from a state to itself is
    ``Route((), 0.0, ())``.

    Ties are broken by insertion order - the order ``outgoing`` yields edges - so
    the same graph always produces the same route.
    """
    if src.value == dst.value:
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
) -> Route:
    """:func:`find_route`, but insisting on an answer.

    Raises:
        RouteNotFound: when :func:`find_route` would return ``None``.
    """
    route = find_route(src, dst, outgoing, policy)
    if route is None:
        raise RouteNotFound(f"no known route from {src.value!r} to {dst.value!r}")
    return route


def _assemble(
    src: Fingerprint, dst: Fingerprint, came_from: dict[str, Transition], cost: float
) -> Route:
    """Walk the predecessor chain back from ``dst`` and flatten it into a Route."""
    edges: list[Transition] = []
    node = dst.value
    while node != src.value:
        edge = came_from[node]
        edges.append(edge)
        node = edge.src.value
    edges.reverse()
    steps = tuple(action for edge in edges for action in edge.actions)
    return Route(steps=steps, cost=cost, edges=tuple(edges))
