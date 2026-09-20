"""The in-memory site graph: screens as nodes, replayable action sequences as edges.

Implements ``contracts.SiteGraph``. States are keyed by ``Fingerprint.value``, edges by
``(src, dst, actions)``.

Node lookup is FUZZY - nearest match above :attr:`InMemorySiteGraph.match_threshold` by
``Fingerprint.similarity`` - because two visits to one screen never fingerprint
identically, and an exact-match graph grows a pile of singletons with no edges between
them. The first fingerprint seen becomes the node's canonical id; later near-matches fold
into it.

Fuzziness stops at the node table: ``route`` and ``neighbors`` match exactly on
``value``, per ``contracts.GraphView``, unless ``approximate=True`` (which is usually
what a fingerprint straight off a live observation wants).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from skillweaver.contracts import (
    Action,
    Fingerprint,
    Route,
    Transition,
    UIState,
    utcnow,
)
from skillweaver.graph.route import (
    VERIFIED_ONLY,
    RoutingPolicy,
    find_route,
    success_rate,
)
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD

DEFAULT_MATCH_THRESHOLD = SAME_STATE_THRESHOLD
"""Similarity at or above which two fingerprints are the same screen.

The project holds ONE number for "am I looking at the screen I recorded", and a threshold
is only meaningful against the SHAPE of the signal it judges, so a fingerprinter and the
graph keying on it cannot pick cuts independently. This was an independent ``0.6``,
reasoned about a fingerprint of three parts; the shipped one emits ~175, whose same-state
floor is 0.305, and 0.6 rejects every screen that moved.

``match_threshold=1.0`` demands exact equality and disables near matching."""

_EdgeKey = tuple[str, str, tuple[Action, ...]]


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """One domain's states and transitions, detached from any graph.

    The unit of persistence and of merging; :mod:`skillweaver.graph.store` reads and
    writes exactly this. Detached, so it is safe to hold while the graph changes.
    """

    domain: str
    states: tuple[UIState, ...] = ()
    transitions: tuple[Transition, ...] = ()


@runtime_checkable
class GraphPersistence(Protocol):
    """Where a graph puts its domains. A Protocol so the model does not import the store."""

    def load(self, domain: str) -> GraphSnapshot:
        """One domain as stored; an empty snapshot when nothing is."""
        ...

    def save_merged(self, snapshot: GraphSnapshot) -> GraphSnapshot:
        """Fold ``snapshot`` into what is stored and return the combined result."""
        ...


class InMemorySiteGraph:
    """A ``contracts.SiteGraph`` held in dicts, optionally backed by a store.

    Args:
        match_threshold: Similarity at or above which a fingerprint resolves to an
            existing node. ``1.0`` demands exact equality and disables near
            matching entirely.
        policy: How :meth:`route` prices edges. The default is the contract's
            verified-edges-only cost model.
        store: Where :meth:`save` and :meth:`load` go. ``None`` makes this a purely
            in-memory graph whose ``save`` is a no-op, as ``contracts`` allows.

    Raises:
        ValueError: if ``match_threshold`` is outside ``0.0..1.0``.

    Not thread-safe. Concurrency between processes is handled at the store layer,
    which merges rather than overwrites.
    """

    def __init__(
        self,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        policy: RoutingPolicy = VERIFIED_ONLY,
        store: GraphPersistence | None = None,
    ) -> None:
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(f"match_threshold must be in 0.0..1.0, got {match_threshold}")
        self.match_threshold = match_threshold
        self.policy = policy
        self.store = store
        self._states: dict[str, UIState] = {}
        self._edges: dict[_EdgeKey, Transition] = {}
        self._exchanged: dict[str, dict[_EdgeKey, Transition]] = {}

    # -- node identity -----------------------------------------------------------

    def resolve(self, fp: Fingerprint) -> UIState | None:
        """The known state ``fp`` refers to, or ``None`` when this screen is new.

        Exact ``value`` wins; otherwise the most similar at or above
        :attr:`match_threshold`, ties to the state seen first so dict order cannot decide.
        """
        exact = self._states.get(fp.value)
        if exact is not None:
            return exact
        if self.match_threshold >= 1.0:
            return None
        # Same rule as ``route.same_state``, but this needs the SCORE to pick the best of
        # several candidates, not a yes or no.
        best: UIState | None = None
        best_score = -1.0
        for state in self._states.values():
            score = fp.similarity(state.fingerprint)
            if score < self.match_threshold:
                continue
            if (
                best is None
                or score > best_score
                or (score == best_score and state.first_seen < best.first_seen)
            ):
                best, best_score = state, score
        return best

    def canonical(self, fp: Fingerprint) -> Fingerprint:
        """``fp`` mapped onto the node it belongs to, or ``fp`` itself when new. Idempotent."""
        known = self.resolve(fp)
        return known.fingerprint if known is not None else fp

    # -- SiteGraph: writes -------------------------------------------------------

    def upsert_state(self, state: UIState) -> UIState:
        """Add ``state``, or refresh the label, URL pattern and thumbnail of the node it
        resolves to, keeping ``first_seen`` and the canonical fingerprint.

        A near match updates the node it matched, so a drifted fingerprint enriches the
        existing node instead of forking one.
        """
        known = self.resolve(state.fingerprint)
        if known is None:
            self._states[state.fingerprint.value] = state
            return state
        merged = replace(
            known,
            domain=state.domain or known.domain,
            label=state.label or known.label,
            url_pattern=state.url_pattern or known.url_pattern,
            thumbnail=state.thumbnail if state.thumbnail is not None else known.thumbnail,
        )
        self._states[known.fingerprint.value] = merged
        return merged

    def observe_transition(
        self,
        src: Fingerprint,
        actions: Sequence[Action],
        dst: Fingerprint,
        ok: bool,
        ms: float,
    ) -> Transition:
        """Record one attempt at the edge ``(src, dst, actions)`` and return it updated.

        Both fingerprints are resolved first, so a drifted one reinforces the edge it
        belongs to. Statistics fold in place with no history: ``mean_ms`` is the running
        mean over SUCCESSFUL traversals only, a failure's duration saying nothing about
        how long the edge takes when it works.
        """
        src_fp = self.canonical(src)
        dst_fp = self.canonical(dst)
        domain = self._states[src_fp.value].domain if src_fp.value in self._states else ""
        for fp in (src_fp, dst_fp):
            if fp.value not in self._states:
                self._states[fp.value] = UIState(fingerprint=fp, domain=domain)

        key: _EdgeKey = (src_fp.value, dst_fp.value, tuple(actions))
        old = self._edges.get(key) or Transition(src_fp, dst_fp, tuple(actions))
        edge = replace(
            old,
            attempts=old.attempts + 1,
            successes=old.successes + (1 if ok else 0),
            mean_ms=_fold_mean(old.mean_ms, old.successes, ms) if ok else old.mean_ms,
            last_verified=utcnow() if ok else old.last_verified,
        )
        self._edges[key] = edge
        return edge

    # -- SiteGraph: reads --------------------------------------------------------

    def route(
        self, src_fp: Fingerprint, dst_fp: Fingerprint, *, approximate: bool = False
    ) -> Route | None:
        """The lowest-cost known route, or ``None`` when no path is known.

        Exact on ``Fingerprint.value``, proven edges only, per ``contracts.GraphView`` -
        both consequences of the defaults, so another ``policy`` routes by that policy.

        Args:
            approximate: Resolve both endpoints through :meth:`resolve` first, for
                fingerprints that came from live observations.
        """
        # ``find_route`` treats endpoints within SAME_STATE_THRESHOLD as one screen,
        # which is right for a live fingerprint and wrong here: the contract says exact.
        return find_route(
            src_fp,
            dst_fp,
            self._outgoing,
            self.policy,
            same_state_threshold=self.match_threshold if approximate else 1.0,
            resolve=self.canonical if approximate else None,
        )

    def neighbors(self, fp: Fingerprint, *, approximate: bool = False) -> list[Transition]:
        """Outgoing edges of ``fp``, most reliable first then fastest; empty when unknown."""
        if approximate:
            fp = self.canonical(fp)
        return sorted(self._outgoing(fp.value), key=lambda e: (-success_rate(e), e.mean_ms))

    def states(self, domain: str) -> list[UIState]:
        """Every known state of ``domain``, oldest first. Empty list when none."""
        return sorted(
            (s for s in self._states.values() if s.domain == domain), key=lambda s: s.first_seen
        )

    def domains(self) -> list[str]:
        """Every domain with at least one state, alphabetically."""
        return sorted({s.domain for s in self._states.values()})

    def transitions(self, domain: str | None = None) -> list[Transition]:
        """Every edge, or only those sourced in ``domain``, in first-observed order."""
        if domain is None:
            return list(self._edges.values())
        nodes = {s.fingerprint.value for s in self._states.values() if s.domain == domain}
        return [e for e in self._edges.values() if e.src.value in nodes]

    # -- snapshots and persistence -----------------------------------------------

    def snapshot(self, domain: str) -> GraphSnapshot:
        """Everything known about ``domain``, detached from this graph."""
        return GraphSnapshot(
            domain=domain,
            states=tuple(self.states(domain)),
            transitions=tuple(self.transitions(domain)),
        )

    def absorb(self, snapshot: GraphSnapshot, *, resolve: bool = False) -> None:
        """Fold a snapshot's states and edges into this graph.

        Verbatim by default, so a snapshot legitimately holding two similar-but-distinct
        screens is not collapsed and a save/load round-trip is exact.

        Args:
            resolve: Consolidate near matches against what is already here, for a graph
                recorded by another run whose fingerprints will have drifted.
        """
        for state in snapshot.states:
            if resolve:
                self.upsert_state(state)
            elif state.fingerprint.value not in self._states:
                self._states[state.fingerprint.value] = state
            else:
                self._states[state.fingerprint.value] = merge_states(
                    self._states[state.fingerprint.value], state
                )
        for edge in snapshot.transitions:
            src = self.canonical(edge.src) if resolve else edge.src
            dst = self.canonical(edge.dst) if resolve else edge.dst
            for fp in (src, dst):
                if fp.value not in self._states:
                    self._states[fp.value] = UIState(fingerprint=fp, domain=snapshot.domain)
            key: _EdgeKey = (src.value, dst.value, edge.actions)
            existing = self._edges.get(key)
            incoming = replace(edge, src=src, dst=dst)
            self._edges[key] = (
                incoming if existing is None else merge_transitions(existing, incoming)
            )

    def unsaved(self, domain: str) -> GraphSnapshot:
        """What this graph has OBSERVED of ``domain`` since it last met the store.

        The store SUMS statistics, so handing it the whole in-memory graph hands back the
        counts :meth:`load` just supplied and every save DOUBLES them - a graph loaded and
        saved ten times reported 1023 attempts on an edge walked ten, and routing then
        preferred whichever edges had been persisted most often. So edges here carry
        DIFFERENCES, exactly the inverse of :func:`merge_transitions`'s weighting; an edge
        with nothing new is left out, an edge the store never saw is passed whole. States
        go in full: they hold no summed statistics.

        The baseline is per domain, set by :meth:`load` and :meth:`save`, so a domain this
        graph never loaded persists everything on its first save.
        """
        already = self._exchanged.get(domain, {})
        states = self.states(domain)
        nodes = {s.fingerprint.value for s in states}
        edges = [
            delta
            for key, edge in self._edges.items()
            if key[0] in nodes
            and (delta := subtract_transitions(edge, already.get(key))) is not None
        ]
        return GraphSnapshot(domain=domain, states=tuple(states), transitions=tuple(edges))

    def save(self) -> None:
        """Persist every loaded domain, merging with what is already stored.

        Merging rather than overwriting because several runs write one domain
        concurrently. What is handed over is :meth:`unsaved`, not the whole graph, so
        saving twice with nothing observed in between changes no number.

        Raises:
            SkillWeaverError: the data directory cannot be written.
        """
        if self.store is None:
            return
        for domain in self.domains():
            self.store.save_merged(self.unsaved(domain))
            self._mark_exchanged(domain)

    def load(self, domain: str) -> None:
        """Load ``domain`` from storage, replacing what is in memory for it.

        Nothing stored loads as empty; a graph with no store simply drops the domain.
        """
        self.forget(domain)
        if self.store is None:
            return
        self.absorb(self.store.load(domain))
        self._mark_exchanged(domain)

    def forget(self, domain: str) -> None:
        """Drop every state of ``domain`` and every edge touching one. In-memory only."""
        self._exchanged.pop(domain, None)
        doomed = {s.fingerprint.value for s in self._states.values() if s.domain == domain}
        if not doomed:
            return
        for value in doomed:
            del self._states[value]
        for key in [k for k in self._edges if k[0] in doomed or k[1] in doomed]:
            del self._edges[key]

    # -- internals ---------------------------------------------------------------

    def _outgoing(self, value: str) -> Iterable[Transition]:
        return [e for e in self._edges.values() if e.src.value == value]

    def _mark_exchanged(self, domain: str) -> None:
        """Record this domain's edges as the baseline :meth:`unsaved` measures against.

        Called after a load and a save - the two moments memory and disk are known to
        agree about what THIS graph contributed. A later writer's additions arrive at the
        next :meth:`load`.
        """
        nodes = {s.fingerprint.value for s in self.states(domain)}
        self._exchanged[domain] = {
            key: edge for key, edge in self._edges.items() if key[0] in nodes
        }

    def __repr__(self) -> str:
        return (
            f"InMemorySiteGraph(states={len(self._states)}, edges={len(self._edges)}, "
            f"match_threshold={self.match_threshold})"
        )


def merge_transitions(left: Transition, right: Transition) -> Transition:
    """Combine two observations of the SAME edge by summing their statistics.

    ``mean_ms`` is the success-WEIGHTED mean of the two, which is the mean over all
    successful traversals on both sides; a plain average would let one side's single
    lucky run outvote the other's hundred.

    Raises:
        ValueError: the two are not the same edge.
    """
    if (left.src.value, left.dst.value, left.actions) != (
        right.src.value,
        right.dst.value,
        right.actions,
    ):
        raise ValueError("merge_transitions needs two observations of the same edge")
    successes = left.successes + right.successes
    if successes:
        mean_ms = (left.mean_ms * left.successes + right.mean_ms * right.successes) / successes
    else:
        mean_ms = 0.0
    stamps = [t for t in (left.last_verified, right.last_verified) if t is not None]
    return replace(
        left,
        attempts=left.attempts + right.attempts,
        successes=successes,
        mean_ms=mean_ms,
        last_verified=max(stamps) if stamps else None,
    )


def merge_states(left: UIState, right: UIState) -> UIState:
    """Combine two records of one node: earliest sighting, ``left``'s non-empty fields."""
    return replace(
        left,
        domain=left.domain or right.domain,
        label=left.label or right.label,
        url_pattern=left.url_pattern or right.url_pattern,
        thumbnail=left.thumbnail if left.thumbnail is not None else right.thumbnail,
        first_seen=min(left.first_seen, right.first_seen),
    )


def subtract_transitions(edge: Transition, already: Transition | None) -> Transition | None:
    """What ``edge`` has to say beyond ``already``, or ``None`` when it says nothing.

    The inverse of :func:`merge_transitions`, so merging the result back onto ``already``
    reproduces ``edge`` exactly. ``last_verified`` is carried only when there IS a new
    success - a delta of pure failures has verified nothing. ``already`` of ``None``
    returns the whole edge.

    Negative differences cannot arise from :meth:`InMemorySiteGraph.save`, whose baseline
    is a past state of the same graph, and are clamped rather than trusted: a statistic
    that can go backwards is worse than one that stalls.
    """
    if already is None:
        return edge
    attempts = max(edge.attempts - already.attempts, 0)
    successes = max(edge.successes - already.successes, 0)
    if not attempts and not successes:
        return None
    if successes:
        mean_ms = max(
            (edge.mean_ms * edge.successes - already.mean_ms * already.successes) / successes, 0.0
        )
    else:
        mean_ms = 0.0
    return replace(
        edge,
        attempts=attempts,
        successes=successes,
        mean_ms=mean_ms,
        last_verified=edge.last_verified if successes else None,
    )


def _fold_mean(mean: float, count_before: int, value: float) -> float:
    """The running mean after adding ``value`` to ``count_before`` samples."""
    return (mean * count_before + value) / (count_before + 1)
