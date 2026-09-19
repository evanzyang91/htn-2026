"""The in-memory site graph: screens as nodes, replayable action sequences as edges.

Implements ``contracts.SiteGraph``. States are keyed by ``Fingerprint.value`` and
edges by ``(src, dst, actions)``, matching the identity rules in ``contracts``.

Why node lookup is fuzzy
------------------------

Two visits to the same screen never fingerprint identically. A cart badge ticks
from 2 to 3, an ad rotates, a timestamp advances - the ``value`` hash changes and
an exact-match graph grows a brand new node for a screen it already knows. Do that
a few dozen times and the graph is a pile of singletons with no edges between them,
which is worth nothing.

So node identity here is *nearest match above a threshold*, using
``Fingerprint.similarity`` (the fraction of agreeing ``parts`` sub-hashes). The
first fingerprint seen for a screen becomes that node's canonical id and later
near-matches fold into it. :attr:`InMemorySiteGraph.match_threshold` tunes how
forgiving that is: too low and two genuinely different screens collapse into one
node whose edges lead somewhere unpredictable, too high and the graph fragments
again. It is a constructor argument because the right value depends on what the
fingerprinter puts in ``parts``.

Fuzziness stops at the node table. ``route`` and ``neighbors`` match exactly on
``Fingerprint.value``, as ``contracts.GraphView`` specifies; pass
``approximate=True`` to opt into the same near-match lookup, which is usually what
you want when the fingerprint came straight off a fresh observation. The explicit
way to do it is :meth:`InMemorySiteGraph.resolve`.
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

DEFAULT_MATCH_THRESHOLD = 0.6
"""Similarity at or above which two fingerprints are taken to be the same screen.

Pitched against the shape a fingerprinter in this project actually produces: three
parts - ``url``, ``layout`` and ``text``. With three parts the only similarities
reachable are ``1.0``, ``0.667``, ``0.333`` and ``0.0``, so this default draws the
line in the one gap that matters. One part drifting (``0.667``) still resolves,
which is the badge-ticked, clock-advanced, ad-rotated case. Two parts drifting
(``0.333``) does not, because a screen whose layout AND text both changed is a
different screen.

A threshold has to be read against the part count, not in the abstract: ``0.75``
sounds stricter by a hair but silently rejects every single-part drift a
three-part fingerprint can express, and the graph fragments into singletons.
Re-check this value if a fingerprinter changes how many parts it emits.
"""

_EdgeKey = tuple[str, str, tuple[Action, ...]]


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """One domain's states and transitions, detached from any graph.

    The unit of persistence and of merging: :mod:`skillweaver.graph.store` writes
    and reads exactly this. Detached means nothing here aliases a live graph, so a
    snapshot is safe to hold, compare or merge while the graph keeps changing.
    """

    domain: str
    states: tuple[UIState, ...] = ()
    transitions: tuple[Transition, ...] = ()


@runtime_checkable
class GraphPersistence(Protocol):
    """Where a graph puts its domains. ``JSONGraphStore`` in
    :mod:`skillweaver.graph.store` is the implementation; this Protocol exists so
    the model does not import the store and the two stay independently testable."""

    def load(self, domain: str) -> GraphSnapshot:
        """One domain as stored; an empty snapshot when nothing is stored for it."""
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

    # -- node identity -----------------------------------------------------------

    def resolve(self, fp: Fingerprint) -> UIState | None:
        """The known state ``fp`` refers to, or ``None`` when this screen is new.

        An exact ``value`` match wins immediately. Otherwise the most similar known
        state at or above :attr:`match_threshold` wins; ties go to the state seen
        first, so the answer does not depend on dict ordering. Never raises.
        """
        exact = self._states.get(fp.value)
        if exact is not None:
            return exact
        if self.match_threshold >= 1.0:
            return None
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
        """``fp`` mapped onto the node it belongs to, or ``fp`` itself when new.

        The id every edge and lookup should be keyed by. Idempotent.
        """
        known = self.resolve(fp)
        return known.fingerprint if known is not None else fp

    # -- SiteGraph: writes -------------------------------------------------------

    def upsert_state(self, state: UIState) -> UIState:
        """Add ``state``, or refresh the label, URL pattern and thumbnail of the node
        it resolves to. ``first_seen`` and the canonical fingerprint of an existing
        node are kept. Returns the stored state.

        A near match updates the node it matched, so re-observing a screen whose
        fingerprint drifted enriches the existing node instead of forking a new one.
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

        Both fingerprints are resolved to existing nodes first, so a drifted
        fingerprint reinforces the edge it belongs to rather than creating a
        parallel one. Unknown states are added under ``src``'s domain when that is
        known, else ``""``.

        Statistics fold in place, without keeping any history: ``attempts`` always
        rises, ``successes`` rises only when ``ok``, and ``mean_ms`` is the running
        mean over SUCCESSFUL traversals only - a failure's duration says nothing
        about how long the edge takes when it works. ``last_verified`` stamps the
        latest success.
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

        Matching is exact on ``Fingerprint.value`` and only proven edges are used,
        per ``contracts.GraphView.route``; both are consequences of the defaults,
        so a graph built with a different ``policy`` routes by that policy instead.
        Never raises for unknown fingerprints.

        Args:
            approximate: Resolve both endpoints through :meth:`resolve` first. Use
                it when the fingerprints came from live observations rather than
                from the graph.
        """
        if approximate:
            src_fp, dst_fp = self.canonical(src_fp), self.canonical(dst_fp)
        return find_route(src_fp, dst_fp, self._outgoing, self.policy)

    def neighbors(self, fp: Fingerprint, *, approximate: bool = False) -> list[Transition]:
        """Outgoing edges of ``fp``, most reliable first, then fastest.

        Empty list when the state is unknown or has no edges. ``approximate``
        resolves ``fp`` through :meth:`resolve` first.
        """
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
        """Every edge, or only those whose source state belongs to ``domain``.

        In insertion order, which is the order the edges were first observed.
        """
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

        Verbatim by default: every state keeps its own fingerprint, so a snapshot
        that legitimately holds two similar-but-distinct screens is not collapsed.
        That is what makes a save/load round-trip exact.

        Args:
            resolve: Route each state and edge endpoint through :meth:`resolve`
                first, consolidating near matches against what is already here.
                Use it when folding in a graph recorded by another run, whose
                fingerprints for the same screens will have drifted.
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

    def save(self) -> None:
        """Persist every loaded domain, merging with what is already stored.

        A no-op when this graph has no store. Merging rather than overwriting is
        the point: several runs write the same domain concurrently, and
        last-write-wins would silently drop the other run's observations.

        Raises:
            SkillWeaverError: if the data directory cannot be written.
        """
        if self.store is None:
            return
        for domain in self.domains():
            self.store.save_merged(self.snapshot(domain))

    def load(self, domain: str) -> None:
        """Load ``domain`` from storage, replacing what is in memory for it.

        A domain with nothing stored loads as empty; never raises for "missing".
        A graph with no store simply drops the domain.
        """
        self.forget(domain)
        if self.store is None:
            return
        self.absorb(self.store.load(domain))

    def forget(self, domain: str) -> None:
        """Drop every state of ``domain`` and every edge touching one. In-memory only."""
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

    def __repr__(self) -> str:
        return (
            f"InMemorySiteGraph(states={len(self._states)}, edges={len(self._edges)}, "
            f"match_threshold={self.match_threshold})"
        )


def merge_transitions(left: Transition, right: Transition) -> Transition:
    """Combine two observations of the SAME edge by summing their statistics.

    Attempts and successes add. ``mean_ms`` becomes the success-weighted mean of
    the two means, which is exactly the mean of all successful traversals on both
    sides - a plain average would let one side's single lucky run outvote the
    other's hundred. ``last_verified`` is the later of the two.

    Raises:
        ValueError: if the two are not the same edge.
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
    """Combine two records of the same node, keeping the earliest sighting and
    preferring ``left``'s non-empty descriptive fields."""
    return replace(
        left,
        domain=left.domain or right.domain,
        label=left.label or right.label,
        url_pattern=left.url_pattern or right.url_pattern,
        thumbnail=left.thumbnail if left.thumbnail is not None else right.thumbnail,
        first_seen=min(left.first_seen, right.first_seen),
    )


def _fold_mean(mean: float, count_before: int, value: float) -> float:
    """The running mean after adding ``value`` to ``count_before`` samples."""
    return (mean * count_before + value) / (count_before + 1)
