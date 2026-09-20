"""Per-domain JSON persistence for site graphs.

One file per domain under a root directory, named after a SANITIZED domain so
``app.example.com`` and ``../etc/passwd`` both land somewhere harmless. Every file carries
:data:`SCHEMA_VERSION`; one written by a newer build is refused rather than half-read into
a graph that then routes an agent somewhere wrong.

:meth:`JSONGraphStore.save_merged` MERGES rather than overwrites, because several runs
write one domain at once and last-write-wins would silently drop the other run's
observations. :func:`atomic_write` makes every writer's file appear whole, so a reader
never sees a torn one; a writer that loses the read-modify-write race loses only the
observations it merged in that instant, which the next ``save`` merges back in. That is
the right trade for a cache of learned shortcuts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillweaver.contracts import (
    Fingerprint,
    Transition,
    UIState,
    action_from_dict,
    action_to_dict,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.graph.model import GraphSnapshot, merge_states, merge_transitions

SCHEMA_VERSION = 1
"""The on-disk format this build writes and is willing to read."""

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_RUNS = re.compile(r"_{2,}")


def domain_filename(domain: str) -> str:
    """The file name for ``domain``; an ordinary hostname keeps its own name.

    Anything else is sanitized - unsafe characters and ``..`` become ``_`` - so a hostile
    or empty domain cannot escape the root, and a hash of the original is appended.
    Without that hash ``a/b`` and ``a_b`` would share one file and silently merge two
    sites' graphs, a corruption nothing downstream could detect.
    """
    safe = _UNSAFE.sub("_", domain).replace("..", "_")
    safe = _RUNS.sub("_", safe).strip("._-")
    if safe and safe == domain:
        return f"{safe}.json"
    digest = hashlib.sha256(domain.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'domain'}-{digest}.json"


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers see either the old file or the new one.

    Temp file in the SAME directory, fsync, then ``os.replace``, which is atomic within a
    filesystem - which sharing the directory guarantees.

    Raises:
        SkillWeaverError: the directory or file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise SkillWeaverError(f"cannot write site graph to {path}: {exc}") from exc


class JSONGraphStore:
    """A ``GraphPersistence`` over a directory of JSON files, one per domain.

    Args:
        root: Created on first write; ``Settings.graphs_dir`` is the usual location.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, domain: str) -> Path:
        """Where ``domain`` is stored. The file need not exist."""
        return self.root / domain_filename(domain)

    def load(self, domain: str) -> GraphSnapshot:
        """Read ``domain``, or an empty snapshot when nothing is stored.

        Raises:
            SkillWeaverError: the file exists but is unreadable, is not valid JSON, or
                carries an unsupported schema version.
        """
        path = self.path_for(domain)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return GraphSnapshot(domain=domain)
        except OSError as exc:
            raise SkillWeaverError(f"cannot read site graph from {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SkillWeaverError(f"site graph at {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise SkillWeaverError(f"site graph at {path} is not a JSON object")
        return snapshot_from_dict(data, where=str(path))

    def save(self, snapshot: GraphSnapshot) -> Path:
        """Write ``snapshot`` over whatever is stored for its domain.

        OVERWRITES; prefer :meth:`save_merged` unless you mean to discard what is stored.
        """
        path = self.path_for(snapshot.domain)
        atomic_write(path, json.dumps(snapshot_to_dict(snapshot), indent=2, sort_keys=False))
        return path

    def save_merged(self, snapshot: GraphSnapshot) -> GraphSnapshot:
        """Fold ``snapshot`` into what is stored, write the result, and return it.

        The safe way to persist: another run's observations survive. See the module
        docstring for the concurrency this does and does not protect against.
        """
        combined = merge(self.load(snapshot.domain), snapshot)
        self.save(combined)
        return combined

    def domains(self) -> list[str]:
        """Every domain with a stored file, alphabetically.

        Reads each file's recorded ``domain`` rather than trusting the sanitized name.
        """
        if not self.root.is_dir():
            return []
        found: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, Mapping) and isinstance(data.get("domain"), str):
                found.append(data["domain"])
        return sorted(found)


def merge(left: GraphSnapshot, right: GraphSnapshot) -> GraphSnapshot:
    """Combine two snapshots of the same domain, summing every statistic.

    Anything in only one side is carried over untouched. Ordering is deterministic:
    ``left``'s items keep their positions and ``right``'s new ones follow. Matching is
    EXACT on ``Fingerprint.value`` - consolidating near-matching screens is node-identity
    work and belongs to ``InMemorySiteGraph.absorb(..., resolve=True)``, which can see the
    whole graph.

    Raises:
        ValueError: the two snapshots are for different domains.
    """
    if left.domain != right.domain:
        raise ValueError(
            f"cannot merge site graphs of different domains: {left.domain!r} and {right.domain!r}"
        )
    states: dict[str, UIState] = {s.fingerprint.value: s for s in left.states}
    for state in right.states:
        known = states.get(state.fingerprint.value)
        states[state.fingerprint.value] = state if known is None else merge_states(known, state)

    edges: dict[tuple[str, str, tuple[Any, ...]], Transition] = {
        (e.src.value, e.dst.value, e.actions): e for e in left.transitions
    }
    for edge in right.transitions:
        key = (edge.src.value, edge.dst.value, edge.actions)
        known = edges.get(key)
        edges[key] = edge if known is None else merge_transitions(known, edge)

    return GraphSnapshot(
        domain=left.domain, states=tuple(states.values()), transitions=tuple(edges.values())
    )


def snapshot_to_dict(snapshot: GraphSnapshot) -> dict[str, Any]:
    """A snapshot as the JSON-safe object stored on disk; :func:`snapshot_from_dict`
    inverts it, round-tripping statistics, ``parts`` sub-hashes and thumbnails."""
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": snapshot.domain,
        "states": [_state_to_dict(s) for s in snapshot.states],
        "transitions": [_transition_to_dict(t) for t in snapshot.transitions],
    }


def snapshot_from_dict(data: Mapping[str, Any], where: str = "<memory>") -> GraphSnapshot:
    """Rebuild a snapshot from :func:`snapshot_to_dict` output.

    Args:
        data: The stored object.
        where: A path or label naming the source, for error messages.

    Raises:
        SkillWeaverError: the schema version is missing, not an integer or not
            :data:`SCHEMA_VERSION`, or a record is malformed.
    """
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SkillWeaverError(
            f"site graph at {where} has no usable schema_version (got {version!r}); "
            f"this build reads version {SCHEMA_VERSION}"
        )
    if version != SCHEMA_VERSION:
        raise SkillWeaverError(
            f"site graph at {where} has schema version {version}, but this build reads "
            f"version {SCHEMA_VERSION}; upgrade skillweaver or delete the file to relearn it"
        )
    domain = data.get("domain")
    if not isinstance(domain, str):
        raise SkillWeaverError(f"site graph at {where} has no domain")
    try:
        states = tuple(_state_from_dict(s) for s in data.get("states", ()))
        transitions = tuple(_transition_from_dict(t) for t in data.get("transitions", ()))
    except (KeyError, TypeError, ValueError) as exc:
        raise SkillWeaverError(f"site graph at {where} is malformed: {exc}") from exc
    return GraphSnapshot(domain=domain, states=states, transitions=transitions)


def _fingerprint_to_dict(fp: Fingerprint) -> dict[str, Any]:
    return {"value": fp.value, "parts": dict(fp.parts)}


def _fingerprint_from_dict(data: Mapping[str, Any]) -> Fingerprint:
    parts = data.get("parts") or {}
    return Fingerprint(value=str(data["value"]), parts={str(k): str(v) for k, v in parts.items()})


def _state_to_dict(state: UIState) -> dict[str, Any]:
    return {
        "fingerprint": _fingerprint_to_dict(state.fingerprint),
        "domain": state.domain,
        "label": state.label,
        "url_pattern": state.url_pattern,
        "first_seen": state.first_seen.isoformat(),
        "thumbnail": (
            None if state.thumbnail is None else base64.b64encode(state.thumbnail).decode("ascii")
        ),
    }


def _state_from_dict(data: Mapping[str, Any]) -> UIState:
    thumbnail = data.get("thumbnail")
    return UIState(
        fingerprint=_fingerprint_from_dict(data["fingerprint"]),
        domain=str(data.get("domain", "")),
        label=str(data.get("label", "")),
        url_pattern=data.get("url_pattern"),
        first_seen=_time_from_iso(data["first_seen"]),
        thumbnail=None if thumbnail is None else base64.b64decode(thumbnail),
    )


def _transition_to_dict(edge: Transition) -> dict[str, Any]:
    return {
        "src": _fingerprint_to_dict(edge.src),
        "dst": _fingerprint_to_dict(edge.dst),
        "actions": [action_to_dict(a) for a in edge.actions],
        "attempts": edge.attempts,
        "successes": edge.successes,
        "mean_ms": edge.mean_ms,
        "last_verified": None if edge.last_verified is None else edge.last_verified.isoformat(),
    }


def _transition_from_dict(data: Mapping[str, Any]) -> Transition:
    stamp = data.get("last_verified")
    return Transition(
        src=_fingerprint_from_dict(data["src"]),
        dst=_fingerprint_from_dict(data["dst"]),
        actions=tuple(action_from_dict(a) for a in data.get("actions", ())),
        attempts=int(data.get("attempts", 0)),
        successes=int(data.get("successes", 0)),
        mean_ms=float(data.get("mean_ms", 0.0)),
        last_verified=None if stamp is None else _time_from_iso(stamp),
    )


def _time_from_iso(text: str) -> datetime:
    """An ISO timestamp, forced to timezone-aware UTC."""
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
