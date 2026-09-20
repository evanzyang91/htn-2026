"""Full-text search over what the agent has recorded, through Elasticsearch - READ-ONLY
from the agent's side.

What this is for
----------------

The data directory holds everything a run leaves behind - trajectories, site graphs,
the skill library - as JSON files nobody can search without opening them one at a
time. Questions like *"which runs on walmart.com ended rejected?"*, *"which step ever
saw the words 'Your cart' on screen?"* or *"which stored skill calls
``find_text("Keycaps")``?"* are the ones a person asks when a live run went wrong, and
each currently means a grep over screenshots' worth of JSON.

This module flattens those records into five indices and asks Elasticsearch the
question. It changes NO agent decision: retrieval still ranks by the word-overlap in
``skills/retrieve.py``, the planner still decides, and nothing here is reachable from a
run. That is on purpose - a better RANKING was measured not to pay
(``skills/embed.py``, ``scripts/bench_retrieval.py``), and this module is not a second
attempt at it. It is observability.

Read-only, enforced and not promised
------------------------------------

Two doors, one on each side of the line:

* ``scripts/index_elastic.py`` is the ONLY writer. It reads the data directory and
  bulk-loads it; it is run by hand, after the runs it indexes, and never by a run.
* ``skillweaver search`` (``cli.py``) is the reader. It builds an
  :class:`ElasticClient` and calls :func:`search`; the client's ``search`` method issues
  ``GET``/``POST _search`` requests only. Nothing under ``src/`` ever calls
  :meth:`ElasticClient.bulk` or :meth:`ElasticClient.put_index`.

Both talk plain HTTP through ``urllib`` - Elasticsearch's REST API is JSON in, JSON out
- so no client library is added to ``pyproject.toml``, and a machine without
Elasticsearch loses exactly one subcommand. The address is ``SKILLWEAVER_ELASTIC_URL``
(``Settings.elastic_url``); unset, the reader says so and stops.

What is indexed
---------------

One document per thing a person might search for, each carrying ``domain`` and enough
context to be read on its own in a result list:

``skillweaver-runs``    one per recorded trajectory: task, outcome, the closing note,
                        how many steps, which action kinds, the URLs it passed through.
``skillweaver-steps``   one per action: the action, the policy's stated reason, the
                        critic's verdict, and the TEXT ON SCREEN before and after
                        (every element's text joined) - which is what makes "which
                        step saw X" answerable.
``skillweaver-states``  one per site-graph node: label, URL pattern, fingerprint.
``skillweaver-edges``   one per site-graph transition: the actions rendered as text,
                        attempts and successes, mean time.
``skillweaver-skills``  one per stored skill, latest version: summary, docstring, the
                        learned sentence, the code and the verifier. Searching this
                        index is a ``grep`` over the library and nothing more.

Screenshots are not indexed - the PNGs outweigh the JSON by three orders of magnitude
and Elasticsearch could do nothing with them. A result names ``run_id`` and step
``index``, which is what ``skillweaver replay <run_id>`` takes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from skillweaver.errors import SkillWeaverError

__all__ = [
    "INDICES",
    "MAPPINGS",
    "SETTINGS",
    "ElasticClient",
    "ElasticError",
    "Hit",
    "documents",
    "graph_documents",
    "run_documents",
    "search",
    "skill_documents",
]

INDEX_PREFIX = "skillweaver-"
KINDS: tuple[str, ...] = ("runs", "steps", "states", "edges", "skills")
"""The searchable kinds, in the order a mixed result list shows them."""

INDICES: dict[str, str] = {kind: INDEX_PREFIX + kind for kind in KINDS}

_TEXT = {"type": "text"}
_CODE = {"type": "text", "analyzer": "code"}
"""Source code and rendered actions. The standard analyzer keeps ``ctx.see.find_text``
as ONE token (a ``.`` between letters does not break a word under UAX#29), so a search
for ``find_text`` found nothing in a library where every skill calls it - measured on
the seven stored skills. The ``code`` analyzer splits on anything that is not a word
character, so identifiers, keys and quoted strings are each their own term."""

SETTINGS: dict[str, Any] = {
    "analysis": {
        "tokenizer": {"code_words": {"type": "pattern", "pattern": "[^A-Za-z0-9_]+"}},
        "analyzer": {"code": {"tokenizer": "code_words", "filter": ["lowercase"]}},
    }
}
"""Index settings every index is created with (``ElasticClient.put_index``)."""
_KEYWORD = {"type": "keyword"}
_DATE = {"type": "date"}
_BOOL = {"type": "boolean"}
_INT = {"type": "long"}
_FLOAT = {"type": "double"}

MAPPINGS: dict[str, dict[str, Any]] = {
    "runs": {
        "run_id": _KEYWORD,
        "domain": _KEYWORD,
        "task": _TEXT,
        "ok": _BOOL,
        "complete": _BOOL,
        "started_at": _DATE,
        "finished_at": _DATE,
        "duration_ms": _FLOAT,
        "note": _TEXT,
        "steps": _INT,
        "action_kinds": _KEYWORD,
        "rejected_steps": _INT,
        "failed_actions": _INT,
        "urls": _KEYWORD,
        "text": _TEXT,
    },
    "steps": {
        "run_id": _KEYWORD,
        "domain": _KEYWORD,
        "task": _TEXT,
        "index": _INT,
        "at": _DATE,
        "action_kind": _KEYWORD,
        "action": _CODE,
        "note": _TEXT,
        "result_ok": _BOOL,
        "error": _TEXT,
        "elapsed_ms": _FLOAT,
        "verdict_ok": _BOOL,
        "verdict_reason": _TEXT,
        "verdict_source": _KEYWORD,
        "url_before": _KEYWORD,
        "url_after": _KEYWORD,
        "fingerprint_before": _KEYWORD,
        "fingerprint_after": _KEYWORD,
        "screen_before": _TEXT,
        "screen_after": _TEXT,
    },
    "states": {
        "domain": _KEYWORD,
        "fingerprint": _KEYWORD,
        "label": _TEXT,
        "url_pattern": _KEYWORD,
        "first_seen": _DATE,
        "text": _TEXT,
    },
    "edges": {
        "domain": _KEYWORD,
        "src": _KEYWORD,
        "dst": _KEYWORD,
        "src_label": _TEXT,
        "dst_label": _TEXT,
        "actions": _CODE,
        "action_kinds": _KEYWORD,
        "attempts": _INT,
        "successes": _INT,
        "mean_ms": _FLOAT,
        "text": _TEXT,
    },
    "skills": {
        "name": _KEYWORD,
        "domain": _KEYWORD,
        "version": _INT,
        "summary": _TEXT,
        "docstring": _TEXT,
        "learned_from": _TEXT,
        "params": _KEYWORD,
        "code": _CODE,
        "verifier_code": _CODE,
        "action_signature": _KEYWORD,
        "demoted_reason": _TEXT,
        "runs": _INT,
        "successes": _INT,
        "mean_ms": _FLOAT,
        "text": _TEXT,
    },
}
"""Explicit mappings, so a ``keyword`` field filters exactly and a ``text`` field is
analysed. Every kind has a ``text`` field (steps: ``screen_before``/``screen_after``)
that :func:`search` queries by default, with the named fields boosted above it."""

_SEARCH_FIELDS: dict[str, list[str]] = {
    "runs": ["task^3", "note^2", "text", "urls"],
    "steps": [
        "action^3",
        "note^2",
        "verdict_reason^2",
        "error^2",
        "task",
        "screen_before",
        "screen_after",
    ],
    "states": ["label^3", "url_pattern^2", "text"],
    "edges": ["actions^3", "src_label", "dst_label", "text"],
    "skills": ["name^4", "summary^3", "learned_from^3", "docstring^2", "code", "verifier_code"],
}

_SCREEN_TEXT_LIMIT = 20_000
"""Characters of on-screen text kept per observation. A dense page has ~300 elements
of a few words each, so this is never reached in practice; it caps a pathological one."""


class ElasticError(SkillWeaverError):
    """Elasticsearch could not be reached or refused the request."""


# -- transport ---------------------------------------------------------------------------


class ElasticClient:
    """The smallest client that covers what this module does, over ``urllib``.

    Args:
        url: Base URL, e.g. ``http://127.0.0.1:9200``. A trailing slash is fine.
        api_key: Sent as ``Authorization: ApiKey ...`` when set. Read-only access is a
            property of the KEY (an ``read`` role over ``skillweaver-*``), not of this
            class; the reader path simply never calls the writing methods.
        timeout: Seconds per request.
    """

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.url}/{path.lstrip('/')}", data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", content_type)
        if self.api_key:
            request.add_header("Authorization", f"ApiKey {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ElasticError(f"{method} {path}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ElasticError(f"cannot reach Elasticsearch at {self.url}: {exc}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ElasticError(f"{method} {path}: reply is not JSON: {exc}") from exc

    # -- reading, the only methods the CLI path uses --

    def info(self) -> dict[str, Any]:
        """``GET /``: the cluster's name and version, or :class:`ElasticError`."""
        return self._request("GET", "/")

    def count(self, index: str) -> int:
        """Documents in ``index``, ``0`` when it does not exist."""
        try:
            return int(self._request("GET", f"/{index}/_count").get("count", 0))
        except ElasticError as exc:
            if "HTTP 404" in str(exc):
                return 0
            raise

    def search(self, index: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /<index>/_search`` with a query DSL ``body``."""
        return self._request("POST", f"/{index}/_search", json.dumps(body).encode("utf-8"))

    # -- writing, used by scripts/index_elastic.py ONLY --

    def put_index(self, index: str, properties: Mapping[str, Any], *, recreate: bool) -> bool:
        """Create ``index`` with :data:`SETTINGS` and ``properties`` as its mapping.
        Returns whether it was created; an existing index is left alone unless
        ``recreate``."""
        exists = "HTTP 404" not in str(self._probe(index))
        if exists and not recreate:
            return False
        if exists:
            self._request("DELETE", f"/{index}")
        self._request(
            "PUT",
            f"/{index}",
            json.dumps({"settings": SETTINGS, "mappings": {"properties": dict(properties)}}).encode(
                "utf-8"
            ),
        )
        return True

    def _probe(self, index: str) -> str:
        try:
            self._request("GET", f"/{index}")
        except ElasticError as exc:
            return str(exc)
        return ""

    def bulk(self, index: str, docs: Sequence[tuple[str, Mapping[str, Any]]]) -> int:
        """Index ``(id, document)`` pairs with one ``_bulk`` request; returns how many
        were accepted and raises on the first refused one."""
        if not docs:
            return 0
        lines: list[str] = []
        for doc_id, doc in docs:
            lines.append(json.dumps({"index": {"_index": index, "_id": doc_id}}))
            lines.append(json.dumps(doc, default=str))
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        reply = self._request("POST", "/_bulk?refresh=true", payload, "application/x-ndjson")
        if reply.get("errors"):
            for item in reply.get("items", []):
                error = item.get("index", {}).get("error")
                if error:
                    raise ElasticError(f"{index}: {error.get('type')}: {error.get('reason')}")
        return len(docs)


# -- documents ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Doc:
    """One document to index: its ``kind`` (a key of :data:`INDICES`), id and body."""

    kind: str
    id: str
    body: dict[str, Any]


def _screen_text(observation: Mapping[str, Any]) -> str:
    """Every element's text on one observation, in file order, joined by newlines."""
    elements = observation.get("elements")
    if not isinstance(elements, list):
        return ""
    texts = [
        str(e.get("text", "")).strip()
        for e in elements
        if isinstance(e, Mapping) and str(e.get("text", "")).strip()
    ]
    return "\n".join(texts)[:_SCREEN_TEXT_LIMIT]


def _action_text(action: Mapping[str, Any]) -> str:
    """A one-line rendering of an action dict, kind first, its salient value after."""
    kind = str(action.get("kind", "?"))
    if kind == "type_text":
        return f"type_text {action.get('text', '')!r}"
    if kind == "press_key":
        keys = action.get("keys") or action.get("key") or action.get("combo") or ""
        return f"press_key {keys}"
    if kind == "navigate":
        return f"navigate {action.get('url', '')}"
    if kind == "click":
        point = action.get("point") or {}
        return f"click ({point.get('x', '?')}, {point.get('y', '?')})"
    if kind == "scroll":
        return f"scroll dx={action.get('dx', 0)} dy={action.get('dy', 0)}"
    if kind == "wait":
        return f"wait {action.get('ms', '?')}ms"
    return kind


def _moment(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        datetime.fromisoformat(raw)
    except ValueError:
        return None
    return raw


def _duration_ms(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except ValueError:
        return None
    return delta.total_seconds() * 1000.0


def run_documents(trajectories_dir: Path) -> Iterator[Doc]:
    """One ``runs`` document and one ``steps`` document per action, for every
    ``trajectory.jsonl`` under ``trajectories_dir``. Screenshots are never read."""
    from skillweaver.trajectory.store import TRAJECTORY_FILE, read_lines

    if not trajectories_dir.is_dir():
        return
    for run_dir in sorted(trajectories_dir.iterdir()):
        path = run_dir / TRAJECTORY_FILE
        if not path.is_file():
            continue
        header: Mapping[str, Any] | None = None
        footer: Mapping[str, Any] | None = None
        steps: list[Mapping[str, Any]] = []
        try:
            for record in read_lines(path):
                kind = record.get("type")
                if kind == "header":
                    header = record
                elif kind == "footer":
                    footer = record
                elif kind == "step":
                    steps.append(record)
        except SkillWeaverError:
            continue  # a corrupt recording is not searchable; the store reports it
        if header is None or not header.get("run_id"):
            continue
        run_id = str(header["run_id"])
        domain = str(header.get("domain", ""))
        task = str(header.get("task", ""))
        started = _moment(header.get("started_at"))
        finished = _moment(footer.get("finished_at")) if footer else None
        urls: list[str] = []
        kinds: list[str] = []
        rejected = 0
        failed = 0
        notes: list[str] = []
        for step in steps:
            action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
            before = step.get("before") if isinstance(step.get("before"), Mapping) else {}
            after = step.get("after") if isinstance(step.get("after"), Mapping) else {}
            result = step.get("result") if isinstance(step.get("result"), Mapping) else {}
            verdict = step.get("verdict") if isinstance(step.get("verdict"), Mapping) else None
            kinds.append(str(action.get("kind", "?")))
            for url in (before.get("url"), after.get("url")):
                if isinstance(url, str) and url and url not in urls:
                    urls.append(url)
            if result.get("ok") is False:
                failed += 1
            if verdict is not None and verdict.get("ok") is False:
                rejected += 1
            note = str(step.get("note", "") or "")
            if note:
                notes.append(note)
            index = int(step.get("index", len(steps)))
            body = {
                "run_id": run_id,
                "domain": domain,
                "task": task,
                "index": index,
                "at": _moment((before.get("screenshot") or {}).get("captured_at")),
                "action_kind": str(action.get("kind", "?")),
                "action": _action_text(action),
                "note": note,
                "result_ok": result.get("ok"),
                "error": result.get("error"),
                "elapsed_ms": result.get("elapsed_ms"),
                "verdict_ok": None if verdict is None else verdict.get("ok"),
                "verdict_reason": None if verdict is None else verdict.get("reason"),
                "verdict_source": None if verdict is None else verdict.get("source"),
                "url_before": before.get("url"),
                "url_after": after.get("url"),
                "fingerprint_before": (before.get("fingerprint") or {}).get("value"),
                "fingerprint_after": (after.get("fingerprint") or {}).get("value"),
                "screen_before": _screen_text(before),
                "screen_after": _screen_text(after),
            }
            yield Doc("steps", f"{run_id}:{index}", body)
        yield Doc(
            "runs",
            run_id,
            {
                "run_id": run_id,
                "domain": domain,
                "task": task,
                "ok": bool(footer.get("ok", False)) if footer else False,
                "complete": footer is not None,
                "started_at": started,
                "finished_at": finished,
                "duration_ms": _duration_ms(started, finished),
                "note": str(footer.get("note", "")) if footer else "incomplete",
                "steps": len(steps),
                "action_kinds": sorted(set(kinds)),
                "rejected_steps": rejected,
                "failed_actions": failed,
                "urls": urls,
                "text": "\n".join(notes),
            },
        )


def graph_documents(graphs_dir: Path) -> Iterator[Doc]:
    """One ``states`` document per node and one ``edges`` document per transition, for
    every site-graph JSON under ``graphs_dir``. Thumbnails are dropped."""
    if not graphs_dir.is_dir():
        return
    for path in sorted(graphs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        domain = str(data.get("domain", path.stem))
        labels: dict[str, str] = {}
        for state in data.get("states") or []:
            if not isinstance(state, Mapping):
                continue
            fingerprint = str((state.get("fingerprint") or {}).get("value", ""))
            if not fingerprint:
                continue
            label = str(state.get("label", "") or "")
            labels[fingerprint] = label
            url_pattern = state.get("url_pattern")
            yield Doc(
                "states",
                f"{domain}:{fingerprint}",
                {
                    "domain": domain,
                    "fingerprint": fingerprint,
                    "label": label,
                    "url_pattern": url_pattern,
                    "first_seen": _moment(state.get("first_seen")),
                    "text": " ".join(s for s in (label, str(url_pattern or "")) if s),
                },
            )
        for position, edge in enumerate(data.get("transitions") or []):
            if not isinstance(edge, Mapping):
                continue
            src = str((edge.get("src") or {}).get("value", ""))
            dst = str((edge.get("dst") or {}).get("value", ""))
            actions = [a for a in (edge.get("actions") or []) if isinstance(a, Mapping)]
            rendered = " -> ".join(_action_text(a) for a in actions) or "no actions"
            yield Doc(
                "edges",
                f"{domain}:{src}:{dst}:{position}",
                {
                    "domain": domain,
                    "src": src,
                    "dst": dst,
                    "src_label": labels.get(src, ""),
                    "dst_label": labels.get(dst, ""),
                    "actions": rendered,
                    "action_kinds": sorted({str(a.get("kind", "?")) for a in actions}),
                    "attempts": int(edge.get("attempts", 0) or 0),
                    "successes": int(edge.get("successes", 0) or 0),
                    "mean_ms": float(edge.get("mean_ms", 0.0) or 0.0),
                    "text": " ".join(
                        s for s in (labels.get(src, ""), rendered, labels.get(dst, "")) if s
                    ),
                },
            )


def skill_documents(skills_dir: Path) -> Iterator[Doc]:
    """One ``skills`` document per stored skill, latest version on disk, from the
    ``meta.json`` / ``skill.py`` / ``verify.py`` triple the file store writes."""
    if not skills_dir.is_dir():
        return
    for domain_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        for skill_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            versions = sorted(
                (int(p.name[1:]), p)
                for p in skill_dir.iterdir()
                if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
            )
            if not versions:
                continue
            version, version_dir = versions[-1]
            try:
                meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, Mapping):
                continue
            code = _read_optional(version_dir / "skill.py")
            verifier = _read_optional(version_dir / "verify.py")
            provenance = (
                meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
            )
            stats = meta.get("stats") if isinstance(meta.get("stats"), Mapping) else {}
            name = str(meta.get("name", skill_dir.name))
            domain = str(meta.get("domain", domain_dir.name))
            summary = str(meta.get("summary", "") or "")
            docstring = str(meta.get("docstring", "") or "")
            learned_from = str(provenance.get("task_text", "") or "")
            precedents = [
                str(p.get("task_text", "") or "")
                for p in (meta.get("precedents") or [])
                if isinstance(p, Mapping)
            ]
            yield Doc(
                "skills",
                f"{domain}:{name}",
                {
                    "name": name,
                    "domain": domain,
                    "version": version,
                    "summary": summary,
                    "docstring": docstring,
                    "learned_from": learned_from,
                    "params": sorted((meta.get("params") or {}).keys()),
                    "code": code,
                    "verifier_code": verifier,
                    "action_signature": list(meta.get("action_signature") or []),
                    "demoted_reason": meta.get("demoted_reason"),
                    "runs": int(stats.get("runs", 0) or 0),
                    "successes": int(stats.get("successes", 0) or 0),
                    "mean_ms": float(stats.get("mean_ms", 0.0) or 0.0),
                    "text": "\n".join(s for s in [summary, learned_from, *precedents] if s),
                },
            )


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def documents(data_dir: Path, kinds: Sequence[str] = KINDS) -> Iterator[Doc]:
    """Every document of the requested ``kinds`` under one data directory."""
    wanted = set(kinds)
    if wanted & {"runs", "steps"}:
        for doc in run_documents(data_dir / "trajectories"):
            if doc.kind in wanted:
                yield doc
    if wanted & {"states", "edges"}:
        for doc in graph_documents(data_dir / "graphs"):
            if doc.kind in wanted:
                yield doc
    if "skills" in wanted:
        yield from skill_documents(data_dir / "skills")


# -- reading -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: which ``kind`` it came from, its score, the stored document
    and the analysed fragments Elasticsearch highlighted (``field -> fragments``)."""

    kind: str
    score: float
    doc: dict[str, Any]
    highlights: dict[str, list[str]] = field(default_factory=dict)

    def where(self) -> str:
        """A one-line locator a person can act on: run and step, node, edge or skill."""
        d = self.doc
        if self.kind == "runs":
            return f"run {d.get('run_id')}"
        if self.kind == "steps":
            return f"run {d.get('run_id')} step {d.get('index')}"
        if self.kind == "states":
            return f"state {d.get('fingerprint')}"
        if self.kind == "edges":
            return f"edge {str(d.get('src'))[:8]} -> {str(d.get('dst'))[:8]}"
        if self.kind == "skills":
            return f"skill {d.get('name')} v{d.get('version')}"
        return self.kind

    def headline(self) -> str:
        """What the result IS, in the record's own words."""
        d = self.doc
        if self.kind == "runs":
            state = "ok" if d.get("ok") else ("incomplete" if not d.get("complete") else "failed")
            return f"[{state}] {d.get('task', '')}"
        if self.kind == "steps":
            verdict = d.get("verdict_ok")
            mark = "" if verdict is None else (" [accepted]" if verdict else " [rejected]")
            return f"{d.get('action', '')}{mark}: {d.get('note', '')}"
        if self.kind == "states":
            return f"{d.get('label') or '(unlabelled)'} {d.get('url_pattern') or ''}".strip()
        if self.kind == "edges":
            return f"{d.get('actions', '')} ({d.get('successes')}/{d.get('attempts')})"
        if self.kind == "skills":
            return str(d.get("summary", ""))
        return ""


def search(
    client: ElasticClient,
    query: str,
    *,
    kinds: Sequence[str] = KINDS,
    domain: str | None = None,
    k: int = 10,
) -> tuple[list[Hit], float]:
    """Search ``kinds`` for ``query``, best score first across all of them, and the
    wall time in milliseconds. ``domain`` filters exactly on the stored domain.

    The query is Elasticsearch ``query_string`` syntax over each kind's own fields
    (``_SEARCH_FIELDS``) with ``AND`` between terms, so every word must appear somewhere
    in a document; a phrase in double quotes is matched as a phrase, and a
    ``field:value`` term reaches ANY stored field, which is how a count or a flag is
    asked about - ``ok:false AND rejected_steps:>0`` for runs the critic turned down,
    ``verdict_ok:false`` for the steps it rejected, ``action_kind:navigate``.
    Highlights come back per field.
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ElasticError(f"unknown kinds {sorted(unknown)}; choose from {list(KINDS)}")
    if not query.strip():
        raise ElasticError("an empty query matches nothing")
    hits: list[Hit] = []
    started = time.perf_counter()
    for kind in kinds:
        must: dict[str, Any] = {
            "query_string": {
                "query": query,
                "fields": _SEARCH_FIELDS[kind],
                "default_operator": "AND",
                "lenient": True,
            }
        }
        body: dict[str, Any] = {
            "size": k,
            "query": {"bool": {"must": [must]}},
            "highlight": {
                "fields": {f.split("^")[0]: {} for f in _SEARCH_FIELDS[kind]},
                "fragment_size": 160,
                "number_of_fragments": 2,
                "pre_tags": ["«"],
                "post_tags": ["»"],
            },
        }
        if domain:
            body["query"]["bool"]["filter"] = [{"term": {"domain": domain}}]
        try:
            reply = client.search(INDICES[kind], body)
        except ElasticError as exc:
            if "HTTP 404" in str(exc):
                continue  # not indexed yet: nothing of this kind to find
            raise
        for raw in reply.get("hits", {}).get("hits", []):
            hits.append(
                Hit(
                    kind=kind,
                    score=float(raw.get("_score") or 0.0),
                    doc=dict(raw.get("_source") or {}),
                    highlights={k_: list(v) for k_, v in (raw.get("highlight") or {}).items()},
                )
            )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k], (time.perf_counter() - started) * 1000.0
