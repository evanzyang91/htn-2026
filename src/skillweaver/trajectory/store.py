"""The on-disk trajectory format, and :class:`TrajectoryFileStore` over it.

Four parts of the project read this: skill synthesis compiles from it, the admission gate
replays against it, the eval harness scores it, the dashboard shows it as a filmstrip.
They want different slices, so the layout separates them::

    <settings().trajectories_dir>/
        20260919T120000000000Z__a1b2c3d4e5f6/       one directory per run
            trajectory.jsonl                        append-only, one JSON object per line
            screens/2f6c1b3d9a04e7f5.png            one file per DISTINCT screenshot

``trajectory.jsonl`` holds one ``header`` line, one ``step`` line per action, and one
``footer``, in that order. A file WITHOUT a footer is a run that crashed: it still loads,
as the steps that reached the disk, with ``ok=False``.

Each read costs what it needs: :meth:`~TrajectoryFileStore.list` is one ``scandir`` and no
open; ``summaries`` reads two lines per run; ``frames`` reads the step lines and returns
PNG PATHS; ``load(screenshots=False)`` skips the images; ``load`` reads each distinct PNG
once. The directory name carries the start time so listing oldest-first is a name sort,
and screenshots are stored by content digest because a run's PNGs outweigh its JSON by
three orders of magnitude and consecutive steps share a screen.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillweaver.config import settings
from skillweaver.contracts import (
    ActionResult,
    Box,
    Element,
    ElementIndex,
    ElementKind,
    ElementSource,
    Fingerprint,
    Observation,
    Point,
    Screenshot,
    Trajectory,
    TrajectoryStep,
    Verdict,
    action_from_dict,
    action_to_dict,
)
from skillweaver.errors import SkillWeaverError

SCHEMA_VERSION = 1
"""The format version written into every header line. :func:`decode_header` refuses
anything else, because a newer writer may inline fields this reader would drop."""

TRAJECTORY_FILE = "trajectory.jsonl"
SCREENS_DIR = "screens"
INCOMPLETE_NOTE = "incomplete: the run did not finish"
"""The ``note`` given to a trajectory whose file has no footer line."""

_TAIL_BYTES = 16384
"""How much of a file's end :func:`read_ends` reads to find the footer. A footer
line is ~120 bytes; this leaves room for a long torn step line in front of it."""

_DIR_TIME_FORMAT = "%Y%m%dT%H%M%S%f"
_DIR_RE = re.compile(r"^\d{8}T\d{12}Z__(?P<run_id>[A-Za-z0-9_-]+)$")


def run_dir_name(started_at: datetime, run_id: str) -> str:
    """A sortable UTC start time, then the ``run_id``: sorting these names is sorting runs
    oldest-first, which is why :meth:`TrajectoryFileStore.list` opens no file."""
    stamp = started_at.astimezone(UTC).strftime(_DIR_TIME_FORMAT)
    return f"{stamp}Z__{run_id}"


def run_id_of(dir_name: str) -> str | None:
    """The ``run_id`` encoded in ``dir_name``, or ``None`` if it is not a run dir."""
    match = _DIR_RE.match(dir_name)
    return match.group("run_id") if match else None


def screenshot_digest(png: bytes) -> str:
    """The content address a screenshot is stored under: 16 hex chars of SHA-256."""
    return hashlib.sha256(png).hexdigest()[:16]


def write_screenshot(screens_dir: Path, png: bytes) -> str:
    """Store ``png`` under its digest if it is not there, and return the digest.

    Atomic (temp file then ``os.replace``), so a crash mid-write never leaves a half PNG a
    later step line would claim to be whole.
    """
    digest = screenshot_digest(png)
    path = screens_dir / f"{digest}.png"
    if path.exists():
        return digest
    screens_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".png.{os.getpid()}.tmp")
    tmp.write_bytes(png)
    os.replace(tmp, path)
    return digest


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _moment(raw: Any, field: str) -> datetime:
    if not isinstance(raw, str):
        raise SkillWeaverError(f"{field} must be an ISO timestamp string, got {raw!r}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SkillWeaverError(f"{field} is not an ISO timestamp: {raw!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _mapping(raw: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SkillWeaverError(f"{field} must be an object, got {type(raw).__name__}")
    return raw


def encode_element(element: Element) -> dict[str, Any]:
    """An :class:`Element` as a JSON-safe dict; boxes stay LOGICAL pixels."""
    box = element.box
    return {
        "box": [box.x, box.y, box.w, box.h],
        "kind": str(element.kind),
        "text": element.text,
        "confidence": element.confidence,
        "stable_id": element.stable_id,
        "source": str(element.source),
    }


def decode_element(data: Mapping[str, Any]) -> Element:
    """Rebuild an :class:`Element` from :func:`encode_element` output."""
    box = data.get("box")
    if not isinstance(box, Sequence) or len(box) != 4:
        raise SkillWeaverError(f"element box must be [x, y, w, h], got {box!r}")
    return Element(
        box=Box(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        kind=ElementKind(data.get("kind", ElementKind.other)),
        text=str(data.get("text", "")),
        confidence=float(data.get("confidence", 1.0)),
        stable_id=data.get("stable_id"),
        source=ElementSource(data.get("source", ElementSource.merged)),
    )


def encode_observation(observation: Observation, digest: str) -> dict[str, Any]:
    """An :class:`Observation` as a dict, its screenshot referenced by ``digest``.

    ``index`` is not stored: it is a view over ``elements``, rebuilt on load.
    """
    shot = observation.screenshot
    return {
        "screenshot": {
            "png": digest,
            "width": shot.width,
            "height": shot.height,
            "scale": shot.scale,
            "captured_at": _iso(shot.captured_at),
        },
        "elements": [encode_element(e) for e in observation.elements],
        "fingerprint": {
            "value": observation.fingerprint.value,
            "parts": dict(observation.fingerprint.parts),
        },
        "url": observation.url,
        "taken_at": _iso(observation.taken_at),
    }


def encode_step(step: TrajectoryStep, before_digest: str, after_digest: str) -> dict[str, Any]:
    """One ``step`` line: the action, the screens around it, and what came of it."""
    line: dict[str, Any] = {
        "type": "step",
        "index": step.index,
        "action": action_to_dict(step.action),
        "before": encode_observation(step.before, before_digest),
        "after": encode_observation(step.after, after_digest),
        "result": {
            "ok": step.result.ok,
            "error": step.result.error,
            "elapsed_ms": step.result.elapsed_ms,
        },
        "note": step.note,
    }
    if step.verdict is not None:
        line["verdict"] = {
            "ok": step.verdict.ok,
            "reason": step.verdict.reason,
            "confidence": step.verdict.confidence,
            "source": step.verdict.source,
        }
    return line


def encode_header(trajectory_or_run: Mapping[str, Any]) -> dict[str, Any]:
    """The ``header`` line from ``run_id``/``task``/``domain``/``started_at`` values."""
    return {"type": "header", "schema": SCHEMA_VERSION, **dict(trajectory_or_run)}


def encode_footer(ok: bool, finished_at: datetime, note: str) -> dict[str, Any]:
    """The ``footer`` line closing a run."""
    return {"type": "footer", "ok": ok, "finished_at": _iso(finished_at), "note": note}


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Header:
    """The opening line of a trajectory file."""

    run_id: str
    task: str
    domain: str
    started_at: datetime


@dataclass(frozen=True, slots=True)
class Footer:
    """The closing line of a trajectory file; absent when the run crashed."""

    ok: bool
    finished_at: datetime
    note: str


def decode_header(data: Mapping[str, Any]) -> Header:
    """Read a ``header`` line.

    Raises:
        SkillWeaverError: the schema version is not :data:`SCHEMA_VERSION`, or a required
            field is missing.
    """
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise SkillWeaverError(
            f"trajectory schema version {schema!r} is not supported "
            f"(this build reads version {SCHEMA_VERSION})"
        )
    for field in ("run_id", "task", "domain", "started_at"):
        if field not in data:
            raise SkillWeaverError(f"trajectory header is missing {field!r}")
    return Header(
        run_id=str(data["run_id"]),
        task=str(data["task"]),
        domain=str(data["domain"]),
        started_at=_moment(data["started_at"], "header started_at"),
    )


def decode_footer(data: Mapping[str, Any]) -> Footer:
    """Read a ``footer`` line."""
    return Footer(
        ok=bool(data.get("ok", False)),
        finished_at=_moment(data.get("finished_at"), "footer finished_at"),
        note=str(data.get("note", "")),
    )


def _decode_observation(
    data: Mapping[str, Any],
    screens_dir: Path,
    cache: dict[str, bytes],
    *,
    screenshots: bool,
    index_factory: Callable[[Sequence[Element]], ElementIndex],
) -> Observation:
    shot = _mapping(data.get("screenshot"), "observation screenshot")
    digest = str(shot.get("png", ""))
    elements = tuple(decode_element(_mapping(e, "element")) for e in data.get("elements", ()))
    fingerprint = _mapping(data.get("fingerprint"), "observation fingerprint")
    return Observation(
        screenshot=Screenshot(
            png=_read_png(screens_dir, digest, cache) if screenshots else b"",
            width=int(shot.get("width", 0)),
            height=int(shot.get("height", 0)),
            scale=float(shot.get("scale", 1.0)),
            captured_at=_moment(shot.get("captured_at"), "screenshot captured_at"),
        ),
        elements=elements,
        index=index_factory(elements),
        fingerprint=Fingerprint(
            str(fingerprint.get("value", "")), dict(fingerprint.get("parts", {}))
        ),
        url=data.get("url"),
        taken_at=_moment(data.get("taken_at"), "observation taken_at"),
    )


def _read_png(screens_dir: Path, digest: str, cache: dict[str, bytes]) -> bytes:
    """Read one stored screenshot, at most once per load however often it is shared."""
    png = cache.get(digest)
    if png is None:
        path = screens_dir / f"{digest}.png"
        try:
            png = path.read_bytes()
        except OSError as exc:
            raise SkillWeaverError(f"screenshot {digest} is missing or unreadable: {exc}") from exc
        cache[digest] = png
    return png


def _decode_step(
    data: Mapping[str, Any],
    screens_dir: Path,
    cache: dict[str, bytes],
    *,
    screenshots: bool,
    index_factory: Callable[[Sequence[Element]], ElementIndex],
) -> TrajectoryStep:
    try:
        action = action_from_dict(_mapping(data.get("action"), "step action"))
    except ValueError as exc:
        raise SkillWeaverError(f"step {data.get('index')} has an unreadable action: {exc}") from exc
    result = _mapping(data.get("result", {}), "step result")
    verdict = data.get("verdict")
    return TrajectoryStep(
        index=int(data.get("index", 0)),
        action=action,
        before=_decode_observation(
            _mapping(data.get("before"), "step before"),
            screens_dir,
            cache,
            screenshots=screenshots,
            index_factory=index_factory,
        ),
        after=_decode_observation(
            _mapping(data.get("after"), "step after"),
            screens_dir,
            cache,
            screenshots=screenshots,
            index_factory=index_factory,
        ),
        result=ActionResult(
            ok=bool(result.get("ok", False)),
            error=result.get("error"),
            elapsed_ms=float(result.get("elapsed_ms", 0.0)),
        ),
        verdict=(
            Verdict(
                ok=bool(verdict.get("ok", False)),
                reason=str(verdict.get("reason", "")),
                confidence=float(verdict.get("confidence", 1.0)),
                source=verdict.get("source", "programmatic"),
            )
            if isinstance(verdict, Mapping)
            else None
        ),
        note=str(data.get("note", "")),
    )


def read_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a trajectory file, tolerating a truncated tail.

    A partial FINAL line is dropped - an interrupted run is still evidence - while an
    unreadable line with more after it is real corruption.

    Raises:
        SkillWeaverError: a malformed line that is not the last, or an unreadable file.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SkillWeaverError(f"cannot read {path}: {exc}") from exc
    lines = text.splitlines()
    ends_cleanly = text.endswith("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines) and not ends_cleanly:
                return  # torn tail of a crashed run: everything before it still counts
            raise SkillWeaverError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise SkillWeaverError(f"{path}:{number} is not a JSON object")
        yield record


def read_ends(path: Path) -> tuple[Header, Footer | None]:
    """Read only the first and last line of a trajectory file.

    What a listing wants: the steps in between are kilobytes of elements per action and
    are never touched. The footer is ``None`` for a run that crashed.

    Raises:
        SkillWeaverError: the file cannot be read, or its first line is not a header.
    """
    try:
        with open(path, "rb") as handle:
            first = handle.readline()
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - _TAIL_BYTES))
            tail = handle.read()
    except OSError as exc:
        raise SkillWeaverError(f"cannot read {path}: {exc}") from exc
    try:
        opening = json.loads(first)
    except json.JSONDecodeError as exc:
        raise SkillWeaverError(f"{path}:1 is not valid JSON: {exc}") from exc
    if not isinstance(opening, dict):
        raise SkillWeaverError(f"{path}:1 is not a JSON object")
    header = decode_header(opening)

    segments = [segment for segment in tail.split(b"\n") if segment.strip()]
    if not tail.endswith(b"\n"):
        segments = segments[:-1]  # a torn final line is not a footer
    if not segments:
        return header, None
    try:
        closing = json.loads(segments[-1])
    except json.JSONDecodeError:
        return header, None
    if isinstance(closing, dict) and closing.get("type") == "footer":
        return header, decode_footer(closing)
    return header, None


def _reading_order(element: Element) -> tuple[int, int]:
    return (element.box.y, element.box.x)


class ReplayElementIndex:
    """A small ``contracts.ElementIndex`` over a reloaded observation's elements.

    A loaded ``Observation`` needs an index and its original was a view, not data.
    Deliberately plain; pass ``index_factory`` to use the project's real index.
    """

    def __init__(self, elements: Sequence[Element]) -> None:
        self._elements = sorted(elements, key=_reading_order)

    def all(self) -> list[Element]:
        return list(self._elements)

    def by_kind(self, kind: ElementKind) -> list[Element]:
        return [e for e in self._elements if e.kind == kind]

    def find_text(
        self, query: str, kind: ElementKind | None = None, fuzzy: bool = True
    ) -> list[Element]:
        wanted = query.strip().lower()
        if not wanted:
            return []
        scored: list[tuple[float, Element]] = []
        for element in self._elements:
            if kind is not None and element.kind != kind:
                continue
            text = element.text.strip().lower()
            if not text:
                continue
            if text == wanted:
                score = 3.0
            elif wanted in text:
                score = 2.0 + len(wanted) / len(text)
            elif fuzzy:
                ratio = difflib.SequenceMatcher(None, wanted, text).ratio()
                score = ratio if ratio >= 0.6 else 0.0
            else:
                score = 0.0
            if score > 0:
                scored.append((score, element))
        scored.sort(key=lambda pair: -pair[0])
        return [element for _, element in scored]

    def nearest(self, point: Point, kind: ElementKind | None = None) -> list[Element]:
        def distance(element: Element) -> float:
            box = element.box
            dx = max(box.x - point.x, 0, point.x - (box.x + box.w - 1))
            dy = max(box.y - point.y, 0, point.y - (box.y + box.h - 1))
            return math.hypot(dx, dy)

        return sorted((e for e in self._elements if kind is None or e.kind == kind), key=distance)

    def containing(self, point: Point) -> list[Element]:
        return sorted(
            (e for e in self._elements if e.box.contains(point)), key=lambda e: e.box.area
        )

    def best(self, description: str) -> list[Element]:
        words = set(description.lower().split())
        scored: list[tuple[float, Element]] = []
        for element in self._elements:
            score = float(len(words & set(element.text.lower().split())))
            if element.kind.value in words:
                score += 1.5
            if score > 0:
                scored.append((score, element))
        scored.sort(key=lambda pair: -pair[0])
        return [element for _, element in scored]


@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    """One run at a glance: everything but its steps and its pixels.

    ``complete`` is ``False`` for a run whose file has no footer - it was killed - in
    which case ``ok`` is ``False`` and ``finished_at`` is the start time.
    """

    run_id: str
    task: str
    domain: str
    ok: bool
    started_at: datetime
    finished_at: datetime
    note: str
    complete: bool
    path: Path


@dataclass(frozen=True, slots=True)
class Frame:
    """Where one step's two screenshots live, without reading either of them."""

    index: int
    before: Path
    after: Path


def _summarize(header: Header, footer: Footer | None, directory: Path) -> TrajectorySummary:
    """Fold a file's two end lines into a :class:`TrajectorySummary`."""
    return TrajectorySummary(
        run_id=header.run_id,
        task=header.task,
        domain=header.domain,
        ok=footer.ok if footer else False,
        started_at=header.started_at,
        finished_at=footer.finished_at if footer else header.started_at,
        note=footer.note if footer else INCOMPLETE_NOTE,
        complete=footer is not None,
        path=directory,
    )


class TrajectoryFileStore:
    """A ``contracts.TrajectoryStore`` over the directory layout described above.

    Never caches: every call reads what is on disk now, so a recorder writing a run and a
    store reading it can be different processes.

    Args:
        root: The directory holding run directories. ``None`` means
            ``settings().trajectories_dir``; nothing here hardcodes a path.
        index_factory: Builds a loaded observation's ``ElementIndex``.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        index_factory: Callable[[Sequence[Element]], ElementIndex] = ReplayElementIndex,
    ) -> None:
        self.root = Path(root) if root is not None else settings().trajectories_dir
        self._index_factory = index_factory

    # -- TrajectoryStore protocol ------------------------------------------------------

    def save(self, trajectory: Trajectory) -> None:
        """Write ``trajectory`` whole, replacing any run with the same ``run_id``.

        A recorder has usually written this file step by step already; saving over it is
        the same bytes and safe to repeat.

        Raises:
            SkillWeaverError: the data directory cannot be written.
        """
        directory = self._existing_dir(trajectory.run_id) or self.root / run_dir_name(
            trajectory.started_at, trajectory.run_id
        )
        screens = directory / SCREENS_DIR
        try:
            screens.mkdir(parents=True, exist_ok=True)
            lines: list[dict[str, Any]] = [
                encode_header(
                    {
                        "run_id": trajectory.run_id,
                        "task": trajectory.task,
                        "domain": trajectory.domain,
                        "started_at": _iso(trajectory.started_at),
                    }
                )
            ]
            for step in trajectory.steps:
                lines.append(
                    encode_step(
                        step,
                        write_screenshot(screens, step.before.screenshot.png),
                        write_screenshot(screens, step.after.screenshot.png),
                    )
                )
            lines.append(encode_footer(trajectory.ok, trajectory.finished_at, trajectory.note))
            body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
            path = directory / TRAJECTORY_FILE
            tmp = path.with_suffix(f".jsonl.{os.getpid()}.tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            raise SkillWeaverError(f"cannot write trajectory {trajectory.run_id}: {exc}") from exc

    def load(self, run_id: str, *, screenshots: bool = True) -> Trajectory:
        """Return one trajectory, whole.

        With ``screenshots=False`` every ``Screenshot.png`` is ``b""`` and no PNG is read -
        what skill synthesis wants. A run whose file has no footer loads as the steps that
        reached the disk, with ``ok=False`` and :data:`INCOMPLETE_NOTE`.

        Raises:
            SkillWeaverError: unknown ``run_id``, unsupported schema, or unreadable data.
        """
        directory = self._require_dir(run_id)
        screens = directory / SCREENS_DIR
        cache: dict[str, bytes] = {}
        header: Header | None = None
        footer: Footer | None = None
        steps: list[TrajectoryStep] = []
        for record in read_lines(directory / TRAJECTORY_FILE):
            match record.get("type"):
                case "header":
                    header = decode_header(record)
                case "step":
                    steps.append(
                        _decode_step(
                            record,
                            screens,
                            cache,
                            screenshots=screenshots,
                            index_factory=self._index_factory,
                        )
                    )
                case "footer":
                    footer = decode_footer(record)
                case other:
                    raise SkillWeaverError(
                        f"trajectory {run_id} has an unknown line type {other!r}"
                    )
        if header is None:
            raise SkillWeaverError(f"trajectory {run_id} has no header line")
        last_seen = steps[-1].after.taken_at if steps else header.started_at
        return Trajectory(
            run_id=header.run_id,
            task=header.task,
            domain=header.domain,
            steps=tuple(steps),
            ok=footer.ok if footer else False,
            started_at=header.started_at,
            finished_at=footer.finished_at if footer else last_seen,
            note=footer.note if footer else INCOMPLETE_NOTE,
        )

    def list(self) -> list[str]:
        """Every stored ``run_id``, oldest first, from one ``scandir`` and no open."""
        return [run_id for run_id, _ in self._run_dirs()]

    # -- cheap views -------------------------------------------------------------------

    def summary(self, run_id: str) -> TrajectorySummary:
        """One run's header and footer, reading neither its steps nor its pixels."""
        directory = self._require_dir(run_id)
        return _summarize(*read_ends(directory / TRAJECTORY_FILE), directory)

    def summaries(self) -> list[TrajectorySummary]:
        """A :meth:`summary` per stored run, oldest first. Unparseable runs are skipped:
        one bad run must not hide the other nine hundred and ninety-nine."""
        out: list[TrajectorySummary] = []
        for _, directory in self._run_dirs():  # one scandir, then two reads per run
            try:
                out.append(_summarize(*read_ends(directory / TRAJECTORY_FILE), directory))
            except SkillWeaverError:
                continue
        return out

    def frames(self, run_id: str) -> list[Frame]:
        """Where each step's screenshots live, in step order; reads no image bytes."""
        directory = self._require_dir(run_id)
        screens = directory / SCREENS_DIR
        out: list[Frame] = []
        for record in read_lines(directory / TRAJECTORY_FILE):
            if record.get("type") != "step":
                continue
            before = _mapping(record.get("before"), "step before")
            after = _mapping(record.get("after"), "step after")
            before_digest = _mapping(before.get("screenshot"), "screenshot")["png"]
            after_digest = _mapping(after.get("screenshot"), "screenshot")["png"]
            out.append(
                Frame(
                    index=int(record.get("index", len(out))),
                    before=screens / f"{before_digest}.png",
                    after=screens / f"{after_digest}.png",
                )
            )
        return out

    def path_of(self, run_id: str) -> Path:
        """The directory holding ``run_id``."""
        return self._require_dir(run_id)

    # -- internals ---------------------------------------------------------------------

    def _run_dirs(self) -> list[tuple[str, Path]]:
        """Every run directory, oldest first, from ONE ``scandir`` and no open."""
        if not self.root.is_dir():
            return []
        found: list[tuple[str, str, Path]] = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                run_id = run_id_of(entry.name)
                if run_id is not None and entry.is_dir():
                    found.append((entry.name, run_id, Path(entry.path)))
        return [(run_id, path) for _, run_id, path in sorted(found)]

    def _existing_dir(self, run_id: str) -> Path | None:
        if not self.root.is_dir():
            return None
        matches = sorted(self.root.glob(f"*__{run_id}"))
        return matches[0] if matches else None

    def _require_dir(self, run_id: str) -> Path:
        directory = self._existing_dir(run_id)
        if directory is None or not (directory / TRAJECTORY_FILE).is_file():
            raise SkillWeaverError(f"no trajectory with run_id {run_id!r} under {self.root}")
        return directory
