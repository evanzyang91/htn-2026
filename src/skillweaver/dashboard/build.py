"""Build the one-file dashboard: what the agent has learned, as a page you can open.

``build_dashboard`` reads a skillweaver data directory and writes ONE HTML file that
opens from the filesystem with no server and no network - stylesheet, script and every
screenshot as a ``data:`` URI, all inlined. Nothing in the output may reference the
network, which is why inline SVG carries no ``xmlns`` (its value is an ``http://`` URL,
and the HTML parser namespaces the element anyway) and why every URL read out of the
data is shown with its scheme stripped.

Six panels over ``<data>/{skills,graphs,trajectories,eval}``: cold-versus-warm, the
library, the site graph drawn, a run as a filmstrip, what each skill DOES step by step
(``read_skill_steps`` reads the stored source with ``ast`` and correlates it with its
recording through ``Skill.provenance.trajectory_id``), and what the library has SAVED.
Every panel is independent and DEFENSIVE: missing, empty or malformed input makes that
panel say what is missing, never raise and never render an empty frame.

The eval harness is a separate piece of work, so this module imports nothing from it and
defines the shape it reads instead - ``EvalReport`` / ``EvalTask`` / ``EvalRun``, whose
field names are the JSON keys, out of ``<data>/eval/*.json``. Unknown keys are ignored
and optional ones default. The run with the LOWEST ``attempt`` is the cold run and every
later one is warm; a top-level JSON list is accepted as the ``tasks`` array.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillweaver.contracts import (
    Action,
    Fingerprint,
    Skill,
    Transition,
    UIState,
    utcnow,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.logging_ import get_logger

__all__ = [
    "EvalReport",
    "EvalRun",
    "EvalTask",
    "Dashboard",
    "SavingRow",
    "SavingsPanel",
    "SkillRoutine",
    "SkillStep",
    "TrajectoryPanel",
    "build_dashboard",
    "build_savings_panel",
    "build_trajectory_panel",
    "read_eval_reports",
    "read_skill_steps",
]

log = get_logger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

MAX_FILMSTRIP_STEPS = 40
"""Forty screenshots is already more than anybody scrolls; beyond it the strip says it
truncated."""

THUMBNAIL_WIDTH = 520
"""Screenshots downscale to this before base64: a strip of full-resolution frames is tens
of megabytes."""


@dataclass(frozen=True, slots=True)
class EvalRun:
    """One measured attempt at one task. ``attempt`` counts from ``1`` and the lowest is
    the cold run; ``skill_used`` is ``None`` when exploring solved it.

    ``usd``, ``input_tokens`` and ``output_tokens`` are ``None`` when nobody was measuring
    rather than ``0``, and the savings panel must never render an absent measurement as a
    saving of nothing.
    """

    attempt: int
    ok: bool = False
    wall_ms: float = 0.0
    llm_calls: int = 0
    steps: int | None = None
    skill_used: str | None = None
    usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    run_id: str | None = None
    started_at: datetime | None = None

    @property
    def tokens(self) -> int | None:
        """Input plus output, or ``None`` when NEITHER was measured - a subtraction against
        an unknown is not zero. A half-measured run counts the side that is there."""
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True, slots=True)
class EvalTask:
    """Every measured run of one task, in the order the harness recorded them."""

    task_id: str
    task_text: str = ""
    domain: str = ""
    runs: tuple[EvalRun, ...] = ()

    @property
    def cold(self) -> EvalRun | None:
        """The first encounter: the run with the lowest ``attempt``."""
        return min(self.runs, key=lambda r: r.attempt, default=None)

    @property
    def warm(self) -> tuple[EvalRun, ...]:
        """Every run after the cold one, in attempt order."""
        cold = self.cold
        if cold is None:
            return ()
        return tuple(sorted((r for r in self.runs if r is not cold), key=lambda r: r.attempt))


@dataclass(frozen=True, slots=True)
class EvalReport:
    """One ``<data>/eval/*.json`` file: a suite of tasks, each with its runs."""

    tasks: tuple[EvalTask, ...] = ()
    suite: str = ""
    generated_at: datetime | None = None
    source: str = ""


# Conversions, all of them total: they answer for any input rather than raising.
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_ANY_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://")
_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9_-]+")


def strip_scheme(url: str | None) -> str:
    """``https://sandbox.test/x`` as ``sandbox.test/x``. Display only: a page needing no
    network should not be sprinkled with things that look like live links."""
    if not url:
        return ""
    return _SCHEME.sub("", url).rstrip("/") or url


def dom_key(value: str, prefix: str = "k") -> str:
    """A short, stable, HTML-id-safe key: a fingerprint hash or an action tuple is safe in
    neither an ``id`` nor a CSS selector, and both must survive the embedded JSON."""
    safe = _UNSAFE_KEY.sub("", value)[-24:]
    return f"{prefix}{safe or 'x'}"


def _number(raw: Any, default: float = 0.0) -> float:
    """``raw`` as a finite float, or ``default`` for anything else (including NaN)."""
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _integer(raw: Any, default: int = 0) -> int:
    if isinstance(raw, bool):
        return default
    value = _number(raw, float(default))
    return int(value)


def _optional_int(raw: Any) -> int | None:
    """``raw`` as an int, or ``None``. The distinction this exists to keep: a missing or
    unparseable measurement is "not measured", and only a real ``0`` means zero."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return None if not math.isfinite(raw) else int(raw)


def _text(raw: Any, default: str = "") -> str:
    return raw if isinstance(raw, str) else default


def _moment(raw: Any) -> datetime | None:
    """An ISO-8601 string as an aware UTC datetime, or ``None`` if it is not one."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def human_ms(ms: float) -> str:
    """Milliseconds as something a human reads at a glance: ``820ms``, ``6.2s``, ``1m 4s``."""
    if ms <= 0:
        return "0ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def human_date(moment: datetime | None) -> str:
    """``2026-09-19 12:04 UTC``, or ``"unknown"``."""
    if moment is None:
        return "unknown"
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def human_ago(moment: datetime | None, now: datetime) -> str:
    """How long before ``now``: ``just now``, ``4h ago``, ``3d ago``."""
    if moment is None:
        return ""
    delta = (now - moment.astimezone(UTC)).total_seconds()
    if delta < 0:
        return "just now"
    for size, unit in ((86400.0, "d"), (3600.0, "h"), (60.0, "m")):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return "just now"


def percent(part: float, whole: float) -> float:
    """Clamped to ``0..100``, so no bar overflows its track whatever the data says."""
    if whole <= 0:
        return 0.0
    return max(0.0, min(100.0, part / whole * 100.0))


def success_rate(successes: int, attempts: int) -> float | None:
    """``0.0..1.0``, or ``None`` when never attempted - a different thing from zero."""
    if attempts <= 0:
        return None
    return max(0.0, min(1.0, successes / attempts))


def data_uri(png: bytes, *, max_width: int = THUMBNAIL_WIDTH) -> str:
    """PNG bytes as an inline ``data:`` URI, downscaled if wider. Falls back to the original
    bytes when Pillow cannot read them: file size beats losing the panel."""
    if not png:
        return ""
    shrunk = png
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png)) as image:
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                resized = image.convert("RGB").resize((max_width, height), Image.LANCZOS)
                buffer = io.BytesIO()
                resized.save(buffer, format="PNG", optimize=True)
                shrunk = buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - a thumbnail is never worth failing a build
        log.debug("dashboard.thumbnail.skipped", reason=str(exc))
    return "data:image/png;base64," + base64.b64encode(shrunk).decode("ascii")


def action_label(action: Action) -> tuple[str, str]:
    """An action as ``(verb, detail)`` for the filmstrip: ``("type", '"Acme"')``."""
    match action.kind:
        case "click":
            suffix = " x2" if getattr(action, "clicks", 1) > 1 else ""
            button = getattr(action, "button", "left")
            where = f"{action.point.x:.0f}, {action.point.y:.0f}"  # type: ignore[union-attr]
            return f"click{suffix}", f"{button} at {where}"
        case "move":
            return "move", f"to {action.point.x:.0f}, {action.point.y:.0f}"  # type: ignore[union-attr]
        case "drag":
            start, end = action.start, action.end  # type: ignore[union-attr]
            return "drag", f"{start.x:.0f}, {start.y:.0f} to {end.x:.0f}, {end.y:.0f}"
        case "type_text":
            return "type", f'"{action.text}"'  # type: ignore[union-attr]
        case "press_key":
            return "press", " + ".join(action.keys)  # type: ignore[union-attr]
        case "scroll":
            return "scroll", f"dx {action.dx:+d}, dy {action.dy:+d}"  # type: ignore[union-attr]
        case "wait":
            return "wait", human_ms(float(action.ms))  # type: ignore[union-attr]
        case "navigate":
            return "navigate", strip_scheme(action.url)  # type: ignore[union-attr]
        case "back":
            return "back", "to the previous page"
    return str(action.kind), ""  # pragma: no cover - the union above is closed


def actions_summary(actions: Sequence[Action]) -> str:
    """A short one-line rendering of an edge's action sequence."""
    if not actions:
        return "no actions"
    parts = []
    for action in actions[:3]:
        verb, detail = action_label(action)
        parts.append(f"{verb} {detail}".strip())
    if len(actions) > 3:
        parts.append(f"+{len(actions) - 3} more")
    return " -> ".join(parts)


def _eval_run_from(raw: Any, position: int) -> EvalRun | None:
    if not isinstance(raw, Mapping):
        return None
    return EvalRun(
        attempt=_integer(raw.get("attempt"), position + 1) or position + 1,
        ok=bool(raw.get("ok", False)),
        wall_ms=_number(raw.get("wall_ms")),
        llm_calls=_integer(raw.get("llm_calls")),
        steps=_integer(raw["steps"]) if isinstance(raw.get("steps"), int | float) else None,
        skill_used=_text(raw.get("skill_used")) or None,
        usd=_number(raw["usd"]) if isinstance(raw.get("usd"), int | float) else None,
        input_tokens=_optional_int(raw.get("input_tokens")),
        output_tokens=_optional_int(raw.get("output_tokens")),
        run_id=_text(raw.get("run_id")) or None,
        started_at=_moment(raw.get("started_at")),
    )


def _eval_task_from(raw: Any) -> EvalTask | None:
    if not isinstance(raw, Mapping):
        return None
    task_id = _text(raw.get("task_id")) or _text(raw.get("id"))
    if not task_id:
        return None
    raw_runs = raw.get("runs")
    runs = (
        tuple(
            run
            for index, item in enumerate(raw_runs)
            if (run := _eval_run_from(item, index)) is not None
        )
        if isinstance(raw_runs, Sequence) and not isinstance(raw_runs, str | bytes)
        else ()
    )
    return EvalTask(
        task_id=task_id,
        task_text=_text(raw.get("task_text")) or task_id,
        domain=_text(raw.get("domain")),
        runs=runs,
    )


def read_eval_reports(eval_dir: Path) -> tuple[list[EvalReport], list[str]]:
    """Every readable ``<eval_dir>/*.json``, plus a note per file that was not. Never raises:
    a missing directory is ``([], [])``, and an unreadable file becomes a problem string
    the panel's empty state carries."""
    reports: list[EvalReport] = []
    problems: list[str] = []
    if not eval_dir.is_dir():
        return reports, problems
    for path in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{path.name}: not readable JSON ({type(exc).__name__})")
            continue
        if isinstance(data, Mapping):
            raw_tasks = data.get("tasks")
            suite = _text(data.get("suite"))
            generated = _moment(data.get("generated_at"))
        elif isinstance(data, list):
            raw_tasks, suite, generated = data, "", None
        else:
            problems.append(f"{path.name}: expected an object or a list of tasks")
            continue
        if not isinstance(raw_tasks, list):
            problems.append(f"{path.name}: has no 'tasks' list")
            continue
        tasks = tuple(task for item in raw_tasks if (task := _eval_task_from(item)) is not None)
        if not tasks:
            problems.append(f"{path.name}: holds no task with a task_id")
            continue
        reports.append(
            EvalReport(tasks=tasks, suite=suite, generated_at=generated, source=path.name)
        )
    return reports, problems


def read_skills(skills_dir: Path) -> tuple[list[Skill], str | None]:
    """Every stored skill, latest version, newest-learned first. Demoted ones are INCLUDED
    and marked by the caller: a library that hides its retired skills tells half the story."""
    if not skills_dir.is_dir():
        return [], None
    try:
        from skillweaver.skills.store import FileSkillStore

        skills = FileSkillStore(skills_dir).list(include_demoted=True)
    except (SkillWeaverError, OSError, KeyError, TypeError, ValueError) as exc:
        return [], f"the skill library at {skills_dir.name}/ could not be read: {exc}"
    skills.sort(key=lambda s: (s.provenance.created_at, s.domain, s.name), reverse=True)
    return skills, None


GraphTriple = tuple[str, list[UIState], list[Transition]]
"""One stored domain as the dashboard reads it: its name, screens and edges."""


def read_graphs(graphs_dir: Path) -> tuple[list[GraphTriple], str | None]:
    """One ``(domain, states, transitions)`` per stored domain, alphabetically; an unreadable
    one is skipped rather than failing the panel."""
    if not graphs_dir.is_dir():
        return [], None
    try:
        from skillweaver.graph.store import JSONGraphStore

        store = JSONGraphStore(graphs_dir)
        domains = store.domains()
    except (SkillWeaverError, OSError, ValueError) as exc:
        return [], f"the site graphs at {graphs_dir.name}/ could not be listed: {exc}"
    found: list[GraphTriple] = []
    skipped: list[str] = []
    for domain in domains:
        try:
            snapshot = store.load(domain)
        except (SkillWeaverError, OSError, ValueError) as exc:
            skipped.append(f"{domain} ({exc})")
            continue
        found.append((domain, list(snapshot.states), list(snapshot.transitions)))
    problem = f"skipped unreadable domains: {', '.join(skipped)}" if skipped else None
    return found, problem


@dataclass(frozen=True, slots=True)
class SpeedupRow:
    """One task's first encounter against every later one. Widths are percentages of the
    widest bar in the WHOLE panel, so tasks compare against each other too."""

    task_id: str
    task_text: str
    domain: str
    warm_runs: int
    cold_ms: float
    warm_ms: float
    cold_calls: float
    warm_calls: float
    cold_ok: bool
    warm_ok: int
    skill_used: str
    time_factor: float | None
    call_factor: float | None
    cold_ms_width: float = 0.0
    warm_ms_width: float = 0.0
    cold_calls_width: float = 0.0
    warm_calls_width: float = 0.0


@dataclass(frozen=True, slots=True)
class SpeedupPanel:
    """The money panel: cold versus warm, plus whatever could not be charted."""

    rows: tuple[SpeedupRow, ...] = ()
    pending: tuple[EvalTask, ...] = ()
    suites: tuple[str, ...] = ()
    generated_at: datetime | None = None
    empty_reason: str = ""
    notes: tuple[str, ...] = ()
    total_cold_ms: float = 0.0
    total_warm_ms: float = 0.0
    total_cold_calls: float = 0.0
    total_warm_calls: float = 0.0

    @property
    def time_factor(self) -> float | None:
        if self.total_warm_ms <= 0 or self.total_cold_ms <= 0:
            return None
        return self.total_cold_ms / self.total_warm_ms

    @property
    def call_factor(self) -> float | None:
        if self.total_warm_calls <= 0 or self.total_cold_calls <= 0:
            return None
        return self.total_cold_calls / self.total_warm_calls


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def build_speedup_panel(reports: Sequence[EvalReport], problems: Sequence[str]) -> SpeedupPanel:
    """Fold every report's tasks into one chartable panel. Warm numbers are the mean over
    SUCCESSFUL warm runs - a failed one would flatter or smear the comparison either way -
    and when none succeeded every run is used and ``warm_ok`` says so."""
    notes = tuple(problems)
    tasks = [task for report in reports for task in report.tasks]
    if not tasks:
        reason = (
            "No evaluation results yet. Run the eval harness so it writes "
            "<data>/eval/*.json, and this panel fills in."
        )
        return SpeedupPanel(empty_reason=reason, notes=notes)

    rows: list[SpeedupRow] = []
    pending: list[EvalTask] = []
    for task in tasks:
        cold, warm = task.cold, task.warm
        if cold is None or not warm:
            pending.append(task)
            continue
        good = [run for run in warm if run.ok] or list(warm)
        warm_ms = _mean(run.wall_ms for run in good)
        warm_calls = _mean(float(run.llm_calls) for run in good)
        skills = sorted({run.skill_used for run in good if run.skill_used})
        rows.append(
            SpeedupRow(
                task_id=task.task_id,
                task_text=task.task_text or task.task_id,
                domain=task.domain,
                warm_runs=len(warm),
                cold_ms=cold.wall_ms,
                warm_ms=warm_ms,
                cold_calls=float(cold.llm_calls),
                warm_calls=warm_calls,
                cold_ok=cold.ok,
                warm_ok=sum(1 for run in warm if run.ok),
                skill_used=", ".join(skills),
                time_factor=(cold.wall_ms / warm_ms) if warm_ms > 0 and cold.wall_ms > 0 else None,
                call_factor=(
                    (cold.llm_calls / warm_calls) if warm_calls > 0 and cold.llm_calls > 0 else None
                ),
            )
        )

    if not rows:
        reason = (
            f"{len(pending)} task(s) have been run once, but none has a second run yet. "
            "Cold versus warm needs a repeat encounter."
        )
        return SpeedupPanel(pending=tuple(pending), empty_reason=reason, notes=notes)

    rows.sort(key=lambda r: r.time_factor or 0.0, reverse=True)
    widest_ms = max(max(r.cold_ms, r.warm_ms) for r in rows)
    widest_calls = max(max(r.cold_calls, r.warm_calls) for r in rows)
    sized = tuple(dataclass_replace_widths(row, widest_ms, widest_calls) for row in rows)
    suites = tuple(sorted({r.suite for r in reports if r.suite}))
    generated = max((r.generated_at for r in reports if r.generated_at), default=None)
    return SpeedupPanel(
        rows=sized,
        pending=tuple(pending),
        suites=suites,
        generated_at=generated,
        notes=notes,
        total_cold_ms=sum(r.cold_ms for r in sized),
        total_warm_ms=sum(r.warm_ms for r in sized),
        total_cold_calls=sum(r.cold_calls for r in sized),
        total_warm_calls=sum(r.warm_calls for r in sized),
    )


def dataclass_replace_widths(row: SpeedupRow, widest_ms: float, widest_calls: float) -> SpeedupRow:
    """``row`` with its four bar widths filled in against the panel-wide maxima."""
    from dataclasses import replace

    return replace(
        row,
        cold_ms_width=percent(row.cold_ms, widest_ms),
        warm_ms_width=percent(row.warm_ms, widest_ms),
        cold_calls_width=percent(row.cold_calls, widest_calls),
        warm_calls_width=percent(row.warm_calls, widest_calls),
    )


@dataclass(frozen=True, slots=True)
class SkillParam:
    name: str
    type_: str
    description: str


@dataclass(frozen=True, slots=True)
class SkillCard:
    """One stored skill, as the page shows it."""

    name: str
    domain: str
    summary: str
    docstring: str
    params: tuple[SkillParam, ...]
    version: int
    runs: int
    successes: int
    rate: float | None
    rate_percent: float
    mean_ms: float
    learned_at: datetime | None
    learned_ago: str
    model: str
    from_task: str
    trajectory_id: str
    requires: tuple[str, ...]
    has_verifier: bool
    demoted_reason: str
    code: str
    code_lines: int


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    """One step of the cumulative "skills known" curve."""

    at: datetime
    count: int
    x: float
    y: float
    label: str


@dataclass(frozen=True, slots=True)
class SkillPanel:
    cards: tuple[SkillCard, ...] = ()
    timeline: tuple[TimelinePoint, ...] = ()
    timeline_area: str = ""
    timeline_line: str = ""
    timeline_width: float = 720.0
    timeline_height: float = 120.0
    timeline_span: str = ""
    domains: tuple[str, ...] = ()
    live: int = 0
    demoted: int = 0
    total_runs: int = 0
    empty_reason: str = ""
    note: str = ""


def _skill_params(raw: Mapping[str, Any]) -> tuple[SkillParam, ...]:
    """A skill's JSON-schema-ish ``params`` as rows, whatever shape they arrived in."""
    out: list[SkillParam] = []
    for name, spec in raw.items():
        if isinstance(spec, Mapping):
            type_ = _text(spec.get("type"), "any")
            description = _text(spec.get("description"))
        else:
            type_, description = _text(spec, "any"), ""
        out.append(SkillParam(name=str(name), type_=type_ or "any", description=description))
    return tuple(out)


def _timeline(
    cards: Sequence[SkillCard], width: float, height: float
) -> tuple[tuple[TimelinePoint, ...], str, str, str]:
    """The cumulative count-over-time curve. Empty paths unless the timestamps support one:
    at least two skills over at least two distinct moments."""
    moments = sorted(c.learned_at for c in cards if c.learned_at is not None)
    if len(moments) < 2 or moments[0] == moments[-1]:
        return (), "", "", ""
    first, last = moments[0], moments[-1]
    span = (last - first).total_seconds()
    top, bottom, left, right = 12.0, height - 22.0, 6.0, width - 6.0
    total = len(moments)
    points: list[TimelinePoint] = []
    for index, moment in enumerate(moments, start=1):
        fraction = (moment - first).total_seconds() / span
        points.append(
            TimelinePoint(
                at=moment,
                count=index,
                x=left + fraction * (right - left),
                y=bottom - (index / total) * (bottom - top),
                label=f"skill {index} of {total}, {human_date(moment)}",
            )
        )
    steps: list[str] = [f"M {left:.1f},{bottom:.1f}"]
    for point in points:
        steps.append(f"L {point.x:.1f},{bottom - (point.count - 1) / total * (bottom - top):.1f}")
        steps.append(f"L {point.x:.1f},{point.y:.1f}")
    steps.append(f"L {right:.1f},{points[-1].y:.1f}")
    line = " ".join(steps)
    area = f"{line} L {right:.1f},{bottom:.1f} Z"
    label = f"{human_date(first)} to {human_date(last)}"
    return tuple(points), area, line, label


def build_skill_panel(skills: Sequence[Skill], note: str | None, now: datetime) -> SkillPanel:
    """Every stored skill as a card, newest-learned first, plus the growth curve."""
    if not skills:
        reason = (
            "The library is empty. Once the agent solves a task and synthesizes it into "
            "code, every skill it learns shows up here, newest first."
        )
        return SkillPanel(empty_reason=reason, note=note or "")

    cards = tuple(
        SkillCard(
            name=skill.name,
            domain=skill.domain,
            summary=skill.summary,
            docstring=skill.docstring,
            params=_skill_params(skill.params or {}),
            version=skill.version,
            runs=skill.stats.runs,
            successes=skill.stats.successes,
            rate=(rate := success_rate(skill.stats.successes, skill.stats.runs)),
            rate_percent=(rate or 0.0) * 100.0,
            mean_ms=skill.stats.mean_ms,
            learned_at=skill.provenance.created_at,
            learned_ago=human_ago(skill.provenance.created_at, now),
            model=skill.provenance.model,
            from_task=skill.provenance.task_text,
            trajectory_id=skill.provenance.trajectory_id,
            requires=tuple(skill.requires),
            has_verifier=skill.verifier_code is not None,
            demoted_reason=skill.demoted_reason or "",
            code=skill.code,
            code_lines=skill.code.count("\n") + 1 if skill.code else 0,
        )
        for skill in skills
    )
    width, height = 720.0, 120.0
    points, area, line, span = _timeline(cards, width, height)
    return SkillPanel(
        cards=cards,
        timeline=points,
        timeline_area=area,
        timeline_line=line,
        timeline_width=width,
        timeline_height=height,
        timeline_span=span,
        domains=tuple(sorted({c.domain for c in cards if c.domain})),
        live=sum(1 for c in cards if not c.demoted_reason),
        demoted=sum(1 for c in cards if c.demoted_reason),
        total_runs=sum(c.runs for c in cards),
        note=note or "",
    )


NODE_W = 176.0
NODE_H = 58.0
COL_GAP = 108.0
ROW_GAP = 36.0
PAD = 28.0


@dataclass(frozen=True, slots=True)
class GraphNode:
    key: str
    fingerprint: str
    label: str
    sublabel: str
    label_fit: str
    sublabel_fit: str
    text_x: float
    x: float
    y: float
    cx: float
    cy: float
    is_entry: bool
    out_degree: int
    in_degree: int
    first_seen: str
    thumbnail: str = ""


@dataclass(frozen=True, slots=True)
class GraphEdge:
    key: str
    src: str
    dst: str
    path: str
    label: str
    label_x: float
    label_y: float
    label_w: float
    title: str
    rate: float | None
    verified: bool


@dataclass(frozen=True, slots=True)
class GraphRoute:
    """The cheapest known way from the entry state to one other state."""

    target: str
    edges: tuple[str, ...]
    nodes: tuple[str, ...]
    cost_ms: float
    steps: int
    hops: int


@dataclass(frozen=True, slots=True)
class GraphPanel:
    domain: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    routes: tuple[GraphRoute, ...] = ()
    width: float = 0.0
    height: float = 0.0
    view_box: str = ""
    entry: str = ""
    entry_label: str = ""
    verified_edges: int = 0
    prefix: str = "g0"

    @property
    def route_map(self) -> dict[str, Any]:
        """What the page's script needs to light a route up, as plain JSON."""
        return {
            route.target: {
                "edges": list(route.edges),
                "nodes": list(route.nodes),
                "cost": round(route.cost_ms, 1),
                "steps": route.steps,
                "hops": route.hops,
            }
            for route in self.routes
        }


@dataclass(frozen=True, slots=True)
class GraphSection:
    panels: tuple[GraphPanel, ...] = ()
    empty_reason: str = ""
    note: str = ""
    total_states: int = 0
    total_edges: int = 0


LABEL_CHAR = 7.15
"""Approximate width of a character of the 13px semibold node label, in pixels."""
SUB_CHAR = 6.3
"""Approximate width of a character of the 10.5px monospace node sublabel."""


def fit(text: str, available: float, char_width: float) -> str:
    """Ellipsized until it fits: SVG text neither wraps nor clips to its parent, so a long
    label spills over whatever is beside it. A per-font average character width is enough
    for a box whose width we chose ourselves."""
    limit = max(4, int(available // char_width))
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "\u2026"


def _separate_labels(edges: list[GraphEdge]) -> list[GraphEdge]:
    """Nudge edge labels apart: bezier midpoints collide whenever two edges leave a node at
    similar angles, and stacked percentages are exactly the number this panel exists to
    show. Each keeps its x and moves in y alone, alternately down then up."""
    from dataclasses import replace

    placed: list[tuple[float, float, float]] = []  # x centre, half width, y centre
    out: list[GraphEdge] = []
    for edge in sorted(edges, key=lambda e: (e.label_y, e.label_x)):
        half = edge.label_w / 2.0 + 3.0
        y = edge.label_y
        for attempt in range(12):
            if not any(
                abs(y - other_y) < 20.0 and abs(edge.label_x - other_x) < half + other_half
                for other_x, other_half, other_y in placed
            ):
                break
            shift = 21.0 * ((attempt // 2) + 1)
            y = edge.label_y + (shift if attempt % 2 == 0 else -shift)
        placed.append((edge.label_x, half, y))
        out.append(replace(edge, label_y=y))
    return sorted(out, key=lambda e: e.key)


def _cubic(
    p0: tuple[float, float],
    c1: tuple[float, float],
    c2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[str, float, float]:
    """A cubic bezier as ``(path, mid_x, mid_y)``, the midpoint taken at ``t=0.5``: the
    average of the four endpoints sits OFF the curve on anything bowed."""
    path = (
        f"M {p0[0]:.1f},{p0[1]:.1f} C {c1[0]:.1f},{c1[1]:.1f} "
        f"{c2[0]:.1f},{c2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"
    )
    mid_x = (p0[0] + 3 * c1[0] + 3 * c2[0] + p3[0]) / 8.0
    mid_y = (p0[1] + 3 * c1[1] + 3 * c2[1] + p3[1]) / 8.0
    return path, mid_x, mid_y


def _layer_nodes(
    ids: Sequence[str], edges: Sequence[tuple[str, str]], order: Mapping[str, int]
) -> dict[str, int]:
    """A column per node, by breadth-first distance from the entry states.

    BFS and not longest-path: a site graph has cycles (every "back to the list" edge), and
    longest-path layering does not terminate on one. BFS depth is defined for every graph,
    stable under insertion order because the frontier is seeded in ``order``, and puts each
    screen as close to the entry as it really is."""
    outgoing: dict[str, list[str]] = {node: [] for node in ids}
    indegree = dict.fromkeys(ids, 0)
    for src, dst in edges:
        if src == dst:
            continue
        outgoing[src].append(dst)
        indegree[dst] += 1
    layer: dict[str, int] = {}
    roots = sorted((n for n in ids if indegree[n] == 0), key=lambda n: order[n])
    queue: deque[str] = deque()
    for root in roots:
        layer[root] = 0
        queue.append(root)
    remaining = deque(sorted(ids, key=lambda n: order[n]))
    while len(layer) < len(ids):
        while queue:
            node = queue.popleft()
            for neighbor in sorted(outgoing[node], key=lambda n: order[n]):
                if neighbor not in layer:
                    layer[neighbor] = layer[node] + 1
                    queue.append(neighbor)
        while remaining and remaining[0] in layer:
            remaining.popleft()
        if remaining:  # a component with no root of its own, e.g. a pure cycle
            seed = remaining.popleft()
            layer[seed] = 0
            queue.append(seed)
    return layer


def _build_graph_panel(
    domain: str, states: Sequence[UIState], transitions: Sequence[Transition], index: int
) -> GraphPanel:
    """Lay one domain out left to right and hand the template finished geometry."""
    known = {state.fingerprint.value: state for state in states}
    ids: list[str] = list(known)
    for edge in transitions:  # an edge may name a screen the snapshot has no state for
        for endpoint in (edge.src.value, edge.dst.value):
            if endpoint not in known and endpoint not in ids:
                ids.append(endpoint)
    order = {node: position for position, node in enumerate(ids)}

    layer = _layer_nodes(ids, [(e.src.value, e.dst.value) for e in transitions], order)
    columns: dict[int, list[str]] = {}
    for node in sorted(ids, key=lambda n: (layer[n], order[n])):
        columns.setdefault(layer[node], []).append(node)

    rows = max((len(column) for column in columns.values()), default=1)

    out_degree: dict[str, int] = dict.fromkeys(ids, 0)
    in_degree: dict[str, int] = dict.fromkeys(ids, 0)
    for edge in transitions:
        out_degree[edge.src.value] += 1
        in_degree[edge.dst.value] += 1

    entry = columns.get(0, ids[:1])[0] if ids else ""
    placed: dict[str, tuple[float, float]] = {}
    nodes: list[GraphNode] = []
    for column_index in sorted(columns):
        column = columns[column_index]
        x = PAD + column_index * (NODE_W + COL_GAP)
        block = len(column) * NODE_H + max(0, len(column) - 1) * ROW_GAP
        top = PAD + 24.0 + (rows * NODE_H + max(0, rows - 1) * ROW_GAP - block) / 2.0
        for row_index, node in enumerate(column):
            y = top + row_index * (NODE_H + ROW_GAP)
            placed[node] = (x, y)
            state = known.get(node)
            label = (state.label if state and state.label else "") or f"state {node[:8]}"
            sublabel = strip_scheme(state.url_pattern if state else None) or node[:16]
            thumbnail = data_uri(state.thumbnail, max_width=96) if state and state.thumbnail else ""
            text_x = x + (54.0 if thumbnail else 14.0)
            room = NODE_W - (text_x - x) - 12.0
            nodes.append(
                GraphNode(
                    key=dom_key(node, f"n{index}_"),
                    fingerprint=node,
                    label=label,
                    sublabel=sublabel,
                    label_fit=fit(label, room, LABEL_CHAR),
                    sublabel_fit=fit(sublabel, room, SUB_CHAR),
                    text_x=text_x,
                    x=x,
                    y=y,
                    cx=x + NODE_W / 2.0,
                    cy=y + NODE_H / 2.0,
                    is_entry=node == entry,
                    out_degree=out_degree[node],
                    in_degree=in_degree[node],
                    first_seen=human_date(state.first_seen) if state else "unknown",
                    thumbnail=thumbnail,
                )
            )

    key_of = {node.fingerprint: node.key for node in nodes}
    extents: list[tuple[float, float]] = [
        (corner_x, corner_y)
        for node in nodes
        for corner_x, corner_y in ((node.x, node.y), (node.x + NODE_W, node.y + NODE_H))
    ]
    edges: list[GraphEdge] = []
    edge_keys: dict[tuple[str, str, tuple[Action, ...]], str] = {}
    parallel: dict[tuple[str, str], int] = {}
    for position, edge in enumerate(transitions):
        src, dst = edge.src.value, edge.dst.value
        if src not in placed or dst not in placed:  # pragma: no cover - ids covers both
            continue
        twin = parallel.get((src, dst), 0)
        parallel[(src, dst)] = twin + 1
        (sx, sy), (dx, dy) = placed[src], placed[dst]
        scy, dcy = sy + NODE_H / 2.0, dy + NODE_H / 2.0
        bow = 26.0 + twin * 20.0
        if src == dst:
            corners = (
                (sx + NODE_W * 0.32, sy),
                (sx + NODE_W * 0.10, sy - bow - 22.0),
                (sx + NODE_W * 0.90, sy - bow - 22.0),
                (sx + NODE_W * 0.68, sy),
            )
        elif layer[dst] > layer[src]:
            reach = max(48.0, (dx - (sx + NODE_W)) * 0.55)
            corners = ((sx + NODE_W, scy), (sx + NODE_W + reach, scy), (dx - reach, dcy), (dx, dcy))
        elif layer[dst] < layer[src]:
            drop = bow + NODE_H * 0.9
            corners = (
                (sx, scy),
                (sx - 64.0, scy + drop),
                (dx + NODE_W + 64.0, dcy + drop),
                (dx + NODE_W, dcy),
            )
        else:
            side = sx + NODE_W
            corners = ((side, scy), (side + bow + 56.0, scy), (side + bow + 56.0, dcy), (side, dcy))
        path, mx, my = _cubic(*corners)
        extents.extend(corners)
        rate = success_rate(edge.successes, edge.attempts)
        label = "untried" if rate is None else f"{rate * 100:.0f}%"
        key = f"e{index}_{position}"
        edge_keys[(src, dst, tuple(edge.actions))] = key
        edges.append(
            GraphEdge(
                key=key,
                src=key_of[src],
                dst=key_of[dst],
                path=path,
                label=label,
                label_x=mx,
                label_y=my,
                label_w=max(30.0, 7.4 * len(label) + 10.0),
                title=(
                    f"{actions_summary(edge.actions)} - {edge.successes}/{edge.attempts} ok"
                    f", {human_ms(edge.mean_ms)} mean"
                ),
                rate=rate,
                verified=edge.successes > 0,
            )
        )

    edges = _separate_labels(edges)
    extents.extend(
        (corner_x, corner_y)
        for e in edges
        for corner_x, corner_y in (
            (e.label_x - e.label_w / 2, e.label_y - 10),
            (e.label_x + e.label_w / 2, e.label_y + 10),
        )
    )

    # A bezier stays inside the hull of its control points, so sizing to every point
    # drawn guarantees nothing is clipped - a bowed back-edge least of all.
    margin = 12.0
    min_x = min((x for x, _ in extents), default=0.0) - margin
    min_y = min((y for _, y in extents), default=0.0) - margin
    width = max((x for x, _ in extents), default=NODE_W) + margin - min_x
    height = max((y for _, y in extents), default=NODE_H) + margin - min_y

    routes = _build_routes(domain, states, transitions, entry, key_of, edge_keys)
    entry_label = next((n.label for n in nodes if n.fingerprint == entry), "")
    return GraphPanel(
        domain=domain,
        nodes=tuple(nodes),
        edges=tuple(edges),
        routes=routes,
        width=width,
        height=height,
        view_box=f"{min_x:.1f} {min_y:.1f} {width:.1f} {height:.1f}",
        entry=key_of.get(entry, ""),
        entry_label=entry_label,
        verified_edges=sum(1 for e in edges if e.verified),
        prefix=f"g{index}",
    )


def _build_routes(
    domain: str,
    states: Sequence[UIState],
    transitions: Sequence[Transition],
    entry: str,
    key_of: Mapping[str, str],
    edge_keys: Mapping[tuple[str, str, tuple[Action, ...]], str],
) -> tuple[GraphRoute, ...]:
    """The cheapest route from the entry state to every reachable one, through the graph
    module's OWN cost model, so the dashboard highlights the path the agent would take.
    Anything going wrong leaves the graph unclickable rather than failing the build."""
    if not entry:
        return ()
    try:
        from skillweaver.graph.model import GraphSnapshot, InMemorySiteGraph

        graph = InMemorySiteGraph()
        graph.absorb(
            GraphSnapshot(domain=domain, states=tuple(states), transitions=tuple(transitions))
        )
        source = Fingerprint(entry)
        routes: list[GraphRoute] = []
        for target in key_of:
            if target == entry:
                continue
            route = graph.route(source, Fingerprint(target))
            if route is None or not route.edges:
                continue
            keys = [
                edge_keys[(e.src.value, e.dst.value, tuple(e.actions))]
                for e in route.edges
                if (e.src.value, e.dst.value, tuple(e.actions)) in edge_keys
            ]
            if len(keys) != len(route.edges):  # pragma: no cover - the maps are built together
                continue
            visited = [key_of[entry]] + [key_of[e.dst.value] for e in route.edges]
            routes.append(
                GraphRoute(
                    target=key_of[target],
                    edges=tuple(keys),
                    nodes=tuple(dict.fromkeys(visited)),
                    cost_ms=route.cost,
                    steps=len(route.steps),
                    hops=len(route.edges),
                )
            )
        return tuple(routes)
    except Exception as exc:  # noqa: BLE001 - a picture without routes still informs
        log.warning("dashboard.routes.skipped", domain=domain, reason=str(exc))
        return ()


def build_graph_section(graphs: Sequence[GraphTriple], note: str | None) -> GraphSection:
    """One drawn panel per stored domain, or the empty state."""
    usable = [(d, s, t) for d, s, t in graphs if s or t]
    if not usable:
        reason = (
            "No site graph yet. As the agent recognizes screens and learns which "
            "actions move between them, each site's map is drawn here."
        )
        return GraphSection(empty_reason=reason, note=note or "")
    panels = tuple(
        _build_graph_panel(domain, states, transitions, index)
        for index, (domain, states, transitions) in enumerate(usable)
    )
    return GraphSection(
        panels=panels,
        note=note or "",
        total_states=sum(len(p.nodes) for p in panels),
        total_edges=sum(len(p.edges) for p in panels),
    )


@dataclass(frozen=True, slots=True)
class FilmFrame:
    """One step of a run: the screen it produced, what was done, how it was judged."""

    index: int
    image: str
    verb: str
    detail: str
    ok: bool
    error: str
    elapsed: str
    verdict: str
    verdict_ok: bool
    verdict_reason: str
    verdict_source: str
    note: str
    url: str


@dataclass(frozen=True, slots=True)
class FilmPanel:
    run_id: str = ""
    task: str = ""
    domain: str = ""
    ok: bool = False
    complete: bool = True
    started_at: str = ""
    duration: str = ""
    frames: tuple[FilmFrame, ...] = ()
    step_count: int = 0
    shown: int = 0
    truncated: bool = False
    other_runs: int = 0
    empty_reason: str = ""
    note: str = ""


def build_film_panel(trajectories_dir: Path) -> FilmPanel:
    """The most interesting stored run, as a strip of screenshots: the newest that finished
    and SUCCEEDED, else the newest that finished, else the newest there is.

    Pixels and structure are read separately - ``load(screenshots=False)`` for actions and
    verdicts, ``frames`` for paths - so only the frames shown are ever decoded."""
    if not trajectories_dir.is_dir():
        reason = (
            "No runs recorded yet. Once the agent runs a task, its screenshots, "
            "actions and verdicts play back here step by step."
        )
        return FilmPanel(empty_reason=reason)
    try:
        from skillweaver.trajectory.store import TrajectoryFileStore

        store = TrajectoryFileStore(trajectories_dir)
        summaries = store.summaries()
    except (SkillWeaverError, OSError, ValueError) as exc:
        return FilmPanel(empty_reason=f"the recorded runs could not be listed: {exc}")
    if not summaries:
        reason = (
            "No runs recorded yet. Once the agent runs a task, its screenshots, "
            "actions and verdicts play back here step by step."
        )
        return FilmPanel(empty_reason=reason)

    chosen = (
        next((s for s in reversed(summaries) if s.complete and s.ok), None)
        or next((s for s in reversed(summaries) if s.complete), None)
        or summaries[-1]
    )
    try:
        trajectory = store.load(chosen.run_id, screenshots=False)
        frame_paths = {frame.index: frame.after for frame in store.frames(chosen.run_id)}
    except (SkillWeaverError, OSError, ValueError) as exc:
        return FilmPanel(
            empty_reason=f"run {chosen.run_id} could not be replayed: {exc}",
            other_runs=max(0, len(summaries) - 1),
        )

    steps = trajectory.steps[:MAX_FILMSTRIP_STEPS]
    frames: list[FilmFrame] = []
    for step in steps:
        verb, detail = action_label(step.action)
        path = frame_paths.get(step.index)
        try:
            png = path.read_bytes() if path is not None and path.is_file() else b""
        except OSError:
            png = b""
        verdict = step.verdict
        frames.append(
            FilmFrame(
                index=step.index,
                image=data_uri(png),
                verb=verb,
                detail=detail,
                ok=step.result.ok,
                error=step.result.error or "",
                elapsed=human_ms(step.result.elapsed_ms),
                verdict=("pass" if verdict.ok else "fail") if verdict else "not judged",
                verdict_ok=bool(verdict and verdict.ok),
                verdict_reason=verdict.reason if verdict else "",
                verdict_source=verdict.source if verdict else "",
                note=step.note,
                url=strip_scheme(step.after.url),
            )
        )
    duration = (trajectory.finished_at - trajectory.started_at).total_seconds() * 1000.0
    return FilmPanel(
        run_id=trajectory.run_id,
        task=trajectory.task,
        domain=trajectory.domain,
        ok=trajectory.ok,
        complete=chosen.complete,
        started_at=human_date(trajectory.started_at),
        duration=human_ms(duration),
        frames=tuple(frames),
        step_count=len(trajectory.steps),
        shown=len(frames),
        truncated=len(trajectory.steps) > len(frames),
        other_runs=max(0, len(summaries) - 1),
        note=trajectory.note,
    )


MAX_SKILL_STEPS = 48
"""How many steps of one skill the panel lists before it says it truncated. A stored
skill is a page of code; a skill that needs more than this to describe is telling you
something on its own, which the truncation note says out loud."""

SKILL_THUMB_WIDTH = 240
"""Width of the recorded screen shown beside a step. Narrower than the filmstrip's,
because these appear once per step of every skill rather than once per run."""

_CTL_VERBS: Mapping[str, str] = {
    "click": "click",
    "double_click": "double-click",
    "move": "move to",
    "drag": "drag",
    "type_text": "type",
    "type": "type",
    "press": "press",
    "press_key": "press",
    "scroll": "scroll",
    "wait": "wait",
    "perform": "perform",
}
"""``ctx.ctl`` methods as the verb a reader wants, including the two spellings
(``press``/``press_key``) that stored code has actually been written with."""

_QUOTE = '"'
"""Written once so :func:`describe_lookup` can unquote a kind without an escape."""

_LOOKUPS = frozenset({"find", "find_text", "best", "by_kind", "nearest", "containing", "all"})
"""Method names that mean "look at the screen" - ``ctx.see``'s index, plus the
``ctx.find`` shorthand some stored code uses."""

_STEP_SHOT_VERBS = frozenset(
    {"click", "double-click", "type", "press", "scroll", "wait", "move to", "drag", "navigate"}
)
"""Verbs that a recorded trajectory step can be matched against. A skill's lookups
and checks have no counterpart in a recording - nothing was performed - so they are
never given a screen."""


@dataclass(frozen=True, slots=True)
class SkillStep:
    """One line of a stored skill, in the words someone deciding whether to trust it uses.

    ``kind`` groups the step for the eye: ``act``, ``look``, ``check`` (where the skill
    refuses to continue), ``call``, ``flow``, ``result``, ``note``. ``anchor`` is the text
    or kind the step aims at rather than a coordinate; ``brittle`` marks the opposite, a
    fixed pixel, which is the one thing in a skill that does not survive a redesign."""

    order: int
    depth: int
    kind: str
    verb: str
    detail: str = ""
    anchor: str = ""
    why: str = ""
    line: int = 0
    brittle: bool = False
    shot: str = ""
    shot_caption: str = ""


@dataclass(frozen=True, slots=True)
class SkillRoutine:
    """One stored skill as a procedure: its steps, its checks, and where it came from."""

    name: str
    domain: str
    summary: str
    version: int
    demoted_reason: str
    params: tuple[str, ...]
    precondition: str
    requires: tuple[str, ...]
    steps: tuple[SkillStep, ...]
    checks: tuple[SkillStep, ...]
    steps_problem: str = ""
    checks_problem: str = ""
    truncated: bool = False
    run_id: str = ""
    run_task: str = ""
    run_ok: bool = False
    run_when: str = ""
    run_steps: int = 0
    matched: int = 0
    run_note: str = ""


@dataclass(frozen=True, slots=True)
class TrajectoryPanel:
    """Every stored skill, laid out as the procedure it runs."""

    routines: tuple[SkillRoutine, ...] = ()
    empty_reason: str = ""
    note: str = ""
    with_screens: int = 0
    with_checks: int = 0
    total_steps: int = 0


def _short(text: str, limit: int = 72) -> str:
    """``text`` on one line, at most ``limit`` characters, ellipsis when cut."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: max(1, limit - 1)].rstrip() + "…"


def _quoted(text: str, limit: int = 44) -> str:
    """A string literal as the page should print it; URL-looking ones lose their scheme."""
    shown = strip_scheme(text) if _SCHEME.match(text) else text
    return '"' + _short(shown, limit) + '"'


def _dotted(node: ast.expr) -> tuple[str, ...]:
    """``ctx.see.find_text`` as ``("ctx", "see", "find_text")``; ``()`` if not dotted."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return ()


def _literal(node: ast.expr) -> str | None:
    """The value of a string constant, else ``None``."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _phrase(node: ast.expr, params: Iterable[str] = ()) -> str:
    """One argument as the words a reader wants: a literal quoted, an f-string keeping its
    fixed parts, a parameter named as one, anything else its own source. Skill code searches
    for text it was HANDED as often as text it hard-codes, and reading ``text "row"`` off
    ``find_text(company, "row")`` would lie about what it anchors on."""
    if (text := _literal(node)) is not None:
        return _quoted(text)
    if isinstance(node, ast.JoinedStr):
        parts = "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "…"
            for value in node.values
        )
        return _quoted(parts, 56)
    if isinstance(node, ast.Name):
        return f"the {node.id} parameter" if node.id in frozenset(params) else node.id
    return _unparse(node)


def _strings(nodes: Iterable[ast.expr]) -> list[str]:
    """Every string literal among ``nodes``, tuples and lists flattened one level."""
    out: list[str] = []
    for node in nodes:
        if (text := _literal(node)) is not None:
            out.append(text)
        elif isinstance(node, ast.Tuple | ast.List):
            out.extend(text for item in node.elts if (text := _literal(item)) is not None)
    return out


def _unparse(node: ast.AST) -> str:
    """``ast.unparse`` on one line, never raising: code this reader has no phrasing for is
    still described, in its own words."""
    try:
        return _short(ast.unparse(node))
    except Exception:  # noqa: BLE001 - unparse can reject a hand-built or exotic node
        return type(node).__name__


def is_lookup(call: ast.Call) -> bool:
    """Whether ``call`` asks the screen a question rather than changing it."""
    path = _dotted(call.func)
    if not path:
        return False
    return path[:2] == ("ctx", "see") or (path[0] == "ctx" and path[-1] in _LOOKUPS)


def describe_lookup(call: ast.Call, params: Iterable[str] = ()) -> str:
    """What a ``ctx.see`` query anchors on: ``text "Reply" (button)``, ``any row`` - the
    sentence this panel exists to print, since a skill that searches for the word on the
    button survives a redesign and one that does not says so here."""
    path = _dotted(call.func)
    method = path[-1] if path else ""
    first = _phrase(call.args[0], params) if call.args else ""
    kinds = _strings(call.args[1:])
    match method:
        case "find_text" | "find":
            if not first:
                return "text on screen"
            anchor = f"text {first}" if first.startswith(_QUOTE) else f"text from {first}"
            return anchor + (f" ({kinds[0]})" if kinds else "")
        case "best":
            return f"the best match for {first}" if first else "the best match"
        case "by_kind":
            return f"any {first.strip(_QUOTE)}" if first else "an element of one kind"
        case "nearest":
            return f"the nearest element to {first or 'a point'}"
        case "containing":
            return f"whatever is under {first or 'a point'}"
        case "all":
            return "every element on screen"
    return first or _unparse(call)


def describe_target(
    node: ast.expr, binds: Mapping[str, str], params: Iterable[str] = ()
) -> tuple[str, bool]:
    """``(anchor, brittle)``: what an action aims at, and whether that is a fixed pixel.
    ``binds`` carries earlier lookups' anchors, so ``click(reply_button[0])`` is described
    by what ``reply_button`` SEARCHED for rather than by its name."""
    current = node
    while True:
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        if isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Name | ast.Subscript | ast.Attribute | ast.Call):
                current = current.value
                continue
            break
        break
    if isinstance(current, ast.Name):
        return binds.get(current.id, f"whatever {current.id} holds"), False
    if isinstance(current, ast.Call):
        if is_lookup(current):
            return describe_lookup(current, params), False
        name = _dotted(current.func)
        if name and name[-1] in {"Point", "Box"}:
            return f"the fixed point {_unparse(current)}", True
        return _unparse(current), False
    if isinstance(current, ast.Tuple | ast.Constant):
        return f"the fixed point {_unparse(current)}", True
    return _unparse(current), False


class _StepReader:
    """One ``SkillStep`` per statement of a skill function. Deliberately TOTAL: a statement
    with no phrasing is rendered in its own source, because unusual code is exactly when
    somebody is reading the page."""

    def __init__(self, params: Iterable[str] = ()) -> None:
        self.steps: list[SkillStep] = []
        self.binds: dict[str, str] = {}
        self.params = frozenset(params)
        self.truncated = False

    def add(
        self,
        node: ast.AST,
        depth: int,
        kind: str,
        verb: str,
        detail: str = "",
        anchor: str = "",
        why: str = "",
        brittle: bool = False,
    ) -> None:
        if len(self.steps) >= MAX_SKILL_STEPS:
            self.truncated = True
            return
        self.steps.append(
            SkillStep(
                order=len(self.steps) + 1,
                depth=depth,
                kind=kind,
                verb=verb,
                detail=_ANY_SCHEME.sub("", detail),
                anchor=_ANY_SCHEME.sub("", anchor),
                why=_ANY_SCHEME.sub("", why),
                line=getattr(node, "lineno", 0),
                brittle=brittle,
            )
        )

    def body(self, statements: Sequence[ast.stmt], depth: int) -> None:
        for statement in statements:
            self.statement(statement, depth)

    def statement(self, node: ast.stmt, depth: int) -> None:
        match node:
            case ast.Expr(value=ast.Constant(value=str())):
                return  # the function's own docstring; the card already shows it
            case ast.Expr(value=ast.Call() as call):
                self.call(call, depth)
            case ast.Assign(targets=[ast.Name(id=name)], value=ast.Call() as call):
                self.assignment(name, call, depth)
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                self.add(node, depth, "note", "note", f"{name} = {_unparse(value)}")
            case ast.Return(value=value):
                self.returned(node, value, depth)
            case ast.If(test=test, body=body, orelse=orelse):
                self.add(node, depth, "flow", "if", _unparse(test))
                self.body(body, depth + 1)
                if orelse:
                    self.add(node, depth, "flow", "otherwise")
                    self.body(orelse, depth + 1)
            case ast.For(target=target, iter=source, body=body):
                detail = f"{_unparse(target)} in {_unparse(source)}"
                self.add(node, depth, "flow", "for each", detail)
                self.body(body, depth + 1)
            case ast.While(test=test, body=body):
                self.add(node, depth, "flow", "while", _unparse(test))
                self.body(body, depth + 1)
            case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=final):
                self.add(node, depth, "flow", "try")
                self.body(body, depth + 1)
                for handler in handlers:
                    caught = _unparse(handler.type) if handler.type is not None else "anything"
                    self.add(handler, depth, "flow", "if that raises", caught)
                    self.body(handler.body, depth + 1)
                self.body(orelse, depth)
                if final:
                    self.add(node, depth, "flow", "either way")
                    self.body(final, depth + 1)
            case ast.With(items=items, body=body):
                held = ", ".join(_unparse(item.context_expr) for item in items)
                self.add(node, depth, "flow", "with", _short(held))
                self.body(body, depth + 1)
            case ast.Assert(test=test, msg=msg):
                why = _phrase(msg, self.params).strip(_QUOTE) if msg is not None else ""
                self.add(node, depth, "check", "check", _unparse(test), why=why)
            case ast.Raise(exc=exc):
                detail = _unparse(exc) if exc is not None else "the error it is handling"
                self.add(node, depth, "check", "give up with", detail)
            case ast.Pass():
                return
            case ast.FunctionDef(name=name, body=body):
                self.add(node, depth, "note", "defines", f"a helper called {name}")
                self.body(body, depth + 1)
            case _:
                self.add(node, depth, "note", "code", _unparse(node))

    def returned(self, node: ast.stmt, value: ast.expr | None, depth: int) -> None:
        """``return <expr>`` - and, for a VERIFIER, the check it really is: that line is where
        the skill decides whether the work landed, not a value handed back."""
        if value is None:
            self.add(node, depth, "result", "hand back", "nothing")
            return
        inner = value
        if isinstance(inner, ast.Call) and _dotted(inner.func)[-1:] == ("bool",) and inner.args:
            inner = inner.args[0]
        anchor, _ = describe_target(inner, self.binds, self.params)
        if isinstance(inner, ast.Call) and is_lookup(inner):
            self.add(node, depth, "check", "confirm", "it is on screen", anchor=anchor)
            return
        if isinstance(inner, ast.Name | ast.Subscript) and anchor in self.binds.values():
            self.add(node, depth, "check", "confirm", "it was found", anchor=anchor)
            return
        self.add(node, depth, "result", "hand back", _unparse(value))

    def assignment(self, name: str, call: ast.Call, depth: int) -> None:
        """``name = <call>``: a remembered lookup, or any other call whose value is kept."""
        if is_lookup(call):
            anchor = describe_lookup(call, self.params)
            self.binds[name] = anchor
            self.add(call, depth, "look", "look for", f"and remember it as {name}", anchor=anchor)
            return
        self.call(call, depth, keeps=name)

    def call(self, call: ast.Call, depth: int, keeps: str = "") -> None:
        path = _dotted(call.func)
        tail = path[-1] if path else ""
        kept = f" (kept as {keeps})" if keeps else ""

        if path == ("ctx", "expect"):
            self.expectation(call, depth)
            return
        if path == ("ctx", "log"):
            said = _phrase(call.args[0], self.params) if call.args else "a trace line"
            self.add(call, depth, "note", "log", said)
            return
        if path == ("ctx", "call"):
            named = _strings(call.args[:1])
            args = ", ".join(kw.arg or "**kwargs" for kw in call.keywords)
            detail = f"{named[0] if named else _unparse(call.func)}({args})"
            self.add(call, depth, "call", "run the skill", detail + kept)
            return
        if path[:2] == ("ctx", "graph"):
            self.add(call, depth, "note", "ask the map", f"{tail}{kept}")
            return
        if is_lookup(call):
            anchor = describe_lookup(call, self.params)
            self.add(call, depth, "look", "look for", kept.strip(), anchor=anchor)
            return
        if tail in {"settle", "wait_for_idle"}:
            self.add(call, depth, "act", "wait", "for the screen to settle")
            return
        if path[:2] == ("ctx", "ctl") and tail in _CTL_VERBS:
            self.action(call, depth, _CTL_VERBS[tail])
            return
        self.add(call, depth, "note", "code", _unparse(call) + kept)

    def expectation(self, call: ast.Call, depth: int) -> None:
        """``ctx.expect(condition, why)`` - the point at which the skill refuses to go on."""
        condition = call.args[0] if call.args else None
        why = _phrase(call.args[1], self.params).strip(_QUOTE) if len(call.args) > 1 else ""
        if condition is None:
            self.add(call, depth, "check", "check", "something about the screen", why=why)
            return
        inner = condition
        if isinstance(inner, ast.Call) and _dotted(inner.func)[-1:] == ("bool",) and inner.args:
            inner = inner.args[0]
        anchor, _ = describe_target(inner, self.binds, self.params)
        # A bare truthiness test says nothing the anchor does not, and repeating
        # ``bool(search_field)`` at a reader who came here to AVOID reading code.
        simple = isinstance(inner, ast.Name | ast.Subscript) or (
            isinstance(inner, ast.Attribute) and bool(_dotted(inner))
        )
        detail = "it is on screen" if simple else _unparse(condition)
        self.add(call, depth, "check", "check", detail, anchor=anchor, why=why)

    def action(self, call: ast.Call, depth: int, verb: str) -> None:
        """One ``ctx.ctl`` call: what it does, and what it aims at."""
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        detail, anchor, brittle = "", "", False

        if verb in {"click", "double-click", "move to"}:
            if call.args:
                anchor, brittle = describe_target(call.args[0], self.binds, self.params)
            clicks = keywords.get("clicks")
            button = _literal(keywords["button"]) if "button" in keywords else None
            if isinstance(clicks, ast.Constant) and clicks.value not in (None, 1):
                detail = f"{clicks.value} times"
            if button and button != "left":
                detail = f"{detail}, {button} button".strip(", ")
        elif verb == "type":
            detail = self.value_of(call.args[0]) if call.args else "the text it was given"
        elif verb == "press":
            keys = _strings(call.args) or _strings(keywords.values())
            detail = " + ".join(keys) if keys else _unparse(call)
        elif verb == "scroll":
            if call.args:
                anchor, brittle = describe_target(call.args[0], self.binds, self.params)
            moves = [
                f"{axis} {_unparse(keywords[axis])}" for axis in ("dx", "dy") if axis in keywords
            ]
            detail = ", ".join(moves) or ", ".join(_unparse(a) for a in call.args[1:])
        elif verb == "wait":
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, int | float):
                detail = human_ms(float(first.value))
            elif first is not None:
                detail = _unparse(first)
        elif verb == "drag":
            parts = [describe_target(a, self.binds, self.params) for a in call.args[:2]]
            anchor = " to ".join(text for text, _ in parts)
            brittle = any(flag for _, flag in parts)
        else:
            detail = _unparse(call.args[0]) if call.args else ""

        self.add(call, depth, "act", verb, detail, anchor=anchor, brittle=brittle)

    def value_of(self, node: ast.expr) -> str:
        """What is being typed, in the reader's terms: a literal, or a parameter."""
        if isinstance(node, ast.Name) and node.id not in self.params:
            return f"whatever {node.id} holds"
        return _phrase(node, self.params)


def read_skill_steps(
    code: str, *, function: str = "run", params: Iterable[str] = ()
) -> tuple[tuple[SkillStep, ...], bool, str]:
    """``(steps, truncated, problem)`` for one stored function; never raises. ``problem`` is
    a sentence for the page when the code does not parse or defines no such function - an
    unreadable skill still appears, saying so, because that is the one to see."""
    if not code.strip():
        return (), False, f"this skill stores no {function}() to describe"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return (), False, f"the stored code does not parse: {exc}"
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function
        ),
        None,
    )
    if target is None:
        return (), False, f"the stored code defines no {function}() function"
    declared = [arg.arg for arg in target.args.args + target.args.kwonlyargs]
    reader = _StepReader(params=[*params, *declared[1:]])
    reader.body(target.body, 0)
    if not reader.steps:
        return (), False, f"{function}() has an empty body: it performs nothing"
    return tuple(reader.steps), reader.truncated, ""


def _recorded_steps(store: Any, run_id: str) -> tuple[list[tuple[str, str, str, Path | None]], str]:
    """``(verb, detail, url, screenshot path)`` per step of one recorded run; structure and
    pixels read separately, so only matched frames are decoded."""
    try:
        trajectory = store.load(run_id, screenshots=False)
        paths = {frame.index: frame.after for frame in store.frames(run_id)}
    except (SkillWeaverError, OSError, ValueError) as exc:
        return [], f"the run it was learned from ({run_id}) could not be replayed: {exc}"
    out: list[tuple[str, str, str, Path | None]] = []
    for step in trajectory.steps:
        verb, detail = action_label(step.action)
        out.append((verb, detail, strip_scheme(step.after.url), paths.get(step.index)))
    return out, ""


def _match_shots(
    steps: Sequence[SkillStep], recorded: Sequence[tuple[str, str, str, Path | None]]
) -> tuple[tuple[SkillStep, ...], int]:
    """Give each performed step its recorded screen, matched by verb and scanning forward.

    Positions do NOT line up - the recording holds steps the skill does not (the navigation
    prologue the hardening pass lifts out, and whatever exploring cost) - but the ORDER of
    the actions that survived into the code does. A verb that never comes up again gets no
    screen, since a wrong picture is worse than none."""
    from dataclasses import replace

    cursor, matched = 0, 0
    out: list[SkillStep] = []
    for step in steps:
        if step.verb not in _STEP_SHOT_VERBS:
            out.append(step)
            continue
        found = next(
            (i for i in range(cursor, len(recorded)) if recorded[i][0] == step.verb),
            None,
        )
        if found is None:
            out.append(step)
            continue
        verb, detail, url, path = recorded[found]
        cursor = found + 1
        try:
            png = path.read_bytes() if path is not None and path.is_file() else b""
        except OSError:
            png = b""
        if not png:
            out.append(step)
            continue
        matched += 1
        caption = f"recorded: {verb} {detail}".strip()
        out.append(
            replace(
                step,
                shot=data_uri(png, max_width=SKILL_THUMB_WIDTH),
                shot_caption=f"{caption} - {url}" if url else caption,
            )
        )
    return tuple(out), matched


def build_trajectory_panel(
    skills: Sequence[Skill], trajectories_dir: Path, note: str | None = None
) -> TrajectoryPanel:
    """Every stored skill as the procedure it runs, with the real screens where they exist.
    The defensive rule applied PER SKILL: one whose recording is gone still gets its steps
    from its own code plus a line saying so, rather than costing the reader the rest."""
    if not skills:
        reason = (
            "Nothing to lay out yet. As soon as the agent writes its first skill, every "
            "step it takes is listed here in order, beside the screens it was learned on."
        )
        return TrajectoryPanel(empty_reason=reason, note=note or "")

    store: Any = None
    listed: set[str] = set()
    store_note = ""
    if trajectories_dir.is_dir():
        try:
            from skillweaver.trajectory.store import TrajectoryFileStore

            store = TrajectoryFileStore(trajectories_dir)
            listed = set(store.list())
        except (SkillWeaverError, OSError, ValueError) as exc:
            store, store_note = None, f"the recorded runs could not be listed: {exc}"

    routines: list[SkillRoutine] = []
    for skill in skills:
        params = tuple((skill.params or {}).keys())
        steps, truncated, steps_problem = read_skill_steps(skill.code, params=params)
        checks, _, checks_problem = read_skill_steps(
            skill.verifier_code or "", function="verify", params=params
        )
        run_id = skill.provenance.trajectory_id
        recorded: list[tuple[str, str, str, Path | None]] = []
        run_note = ""
        summary = None
        if not run_id:
            run_note = "this skill records no trajectory it was written from"
        elif store is None:
            run_note = store_note or "no recorded runs are stored, so there are no screens"
        elif run_id not in listed:
            run_note = f"the run it was learned from ({run_id}) is no longer stored"
        else:
            recorded, run_note = _recorded_steps(store, run_id)
            if not run_note:
                try:
                    summary = store.summary(run_id)
                except (SkillWeaverError, OSError, ValueError):
                    summary = None
        shown, matched = _match_shots(steps, recorded)
        if recorded and not matched and not run_note:
            run_note = (
                f"the recording of {run_id} has {len(recorded)} step(s), none of which "
                "lines up with a step of the stored code"
            )
        routines.append(
            SkillRoutine(
                name=skill.name,
                domain=skill.domain,
                summary=skill.summary or skill.docstring,
                version=skill.version,
                demoted_reason=skill.demoted_reason or "",
                params=params,
                precondition=(strip_scheme(skill.precondition.value) if skill.precondition else ""),
                requires=tuple(skill.requires),
                steps=shown,
                checks=checks,
                steps_problem=steps_problem,
                checks_problem=checks_problem,
                truncated=truncated,
                run_id=run_id,
                run_task=summary.task if summary else "",
                run_ok=bool(summary and summary.ok),
                run_when=human_date(summary.started_at) if summary else "",
                run_steps=len(recorded),
                matched=matched,
                run_note=run_note,
            )
        )
    return TrajectoryPanel(
        routines=tuple(routines),
        note=note or "",
        with_screens=sum(1 for r in routines if r.matched),
        with_checks=sum(1 for r in routines if r.checks),
        total_steps=sum(len(r.steps) for r in routines),
    )


def human_usd(usd: float | None) -> str:
    """``$0.0042``, ``$0.78``, ``$12.40``; ``None`` is ``"not priced"`` and never ``"$0.00"``
    - an unmetered run is a different claim from a free one."""
    if usd is None:
        return "not priced"
    magnitude = abs(usd)
    sign = "-" if usd < 0 else ""
    if magnitude < 0.01:
        return f"{sign}${magnitude:.4f}"
    if magnitude < 100:
        return f"{sign}${magnitude:.2f}"
    return f"{sign}${magnitude:,.0f}"


def human_count(value: float | None) -> str:
    """``840``, ``12.3k``, ``1.4M``; ``None`` is ``"not measured"``, since rendering an
    unmetered run as zero would claim a saving of nothing."""
    if value is None:
        return "not measured"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude < 1000:
        return f"{sign}{magnitude:.0f}"
    if magnitude < 1_000_000:
        return f"{sign}{magnitude / 1000:.1f}k"
    return f"{sign}{magnitude / 1_000_000:.2f}M"


@dataclass(frozen=True, slots=True)
class SavingRow:
    """One task's repeat runs, priced against the one attempt that had no library.

    ``baseline_*`` is the COLD run. ``saved_*`` totals every warm attempt: one that did the
    task is credited the difference, one that FAILED is charged its own cost with no credit,
    since the work still had to be done afterwards. ``*_runs`` counts how many attempts that
    measure existed for, so a partial measurement cannot pass as a complete one."""

    task_id: str
    task_text: str
    domain: str
    skills: str
    repeats: int
    repeats_ok: int
    baseline_calls: float
    saved_calls: float
    baseline_usd: float | None = None
    saved_usd: float | None = None
    usd_runs: int = 0
    baseline_tokens: int | None = None
    saved_tokens: int | None = None
    token_runs: int = 0
    cumulative_usd: float = 0.0
    cumulative_tokens: float = 0.0
    saved_width: float = 0.0
    spent_width: float = 0.0
    measured: bool = True


@dataclass(frozen=True, slots=True)
class ExcludedTask:
    """A task deliberately left out of the arithmetic, and why. The panel prints EVERY one:
    a saving computed over a silently chosen subset is not a measurement."""

    task_id: str
    task_text: str
    reason: str


@dataclass(frozen=True, slots=True)
class SavingPoint:
    """One step of the cumulative saving curve: the total after this task's repeats."""

    task_id: str
    value: float
    x: float
    y: float
    label: str


@dataclass(frozen=True, slots=True)
class SavingsPanel:
    """The library's worth: the cold attempt as the baseline, the repeats as the bill."""

    rows: tuple[SavingRow, ...] = ()
    excluded: tuple[ExcludedTask, ...] = ()
    suites: tuple[str, ...] = ()
    generated_at: datetime | None = None
    empty_reason: str = ""
    notes: tuple[str, ...] = ()
    repeats: int = 0
    baseline_calls: float = 0.0
    saved_calls: float = 0.0
    baseline_usd: float | None = None
    saved_usd: float | None = None
    usd_runs: int = 0
    baseline_tokens: int | None = None
    saved_tokens: int | None = None
    token_runs: int = 0
    chart_metric: str = ""
    chart_area: str = ""
    chart_line: str = ""
    chart_points: tuple[SavingPoint, ...] = ()
    chart_width: float = 720.0
    chart_height: float = 130.0

    @property
    def headline(self) -> str:
        """The one figure the panel is for: dollars when anything was priced, else tokens."""
        if self.saved_usd is not None:
            return human_usd(self.saved_usd)
        if self.saved_tokens is not None:
            return human_count(self.saved_tokens)
        return human_count(self.saved_calls)

    @property
    def headline_unit(self) -> str:
        if self.saved_usd is not None:
            return "saved"
        if self.saved_tokens is not None:
            return "tokens saved"
        return "model calls saved"

    @property
    def token_gap(self) -> int:
        """Repeat runs whose tokens nobody counted. Printed, never zero-filled."""
        return max(0, self.repeats - self.token_runs)

    @property
    def usd_gap(self) -> int:
        """Repeat runs whose dollars nobody counted."""
        return max(0, self.repeats - self.usd_runs)


def _credit(baseline: float, run: EvalRun, measure: float) -> float:
    """One warm attempt against ``baseline``: a run that did the task saves the difference,
    one that FAILED subtracts its own cost, since the expensive path was still ahead."""
    return baseline - measure if run.ok else -measure


def _cumulative_curve(
    values: Sequence[float], labels: Sequence[str], width: float, height: float
) -> tuple[tuple[SavingPoint, ...], str, str]:
    """The running total, indexed by TASK and not by clock: the harness writes tasks in the
    order it ran them, most runs carry no ``started_at``, and a curve against invented
    timestamps would be a prettier lie."""
    if len(values) < 2:
        return (), "", ""
    top, bottom, left, right = 14.0, height - 20.0, 6.0, width - 6.0
    peak = max(values) or 1.0
    floor = min(0.0, min(values))
    # A twelfth of headroom: without it a curve that rises once and stays flat - a suite
    # where only some runs were priced - draws as a block against the top edge, reading
    # as "off the chart" rather than "this is the total".
    span = ((peak - floor) * 1.12) or 1.0
    points: list[SavingPoint] = []
    for index, (value, label) in enumerate(zip(values, labels, strict=True)):
        fraction = index / (len(values) - 1)
        points.append(
            SavingPoint(
                task_id=label,
                value=value,
                x=left + fraction * (right - left),
                y=bottom - ((value - floor) / span) * (bottom - top),
                label=label,
            )
        )
    base = bottom - ((0.0 - floor) / span) * (bottom - top)
    line = "M " + " L ".join(f"{p.x:.1f},{p.y:.1f}" for p in points)
    area = f"{line} L {points[-1].x:.1f},{base:.1f} L {points[0].x:.1f},{base:.1f} Z"
    return tuple(points), area, line


def build_savings_panel(reports: Sequence[EvalReport], problems: Sequence[str]) -> SavingsPanel:
    """Price every repeat run against the first attempt at the same task. The rules, all of
    them also printed ON the page because this is the number a stranger will poke at:

    * The baseline is the COLD run and only if it SUCCEEDED; otherwise excluded and named.
    * A task with no repeat run yet is excluded and named.
    * Dollars and tokens count only warm attempts where BOTH sides were measured.
    * A failed warm attempt subtracts its own cost instead of earning a credit."""
    notes = tuple(problems)
    tasks = [task for report in reports for task in report.tasks]
    if not tasks:
        reason = (
            "No evaluation results yet, so there is nothing to price. Run the eval "
            "harness so it writes <data>/eval/*.json and this panel fills in."
        )
        return SavingsPanel(empty_reason=reason, notes=notes)

    rows: list[SavingRow] = []
    excluded: list[ExcludedTask] = []
    running_usd = 0.0
    running_tokens = 0.0
    for task in tasks:
        cold, warm = task.cold, task.warm
        if cold is None:
            excluded.append(ExcludedTask(task.task_id, task.task_text, "no run was recorded"))
            continue
        if not cold.ok:
            excluded.append(
                ExcludedTask(
                    task.task_id,
                    task.task_text,
                    "the first attempt never succeeded, so there is no baseline to save against",
                )
            )
            continue
        if not warm:
            excluded.append(
                ExcludedTask(task.task_id, task.task_text, "no repeat run yet, so nothing to price")
            )
            continue

        saved_calls = sum(_credit(float(cold.llm_calls), run, float(run.llm_calls)) for run in warm)
        priced = [run for run in warm if run.usd is not None] if cold.usd is not None else []
        counted = [run for run in warm if run.tokens is not None] if cold.tokens is not None else []
        saved_usd = (
            sum(_credit(cold.usd or 0.0, run, run.usd or 0.0) for run in priced) if priced else None
        )
        saved_tokens = (
            sum(_credit(float(cold.tokens or 0), run, float(run.tokens or 0)) for run in counted)
            if counted
            else None
        )
        running_usd += saved_usd or 0.0
        running_tokens += saved_tokens or 0.0
        skills = sorted({run.skill_used for run in warm if run.skill_used})
        rows.append(
            SavingRow(
                task_id=task.task_id,
                task_text=task.task_text or task.task_id,
                domain=task.domain,
                skills=", ".join(skills),
                repeats=len(warm),
                repeats_ok=sum(1 for run in warm if run.ok),
                baseline_calls=float(cold.llm_calls),
                saved_calls=saved_calls,
                baseline_usd=cold.usd,
                saved_usd=saved_usd,
                usd_runs=len(priced),
                baseline_tokens=cold.tokens,
                saved_tokens=None if saved_tokens is None else int(saved_tokens),
                token_runs=len(counted),
                cumulative_usd=running_usd,
                cumulative_tokens=running_tokens,
            )
        )

    if not rows:
        reason = (
            f"Nothing can be priced yet: {len(excluded)} task(s) are on record but none has "
            "both a first attempt that succeeded and a repeat run to compare against it."
        )
        return SavingsPanel(empty_reason=reason, excluded=tuple(excluded), notes=notes)

    usd_runs = sum(r.usd_runs for r in rows)
    token_runs = sum(r.token_runs for r in rows)
    # ONE unit for every bar, the one the headline quotes: bars from whichever measure a
    # row happened to have would put $0.78 and nine model calls on one axis.
    metric, values = (
        ("dollars", [r.cumulative_usd for r in rows])
        if usd_runs
        else (
            ("tokens", [r.cumulative_tokens for r in rows])
            if token_runs
            else ("model calls", list(_running(r.saved_calls for r in rows)))
        )
    )
    width, height = 720.0, 130.0
    points, area, line = _cumulative_curve(values, [r.task_id for r in rows], width, height)
    measures = [_measure_of(row, metric) for row in rows]
    widest = max((abs(value) for value in measures if value is not None), default=0.0)
    sized = tuple(
        _size_saving(row, measure, widest) for row, measure in zip(rows, measures, strict=True)
    )
    return SavingsPanel(
        rows=sized,
        excluded=tuple(excluded),
        suites=tuple(sorted({r.suite for r in reports if r.suite})),
        generated_at=max((r.generated_at for r in reports if r.generated_at), default=None),
        notes=notes,
        repeats=sum(r.repeats for r in sized),
        baseline_calls=sum(r.baseline_calls for r in sized),
        saved_calls=sum(r.saved_calls for r in sized),
        baseline_usd=sum(r.baseline_usd or 0.0 for r in sized) if usd_runs else None,
        saved_usd=sum(r.saved_usd or 0.0 for r in sized) if usd_runs else None,
        usd_runs=usd_runs,
        baseline_tokens=sum(r.baseline_tokens or 0 for r in sized) if token_runs else None,
        saved_tokens=sum(r.saved_tokens or 0 for r in sized) if token_runs else None,
        token_runs=token_runs,
        chart_metric=metric,
        chart_area=area,
        chart_line=line,
        chart_points=points,
        chart_width=width,
        chart_height=height,
    )


def _running(values: Iterable[float]) -> Iterator[float]:
    """The running total of ``values``."""
    total = 0.0
    for value in values:
        total += value
        yield total


def _measure_of(row: SavingRow, metric: str) -> float | None:
    """``row``'s saving in the panel's chosen unit, or ``None`` when it has none."""
    if metric == "dollars":
        return row.saved_usd
    if metric == "tokens":
        return None if row.saved_tokens is None else float(row.saved_tokens)
    return row.saved_calls


def _size_saving(row: SavingRow, measure: float | None, widest: float) -> SavingRow:
    """``row`` with its bar width against the panel-wide maximum. One bar, pointing the way
    the number does, because a saving that is actually a loss must LOOK like one; a row with
    no measurement gets NO bar rather than an empty track reading as a saving of nothing."""
    from dataclasses import replace

    if measure is None:
        return replace(row, measured=False)
    width = percent(abs(measure), widest)
    return replace(
        row,
        saved_width=width if measure >= 0 else 0.0,
        spent_width=0.0 if measure >= 0 else width,
    )


@dataclass(frozen=True, slots=True)
class Stat:
    """One headline tile: the four numbers that carry the ten-second read."""

    value: str
    unit: str
    label: str
    detail: str
    tone: str = "plain"


@dataclass(frozen=True, slots=True)
class Dashboard:
    """Everything the template needs; assembled by :func:`collect`, then rendered."""

    data_dir: str
    built_at: str
    stats: tuple[Stat, ...]
    speedup: SpeedupPanel
    skills: SkillPanel
    graph: GraphSection
    film: FilmPanel
    trajectories: TrajectoryPanel = field(default_factory=TrajectoryPanel)
    savings: SavingsPanel = field(default_factory=SavingsPanel)
    eval_format: str = ""

    @property
    def route_data(self) -> dict[str, Any]:
        return {panel.prefix: panel.route_map for panel in self.graph.panels}


def _headline_stats(
    speedup: SpeedupPanel,
    skills: SkillPanel,
    graph: GraphSection,
    film: FilmPanel,
    savings: SavingsPanel,
) -> tuple[Stat, ...]:
    time_factor = speedup.time_factor
    call_factor = speedup.call_factor
    return (
        Stat(
            value=savings.headline if savings.rows else "--",
            unit="",
            label=f"{savings.headline_unit} by the library" if savings.rows else "saved so far",
            detail=(
                f"{savings.repeats} repeat run(s) of {len(savings.rows)} task(s) against "
                f"what the first attempt cost"
                + (f", {len(savings.excluded)} task(s) excluded" if savings.excluded else "")
                if savings.rows
                else "nothing priced yet"
            ),
            tone="good" if savings.rows else "muted",
        ),
        Stat(
            value=f"{time_factor:.1f}" if time_factor else "--",
            unit="x" if time_factor else "",
            label="faster once learned",
            detail=(
                f"{human_ms(speedup.total_cold_ms)} cold vs {human_ms(speedup.total_warm_ms)} warm"
                f" over {len(speedup.rows)} task(s)"
                if time_factor
                else "waiting on a second run of a task"
            ),
            tone="good" if time_factor else "muted",
        ),
        Stat(
            value=f"{call_factor:.1f}" if call_factor else "--",
            unit="x" if call_factor else "",
            label="fewer model calls",
            detail=(
                f"{speedup.total_cold_calls:.0f} cold vs {speedup.total_warm_calls:.1f} warm"
                if call_factor
                else "waiting on a second run of a task"
            ),
            tone="good" if call_factor else "muted",
        ),
        Stat(
            value=str(skills.live) if skills.cards else "0",
            unit="",
            label="skills in the library",
            detail=(
                f"{skills.total_runs} recorded run(s)"
                + (f", {skills.demoted} demoted" if skills.demoted else "")
                if skills.cards
                else "nothing learned yet"
            ),
            tone="accent" if skills.live else "muted",
        ),
        Stat(
            value=str(graph.total_states) if graph.panels else "0",
            unit="",
            label="screens mapped",
            detail=(
                f"{graph.total_edges} known action(s) between them"
                if graph.panels
                else "no site graph yet"
            ),
            tone="accent" if graph.total_states else "muted",
        ),
    )


EVAL_FORMAT_EXAMPLE = """{
  "schema_version": 1,
  "suite": "sandbox-site",
  "generated_at": "2026-09-19T12:00:00Z",
  "tasks": [
    {
      "task_id": "find_invoice",
      "task_text": "Find the invoice for Acme",
      "domain": "sandbox.test",
      "runs": [
        {"attempt": 1, "ok": true, "wall_ms": 48120.0, "llm_calls": 23,
         "steps": 14, "skill_used": null, "usd": 0.42,
         "input_tokens": 19200, "output_tokens": 2480},
        {"attempt": 2, "ok": true, "wall_ms": 6210.0, "llm_calls": 2,
         "steps": 4, "skill_used": "search_invoice", "usd": 0.03,
         "input_tokens": 2400, "output_tokens": 310}
      ]
    }
  ]
}"""


def collect(data_dir: Path, *, now: datetime | None = None) -> Dashboard:
    """Read every input and assemble the view model. Never raises for bad data."""
    moment = now or utcnow()
    reports, problems = read_eval_reports(data_dir / "eval")
    skills, skills_note = read_skills(data_dir / "skills")
    graphs, graphs_note = read_graphs(data_dir / "graphs")

    speedup = build_speedup_panel(reports, problems)
    savings = build_savings_panel(reports, problems)
    skill_panel = build_skill_panel(skills, skills_note, moment)
    graph_section = build_graph_section(graphs, graphs_note)
    film = build_film_panel(data_dir / "trajectories")
    routines = build_trajectory_panel(skills, data_dir / "trajectories", skills_note)
    return Dashboard(
        data_dir=str(data_dir),
        built_at=human_date(moment),
        stats=_headline_stats(speedup, skill_panel, graph_section, film, savings),
        speedup=speedup,
        skills=skill_panel,
        graph=graph_section,
        film=film,
        trajectories=routines,
        savings=savings,
        eval_format=EVAL_FORMAT_EXAMPLE,
    )


def _environment() -> Any:
    """A Jinja environment over ``templates/``, autoescaping, with the page's filters."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(default_for_string=True, default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["ms"] = human_ms
    env.filters["ago"] = lambda moment: human_ago(moment, utcnow())
    env.filters["date"] = human_date
    env.filters["usd"] = human_usd
    env.filters["count"] = human_count
    env.globals["asset"] = _read_asset
    return env


def _read_asset(name: str) -> str:
    """One file from ``static/``, verbatim, or ``""`` when missing: an ugly dashboard beats
    not building at all."""
    path = STATIC_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("dashboard.asset.missing", asset=name, reason=str(exc))
        return ""


def render(dashboard: Dashboard) -> str:
    """The finished page as one string of HTML."""
    template = _environment().get_template("dashboard.html.j2")
    return template.render(d=dashboard, routes=dashboard.route_data)


def build_dashboard(data_dir: Path | str, out_path: Path | str) -> Path:
    """Read ``data_dir`` and write one self-contained HTML file to ``out_path``, returning
    that path. Raises only if the output itself cannot be written.

    Missing data is never an error: a data directory that does not exist still produces a
    valid page whose panels explain what is missing, because this is built alongside the
    pipelines that fill it and has to be openable before any of them has run."""
    source = Path(data_dir)
    destination = Path(out_path)
    page = render(collect(source))
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page, encoding="utf-8")
    except OSError as exc:
        raise SkillWeaverError(f"cannot write the dashboard to {destination}: {exc}") from exc
    log.info(
        "dashboard.built",
        data_dir=str(source),
        out=str(destination),
        bytes=len(page.encode("utf-8")),
    )
    return destination
