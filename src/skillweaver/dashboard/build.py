"""Build the one-file dashboard: what the agent has learned, as a page you can open.

:func:`build_dashboard` reads a skillweaver data directory and writes a SINGLE HTML
file that opens from the filesystem with no server and no network. Everything is
inlined - the stylesheet, the few lines of script, and every screenshot as a
``data:`` URI - because the thing a judge or a teammate actually does with this is
open it, mail it, or drop it on a USB stick.

What it reads, and what it shows::

    <data>/skills/         FileSkillStore      panel 2: the library growing
    <data>/graphs/         JSONGraphStore      panel 3: the site graph, drawn
    <data>/trajectories/   TrajectoryFileStore panel 4: a run, as a filmstrip
    <data>/eval/*.json     see below           panel 1: cold versus warm

Every panel is independent and every panel is defensive. A missing, empty or
malformed input makes that panel say plainly what is missing; it never raises and
never renders an empty frame. This module is written before the evaluation harness
exists, so it has to behave well against data that is not there yet.

The evaluation metrics format
-----------------------------

The eval harness is a separate piece of work and does not exist yet, so this module
does not import it and writes nothing under ``skillweaver.eval``. Instead it defines
the shape it needs here, as :class:`EvalReport` / :class:`EvalTask` / :class:`EvalRun`,
and reads it out of ``<data>/eval/*.json`` defensively. **The harness author should
write this shape**; deviating is not fatal (unknown keys are ignored and missing
optional keys default), but a file that cannot be understood at all shows the panel's
empty state rather than failing the build.

One JSON object per file, one object per evaluated task, one object per run::

    {
      "schema_version": 1,
      "suite": "sandbox-site",                  optional, free text
      "generated_at": "2026-09-19T12:00:00Z",   optional, ISO-8601
      "tasks": [
        {
          "task_id": "find_invoice",            required, stable across runs
          "task_text": "Find Acme's invoice",   optional, shown as the row title
          "domain": "sandbox.test",             optional
          "runs": [
            {
              "attempt": 1,                     optional; defaults to list position + 1
              "ok": true,                       optional, defaults to false
              "wall_ms": 48120.0,               required for the time chart
              "llm_calls": 23,                  required for the model-call chart
              "steps": 14,                      optional
              "skill_used": null,               optional; null means solved by exploring
              "usd": 0.42,                      optional
              "run_id": "a1b2c3d4",             optional; the Trajectory.run_id
              "started_at": "2026-09-19T11:58:00Z"   optional, ISO-8601
            },
            {"attempt": 2, "ok": true, "wall_ms": 6210.0, "llm_calls": 2,
             "skill_used": "search_invoice"}
          ]
        }
      ]
    }

The run with the lowest ``attempt`` is the COLD run - the first encounter, solved by
trial and error. Every later run is WARM, and the panel charts the cold run against
the mean of the warm ones, in wall-clock milliseconds and in model calls. A task with
only a cold run is listed as awaiting its warm run rather than charted. A top-level
JSON *list* is also accepted and treated as the ``tasks`` array, since that is the
other obvious way to write this file.

A note on self-containment
--------------------------

Nothing in the output may reference the network, so: no external stylesheet, no
external script, no web font, no linked image. Inline SVG here deliberately carries
no ``xmlns`` attribute - the HTML parser puts it in the SVG namespace on its own, and
the attribute's value is an ``http://`` URL that has no business in a file that
claims to need no network. URLs read out of the data (a state's ``url_pattern``, an
observation's ``url``) are displayed with their scheme stripped, as
``sandbox.test/invoices``: shorter to read, and it keeps the page free of anything
that looks like a live link.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    "build_dashboard",
    "read_eval_reports",
]

log = get_logger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

MAX_FILMSTRIP_STEPS = 40
"""How many steps of a run the filmstrip renders before it says it truncated. A page
is a page; forty screenshots is already more than anybody scrolls."""

THUMBNAIL_WIDTH = 520
"""Screenshots are downscaled to at most this many pixels wide before they are
base64-encoded. A filmstrip of full-resolution frames is tens of megabytes."""


# --------------------------------------------------------------------------------------
# The evaluation metrics format (defined here; see the module docstring)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalRun:
    """One measured attempt at one task.

    ``wall_ms`` is wall-clock milliseconds end to end and ``llm_calls`` the number of
    model calls the run made - the two axes the cold-versus-warm panel charts.
    ``attempt`` counts from ``1``; the lowest attempt of a task is its cold run.
    ``skill_used`` names the stored skill that carried the run, or ``None`` when the
    run solved the task by exploring.
    """

    attempt: int
    ok: bool = False
    wall_ms: float = 0.0
    llm_calls: int = 0
    steps: int | None = None
    skill_used: str | None = None
    usd: float | None = None
    run_id: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EvalTask:
    """Every measured run of one task, in the order the harness recorded them."""

    task_id: str
    task_text: str = ""
    domain: str = ""
    runs: tuple[EvalRun, ...] = ()

    @property
    def cold(self) -> EvalRun | None:
        """The first encounter: the run with the lowest ``attempt``, or ``None``."""
        return min(self.runs, key=lambda r: r.attempt, default=None)

    @property
    def warm(self) -> tuple[EvalRun, ...]:
        """Every run after the cold one, in attempt order. Empty when there is none."""
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


# --------------------------------------------------------------------------------------
# Small conversions, all of them total: they answer for any input rather than raising
# --------------------------------------------------------------------------------------

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9_-]+")


def strip_scheme(url: str | None) -> str:
    """``https://sandbox.test/x`` as ``sandbox.test/x``; ``None`` as ``""``.

    Display only. See the module docstring: a dashboard that needs no network should
    not be sprinkled with things that look like live links.
    """
    if not url:
        return ""
    return _SCHEME.sub("", url).rstrip("/") or url


def dom_key(value: str, prefix: str = "k") -> str:
    """A short, stable, HTML-id-safe key for an arbitrary string.

    Fingerprint values are hashes and edge keys are tuples of actions; neither is
    safe to drop into an ``id`` or a CSS selector, and both need to survive the trip
    into the page's embedded JSON unchanged.
    """
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
    """``part`` as a percentage of ``whole``, clamped to ``0..100``; ``0`` when
    ``whole`` is zero. Bar widths go through here so no bar can ever overflow its
    track, whatever the data says."""
    if whole <= 0:
        return 0.0
    return max(0.0, min(100.0, part / whole * 100.0))


def success_rate(successes: int, attempts: int) -> float | None:
    """``successes / attempts`` in ``0.0..1.0``, or ``None`` when never attempted -
    which is a different thing from zero and is shown differently."""
    if attempts <= 0:
        return None
    return max(0.0, min(1.0, successes / attempts))


def data_uri(png: bytes, *, max_width: int = THUMBNAIL_WIDTH) -> str:
    """PNG bytes as an inline ``data:`` URI, downscaled to ``max_width`` if wider.

    Falls back to the original bytes if Pillow cannot read them, so a screenshot the
    decoder dislikes costs file size rather than the whole panel.
    """
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


# --------------------------------------------------------------------------------------
# Reading the data directory. Each reader returns (value, problem): never raises.
# --------------------------------------------------------------------------------------


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
    """Every readable ``<eval_dir>/*.json``, plus a note per file that was not.

    Nothing in here raises. A missing directory yields ``([], [])``; a file that is
    not JSON, is not an object or a list, or holds no recognizable task yields a
    problem string naming the file, and the panel shows its empty state with that
    note attached. This is the whole point: the harness that writes these files has
    not been built yet.
    """
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
    """Every stored skill, latest version first-class, newest-learned first.

    Demoted skills are included - a library that quietly hides its retired skills is
    telling half the story - and marked as such by the caller.
    """
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
    """One ``(domain, states, transitions)`` per stored domain, alphabetically.

    A domain whose file is unreadable is skipped rather than failing the panel.
    """
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


# --------------------------------------------------------------------------------------
# Panel 1: cold versus warm
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeedupRow:
    """One task's first encounter measured against every later one.

    Widths are percentages of the widest bar in the whole panel, so the eye compares
    tasks against each other as well as cold against warm.
    """

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
    """Fold every report's tasks into one chartable panel.

    Warm numbers are the mean over the task's SUCCESSFUL warm runs; a warm run that
    failed took whatever time it took on the way to being wrong and would flatter or
    smear the comparison either way. When no warm run succeeded, every warm run is
    used and the row says so through ``warm_ok``.
    """
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


# --------------------------------------------------------------------------------------
# Panel 2: the skill library
# --------------------------------------------------------------------------------------


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
    """The cumulative count-over-time curve, as points plus an area and a line path.

    Returns empty paths unless the provenance timestamps actually support a curve:
    at least two skills, spanning at least two distinct moments.
    """
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


# --------------------------------------------------------------------------------------
# Panel 3: the site graph, drawn
# --------------------------------------------------------------------------------------

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
    """``text`` shortened with an ellipsis until it fits ``available`` pixels.

    SVG text does not wrap and does not clip to its parent, so a label that is too
    long simply spills out of the node box and over whatever is next to it. Measuring
    properly needs a font engine; a per-font average character width is accurate
    enough to keep text inside a box whose width we chose ourselves.
    """
    limit = max(4, int(available // char_width))
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "\u2026"


def _separate_labels(edges: list[GraphEdge]) -> list[GraphEdge]:
    """Nudge edge labels apart so two of them never print on top of each other.

    Bezier midpoints collide whenever two edges leave the same node at similar
    angles, and two percentages stacked in the same 18 pixels are unreadable - which
    is exactly the number this panel exists to show. Each label keeps its x and moves
    only in y, alternately down then up, to the nearest free slot.
    """
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
    """A cubic bezier as ``(path, mid_x, mid_y)``.

    The midpoint is the curve at ``t=0.5``, which is where an edge label belongs:
    the average of the four endpoints would sit off the curve on anything bowed.
    """
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
    """Assign each node a column by breadth-first distance from the entry states.

    Breadth-first rather than longest-path on purpose: a site graph has cycles in it
    (every "back to the list" edge is one), and longest-path layering does not
    terminate on a cycle without extra machinery. BFS depth is defined for every
    graph, is stable under insertion order because the frontier is seeded in
    ``order``, and puts each screen as close to the entry as it really is - which is
    the thing a viewer is reading off the picture.
    """
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

    # A bezier stays inside the hull of its control points, so sizing the canvas to
    # every point we drew guarantees nothing - a bowed back-edge least of all - is
    # clipped, whatever shape the graph turns out to be.
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
    """The cheapest route from the entry state to every other reachable state.

    Routing is the graph module's own cost model rather than a reimplementation
    here, so what the dashboard highlights is exactly the path the agent would take.
    Anything that goes wrong yields no routes and a graph that simply is not
    clickable - never a failed build.
    """
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


# --------------------------------------------------------------------------------------
# Panel 4: the run filmstrip
# --------------------------------------------------------------------------------------


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
    """The most interesting stored run, as a strip of screenshots.

    "Most interesting" is the newest run that finished and succeeded; failing that,
    the newest that finished; failing that, the newest there is. A viewer opening
    this page wants to watch the agent do the thing, so a successful run wins.

    Pixels and structure are read separately, which is what the trajectory store's
    layout is for: ``load(screenshots=False)`` gives the actions and verdicts without
    touching a PNG, and ``frames`` gives the image paths without decoding them, so
    only the frames this panel actually shows are ever read.
    """
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


# --------------------------------------------------------------------------------------
# The whole page
# --------------------------------------------------------------------------------------


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
    eval_format: str = ""

    @property
    def route_data(self) -> dict[str, Any]:
        return {panel.prefix: panel.route_map for panel in self.graph.panels}


def _headline_stats(
    speedup: SpeedupPanel, skills: SkillPanel, graph: GraphSection, film: FilmPanel
) -> tuple[Stat, ...]:
    time_factor = speedup.time_factor
    call_factor = speedup.call_factor
    return (
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
         "steps": 14, "skill_used": null, "usd": 0.42},
        {"attempt": 2, "ok": true, "wall_ms": 6210.0, "llm_calls": 2,
         "steps": 4, "skill_used": "search_invoice", "usd": 0.03}
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
    skill_panel = build_skill_panel(skills, skills_note, moment)
    graph_section = build_graph_section(graphs, graphs_note)
    film = build_film_panel(data_dir / "trajectories")
    return Dashboard(
        data_dir=str(data_dir),
        built_at=human_date(moment),
        stats=_headline_stats(speedup, skill_panel, graph_section, film),
        speedup=speedup,
        skills=skill_panel,
        graph=graph_section,
        film=film,
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
    env.globals["asset"] = _read_asset
    return env


def _read_asset(name: str) -> str:
    """One file from ``static/``, verbatim, for inlining into the page.

    Returns ``""`` when it is missing: a dashboard with no stylesheet is ugly and
    still readable, which beats not building at all.
    """
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
    """Read ``data_dir`` and write one self-contained HTML file to ``out_path``.

    The output opens straight off the filesystem: no server, no network, no sibling
    files. The stylesheet and script are inlined and every screenshot is a ``data:``
    URI.

    Missing data is never an error. A data directory that does not exist at all still
    produces a valid page in which all four panels explain what is missing, because
    this dashboard is built alongside the pipelines that fill it and has to be
    openable before any of them has run.

    Args:
        data_dir: A skillweaver data directory - the one holding ``skills/``,
            ``graphs/``, ``trajectories/`` and ``eval/``.
        out_path: Where to write the HTML file. Parent directories are created.

    Returns:
        The path written, for convenience.

    Raises:
        SkillWeaverError: only if the output itself cannot be written.
    """
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
