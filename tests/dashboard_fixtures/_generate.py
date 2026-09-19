"""Regenerate the dashboard's committed fixture data directories.

Run it from the repository root::

    uv run python tests/dashboard_fixtures/_generate.py

The generated skill sources are written in the repository's own formatting, because
they land as ``.py`` files under ``tests/`` and ``make lint`` reads them like any
other file here.

The fixtures are written through the REAL stores - ``FileSkillStore``,
``JSONGraphStore``, ``TrajectoryFileStore`` - so they cannot drift away from the
formats the dashboard reads. Hand-writing this JSON would only prove that the
dashboard agrees with whatever the fixture author guessed.

Three corpora:

``full/``      one of everything: six skills across two versions of the library's
               life, a five-screen site graph with a deliberately flaky shortcut,
               a four-step recorded run, and a metrics file in the shape
               ``skillweaver.dashboard.build`` documents.
``broken/``    a metrics file that is valid JSON and nothing else, so the
               cold-versus-warm panel must fall back to its empty state.
``cold_only/`` every task run exactly once: the panel has data but no comparison
               to draw yet, which is a different empty state from having no file.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from skillweaver.contracts import (  # noqa: E402
    ActionResult,
    Box,
    Click,
    Element,
    ElementKind,
    ElementSource,
    Fingerprint,
    Navigate,
    Observation,
    Point,
    PressKey,
    Provenance,
    Screenshot,
    Skill,
    Trajectory,
    TrajectoryStep,
    TypeText,
    UIState,
    Verdict,
)
from skillweaver.graph.model import GraphSnapshot  # noqa: E402
from skillweaver.graph.store import JSONGraphStore  # noqa: E402
from skillweaver.skills.store import FileSkillStore  # noqa: E402
from skillweaver.trajectory.store import TrajectoryFileStore  # noqa: E402

DOMAIN = "sandbox.test"
DAY = datetime(2026, 9, 19, tzinfo=UTC)


def at(days: float = 0.0, hours: float = 0.0, minutes: float = 0.0) -> datetime:
    return DAY + timedelta(days=days, hours=hours, minutes=minutes)


# ------------------------------------------------------------------ screenshots


def screen(title: str, accent: tuple[int, int, int], rows: int, highlight: int = -1) -> bytes:
    """A tiny, recognizable fake of the sandbox site: header, sidebar, list rows.

    Small on purpose - a fixture screenshot is committed to the repository, and the
    dashboard's job is to show it, not to be a faithful browser.
    """
    from PIL import Image, ImageDraw

    width, height = 360, 225
    image = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 34], fill=accent)
    draw.text((12, 12), title, fill=(255, 255, 255))
    draw.rectangle([0, 34, 78, height], fill=(233, 237, 243))
    for index, label in enumerate(("Mail", "Records", "Settings")):
        draw.text((12, 48 + index * 20), label, fill=(80, 92, 110))
    draw.rectangle([90, 46, width - 14, 68], fill=(255, 255, 255), outline=(205, 213, 224))
    draw.text((98, 52), "Search", fill=(150, 160, 175))
    for index in range(rows):
        top = 80 + index * 26
        fill = (255, 246, 230) if index == highlight else (255, 255, 255)
        draw.rectangle([90, top, width - 14, top + 22], fill=fill, outline=(219, 225, 234))
        draw.text((98, top + 5), f"INV-10{index + 1}   Acme Corp", fill=(52, 64, 82))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def shot(png: bytes, when: datetime) -> Screenshot:
    return Screenshot(png=png, width=360, height=225, scale=1.0, captured_at=when)


def observe(png: bytes, fingerprint: str, url: str, when: datetime) -> Observation:
    elements = (
        Element(
            box=Box(90, 46, 256, 22),
            kind=ElementKind.text_field,
            text="Search",
            confidence=0.94,
            stable_id="search",
            source=ElementSource.merged,
        ),
        Element(
            box=Box(90, 80, 256, 22),
            kind=ElementKind.row,
            text="INV-101 Acme Corp",
            confidence=0.88,
            stable_id="row-1",
            source=ElementSource.yolo,
        ),
    )
    return Observation(
        screenshot=shot(png, when),
        elements=elements,
        index=_Index(elements),
        fingerprint=Fingerprint(fingerprint, {"url": fingerprint, "layout": "l1"}),
        url=url,
        taken_at=when,
    )


class _Index:
    """The smallest thing that satisfies ``ElementIndex`` for a fixture writer.

    The store only serializes ``Observation.elements``; the index is rebuilt on load.
    """

    def __init__(self, elements: tuple[Element, ...]) -> None:
        self._elements = list(elements)

    def all(self) -> list[Element]:
        return list(self._elements)

    def by_kind(self, kind: ElementKind) -> list[Element]:
        return [e for e in self._elements if e.kind == kind]

    def find_text(self, *args: object, **kwargs: object) -> list[Element]:
        return []

    def nearest(self, *args: object, **kwargs: object) -> list[Element]:
        return []

    def containing(self, *args: object, **kwargs: object) -> list[Element]:
        return []

    def best(self, *args: object, **kwargs: object) -> list[Element]:
        return []


# ---------------------------------------------------------------------- skills


def skill(
    name: str,
    summary: str,
    docstring: str,
    params: dict[str, object],
    code: str,
    created: datetime,
    *,
    requires: tuple[str, ...] = (),
    verifier: str | None = None,
) -> Skill:
    return Skill(
        name=name,
        domain=DOMAIN,
        summary=summary,
        docstring=docstring,
        params=params,
        code=code,
        requires=requires,
        precondition=None,
        verifier_code=verifier,
        provenance=Provenance(
            trajectory_id=f"run-{name}",
            task_text=summary,
            model="claude-opus-5",
            created_at=created,
        ),
    )


SKILLS = [
    (
        skill(
            "open_records",
            "Open the records list from anywhere in the console.",
            "Clicks Records in the left navigation and waits for the table to settle. "
            "Ends on the records list with no filter applied.",
            {},
            "def run(ctx):\n"
            '    ctx.ctl.click(ctx.find("Records nav item").box.center)\n'
            "    ctx.settle()\n",
            at(days=-2, hours=9, minutes=12),
        ),
        (12, 12, 640.0),
        None,
    ),
    (
        skill(
            "search_records",
            "Search the records table for a company.",
            "Focuses the search field, types the company name and submits. Ends on the "
            "filtered results for that company.",
            {"company": {"type": "string", "description": "Company name to filter by"}},
            "def run(ctx, company):\n"
            '    ctx.ctl.click(ctx.find("search field").box.center)\n'
            "    ctx.ctl.type_text(company)\n"
            '    ctx.ctl.press_key(("Enter",))\n'
            "    ctx.settle()\n",
            at(days=-2, hours=15, minutes=40),
        ),
        (9, 8, 980.0),
        None,
    ),
    (
        skill(
            "open_record_detail",
            "Open one record from a filtered list.",
            "Clicks the first row of the current results and waits for the detail pane. "
            "Expects a filtered list, so it is usually called after search_records.",
            {"row": {"type": "integer", "description": "1-based row to open (default 1)"}},
            "def run(ctx, row=1):\n"
            '    rows = ctx.index.by_kind("row")\n'
            "    ctx.ctl.click(rows[row - 1].box.center)\n"
            "    ctx.settle()\n",
            at(days=-1, hours=11, minutes=5),
            requires=("search_records",),
        ),
        (7, 7, 520.0),
        None,
    ),
    (
        skill(
            "send_reply",
            "Reply to the currently open message.",
            "Opens the reply composer, types the body and sends it. Ends back on the "
            "message with the reply appended to the thread.",
            {"body": {"type": "string", "description": "Text of the reply"}},
            "def run(ctx, body):\n"
            '    ctx.ctl.click(ctx.find("Reply button").box.center)\n'
            "    ctx.ctl.type_text(body)\n"
            '    ctx.ctl.press_key(("Meta", "Enter"))\n'
            "    ctx.settle()\n",
            at(days=-1, hours=20, minutes=30),
            verifier=('def verify(ctx, result):\n    return ctx.index.find_text("Sent") != []\n'),
        ),
        (4, 2, 1450.0),
        None,
    ),
    (
        skill(
            "export_csv",
            "Export the current records view as CSV.",
            "Opens the export menu and chooses CSV. Ends with the download confirmed in "
            "the status bar.",
            {"scope": {"type": "string", "description": '"page" or "all" (default "page")'}},
            'def run(ctx, scope="page"):\n'
            '    ctx.ctl.click(ctx.find("Export menu").box.center)\n'
            "    ctx.ctl.click(ctx.find(scope.title()).box.center)\n"
            "    ctx.settle()\n",
            at(hours=8, minutes=2),
        ),
        (0, 0, 0.0),
        None,
    ),
    (
        skill(
            "old_search",
            "Search the records table using the legacy toolbar.",
            "Superseded by search_records once the console moved its search field into "
            "the table header.",
            {"company": {"type": "string", "description": "Company name to filter by"}},
            "def run(ctx, company):\n"
            '    ctx.ctl.click(ctx.find("toolbar search").box.center)\n'
            "    ctx.ctl.type_text(company)\n",
            at(days=-3, hours=13, minutes=20),
        ),
        (6, 1, 1720.0),
        "selector drifted after the console moved search into the table header",
    ),
]


def write_skills(root: Path) -> None:
    store = FileSkillStore(root)
    for definition, (runs, successes, mean_ms), demoted in SKILLS:
        stored = store.put(definition)
        for index in range(runs):
            store.record_run(stored.name, stored.domain, index < successes, mean_ms)
        if demoted:
            store.demote(stored.name, stored.domain, demoted)


# ----------------------------------------------------------------- site graph

STATES = [
    ("fp-records-list", "records list", "https://sandbox.test/records", at(days=-3, hours=9)),
    (
        "fp-records-found",
        "search results",
        "https://sandbox.test/records?q=acme",
        at(days=-3, hours=9, minutes=4),
    ),
    (
        "fp-record-detail",
        "record detail",
        "https://sandbox.test/records/101",
        at(days=-3, hours=9, minutes=9),
    ),
    ("fp-mail-inbox", "mail inbox", "https://sandbox.test/mail", at(days=-2, hours=10)),
    (
        "fp-mail-compose",
        "compose reply",
        "https://sandbox.test/mail/compose",
        at(days=-2, hours=10, minutes=6),
    ),
]

EDGES = [
    # src, dst, actions, attempts, successes, mean_ms
    (
        "fp-records-list",
        "fp-records-found",
        (Click(Point(210, 57)), TypeText("Acme Corp"), PressKey(("Enter",))),
        9,
        8,
        820.0,
    ),
    ("fp-records-found", "fp-record-detail", (Click(Point(210, 91)),), 7, 7, 410.0),
    # The tempting direct shortcut: fast when it works, and it mostly does not. Cost
    # is mean time over success rate, so the router prefers the two reliable hops.
    ("fp-records-list", "fp-record-detail", (Click(Point(210, 91)),), 3, 1, 1500.0),
    ("fp-record-detail", "fp-records-list", (PressKey(("Escape",)),), 5, 4, 350.0),
    ("fp-records-list", "fp-mail-inbox", (Click(Point(38, 48)),), 4, 4, 260.0),
    ("fp-mail-inbox", "fp-mail-compose", (Click(Point(300, 46)),), 3, 3, 300.0),
    ("fp-mail-compose", "fp-mail-inbox", (PressKey(("Escape",)),), 2, 1, 220.0),
]


def write_graph(root: Path, thumbnails: dict[str, bytes]) -> None:
    states = tuple(
        UIState(
            fingerprint=Fingerprint(value, {"url": value, "layout": "l1"}),
            domain=DOMAIN,
            label=label,
            url_pattern=url,
            first_seen=seen,
            thumbnail=thumbnails.get(value),
        )
        for value, label, url, seen in STATES
    )
    from skillweaver.contracts import Transition

    transitions = tuple(
        Transition(
            src=Fingerprint(src),
            dst=Fingerprint(dst),
            actions=actions,
            attempts=attempts,
            successes=successes,
            mean_ms=mean_ms,
            last_verified=at(hours=-3) if successes else None,
        )
        for src, dst, actions, attempts, successes, mean_ms in EDGES
    )
    JSONGraphStore(root).save(GraphSnapshot(domain=DOMAIN, states=states, transitions=transitions))


# ---------------------------------------------------------------- trajectory


def write_trajectory(root: Path, screens: dict[str, bytes]) -> None:
    start = at(hours=9, minutes=41)
    plain, typed, found, detail = (
        screens["list"],
        screens["typed"],
        screens["found"],
        screens["detail"],
    )
    plan = [
        (
            Navigate("https://sandbox.test/records"),
            plain,
            plain,
            "fp-records-list",
            "https://sandbox.test/records",
            "start from the records list",
            Verdict(True, "the records table is on screen", 0.99, "programmatic"),
            620.0,
        ),
        (
            Click(Point(210, 57)),
            plain,
            typed,
            "fp-records-list",
            "https://sandbox.test/records",
            "focus the search field before typing",
            Verdict(True, "the search field has focus", 0.96, "programmatic"),
            180.0,
        ),
        (
            TypeText("Acme Corp"),
            typed,
            typed,
            "fp-records-list",
            "https://sandbox.test/records",
            "type the company the task named",
            Verdict(True, "the field reads Acme Corp", 0.93, "model"),
            240.0,
        ),
        (
            PressKey(("Enter",)),
            typed,
            found,
            "fp-records-found",
            "https://sandbox.test/records?q=acme",
            "submit the search",
            Verdict(True, "three matching invoices are listed", 0.97, "model"),
            910.0,
        ),
        (
            Click(Point(210, 91)),
            found,
            detail,
            "fp-record-detail",
            "https://sandbox.test/records/101",
            "open the first matching invoice",
            Verdict(True, "INV-101 for Acme Corp is open", 0.98, "model"),
            430.0,
        ),
    ]
    steps: list[TrajectoryStep] = []
    when = start
    for index, (action, before_png, after_png, fingerprint, url, note, verdict, ms) in enumerate(
        plan
    ):
        before = observe(
            before_png, steps[-1].after.fingerprint.value if steps else "fp-records-list", url, when
        )
        when = when + timedelta(milliseconds=ms)
        after = observe(after_png, fingerprint, url, when)
        steps.append(
            TrajectoryStep(
                index=index,
                action=action,
                before=before,
                after=after,
                result=ActionResult(ok=True, error=None, elapsed_ms=ms),
                verdict=verdict,
                note=note,
            )
        )
    trajectory = Trajectory(
        run_id="a1b2c3d4e5f6",
        task="Find the invoice for Acme Corp and open it",
        domain=DOMAIN,
        steps=tuple(steps),
        ok=True,
        started_at=start,
        finished_at=when,
        note="opened INV-101; synthesized search_records from steps 1 to 3",
    )
    TrajectoryFileStore(root).save(trajectory)


# ---------------------------------------------------------------------- eval

METRICS = {
    "schema_version": 1,
    "suite": "sandbox-site",
    "generated_at": "2026-09-19T09:55:00+00:00",
    "tasks": [
        {
            "task_id": "find_invoice",
            "task_text": "Find the invoice for Acme Corp and open it",
            "domain": DOMAIN,
            "runs": [
                {
                    "attempt": 1,
                    "ok": True,
                    "wall_ms": 48120.0,
                    "llm_calls": 23,
                    "steps": 14,
                    "skill_used": None,
                    "usd": 0.42,
                    "run_id": "a1b2c3d4e5f6",
                    "started_at": "2026-09-19T09:41:00+00:00",
                },
                {
                    "attempt": 2,
                    "ok": True,
                    "wall_ms": 6210.0,
                    "llm_calls": 2,
                    "steps": 4,
                    "skill_used": "search_records",
                    "usd": 0.03,
                },
                {
                    "attempt": 3,
                    "ok": True,
                    "wall_ms": 5890.0,
                    "llm_calls": 2,
                    "steps": 4,
                    "skill_used": "search_records",
                    "usd": 0.03,
                },
            ],
        },
        {
            "task_id": "open_record",
            "task_text": "Open the detail view of invoice INV-101",
            "domain": DOMAIN,
            "runs": [
                {"attempt": 1, "ok": True, "wall_ms": 19400.0, "llm_calls": 11, "steps": 7},
                {
                    "attempt": 2,
                    "ok": True,
                    "wall_ms": 4100.0,
                    "llm_calls": 2,
                    "steps": 3,
                    "skill_used": "open_record_detail",
                },
            ],
        },
        {
            "task_id": "send_reply",
            "task_text": "Reply to the newest message in the inbox",
            "domain": DOMAIN,
            "runs": [
                {"attempt": 1, "ok": True, "wall_ms": 62000.0, "llm_calls": 31, "steps": 19},
                {
                    "attempt": 2,
                    "ok": True,
                    "wall_ms": 15800.0,
                    "llm_calls": 5,
                    "steps": 6,
                    "skill_used": "send_reply",
                },
                {
                    "attempt": 3,
                    "ok": False,
                    "wall_ms": 41000.0,
                    "llm_calls": 18,
                    "steps": 12,
                    "skill_used": "send_reply",
                },
            ],
        },
        {
            "task_id": "export_csv",
            "task_text": "Export the current records view as CSV",
            "domain": DOMAIN,
            "runs": [{"attempt": 1, "ok": True, "wall_ms": 27300.0, "llm_calls": 15, "steps": 9}],
        },
    ],
}

COLD_ONLY = {
    "schema_version": 1,
    "suite": "first-pass",
    "tasks": [
        {
            "task_id": "find_invoice",
            "task_text": "Find the invoice for Acme Corp and open it",
            "runs": [{"attempt": 1, "ok": True, "wall_ms": 48120.0, "llm_calls": 23}],
        }
    ],
}

BROKEN = '{"schema_version": 1, "suite": "oops", "tasks": "not a list at all"'


# --------------------------------------------------------------------- driver


def main() -> None:
    full = ROOT / "full"
    broken = ROOT / "broken"
    cold_only = ROOT / "cold_only"
    for path in (full, broken, cold_only):
        shutil.rmtree(path, ignore_errors=True)

    screens = {
        "list": screen("Northwind Console", (47, 109, 246), 4),
        "typed": screen("Northwind Console", (47, 109, 246), 4, highlight=0),
        "found": screen("Northwind Console - Acme Corp", (47, 109, 246), 3),
        "detail": screen("INV-101 - Acme Corp", (15, 138, 126), 2, highlight=0),
    }

    write_skills(full / "skills")
    write_graph(
        full / "graphs",
        {"fp-records-list": screens["list"], "fp-record-detail": screens["detail"]},
    )
    write_trajectory(full / "trajectories", screens)
    (full / "eval").mkdir(parents=True, exist_ok=True)
    (full / "eval" / "suite.json").write_text(
        json.dumps(METRICS, indent=2) + "\n", encoding="utf-8"
    )

    (broken / "eval").mkdir(parents=True, exist_ok=True)
    (broken / "eval" / "truncated.json").write_text(BROKEN, encoding="utf-8")
    (broken / "eval" / "wrong_shape.json").write_text(
        json.dumps({"schema_version": 1, "results": []}) + "\n", encoding="utf-8"
    )

    (cold_only / "eval").mkdir(parents=True, exist_ok=True)
    (cold_only / "eval" / "suite.json").write_text(
        json.dumps(COLD_ONLY, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {full}, {broken} and {cold_only}")


if __name__ == "__main__":
    main()
