"""What the agent already knows, read live while it is running.

The inspector's one panel that upstream's has no counterpart for. It answers one
question - *what does this agent already know how to do?* - and it has to keep answering
it while a ``learn`` run in another terminal is adding to the library, because watching a
skill APPEAR is the moment the whole project is about.

Read through the store, not the directories
-------------------------------------------

Everything here comes from :class:`~skillweaver.skills.store.FileSkillStore`, which is
what ``skills ls`` reads and what the admission gate writes. Walking ``data/skills/``
by hand would be a second reader of the same bytes, and the two would disagree the first
time a version directory was half-written - :meth:`FileSkillStore._scan` skips a version
with no ``meta.json`` precisely because an interrupted ``put`` leaves one.

Why a new store per refresh
---------------------------

A ``FileSkillStore`` caches its manifest in memory on first read, which is right for a
command that runs once and exits and wrong for a page that polls. So :class:`SkillLibrary`
holds no store: it watches the manifest file's mtime and the skills directory's, and when
either moves it builds a FRESH store and reads through that. Nothing is written - in
particular ``rebuild_manifest`` is NOT called, because it rewrites the manifest and a
learn run in another process may be part-way through writing that file itself.

What a row says
---------------

Name, domain, version, when it was recorded and whether it carries a verifier, because
those five are the question. Then the ones a person asks next: the render mode its start
screen was recorded in (a skill recorded headed cannot match a headless screen - see
:mod:`skillweaver.render_mode`), whether it has a precondition at all, its action
signature, how many requests it is PROVEN to have served, and its run statistics with the
warning that belongs beside them.

``SkillStats`` is reported and labelled untrustworthy in the same breath, on purpose:
``record_run`` is told ``ok`` by the sandbox and the sandbox is told ``ok`` by the
skill's own verifier, so a verifier that passes on the wrong screen reads as 14 runs and
14 successes. That is written up in ``AGENTS.md`` and it is the reason this panel shows
``verifier`` as a first-class column rather than a footnote.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillweaver.contracts import Skill
from skillweaver.logging_ import get_logger
from skillweaver.perception_mode import bare, path_of
from skillweaver.skills.store import MANIFEST_FILE, FileSkillStore

log = get_logger(__name__)

__all__ = ["SkillLibrary", "skill_row"]


@dataclass(frozen=True, slots=True)
class _Stamp:
    """What the library looked like on disk, cheaply."""

    manifest_mtime: float
    root_mtime: float
    domains: tuple[float, ...]


class SkillLibrary:
    """A polled, read-only view of the skill library.

    Args:
        skills_dir: The library root - :attr:`~skillweaver.config.Settings.skills_dir`.
        min_interval: The shortest gap between two disk checks, in seconds. The check
            itself is a handful of ``stat`` calls, so this is politeness rather than
            necessity; the full re-read only happens when a stamp actually moved.
    """

    def __init__(self, skills_dir: Path, *, min_interval: float = 1.0) -> None:
        self._root = Path(skills_dir)
        self._min_interval = min_interval
        self._stamp: _Stamp | None = None
        self._checked_at = 0.0
        self._rows: list[dict[str, Any]] = []
        self._read_at = 0.0

    def rows(self) -> list[dict[str, Any]]:
        """Every skill in the library, newest recording first, re-read when it changed."""
        now = time.monotonic()
        if now - self._checked_at >= self._min_interval:
            self._checked_at = now
            stamp = self._stamp_now()
            if stamp != self._stamp:
                self._stamp = stamp
                self._rows = self._read()
        return self._rows

    def as_json(self) -> dict[str, Any]:
        """The panel's whole payload: the rows, and what they were read from."""
        rows = self.rows()
        return {
            "root": str(self._root),
            "count": len(rows),
            "domains": sorted({row["domain"] for row in rows}),
            "read_at": self._read_at,
            "skills": rows,
        }

    # -- reading -----------------------------------------------------------------------

    def _stamp_now(self) -> _Stamp:
        """A cheap fingerprint of the library's shape on disk.

        The per-domain mtimes are in it because a ``put`` of a NEW version of an existing
        skill touches that skill's directory and the manifest, while a put of a new skill
        touches the domain directory too - and a checkout whose manifest is missing
        entirely (``_scan`` rebuilds from the directories) has no manifest mtime to move
        at all. Stat-ing one level of directories covers all three.
        """
        return _Stamp(
            manifest_mtime=_mtime(self._root / MANIFEST_FILE),
            root_mtime=_mtime(self._root),
            domains=tuple(
                sorted(_mtime(path) for path in self._root.glob("*") if path.is_dir())
            ),
        )

    def _read(self) -> list[dict[str, Any]]:
        """Build every row through a FRESH store. Never raises; a broken library is a row
        count of zero and a log line, because a panel that throws takes the page with it."""
        try:
            store = FileSkillStore(self._root)
            skills = store.list(include_demoted=True)
        except Exception as exc:  # noqa: BLE001 - the page must survive a bad library
            log.warning("inspect.library.unreadable", root=str(self._root), error=str(exc))
            return []
        rows = [skill_row(skill, store) for skill in skills]
        rows.sort(key=lambda row: row["created_at"], reverse=True)
        self._read_at = time.time()
        log.info("inspect.library.read", root=str(self._root), skills=len(rows))
        return rows


def skill_row(skill: Skill, store: FileSkillStore) -> dict[str, Any]:
    """One stored skill as the row the page draws.

    ``domain`` is the BARE host and ``path`` the perception path it was filed under,
    split apart here because the two share one namespaced string - see
    :mod:`skillweaver.perception_mode` - and a person reading a library wants to see
    that a skill learned through the DOM is not offered to a pixel run.
    """
    stats = skill.stats
    return {
        "name": skill.name,
        "domain": bare(skill.domain),
        "namespace": skill.domain,
        "path": path_of(skill.domain),
        "version": skill.version,
        "versions": store.versions(skill.name, skill.domain),
        "summary": skill.summary,
        "created_at": skill.provenance.created_at.isoformat(),
        "task_text": skill.provenance.task_text,
        "model": skill.provenance.model,
        "trajectory_id": skill.provenance.trajectory_id,
        "verifier": skill.verifier_code is not None,
        "precondition": skill.precondition.value if skill.precondition else "",
        "render_mode": store.recorded_render_mode(skill.name, skill.domain) or "",
        "params": sorted(skill.params),
        "requires": list(skill.requires),
        "action_signature": list(skill.action_signature),
        "precedents": [precedent.task_text for precedent in skill.precedents],
        "demoted_reason": skill.demoted_reason or "",
        "stats": {
            "runs": stats.runs,
            "successes": stats.successes,
            "mean_ms": round(stats.mean_ms),
            "last_ok_at": stats.last_ok_at.isoformat() if stats.last_ok_at else "",
        },
        "lines": len(skill.code.splitlines()),
    }


def _mtime(path: Path) -> float:
    """A path's modification time, or ``0.0`` when it is not there. Never raises."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
