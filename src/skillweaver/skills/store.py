"""``FileSkillStore``: the skill library on disk.

Layout, rooted at :attr:`~skillweaver.config.Settings.skills_dir`::

    <data>/skills/manifest.json                       index: latest version per skill
    <data>/skills/<domain>/<name>/v1/skill.py         the source, byte for byte
    <data>/skills/<domain>/<name>/v1/meta.json        everything else
    <data>/skills/<domain>/<name>/v1/recorded-in      the render mode, when known
    <data>/skills/<domain>/<name>/v2/...              the next version; v1 stays

Three properties make this safe to grow into:

*Versions are append-only.* ``put`` claims ``v<N+1>`` with ``os.mkdir``, which fails
if the directory exists, so two writers can never agree to overwrite one version. A
regression is one ``put`` away from being reverted, because the old code is still
there.

*Writes are atomic.* Every file is written to a temporary sibling and then
``os.replace``d into place, so a reader never sees half a ``meta.json``.

*The manifest is a cache, not the truth.* It makes ``list`` skip the directory walk;
if it is missing, stale or corrupt, :meth:`FileSkillStore.rebuild_manifest` recovers
it from the directories, which are the real record.

``record_run`` rewrites only ``meta.json`` - the code file is never touched by
statistics - and ``demote`` writes a reason that keeps the skill out of listing and
retrieval while leaving every byte of it on disk.

Why ``recorded-in`` is a file of its own
----------------------------------------

It answers "which renderer produced the screen this skill starts on", which decides
whether a later run can compare against that screen at all - see
:mod:`skillweaver.render_mode`. It is deliberately NOT a key in ``meta.json``, for two
reasons. ``meta.json`` is regenerated wholesale from ``skills.model.to_dict`` on every
``record_run`` and every ``demote``, and ``to_dict`` serializes a
:class:`~skillweaver.contracts.Skill`, which has no field for the mode and cannot grow
one here: ``contracts`` is shared surface. A key written beside that output would
therefore survive the ``put`` and vanish on the skill's first recorded run, which is a
worse record than none. So the mode lives beside ``skill.py`` as its own small file,
written once when the version is claimed and never rewritten - the same shape, and for
the same reason, as the source it describes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from skillweaver.config import settings
from skillweaver.contracts import Precedent, Skill, SkillStats, utcnow
from skillweaver.errors import SkillNotFound, SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.render_mode import MODES
from skillweaver.skills.model import from_dict, to_dict, validate_identity

__all__ = [
    "CODE_FILE",
    "MANIFEST_FILE",
    "META_FILE",
    "RENDER_MODE_FILE",
    "FileSkillStore",
]

log = get_logger(__name__)

CODE_FILE = "skill.py"
META_FILE = "meta.json"
MANIFEST_FILE = "manifest.json"
RENDER_MODE_FILE = "recorded-in"
MANIFEST_VERSION = 1

_VERIFIER_FILE = "verify.py"

MAX_PRECEDENTS = 12
"""How many proven requests one skill keeps. Each is a template the binder tries, so
the list is a cost on every warm lookup; a dozen wordings of one errand is already
more than a person uses."""


def _fold_mean(mean: float, count_before: int, value: float) -> float:
    """The mean of ``count_before`` values known only through ``mean``, plus
    ``value``. Folding one run at a time keeps the mean exact without storing history.
    """
    return (mean * count_before + value) / (count_before + 1)


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` so a reader sees either the old file or the new
    one, never a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillWeaverError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillWeaverError(f"{path} does not hold a JSON object")
    return data


def _dump_json(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


class FileSkillStore:
    """A :class:`~skillweaver.contracts.SkillStore` backed by a directory tree.

    Args:
        root: the skills directory. ``None`` means
            :attr:`~skillweaver.config.Settings.skills_dir`, so nothing is hardcoded.
        render_mode: the mode the browser of THIS process renders in
            (:data:`~skillweaver.render_mode.HEADED` or
            :data:`~skillweaver.render_mode.HEADLESS`), stamped on every version this
            store writes. ``None`` means "no claim" and writes nothing, which is the
            honest answer for a store opened by a command that never opens a world -
            ``skills ls`` must not assert a mode it did not render in.

    Raises:
        SkillWeaverError: if ``render_mode`` is not one of the known modes. A mode
            misspelled here would be written to every skill stored afterwards.
    """

    def __init__(self, root: Path | str | None = None, *, render_mode: str | None = None) -> None:
        if render_mode is not None and render_mode not in MODES:
            raise SkillWeaverError(f"render_mode must be one of {MODES}, got {render_mode!r}")
        self.root = Path(root) if root is not None else settings().skills_dir
        self._render_mode = render_mode
        self._manifest: dict[tuple[str, str], dict[str, Any]] | None = None

    # -- layout ------------------------------------------------------------------

    def skill_dir(self, name: str, domain: str) -> Path:
        """Where every version of ``(name, domain)`` lives."""
        return self.root / domain / name

    def version_dir(self, name: str, domain: str, version: int) -> Path:
        """Where one version lives: ``<root>/<domain>/<name>/v<N>``."""
        return self.skill_dir(name, domain) / f"v{version}"

    @property
    def render_mode(self) -> str | None:
        """The mode this store stamps on what it writes, or ``None`` for no claim."""
        return self._render_mode

    # -- manifest ----------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    def _entries(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> dict[tuple[str, str], dict[str, Any]]:
        path = self.manifest_path
        if not path.is_file():
            return self._scan()
        try:
            raw = _read_json(path)
            if raw.get("manifest_version") != MANIFEST_VERSION:
                raise SkillWeaverError(f"manifest_version {raw.get('manifest_version')!r}")
            entries = {
                (e["name"], e["domain"]): e
                for e in raw["entries"]
                if isinstance(e, dict) and {"name", "domain", "latest"} <= set(e)
            }
        except (SkillWeaverError, KeyError, TypeError) as exc:
            log.warning("skills.manifest.rebuild", path=str(path), reason=str(exc))
            return self._scan()
        return entries

    def _scan(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Rebuild the index from the directories, which are the real record. A
        version directory without a ``meta.json`` is an interrupted ``put`` and is
        skipped."""
        entries: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.root.is_dir():
            return entries
        for domain_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for skill_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
                versions = self._versions_on_disk(skill_dir)
                if not versions:
                    continue
                latest = max(versions)
                meta = _read_json(skill_dir / f"v{latest}" / META_FILE)
                entries[(skill_dir.name, domain_dir.name)] = {
                    "name": skill_dir.name,
                    "domain": domain_dir.name,
                    "latest": latest,
                    "summary": meta.get("summary", ""),
                    "demoted_reason": meta.get("demoted_reason"),
                }
        return entries

    @staticmethod
    def _versions_on_disk(skill_dir: Path) -> list[int]:
        found: list[int] = []
        if not skill_dir.is_dir():
            return found
        for child in skill_dir.iterdir():
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                if (child / META_FILE).is_file():
                    found.append(int(child.name[1:]))
        return found

    def _save_manifest(self) -> None:
        entries = self._entries()
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "entries": [entries[key] for key in sorted(entries, key=lambda k: (k[1], k[0]))],
        }
        _write_atomic(self.manifest_path, _dump_json(payload))

    def rebuild_manifest(self) -> int:
        """Recover the index by walking the directories and return how many skills it
        found. Safe to call at any time; the directories always win."""
        self._manifest = self._scan()
        self._save_manifest()
        log.info("skills.manifest.rebuilt", root=str(self.root), skills=len(self._manifest))
        return len(self._manifest)

    # -- reading -----------------------------------------------------------------

    def _load_version(self, name: str, domain: str, version: int) -> Skill:
        directory = self.version_dir(name, domain, version)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise SkillNotFound(f"skill {name!r} in domain {domain!r} has no version {version}")
        code = (directory / CODE_FILE).read_bytes().decode("utf-8")
        verifier_path = directory / _VERIFIER_FILE
        verifier = verifier_path.read_bytes().decode("utf-8") if verifier_path.is_file() else None
        return from_dict(_read_json(meta_path), code=code, verifier_code=verifier)

    def _latest_version(self, name: str, domain: str) -> int:
        entry = self._entries().get((name, domain))
        if entry is not None and (self.version_dir(name, domain, entry["latest"])).is_dir():
            return int(entry["latest"])
        versions = self._versions_on_disk(self.skill_dir(name, domain))
        if not versions:
            raise SkillNotFound(f"no skill {name!r} in domain {domain!r}")
        return max(versions)

    def get(self, name: str, domain: str, version: int | None = None) -> Skill:
        """Return one skill; ``version=None`` means the latest. Demoted skills are
        still returned - check ``demoted_reason``.

        Raises:
            SkillNotFound: if the name, domain or version does not exist.
        """
        if version is None:
            version = self._latest_version(name, domain)
        elif version < 1:
            raise SkillNotFound(f"version must be 1 or greater, got {version}")
        return self._load_version(name, domain, version)

    def versions(self, name: str, domain: str) -> list[int]:
        """Every stored version number of ``(name, domain)``, ascending. Empty when
        the skill is unknown."""
        return sorted(self._versions_on_disk(self.skill_dir(name, domain)))

    def recorded_render_mode(
        self, name: str, domain: str, version: int | None = None
    ) -> str | None:
        """The mode this version's screens were rendered in, or ``None`` when unknown.

        ``None`` covers every honest way the answer can be missing and they are all the
        same answer to a caller: a version stored before this was recorded, a store that
        was opened with no claim, a file that says something this build does not
        recognize. Never raises and never guesses - a wrong mode in a report is worse
        than an absent one, because it accuses a working library of a mismatch it does
        not have.

        ``version=None`` means the latest, and an unknown skill is ``None`` rather than
        :exc:`~skillweaver.errors.SkillNotFound`: this is a question asked while
        deciding whether to bother comparing, not a read of the library.
        """
        try:
            resolved = self._latest_version(name, domain) if version is None else int(version)
        except SkillNotFound:
            return None
        path = self.version_dir(name, domain, resolved) / RENDER_MODE_FILE
        try:
            found = path.read_bytes().decode("utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return found if found in MODES else None

    def recorded_render_modes(self, domain: str | None = None) -> dict[str, str]:
        """``{skill name: mode}`` for the latest version of every skill that says.

        Skills whose mode is unknown are LEFT OUT rather than mapped to ``None``, so a
        caller can ask "did anything here claim a mode?" by looking at the mapping and
        "do they all disagree with me?" without filtering first. ``domain=None`` spans
        the library; a name appears once, because only latest versions are consulted.
        """
        found: dict[str, str] = {}
        for (name, entry_domain), _entry in self._entries().items():
            if domain is not None and entry_domain != domain:
                continue
            mode = self.recorded_render_mode(name, entry_domain)
            if mode is not None:
                found[name] = mode
        return found

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[Skill]:
        """The latest version of every skill, optionally restricted to one domain,
        sorted by ``(domain, name)``. Demoted skills are omitted unless asked for."""
        wanted = [
            entry
            for (name, entry_domain), entry in self._entries().items()
            if (domain is None or entry_domain == domain)
            and (include_demoted or entry.get("demoted_reason") is None)
        ]
        wanted.sort(key=lambda e: (e["domain"], e["name"]))
        return [self._load_version(e["name"], e["domain"], int(e["latest"])) for e in wanted]

    # -- writing -----------------------------------------------------------------

    def put(self, skill: Skill) -> Skill:
        """Store ``skill`` as the NEXT version of ``(name, domain)`` and return the
        stored copy. The incoming ``version`` is ignored: the first ``put`` yields
        version ``1``, each later one increments. Older versions are kept and stay
        readable through :meth:`get`.

        Admission checks belong before ``put``; the only thing checked here is that
        the name and domain can be a path at all.

        Raises:
            InvalidSkillName, InvalidSkillDomain: if the key is unusable on disk.
        """
        validate_identity(skill)
        on_disk = self._versions_on_disk(self.skill_dir(skill.name, skill.domain))
        entry = self._entries().get((skill.name, skill.domain))
        next_version = max([*on_disk, int(entry["latest"]) if entry else 0], default=0) + 1

        directory = self._claim_version_dir(skill.name, skill.domain, next_version)
        version = int(directory.name[1:])
        stored = replace(skill, version=version)

        _write_atomic(directory / CODE_FILE, stored.code.encode("utf-8"))
        if stored.verifier_code is not None:
            _write_atomic(directory / _VERIFIER_FILE, stored.verifier_code.encode("utf-8"))
        _write_atomic(directory / META_FILE, _dump_json(to_dict(stored, include_code=False)))
        if self._render_mode is not None:
            _write_atomic(directory / RENDER_MODE_FILE, f"{self._render_mode}\n".encode())

        self._entries()[(stored.name, stored.domain)] = {
            "name": stored.name,
            "domain": stored.domain,
            "latest": version,
            "summary": stored.summary,
            "demoted_reason": stored.demoted_reason,
        }
        self._save_manifest()
        log.info(
            "skills.put",
            name=stored.name,
            domain=stored.domain,
            version=version,
            recorded_in=self._render_mode,
        )
        return stored

    def _claim_version_dir(self, name: str, domain: str, start: int) -> Path:
        """Create the first free ``v<N>`` directory at or after ``start`` and return
        it. ``mkdir`` is the lock: it fails if another writer got there first."""
        parent = self.skill_dir(name, domain)
        parent.mkdir(parents=True, exist_ok=True)
        for version in range(start, start + 1000):
            candidate = parent / f"v{version}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate
        raise SkillWeaverError(  # pragma: no cover - a corrupt tree
            f"cannot claim a version for {name!r} in {parent}"
        )

    def _rewrite_meta(self, skill: Skill) -> Skill:
        """Persist everything but the code of ``skill`` over its own version."""
        path = self.version_dir(skill.name, skill.domain, skill.version) / META_FILE
        if not path.is_file():
            raise SkillNotFound(
                f"skill {skill.name!r} in domain {skill.domain!r} has no version {skill.version}"
            )
        _write_atomic(path, _dump_json(to_dict(skill, include_code=False)))
        return skill

    def record_run(self, name: str, domain: str, ok: bool, ms: float) -> Skill:
        """Fold one execution into the latest version's
        :class:`~skillweaver.contracts.SkillStats` and return the updated skill.

        ``mean_ms`` is the mean of SUCCESSFUL runs only - a failure that gave up after
        a timeout would otherwise make a good skill look slow. Only ``meta.json`` is
        rewritten; the code file is untouched.

        Raises:
            SkillNotFound: if the skill does not exist.
        """
        skill = self.get(name, domain)
        old = skill.stats
        stats = SkillStats(
            runs=old.runs + 1,
            successes=old.successes + (1 if ok else 0),
            mean_ms=_fold_mean(old.mean_ms, old.successes, ms) if ok else old.mean_ms,
            last_ok_at=utcnow() if ok else old.last_ok_at,
        )
        updated = self._rewrite_meta(replace(skill, stats=stats))
        log.info(
            "skills.record_run",
            name=name,
            domain=domain,
            ok=ok,
            ms=round(ms, 1),
            runs=stats.runs,
            successes=stats.successes,
        )
        return updated

    def record_precedent(self, name: str, domain: str, precedent: Precedent) -> Skill:
        """Append one PROVEN request to the latest version and return the updated skill.

        The caller owns the proof - see :class:`~skillweaver.contracts.Precedent` for
        what counts - and this only writes it down. A sentence already on record is
        not recorded twice, and the list is capped at :data:`MAX_PRECEDENTS`, oldest
        reuse dropped first and the admission run (the first entry) always kept,
        because every one of them is a template binding is tried against.

        Raises:
            SkillNotFound: if the skill does not exist.
        """
        skill = self.get(name, domain)
        asked = " ".join(precedent.task_text.casefold().split())
        if any(" ".join(p.task_text.casefold().split()) == asked for p in skill.precedents):
            return skill
        kept = (*skill.precedents, precedent)
        if len(kept) > MAX_PRECEDENTS:
            kept = (kept[0], *kept[-(MAX_PRECEDENTS - 1) :])
        updated = self._rewrite_meta(replace(skill, precedents=kept))
        log.info(
            "skills.record_precedent",
            name=name,
            domain=domain,
            task=precedent.task_text,
            precedents=len(kept),
        )
        return updated

    def demote(self, name: str, domain: str, reason: str) -> Skill:
        """Retire the latest version from listing and retrieval by setting
        ``demoted_reason``, and return the updated skill. Nothing is deleted: ``get``
        still returns it, and a later ``put`` of a fixed version is healthy again.

        Raises:
            SkillNotFound: if the skill does not exist.
            SkillWeaverError: if ``reason`` is empty - a demotion a human cannot read
                is worse than none.
        """
        if not reason or not reason.strip():
            raise SkillWeaverError("demote() needs a non-empty reason")
        updated = self._rewrite_meta(replace(self.get(name, domain), demoted_reason=reason))
        self._entries()[(name, domain)] = {
            "name": name,
            "domain": domain,
            "latest": updated.version,
            "summary": updated.summary,
            "demoted_reason": reason,
        }
        self._save_manifest()
        log.info("skills.demote", name=name, domain=domain, reason=reason)
        return updated

    def __repr__(self) -> str:
        mode = "" if self._render_mode is None else f", render_mode={self._render_mode!r}"
        return f"FileSkillStore(root={str(self.root)!r}{mode})"
