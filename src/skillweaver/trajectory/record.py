"""``Recorder``: writes a run to disk as it happens, one line per action.

Deliberately not a buffer that saves at the end. Each step is appended and FSYNCED before
returning, so a run killed by a budget, a crash or an impatient demo operator still leaves
a loadable directory - the header and every completed step, with ``ok=False``. That
partial run is what exploration produces most of the time and is still worth reading.

Screenshots are written BEFORE the step line that names them, and by content digest, so a
step line never points at a missing or half-written PNG and two steps showing one screen
cost one file.

The format lives in :mod:`skillweaver.trajectory.store`; this module only writes it.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from skillweaver.config import settings
from skillweaver.contracts import (
    Action,
    ActionResult,
    Observation,
    Trajectory,
    TrajectoryStep,
    Verdict,
    utcnow,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.trajectory.store import (
    SCREENS_DIR,
    TRAJECTORY_FILE,
    encode_footer,
    encode_header,
    encode_step,
    run_dir_name,
    write_screenshot,
)

log = get_logger(__name__)


def new_run_id() -> str:
    """A fresh run identifier: twelve hex characters, unique in practice."""
    return uuid.uuid4().hex[:12]


class Recorder:
    """A ``contracts.TrajectoryRecorder`` that writes each step through to disk.

    ONE RUN AT A TIME: :meth:`start` on a recorder already recording raises, rather than
    silently interleaving two runs into one file.

    Args:
        root: ``None`` means ``settings().trajectories_dir``; no path is hardcoded here.
        clock: Source of UTC timestamps, injectable for determinism.
        run_ids: Source of run identifiers, injectable for the same reason.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        clock: Callable[[], datetime] = utcnow,
        run_ids: Callable[[], str] = new_run_id,
    ) -> None:
        self.root = Path(root) if root is not None else settings().trajectories_dir
        self._clock = clock
        self._run_ids = run_ids
        self._run_id: str | None = None
        self._path: Path | None = None
        self._task = ""
        self._domain = ""
        self._started_at: datetime | None = None
        self._steps: list[TrajectoryStep] = []

    # -- inspection --------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        """The run being recorded, or ``None`` when idle."""
        return self._run_id

    @property
    def path(self) -> Path | None:
        """The directory of the run being recorded, or ``None`` when idle."""
        return self._path

    @property
    def steps(self) -> int:
        """How many steps have been recorded in the current run."""
        return len(self._steps)

    # -- TrajectoryRecorder protocol ---------------------------------------------------

    def start(self, task: str, domain: str) -> str:
        """Begin a run: create its directory, write the header line, return the id.

        Raises:
            SkillWeaverError: a run is in progress, or the directory cannot be created.
        """
        if self._run_id is not None:
            raise SkillWeaverError(f"run {self._run_id} is still in progress")
        run_id = self._run_ids()
        started_at = self._clock()
        path = self.root / run_dir_name(started_at, run_id)
        try:
            (path / SCREENS_DIR).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SkillWeaverError(f"cannot create trajectory directory {path}: {exc}") from exc
        self._run_id, self._path = run_id, path
        self._task, self._domain, self._started_at = task, domain, started_at
        self._steps = []
        self._append(
            encode_header(
                {
                    "run_id": run_id,
                    "task": task,
                    "domain": domain,
                    "started_at": started_at.isoformat(),
                }
            )
        )
        log.info("trajectory.start", run_id=run_id, domain=domain, path=str(path))
        return run_id

    def step(
        self,
        action: Action,
        before: Observation,
        after: Observation,
        result: ActionResult,
        verdict: Verdict | None = None,
        note: str = "",
    ) -> TrajectoryStep:
        """Append one step: its screenshots first, then its line, then fsync.

        Raises:
            SkillWeaverError: no run is in progress, or the step cannot be written.
        """
        if self._run_id is None or self._path is None:
            raise SkillWeaverError("no run in progress: call start() first")
        recorded = TrajectoryStep(len(self._steps), action, before, after, result, verdict, note)
        screens = self._path / SCREENS_DIR
        try:
            before_digest = write_screenshot(screens, before.screenshot.png)
            after_digest = write_screenshot(screens, after.screenshot.png)
        except OSError as exc:
            raise SkillWeaverError(
                f"cannot write screenshot for run {self._run_id}: {exc}"
            ) from exc
        self._append(encode_step(recorded, before_digest, after_digest))
        self._steps.append(recorded)
        return recorded

    def finish(self, ok: bool, note: str = "") -> Trajectory:
        """Write the footer, close the run and return the finished trajectory.

        The file is already complete when this returns; handing the result to
        ``TrajectoryFileStore.save`` rewrites the same content and is optional.
        """
        if self._run_id is None or self._started_at is None:
            raise SkillWeaverError("no run in progress: call start() first")
        finished_at = self._clock()
        self._append(encode_footer(ok, finished_at, note))
        trajectory = Trajectory(
            run_id=self._run_id,
            task=self._task,
            domain=self._domain,
            steps=tuple(self._steps),
            ok=ok,
            started_at=self._started_at,
            finished_at=finished_at,
            note=note,
        )
        log.info("trajectory.finish", run_id=self._run_id, ok=ok, steps=len(self._steps), note=note)
        self._run_id, self._path, self._started_at = None, None, None
        self._steps = []
        return trajectory

    # -- internals ---------------------------------------------------------------------

    def _append(self, line: dict[str, Any]) -> None:
        """Append one JSON line and fsync it before returning."""
        assert self._path is not None  # noqa: S101 - callers check; this documents the invariant
        path = self._path / TRAJECTORY_FILE
        text = json.dumps(line, ensure_ascii=False) + "\n"
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise SkillWeaverError(f"cannot append to {path}: {exc}") from exc
