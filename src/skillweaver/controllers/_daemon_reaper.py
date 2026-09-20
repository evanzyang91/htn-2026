"""Stop a Browser Harness daemon nobody has used for a while.

The daemon is one process per ``BU_NAME`` and OUTLIVES every run that used it, on
purpose: the next run attaches in ~1ms instead of ~200ms. Nothing ever stops it, so a
name used once stays a process forever - measured 2026-09-20: 47 of them after an
afternoon of runs that each chose a fresh ``BU_NAME``, every one holding a websocket to a
Chrome that had already exited.

So a run that uses a NAMED daemon leaves a heartbeat behind and makes sure one reaper is
watching it: a detached process that stops the daemon, through Browser Harness's own
``restart_daemon`` (which despite its name only stops), once the heartbeat is
:data:`DEFAULT_IDLE_SECONDS` old. The ``default`` daemon is exempt: it is the one attached
to the person's own Chrome, other tools share it, and stopping it means Chrome asks
"Allow remote debugging?" again on the next run.

``SKILLWEAVER_DAEMON_IDLE_S`` overrides the window; ``0`` turns the reaper off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_IDLE_SECONDS = 300.0
"""Five minutes. Long enough to sit between the runs of one sitting, which is what the
daemon outliving a run is for."""

ENV_IDLE = "SKILLWEAVER_DAEMON_IDLE_S"

EXEMPT_NAMES = frozenset({"default"})

_BEAT_EVERY_SECONDS = 5.0
"""A heartbeat is a ``utime``; this keeps it off the per-call path all the same."""

_last_beat: dict[str, float] = {}


def idle_seconds() -> float:
    """The configured window, or the default when unset or unreadable."""
    raw = os.environ.get(ENV_IDLE, "").strip()
    try:
        return max(float(raw), 0.0) if raw else DEFAULT_IDLE_SECONDS
    except ValueError:
        return DEFAULT_IDLE_SECONDS


def _beat_path(name: str) -> Path:
    return Path(tempfile.gettempdir()) / f"skillweaver-bu-{name}.beat"


def beat(name: str) -> None:
    """Say the daemon ``name`` was used just now. Never raises."""
    now = time.monotonic()
    if now - _last_beat.get(name, -_BEAT_EVERY_SECONDS) < _BEAT_EVERY_SECONDS:
        return
    _last_beat[name] = now
    try:
        _beat_path(name).touch()
    except OSError:
        pass


def watch(name: str) -> bool:
    """Make sure a reaper is watching ``name``. ``False`` when it is exempt or turned off.

    Spawning is unconditional and cheap: a second reaper for the same name finds the lock
    held and exits at once, so callers need not know whether one is already there.
    """
    idle = idle_seconds()
    if name in EXEMPT_NAMES or idle <= 0:
        return False
    _last_beat.pop(name, None)
    beat(name)
    try:
        subprocess.Popen(  # noqa: S603 - our own interpreter and module
            [sys.executable, "-m", __name__, name, str(idle)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False
    return True


def _reap(name: str, idle: float) -> int:
    """The reaper's whole life. Returns the exit code."""
    import fcntl

    from browser_harness.admin import daemon_alive, restart_daemon

    beat_path = _beat_path(name)
    with open(beat_path.with_suffix(".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0  # another reaper has this name
        poll = max(min(idle / 10.0, 30.0), 0.05)
        while True:
            time.sleep(poll)
            if not daemon_alive(name):
                break
            try:
                age = time.time() - beat_path.stat().st_mtime
            except OSError:
                age = idle + 1.0  # no heartbeat left means nobody is using it
            if age >= idle:
                restart_daemon(name)
                break
        try:
            beat_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(_reap(sys.argv[1], float(sys.argv[2])))
