"""Import, in the background, what the run is about to need - while it waits on something else.

A run's first seconds are spent WAITING - for the browser daemon, for the first page to load,
for the first policy answer - and then, one after another, on imports nobody asked for until
that moment: Pillow at the first fingerprint, the Anthropic SDK at the model client. Measured
on the ``--perception dom --policy jev`` path, 2026-09-20, same machine, two consecutive
runs: ``import anthropic`` 1.26s then 0.30s, and the first observation 1.00s then 0.07s with
an identical 45ms of browser calls inside it. That the slow run was reading the files from
disk and the fast one from the OS cache is an INFERENCE - it was the first run after a
``uv sync`` - and not something this project measured. Either way none of it has to be on the
critical path, because the main thread is asleep on a socket for longer than all of it takes.

This changes WHEN a module is imported and nothing else. A name that fails to import is left
for the real ``import`` statement to fail on, in the place and with the message it always had;
a name already imported costs nothing; and the importer's own per-module locks make a real
``import`` that arrives mid-prefetch wait for it rather than race it. ONE thread imports the
names in order, so two prefetched modules can never hold each other's import lock.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Sequence

__all__ = ["prefetch"]


def prefetch(*names: str) -> threading.Thread | None:
    """Start importing ``names`` on one daemon thread; ``None`` if all are already in.

    Never raises and never reports: a module that cannot be imported here is imported again,
    for real, by the code that needs it, and that is where the failure belongs.
    """
    wanted = [name for name in names if name not in sys.modules]
    if not wanted:
        return None
    thread = threading.Thread(
        target=_import_all, args=(wanted,), name="skillweaver-prefetch", daemon=True
    )
    thread.start()
    return thread


def _import_all(names: Sequence[str]) -> None:
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - the real import reports it, where it is needed
            continue
