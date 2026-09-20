"""Structured key-value logging: ``log.info("skill.run", name=..., ok=True)``.

Importing configures nothing; the first ``get_logger`` attaches one stderr handler to
the ``skillweaver`` logger only, never the root, so an embedding app is left alone.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

_ROOT = "skillweaver"
_lock = threading.Lock()
_configured = False


def _format_value(value: Any) -> str:
    text = str(value)
    if text == "" or any(ch in text for ch in ' "=\n\t'):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return text


def format_event(event: str, fields: dict[str, Any]) -> str:
    """``event`` plus ``key=value`` pairs; values with spaces, quotes, ``=`` or newlines
    are quoted and escaped."""
    if not fields:
        return event
    return event + " " + " ".join(f"{k}={_format_value(v)}" for k, v in fields.items())


class KVLogger:
    """Stdlib logger wrapper taking an event name plus keyword fields. The event is a
    short dotted identifier (``"graph.route"``), not a sentence."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def stdlib(self) -> logging.Logger:
        """The underlying logger, for handlers and levels."""
        return self._logger

    def _log(self, level: int, event: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, format_event(event, fields), exc_info=exc_info, stacklevel=3)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        """``error`` plus the active exception's traceback."""
        self._log(logging.ERROR, event, fields, exc_info=True)


def _configure_once() -> None:
    global _configured
    with _lock:
        if _configured:
            return
        root = logging.getLogger(_ROOT)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%H:%M:%S")
            )
            root.addHandler(handler)
        if root.level == logging.NOTSET:
            from skillweaver.config import settings

            root.setLevel(settings().log_level)
        _configured = True


def get_logger(name: str) -> KVLogger:
    """The structured logger for ``name`` (pass ``__name__``); names outside the
    ``skillweaver`` namespace are nested under it to share its handler and level."""
    _configure_once()
    if name != _ROOT and not name.startswith(_ROOT + "."):
        name = f"{_ROOT}.{name}"
    return KVLogger(logging.getLogger(name))
