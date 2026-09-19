"""Structured key-value logging.

::

    log = get_logger(__name__)
    log.info("skill.run", name="search_invoice", ok=True, ms=412.5)
    # 12:00:01 INFO skillweaver.skills.runner skill.run name=search_invoice ok=True ms=412.5

Importing this module configures nothing. The first :func:`get_logger` call attaches
one stderr handler to the ``skillweaver`` logger only - never the root logger - so
an embedding application's logging is left alone.
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
    """Render ``event`` followed by ``key=value`` pairs; values containing spaces,
    quotes, ``=`` or newlines are double-quoted and escaped."""
    if not fields:
        return event
    return event + " " + " ".join(f"{k}={_format_value(v)}" for k, v in fields.items())


class KVLogger:
    """A thin wrapper over a stdlib logger whose methods take an event name plus
    keyword fields. The event name is a short dotted identifier (``"graph.route"``),
    not a sentence; put the variable parts in fields."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def stdlib(self) -> logging.Logger:
        """The underlying ``logging.Logger`` (for handlers, levels, ``caplog``)."""
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
        """Like :meth:`error`, and appends the active exception's traceback."""
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
    """Return the structured logger for ``name`` (pass ``__name__``).

    Names outside the ``skillweaver`` namespace (tests, scripts) are nested under it
    so they share its handler and level.
    """
    _configure_once()
    if name != _ROOT and not name.startswith(_ROOT + "."):
        name = f"{_ROOT}.{name}"
    return KVLogger(logging.getLogger(name))
