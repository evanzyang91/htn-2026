"""The live inspector: a loopback page that drives a real browser session by hand.

Not :mod:`skillweaver.dashboard`, which writes a static, after-the-fact report. This is a
control surface - a prompt bar, step buttons, a running log, resets, and a live view of
the skill library - over the same agent the ``learn`` command runs.

:mod:`~skillweaver.inspector.session` is the run, :mod:`~skillweaver.inspector.library`
is the cached-skills view, and :mod:`~skillweaver.inspector.server` is the HTTP surface
and the security that goes with driving somebody's real browser. Start it with
``python -m skillweaver.cli inspect``.
"""

from skillweaver.inspector.server import DEFAULT_PORT, Inspector, serve
from skillweaver.inspector.session import LiveSession, SessionError

__all__ = ["DEFAULT_PORT", "Inspector", "LiveSession", "SessionError", "serve"]
