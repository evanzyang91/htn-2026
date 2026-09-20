"""The loopback inspector: a small HTTP server in front of one live agent session.

Shaped after the inspector on ``origin/jev-ultrafast`` - a
:class:`~http.server.ThreadingHTTPServer` bound to ``127.0.0.1``, ``GET /api/state`` for
polled state, ``POST /api/<command>`` for the buttons, and a
:func:`secrets.token_urlsafe` minted at startup that every request has to carry. What is
driven behind it is this project's own stack: :class:`~skillweaver.inspector.session.
LiveSession`, which is the explorer's loop one step at a time.

Two things this server does that that one does not have to
----------------------------------------------------------

**One thread owns the browser.** Playwright's sync API runs an event loop per thread and
refuses to be touched from another, and ``ThreadingHTTPServer`` answers every request on
a new thread - so the agent lives on :class:`_Worker` and the handlers post jobs to it and
wait. This is not a nicety: without it the first button press from a second HTTP thread
kills the driver. :class:`~skillweaver.controllers.browser._ThreadDriver` is where that
constraint is written down.

**Reading the state never waits for the agent.** A ``predict`` is a model call and can
take ten seconds; a page that polled through the same queue would freeze for all ten and
could not even show that it was thinking. So the worker PUBLISHES a snapshot after every
job and ``GET /api/state`` reads that published copy, with ``busy`` saying whether a job
is in flight. Nothing is computed on an HTTP thread.

**An automatic run is the worker's loop, not the page's.** Upstream's page runs its own
``for`` loop of ``tick`` requests, so closing the tab stops the run and a second tab
cannot see that one is going. Here auto-run is a SWITCH the worker reads between jobs
(:meth:`_Worker.set_auto`): while it is on and the run can move, the worker gives itself
one ``advance`` at a time, and anything a person posts goes ahead of the next one. The
switch is flipped from the HTTP thread without queueing, which is the point of it - a
"stop" that waited its turn behind the move it was meant to stop would not be one - and
it takes effect between moves, never inside one, because the browser is mid-action there.

Security
--------

Loopback only, and every ``/api`` request - GET included - must carry the token. The GET
matters as much as the POST here: ``/api/frame`` is a photograph of a browser that may be
carrying the captain's own logged-in profile, and a page on another origin can load an
image it is not allowed to read. So the frame is fetched with the token in a header and
turned into a blob, which also keeps the secret out of every URL, out of the address bar
and out of any log that records one.

Three more checks, each for a named attack. The ``Host`` header must be the loopback
address this server bound, which is what stops a DNS-rebinding page from reaching a
service it resolved to ``127.0.0.1``. ``Origin`` on a POST must be this server or absent.
And nothing here emits an ``Access-Control-Allow-Origin``, so a cross-origin ``fetch``
gets an opaque response even when it somehow guesses the token.

The static page is served without a token, because a URL a person pastes into a browser
cannot carry a header - it is the page that then reads the token out of its own meta tag.
A web origin cannot read that body: there is no CORS header on it, so a cross-origin
fetch of ``/`` is opaque. That is the same trade upstream makes and it holds for the same
reason.
"""

from __future__ import annotations

import errno
import json
import queue
import secrets
import socket
import threading
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from skillweaver.config import Settings
from skillweaver.contracts import Budget
from skillweaver.errors import SkillWeaverError
from skillweaver.inspector.library import SkillLibrary
from skillweaver.inspector.session import (
    LiveSession,
    SessionBusy,
    SessionError,
    run_overrides,
)
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

__all__ = ["DEFAULT_PORT", "MAX_BODY", "PORT_SPAN", "Inspector", "serve"]

STATIC = Path(__file__).parent / "static"
"""Where the page lives. Three files, served by name and nothing else."""

DEFAULT_PORT = 8767
"""The loopback port, unless ``--port`` says otherwise. One above upstream's 8766, so the
two inspectors can be open side by side, which is what comparing them takes.

Binding ``127.0.0.1`` does NOT fail when another process already holds the same port on
the wildcard address - measured on macOS, 2026-09-20, against a stray
``python -m http.server`` on ``*:8799``: both listened, and ``localhost`` then reached
whichever the resolver preferred. The printed URL names ``127.0.0.1`` for that reason and
is the one to open; a port that "works" is not proof it is this server answering.
:func:`_answering` is the check that follows from it."""

PORT_SPAN = 10
"""How many ports, counting up from the default, are tried before giving up. Upstream's
number. Only the DEFAULT falls forward: a port somebody named is a port something else -
a bookmark, a tunnel, a script - was told about, and quietly serving on its neighbour
would leave that thing talking to whoever holds the one that was asked for."""

MAX_BODY = 16384
"""Largest accepted request body. A command is a handful of short strings; an undo recipe
quoted inline is the biggest of them and is still far under this."""

_FILES = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}

_CSP = (
    "default-src 'none'; img-src 'self' blob:; script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)
"""What the page is allowed to load. Everything is same-origin and there is no inline
script, so this is tight rather than aspirational. Inline STYLE is allowed because the
target boxes over the screenshot are positioned with a ``style`` attribute; every value
that reaches one is a number computed by the page - and ``frame-ancestors 'none'`` is
what stops another page from framing the inspector and clicking its buttons."""


# --------------------------------------------------------------------------------------
# The worker that owns the browser
# --------------------------------------------------------------------------------------


class _Worker(threading.Thread):
    """The one thread that touches the session, and the only publisher of its state.

    Jobs arrive on a queue, run one at a time, and each one ends by re-publishing the
    snapshot. A job's exception is carried back to whoever posted it rather than killing
    the thread: a failed button is a message on the page, not the end of the session.
    """

    def __init__(self, session: LiveSession, library: SkillLibrary) -> None:
        super().__init__(name="skillweaver-inspector", daemon=True)
        self._session = session
        self._library = library
        self._jobs: queue.Queue[tuple[Callable[[LiveSession], None] | None, _Slot] | None] = (
            queue.Queue()
        )
        self._auto = False
        self._auto_note = ""
        self._published: dict[str, Any] = {"status": "idle", "note": "", "history": []}
        self._frame = b""
        self._frame_id = 0
        self._lock = threading.Lock()
        self._busy = False
        self._doing = ""

    # -- posting work ------------------------------------------------------------------

    def submit(self, what: str, job: Callable[[LiveSession], None], *, timeout: float) -> None:
        """Run ``job`` on the agent thread and wait for it.

        Raises:
            SessionError: whatever the job raised, or a sentence saying it is still
                running when ``timeout`` passes. The job is NOT cancelled by the
                timeout - the browser is still doing whatever it was doing - so the
                message says exactly that rather than implying a rollback.
        """
        slot = _Slot(what)
        self._jobs.put((job, slot))
        if not slot.done.wait(timeout):
            raise SessionError(
                f"'{what}' is still running after {timeout:.0f}s. The browser is still "
                "working; the page will catch up when it finishes."
            )
        if slot.error is not None:
            raise slot.error

    def set_auto(self, on: bool) -> None:
        """Switch the automatic run on or off. Safe from any thread; never waits.

        On stays on: across steps, across a finished run and into the next ``start`` or
        ``reset_run``, which then runs without a second press - upstream's checkbox. Off
        lets the move in flight finish and gives the worker no next one. The one thing
        that switches it off by itself is a move that RAISED (see :meth:`run`).
        """
        with self._lock:
            self._auto = on
            if on:
                self._auto_note = ""
        if on:
            # The loop may be asleep on an empty queue; a job that does nothing wakes it
            # to notice the switch.
            self._jobs.put((None, _Slot("wake")))

    def stop(self) -> None:
        """Close the session on its own thread and end the loop. Idempotent."""
        with self._lock:
            self._auto = False
        self._jobs.put(None)
        self.join(timeout=30)

    # -- reading state -----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The published snapshot, plus what the worker is doing and what is stored.

        Never touches the session, so a ten-second model call does not stop the page from
        rendering - which is the difference between a demo that looks alive while it
        thinks and one that looks hung.
        """
        with self._lock:
            state = dict(self._published)
            state["busy"] = self._busy
            state["doing"] = self._doing
            state["frame"] = self._frame_id
            state["auto"] = self._auto
            state["auto_note"] = self._auto_note
        state["library"] = self._library.as_json()
        return state

    def frame(self) -> bytes:
        """The current screen as PNG bytes; empty before the first observation."""
        with self._lock:
            return self._frame

    # -- the loop ----------------------------------------------------------------------

    def run(self) -> None:
        # Before the first job: an idle page should already say which perception, policy
        # and render mode it will drive, because that is what a person checks BEFORE
        # pointing it at a site that only accepts one of them.
        self._publish()
        while True:
            item = self._next()
            if item is None:
                try:
                    self._session.close()
                finally:
                    self._publish()
                return
            job, slot = item
            if job is None:
                slot.done.set()
                continue
            with self._lock:
                self._busy, self._doing = True, slot.what
            try:
                job(self._session)
            except SkillWeaverError as exc:
                slot.error = exc
            except Exception as exc:  # noqa: BLE001 - a broken button is not a dead server
                log.warning("inspect.job.failed", what=slot.what, error=str(exc))
                slot.error = SessionError(f"{slot.what} failed: {exc}")
            finally:
                if slot.what == _AUTO and slot.error is not None:
                    # Nobody is waiting on this slot, so the page is told here. And the
                    # switch goes off: a provider that is down fails the same way every
                    # time, and a loop that retries it is a bill with no run behind it.
                    with self._lock:
                        self._auto = False
                        self._auto_note = f"Auto-run stopped: {slot.error}"
                    log.warning("inspect.auto.stopped", error=str(slot.error))
                with self._lock:
                    self._busy, self._doing = False, ""
                self._publish()
                slot.done.set()

    def _next(self) -> tuple[Callable[[LiveSession], None] | None, _Slot] | None:
        """The next job: whatever was posted, else one automatic move, else wait.

        Posted work goes FIRST, so a reset or a pause pressed during an automatic run is
        the very next thing that happens and not something queued behind the rest of it.
        The switch is read here and only here, which is what makes "between moves" exact.
        """
        try:
            return self._jobs.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            auto = self._auto
        if auto and getattr(self._session, "status", "") in ("ready", "predicted"):
            return (lambda session: session.advance()), _Slot(_AUTO)
        return self._jobs.get()

    def _publish(self) -> None:
        """Snapshot the session once, on the agent thread, and hand it to the readers."""
        try:
            state = self._session.snapshot()
            png = self._session.frame_png
            frame_id = self._session.frame
        except Exception as exc:  # noqa: BLE001 - never lose the thread over a render
            log.warning("inspect.publish.failed", error=str(exc))
            return
        with self._lock:
            self._published = state
            self._frame = png
            self._frame_id = frame_id


_AUTO = "auto"
"""What the worker's own moves are called in ``doing``, as ``tick`` names a pressed one."""


class _Slot:
    """One posted job's result, waited on by the thread that posted it."""

    __slots__ = ("done", "error", "what")

    def __init__(self, what: str) -> None:
        self.what = what
        self.done = threading.Event()
        self.error: BaseException | None = None


# --------------------------------------------------------------------------------------
# The commands the buttons send
# --------------------------------------------------------------------------------------


def _command(
    name: str, body: Mapping[str, Any], settings: Settings
) -> Callable[[LiveSession], None]:
    """One request body as a job for the agent thread.

    Every argument is validated HERE, on the HTTP thread, so a typo in the prompt bar
    comes back as a message without ever reaching the browser.

    Raises:
        SessionError: for an unknown command or an unusable argument.
    """
    if name == "start":
        budget = _budget(body, settings)
        url = str(body.get("url", ""))
        goal = str(body.get("goal", ""))
        recipe = str(body.get("reset_steps", "") or "")
        read_only = bool(body.get("read_only", False))
        reset_url = str(body.get("reset_url", "") or "") or None
        steps = _recipe_text(recipe)
        overrides = run_overrides(body, settings)
        return lambda session: session.start(
            url=url,
            goal=goal,
            reset_url=reset_url,
            reset_steps=steps,
            read_only=read_only,
            budget=budget,
            overrides=overrides,
        )
    if name == "predict":
        return lambda session: session.predict()
    if name == "act":
        return lambda session: session.act()
    if name == "tick":
        return lambda session: session.tick()
    if name == "reset_run":
        return lambda session: session.reset_run()
    if name == "reset_browser":
        return lambda session: session.reset_browser()
    if name == "reset_site":
        recipe = str(body.get("recipe", "task"))
        return lambda session: session.reset_site(recipe)
    if name == "close":
        return lambda session: session.close()
    raise SessionError(f"There is no command called {name!r}.")


def _budget(body: Mapping[str, Any], settings: Settings) -> Budget:
    """The run's limits: the configured ones, with anything the prompt bar set on top.

    Configuration is the floor and a field is the override, which is the rule
    :func:`skillweaver.orchestrator.budget_from` keeps for the command line. A blank
    field is not an override.
    """
    base = settings.default_budget
    return Budget(
        max_steps=_positive(body.get("max_steps"), base.max_steps, int),
        max_seconds=_positive(body.get("max_seconds"), base.max_seconds, float),
        max_usd=_positive(body.get("max_usd"), base.max_usd, float),
        max_llm_calls=_positive(body.get("max_llm_calls"), base.max_llm_calls, int),
    )


def _positive[T: (int, float)](raw: Any, default: T, cast: type[T]) -> T:
    """One numeric override, or the default when the field was left alone.

    Raises:
        SessionError: when the field holds something that is not a positive number.
    """
    if raw is None or raw == "":
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise SessionError(f"{raw!r} is not a number.") from exc
    if value <= 0:
        raise SessionError(f"{raw!r} must be greater than zero.")
    return value


def _recipe_text(value: str) -> Any:
    """``--reset-steps`` as the prompt bar can hold it: inline JSON, or a file path.

    The same two forms the command line takes, read the same way, so an undo tried in the
    inspector is the undo the ``learn`` command would run.

    Raises:
        SessionError: if a path was given and cannot be read.
    """
    text = value.strip()
    if not text:
        return None
    if text.startswith(("[", "{")):
        return text
    path = Path(text).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SessionError(
            f"--reset-steps is inline JSON or a path to a file holding it, and {text!r} "
            f"is neither: {exc}"
        ) from exc


# --------------------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------------------


class Inspector:
    """A running inspector: a worker, a token and an HTTP server on the loopback.

    Args:
        settings: The resolved configuration. Every browser and policy choice on it is
            honoured; the inspector forces nothing, because the mode a live site accepts
            may be the only one that works.
        port: The loopback port. ``None`` means :data:`DEFAULT_PORT` or the next free
            one above it (:data:`PORT_SPAN`); a number means that port or an ``OSError``;
            ``0`` asks the OS for a free one. :attr:`url` reports whichever it was.
        step_timeout: How long a single button may take before the page is told it is
            still running. Generous by default: a cold ``predict`` on a real page is a
            model call and an observation, and OCR is most of the second one.
        session: The session to drive, for a caller that has one - a check script with
            no browser behind it. Defaults to a real :class:`LiveSession`.

    Raises:
        OSError: if the port cannot be had. ``EADDRINUSE`` also when the bind would have
            SUCCEEDED beside another listener - see :func:`_answering`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        port: int | None = None,
        step_timeout: float = 240.0,
        session: LiveSession | None = None,
    ) -> None:
        self.settings = settings
        self.token = secrets.token_urlsafe(32)
        self._step_timeout = step_timeout
        self._worker = _Worker(session or LiveSession(settings), SkillLibrary(settings.skills_dir))
        self._gate = threading.Lock()
        self._serving = False
        self._server = _bind(port, _handler(self))
        self._server.daemon_threads = True
        self.asked_port = DEFAULT_PORT if port is None else port
        self.port = self._server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.url = f"http://{self.host}"

    def serve_forever(self) -> None:
        """Run until interrupted, then close the browser. Always closes the browser."""
        self._worker.start()
        self._serving = True
        log.info("inspect.serving", url=self.url, perception=self.settings.perception)
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Stop the server and close the browser. Idempotent; never raises."""
        # ``shutdown`` waits for ``serve_forever`` to notice, and waits for ever when it
        # was never called - an inspector built and closed without serving.
        if self._serving:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass
        self._server.server_close()
        if self._worker.is_alive():
            self._worker.stop()

    # -- what a handler needs ----------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The published snapshot, plus what this invocation is configured as."""
        state = self._worker.state()
        state["server"] = {
            "url": self.url,
            "data_dir": str(self.settings.data_dir),
            "recipes": _recipes(),
            "started_at": _STARTED_AT,
        }
        return state

    def frame(self) -> bytes:
        return self._worker.frame()

    def run(self, name: str, body: Mapping[str, Any]) -> None:
        """Validate a command and run it on the agent thread.

        Raises:
            SessionError: on a bad argument, a refused step, or a step still running.
        """
        if name == "auto":
            # Not a job and not gated: it has to land WHILE a move holds the gate.
            if not isinstance(body.get("on"), bool):
                raise SessionError('auto takes {"on": true} or {"on": false}.')
            self._worker.set_auto(body["on"])
            return
        job = _command(name, body, self.settings)
        # One command at a time, refused rather than queued: a second press of Choose
        # while the first is still thinking would otherwise buy a second model call for
        # an answer nobody is going to look at.
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("A step is already running - wait for it, or pause.")
        try:
            self._worker.submit(name, job, timeout=self._step_timeout)
        finally:
            self._gate.release()


_STARTED_AT = time.time()


def _answering(port: int) -> bool:
    """Whether something already accepts connections on ``127.0.0.1:<port>``.

    Asked BEFORE binding because the bind does not answer it: ``DEFAULT_PORT`` has the
    measurement - a listener on ``*:<port>`` and ours on ``127.0.0.1:<port>`` coexist on
    macOS, so a bind that succeeded says nothing about who a browser will reach. A
    connection attempt is the question the browser is going to ask.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _bind(port: int | None, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """A server on the loopback: the named port, or the default falling forward.

    Raises:
        OSError: a named port that is taken, or a default whose whole span is.
    """
    if port == 0:
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)
    candidates = [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + PORT_SPAN)
    refusal: OSError | None = None
    for candidate in candidates:
        if _answering(candidate):
            refusal = OSError(errno.EADDRINUSE, f"something is already serving on port {candidate}")
            continue
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidate), handler)
        except OSError as exc:
            refusal = exc
    if port is None:
        raise OSError(
            errno.EADDRINUSE,
            f"ports {DEFAULT_PORT}-{DEFAULT_PORT + PORT_SPAN - 1} are all in use; "
            "free one, or name another with --port",
        )
    assert refusal is not None
    raise refusal


def _recipes() -> list[str]:
    """The undo recipes this checkout ships, offered even before a task is started.

    Read from ``undo/*.json``, which is where the cart resets proven against live sites
    live. ``task`` is offered too and means "whatever the running task declared", which
    is the undo the admission gate itself would use.
    """
    directory = Path.cwd() / "undo"
    names = ["task"]
    if directory.is_dir():
        names += sorted(path.stem for path in directory.glob("*.json"))
    return names


def _handler(inspector: Inspector) -> type[BaseHTTPRequestHandler]:
    """The request handler, closed over one inspector.

    A class rather than an instance because that is what ``ThreadingHTTPServer`` takes,
    and a closure rather than a class attribute so two inspectors in one process cannot
    end up sharing a token.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "skillweaver-inspector"
        sys_version = ""

        # -- replying ------------------------------------------------------------------

        def send(self, status: int, content: bytes | str, mime: str = "application/json") -> None:
            body = content if isinstance(content, bytes) else content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.end_headers()
            self.wfile.write(body)

        def fail(self, status: int, message: str, code: str = "") -> None:
            self.send(status, json.dumps({"error": message, "code": code}))

        def refuse(self) -> bool:
            """Answer 403 if any check fails, saying WHICH - and whether one did.

            The three used to share one sentence, and the commonest of them by far is not
            an attack: a tab left open across a restart still holds the last server's
            token, and "Local inspector requests only." told its owner nothing they could
            act on. ``stale-token`` is what the page turns into "reload". Naming the
            check gives nothing away - each is a header the caller sent and already
            knows - and no check is skipped, reordered or loosened to do it.
            """
            if not self.local():
                self.fail(
                    403,
                    f"Local requests only. Open {inspector.url} - not localhost, and "
                    "not another name for this machine.",
                    "not-local",
                )
            elif not self.carries_token():
                self.fail(
                    403,
                    "This page belongs to an inspector that has since been restarted. "
                    "Reload it to pick up the new one.",
                    "stale-token",
                )
            elif self.command == "POST" and not self.same_origin():
                self.fail(
                    403, "Commands are accepted from the inspector's own page only.", "origin"
                )
            else:
                return False
            return True

        # -- the three checks ----------------------------------------------------------

        def local(self) -> bool:
            """Whether this request reached the loopback address it claims to have.

            Checked on EVERY request, GET included: a page that resolved its own hostname
            to 127.0.0.1 gets here with somebody else's ``Host``, and that is the whole
            of a DNS-rebinding attack on a local service.
            """
            return self.headers.get("Host") == inspector.host

        def carries_token(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("X-Inspector-Token", ""), inspector.token
            )

        def same_origin(self) -> bool:
            origin = self.headers.get("Origin")
            return origin in (None, inspector.url)

        # -- GET -----------------------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            path = urlparse(self.path).path
            if path.startswith("/api/"):
                if self.refuse():
                    return None
                if path == "/api/state":
                    return self.send(200, json.dumps(inspector.state(), default=str))
                if path == "/api/frame":
                    png = inspector.frame()
                    if not png:
                        return self.fail(404, "No screen has been observed yet.")
                    return self.send(200, png, "image/png")
                return self.fail(404, "No such endpoint.")
            if not self.local():
                return self.fail(403, "Local requests only.", "not-local")
            if path == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            if path not in _FILES:
                return self.fail(404, "Not found.")
            name, mime = _FILES[path]
            text = (STATIC / name).read_text(encoding="utf-8").replace("__TOKEN__", inspector.token)
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            body = text.encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", _CSP)
            self.end_headers()
            self.wfile.write(body)

        # -- POST ----------------------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            if self.refuse():
                return None
            path = urlparse(self.path).path
            if not path.startswith("/api/"):
                return self.fail(404, "No such endpoint.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.fail(400, "A command needs a JSON body.")
            if not 0 <= length <= MAX_BODY:
                return self.fail(413, "That command is too large to be one.")
            try:
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
            except (OSError, json.JSONDecodeError):
                return self.fail(400, "A command's body must be a JSON object.")
            if not isinstance(body, dict):
                return self.fail(400, "A command's body must be a JSON object.")
            try:
                inspector.run(path.removeprefix("/api/"), body)
            except SessionBusy as exc:
                return self.fail(409, str(exc))
            except SkillWeaverError as exc:
                return self.fail(400, str(exc))
            return self.send(200, json.dumps(inspector.state(), default=str))

        def log_message(self, *_args: Any) -> None:
            """Silent: this project logs through :mod:`skillweaver.logging_`, and an
            access line per poll would bury everything the run says."""

    return Handler


def serve(settings: Settings, *, port: int | None = None, open_browser: bool = False) -> None:
    """Run an inspector until interrupted.

    Args:
        settings: The resolved configuration, honoured as-is.
        port: The loopback port: ``None`` for the default falling forward to the next
            free one, a number for exactly that port, ``0`` for one of the OS's choosing.
        open_browser: Open the page in the default browser once the server is up.

    Raises:
        OSError: if the port cannot be had.
    """
    inspector = Inspector(settings, port=port)
    if inspector.port != inspector.asked_port and inspector.asked_port:
        print(
            f"port {inspector.asked_port} is in use; serving on {inspector.port} instead.",
            flush=True,
        )
    print(f"skillweaver inspector: {inspector.url}", flush=True)
    print(
        f"  {settings.perception} perception, {settings.policy} policy, "
        f"{'headless' if settings.headless else 'headed'} browser, "
        f"library {settings.skills_dir}",
        flush=True,
    )
    if open_browser:
        import webbrowser

        webbrowser.open(inspector.url)
    inspector.serve_forever()
