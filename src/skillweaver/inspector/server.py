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

import json
import queue
import secrets
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
from skillweaver.inspector.session import LiveSession, SessionBusy, SessionError
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

__all__ = ["DEFAULT_PORT", "MAX_BODY", "Inspector", "serve"]

STATIC = Path(__file__).parent / "static"
"""Where the page lives. Three files, served by name and nothing else."""

DEFAULT_PORT = 8767
"""The loopback port, unless ``--port`` says otherwise. One above upstream's 8766, so the
two inspectors can be open side by side, which is what comparing them takes.

Binding ``127.0.0.1`` does NOT fail when another process already holds the same port on
the wildcard address - measured on macOS, 2026-09-20, against a stray
``python -m http.server`` on ``*:8799``: both listened, and ``localhost`` then reached
whichever the resolver preferred. The printed URL names ``127.0.0.1`` for that reason and
is the one to open; a port that "works" is not proof it is this server answering."""

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
        self._jobs: queue.Queue[tuple[Callable[[LiveSession], None], _Slot] | None] = queue.Queue()
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

    def stop(self) -> None:
        """Close the session on its own thread and end the loop. Idempotent."""
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
            item = self._jobs.get()
            if item is None:
                try:
                    self._session.close()
                finally:
                    self._publish()
                return
            job, slot = item
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
                with self._lock:
                    self._busy, self._doing = False, ""
                self._publish()
                slot.done.set()

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
        return lambda session: session.start(
            url=url,
            goal=goal,
            reset_url=reset_url,
            reset_steps=steps,
            read_only=read_only,
            budget=budget,
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
        port: The loopback port. ``0`` asks the OS for a free one, which
            :attr:`url` then reports.
        step_timeout: How long a single button may take before the page is told it is
            still running. Generous by default: a cold ``predict`` on a real page is a
            model call and an observation, and OCR is most of the second one.
    """

    def __init__(
        self, settings: Settings, *, port: int = DEFAULT_PORT, step_timeout: float = 240.0
    ) -> None:
        self.settings = settings
        self.token = secrets.token_urlsafe(32)
        self._step_timeout = step_timeout
        self._worker = _Worker(LiveSession(settings), SkillLibrary(settings.skills_dir))
        self._gate = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.url = f"http://{self.host}"

    def serve_forever(self) -> None:
        """Run until interrupted, then close the browser. Always closes the browser."""
        self._worker.start()
        log.info("inspect.serving", url=self.url, perception=self.settings.perception)
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Stop the server and close the browser. Idempotent; never raises."""
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

        def fail(self, status: int, message: str) -> None:
            self.send(status, json.dumps({"error": message}))

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
            if not self.local():
                return self.fail(403, "Local requests only.")
            path = urlparse(self.path).path
            if path.startswith("/api/"):
                if not self.carries_token():
                    return self.fail(403, "This inspector needs its own page's token.")
                if path == "/api/state":
                    return self.send(200, json.dumps(inspector.state(), default=str))
                if path == "/api/frame":
                    png = inspector.frame()
                    if not png:
                        return self.fail(404, "No screen has been observed yet.")
                    return self.send(200, png, "image/png")
                return self.fail(404, "No such endpoint.")
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
            if not self.local() or not self.carries_token() or not self.same_origin():
                return self.fail(403, "Local inspector requests only.")
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


def serve(settings: Settings, *, port: int = DEFAULT_PORT, open_browser: bool = False) -> None:
    """Run an inspector until interrupted.

    Args:
        settings: The resolved configuration, honoured as-is.
        port: The loopback port; ``0`` asks the OS for a free one.
        open_browser: Open the page in the default browser once the server is up.
    """
    inspector = Inspector(settings, port=port)
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
