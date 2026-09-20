"""Start the real Google Chrome as an ORDINARY PROCESS and hand back its debugging address.

This module owns one OS process: it finds Chrome, starts it the way a person's launcher
does, waits for the debugging interface, proves that interface belongs to the process it
started, and kills that process afterwards. Attaching is somebody else's job - any caller
can read :attr:`ChromeProcess.endpoint`.

Read :data:`PLAINLY_LAUNCHED` before changing anything here.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from urllib.parse import urlparse

from skillweaver.errors import ControllerError

PLAINLY_LAUNCHED = "plainly-launched Chrome, attached over CDP"
"""The third launch configuration this project has, and the only one a live DoorDash
serves. What it changes is WHO STARTED THE BROWSER.

Measured by hand 2026-09-19 against live doordash.com, same machine and network, minutes
apart, signed out: framework-launched Chromium REFUSED; framework-launched real Chrome
(``REAL_CHROME_CHANNEL`` + ``launch_persistent_context``) REFUSED, 0 of 6 loads and 0 of 5
on a brand-new profile; plainly launched real Chrome attached over the debugging channel
LOADED 6 of 6, including a 32,596-character city listing and four store menus - the same
navigation that scored 0 of 5 the other way. The user's everyday Chrome loads the site,
which ruled out the network and the address.

So the variable is not the binary, the profile, the IP or the debugging channel, all of
which the three modes can share: it is the flags the framework adds when IT starts the
browser. Mechanism, measured on Chrome 153: ``navigator.webdriver`` is TRUE in both
framework-launched modes and FALSE here, because ``--enable-automation`` is what sets it
and an ordinary process does not carry it. The page is reading the browser correctly in
all three cases, which is why the other two cannot be fixed by adding a setting.

That is the line. NOT PASSING A FLAG OF OUR OWN is allowed; CONTRADICTING THE BROWSER -
rewriting ``navigator.webdriver``, a spoofed user agent, a patched fingerprint - never is,
however similar the resulting page looks. Nothing here defeats, masks or retries past a
human-verification page: a challenge FAILS the run and a person clears it by hand, once,
in the profile. Do not extend it into one: the flags below are the job's own.

Headless has its own tells (the user agent still says ``HeadlessChrome``), which is why
the measurement was taken HEADED. And do not describe this as making a site accept us; it
makes us STOP ANNOUNCING OURSELVES, and a site can change its mind - the second row above
was measured working earlier the same day.
"""

CHROME_BINARIES: dict[str, tuple[str, ...]] = {
    "Darwin": ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",),
    "Linux": (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/opt/google/chrome/chrome",
    ),
    "Windows": (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ),
}
"""Where the real Google Chrome lives, per platform, first match wins. Deliberately not
Playwright's bundled Chromium: that is the build the first row of :data:`PLAINLY_LAUNCHED`
measures being refused, so a fallback would trade a loud failure for a quiet block."""

_ANNOUNCEMENT = re.compile(r"DevTools listening on (ws://\S+)")
"""Chrome telling ITS OWN STDERR where its debugging interface is. On Chrome
153.0.8010.50 the line is printed on every start, headed and headless, and carries the
port and the browser's unique WebSocket path.

Chrome 153 does NOT write :data:`_PORT_FILE` in either render mode, so a launch that waits
for that file waits out its whole timeout beside a healthy browser; it is read where it
exists but never waited for. Either source is an IDENTITY as well as an address (see
:meth:`ChromeProcess._confirm`) and both are private to THIS run."""

_PORT_FILE = "DevToolsActivePort"
"""Port on line one, the browser's unique WebSocket path on line two. Read where it
exists, never waited for; see :data:`_ANNOUNCEMENT`."""

_STARTUP_TIMEOUT_S = 30.0
"""How long Chrome gets to publish its debugging address; a cold profile on a busy machine
is the slow case. An already-exited process is noticed at once rather than waited out."""

_POLL_S = 0.05
_LAUNCH_ATTEMPTS = 3
"""Ports are picked, not reserved, so another process can take one in the gap. Chrome then
refuses to start rather than share, which shows up here as an exited process."""

_TERMINATE_GRACE_S = 5.0
"""How long a terminated Chrome gets to exit on its own before it is killed outright."""


def chrome_executable(explicit: str | Path | None = None) -> Path:
    """The real Google Chrome on this machine.

    Args:
        explicit: A path to use instead of searching, checked and used as-is.

    Raises:
        ControllerError: there is no Chrome to run, naming the paths tried. There is no
            fallback to bundled Chromium; see :data:`CHROME_BINARIES`.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ControllerError(f"no Chrome executable at {path}")
        return path
    candidates = CHROME_BINARIES.get(platform.system(), ())
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    tried = ", ".join(candidates) or "nothing - this platform has no known location"
    raise ControllerError(
        f"could not find Google Chrome on this {platform.system()} machine (tried: "
        f"{tried}). Install Chrome, or pass its path explicitly; there is no fallback "
        f"to bundled Chromium, which is the build a bot wall refuses."
    )


def _free_port() -> int:
    """A port nothing is listening on right now.

    Picked, not reserved: the socket closes before Chrome starts, because Chrome has to
    bind it itself. The gap is why :data:`_LAUNCH_ATTEMPTS` exists.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ChromeProcess:
    """A real Chrome started as an ordinary process, with its debugging port open.

    The process is OWNED: it is killed on :meth:`close`, and the only pid this class will
    ever signal is the one its own ``Popen`` returned. Use it as a context manager.

    Args:
        user_data_dir: The profile directory, created if missing and left on disk. **One
            directory per run**: a profile is exclusive, and concurrent runs sharing one
            fight over the lock and spoil the stored state that is the point of keeping it.
        headless: The :data:`PLAINLY_LAUNCHED` measurement was taken HEADED, and headless
            is a different screen - see :mod:`skillweaver.render_mode`.
        binary: Chrome's path, or ``None`` to search.
        device_scale_factor: Applied with ``--force-device-scale-factor``; ``1.0`` passes
            no flag and lets the display's own scale stand.
        window: ``(width, height)`` for the window Chrome opens. The page's viewport is
            the attaching caller's business.
        extra_args: Further arguments. Read the boundary in :data:`PLAINLY_LAUNCHED`
            first: this is not a door for masking arguments.
        startup_timeout_s: How long Chrome gets to open its debugging port.

    Raises:
        ControllerError: Chrome cannot be found or started, never opens its debugging
            port, or opens one that proves to belong to a different browser.
    """

    def __init__(
        self,
        *,
        user_data_dir: str | Path,
        headless: bool = False,
        binary: str | Path | None = None,
        device_scale_factor: float = 1.0,
        window: tuple[int, int] | None = None,
        extra_args: tuple[str, ...] = (),
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
    ) -> None:
        self._profile = Path(user_data_dir).expanduser()
        self._executable = chrome_executable(binary)
        self._proc: subprocess.Popen[bytes] | None = None
        self._endpoint: str | None = None
        self._port: int | None = None
        self._log = self._profile / "skillweaver-chrome.log"

        try:
            self._profile.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ControllerError(
                f"could not create Chrome profile {self._profile}: {exc}"
            ) from exc

        last: str = ""
        for attempt in range(1, _LAUNCH_ATTEMPTS + 1):
            port = _free_port()
            self._spawn(port, headless, device_scale_factor, window, extra_args)
            try:
                actual_port, token = self._await_debug_address(startup_timeout_s)
            except ControllerError as exc:
                self.close()
                last = str(exc)
                if attempt < _LAUNCH_ATTEMPTS and self._raced_for_the_port(exc):
                    continue
                raise ControllerError(last) from exc
            self._confirm(actual_port, token)
            self._port = actual_port
            self._endpoint = f"http://127.0.0.1:{actual_port}"
            return
        raise ControllerError(last)  # pragma: no cover - the loop returns or raises

    # -- what a caller needs -------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """``http://127.0.0.1:<port>``, the address to attach to. Any caller may."""
        if self._endpoint is None:  # pragma: no cover - __init__ returns or raises
            raise ControllerError("Chrome is not running")
        return self._endpoint

    @property
    def port(self) -> int:
        """The port Chrome ACTUALLY bound, read back rather than assumed."""
        if self._port is None:  # pragma: no cover - __init__ returns or raises
            raise ControllerError("Chrome is not running")
        return self._port

    @property
    def profile_dir(self) -> Path:
        """The directory this Chrome runs out of. Left on disk by :meth:`close`."""
        return self._profile

    @property
    def pid(self) -> int | None:
        """The process this class started, or ``None`` once it is gone."""
        return None if self._proc is None else self._proc.pid

    def is_running(self) -> bool:
        """Whether the Chrome this class started is still alive."""
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> ChromeProcess:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Kill the Chrome this class started. Idempotent, and never raises.

        The whole process GROUP goes, which is safe precisely because :meth:`_spawn` put
        Chrome in a session of its own. The profile directory is left on disk: the
        clearance a person granted by hand lives in it.
        """
        proc, self._proc = self._proc, None
        self._endpoint = None
        self._port = None
        if proc is None or proc.poll() is not None:
            if proc is not None:
                with contextlib.suppress(Exception):
                    proc.wait(timeout=0.1)
            return
        self._signal(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=_TERMINATE_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        self._signal(proc, signal.SIGKILL)
        with contextlib.suppress(Exception):  # noqa: BLE001 - teardown must never raise
            proc.wait(timeout=_TERMINATE_GRACE_S)

    # -- internals -----------------------------------------------------------------------

    def _spawn(
        self,
        port: int,
        headless: bool,
        device_scale_factor: float,
        window: tuple[int, int] | None,
        extra_args: tuple[str, ...],
    ) -> None:
        """Start Chrome the way a launcher does, with as few flags as the job needs."""
        # Stale from a previous run or a crash; removing it is what makes "the file
        # exists" mean "this Chrome is listening".
        with contextlib.suppress(OSError):
            (self._profile / _PORT_FILE).unlink()

        args = [
            str(self._executable),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self._profile}",
            "--no-first-run",
            "--no-default-browser-check",
            # Chromium animates wheel scrolling; a capture mid-animation is half-scrolled.
            "--disable-smooth-scrolling",
        ]
        if headless:
            args.append("--headless=new")
        if abs(device_scale_factor - 1.0) > 1e-9:
            args.append(f"--force-device-scale-factor={device_scale_factor:g}")
        if window is not None:
            args.append(f"--window-size={int(window[0])},{int(window[1])}")
        args.extend(extra_args)

        try:
            log = self._log.open("wb")
        except OSError:  # pragma: no cover - an unwritable profile fails at mkdir first
            log = None
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - a path this module resolved
                args,
                stdout=subprocess.DEVNULL,
                stderr=log or subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                # Its own session, so close() takes the tree down without reaching
                # anything this class did not start.
                start_new_session=True,
            )
        except OSError as exc:
            raise ControllerError(f"could not start Chrome at {self._executable}: {exc}") from exc
        finally:
            if log is not None:
                log.close()

    def _await_debug_address(self, timeout_s: float) -> tuple[int, str]:
        """``(port, websocket path)`` as Chrome itself announced them.

        Raises:
            ControllerError: Chrome exited, or never announced an address.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ControllerError(
                    f"Chrome exited with status {self._proc.returncode} before opening its "
                    f"debugging port{self._log_tail()}"
                )
            published = self._published_address()
            if published is not None:
                return published
            time.sleep(_POLL_S)
        raise ControllerError(
            f"Chrome did not open its debugging port within {timeout_s:g}s{self._log_tail()}"
        )

    def _published_address(self) -> tuple[int, str] | None:
        """The address Chrome has published so far, from either source, or ``None``."""
        try:
            said = self._log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            said = ""
        found = _ANNOUNCEMENT.search(said)
        if found is not None:
            parsed = urlparse(found.group(1))
            if parsed.port:
                return int(parsed.port), parsed.path
        path = self._profile / _PORT_FILE
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        if len(lines) >= 2 and lines[0].strip().isdigit():
            return int(lines[0].strip()), lines[1].strip()
        return None

    def _confirm(self, port: int, token: str) -> None:
        """Prove the browser listening on ``port`` is the one this class started.

        Attaching to the wrong Chrome would drive somebody's real session, so it is made
        impossible rather than unlikely by two facts a stranger's browser cannot both
        satisfy: the port came out of the mouth of the process this class started, and the
        endpoint's ``/json/version`` WebSocket path - unique per browser instance - must be
        the one that process announced.

        Raises:
            ControllerError: the endpoint cannot be read, or names another browser.
        """
        url = f"http://127.0.0.1:{port}/json/version"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - literal
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.close()
            raise ControllerError(
                f"Chrome published port {port} but its debugging interface did not "
                f"answer at {url}: {exc}"
            ) from exc
        socket_url = str(payload.get("webSocketDebuggerUrl", ""))
        if urlparse(socket_url).path != token:
            self.close()
            raise ControllerError(
                f"the browser answering on port {port} is not the one this run started "
                f"(it identifies as {socket_url!r}, and the Chrome started here "
                f"announced {token!r}). Refusing to attach rather than drive somebody else's "
                f"browser."
            )

    def _raced_for_the_port(self, exc: ControllerError) -> bool:
        """Whether a failed start looks like the picked port having been taken.

        Chrome refuses to share a port, so an immediate exit is what losing the race looks
        like. A timeout is not: Chrome is alive, and another port would not help.
        """
        return "exited with status" in str(exc)

    def _signal(self, proc: subprocess.Popen[bytes], sig: int) -> None:
        """Signal the process group this class started, or the process if it has none."""
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):  # noqa: BLE001 - teardown must not raise
                proc.send_signal(sig)

    def _log_tail(self, limit: int = 400) -> str:
        """The end of Chrome's own stderr, for a failure message that can be acted on."""
        try:
            text = self._log.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        return f": {text[-limit:]}" if text else ""
