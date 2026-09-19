"""Start the real Google Chrome as an ORDINARY PROCESS and hand back its debugging address.

This module owns one OS process and nothing else. It knows how to find Chrome on this
machine, start it the way a person's launcher does, wait until its debugging interface is
up, prove that the interface belongs to the process it started, and kill that process
afterwards. Attaching a browser automation framework to the address it returns is
somebody else's job - :class:`~skillweaver.controllers.browser.BrowserController` does it
for the pixel path, and any other caller that wants the same browser can read
:attr:`ChromeProcess.endpoint` and connect to it too.

Read :data:`PLAINLY_LAUNCHED` before changing anything here: it carries the measurement
that says why this mode exists, and the boundary that says what it must never become.
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
"""The third browser launch configuration this project has, and the only one a live
DoorDash serves. What it changes is WHO STARTED THE BROWSER.

Measured by hand on 2026-09-19 against live doordash.com, same machine, same network,
within minutes of each other, signed out:

=================================================================  ==================
how the browser was started                                        result
=================================================================  ==================
framework-launched Chromium, fresh profile                         refused
framework-launched real Chrome (``channel="chrome"`` +             refused; 0 of 6
``launch_persistent_context``) - see ``REAL_CHROME_CHANNEL``       loads, and 0 of 5
                                                                   from a brand-new
                                                                   profile
**plainly launched real Chrome, attached over the debugging        **6 of 6 loads
channel**                                                          real**
=================================================================  ==================

The working run was a city listing page of 32,596 characters, four different store pages
with real priced menus, and a return to the listings - the SAME navigation pattern that
scored 0 of 5 the framework-launched way. The user's own everyday Chrome loads the site
fine, which is what ruled out the network and the address.

So the variable is not the Chrome binary, not the profile, not the IP and not the
debugging channel - all three configurations can have any of those. It is THE FLAGS THE
AUTOMATION FRAMEWORK ADDS WHEN IT STARTS THE BROWSER. Started as an ordinary process,
Chrome carries none of them.

**The mechanism, measured, because a reader will otherwise guess at it.** ``navigator.
webdriver`` is TRUE in both framework-launched modes and FALSE here (Chrome 153, all
three modes, same machine). Nothing in this module touches that flag. It is set by
``--enable-automation``, which the automation framework adds when IT starts the browser
and which a browser started as an ordinary process simply does not carry - the page is
reading the browser correctly in all three cases. That is the whole of the difference,
and it is why the other two modes cannot be fixed by adding a setting to them: what a
site is reading is a property of the launch, not of the driving.

This is the line, and it is a real one. REMOVING OUR OWN ANNOUNCEMENT is not the same
act as CONTRADICTING THE BROWSER: a launch flag we do not pass is a flag we do not pass,
while a page script that rewrites ``navigator.webdriver``, a spoofed user agent or a
patched fingerprint would be telling the site something untrue. The first is allowed
here and the second never is, however similar the resulting page looks.

Headless is not free of tells either, and this mode does not hide those: the user agent
still says ``HeadlessChrome`` in headless mode, which is why the measurement below was
taken HEADED and why a live run should be.

**What this is, stated plainly, because a future reader must not mistake it for a trick.**
It starts an ordinary browser and drives it through Chrome's own documented remote
debugging interface - the same interface Chrome DevTools itself uses, enabled by a
documented command-line switch. **It does not defeat, mask, auto-solve or retry past a
human-verification page, and nothing built on it may.** A challenge FAILS the run and a
person clears it by hand, once, in the profile directory that outlives the run.

**Do not extend it into one.** No ``--disable-blink-features=AutomationControlled``, no
spoofed fingerprint, no user-agent override, no proxy, no retry-until-it-passes loop. The
flags below are the job's own: a debugging port, a profile directory, the two
first-run suppressions any scripted launch needs, a window size, and the scroll and
headless settings every other mode in this project already sets.

And do not describe this mode as making a site accept us. It makes us STOP ANNOUNCING
OURSELVES, which is a different and much smaller claim. Whether a site then serves an
ordinary browser is the site's business, and it can change its mind - the second row of
that table was measured working earlier the same day.
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
"""Where the real Google Chrome lives, per platform, first match wins.

Deliberately not Playwright's bundled Chromium: that is the build the first row of
:data:`PLAINLY_LAUNCHED` measures being refused, so falling back to it would trade a
loud failure for a run that is quietly blocked.
"""

_ANNOUNCEMENT = re.compile(r"DevTools listening on (ws://\S+)")
"""Chrome telling ITS OWN STDERR where its debugging interface is, which is where the
address comes from. Measured on Chrome 153.0.8010.50: the line is printed on every
start, headed and headless alike, and carries the port and the browser's unique
WebSocket path in one string.

The older way to learn this was :data:`_PORT_FILE`, and it is still read where it
exists, but it cannot be relied on alone: Chrome 153 does NOT write that file, in either
render mode, and a launch that waits for it waits out its whole timeout while a perfectly
healthy browser sits there serving.

Either source is an identity as well as an address, which is the point - see
:meth:`ChromeProcess._confirm`. Both are private to THIS run: the stderr of the process
this class started, and a file inside the profile directory this run owns and emptied of
it before launching."""

_PORT_FILE = "DevToolsActivePort"
"""What Chrome used to write into its own profile directory: the port on line one, the
browser's unique WebSocket path on line two. Read when it is there, not waited for; see
:data:`_ANNOUNCEMENT`."""

_STARTUP_TIMEOUT_S = 30.0
"""How long Chrome gets to publish its debugging address before the launch is called
failed. A cold profile on a busy machine is the slow case; a process that has already
exited is noticed immediately rather than waited out."""

_POLL_S = 0.05
_LAUNCH_ATTEMPTS = 3
"""Ports are picked, not reserved, so another process can take one in the gap between
picking and launching. Chrome then refuses to start rather than sharing, which this
module sees as an exited process and answers by picking another port."""

_TERMINATE_GRACE_S = 5.0
"""How long a terminated Chrome gets to exit on its own before it is killed outright."""


def chrome_executable(explicit: str | Path | None = None) -> Path:
    """The real Google Chrome on this machine.

    Args:
        explicit: A path to use instead of searching. Given, it is checked and used
            as-is, which is how a machine with Chrome somewhere unusual, or a test
            standing a fake process in its place, says so.

    Returns:
        The executable's path.

    Raises:
        ControllerError: if there is no Chrome to run, naming the paths that were tried.
            There is no fallback to bundled Chromium on purpose; see
            :data:`CHROME_BINARIES`.
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
    """A port nothing is listening on, right now.

    Picked rather than reserved: the socket is closed before Chrome is started, because
    Chrome has to bind it itself. The gap is why :data:`_LAUNCH_ATTEMPTS` exists.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ChromeProcess:
    """A real Chrome started as an ordinary process, with its debugging port open.

    The process is OWNED: it is killed on :meth:`close`, and nothing else on the machine
    is ever killed, because the only pid this class will signal is the one it got back
    from its own ``Popen``. Use it as a context manager so a raising body still tears the
    browser down::

        with ChromeProcess(user_data_dir=fresh_dir) as chrome:
            browser = playwright.chromium.connect_over_cdp(chrome.endpoint)

    Args:
        user_data_dir: The profile directory Chrome runs out of, created if missing and
            left on disk afterwards. **One directory per run**: a profile is exclusive,
            and concurrent runs sharing one fight over the lock and spoil the stored
            state that is the whole reason for keeping it.
        headless: Start with no visible window. The measurement in
            :data:`PLAINLY_LAUNCHED` was taken HEADED; headless is a different screen,
            which :mod:`skillweaver.render_mode` is what says so.
        binary: Chrome's path, or ``None`` to search - see :func:`chrome_executable`.
        device_scale_factor: Physical pixels per logical pixel, applied with Chrome's own
            ``--force-device-scale-factor``. ``1.0`` passes no flag at all and lets the
            display's own scale stand, which a capture measures and reports either way.
        window: ``(width, height)`` for the window Chrome opens, so a headed run is not
            an awkward shape on screen. The page's viewport is the attaching caller's
            business, not this class's.
        extra_args: Further command-line arguments. Read the boundary in
            :data:`PLAINLY_LAUNCHED` before adding one: this is not a door for masking
            arguments.
        startup_timeout_s: How long Chrome gets to open its debugging port.

    Raises:
        ControllerError: if Chrome cannot be found, cannot be started, never opens its
            debugging port, or opens one that proves to belong to a different browser.
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
        """``http://127.0.0.1:<port>``, the address to attach to.

        Any caller can use it, not only the pixel controller: a policy that reads the DOM
        over CDP attaches to this same browser the same way.
        """
        if self._endpoint is None:  # pragma: no cover - __init__ returns or raises
            raise ControllerError("Chrome is not running")
        return self._endpoint

    @property
    def port(self) -> int:
        """The port Chrome ACTUALLY bound, read back from its own :data:`_PORT_FILE`
        rather than assumed from what it was asked for."""
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

        Chrome spawns a tree of helper processes, so the whole process GROUP goes - which
        is safe precisely because :meth:`_spawn` put Chrome in a session of its own, so
        that group contains nothing but the browser this class started. A Chrome this
        class did not start is never signalled.

        The profile directory is left on disk: the clearance a person granted by hand
        lives in it, and deleting it would put the next run back behind the wall.
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
        # Stale from a previous run of the same directory, or from a crash. Removing it
        # is what makes "the file exists" mean "this Chrome is listening".
        with contextlib.suppress(OSError):
            (self._profile / _PORT_FILE).unlink()

        args = [
            str(self._executable),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self._profile}",
            "--no-first-run",
            "--no-default-browser-check",
            # Chromium animates wheel scrolling, exactly as it does in the other modes,
            # and a capture taken mid-animation is a half-scrolled frame.
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
                # Its own session, so close() can take the whole process tree down
                # without reaching anything this class did not start.
                start_new_session=True,
            )
        except OSError as exc:
            raise ControllerError(f"could not start Chrome at {self._executable}: {exc}") from exc
        finally:
            if log is not None:
                log.close()

    def _await_debug_address(self, timeout_s: float) -> tuple[int, str]:
        """Wait for Chrome to publish its debugging address, and read it back.

        Returns:
            ``(port, websocket path)`` as Chrome itself announced them.

        Raises:
            ControllerError: if Chrome exits, or never announces an address.
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

        Attaching to the wrong Chrome is the worst failure available here - a run would
        drive somebody's real browsing session - so it is made impossible rather than
        unlikely, by two facts that a stranger's browser cannot both satisfy. The port
        came out of the mouth of the process this class started - its own stderr, or a
        file inside this run's own profile directory that was emptied before launching -
        and the endpoint is then asked for its identity, whose WebSocket path is unique
        per browser instance and must be the one that process announced.

        Raises:
            ControllerError: if the endpoint cannot be read or names another browser.
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

        Chrome refuses to start rather than share a port, so an immediate exit is what
        losing the race looks like from here. A timeout is not: Chrome is alive and
        simply slow or wedged, and picking another port would not help.
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
