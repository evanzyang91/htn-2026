"""Checks for ``HarnessBrowserController``: the first document, Enter, and the frame.

Needs a local Chrome and NO network. It starts a Chrome of its own out of a temporary
profile (``ChromeProcess``) and points Browser Harness at it with ``BU_CDP_URL`` and a
``BU_NAME`` of its own, so the person's running browser is never touched; the pages are
served from ``127.0.0.1`` by this process. Headless unless ``CHECK_D_HEADED`` is set.

    uv run python scripts/check_d_harness.py

One line per check; exits non-zero if any failed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from skillweaver.contracts import Back, Click, Navigate, Point, PressKey, TypeText
from skillweaver.controllers import _coords
from skillweaver.controllers.chrome_launch import ChromeProcess

_SLOW_S = 0.4

_FORM = b"""<!doctype html><html><body style="margin:0">
<form action="/submitted" method="get">
<input name="q" autofocus style="position:absolute;left:20px;top:20px;width:300px;height:30px">
</form></body></html>"""

_PAGES: dict[str, bytes] = {
    "/slow": b"<!doctype html><html><body><h1>slow page</h1></body></html>",
    "/form": _FORM,
    "/still": b"<!doctype html><html><body><h1>still</h1><p>nothing moves here</p></body></html>",
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        path = self.path.split("?", 1)[0]
        if path == "/slow":
            time.sleep(_SLOW_S)
        body = _PAGES.get(path, f"<html><body><h1>{self.path}</h1></body></html>".encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail != '' else ''}")
    if not ok:
        _failures.append(name)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    with ChromeProcess(
        user_data_dir=tempfile.mkdtemp(prefix="check-d-chrome-"),
        headless=os.environ.get("CHECK_D_HEADED") is None,
    ) as chrome:
        # Before the import: the harness reads both when its modules load.
        os.environ["BU_CDP_URL"] = chrome.endpoint
        os.environ["BU_NAME"] = f"checkd{os.getpid()}"
        from skillweaver.controllers.harness import HarnessBrowserController

        # 1. The first read after construction is the page asked for, never about:blank.
        firsts = []
        for _ in range(5):
            with HarnessBrowserController(start_url=f"{base}/slow") as ctl:
                firsts.append(ctl.evaluate("() => location.href + ' ' + document.readyState"))
        check(
            "first read after construction is the start_url, complete",
            all(found == f"{base}/slow complete" for found in firsts),
            sorted(set(firsts)),
        )

        with HarnessBrowserController(start_url=f"{base}/form") as ctl:
            # A navigation AIMED at about:blank must not wait out its timeout for it.
            started = time.perf_counter()
            result = ctl.perform(Navigate("about:blank"))
            took = time.perf_counter() - started
            check("navigate to about:blank is prompt", result.ok and took < 5.0, f"{took:.2f}s")
            result = ctl.perform(Navigate(f"{base}/form"))
            check("navigate away from about:blank", result.ok, result.error or "")

            # 2. Enter submits a form that has no button.
            ctl.perform(Click(Point(100, 35)))
            ctl.perform(TypeText("keycaps"))
            result = ctl.perform(PressKey(("Enter",)))
            href = ctl.evaluate("() => location.href")
            check(
                "Enter submits a button-less form",
                result.ok and href == f"{base}/submitted?q=keycaps",
                href,
            )

            # Back is not asked to leave about:blank, and lands on the page behind.
            result = ctl.perform(Back())
            href = ctl.evaluate("() => location.href")
            check("Back returns to the form", result.ok and href == f"{base}/form", href)

        # 3. The frame: a valid PNG of the viewport, and the same bytes for the same pixels.
        with HarnessBrowserController(viewport=(1024, 640), start_url=f"{base}/still") as ctl:
            time.sleep(0.5)
            one, two = ctl.capture(), ctl.capture()
            check("capture is a PNG", one.png[:8] == b"\x89PNG\r\n\x1a\n")
            size = _coords.png_size(one.png)
            check("capture is the viewport, 1x", size == (1024, 640) and one.scale == 1.0, size)
            check(
                "two captures of a still page are the same bytes (the OCR cache key)",
                one.png == two.png,
                f"{len(one.png)} and {len(two.png)} bytes",
            )

    server.shutdown()
    print(f"{len(_failures)} failed" if _failures else "all checks passed")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
