"""A one-box web UI that starts the agent.

Standard library plus what the project already ships (``pillow``), on purpose:
``pyproject.toml`` is shared surface, so this adds no dependency. The launcher page is
a dark screen with one text box. Submitting it runs the real command line -
``python -m skillweaver.cli run <task>`` - as a subprocess out of the repository root,
and opens a SECOND TAB for that run: the agent's own browser window, mirrored live,
with the command's output streaming beneath it. The mirror is a screen capture of the
headed Chromium the controller opens, found by walking the run's process tree, so
nothing about the run is different from one typed into a terminal.

    uv run python apps/webui/serve.py                # http://127.0.0.1:8765
    uv run python apps/webui/serve.py --port 9000 --no-open
    uv run python apps/webui/serve.py -- --perception dom   # flags handed to every run

A URL typed into the box (``add 2 apples to the cart on http://localhost:5173``) is
lifted out and passed as ``--url``; everything else is the task, verbatim. With no URL
a SIDE AGENT (:func:`pick_site`, one small model call) chooses the website from the
task and its choice is printed at the top of the run; when it cannot choose, the run
goes ahead without ``--url`` and the library resolves the domain itself.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATIC = HERE / "static"

_URL = re.compile(r"https?://\S+")
_TRAILING_PREP = re.compile(r"\s+(on|at|in|from|using)\s*$", re.IGNORECASE)


def split_task(text: str) -> tuple[str, str | None]:
    """Lift the first URL out of the sentence; the rest is the task."""
    match = _URL.search(text)
    if not match:
        return text.strip(), None
    url = match.group(0).rstrip(".,;)")
    task = (text[: match.start()] + text[match.end() :]).strip()
    task = _TRAILING_PREP.sub("", task).strip()
    return task or text.strip(), url


def command_for(task: str, url: str | None, extra: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "skillweaver.cli", *extra, "run", task]
    if url:
        cmd += ["--url", url]
    return cmd


# --------------------------------------------------------------------------------------
# Mirroring the agent's browser window (Windows only; elsewhere the tab shows the log)
# --------------------------------------------------------------------------------------


def _descendants(root_pid: int) -> set[int]:
    """Every process under ``root_pid``, via a Toolhelp snapshot."""
    if sys.platform != "win32":
        return set()
    k32 = ctypes.windll.kernel32

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_char * 260),
        ]

    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    parents: dict[int, int] = {}
    try:
        e = Entry()
        e.dwSize = ctypes.sizeof(Entry)
        if k32.Process32First(snap, ctypes.byref(e)):
            while True:
                parents[e.th32ProcessID] = e.th32ParentProcessID
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    break
    finally:
        k32.CloseHandle(snap)
    found = {root_pid}
    grew = True
    while grew:
        grew = False
        for pid, parent in parents.items():
            if parent in found and pid not in found:
                found.add(pid)
                grew = True
    return found


def _browser_window(pids: set[int]) -> int | None:
    """Handle of the largest visible top-level window owned by one of ``pids``."""
    if sys.platform != "win32":
        return None
    u32 = ctypes.windll.user32
    best: tuple[int, int] | None = None

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd: int, _: int) -> bool:
        nonlocal best
        if not u32.IsWindowVisible(hwnd) or u32.IsIconic(hwnd):
            return True
        pid = ctypes.c_ulong()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in pids:
            return True
        rect = _Rect()
        u32.GetWindowRect(hwnd, ctypes.byref(rect))
        w, h = rect.r - rect.l, rect.b - rect.t
        if w < 200 or h < 200:
            return True
        if best is None or w * h > best[0]:
            best = (w * h, hwnd)
        return True

    u32.EnumWindows(visit, 0)
    return best[1] if best else None


class _Rect(ctypes.Structure):
    _fields_ = [
        ("l", ctypes.c_long),
        ("t", ctypes.c_long),
        ("r", ctypes.c_long),
        ("b", ctypes.c_long),
    ]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


def _print_window(hwnd: int) -> tuple[int, int, bytes] | None:
    """The window's own pixels through ``PrintWindow``, so a window on top of it
    - the user's browser, this very page - does not end up in the mirror. A plain
    screen grab of the window's rectangle did exactly that."""
    u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
    u32.GetWindowDC.restype = ctypes.c_void_p
    g32.CreateCompatibleDC.restype = ctypes.c_void_p
    g32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    g32.SelectObject.restype = ctypes.c_void_p
    rect = _Rect()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.r - rect.l, rect.b - rect.t
    if w <= 0 or h <= 0:
        return None
    hwnd_dc = u32.GetWindowDC(ctypes.c_void_p(hwnd))
    mem_dc = g32.CreateCompatibleDC(ctypes.c_void_p(hwnd_dc))
    bmp = g32.CreateCompatibleBitmap(ctypes.c_void_p(hwnd_dc), w, h)
    try:
        g32.SelectObject(ctypes.c_void_p(mem_dc), ctypes.c_void_p(bmp))
        # PW_RENDERFULLCONTENT (2): needed for a GPU-composited window like Chromium.
        if not u32.PrintWindow(ctypes.c_void_p(hwnd), ctypes.c_void_p(mem_dc), 2):
            return None
        info = _BitmapInfoHeader()
        info.biSize = ctypes.sizeof(info)
        info.biWidth, info.biHeight = w, -h  # top-down rows
        info.biPlanes, info.biBitCount = 1, 32
        buf = ctypes.create_string_buffer(w * h * 4)
        got = g32.GetDIBits(
            ctypes.c_void_p(mem_dc), ctypes.c_void_p(bmp), 0, h, buf, ctypes.byref(info), 0
        )
        if got != h:
            return None
        return w, h, buf.raw
    finally:
        g32.DeleteObject(ctypes.c_void_p(bmp))
        g32.DeleteDC(ctypes.c_void_p(mem_dc))
        u32.ReleaseDC(ctypes.c_void_p(hwnd), ctypes.c_void_p(hwnd_dc))


class Mirror:
    """Captures the run's browser window at a modest rate while someone is watching."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self.frame: bytes | None = None
        self.stamp = 0.0
        self._lock = threading.Lock()

    def latest(self) -> bytes | None:
        with self._lock:
            if time.monotonic() - self.stamp < 0.4:
                return self.frame
            self.frame = self._grab()
            self.stamp = time.monotonic()
            return self.frame

    def _grab(self) -> bytes | None:
        try:
            from PIL import Image
        except ImportError:
            return None
        hwnd = _browser_window(_descendants(self.root_pid))
        if hwnd is None:
            return None
        try:
            got = _print_window(hwnd)
            if got is None:
                return None
            w, h, raw = got
            image = Image.frombytes("RGB", (w, h), raw, "raw", "BGRX")
            if image.width > 1280:
                image = image.resize((1280, int(image.height * 1280 / image.width)))
            out = io.BytesIO()
            image.save(out, "JPEG", quality=70)
            return out.getvalue()
        except Exception:
            return None


# --------------------------------------------------------------------------------------
# The side agent: which website, when the sentence does not say
# --------------------------------------------------------------------------------------

SITE_PICKER_MODEL = "claude-sonnet-5"

_SITE_PICKER_PROMPT = """You choose the website a browser agent should start on for a task.
The agent drives an ordinary browser; it is not logged in anywhere and never pays.

Task: {task}

Answer with ONE JSON object and nothing else, of this shape:
{{"url": "https://...", "site": "short site name", "why": "one short sentence"}}

Rules:
- If the task names or clearly implies a website, use that site.
- Otherwise pick the single best-known public website where an ordinary person would do
  this, and its home page or the most useful landing page for the task.
- The URL must be public and need no login. Prefer https.
- If no website can reasonably be chosen, answer {{"url": null, "site": null, "why": "..."}}.
"""


def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # the same .env the agent itself reads
        for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name == "ANTHROPIC_API_KEY":
                return value.strip().strip("\"'")
    except OSError:
        pass
    return None


def pick_site(task: str) -> dict[str, Any]:
    """Ask a small model for the site. Never prefilled - asked tolerantly and parsed."""
    import anthropic

    key = _api_key()
    if not key:
        return {"url": None, "site": None, "why": "no ANTHROPIC_API_KEY"}
    client = anthropic.Anthropic(api_key=key)
    reply = client.messages.create(
        model=SITE_PICKER_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": _SITE_PICKER_PROMPT.format(task=task)}],
    )
    text = "".join(getattr(block, "text", "") for block in reply.content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"url": None, "site": None, "why": f"unparseable reply: {text[:120]!r}"}
    try:
        chosen = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"url": None, "site": None, "why": f"unparseable reply: {text[:120]!r}"}
    url = chosen.get("url")
    if not isinstance(url, str) or not _URL.fullmatch(url.strip()):
        chosen["url"] = None
    return chosen


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


class Run:
    """One subprocess and the lines it has printed so far."""

    _ids = itertools.count(1)

    def __init__(self, text: str, extra: list[str]) -> None:
        self.id = next(self._ids)
        self.text = text
        self.extra = extra
        self.task, self.url = split_task(text)
        self.site: dict[str, Any] | None = None
        self.lines: list[str] = []
        self.done = False
        self.exit_code: int | None = None
        self.listeners: list[queue.Queue[str | None]] = []
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.mirror: Mirror | None = None
        self.stopped = False
        threading.Thread(target=self._pump, daemon=True).start()

    def _emit(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            for q in self.listeners:
                q.put(line)

    def _pump(self) -> None:
        if self.url is None:
            self._emit("choosing a website for this task...")
            try:
                self.site = pick_site(self.task)
            except Exception as exc:  # the run still goes ahead, on the library alone
                self.site = {"url": None, "site": None, "why": f"{type(exc).__name__}: {exc}"}
            if self.site.get("url"):
                self.url = self.site["url"]
                self._emit(f"site: {self.site.get('site')} - {self.site.get('why')}")
            else:
                self._emit(
                    f"no site chosen ({self.site.get('why')}); "
                    "the run will look for a stored skill that knows where to go"
                )
        if self.stopped:
            self._finish(-1)
            return
        env = dict(os.environ)
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            NO_COLOR="1",
            TERM="dumb",
            COLUMNS="110",
        )
        shown = ["skillweaver", "run", json.dumps(self.task)]
        if self.url:
            shown += ["--url", self.url]
        self._emit("$ " + " ".join(shown))
        self.proc = subprocess.Popen(
            command_for(self.task, self.url, self.extra),
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.mirror = Mirror(self.proc.pid)
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            self._emit(raw.rstrip("\r\n"))
        self._finish(self.proc.wait())

    def _finish(self, exit_code: int) -> None:
        with self.lock:
            self.exit_code = exit_code
            self.done = True
            for q in self.listeners:
                q.put(None)

    def subscribe(self) -> tuple[list[str], queue.Queue[str | None] | None]:
        with self.lock:
            if self.done:
                return list(self.lines), None
            q: queue.Queue[str | None] = queue.Queue()
            self.listeners.append(q)
            return list(self.lines), q

    def stop(self) -> None:
        self.stopped = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def frame(self) -> bytes | None:
        return self.mirror.latest() if self.mirror is not None else None


RUNS: dict[int, Run] = {}
EXTRA: list[str] = []


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        first = str(args[0]) if args else ""
        if "/events" not in first and "/frame" not in first:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _json(self, status: HTTPStatus, body: object) -> None:
        self._bytes(status, json.dumps(body).encode(), "application/json")

    def _bytes(self, status: HTTPStatus, data: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _page(self, name: str) -> None:
        self._bytes(HTTPStatus.OK, (STATIC / name).read_bytes(), "text/html; charset=utf-8")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._page("index.html")
        m = re.fullmatch(r"/run/(\d+)", path)
        if m:
            if int(m.group(1)) not in RUNS:
                return self.send_error(HTTPStatus.NOT_FOUND, "no such run")
            return self._page("run.html")
        m = re.fullmatch(r"/api/runs/(\d+)", path)
        if m and int(m.group(1)) in RUNS:
            run = RUNS[int(m.group(1))]
            return self._json(
                HTTPStatus.OK,
                {
                    "id": run.id,
                    "text": run.text,
                    "task": run.task,
                    "url": run.url,
                    "site": run.site,
                    "done": run.done,
                },
            )
        m = re.fullmatch(r"/api/runs/(\d+)/events", path)
        if m:
            return self._events(int(m.group(1)))
        m = re.fullmatch(r"/api/runs/(\d+)/frame\.jpg", path)
        if m and int(m.group(1)) in RUNS:
            frame = RUNS[int(m.group(1))].frame()
            if frame is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return None
            return self._bytes(HTTPStatus.OK, frame, "image/jpeg")
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if path == "/api/run":
            text = str(body.get("text", "")).strip()
            if not text:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "say what to do"})
            run = Run(text, EXTRA)
            RUNS[run.id] = run
            return self._json(HTTPStatus.OK, {"id": run.id, "task": run.task, "url": run.url})
        m = re.fullmatch(r"/api/runs/(\d+)/stop", path)
        if m and int(m.group(1)) in RUNS:
            RUNS[int(m.group(1))].stop()
            return self._json(HTTPStatus.OK, {"ok": True})
        self.send_error(HTTPStatus.NOT_FOUND)

    def _events(self, run_id: int) -> None:
        run = RUNS.get(run_id)
        if run is None:
            return self.send_error(HTTPStatus.NOT_FOUND)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def send(event: str, data: object) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        backlog, q = run.subscribe()
        try:
            for line in backlog:
                send("line", line)
            if q is not None:
                while True:
                    try:
                        item = q.get(timeout=15)
                    except queue.Empty:
                        send("ping", "")
                        continue
                    if item is None:
                        break
                    send("line", item)
            send("done", {"exit_code": run.exit_code})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            if q is not None:
                with run.lock:
                    if q in run.listeners:
                        run.listeners.remove(q)


def main() -> None:
    ap = argparse.ArgumentParser(description="The one-box web UI that starts the agent.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    ap.add_argument(
        "cli_flags",
        nargs="*",
        help="global skillweaver flags handed to every run, e.g. -- --perception dom",
    )
    args = ap.parse_args()
    EXTRA.extend(args.cli_flags)
    if sys.platform == "win32":
        try:  # capture in physical pixels, so window bounds and the screen agree
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
    # Not reusing the address on purpose: with it, Windows lets this server bind
    # BESIDE whatever already owns the port (the sandbox site, on 8765) and the two
    # split the requests. A taken port should fail loudly instead.
    ThreadingHTTPServer.allow_reuse_address = False
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    address = f"http://{args.host}:{args.port}"
    print(f"skillweaver web ui: {address}  (repo {REPO})", flush=True)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, (address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        for run in RUNS.values():
            run.stop()


if __name__ == "__main__":
    main()
