"""Loopback-only inspector for the Jev browser agent."""

import atexit
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import Agent
from .model import EFFORTS, refine_goal
from .questions import MAX_STEPS

ROOT = Path(__file__).parent
PORT = int(os.environ.get("TYPESAFE_DEMO_PORT", "8766"))
ORIGIN = f"http://127.0.0.1:{PORT}"
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
AGENT = None


def load_environment():
    path = Path.cwd() / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


DEFAULT_REFINE_MODELS = "gpt-6-astra,gpt-5.6-luna,gpt-5.6-sol,gpt-4o-mini"


def refine_models():
    """Offered in the inspector and used as the allowlist for the requested model."""
    names = [m.strip() for m in os.environ.get("REFINE_MODELS", DEFAULT_REFINE_MODELS).split(",") if m.strip()]
    current = os.environ.get("REFINE_MODEL") or os.environ.get("TEXT_MODEL", "deepseek-chat")
    return ([current] + [m for m in names if m != current]) if current not in names else names


def response_state():
    state = AGENT.snapshot() if AGENT else {"page": None, "status": "idle", "history": [], "decision": None}
    return {
        **state,
        "text_model": os.environ.get("TEXT_MODEL", "deepseek-chat"),
        "refine_models": refine_models(),
        "refine_model": os.environ.get("REFINE_MODEL") or os.environ.get("TEXT_MODEL", "deepseek-chat"),
        "refine_effort": os.environ.get("REFINE_EFFORT", ""),
        "efforts": list(EFFORTS),
        "max_steps": MAX_STEPS,
    }


def close_browser():
    global AGENT
    if AGENT:
        AGENT.close()
        AGENT = None


def command(name, body):
    global AGENT
    if name == "reset":
        refinement = None
        scenario = body.get("scenario", "flights")
        if scenario not in {"travel", "research", "flights", "custom"}:
            raise ValueError("Unknown demo scenario")
        goal = body.get("goal", "").strip()
        if not goal or len(goal) > 2000:
            raise ValueError("Enter 1–2,000 characters")
        if scenario == "custom":
            url = body.get("url", "").strip()
            parts = urlparse(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or len(url) > 2000:
                raise ValueError("Enter a full http(s):// address")
        elif scenario == "flights":
            url = "https://www.google.com/travel/flights?hl=en"
        else:
            url = f"{ORIGIN}/fixture.html?scenario={scenario}"
        if body.get("refine"):
            chosen = body.get("refine_model") or None
            if chosen and chosen not in refine_models():
                raise ValueError("Unknown refinement model")
            effort = body.get("refine_effort") or None
            if effort and effort not in EFFORTS:
                raise ValueError("Unknown thinking level")
            goal, refinement = refine_goal(goal, url, chosen, effort)
        close_browser()
        AGENT = Agent(
            url,
            goal,
            screenshots=True,
            record_dir=Path.cwd() / "artifacts" / "frames" if body.get("record") else None,
        )
        AGENT.state["scenario"] = scenario
        AGENT.state["refinement"] = refinement
    else:
        if AGENT is None:
            raise ValueError("Start a demo first")
        AGENT.command(name, body)
    return response_state()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, mime="application/json"):
        content = content if isinstance(content, bytes) else content.encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{PORT}":
            return self.send(403, "Forbidden", "text/plain")
        path = urlparse(self.path).path
        if path == "/api/state":
            with LOCK:
                return self.send(200, json.dumps(response_state()))
        files = {
            "/": ("index.html", "text/html"),
            "/app.js": ("app.js", "text/javascript"),
            "/style.css": ("style.css", "text/css"),
            "/fixture.html": ("fixture.html", "text/html"),
        }
        if path not in files:
            return self.send(404, "Not found", "text/plain")
        name, mime = files[path]
        content = (ROOT / "static" / name).read_text().replace("__TOKEN__", TOKEN)
        self.send(200, content, mime + "; charset=utf-8")

    def do_POST(self):
        if (
            self.headers.get("Host") != f"127.0.0.1:{PORT}"
            or self.headers.get("X-Demo-Token") != TOKEN
            or self.headers.get("Origin") not in (None, ORIGIN)
        ):
            return self.send(403, json.dumps({"error": "Local demo requests only"}))
        if not LOCK.acquire(blocking=False):
            return self.send(409, json.dumps({"error": "A browser step is already running"}))
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 8192:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            result = command(self.path.removeprefix("/api/"), body)
            self.send(200, json.dumps(result))
        except (ValueError, RuntimeError, TimeoutError) as error:
            self.send(400, json.dumps({"error": str(error)}))
        except Exception:
            self.send(500, json.dumps({"error": "Local demo failed; no automatic retry. Reset to recover."}))
        finally:
            LOCK.release()

    def log_message(self, *_args):
        pass


def main():
    load_environment()
    atexit.register(close_browser)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Jevis: {ORIGIN}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
