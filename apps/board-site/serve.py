#!/usr/bin/env python3
"""Lantern Board - a deterministic sandbox web app for computer-use agents.

Standard library only. No framework, no build step, no database.

All application state derives from ``seed.json``. The server owns the state and
every mutation goes through ``POST /api/act``, so ``GET /__state`` is always an
exact description of what is on screen and ``GET /__reset`` puts the app back to
a byte-identical starting point.

Run:
    python3 serve.py --port 8767
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(HERE, "seed.json")
STATIC_DIR = os.path.join(HERE, "static")

COMPOSE_FIELDS = ("title", "description", "assignee", "priority")


def load_seed():
    with open(SEED_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def closed_compose(board):
    return {
        "open": False,
        "title": "",
        "description": "",
        "assignee": board["assignees"][0],
        "priority": "Medium",
    }


def initial_ui(data):
    """UI state is part of the state snapshot so that a reset clears it too."""
    return {
        "search": "",
        "assignee": "all",
        "priority": "all",
        "openId": None,
        "moveMenuFor": None,
        "compose": closed_compose(data["board"]),
        "dialogOpen": False,
        "banner": None,
        "draft": {"title": "", "comment": ""},
    }


def fresh_state():
    seed = load_seed()
    data = copy.deepcopy(seed)
    return {
        "meta": data["meta"],
        "board": data["board"],
        "archived": data["archived"],
        "ui": initial_ui(data),
    }


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = fresh_state()

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def reset(self):
        with self._lock:
            self._state = fresh_state()
            return copy.deepcopy(self._state)

    def act(self, action, payload):
        with self._lock:
            apply_action(self._state, action, payload or {})
            return copy.deepcopy(self._state)


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------


def _ticket(state, tid):
    for t in state["board"]["tickets"]:
        if t["id"] == tid:
            return t
    return None


def _column_name(board, col_id):
    for c in board["columns"]:
        if c["id"] == col_id:
            return c["name"]
    return col_id


def apply_action(state, action, p):
    ui = state["ui"]
    board = state["board"]

    # ---- filters ----
    if action == "board.search":
        ui["search"] = str(p.get("q", ""))
        ui["moveMenuFor"] = None
    elif action == "board.filterAssignee":
        assignee = p.get("assignee", "all")
        if assignee == "all" or assignee in board["assignees"]:
            ui["assignee"] = assignee
            ui["moveMenuFor"] = None
    elif action == "board.filterPriority":
        priority = p.get("priority", "all")
        if priority == "all" or priority in board["priorities"]:
            ui["priority"] = priority
            ui["moveMenuFor"] = None

    # ---- moving tickets ----
    elif action == "board.toggleMoveMenu":
        tid = p.get("id")
        ui["moveMenuFor"] = None if ui["moveMenuFor"] == tid else tid
    elif action == "board.move":
        t = _ticket(state, p.get("id"))
        col = p.get("column")
        if t and any(c["id"] == col for c in board["columns"]) and t["column"] != col:
            t["column"] = col
            ui["banner"] = f"Moved {t['key']} to {_column_name(board, col)}"
        ui["moveMenuFor"] = None

    # ---- detail panel ----
    elif action == "board.open":
        t = _ticket(state, p.get("id"))
        if t:
            ui["openId"] = t["id"]
            ui["draft"] = {"title": t["title"], "comment": ""}
            ui["moveMenuFor"] = None
    elif action == "board.close":
        ui["openId"] = None
        ui["draft"] = {"title": "", "comment": ""}
    elif action == "board.draftTitle":
        ui["draft"]["title"] = str(p.get("value", ""))
    elif action == "board.saveTitle":
        t = _ticket(state, ui["openId"])
        value = ui["draft"]["title"].strip()
        if t and value:
            t["title"] = value
            ui["banner"] = f"Renamed {t['key']} to {value}"
    elif action == "board.assign":
        t = _ticket(state, ui["openId"])
        assignee = p.get("assignee")
        if t and assignee in board["assignees"] and t["assignee"] != assignee:
            t["assignee"] = assignee
            ui["banner"] = f"Assigned {t['key']} to {assignee}"
    elif action == "board.draftComment":
        ui["draft"]["comment"] = str(p.get("value", ""))
    elif action == "board.addComment":
        t = _ticket(state, ui["openId"])
        text = ui["draft"]["comment"].strip()
        if t and text:
            board["counters"]["comment"] += 1
            t["comments"].append(
                {
                    "id": f"c{board['counters']['comment']:02d}",
                    "author": state["meta"]["user"]["name"],
                    "text": text,
                    "date": state["meta"]["today"],
                }
            )
            ui["draft"]["comment"] = ""
            ui["banner"] = f"Comment added to {t['key']}"

    # ---- new ticket ----
    elif action == "board.composeOpen":
        ui["compose"] = dict(closed_compose(board), open=True)
        ui["moveMenuFor"] = None
    elif action == "board.composeClose":
        ui["compose"] = closed_compose(board)
    elif action == "board.composeField":
        field = p.get("field")
        if field in COMPOSE_FIELDS:
            ui["compose"][field] = str(p.get("value", ""))
    elif action == "board.create":
        c = ui["compose"]
        title = c["title"].strip()
        if c["open"] and title:
            board["counters"]["ticket"] += 1
            n = board["counters"]["ticket"]
            board["tickets"].append(
                {
                    "id": f"t{n:02d}",
                    "key": f"LAN-{n}",
                    "title": title,
                    "description": c["description"].strip(),
                    "assignee": c["assignee"] if c["assignee"] in board["assignees"] else board["assignees"][0],
                    "priority": c["priority"] if c["priority"] in board["priorities"] else "Medium",
                    "column": "backlog",
                    "comments": [],
                }
            )
            ui["compose"] = closed_compose(board)
            ui["banner"] = f"Created LAN-{n} in Backlog"

    # ---- archive done ----
    elif action == "board.archiveDialog":
        if any(t["column"] == "done" for t in board["tickets"]):
            ui["dialogOpen"] = True
            ui["moveMenuFor"] = None
    elif action == "board.archiveCancel":
        ui["dialogOpen"] = False
    elif action == "board.archiveConfirm":
        if ui["dialogOpen"]:
            done = [t for t in board["tickets"] if t["column"] == "done"]
            board["tickets"] = [t for t in board["tickets"] if t["column"] != "done"]
            state["archived"].extend(done)
            if ui["openId"] in {t["id"] for t in done}:
                ui["openId"] = None
                ui["draft"] = {"title": "", "comment": ""}
            ui["dialogOpen"] = False
            ui["banner"] = f"Archived {len(done)} ticket{'' if len(done) == 1 else 's'}"

    elif action == "board.dismissBanner":
        ui["banner"] = None
    else:
        raise ValueError(f"unknown action: {action}")


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

STORE = Store()


class Handler(BaseHTTPRequestHandler):
    server_version = "LanternSandbox/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console quiet during agent runs
        pass

    # -- helpers --
    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # -- routes --
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/__state":
            self._json(STORE.snapshot())
        elif path == "/__reset":
            self._json({"ok": True, "state": STORE.reset()})
        elif path == "/__seed":
            self._json(load_seed())
        elif path == "/" or path == "":
            self._static("index.html")
        else:
            self._static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            self._json({"error": "bad json"}, 400)
            return
        if path == "/api/act":
            try:
                state = STORE.act(body.get("action"), body.get("payload"))
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"seq": body.get("seq", 0), "state": state})
        elif path == "/__reset":
            self._json({"ok": True, "state": STORE.reset()})
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser(description="Lantern Board sandbox server")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Lantern Board sandbox on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
