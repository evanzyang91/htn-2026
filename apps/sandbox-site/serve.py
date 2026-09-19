#!/usr/bin/env python3
"""Northwind Console - a deterministic sandbox web app for computer-use agents.

Standard library only. No framework, no build step, no database.

All application state derives from ``seed.json``. The server owns the state and
every mutation goes through ``POST /api/act``, so ``GET /__state`` is always an
exact description of what is on screen and ``GET /__reset`` puts the app back to
a byte-identical starting point.

Run:
    python3 serve.py --port 8765
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

RECORD_COLUMNS = ("name", "owner", "category", "status", "priority", "records", "updated")
BULK_STATUSES = ("Active", "Paused", "Draft", "Archived")


def load_seed():
    with open(SEED_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def initial_ui(data):
    """UI state is part of the state snapshot so that a reset clears it too."""
    settings = data["settings"]
    return {
        "screen": "mail",
        "mail": {
            "folder": "inbox",
            "search": "",
            "selected": [],
            "openId": None,
            "labelMenuOpen": False,
            "banner": None,
            "compose": {"open": False, "to": "", "subject": "", "body": ""},
        },
        "records": {
            "filter": "",
            "sortKey": "name",
            "sortDir": "asc",
            "selected": [],
            "editingId": None,
            "editValue": "",
            "bulkMenuOpen": False,
            "banner": None,
        },
        "settings": {
            "draft": {
                "displayName": settings["displayName"],
                "timezone": settings["timezone"],
                "density": settings["density"],
                "notifyEmail": settings["notifyEmail"],
                "notifyDesktop": settings["notifyDesktop"],
                "weeklyDigest": settings["weeklyDigest"],
                "autoArchive": settings["autoArchive"],
            },
            "dialogOpen": False,
            "success": False,
        },
    }


def fresh_state():
    seed = load_seed()
    data = copy.deepcopy(seed)
    return {
        "meta": data["meta"],
        "mail": data["mail"],
        "records": data["records"],
        "settings": data["settings"],
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


def _msg(state, mid):
    for m in state["mail"]["messages"]:
        if m["id"] == mid:
            return m
    return None


def _row(state, rid):
    for r in state["records"]["rows"]:
        if r["id"] == rid:
            return r
    return None


def _toggle(lst, value):
    if value in lst:
        lst.remove(value)
    else:
        lst.append(value)


def apply_action(state, action, p):
    ui = state["ui"]
    mail, rec, setg = ui["mail"], ui["records"], ui["settings"]

    if action == "nav":
        screen = p.get("screen", "mail")
        if screen in ("mail", "records", "settings"):
            ui["screen"] = screen
            mail["labelMenuOpen"] = False
            rec["bulkMenuOpen"] = False

    # ---- mail ----
    elif action == "mail.folder":
        folder = p.get("folder", "inbox")
        if folder in ("inbox", "archived", "sent"):
            mail["folder"] = folder
            mail["selected"] = []
            mail["openId"] = None
            mail["labelMenuOpen"] = False
    elif action == "mail.search":
        mail["search"] = str(p.get("q", ""))
        mail["selected"] = []
    elif action == "mail.toggleSelect":
        _toggle(mail["selected"], p.get("id"))
        mail["labelMenuOpen"] = False
    elif action == "mail.selectAll":
        mail["selected"] = list(p.get("ids") or []) if p.get("checked") else []
        mail["labelMenuOpen"] = False
    elif action == "mail.open":
        m = _msg(state, p.get("id"))
        if m:
            m["unread"] = False
            mail["openId"] = m["id"]
    elif action == "mail.close":
        mail["openId"] = None
    elif action == "mail.archive":
        n = 0
        for mid in list(mail["selected"]):
            m = _msg(state, mid)
            if m and not m["archived"]:
                m["archived"] = True
                n += 1
            if mail["openId"] == mid:
                mail["openId"] = None
        mail["selected"] = []
        mail["labelMenuOpen"] = False
        mail["banner"] = f"Archived {n} conversation{'' if n == 1 else 's'}"
    elif action == "mail.toggleLabelMenu":
        mail["labelMenuOpen"] = not mail["labelMenuOpen"]
    elif action == "mail.label":
        label = p.get("label")
        n = 0
        if label in state["mail"]["labels"]:
            for mid in mail["selected"]:
                m = _msg(state, mid)
                if m and label not in m["labels"]:
                    m["labels"].append(label)
                    n += 1
        mail["labelMenuOpen"] = False
        mail["selected"] = []
        mail["banner"] = f"Labelled {n} conversation{'' if n == 1 else 's'} as {label}"
    elif action == "mail.dismissBanner":
        mail["banner"] = None
    elif action == "mail.composeOpen":
        mail["compose"] = {"open": True, "to": "", "subject": "", "body": ""}
    elif action == "mail.composeClose":
        mail["compose"] = {"open": False, "to": "", "subject": "", "body": ""}
    elif action == "mail.composeField":
        field = p.get("field")
        if field in ("to", "subject", "body"):
            mail["compose"][field] = str(p.get("value", ""))
    elif action == "mail.send":
        c = mail["compose"]
        if c["open"] and c["to"].strip():
            state["mail"]["sent"].append(
                {
                    "id": f"s{len(state['mail']['sent']) + 1:02d}",
                    "to": c["to"].strip(),
                    "subject": c["subject"].strip(),
                    "body": c["body"],
                    "date": state["meta"]["today"],
                }
            )
            mail["compose"] = {"open": False, "to": "", "subject": "", "body": ""}
            mail["banner"] = "Message sent"

    # ---- records ----
    elif action == "records.filter":
        rec["filter"] = str(p.get("q", ""))
        rec["selected"] = []
        rec["editingId"] = None
    elif action == "records.sort":
        key = p.get("key")
        if key in RECORD_COLUMNS:
            if rec["sortKey"] == key:
                rec["sortDir"] = "desc" if rec["sortDir"] == "asc" else "asc"
            else:
                rec["sortKey"] = key
                rec["sortDir"] = "asc"
    elif action == "records.toggleSelect":
        _toggle(rec["selected"], p.get("id"))
        rec["bulkMenuOpen"] = False
    elif action == "records.selectAll":
        rec["selected"] = list(p.get("ids") or []) if p.get("checked") else []
        rec["bulkMenuOpen"] = False
    elif action == "records.editStart":
        r = _row(state, p.get("id"))
        if r:
            rec["editingId"] = r["id"]
            rec["editValue"] = r["name"]
    elif action == "records.editValue":
        rec["editValue"] = str(p.get("value", ""))
    elif action == "records.editCommit":
        r = _row(state, rec["editingId"])
        value = rec["editValue"].strip()
        if r and value:
            r["name"] = value
            rec["banner"] = f"Renamed to {value}"
        rec["editingId"] = None
        rec["editValue"] = ""
    elif action == "records.editCancel":
        rec["editingId"] = None
        rec["editValue"] = ""
    elif action == "records.toggleBulkMenu":
        rec["bulkMenuOpen"] = not rec["bulkMenuOpen"]
    elif action == "records.bulkStatus":
        status = p.get("status")
        n = 0
        if status in BULK_STATUSES:
            for rid in rec["selected"]:
                r = _row(state, rid)
                if r and r["status"] != status:
                    r["status"] = status
                    n += 1
            rec["banner"] = f"Set {n} record{'' if n == 1 else 's'} to {status}"
        rec["bulkMenuOpen"] = False
        rec["selected"] = []
    elif action == "records.export":
        ids = list(p.get("ids") or [])
        state["records"]["exports"].append(
            {
                "id": f"e{len(state['records']['exports']) + 1:02d}",
                "rowIds": ids,
                "count": len(ids),
                "filter": rec["filter"],
            }
        )
        rec["banner"] = f"Exported {len(ids)} row{'' if len(ids) == 1 else 's'} to CSV"
    elif action == "records.dismissBanner":
        rec["banner"] = None

    # ---- settings ----
    elif action == "settings.field":
        field = p.get("field")
        if field in setg["draft"]:
            setg["draft"][field] = p.get("value")
            setg["success"] = False
    elif action == "settings.discard":
        saved = state["settings"]
        for key in setg["draft"]:
            setg["draft"][key] = saved[key]
        setg["success"] = False
    elif action == "settings.save":
        setg["dialogOpen"] = True
    elif action == "settings.cancel":
        setg["dialogOpen"] = False
    elif action == "settings.confirm":
        if setg["dialogOpen"]:
            state["settings"].update(setg["draft"])
            state["settings"]["savedCount"] += 1
            setg["dialogOpen"] = False
            setg["success"] = True
    elif action == "settings.dismissSuccess":
        setg["success"] = False
    else:
        raise ValueError(f"unknown action: {action}")


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

STORE = Store()


class Handler(BaseHTTPRequestHandler):
    server_version = "NorthwindSandbox/1.0"
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
    ap = argparse.ArgumentParser(description="Northwind Console sandbox server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Northwind Console sandbox on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
