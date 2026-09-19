#!/usr/bin/env python3
"""Harbour Supply - a deterministic sandbox storefront for computer-use agents.

Standard library only. No framework, no build step, no database.

All application state derives from ``seed.json``. The server owns the state and
every mutation goes through ``POST /api/act``, so ``GET /__state`` is always an
exact description of what is on screen and ``GET /__reset`` puts the app back to
a byte-identical starting point.

Run:
    python3 serve.py --port 8766
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

CATEGORIES = ("All", "Tools", "Outdoor", "Kitchen", "Lighting")
PRICE_BANDS = ("any", "under25", "25to75", "over75")
SORTS = ("featured", "price-asc", "price-desc", "rating-desc", "name-asc")
FORM_FIELDS = ("name", "email", "address", "shipping", "save")


def load_seed():
    with open(SEED_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def blank_form():
    return {"name": "", "email": "", "address": "", "shipping": "standard", "save": False}


def initial_ui():
    """UI state is part of the state snapshot so that a reset clears it too."""
    return {
        "screen": "shop",
        "search": "",
        "category": "All",
        "priceBand": "any",
        "sort": "featured",
        "cartOpen": False,
        "checkoutOpen": False,
        "dialogOpen": False,
        "banner": None,
        "form": blank_form(),
    }


def fresh_state():
    seed = load_seed()
    data = copy.deepcopy(seed)
    return {
        "meta": data["meta"],
        "catalog": data["catalog"],
        "cart": data["cart"],
        "orders": data["orders"],
        "ui": initial_ui(),
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


def _product(state, pid):
    for p in state["catalog"]["products"]:
        if p["id"] == pid:
            return p
    return None


def _line(state, pid):
    for line in state["cart"]["lines"]:
        if line["productId"] == pid:
            return line
    return None


def _shipping_fee(state, shipping_id):
    for opt in state["meta"]["shipping"]:
        if opt["id"] == shipping_id:
            return opt["fee"]
    return 0


def _form_complete(ui):
    f = ui["form"]
    return bool(f["name"].strip() and f["email"].strip() and f["address"].strip())


def apply_action(state, action, p):
    ui = state["ui"]
    lines = state["cart"]["lines"]

    # ---- catalog controls ----
    if action == "shop.search":
        ui["search"] = str(p.get("q", ""))
    elif action == "shop.category":
        category = p.get("category", "All")
        if category in CATEGORIES:
            ui["category"] = category
    elif action == "shop.priceBand":
        band = p.get("band", "any")
        if band in PRICE_BANDS:
            ui["priceBand"] = band
    elif action == "shop.sort":
        sort = p.get("sort", "featured")
        if sort in SORTS:
            ui["sort"] = sort
    elif action == "shop.dismissBanner":
        ui["banner"] = None

    # ---- cart ----
    elif action == "cart.add":
        product = _product(state, p.get("productId"))
        if product:
            line = _line(state, product["id"])
            if line:
                line["qty"] += 1
            else:
                lines.append({"productId": product["id"], "qty": 1})
    elif action == "cart.open":
        ui["cartOpen"] = True
        ui["checkoutOpen"] = False
        ui["dialogOpen"] = False
    elif action == "cart.close":
        ui["cartOpen"] = False
        ui["checkoutOpen"] = False
        ui["dialogOpen"] = False
    elif action == "cart.increment":
        line = _line(state, p.get("productId"))
        if line:
            line["qty"] += 1
    elif action == "cart.decrement":
        line = _line(state, p.get("productId"))
        if line and line["qty"] > 1:
            line["qty"] -= 1
    elif action == "cart.remove":
        line = _line(state, p.get("productId"))
        if line:
            lines.remove(line)
            if not lines:
                ui["checkoutOpen"] = False
                ui["dialogOpen"] = False

    # ---- checkout ----
    elif action == "checkout.open":
        if lines:
            ui["cartOpen"] = True
            ui["checkoutOpen"] = True
            ui["dialogOpen"] = False
    elif action == "checkout.back":
        ui["checkoutOpen"] = False
        ui["dialogOpen"] = False
    elif action == "checkout.field":
        field = p.get("field")
        if field in FORM_FIELDS:
            if field == "save":
                ui["form"]["save"] = bool(p.get("value"))
            elif field == "shipping":
                value = str(p.get("value", "standard"))
                if any(opt["id"] == value for opt in state["meta"]["shipping"]):
                    ui["form"]["shipping"] = value
            else:
                ui["form"][field] = str(p.get("value", ""))
    elif action == "checkout.submit":
        if lines and ui["checkoutOpen"] and _form_complete(ui):
            ui["dialogOpen"] = True
    elif action == "checkout.cancel":
        ui["dialogOpen"] = False
    elif action == "checkout.confirm":
        if ui["dialogOpen"] and lines and _form_complete(ui):
            form = ui["form"]
            items = []
            subtotal = 0
            for line in lines:
                product = _product(state, line["productId"])
                items.append(
                    {
                        "productId": product["id"],
                        "name": product["name"],
                        "price": product["price"],
                        "qty": line["qty"],
                    }
                )
                subtotal += product["price"] * line["qty"]
            fee = _shipping_fee(state, form["shipping"])
            n = len(state["orders"]) + 1
            number = 1000 + n
            state["orders"].append(
                {
                    "id": f"o{n:02d}",
                    "number": number,
                    "items": items,
                    "subtotal": subtotal,
                    "shippingFee": fee,
                    "total": subtotal + fee,
                    "shipping": form["shipping"],
                    "name": form["name"].strip(),
                    "email": form["email"].strip(),
                    "address": form["address"].strip(),
                    "saveDetails": form["save"],
                    "placed": state["meta"]["today"],
                }
            )
            state["cart"]["lines"] = []
            ui["cartOpen"] = False
            ui["checkoutOpen"] = False
            ui["dialogOpen"] = False
            ui["banner"] = f"Order #{number} placed - thank you!"
            if not form["save"]:
                ui["form"] = blank_form()
    else:
        raise ValueError(f"unknown action: {action}")


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

STORE = Store()


class Handler(BaseHTTPRequestHandler):
    server_version = "HarbourSupplySandbox/1.0"
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
    ap = argparse.ArgumentParser(description="Harbour Supply sandbox server")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Harbour Supply sandbox on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
