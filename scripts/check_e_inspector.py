"""Checks for the live inspector's HTTP surface, with no browser and no network.

    uv run python scripts/check_e_inspector.py

Starts real :class:`~skillweaver.inspector.server.Inspector` servers in-process on the
loopback and talks to them over HTTP. Two kinds of session sit behind them: a real, idle
:class:`~skillweaver.inspector.session.LiveSession` - which opens nothing until a run is
started - for what the state JSON carries, and a stub whose ``advance`` only counts, for
the automatic run, because what is being checked there is the worker's loop and not an
agent. Needs no ``.env``, no credentials and no network; it binds loopback ports, one of
them the inspector's default when that is free. Exits non-zero on the first failure.

What this cannot show is that the three timings are RIGHT - that takes a real page, and
is the live run reported with this change.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from skillweaver.config import Settings
from skillweaver.inspector.server import DEFAULT_PORT, PORT_SPAN, Inspector, _answering
from skillweaver.inspector.session import (
    SessionError,
    _Split,
    _WatchedPolicy,
    run_overrides,
    text_models,
)

_FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _FAILED
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail else ''}", flush=True)
    if not ok:
        _FAILED += 1


class StubSession:
    """Just enough of ``LiveSession`` for the worker: a status, a snapshot and a move."""

    def __init__(self, moves_to_done: int = 1000, *, breaks: bool = False) -> None:
        self.status = "ready"
        self.frame = 0
        self.frame_png = b""
        self.moves = 0
        self.started_with: dict[str, Any] = {}
        self._to_done = moves_to_done
        self._breaks = breaks

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status, "note": "", "history": [], "moves": self.moves}

    def advance(self) -> None:
        time.sleep(0.03)
        if self._breaks:
            raise RuntimeError("the provider is down")
        self.moves += 1
        if self.moves >= self._to_done:
            self.status = "done"

    def start(self, **kwargs: Any) -> None:
        self.started_with = kwargs
        self.status, self.moves = "ready", 0

    def close(self) -> None:
        self.status = "idle"


class Running:
    """An inspector serving on a thread, and a client that knows its token."""

    def __init__(self, inspector: Inspector) -> None:
        self.inspector = inspector
        self._thread = threading.Thread(target=inspector.serve_forever, daemon=True)
        self._thread.start()

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        token: str | None = "",
        host: str | None = None,
        origin: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.inspector.port, timeout=20)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Inspector-Token"] = token or self.inspector.token
        if host is not None:
            headers["Host"] = host
        if origin is not None:
            headers["Origin"] = origin
        payload = json.dumps(body or {}) if method == "POST" else None
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        try:
            return response.status, json.loads(raw)
        except ValueError:
            return response.status, {}

    def state(self) -> dict[str, Any]:
        return self.request("GET", "/api/state")[1]

    def until(self, test: Any, seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if test(self.state()):
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self.inspector.close()
        self._thread.join(timeout=10)


def check_door(settings: Settings) -> None:
    running = Running(Inspector(settings, port=0))
    try:
        status, body = running.request("GET", "/api/state", token=None)
        check("state without a token is 403", status == 403, f"{status} {body.get('code')}")
        check("...and says the token is stale", body.get("code") == "stale-token")
        status, body = running.request("GET", "/api/state", token="not-the-token")
        check(
            "state with a wrong token is 403", status == 403 and body.get("code") == "stale-token"
        )
        status, body = running.request("GET", "/api/frame", token=None)
        check("the frame needs the token too", status == 403)
        status, body = running.request(
            "GET", "/api/state", host=f"localhost:{running.inspector.port}"
        )
        check("another Host is 403 not-local", status == 403 and body.get("code") == "not-local")
        status, body = running.request(
            "POST", "/api/auto", {"on": True}, origin="http://evil.example"
        )
        check("a foreign Origin is 403 origin", status == 403 and body.get("code") == "origin")
        check("...and flipped nothing", running.state().get("auto") is False)
        status, body = running.request("POST", "/api/auto", {"on": True}, token=None)
        check(
            "a command without a token is 403", status == 403 and body.get("code") == "stale-token"
        )

        status, state = running.request("GET", "/api/state")
        check("state with the token is 200", status == 200 and state.get("status") == "idle")
        timing = state.get("timing", {})
        wanted = {"wall_ms", "model_ms", "site_ms", "frame_ms", "other_ms", "first_load_ms"}
        check("state carries the timing split", wanted <= set(timing), str(sorted(timing)))
        check("state carries the auto switch", state.get("auto") is False and "auto_note" in state)
        options, mode = state.get("options", {}), state.get("mode", {})
        check(
            "state carries the run's model controls",
            options.get("text_models", [None])[0] == settings.text_model
            and options.get("efforts") == ["", "low", "medium", "high"]
            and mode.get("text_model") == settings.text_model
            and mode.get("refine_goal") is False,
            json.dumps(options),
        )
        status, body = running.request(
            "POST", "/api/start", {"url": "https://x.test", "goal": "g", "text_model": "made-up"}
        )
        check(
            "a model that is not offered is refused before a browser",
            status == 400,
            body.get("error", ""),
        )
        status, body = running.request("POST", "/api/auto", {"on": "yes"})
        check("auto wants a boolean", status == 400)
    finally:
        running.close()


def check_auto(settings: Settings) -> None:
    stub = StubSession()
    running = Running(Inspector(settings, port=0, session=stub))  # type: ignore[arg-type]
    try:
        time.sleep(0.2)
        check("nothing moves while the switch is off", stub.moves == 0)
        status, state = running.request("POST", "/api/auto", {"on": True})
        check("the switch turns on", status == 200 and state.get("auto") is True)
        check("...and the worker moves by itself", running.until(lambda s: s["moves"] >= 3))
        check("...naming what it is doing", running.until(lambda s: s.get("doing") == "auto"))

        # A posted command goes ahead of the next automatic move rather than being refused.
        status, _ = running.request(
            "POST",
            "/api/start",
            {"url": "https://x.test", "goal": "g", "refine_goal": True, "text_effort": "low"},
        )
        check("a posted command lands during an automatic run", status == 200)
        check(
            "start is handed the run's overrides",
            stub.started_with.get("overrides") == {"refine_goal": True, "text_effort": "low"},
            str(stub.started_with.get("overrides")),
        )
        check("...and the run carries on after it", running.until(lambda s: s["moves"] >= 2))

        status, state = running.request("POST", "/api/auto", {"on": False})
        check("the switch turns off mid-run", status == 200 and state.get("auto") is False)
        time.sleep(0.15)  # longer than one move: whatever was in flight has landed
        at = stub.moves
        time.sleep(0.3)
        check("...and no further move begins", stub.moves == at, f"{at} -> {stub.moves}")

        stub._to_done = stub.moves + 2
        running.request("POST", "/api/auto", {"on": True})
        check(
            "switched on again, it runs to the end", running.until(lambda s: s["status"] == "done")
        )
        at = stub.moves
        time.sleep(0.2)
        check("...stops there", stub.moves == at)
        check(
            "...and the switch is still on, for the next run", running.state().get("auto") is True
        )
    finally:
        running.close()

    broken = Running(Inspector(settings, port=0, session=StubSession(breaks=True)))  # type: ignore[arg-type]
    try:
        broken.request("POST", "/api/auto", {"on": True})
        check(
            "a move that raises switches the run off and says why",
            broken.until(lambda s: s["auto"] is False and "provider is down" in s["auto_note"]),
            broken.state().get("auto_note", ""),
        )
    finally:
        broken.close()


def check_ports(settings: Settings) -> None:
    # The trap this guards is a listener on the WILDCARD address: on macOS a bind to
    # 127.0.0.1 succeeds right beside it. So that is what occupies the default here,
    # unless something (a real inspector, say) already holds it - which is left alone.
    squatter = None
    if not _answering(DEFAULT_PORT):
        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("0.0.0.0", DEFAULT_PORT))
        squatter.listen(1)
    try:
        check(f"the default port {DEFAULT_PORT} is answering", _answering(DEFAULT_PORT))
        inspector = Inspector(settings, session=StubSession())  # type: ignore[arg-type]
        try:
            check(
                "the default falls forward to a free port",
                DEFAULT_PORT < inspector.port < DEFAULT_PORT + PORT_SPAN,
                str(inspector.port),
            )
            check("...and the URL and Host follow it", inspector.url.endswith(f":{inspector.port}"))
        finally:
            inspector.close()
        try:
            Inspector(settings, port=DEFAULT_PORT, session=StubSession()).close()  # type: ignore[arg-type]
            check("a NAMED port that is taken fails loudly", False, "it bound")
        except OSError as exc:
            check("a NAMED port that is taken fails loudly", True, str(exc))
    finally:
        if squatter is not None:
            squatter.close()


def check_pure(settings: Settings) -> None:
    split = _Split.of(1000, 400, 500, 300)
    check(
        "a split never sums past its wall",
        split.model_ms + split.site_ms + split.frame_ms <= split.wall_ms and split.other_ms == 0,
        str(split.as_json()),
    )
    split = _Split.of(1000, 200, 300, 100)
    check("...and leaves the remainder as judging", split.other_ms == 400, str(split.as_json()))
    check("...and refuses a clock that ran backwards", _Split.of(100, -5, -5, -5).other_ms == 100)
    total = _Split.of(1000, 200, 300, 100) + _Split.of(500, 100, 100, 50)
    check("splits add", total.as_json()["wall_ms"] == 1500 and total.as_json()["frame_ms"] == 150)

    class _Policy:
        def decide(self, *_args: Any) -> str:
            return "decision"

    watched = _WatchedPolicy(_Policy())
    exclude = {
        "CLICK": {"e2", "e1"},
        "CONTROLS": {"SCROLL_DOWN"},
        "LABELS": {"Next"},
        "TYPE_TEXT": set(),
    }
    watched.decide("goal", None, [], exclude)  # type: ignore[arg-type]
    check(
        "the reserved exclude keys are filed apart from the element ids",
        watched.last_exclude == {"CLICK": ["e1", "e2"]}
        and watched.last_reserved == {"CONTROLS": ["SCROLL_DOWN"], "LABELS": ["Next"]},
        f"{watched.last_exclude} {watched.last_reserved}",
    )

    check("no key is no override", run_overrides({}, settings) == {})
    check(
        "blank effort clears it",
        run_overrides({"text_effort": ""}, settings) == {"text_effort": None},
    )
    check("the configured model is offered first", text_models(settings)[0] == settings.text_model)
    for body in ({"text_effort": "max"}, {"refine_model": "x"}):
        try:
            run_overrides(body, settings)
            check(f"{body} is refused", False)
        except SessionError as exc:
            check(f"{body} is refused", True, str(exc)[:60])


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(data_dir=Path(directory))
        check_pure(settings)
        check_door(settings)
        check_auto(settings)
        check_ports(settings)
    print(f"{_FAILED} failed" if _FAILED else "all checks passed")
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
