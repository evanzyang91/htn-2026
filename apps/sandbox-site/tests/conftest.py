"""Fixtures: a throwaway sandbox server plus a Chromium page pointed at it."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

APP_DIR = Path(__file__).resolve().parent.parent
SERVE = APP_DIR / "serve.py"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SERVE), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: {proc.stdout.read().decode()}")
        try:
            urllib.request.urlopen(url + "/__state", timeout=0.5).read()
            break
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError("server did not come up")
    yield url
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, base_url):
    """A fresh page on a freshly reset server - identical starting state per test."""
    urllib.request.urlopen(base_url + "/__reset").read()
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        device_scale_factor=1,
    )
    pg = context.new_page()
    pg.goto(base_url + "/")
    pg.wait_for_selector("body[data-ready='1']")
    yield pg
    context.close()


@pytest.fixture
def state(base_url):
    """Read the server's authoritative state, so assertions never read the screen."""

    def _read():
        return json.loads(urllib.request.urlopen(base_url + "/__state").read())

    return _read


@pytest.fixture
def seed():
    return json.loads((APP_DIR / "seed.json").read_text())
