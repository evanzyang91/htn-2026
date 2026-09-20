"""Checks for what a command imports before it has to, and for the two lazy seams.

No browser, no model, no network - but NOT free: the pixel check loads the real detector
weights, so it imports ultralytics and takes a few seconds.

    uv run python scripts/check_i_startup.py

One line per check; exits non-zero if any failed. What these CANNOT show is the time any of
it saves - that is ``python -X importtime`` and a timestamped live run, and the numbers are
in ``src/skillweaver/_prefetch.py``.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

_failed: list[str] = []

HEAVY = (
    "numpy",
    "PIL",
    "anthropic",
    "httpx",
    "torch",
    "ultralytics",
    "onnxruntime",
    "rapidocr_onnxruntime",
    "cv2",
    "playwright",
    "browser_harness",
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail else ''}")
    if not ok:
        _failed.append(name)


def loaded_after(statement: str) -> set[str]:
    """Which of :data:`HEAVY` a FRESH interpreter holds after ``statement``."""
    report = f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    probe = f"{statement}\nimport sys, json\n{report}"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def main() -> int:
    # -- what a command that opens nothing imports ---------------------------------------
    got = loaded_after("import skillweaver.cli")
    check("importing the CLI imports no heavy package", not got, f"loaded: {sorted(got)}")

    got = loaded_after("import skillweaver.llm.anthropic_")
    check("the Claude adapter imports without the SDK", "anthropic" not in got, str(sorted(got)))

    dom_path = (
        "import skillweaver.cli, skillweaver.controllers.harness, skillweaver.perception.dom, "
        "skillweaver.llm.anthropic_, skillweaver.llm.jev_, skillweaver.llm.openai_, "
        "skillweaver.agent.jev_driver, skillweaver.agent.move_critic"
    )
    got = loaded_after(dom_path)
    unwanted = got & {"torch", "ultralytics", "onnxruntime", "rapidocr_onnxruntime", "cv2"}
    check("the dom+jev modules import no detector, OCR or Playwright", not unwanted, str(got))
    check("...and not the Anthropic SDK or numpy either", not got & {"anthropic", "numpy", "PIL"})

    # -- the lazy SDK client ------------------------------------------------------------
    from skillweaver.llm.anthropic_ import AnthropicClient, is_retryable

    client = AnthropicClient(model="claude-opus-5", api_key="sk-not-a-key")
    sdk = client._sdk()
    import anthropic

    check("the first use builds a real SDK client", isinstance(sdk, anthropic.Anthropic))
    check("...once", client._sdk() is sdk)
    check("...with our retries switched off", sdk.max_retries == 0)
    check("...and the key it was given", sdk.api_key == "sk-not-a-key")
    results: list[object] = []
    racing = AnthropicClient(model="claude-opus-5", api_key="k")
    threads = [threading.Thread(target=lambda: results.append(racing._sdk())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    check("eight racing first uses get one client", len({id(r) for r in results}) == 1)

    sent: list[dict[str, object]] = []
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        stop_reason="end_turn",
    )
    fake = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: (sent.append(kw), reply)[1])
    )
    from skillweaver.contracts import LLMMessage

    injected = AnthropicClient(model="claude-opus-5", client=fake)
    check("an injected client is used as it is", injected._sdk() is fake)
    answer = injected.complete([LLMMessage(role="user", text="hi")])
    check("...and a completion goes through it", answer.text == "hello" and len(sent) == 1)
    check("a timeout is still retryable", is_retryable(anthropic.APITimeoutError(request=None)))  # type: ignore[arg-type]
    check("a ValueError is still not", not is_retryable(ValueError("no")))

    # -- prefetch -----------------------------------------------------------------------
    from skillweaver._prefetch import prefetch

    check("prefetching what is already imported starts nothing", prefetch("sys", "json") is None)
    thread = prefetch("skillweaver_no_such_module_i", "colorsys")
    assert thread is not None
    thread.join(10)
    check("a name that cannot be imported is skipped, not raised", "colorsys" in sys.modules)

    # -- the fingerprinter, whose numpy is now imported on use ----------------------------
    import numpy as np
    from PIL import Image

    from skillweaver.contracts import Screenshot
    from skillweaver.perception.fingerprint import StateFingerprinter

    pixels = np.full((800, 1280, 3), 255, dtype="uint8")
    pixels[40:90, 100:900] = 0
    pixels[300:340, 200:500] = 90
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    shot = Screenshot(
        png=buffer.getvalue(), width=1280, height=800, scale=1.0, captured_at=datetime.now(UTC)
    )
    prints = [StateFingerprinter().fingerprint(shot, [], "https://a.example/x") for _ in range(2)]
    check("one frame fingerprints the same twice", prints[0].value == prints[1].value)
    check("...with band parts in it", any(k.startswith("band.") for k in prints[0].parts))
    check("...and is its own same state", prints[0].similarity(prints[1]) == 1.0)

    # -- the pixel path is built exactly as before ----------------------------------------
    from skillweaver.config import load_settings
    from skillweaver.orchestrator import ComposedPerceiver, _open_eyes
    from skillweaver.perception.dom import DomPerceiver

    pixel_eyes = _open_eyes(load_settings({"SKILLWEAVER_PERCEPTION": "pixels"}, env_file=None))
    check(
        "--perception pixels still builds the composed perceiver",
        isinstance(pixel_eyes, ComposedPerceiver),
    )
    observation = pixel_eyes.observe(_Still(shot))
    check(
        "...which detects and reads a frame",
        bool(observation.fingerprint.value),
        f"{len(observation.elements)} element(s)",
    )
    check("...through the real detector", "ultralytics" in sys.modules)
    dom_eyes = _open_eyes(load_settings({"SKILLWEAVER_PERCEPTION": "dom"}, env_file=None))
    check("--perception dom still builds the DOM perceiver", isinstance(dom_eyes, DomPerceiver))

    # -- a DOM read is not a detection ----------------------------------------------------
    from skillweaver.perception.ocr import PerceptionCounters, PerceptionCounts

    pixel_words = str(PerceptionCounts(observations=2, ocr_reads=1, ocr_hits=1, detections=2))
    check(
        "the pixel path's wording is unchanged",
        pixel_words == "2 observation(s), 1 OCR read(s) + 1 cached, 2 detection(s)",
        pixel_words,
    )
    dom_words = str(PerceptionCounts(observations=2, dom_reads=3))
    check("the DOM path says what it did", dom_words.endswith("3 DOM read(s)"), dom_words)
    check("...and claims no detection", "detection" not in dom_words)
    tally = PerceptionCounters()
    tally.dom_reads += 2
    mark = tally.snapshot()
    tally.dom_reads += 3
    check("DOM reads survive a snapshot and a since()", tally.since(mark).dom_reads == 3)
    check("...and add", (mark + mark).dom_reads == 4)
    tally.reset()
    check("...and reset", tally.dom_reads == 0 and not tally.snapshot())

    class _Page(_Still):
        def evaluate(self, script: str) -> object:
            return {"url": self.url(), "title": "t", "text": "hello", "controls": []}

    try:
        dom_eyes.observe(_Page(shot))
        counted = dom_eyes.counters.snapshot()
        check(
            "the DOM perceiver counts a DOM read and no detection",
            counted.dom_reads >= 1 and counted.detections == 0,
            str(counted),
        )
    except Exception as exc:  # noqa: BLE001 - the snapshot shape is dom.py's, not this check's
        check("the DOM perceiver counts a DOM read and no detection", False, repr(exc))

    print(f"\n{len(_failed)} failed" if _failed else "\nall passed")
    return 1 if _failed else 0


class _Still:
    """A controller that only ever shows one frame."""

    def __init__(self, shot: object) -> None:
        self._shot = shot

    def capture(self) -> object:
        return self._shot

    def url(self) -> str:
        return "https://a.example/x"

    def describe(self) -> str:
        return "a still frame"


if __name__ == "__main__":
    raise SystemExit(main())
