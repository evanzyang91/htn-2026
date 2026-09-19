"""The three scenarios every cassette holds, and the script that records them.

One definition, used three ways:

* ``tests/llm/test_replay.py`` replays the committed cassettes through both
  adapters and asserts the replies.
* ``python tests/llm/record_fixtures.py`` re-records them offline, from the stub
  provider SDKs in ``tests/llm/conftest.py``.
* ``python tests/llm/record_fixtures.py --live`` re-records them from the real
  APIs, which is how a cassette gets a genuine provider reply in it. This is the
  only code path in ``tests/`` that touches a network, and nothing in the suite
  calls it.

Keeping the scenarios here rather than in the test is what makes the recording and
the replay provably the same request: the digest is computed from these values.

Provenance of the committed cassettes, which differs by backend and matters:

* ``anthropic_basics.json`` was recorded LIVE from ``claude-opus-5`` on
  2026-09-19. Every reply in it is a real API response.
* ``gemini_basics.json`` is still stub-recorded. No ``GEMINI_API_KEY`` has been
  supplied, so that adapter has never made a real call. Re-record it with
  ``--live --only gemini`` when a key exists.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillweaver.contracts import LLMMessage, LLMResponse, ToolSpec
from skillweaver.llm.cassette import CassetteClient

CASSETTE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes"
ANTHROPIC_CASSETTE = CASSETTE_DIR / "anthropic_basics.json"
GEMINI_CASSETTE = CASSETTE_DIR / "gemini_basics.json"
SCREENSHOT = CASSETTE_DIR / "screenshot.png"

SCREENSHOT_BYTES = SCREENSHOT.read_bytes()

CLICK_TOOL = ToolSpec(
    name="click",
    description="Click the mouse at a point, in logical pixels from the top-left of the viewport.",
    input_schema={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Logical pixels from the left edge."},
            "y": {"type": "integer", "description": "Logical pixels from the top edge."},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One ``complete()`` call, recorded and replayed byte-identically."""

    key: str
    messages: tuple[LLMMessage, ...]
    system: str | None = None
    tools: tuple[ToolSpec, ...] | None = None
    max_tokens: int = 256

    def run(self, client: Any) -> LLMResponse:
        return client.complete(
            self.messages,
            system=self.system,
            tools=self.tools,
            max_tokens=self.max_tokens,
        )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="plain",
        messages=(LLMMessage("user", "Reply with exactly one word: ready"),),
        system="You are a terse test fixture. Answer in as few words as possible.",
    ),
    Scenario(
        key="image",
        messages=(
            LLMMessage(
                "user",
                "This is a screenshot. Name the single word written on it, nothing else.",
                images=(SCREENSHOT_BYTES,),
            ),
        ),
        system="You are a terse test fixture. Answer in as few words as possible.",
    ),
    Scenario(
        key="tool_call",
        messages=(
            LLMMessage(
                "user",
                "Click the Submit button. It is at x=120, y=240 in logical pixels. "
                "Use the click tool.",
            ),
        ),
        system="You drive a computer with the tools you are given. Prefer a tool over prose.",
        tools=(CLICK_TOOL,),
    ),
)

#: A canned reply per scenario per backend, used to re-record a cassette offline.
#:
#: The ``anthropic`` entries are TRANSCRIBED FROM THE LIVE RECORDING - the text,
#: the tool-call id and the token counts are what claude-opus-5 actually returned
#: on 2026-09-19. They are not invented, and they are what makes the committed
#: live cassette reproducible without a key. If a later live re-record changes
#: them, ``test_offline_rerecord_matches_the_committed_file`` fails and the new
#: values belong here.
#:
#: The ``gemini`` entries are still INVENTED stubs: no GEMINI_API_KEY has ever
#: been supplied, so that cassette has never seen a real response.
OFFLINE_REPLIES: dict[str, dict[str, dict[str, Any]]] = {
    "anthropic": {
        "plain": {"text": "ready", "input_tokens": 40, "output_tokens": 19},
        "image": {"text": "Submit", "input_tokens": 125, "output_tokens": 5},
        "tool_call": {
            "text": "",
            "tool_calls": [("toolu_01JDZkibSUJssN1UuiZVbi5a", "click", {"x": 120, "y": 240})],
            "input_tokens": 498,
            "output_tokens": 66,
        },
    },
    "gemini": {
        "plain": {"text": "ready", "input_tokens": 27, "output_tokens": 3},
        "image": {"text": "Submit", "input_tokens": 312, "output_tokens": 4},
        "tool_call": {
            "text": "",
            "tool_calls": [("call_fixture_1", "click", {"x": 120, "y": 240})],
            "input_tokens": 415,
            "output_tokens": 38,
        },
    },
}


def record(client: Any, path: Path, scenarios: Sequence[Scenario] = SCENARIOS) -> None:
    """Run every scenario through ``client`` and write the cassette at ``path``.

    Recording goes to a sibling temporary file and is moved into place only after
    EVERY scenario has succeeded. A live run that dies on the first call - an
    unscoped key, a rate limit, a network drop - then leaves the committed
    cassette untouched instead of deleting it. Recording in place cost exactly
    that once, and a half-written fixture is worse than a stale one because the
    suite fails somewhere unrelated to the real problem.
    """
    staging = path.with_name(path.name + ".recording")
    staging.unlink(missing_ok=True)
    recorder = CassetteClient(staging, mode="record", inner=client)
    try:
        for scenario in scenarios:
            response = scenario.run(recorder)
            print(f"  {scenario.key:<10} -> {response.stop_reason:<9} {response.text[:48]!r}")
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    staging.replace(path)


def _offline_anthropic() -> Any:
    from tests.llm.conftest import StubAnthropic, anthropic_message

    script = []
    for scenario in SCENARIOS:
        reply = OFFLINE_REPLIES["anthropic"][scenario.key]
        script.append(
            anthropic_message(
                text=reply["text"],
                tool_calls=reply.get("tool_calls", ()),
                stop_reason="tool_use" if reply.get("tool_calls") else "end_turn",
                input_tokens=reply["input_tokens"],
                output_tokens=reply["output_tokens"],
            )
        )
    from skillweaver.llm.anthropic_ import AnthropicClient

    return AnthropicClient(client=StubAnthropic(script), model="claude-opus-5")


def _offline_gemini() -> Any:
    from google.genai import types as genai_types

    from skillweaver.llm.gemini_ import GeminiClient
    from tests.llm.conftest import StubGenai, gemini_response

    script = []
    for scenario in SCENARIOS:
        reply = OFFLINE_REPLIES["gemini"][scenario.key]
        script.append(
            gemini_response(
                text=reply["text"],
                tool_calls=[(cid, name, args) for cid, name, args in reply.get("tool_calls", ())],
                finish_reason=genai_types.FinishReason.STOP,
                prompt_tokens=reply["input_tokens"],
                output_tokens=reply["output_tokens"],
            )
        )
    return GeminiClient(client=StubGenai(script), model="gemini-2.5-computer-use-preview-10-2025")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="record from the real APIs instead of the stub SDKs (spends money)",
    )
    parser.add_argument(
        "--only",
        choices=("anthropic", "gemini"),
        help="record just one backend",
    )
    parser.add_argument(
        "--workspace-id",
        help="anthropic-workspace-id header, for a key scoped to an org not a workspace",
    )
    args = parser.parse_args(argv)

    backends = []
    if args.only in (None, "anthropic"):
        if args.live:
            from skillweaver.llm.anthropic_ import AnthropicClient

            backends.append(
                (
                    "anthropic",
                    AnthropicClient(workspace_id=args.workspace_id),
                    ANTHROPIC_CASSETTE,
                ),
            )
        else:
            backends.append(("anthropic", _offline_anthropic(), ANTHROPIC_CASSETTE))
    if args.only in (None, "gemini"):
        if args.live:
            from skillweaver.llm.gemini_ import GeminiClient

            backends.append(("gemini", GeminiClient(), GEMINI_CASSETTE))
        else:
            backends.append(("gemini", _offline_gemini(), GEMINI_CASSETTE))

    for label, client, path in backends:
        print(f"{label} ({'live' if args.live else 'offline'}) -> {path}")
        record(client, path)
        print(f"  total usage: {client.total_usage()}")
    return 0


def _make_tests_importable() -> None:
    """Point ``import tests`` at this repository's ``tests/`` directory.

    ``ultralytics`` installs its own top-level ``tests`` package into
    site-packages. Ours has no ``__init__.py`` (see the note in ``pyproject.toml``
    about ``--import-mode=importlib``), and Python only falls back to a namespace
    package when no regular package is found anywhere on the path - so a bare
    ``import tests`` outside pytest picks up ultralytics' one no matter how the
    path is ordered. pytest is unaffected; this is only for running this file as a
    script.
    """
    import types

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("tests")
    package.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules["tests"] = package
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))


if __name__ == "__main__":
    _make_tests_importable()
    sys.exit(main())
