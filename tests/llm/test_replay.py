"""Both adapters replay the committed cassettes, offline and identically.

The point of this file is the gate: a plain completion, a completion with an
attached image, and a completion returning a tool call, replayed for Claude and
for Gemini, with nothing behind the cassette. Anything downstream that wants a
model in a test wires it up exactly like this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillweaver.contracts import LLMClient, LLMMessage
from skillweaver.errors import ProviderError
from skillweaver.llm.cassette import Cassette, CassetteClient
from tests.llm.record_fixtures import (
    ANTHROPIC_CASSETTE,
    GEMINI_CASSETTE,
    SCENARIOS,
    SCREENSHOT_BYTES,
    Scenario,
    record,
)

BACKENDS = pytest.mark.parametrize(
    "cassette_path",
    [pytest.param(ANTHROPIC_CASSETTE, id="anthropic"), pytest.param(GEMINI_CASSETTE, id="gemini")],
)

BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


def player(path: Path) -> CassetteClient:
    return CassetteClient(path, mode="replay")


@BACKENDS
class TestCommittedCassettes:
    def test_the_cassette_exists_and_holds_every_scenario(self, cassette_path: Path) -> None:
        cassette = Cassette.load(cassette_path)
        assert len(cassette.interactions) == len(SCENARIOS)
        assert cassette.model

    def test_replays_a_plain_completion(self, cassette_path: Path) -> None:
        response = BY_KEY["plain"].run(player(cassette_path))
        assert response.text.strip().lower() == "ready"
        assert response.stop_reason == "end"
        assert response.tool_calls == ()
        assert response.usage.calls == 1
        assert response.usage.input_tokens > 0
        assert response.usage.cost_usd > 0, "a priced model must record a real cost"

    def test_replays_a_completion_with_an_attached_image(self, cassette_path: Path) -> None:
        scenario = BY_KEY["image"]
        assert scenario.messages[0].images == (SCREENSHOT_BYTES,)
        response = scenario.run(player(cassette_path))
        assert "submit" in response.text.strip().lower()

    def test_replays_a_completion_returning_a_tool_call(self, cassette_path: Path) -> None:
        response = BY_KEY["tool_call"].run(player(cassette_path))
        assert response.stop_reason == "tool_use"
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "click"
        # Logical pixels, as the prompt gave them: nothing rescales a coordinate.
        assert call.args == {"x": 120, "y": 240}

    def test_the_player_is_an_llm_client(self, cassette_path: Path) -> None:
        assert isinstance(player(cassette_path), LLMClient)

    def test_usage_accumulates_across_replayed_calls(self, cassette_path: Path) -> None:
        client = player(cassette_path)
        for scenario in SCENARIOS:
            scenario.run(client)
        total = client.total_usage()
        assert total.calls == len(SCENARIOS)
        assert total.input_tokens > 0
        assert total.cost_usd > 0

    def test_a_different_image_does_not_match(self, cassette_path: Path) -> None:
        """The digest covers image content, so a cassette cannot answer for a
        screenshot it never saw."""
        scenario = BY_KEY["image"]
        wrong = Scenario(
            key="image",
            messages=(LLMMessage("user", scenario.messages[0].text, images=(b"other bytes",)),),
            system=scenario.system,
            tools=scenario.tools,
            max_tokens=scenario.max_tokens,
        )
        with pytest.raises(ProviderError, match="no recorded interaction"):
            wrong.run(player(cassette_path))

    def test_an_unrecorded_request_names_itself(self, cassette_path: Path) -> None:
        with pytest.raises(ProviderError) as excinfo:
            player(cassette_path).complete([LLMMessage("user", "never recorded")])
        message = str(excinfo.value)
        assert "never recorded" in message
        assert str(cassette_path) in message

    def test_no_credential_is_in_the_committed_file(self, cassette_path: Path) -> None:
        body = cassette_path.read_text(encoding="utf-8")
        for marker in ("sk-ant-", "AIza", "ya29.", "x-api-key"):
            assert marker not in body


class TestRerecording:
    """Re-recording offline reproduces the committed cassettes exactly, which is
    what keeps ``python tests/llm/record_fixtures.py`` honest."""

    @pytest.mark.parametrize("backend", ["anthropic", "gemini"])
    def test_offline_rerecord_matches_the_committed_file(
        self, backend: str, tmp_path: Path
    ) -> None:
        from tests.llm.record_fixtures import _offline_anthropic, _offline_gemini

        committed = ANTHROPIC_CASSETTE if backend == "anthropic" else GEMINI_CASSETTE
        client = _offline_anthropic() if backend == "anthropic" else _offline_gemini()
        fresh = tmp_path / "fresh.json"
        record(client, fresh)

        expected = Cassette.load(committed)
        actual = Cassette.load(fresh)
        assert actual.model == expected.model
        assert [i.digest for i in actual.interactions] == [i.digest for i in expected.interactions]
        assert [i.response for i in actual.interactions] == [
            i.response for i in expected.interactions
        ]

    def test_a_failed_recording_leaves_the_existing_cassette_intact(self, tmp_path: Path) -> None:
        """A live run that dies partway must not destroy the committed fixture.

        Recording in place did exactly that the first time a real key was tried:
        the call failed on scenario one and the cassette was already gone.
        """
        from tests.llm.record_fixtures import _offline_anthropic

        target = tmp_path / "existing.json"
        record(_offline_anthropic(), target)
        before = target.read_bytes()

        dying = _offline_anthropic()
        dying._client.messages.script.clear()  # the stub runs out on the first call

        with pytest.raises(ProviderError):
            record(dying, target)

        assert target.read_bytes() == before, "the previous cassette must survive"
        assert not list(tmp_path.glob("*.recording")), "no staging file left behind"
