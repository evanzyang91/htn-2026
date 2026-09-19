"""Cassette scrubbing, digesting, recording and replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillweaver.contracts import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm.cassette import (
    REDACTED,
    Cassette,
    CassetteClient,
    Interaction,
    digest,
    register_secret,
    request_payload,
    response_from_dict,
    response_to_dict,
    scrub,
)
from tests.fakes import FakeLLM

REAL_KEY = "sk-ant-api03-ZZZZtotallyrealsecretkeymaterial9999"


def payload(**overrides: object) -> dict:
    base = {
        "model": "claude-opus-5",
        "messages": [LLMMessage("user", "hello")],
        "system": None,
        "tools": None,
        "max_tokens": 16000,
        "temperature": None,
    }
    base.update(overrides)
    return request_payload(**base)  # type: ignore[arg-type]


class TestScrub:
    def test_redacts_a_credential_named_key(self) -> None:
        assert scrub({"api_key": "whatever", "x-api-key": "y"}) == {
            "api_key": REDACTED,
            "x-api-key": REDACTED,
        }

    def test_redacts_a_key_shaped_value_anywhere(self) -> None:
        text = f"use {REAL_KEY} to authenticate"
        assert REAL_KEY not in scrub(text)
        assert REDACTED in scrub(text)

    def test_redacts_a_google_key(self) -> None:
        assert "AIza" not in scrub("key=AIzaSyA1234567890abcdefghijklmnop")

    def test_redacts_an_environment_key_by_exact_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "not-key-shaped-at-all-4711")
        assert scrub("token is not-key-shaped-at-all-4711") == f"token is {REDACTED}"

    def test_redacts_a_registered_secret(self) -> None:
        register_secret("hunter2-and-then-some")
        assert "hunter2" not in scrub("password: hunter2-and-then-some")

    def test_ignores_a_too_short_registration(self) -> None:
        register_secret("")
        assert scrub("nothing sensitive here") == "nothing sensitive here"

    def test_recurses_through_lists_and_dicts(self) -> None:
        got = scrub({"a": [{"secret": 1}, "plain"], "b": ("x", REAL_KEY)})
        assert got == {"a": [{"secret": REDACTED}, "plain"], "b": ["x", REDACTED]}

    def test_leaves_non_strings_alone(self) -> None:
        assert scrub({"n": 5, "f": 1.5, "b": True, "z": None}) == {
            "n": 5,
            "f": 1.5,
            "b": True,
            "z": None,
        }


class TestDigest:
    def test_is_stable_across_key_order(self) -> None:
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_changes_with_the_message(self) -> None:
        assert digest(payload()) != digest(payload(messages=[LLMMessage("user", "goodbye")]))

    def test_changes_with_max_tokens(self) -> None:
        assert digest(payload()) != digest(payload(max_tokens=64))

    def test_changes_with_the_tool_set(self) -> None:
        tool = ToolSpec("click", "click a thing", {"type": "object", "properties": {}})
        assert digest(payload()) != digest(payload(tools=[tool]))

    def test_an_image_contributes_by_content(self, png: bytes) -> None:
        with_image = payload(messages=[LLMMessage("user", "hi", images=(png,))])
        other_image = payload(messages=[LLMMessage("user", "hi", images=(b"different",))])
        assert digest(with_image) != digest(payload())
        assert digest(with_image) != digest(other_image)

    def test_an_image_is_recorded_by_hash_not_inline(self, png: bytes) -> None:
        body = payload(messages=[LLMMessage("user", "hi", images=(png,))])
        ref = body["messages"][0]["images"][0]
        assert set(ref) == {"sha256", "bytes"}
        assert ref["bytes"] == len(png)

    def test_a_key_in_the_prompt_does_not_change_the_digest(self) -> None:
        """Scrub happens BEFORE the digest, so the same call replays whatever key
        was in play when it was recorded."""
        one = payload(system=f"auth with {REAL_KEY}")
        two = payload(system="auth with sk-ant-api03-AAAAdifferentkeymaterialentirely1")
        assert digest(one) == digest(two)


class TestSerialization:
    def test_response_round_trips(self) -> None:
        response = LLMResponse(
            text="done",
            tool_calls=(ToolCall("click", {"x": 1, "y": 2}, "tu_1"),),
            usage=Usage(10, 5, 1, 0.25),
            stop_reason="tool_use",
        )
        assert response_from_dict(response_to_dict(response)) == response

    def test_an_empty_response_round_trips(self) -> None:
        assert response_from_dict(response_to_dict(LLMResponse())) == LLMResponse()


class TestCassetteFile:
    def test_a_missing_file_loads_empty(self, tmp_path: Path) -> None:
        cassette = Cassette.load(tmp_path / "nope.json")
        assert cassette.interactions == []

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        cassette = Cassette(path=path, model="claude-opus-5")
        cassette.append(Interaction("abc", {"m": 1}, {"text": "hi"}))
        cassette.save()
        again = Cassette.load(path)
        assert again.model == "claude-opus-5"
        assert again.find("abc") is not None
        assert again.find("missing") is None

    def test_re_recording_replaces_rather_than_grows(self, tmp_path: Path) -> None:
        cassette = Cassette(path=tmp_path / "c.json")
        cassette.append(Interaction("abc", {}, {"text": "one"}))
        cassette.append(Interaction("abc", {}, {"text": "two"}))
        assert len(cassette.interactions) == 1
        assert cassette.interactions[0].response["text"] == "two"

    def test_a_wrong_version_is_a_provider_error(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"version": 99, "interactions": []}))
        with pytest.raises(ProviderError, match="version 99"):
            Cassette.load(path)

    def test_garbage_is_a_provider_error(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text("not json at all")
        with pytest.raises(ProviderError, match="unreadable"):
            Cassette.load(path)


class TestRecordAndReplay:
    def test_record_then_replay_returns_the_same_reply(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        inner = FakeLLM([LLMResponse(text="recorded reply", usage=Usage(7, 3, 1, 0.01))])
        recorder = CassetteClient(path, mode="record", inner=inner)
        live = recorder.complete([LLMMessage("user", "hello")])

        player = CassetteClient(path, mode="replay")
        replayed = player.complete([LLMMessage("user", "hello")])
        assert replayed == live
        assert inner.calls == 1, "replay must not reach the inner client"

    def test_replay_reports_the_model_the_cassette_was_recorded_from(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        inner = FakeLLM([LLMResponse(text="x")], model="claude-opus-5")
        CassetteClient(path, mode="record", inner=inner).complete([LLMMessage("user", "hi")])
        assert CassetteClient(path, mode="replay").name() == "claude-opus-5"

    def test_replay_accumulates_usage(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        inner = FakeLLM(
            [
                LLMResponse(text="a", usage=Usage(10, 2, 1, 0.1)),
                LLMResponse(text="b", usage=Usage(20, 4, 1, 0.2)),
            ]
        )
        recorder = CassetteClient(path, mode="record", inner=inner)
        recorder.complete([LLMMessage("user", "one")])
        recorder.complete([LLMMessage("user", "two")])

        player = CassetteClient(path, mode="replay")
        player.complete([LLMMessage("user", "one")])
        player.complete([LLMMessage("user", "two")])
        assert player.total_usage() == Usage(30, 6, 2, pytest.approx(0.3))

    def test_an_unmatched_request_names_itself(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        inner = FakeLLM([LLMResponse(text="x")])
        CassetteClient(path, mode="record", inner=inner).complete([LLMMessage("user", "hello")])

        player = CassetteClient(path, mode="replay")
        with pytest.raises(ProviderError) as excinfo:
            player.complete([LLMMessage("user", "something else entirely")])
        message = str(excinfo.value)
        assert "no recorded interaction" in message
        assert "something else entirely" in message
        assert str(path) in message
        assert "1 interaction(s)" in message

    def test_recording_a_key_does_not_write_the_key(self, tmp_path: Path) -> None:
        """The gate: a cassette written from a request containing a key does not
        contain that key."""
        path = tmp_path / "c.json"
        inner = FakeLLM([LLMResponse(text=f"I will use {REAL_KEY}")])
        recorder = CassetteClient(path, mode="record", inner=inner)
        recorder.complete(
            [LLMMessage("user", f"my key is {REAL_KEY}")],
            system=f"Authenticate with {REAL_KEY}.",
            tools=[ToolSpec("auth", f"pass {REAL_KEY}", {"type": "object"})],
        )
        written = path.read_text(encoding="utf-8")
        assert REAL_KEY not in written
        assert "sk-ant" not in written
        assert REDACTED in written
        # The response is scrubbed on the way out too, not just the request.
        assert REAL_KEY not in json.dumps(Cassette.load(path).interactions[0].response)

    def test_record_mode_needs_an_inner_client(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="inner client"):
            CassetteClient(tmp_path / "c.json", mode="record")

    def test_an_unknown_mode_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="record"):
            CassetteClient(tmp_path / "c.json", mode="sideways")  # type: ignore[arg-type]

    def test_record_forwards_every_argument(self, tmp_path: Path) -> None:
        inner = FakeLLM([LLMResponse(text="x")])
        recorder = CassetteClient(tmp_path / "c.json", mode="record", inner=inner)
        tool = ToolSpec("click", "click", {"type": "object"})
        recorder.complete(
            [LLMMessage("user", "go")],
            system="be brief",
            tools=[tool],
            max_tokens=99,
            temperature=0.3,
        )
        request = inner.requests[0]
        assert request.system == "be brief"
        assert request.tools == (tool,)
        assert request.max_tokens == 99
        assert request.temperature == 0.3
