"""The Claude adapter: request translation, reply parsing, retry and error mapping."""

from __future__ import annotations

import base64

import pytest

from skillweaver.contracts import LLMClient, LLMMessage, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm.anthropic_ import (
    COMPUTER_USE_ACTIONS,
    COMPUTER_USE_MODELS,
    COMPUTER_USE_TOOL,
    AnthropicClient,
    build_messages,
    is_retryable,
)
from tests.llm.conftest import (
    StubAnthropic,
    anthropic_connection_error,
    anthropic_error,
    anthropic_message,
)

CLICK = ToolSpec(
    "click",
    "Click at a point.",
    {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
)


def client(*script: object, **kwargs: object) -> AnthropicClient:
    """An adapter over a scripted stub, with retry sleeping instantly."""
    kwargs.setdefault("model", "claude-opus-5")
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("jitter", lambda: 1.0)
    return AnthropicClient(client=StubAnthropic(list(script)), **kwargs)  # type: ignore[arg-type]


class TestProtocol:
    def test_satisfies_the_llm_client_protocol(self) -> None:
        assert isinstance(client(anthropic_message(text="hi")), LLMClient)

    def test_name_is_the_model(self) -> None:
        assert client(anthropic_message(text="hi"), model="claude-sonnet-5").name() == (
            "claude-sonnet-5"
        )


class TestPlainCompletion:
    def test_returns_the_text_and_usage(self) -> None:
        c = client(anthropic_message(text="Paris.", input_tokens=1_000_000, output_tokens=200_000))
        response = c.complete([LLMMessage("user", "Capital of France?")])
        assert response.text == "Paris."
        assert response.tool_calls == ()
        assert response.stop_reason == "end"
        assert response.usage == Usage(1_000_000, 200_000, 1, pytest.approx(10.00))

    def test_sends_the_system_prompt_and_the_conversation(self) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        c = AnthropicClient(client=stub, model="claude-opus-5")
        c.complete(
            [
                LLMMessage("user", "one"),
                LLMMessage("assistant", "two"),
                LLMMessage("user", "three"),
            ],
            system="be terse",
            max_tokens=256,
        )
        request = stub.calls[0]
        assert request["model"] == "claude-opus-5"
        assert request["system"] == "be terse"
        assert request["max_tokens"] == 256
        assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]
        assert request["messages"][0]["content"] == [{"type": "text", "text": "one"}]

    def test_omits_system_and_tools_when_unset(self) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5").complete([LLMMessage("user", "hi")])
        assert "system" not in stub.calls[0]
        assert "tools" not in stub.calls[0]

    def test_temperature_is_dropped_not_forwarded(self) -> None:
        """Current Claude models reject sampling parameters; the contract says drop
        it silently rather than fail."""
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5").complete(
            [LLMMessage("user", "hi")], temperature=0.7
        )
        assert "temperature" not in stub.calls[0]

    def test_effort_is_sent_as_output_config(self) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5", effort="low").complete(
            [LLMMessage("user", "hi")]
        )
        assert stub.calls[0]["output_config"] == {"effort": "low"}

    def test_usage_accumulates_over_calls(self) -> None:
        c = client(
            anthropic_message(text="a", input_tokens=1_000_000, output_tokens=0),
            anthropic_message(text="b", input_tokens=1_000_000, output_tokens=0),
        )
        c.complete([LLMMessage("user", "one")])
        c.complete([LLMMessage("user", "two")])
        assert c.total_usage() == Usage(2_000_000, 0, 2, pytest.approx(10.00))

    def test_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="at least one message"):
            client().complete([])


class TestImages:
    def test_attaches_a_screenshot_as_an_image_block(self, png: bytes) -> None:
        stub = StubAnthropic([anthropic_message(text="a login page")])
        AnthropicClient(client=stub, model="claude-opus-5").complete(
            [LLMMessage("user", "What is on screen?", images=(png,))]
        )
        content = stub.calls[0]["messages"][0]["content"]
        # Image first, question second: the model reads the screenshot better that way.
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/png"
        assert base64.standard_b64decode(content[0]["source"]["data"]) == png
        assert content[1] == {"type": "text", "text": "What is on screen?"}

    def test_attaches_several_images_in_order(self, png: bytes) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5").complete(
            [LLMMessage("user", "diff these", images=(png, b"second"))]
        )
        content = stub.calls[0]["messages"][0]["content"]
        assert [b["type"] for b in content] == ["image", "image", "text"]
        assert base64.standard_b64decode(content[1]["source"]["data"]) == b"second"


class TestToolUse:
    def test_returns_the_tool_call_and_stop_reason(self) -> None:
        c = client(
            anthropic_message(
                tool_calls=[("tu_1", "click", {"x": 120, "y": 240})], stop_reason="tool_use"
            )
        )
        response = c.complete([LLMMessage("user", "click login")], tools=[CLICK])
        assert response.stop_reason == "tool_use"
        assert response.tool_calls == (ToolCall("click", {"x": 120, "y": 240}, "tu_1"),)
        assert response.text == ""

    def test_returns_text_alongside_a_tool_call(self) -> None:
        c = client(
            anthropic_message(
                text="I will click login.",
                tool_calls=[("tu_1", "click", {"x": 1})],
                stop_reason="tool_use",
            )
        )
        response = c.complete([LLMMessage("user", "go")], tools=[CLICK])
        assert response.text == "I will click login."
        assert len(response.tool_calls) == 1

    def test_sends_the_tool_definition(self) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5").complete(
            [LLMMessage("user", "go")], tools=[CLICK]
        )
        assert stub.calls[0]["tools"] == [
            {
                "name": "click",
                "description": "Click at a point.",
                "input_schema": CLICK.input_schema,
            }
        ]

    def test_computer_use_tool_is_appended_when_asked(self) -> None:
        stub = StubAnthropic([anthropic_message(text="ok")])
        AnthropicClient(client=stub, model="claude-opus-5", computer_use=True).complete(
            [LLMMessage("user", "go")], tools=[CLICK]
        )
        assert stub.calls[0]["tools"][-1] == dict(COMPUTER_USE_TOOL)

    def test_computer_use_tool_is_schema_less(self) -> None:
        """The 2026-08-01 toolset rejects name/display_*; only `type` belongs."""
        assert dict(COMPUTER_USE_TOOL) == {"type": "computer_toolset_20260801"}
        assert "screenshot" in COMPUTER_USE_ACTIONS
        assert "left_click" in COMPUTER_USE_ACTIONS
        assert len(COMPUTER_USE_ACTIONS) == 17
        assert "claude-opus-5" in COMPUTER_USE_MODELS

    def test_warns_on_a_model_that_cannot_use_it(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="skillweaver"):
            client(model="claude-opus-4-6", computer_use=True)
        assert "anthropic.computer_use_unsupported" in caplog.text


class TestMessageTranslation:
    def test_an_assistant_turn_carries_its_tool_calls(self) -> None:
        turns = build_messages(
            [
                LLMMessage("user", "go"),
                LLMMessage(
                    "assistant", "clicking", tool_calls=(ToolCall("click", {"x": 1}, "t1"),)
                ),
                LLMMessage("tool", "clicked", tool_call_id="t1"),
            ]
        )
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"][1] == {
            "type": "tool_use",
            "id": "t1",
            "name": "click",
            "input": {"x": 1},
        }

    def test_consecutive_tool_results_collapse_into_one_user_turn(self) -> None:
        """Splitting tool_result blocks across user messages teaches Claude to stop
        making parallel tool calls, so they must arrive together."""
        turns = build_messages(
            [
                LLMMessage("user", "go"),
                LLMMessage(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("click", {}, "t1"), ToolCall("type", {}, "t2")),
                ),
                LLMMessage("tool", "clicked", tool_call_id="t1"),
                LLMMessage("tool", "typed", tool_call_id="t2"),
            ]
        )
        assert len(turns) == 3
        assert turns[2]["role"] == "user"
        assert [b["tool_use_id"] for b in turns[2]["content"]] == ["t1", "t2"]

    def test_a_tool_result_can_carry_a_screenshot(self, png: bytes) -> None:
        turns = build_messages(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "", tool_calls=(ToolCall("screenshot", {}, "t1"),)),
                LLMMessage("tool", "here it is", images=(png,), tool_call_id="t1"),
            ]
        )
        block = turns[2]["content"][0]
        assert block["type"] == "tool_result"
        assert [c["type"] for c in block["content"]] == ["text", "image"]

    def test_an_empty_tool_result_still_produces_content(self) -> None:
        turns = build_messages(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "", tool_calls=(ToolCall("wait", {}, "t1"),)),
                LLMMessage("tool", "", tool_call_id="t1"),
            ]
        )
        assert turns[2]["content"][0]["content"] == [{"type": "text", "text": ""}]

    def test_an_entirely_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="empty content"):
            build_messages([LLMMessage("user", "")])


class TestStopReasons:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("end_turn", "end"),
            ("stop_sequence", "end"),
            ("tool_use", "tool_use"),
            ("max_tokens", "max_tokens"),
            ("refusal", "other"),
            ("pause_turn", "other"),
            (None, "other"),
        ],
    )
    def test_normalizes(self, raw: str | None, expected: str) -> None:
        c = client(anthropic_message(text="x", stop_reason=raw))  # type: ignore[arg-type]
        assert c.complete([LLMMessage("user", "hi")]).stop_reason == expected


class TestRetry:
    def test_retries_a_transient_failure_then_succeeds(self) -> None:
        slept: list[float] = []
        c = client(
            anthropic_connection_error(),
            anthropic_error(429),
            anthropic_message(text="third time lucky"),
            sleep=slept.append,
        )
        assert c.complete([LLMMessage("user", "hi")]).text == "third time lucky"
        assert len(slept) == 2, "one sleep per retry"
        assert slept[0] < slept[1], "backoff grows"

    def test_retries_a_server_error(self) -> None:
        c = client(anthropic_error(503), anthropic_message(text="recovered"))
        assert c.complete([LLMMessage("user", "hi")]).text == "recovered"

    def test_gives_up_after_max_attempts(self) -> None:
        c = client(*[anthropic_error(503) for _ in range(3)], max_attempts=3)
        with pytest.raises(ProviderError, match="after 3 attempt"):
            c.complete([LLMMessage("user", "hi")])

    def test_max_attempts_one_disables_retry(self) -> None:
        c = client(anthropic_error(503), max_attempts=1)
        with pytest.raises(ProviderError, match="after 1 attempt"):
            c.complete([LLMMessage("user", "hi")])

    def test_backoff_is_capped(self) -> None:
        slept: list[float] = []
        c = client(
            *[anthropic_error(503) for _ in range(5)],
            anthropic_message(text="ok"),
            max_attempts=6,
            base_delay=1.0,
            max_delay=2.0,
            sleep=slept.append,
        )
        c.complete([LLMMessage("user", "hi")])
        assert max(slept) <= 2.0

    def test_rejects_a_nonsense_attempt_count(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            AnthropicClient(client=StubAnthropic([]), max_attempts=0)

    @pytest.mark.parametrize("status", [429, 500, 503, 408, 409])
    def test_retryable_statuses(self, status: int) -> None:
        assert is_retryable(anthropic_error(status))

    @pytest.mark.parametrize("status", [400, 401, 404])
    def test_non_retryable_statuses(self, status: int) -> None:
        assert not is_retryable(anthropic_error(status))

    def test_a_connection_error_is_retryable(self) -> None:
        assert is_retryable(anthropic_connection_error())

    def test_an_unrelated_exception_is_not_retryable(self) -> None:
        assert not is_retryable(ValueError("nope"))


class TestCredentialResolution:
    """What the first live run got wrong.

    The SDK reads only the process environment, so a key that lives in ``.env``
    reaches it only if the adapter passes it explicitly. ``settings()`` is the one
    place that merges the two.
    """

    def test_a_dot_env_key_is_passed_to_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import skillweaver.llm.anthropic_ as module
        from skillweaver.config import Settings

        captured: dict[str, object] = {}

        class FakeSDK:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(module.anthropic, "Anthropic", FakeSDK)
        monkeypatch.setattr(
            module, "settings", lambda: Settings(anthropic_api_key="key-from-dot-env")
        )
        AnthropicClient()
        assert captured["api_key"] == "key-from-dot-env"
        assert captured["max_retries"] == 0, "our backoff must be the only one"

    def test_an_explicit_key_wins_over_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import skillweaver.llm.anthropic_ as module
        from skillweaver.config import Settings

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            module.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object()
        )
        monkeypatch.setattr(
            module, "settings", lambda: Settings(anthropic_api_key="key-from-dot-env")
        )
        AnthropicClient(api_key="explicit")
        assert captured["api_key"] == "explicit"

    def test_no_key_anywhere_leaves_sdk_resolution_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An auth token or a stored profile is still a valid credential, so the
        adapter must not force api_key=None and break them."""
        import skillweaver.llm.anthropic_ as module
        from skillweaver.config import Settings

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            module.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object()
        )
        monkeypatch.setattr(module, "settings", lambda: Settings(anthropic_api_key=None))
        AnthropicClient()
        assert "api_key" not in captured

    def test_workspace_id_becomes_a_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An org-scoped key 400s on every endpoint without this header."""
        import skillweaver.llm.anthropic_ as module
        from skillweaver.config import Settings

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            module.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object()
        )
        monkeypatch.setattr(module, "settings", lambda: Settings(anthropic_api_key="k"))
        AnthropicClient(workspace_id="wrkspc_abc")
        assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_abc"}

    def test_no_workspace_header_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import skillweaver.llm.anthropic_ as module
        from skillweaver.config import Settings

        captured: dict[str, object] = {}
        monkeypatch.setattr(
            module.anthropic, "Anthropic", lambda **kw: captured.update(kw) or object()
        )
        monkeypatch.setattr(module, "settings", lambda: Settings(anthropic_api_key="k"))
        AnthropicClient()
        assert "default_headers" not in captured


class TestErrorMapping:
    def test_a_non_retryable_error_raises_provider_error_with_the_cause(self) -> None:
        original = anthropic_error(400, "messages.0: invalid role")
        c = client(original)
        with pytest.raises(ProviderError) as excinfo:
            c.complete([LLMMessage("user", "hi")])
        assert excinfo.value.__cause__ is original
        assert "invalid role" in str(excinfo.value)
        assert "BadRequestError" in str(excinfo.value)

    def test_an_auth_error_is_not_retried(self) -> None:
        stub = StubAnthropic([anthropic_error(401, "invalid x-api-key")])
        c = AnthropicClient(client=stub, model="claude-opus-5", sleep=lambda _s: None)
        with pytest.raises(ProviderError, match="after 1 attempt"):
            c.complete([LLMMessage("user", "hi")])
        assert len(stub.calls) == 1

    def test_an_unreadable_reply_raises_provider_error(self) -> None:
        class Broken:
            content = "not a list of blocks"

        c = client(Broken())
        with pytest.raises(ProviderError, match="could not read"):
            c.complete([LLMMessage("user", "hi")])
