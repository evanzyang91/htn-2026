"""The Gemini adapter: the same surface, over Gemini's own shapes."""

from __future__ import annotations

import pytest
from google.genai import types as genai_types

from skillweaver.contracts import LLMClient, LLMMessage, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm.gemini_ import (
    ENVIRONMENTS,
    GeminiClient,
    build_contents,
    computer_use_tool,
    is_retryable,
)
from tests.llm.conftest import StubGenai, gemini_error, gemini_response

MODEL = "gemini-2.5-computer-use-preview-10-2025"

CLICK = ToolSpec(
    "click",
    "Click at a point.",
    {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
)


def client(*script: object, **kwargs: object) -> GeminiClient:
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("jitter", lambda: 1.0)
    return GeminiClient(client=StubGenai(list(script)), **kwargs)  # type: ignore[arg-type]


class TestProtocol:
    def test_satisfies_the_llm_client_protocol(self) -> None:
        assert isinstance(client(gemini_response(text="hi")), LLMClient)

    def test_name_is_the_model(self) -> None:
        assert client(gemini_response(text="hi")).name() == MODEL


class TestPlainCompletion:
    def test_returns_the_text_and_usage(self) -> None:
        c = client(gemini_response(text="Paris.", prompt_tokens=1_000_000, output_tokens=1_000_000))
        response = c.complete([LLMMessage("user", "Capital of France?")])
        assert response.text == "Paris."
        assert response.stop_reason == "end"
        # $2.00 in + $10.00 out per million.
        assert response.usage == Usage(1_000_000, 1_000_000, 1, pytest.approx(12.00))

    def test_sends_system_prompt_max_tokens_and_the_conversation(self) -> None:
        stub = StubGenai([gemini_response(text="ok")])
        GeminiClient(client=stub, model=MODEL).complete(
            [
                LLMMessage("user", "one"),
                LLMMessage("assistant", "two"),
                LLMMessage("user", "three"),
            ],
            system="be terse",
            max_tokens=256,
        )
        request = stub.calls[0]
        assert request["model"] == MODEL
        assert request["config"].system_instruction == "be terse"
        assert request["config"].max_output_tokens == 256
        # Gemini calls the assistant role "model".
        assert [c.role for c in request["contents"]] == ["user", "model", "user"]

    def test_temperature_is_forwarded(self) -> None:
        """Unlike Claude, Gemini accepts sampling parameters."""
        stub = StubGenai([gemini_response(text="ok")])
        GeminiClient(client=stub, model=MODEL).complete(
            [LLMMessage("user", "hi")], temperature=0.25
        )
        assert stub.calls[0]["config"].temperature == 0.25

    def test_temperature_is_omitted_when_none(self) -> None:
        stub = StubGenai([gemini_response(text="ok")])
        GeminiClient(client=stub, model=MODEL).complete([LLMMessage("user", "hi")])
        assert stub.calls[0]["config"].temperature is None

    def test_usage_accumulates_over_calls(self) -> None:
        c = client(
            gemini_response(text="a", prompt_tokens=1_000_000, output_tokens=0),
            gemini_response(text="b", prompt_tokens=1_000_000, output_tokens=0),
        )
        c.complete([LLMMessage("user", "one")])
        c.complete([LLMMessage("user", "two")])
        assert c.total_usage() == Usage(2_000_000, 0, 2, pytest.approx(4.00))

    def test_a_thought_part_is_not_the_answer(self) -> None:
        raw = gemini_response(text="the answer")
        raw.candidates[0].content.parts.insert(
            0, genai_types.Part(text="let me think", thought=True)
        )
        assert client(raw).complete([LLMMessage("user", "hi")]).text == "the answer"

    def test_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="at least one message"):
            client().complete([])

    def test_no_candidates_is_a_provider_error(self) -> None:
        raw = genai_types.GenerateContentResponse(candidates=[])
        with pytest.raises(ProviderError, match="no candidates"):
            client(raw).complete([LLMMessage("user", "hi")])


class TestImages:
    def test_attaches_a_screenshot_as_an_inline_part(self, png: bytes) -> None:
        stub = StubGenai([gemini_response(text="a login page")])
        GeminiClient(client=stub, model=MODEL).complete(
            [LLMMessage("user", "What is on screen?", images=(png,))]
        )
        parts = stub.calls[0]["contents"][0].parts
        assert parts[0].inline_data.mime_type == "image/png"
        assert parts[0].inline_data.data == png
        assert parts[1].text == "What is on screen?"


class TestToolUse:
    def test_returns_the_tool_call(self) -> None:
        c = client(gemini_response(tool_calls=[("fc_1", "click", {"x": 120, "y": 240})]))
        response = c.complete([LLMMessage("user", "click login")], tools=[CLICK])
        assert response.tool_calls == (ToolCall("click", {"x": 120, "y": 240}, "fc_1"),)

    def test_a_tool_call_reports_stop_reason_tool_use(self) -> None:
        """Gemini says STOP for a turn that asked for a function call; the contract
        wants tool_use."""
        c = client(gemini_response(tool_calls=[("fc_1", "click", {"x": 1})]))
        assert c.complete([LLMMessage("user", "go")], tools=[CLICK]).stop_reason == "tool_use"

    def test_a_missing_call_id_becomes_empty_string(self) -> None:
        c = client(gemini_response(tool_calls=[(None, "click", {"x": 1})]))
        assert c.complete([LLMMessage("user", "go")], tools=[CLICK]).tool_calls[0].id == ""

    def test_sends_the_tool_declaration_with_its_json_schema(self) -> None:
        stub = StubGenai([gemini_response(text="ok")])
        GeminiClient(client=stub, model=MODEL).complete([LLMMessage("user", "go")], tools=[CLICK])
        declarations = stub.calls[0]["config"].tools[0].function_declarations
        assert declarations[0].name == "click"
        assert declarations[0].parameters_json_schema == dict(CLICK.input_schema)

    def test_extra_tools_are_sent_too(self) -> None:
        stub = StubGenai([gemini_response(text="ok")])
        GeminiClient(client=stub, model=MODEL, extra_tools=[computer_use_tool("browser")]).complete(
            [LLMMessage("user", "go")], tools=[CLICK]
        )
        tools = stub.calls[0]["config"].tools
        assert tools[0].computer_use is not None
        assert tools[1].function_declarations is not None


class TestComputerUseTool:
    def test_builds_the_browser_environment(self) -> None:
        tool = computer_use_tool("browser")
        assert tool.computer_use.environment == genai_types.Environment.ENVIRONMENT_BROWSER
        assert tool.computer_use.enable_prompt_injection_detection is True

    def test_environments_cover_the_project_targets(self) -> None:
        assert {"browser", "desktop"} <= set(ENVIRONMENTS)

    def test_can_exclude_functions(self) -> None:
        tool = computer_use_tool("desktop", excluded_functions=["drag_and_drop"])
        assert tool.computer_use.excluded_predefined_functions == ["drag_and_drop"]

    def test_rejects_an_unknown_environment(self) -> None:
        with pytest.raises(ValueError, match="environment must be one of"):
            computer_use_tool("holodeck")


class TestMessageTranslation:
    def test_a_tool_result_becomes_a_function_response(self) -> None:
        contents = build_contents(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "", tool_calls=(ToolCall("click", {"x": 1}, "fc_1"),)),
                LLMMessage("tool", "clicked", tool_call_id="fc_1"),
            ]
        )
        response = contents[2].parts[0].function_response
        assert contents[2].role == "user"
        assert response.name == "click"
        assert response.id == "fc_1"
        assert response.response == {"output": "clicked"}

    def test_a_tool_result_carries_a_screenshot(self, png: bytes) -> None:
        contents = build_contents(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "", tool_calls=(ToolCall("screenshot", {}, "fc_1"),)),
                LLMMessage("tool", "captured", images=(png,), tool_call_id="fc_1"),
            ]
        )
        blobs = contents[2].parts[0].function_response.parts
        assert blobs[0].inline_data.mime_type == "image/png"
        assert blobs[0].inline_data.data == png

    def test_consecutive_tool_results_collapse_into_one_turn(self) -> None:
        contents = build_contents(
            [
                LLMMessage("user", "go"),
                LLMMessage(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("click", {}, "a"), ToolCall("type", {}, "b")),
                ),
                LLMMessage("tool", "clicked", tool_call_id="a"),
                LLMMessage("tool", "typed", tool_call_id="b"),
            ]
        )
        assert len(contents) == 3
        assert [p.function_response.name for p in contents[2].parts] == ["click", "type"]

    def test_an_unlabelled_result_after_a_single_call_is_matched(self) -> None:
        """Gemini does not always hand out call ids; one pending call is unambiguous."""
        contents = build_contents(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "", tool_calls=(ToolCall("click", {}, ""),)),
                LLMMessage("tool", "clicked"),
            ]
        )
        assert contents[2].parts[0].function_response.name == "click"

    def test_an_ambiguous_result_is_a_clear_error(self) -> None:
        with pytest.raises(ProviderError, match="cannot tell which function"):
            build_contents(
                [
                    LLMMessage("user", "go"),
                    LLMMessage(
                        "assistant",
                        "",
                        tool_calls=(ToolCall("click", {}, "a"), ToolCall("type", {}, "b")),
                    ),
                    LLMMessage("tool", "clicked"),
                ]
            )

    def test_an_assistant_turn_carries_its_function_calls(self) -> None:
        contents = build_contents(
            [
                LLMMessage("user", "go"),
                LLMMessage("assistant", "clicking", tool_calls=(ToolCall("click", {"x": 1}, "a"),)),
            ]
        )
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "clicking"
        assert contents[1].parts[1].function_call.name == "click"

    def test_an_entirely_empty_conversation_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="empty content"):
            build_contents([LLMMessage("user", "")])


class TestStopReasons:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (genai_types.FinishReason.STOP, "end"),
            (genai_types.FinishReason.MAX_TOKENS, "max_tokens"),
            (genai_types.FinishReason.SAFETY, "other"),
            (genai_types.FinishReason.MALFORMED_FUNCTION_CALL, "other"),
            (None, "end"),
        ],
    )
    def test_normalizes(self, raw: genai_types.FinishReason | None, expected: str) -> None:
        c = client(gemini_response(text="x", finish_reason=raw))  # type: ignore[arg-type]
        assert c.complete([LLMMessage("user", "hi")]).stop_reason == expected


class TestRetry:
    def test_retries_a_transient_failure_then_succeeds(self) -> None:
        slept: list[float] = []
        c = client(
            gemini_error(503),
            gemini_error(429),
            gemini_response(text="third time lucky"),
            sleep=slept.append,
        )
        assert c.complete([LLMMessage("user", "hi")]).text == "third time lucky"
        assert len(slept) == 2
        assert slept[0] < slept[1]

    def test_gives_up_after_max_attempts(self) -> None:
        c = client(*[gemini_error(503) for _ in range(3)], max_attempts=3)
        with pytest.raises(ProviderError, match="after 3 attempt"):
            c.complete([LLMMessage("user", "hi")])

    def test_rejects_a_nonsense_attempt_count(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            GeminiClient(client=StubGenai([]), max_attempts=0)

    @pytest.mark.parametrize("status", [429, 500, 503, 408])
    def test_retryable_statuses(self, status: int) -> None:
        assert is_retryable(gemini_error(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_non_retryable_statuses(self, status: int) -> None:
        assert not is_retryable(gemini_error(status))

    def test_a_connection_error_is_retryable(self) -> None:
        assert is_retryable(ConnectionError("reset by peer"))

    def test_an_unrelated_exception_is_not_retryable(self) -> None:
        assert not is_retryable(ValueError("nope"))


class TestErrorMapping:
    def test_a_non_retryable_error_raises_provider_error_with_the_cause(self) -> None:
        original = gemini_error(400, "invalid function declaration")
        c = client(original)
        with pytest.raises(ProviderError) as excinfo:
            c.complete([LLMMessage("user", "hi")])
        assert excinfo.value.__cause__ is original
        assert "invalid function declaration" in str(excinfo.value)

    def test_an_auth_error_is_not_retried(self) -> None:
        stub = StubGenai([gemini_error(403, "permission denied")])
        c = GeminiClient(client=stub, model=MODEL, sleep=lambda _s: None)
        with pytest.raises(ProviderError, match="after 1 attempt"):
            c.complete([LLMMessage("user", "hi")])
        assert len(stub.calls) == 1

    def test_an_unreadable_reply_raises_provider_error(self) -> None:
        class Broken:
            candidates = [object()]
            usage_metadata = None

        with pytest.raises(ProviderError, match="could not read"):
            client(Broken()).complete([LLMMessage("user", "hi")])
