"""Provider-SDK doubles for the two adapters.

These stand in for the transport only: they are shaped like the real SDK client
(``messages.create`` / ``models.generate_content``) and they hand back REAL SDK
response objects, so every translation path in the adapters - content blocks,
image parts, tool calls, usage, stop reasons - runs against the real types.
Nothing here touches a network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def anthropic_message(
    *,
    text: str = "",
    tool_calls: Sequence[tuple[str, str, dict[str, Any]]] = (),
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    model: str = "claude-opus-5",
) -> Message:
    """A real ``anthropic.types.Message``. ``tool_calls`` items are ``(id, name, input)``."""
    content: list[Any] = []
    if text:
        content.append(TextBlock(type="text", text=text))
    content.extend(
        ToolUseBlock(type="tool_use", id=call_id, name=name, input=args)
        for call_id, name, args in tool_calls
    )
    return Message(
        id="msg_stub",
        model=model,
        role="assistant",
        type="message",
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def anthropic_error(status: int, message: str = "boom") -> anthropic.APIStatusError:
    """A real SDK error carrying ``status``, so ``is_retryable`` sees what it would
    see in production."""
    response = httpx2.Response(status, request=_REQUEST, json={"error": {"message": message}})
    cls = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
    }.get(status, anthropic.InternalServerError)
    return cls(message, response=response, body=None)


def anthropic_connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_REQUEST)


@dataclass
class StubMessages:
    """The ``client.messages`` namespace."""

    script: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(
                f"stub Anthropic client ran out of script on call #{len(self.calls)}"
            )
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@dataclass
class StubAnthropic:
    """An object shaped like ``anthropic.Anthropic`` for ``AnthropicClient(client=...)``.

    ``script`` is served in order; an exception in it is raised instead of returned,
    which is how a transient failure gets injected.
    """

    script: list[Any]

    def __post_init__(self) -> None:
        self.messages = StubMessages(list(self.script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


def gemini_response(
    *,
    text: str = "",
    tool_calls: Sequence[tuple[str | None, str, dict[str, Any]]] = (),
    finish_reason: genai_types.FinishReason = genai_types.FinishReason.STOP,
    prompt_tokens: int = 10,
    output_tokens: int = 5,
) -> genai_types.GenerateContentResponse:
    """A real ``GenerateContentResponse``. ``tool_calls`` items are ``(id, name, args)``."""
    parts: list[genai_types.Part] = []
    if text:
        parts.append(genai_types.Part(text=text))
    parts.extend(
        genai_types.Part(function_call=genai_types.FunctionCall(id=cid, name=name, args=args))
        for cid, name, args in tool_calls
    )
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
        ),
    )


def gemini_error(status: int, message: str = "boom") -> genai_errors.APIError:
    cls = genai_errors.ServerError if status >= 500 else genai_errors.ClientError
    return cls(status, {"error": {"message": message, "status": str(status)}})


@dataclass
class StubModels:
    """The ``client.models`` namespace."""

    script: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(f"stub Gemini client ran out of script on call #{len(self.calls)}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@dataclass
class StubGenai:
    """An object shaped like ``genai.Client`` for ``GeminiClient(client=...)``."""

    script: list[Any]

    def __post_init__(self) -> None:
        self.models = StubModels(list(self.script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.models.calls


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

#: A 1x1 red PNG - the smallest thing that is genuinely a PNG.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d4944415478da63f8cfc0000003010100185dd1d20000000049454e44ae426082"
)


@pytest.fixture
def png() -> bytes:
    """One-pixel PNG bytes, for the image-attachment paths."""
    return PNG_1X1


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure nothing in this package can reach a provider by accident, and that
    the scrubber's environment-secret path has known values to work with."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
