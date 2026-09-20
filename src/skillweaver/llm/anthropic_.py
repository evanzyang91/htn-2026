"""The Claude backend: ``contracts.LLMClient`` over the Anthropic Messages API.

Translates the provider-neutral ``LLMMessage`` conversation into Messages API content
blocks and back, so nothing above this module knows an Anthropic SDK type.

Sampling parameters are DROPPED, not forwarded: current Claude models reject
``temperature``, and the contract says an adapter drops it rather than failing, so a
caller written against Gemini's surface still works here.

Thinking is left at the MODEL DEFAULT. On Claude Opus 5 that is adaptive thinking, which
- unlike the older fixed-budget mode - does not require the caller to echo thinking
blocks back on the next turn. That is what lets ``LLMMessage`` stay lossy.

Retries are OURS: the SDK client is built with ``max_retries=0`` so the backoff here is
the only one. Anything not retryable is re-raised as ``ProviderError``.

Proven against the live API on 2026-09-19 for a plain completion, one with an attached
screenshot, and one returning a tool call. NOT yet exercised live: the computer-use
toolset entry (:data:`COMPUTER_USE_TOOL`), ``effort``, and the ``workspace_id`` header.
"""

from __future__ import annotations

import base64
import random
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

import anthropic

from skillweaver.config import settings
from skillweaver.contracts import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm.usage import UsageMeter
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

COMPUTER_USE_TOOL: Final[Mapping[str, Any]] = MappingProxyType(
    {"type": "computer_toolset_20260801"}
)
"""The computer-use toolset entry, ready to put in a request's ``tools``.

The 2026-08-01 toolset is schema-less on purpose: member names are fixed by the version
and coordinates are always in the pixel space of the screenshots you send back, so
``name``, ``display_width_px``, ``display_height_px`` and ``display_number`` are all
REJECTED. It needs no beta header. Pass ``computer_use=True`` rather than hand-rolling it."""

COMPUTER_USE_ACTIONS: Final[tuple[str, ...]] = (
    "screenshot",
    "zoom",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "mouse_move",
    "left_mouse_down",
    "left_mouse_up",
    "cursor_position",
    "scroll",
    "type",
    "key",
    "hold_key",
    "wait",
)
"""Every member tool of :data:`COMPUTER_USE_TOOL`, in the order the docs list them.

A returned ``ToolCall`` uses one as its ``name``. Coordinates arrive in the pixel space
of the screenshot that was sent - which, since ``Screenshot.to_array`` hands out a
logical-size image, is LOGICAL pixels. Do not rescale them."""

COMPUTER_USE_MODELS: Final[frozenset[str]] = frozenset(
    {
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
    }
)
"""Models that accept :data:`COMPUTER_USE_TOOL`. Older models take the previous
``computer_20251124`` tool instead, which this adapter does not offer."""

_STOP_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "end_turn": "end",
        "stop_sequence": "end",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
    }
)

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 429})


def is_retryable(exc: BaseException) -> bool:
    """True for a transient failure worth another attempt: a connection or timeout
    error, a rate limit, or a 5xx/408/409 from the API. A 400, 401, 403 or 404 is
    a bug in the request and is never retried."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500 or exc.status_code in _RETRYABLE_STATUS
    return False


class AnthropicClient:
    """A ``contracts.LLMClient`` backed by the Anthropic Messages API.

    Args:
        model: The model id. Defaults to ``settings().claude_model``.
        api_key: An explicit key. ``None`` falls back to ``settings().anthropic_api_key``
            (which reads both the environment and ``.env``) and then to the SDK's own
            resolution (auth token, stored profile).
        client: A ready-made SDK client, or any object exposing
            ``messages.create(**kwargs)``. When given, ``api_key``, ``workspace_id``
            and ``timeout`` are ignored.
        workspace_id: Sent as the ``anthropic-workspace-id`` header. Required when
            the key is scoped to an organization rather than a workspace - such a
            key gets a 400 on EVERY endpoint without it, ``models.list`` included.
        computer_use: Append :data:`COMPUTER_USE_TOOL` to every request's tools.
        effort: ``"low"``..``"max"``, sent as ``output_config.effort``. ``None``
            leaves the model default (``"high"``).
        max_attempts: Total tries per call, including the first. ``1`` disables retry.
        base_delay / max_delay: Exponential backoff bounds in seconds.
        timeout: Per-request timeout in seconds for a client built here.
        sleep / jitter: Injectable, so a retry need not cost wall time.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        workspace_id: str | None = None,
        computer_use: bool = False,
        effort: str | None = None,
        max_attempts: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        timeout: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        resolved = settings()
        self._model = model or resolved.claude_model
        self._computer_use = computer_use
        self._effort = effort
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._meter = UsageMeter()
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {"max_retries": 0}
            # settings() resolves ANTHROPIC_API_KEY from the process environment
            # AND from .env; the SDK only looks at the environment, so a key that
            # lives only in .env never reaches it unless it is passed explicitly.
            # Leaving this to the SDK cost a failed live run. Falling through with
            # no key is still correct: the SDK then tries its own auth token and
            # stored profile.
            resolved_key = api_key or resolved.anthropic_api_key
            if resolved_key is not None:
                kwargs["api_key"] = resolved_key
            if workspace_id:
                kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._client = anthropic.Anthropic(**kwargs)
        if computer_use and self._model not in COMPUTER_USE_MODELS:
            log.warning(
                "anthropic.computer_use_unsupported",
                model=self._model,
                supported=",".join(sorted(COMPUTER_USE_MODELS)),
            )

    # -- LLMClient ---------------------------------------------------------

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ProviderError("complete() needs at least one message")
        if temperature is not None:
            log.debug("anthropic.temperature_dropped", model=self._model, value=temperature)

        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": build_messages(messages),
        }
        if system:
            request["system"] = system
        wire_tools = self._build_tools(tools)
        if wire_tools:
            request["tools"] = wire_tools
        if self._effort is not None:
            request["output_config"] = {"effort": self._effort}

        raw = self._call_with_retry(request)
        return self._parse(raw)

    def total_usage(self) -> Usage:
        return self._meter.total()

    def name(self) -> str:
        return self._model

    # -- internals ---------------------------------------------------------

    def _build_tools(self, tools: Sequence[ToolSpec] | None) -> list[Any]:
        wire: list[Any] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in tools or ()
        ]
        if self._computer_use:
            wire.append(dict(COMPUTER_USE_TOOL))
        return wire

    def _call_with_retry(self, request: Mapping[str, Any]) -> Any:
        """Call the API, retrying transient failures with exponential backoff and
        full jitter. Anything else - and a transient failure that outlives the
        attempts - becomes a :class:`ProviderError` with the cause attached."""
        last: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.messages.create(**request)
            except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
                last = exc
                if not is_retryable(exc) or attempt == self._max_attempts:
                    raise ProviderError(
                        f"Anthropic call to {self._model} failed after {attempt} "
                        f"attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                delay = min(self._base_delay * 2 ** (attempt - 1), self._max_delay)
                delay *= self._jitter()
                log.warning(
                    "anthropic.retry",
                    model=self._model,
                    attempt=attempt,
                    of=self._max_attempts,
                    delay_s=round(delay, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._sleep(delay)
        raise ProviderError(f"Anthropic call to {self._model} exhausted retries") from last

    def _parse(self, raw: Any) -> LLMResponse:
        try:
            texts: list[str] = []
            calls: list[ToolCall] = []
            for block in raw.content:
                kind = getattr(block, "type", None)
                if kind == "text":
                    texts.append(block.text)
                elif kind == "tool_use":
                    calls.append(
                        ToolCall(name=block.name, args=dict(block.input or {}), id=block.id or "")
                    )
            usage = self._meter.record(
                self._model, _input_tokens(raw.usage), int(raw.usage.output_tokens or 0)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"Anthropic returned a reply this adapter could not read: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return LLMResponse(
            text="\n".join(texts),
            tool_calls=tuple(calls),
            usage=usage,
            stop_reason=_STOP_REASONS.get(getattr(raw, "stop_reason", "") or "", "other"),
        )


def _input_tokens(usage: Any) -> int:
    """Total billed input tokens. Cache reads and writes are folded in at the plain
    input rate; skillweaver does not use prompt caching, and over-reporting is the
    safe direction for a budget."""
    return (
        int(getattr(usage, "input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )


def build_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Translate the conversation into Messages API turns.

    Consecutive ``"tool"`` messages collapse into ONE user turn. That is not a
    tidiness choice: splitting ``tool_result`` blocks across several user messages
    silently teaches Claude to stop issuing parallel tool calls.
    """
    turns: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            turns.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if message.role == "tool":
            pending_results.append(_tool_result_block(message))
            continue
        flush()
        if message.role == "assistant":
            content = _assistant_content(message)
        else:
            content = _user_content(message)
        if content:
            turns.append({"role": message.role, "content": content})
    flush()
    if not turns:
        raise ProviderError("every message translated to empty content")
    return turns


def _user_content(message: LLMMessage) -> list[dict[str, Any]]:
    # Images first: the model reads a screenshot better when the question follows it.
    content: list[dict[str, Any]] = [_image_block(img) for img in message.images]
    if message.text:
        content.append({"type": "text", "text": message.text})
    return content


def _assistant_content(message: LLMMessage) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    content.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.args)}
        for call in message.tool_calls
    )
    return content


def _tool_result_block(message: LLMMessage) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    content.extend(_image_block(img) for img in message.images)
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id or "",
        "content": content,
    }


def _image_block(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }
