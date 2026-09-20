"""The Gemini backend: the same ``contracts.LLMClient`` over ``google-genai``.

Interchangeable with the Claude adapter, sharing :mod:`skillweaver.llm.usage` and
:mod:`skillweaver.llm.cassette`, so a run can be repeated on a second model.

Three differences from Claude's surface. Gemini has ``"user"`` and ``"model"`` and no
tool role, so a ``"tool"`` message becomes a ``function_response`` part inside a user
turn. A function response is keyed by NAME, not only by id, and ``LLMMessage`` carries no
function name - which is why the whole conversation is walked in order to resolve it. And
sampling is real here: a non-``None`` ``temperature`` is forwarded rather than dropped.

**Verification status: this adapter has NEVER made a real call.** No ``GEMINI_API_KEY``
has been supplied, so the computer-use tool, the screenshot-carrying function response
and the finish-reason mapping follow the current docs and are unconfirmed by any live
round trip. The sibling Claude adapter IS live-proven; do not read its status as covering
this one.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from skillweaver.config import settings
from skillweaver.contracts import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm.usage import UsageMeter
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

ENVIRONMENTS: Final[Mapping[str, genai_types.Environment]] = MappingProxyType(
    {
        "browser": genai_types.Environment.ENVIRONMENT_BROWSER,
        "desktop": genai_types.Environment.ENVIRONMENT_DESKTOP,
        "mobile": genai_types.Environment.ENVIRONMENT_MOBILE,
    }
)
"""skillweaver target name -> the Gemini computer-use environment. The keys match
``Settings.default_target``, so a caller can pass the configured target straight through."""

_STOP_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "STOP": "end",
        "MAX_TOKENS": "max_tokens",
    }
)

#: HTTP statuses worth another attempt: rate limit, server error, gateway.
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})


def computer_use_tool(
    environment: str = "browser",
    *,
    enable_prompt_injection_detection: bool = True,
    excluded_functions: Sequence[str] = (),
) -> genai_types.Tool:
    """The Gemini computer-use tool, ready for :class:`GeminiClient`'s ``extra_tools``.

    Args:
        environment: A key of :data:`ENVIRONMENTS` (``"browser"``, ``"desktop"``,
            ``"mobile"``).
        enable_prompt_injection_detection: Leave on. The model then flags a page
            that tries to hijack the agent instead of following it.
        excluded_functions: Predefined actions to withhold, for a controller that
            cannot perform them.

    Raises:
        ValueError: on an unknown environment name.
    """
    try:
        env = ENVIRONMENTS[environment]
    except KeyError:
        raise ValueError(
            f"environment must be one of {sorted(ENVIRONMENTS)}, not {environment!r}"
        ) from None
    return genai_types.Tool(
        computer_use=genai_types.ComputerUse(
            environment=env,
            enable_prompt_injection_detection=enable_prompt_injection_detection,
            excluded_predefined_functions=list(excluded_functions) or None,
        )
    )


def is_retryable(exc: BaseException) -> bool:
    """True for a transient failure: a server error, a rate limit, or a connection
    problem. A 4xx that is not 408/409/429 means the request is wrong and is never
    retried."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return _status_of(exc) in _RETRYABLE_STATUS
    if isinstance(exc, genai_errors.APIError):
        status = _status_of(exc)
        return status is None or status >= 500 or status in _RETRYABLE_STATUS
    return isinstance(exc, (ConnectionError, TimeoutError))


def _status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


class GeminiClient:
    """A ``contracts.LLMClient`` backed by Gemini's ``generate_content``.

    Args:
        model: The model id. Defaults to ``settings().gemini_model``.
        api_key: An explicit key; ``None`` uses ``settings().gemini_api_key`` and
            then the SDK's own resolution (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``).
        client: A ready-made ``genai.Client``, or any object exposing
            ``models.generate_content(**kwargs)``.
        extra_tools: Provider-level tools appended to every request - notably
            :func:`computer_use_tool`.
        max_attempts / base_delay / max_delay: Retry with exponential backoff.
        sleep / jitter: Injectable, so a retry need not cost wall time.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        extra_tools: Sequence[genai_types.Tool] = (),
        max_attempts: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        resolved = settings()
        self._model = model or resolved.gemini_model
        self._extra_tools = tuple(extra_tools)
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._meter = UsageMeter()
        if client is not None:
            self._client = client
        else:
            key = api_key or resolved.gemini_api_key
            self._client = genai.Client(api_key=key) if key else genai.Client()

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

        config = genai_types.GenerateContentConfig(max_output_tokens=max_tokens)
        if system:
            config.system_instruction = system
        if temperature is not None:
            config.temperature = temperature
        wire_tools = self._build_tools(tools)
        if wire_tools:
            config.tools = wire_tools

        raw = self._call_with_retry(
            {
                "model": self._model,
                "contents": build_contents(messages),
                "config": config,
            }
        )
        return self._parse(raw)

    def total_usage(self) -> Usage:
        return self._meter.total()

    def name(self) -> str:
        return self._model

    # -- internals ---------------------------------------------------------

    def _build_tools(self, tools: Sequence[ToolSpec] | None) -> list[genai_types.Tool]:
        wire = list(self._extra_tools)
        declarations = [
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=dict(tool.input_schema),
            )
            for tool in tools or ()
        ]
        if declarations:
            wire.append(genai_types.Tool(function_declarations=declarations))
        return wire

    def _call_with_retry(self, request: Mapping[str, Any]) -> Any:
        """Call the API, retrying transient failures with exponential backoff and
        full jitter; everything else becomes a :class:`ProviderError` carrying the
        original exception as its cause."""
        last: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.models.generate_content(**request)
            except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
                last = exc
                if not is_retryable(exc) or attempt == self._max_attempts:
                    raise ProviderError(
                        f"Gemini call to {self._model} failed after {attempt} "
                        f"attempt(s): {type(exc).__name__}: {exc}"
                    ) from exc
                delay = min(self._base_delay * 2 ** (attempt - 1), self._max_delay) * self._jitter()
                log.warning(
                    "gemini.retry",
                    model=self._model,
                    attempt=attempt,
                    of=self._max_attempts,
                    delay_s=round(delay, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._sleep(delay)
        raise ProviderError(f"Gemini call to {self._model} exhausted retries") from last

    def _parse(self, raw: Any) -> LLMResponse:
        try:
            candidates = raw.candidates or ()
            if not candidates:
                raise ProviderError(
                    f"Gemini returned no candidates for {self._model}; "
                    f"prompt_feedback={getattr(raw, 'prompt_feedback', None)}"
                )
            candidate = candidates[0]
            texts: list[str] = []
            calls: list[ToolCall] = []
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or ():
                # A thought summary is still a text part; it is not the answer.
                if getattr(part, "thought", False):
                    continue
                if getattr(part, "text", None):
                    texts.append(part.text)
                call = getattr(part, "function_call", None)
                if call is not None and call.name:
                    calls.append(
                        ToolCall(name=call.name, args=dict(call.args or {}), id=call.id or "")
                    )
            metadata = getattr(raw, "usage_metadata", None)
            if metadata is None:
                # Silently pricing this at zero would under-report spend against
                # the run's Budget, which is the one number that must not drift.
                raise ProviderError(
                    f"Gemini returned a reply this adapter could not read: "
                    f"no usage_metadata on the {self._model} response"
                )
            usage = self._meter.record(
                self._model,
                int(getattr(metadata, "prompt_token_count", 0) or 0),
                int(getattr(metadata, "candidates_token_count", 0) or 0),
            )
        except ProviderError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"Gemini returned a reply this adapter could not read: {type(exc).__name__}: {exc}"
            ) from exc
        return LLMResponse(
            text="\n".join(texts),
            tool_calls=tuple(calls),
            usage=usage,
            stop_reason=_stop_reason(candidate, bool(calls)),
        )


def _stop_reason(candidate: Any, has_tool_calls: bool) -> str:
    """Normalize Gemini's ``finish_reason``.

    Gemini reports ``STOP`` for a turn that asked for a function call, where the contract
    wants ``"tool_use"``; safety, recitation and malformed-call reasons land on ``"other"``.
    """
    raw = getattr(candidate, "finish_reason", None)
    name = getattr(raw, "value", raw)
    reason = _STOP_REASONS.get(str(name or "STOP"), "other")
    if has_tool_calls and reason == "end":
        return "tool_use"
    return reason


def build_contents(messages: Sequence[LLMMessage]) -> list[genai_types.Content]:
    """Translate the conversation into Gemini ``Content`` turns.

    Consecutive ``"tool"`` messages collapse into one user turn, mirroring the Claude
    adapter so both backends see the same conversation shape.

    Raises:
        ProviderError: a tool result cannot be matched to the call it answers, which
            would otherwise be sent under the wrong function name.
    """
    names_by_id: dict[str, str] = {}
    pending_call_names: list[str] = []
    turns: list[genai_types.Content] = []
    pending_parts: list[genai_types.Part] = []

    def flush() -> None:
        if pending_parts:
            turns.append(genai_types.Content(role="user", parts=list(pending_parts)))
            pending_parts.clear()

    for message in messages:
        if message.role == "tool":
            name = _resolve_tool_name(message, names_by_id, pending_call_names)
            pending_parts.append(_function_response_part(message, name))
            continue
        flush()
        if message.role == "assistant":
            pending_call_names = [c.name for c in message.tool_calls]
            for call in message.tool_calls:
                if call.id:
                    names_by_id[call.id] = call.name
            parts = _assistant_parts(message)
        else:
            parts = _user_parts(message)
        if parts:
            turns.append(
                genai_types.Content(
                    role="model" if message.role == "assistant" else "user", parts=parts
                )
            )
    flush()
    if not turns:
        raise ProviderError("every message translated to empty content")
    return turns


def _resolve_tool_name(
    message: LLMMessage, names_by_id: Mapping[str, str], pending: Sequence[str]
) -> str:
    if message.tool_call_id and message.tool_call_id in names_by_id:
        return names_by_id[message.tool_call_id]
    # Gemini does not always hand out call ids. When the preceding assistant turn
    # made exactly one call there is no ambiguity to resolve.
    if len(pending) == 1:
        return pending[0]
    raise ProviderError(
        f"cannot tell which function this tool result answers: tool_call_id="
        f"{message.tool_call_id!r}, candidates={list(pending)}. "
        "Set LLMMessage.tool_call_id to the id of the ToolCall it answers."
    )


def _user_parts(message: LLMMessage) -> list[genai_types.Part]:
    parts = [genai_types.Part.from_bytes(data=img, mime_type="image/png") for img in message.images]
    if message.text:
        parts.append(genai_types.Part(text=message.text))
    return parts


def _assistant_parts(message: LLMMessage) -> list[genai_types.Part]:
    parts: list[genai_types.Part] = []
    if message.text:
        parts.append(genai_types.Part(text=message.text))
    parts.extend(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name=call.name, args=dict(call.args), id=call.id or None
            )
        )
        for call in message.tool_calls
    )
    return parts


def _function_response_part(message: LLMMessage, name: str) -> genai_types.Part:
    """One ``function_response`` part. A screenshot returned by a computer-use
    action rides along as an inline blob inside the same response."""
    blobs = [
        genai_types.FunctionResponsePart(
            inline_data=genai_types.FunctionResponseBlob(mime_type="image/png", data=img)
        )
        for img in message.images
    ]
    return genai_types.Part(
        function_response=genai_types.FunctionResponse(
            id=message.tool_call_id or None,
            name=name,
            response={"output": message.text},
            parts=blobs or None,
        )
    )
