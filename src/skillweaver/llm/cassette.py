"""Record and replay LLM calls, so tests can exercise real provider replies offline.

A cassette is one JSON file under ``tests/fixtures/cassettes/`` holding
``(request, response)`` pairs. :class:`CassetteClient` is itself an
``LLMClient``, so it wraps :class:`~skillweaver.llm.anthropic_.AnthropicClient` or
:class:`~skillweaver.llm.gemini_.GeminiClient` transparently and the code under
test cannot tell which - or whether anything is behind it at all.

Two properties make this trustworthy:

**The digest is provider-neutral.** It is computed from the ``complete()``
arguments, not from provider wire bytes, so the same cassette shape works for
every adapter and a request that "looks the same" to skillweaver replays.

**Scrubbing happens before the digest, not after.** The recorded request is
scrubbed first and then hashed, so replay matches on the scrubbed form. A key that
leaked into a prompt is redacted on disk *and* the cassette still matches when the
same call is made with a different key.

Images are recorded by SHA-256 and byte length rather than inline: a screenshot
does not change what reply to replay, and a megabyte of base64 per turn makes the
fixtures unreviewable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from skillweaver.contracts import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from skillweaver.errors import ProviderError
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

REDACTED = "<redacted>"
"""What every scrubbed value becomes. Stable, so digests stay stable."""

CASSETTE_VERSION = 1

Mode = Literal["record", "replay"]

#: Mapping keys whose *value* is a credential whatever it looks like.
_SECRET_KEYS = re.compile(
    r"(?i)(api[_-]?key|apikey|authorization|auth[_-]?token|access[_-]?token|"
    r"x-api-key|secret|password|passwd|credential|bearer|session[_-]?token)"
)

#: Values that are recognisably credentials wherever they appear - including
#: inside a prompt, which is the leak the scrubber actually exists for.
_SECRET_VALUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    re.compile(r"AIza[A-Za-z0-9_\-]{10,}"),  # Google
    re.compile(r"ya29\.[A-Za-z0-9_\-]{10,}"),  # Google OAuth
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}"),  # generic provider key
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
)

#: Environment variables whose exact value is scrubbed wherever it appears.
_SECRET_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    # The Jev policy and the text model it is paired with; see skillweaver.llm.jev_.
    # TYPESAFE_API_KEY has no recognisable shape, so the literal value is the only
    # thing that can redact it and this is where that value is named.
    "TYPESAFE_API_KEY",
    "TEXT_MODEL_API_KEY",
)

_extra_secrets: set[str] = set()
_secret_lock = threading.Lock()


def register_secret(value: str) -> None:
    """Also scrub this exact string. Use it for a credential that is not in the
    environment and does not match a known key shape. Values shorter than 8
    characters are ignored, so registering ``""`` cannot redact everything."""
    if len(value) < 8:
        return
    with _secret_lock:
        _extra_secrets.add(value)


def _literal_secrets() -> tuple[str, ...]:
    with _secret_lock:
        extra = tuple(_extra_secrets)
    env = tuple(v for name in _SECRET_ENV_VARS if (v := os.environ.get(name)) and len(v) >= 8)
    return env + extra


def scrub(value: Any) -> Any:
    """Return ``value`` with every credential replaced by :data:`REDACTED`.

    Recurses through mappings and sequences. A mapping key that names a credential
    redacts its whole value; any string is additionally scanned for known key
    shapes and for the exact value of a credential in the environment or
    registered with :func:`register_secret`.
    """
    return _scrub(value, _literal_secrets())


def _scrub(value: Any, literals: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): (REDACTED if _SECRET_KEYS.search(str(k)) else _scrub(v, literals))
            for k, v in value.items()
        }
    if isinstance(value, str):
        return _scrub_text(value, literals)
    if isinstance(value, (list, tuple)):
        return [_scrub(v, literals) for v in value]
    return value


def _scrub_text(text: str, literals: tuple[str, ...]) -> str:
    for literal in literals:
        text = text.replace(literal, REDACTED)
    for pattern in _SECRET_VALUES:
        text = pattern.sub(REDACTED, text)
    return text


# ---------------------------------------------------------------------------
# The provider-neutral request payload and its digest
# ---------------------------------------------------------------------------


def request_payload(
    *,
    model: str,
    messages: Sequence[LLMMessage],
    system: str | None,
    tools: Sequence[ToolSpec] | None,
    max_tokens: int,
    temperature: float | None,
) -> dict[str, Any]:
    """The JSON-able, already-scrubbed form of one ``complete()`` call."""
    payload = {
        "model": model,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [_message_to_dict(m) for m in messages],
        "tools": None if tools is None else [_tool_to_dict(t) for t in tools],
    }
    return scrub(payload)


def digest(payload: Mapping[str, Any]) -> str:
    """A stable SHA-256 over an already-scrubbed payload.

    ``sort_keys`` makes it independent of dict ordering, so a cassette recorded by
    one adapter matches a request built by another.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _message_to_dict(message: LLMMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "text": message.text,
        "images": [_image_ref(b) for b in message.images],
        "tool_calls": [_tool_call_to_dict(c) for c in message.tool_calls],
        "tool_call_id": message.tool_call_id,
    }


def _image_ref(data: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _tool_to_dict(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": json.loads(json.dumps(tool.input_schema)),
    }


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "args": json.loads(json.dumps(call.args)), "id": call.id}


def response_to_dict(response: LLMResponse) -> dict[str, Any]:
    """The JSON-able form of an :class:`LLMResponse` (not scrubbed; callers do that)."""
    return {
        "text": response.text,
        "tool_calls": [_tool_call_to_dict(c) for c in response.tool_calls],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "calls": response.usage.calls,
            "cost_usd": response.usage.cost_usd,
        },
        "stop_reason": response.stop_reason,
    }


def response_from_dict(data: Mapping[str, Any]) -> LLMResponse:
    """Rebuild an :class:`LLMResponse` written by :func:`response_to_dict`."""
    usage = data.get("usage") or {}
    return LLMResponse(
        text=data.get("text", ""),
        tool_calls=tuple(
            ToolCall(name=c["name"], args=c.get("args") or {}, id=c.get("id", ""))
            for c in data.get("tool_calls") or ()
        ),
        usage=Usage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            calls=int(usage.get("calls", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
        ),
        stop_reason=data.get("stop_reason", "end"),
    )


# ---------------------------------------------------------------------------
# The cassette file
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interaction:
    """One recorded ``(request, response)`` pair, keyed by the request digest."""

    digest: str
    request: Mapping[str, Any]
    response: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "request": dict(self.request),
            "response": dict(self.response),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Interaction:
        return cls(data["digest"], data["request"], data["response"])


@dataclass(slots=True)
class Cassette:
    """The interactions in one JSON file, plus the model they were recorded from."""

    path: Path
    model: str = ""
    interactions: list[Interaction] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str) -> Cassette:
        """Read a cassette. A missing file gives an empty one, so recording into a
        new path just works.

        Raises:
            ProviderError: if the file exists but is not a cassette this code can read.
        """
        path = Path(path)
        if not path.is_file():
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = int(data.get("version", 0))
            if version != CASSETTE_VERSION:
                raise ProviderError(
                    f"cassette {path} has version {version}, expected {CASSETTE_VERSION}"
                )
            return cls(
                path=path,
                model=data.get("model", ""),
                interactions=[Interaction.from_dict(i) for i in data.get("interactions", [])],
            )
        except ProviderError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError(f"cassette {path} is unreadable: {exc}") from exc

    def save(self) -> None:
        """Write the cassette, creating parent directories. Trailing newline and
        two-space indent so a fixture stays reviewable in a diff."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": CASSETTE_VERSION,
            "model": self.model,
            "interactions": [i.to_dict() for i in self.interactions],
        }
        self.path.write_text(
            json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def find(self, request_digest: str) -> Interaction | None:
        """The first interaction with this digest, or ``None``."""
        for interaction in self.interactions:
            if interaction.digest == request_digest:
                return interaction
        return None

    def append(self, interaction: Interaction) -> None:
        """Add an interaction, replacing any earlier one with the same digest so
        re-recording does not grow the file."""
        self.interactions = [i for i in self.interactions if i.digest != interaction.digest]
        self.interactions.append(interaction)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class CassetteClient:
    """An ``LLMClient`` that records what it forwards, or replays what was recorded.

    Args:
        path: The cassette file.
        mode: ``"record"`` forwards to ``inner`` and saves the pair; ``"replay"``
            never touches ``inner`` (which may be ``None``).
        inner: The real client. Required in ``"record"`` mode.
        model: The model name reported by :meth:`name` in ``"replay"`` mode when
            the cassette does not carry one.

    Raises:
        ProviderError: in ``"replay"`` mode when no recorded request matches, with
            the unmatched digest and a readable summary of the request in the message.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        mode: Mode = "replay",
        inner: Any | None = None,
        model: str = "",
    ) -> None:
        if mode not in ("record", "replay"):
            raise ValueError(f"mode must be 'record' or 'replay', not {mode!r}")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs an inner client to forward to")
        self._mode: Mode = mode
        self._inner = inner
        self._cassette = Cassette.load(path)
        self._lock = threading.Lock()
        if inner is not None:
            self._cassette.model = inner.name()
        elif model:
            self._cassette.model = self._cassette.model or model
        self._meter_total = Usage()

    @property
    def cassette(self) -> Cassette:
        """The loaded cassette, for assertions in tests."""
        return self._cassette

    @property
    def mode(self) -> Mode:
        return self._mode

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = request_payload(
            model=self.name(),
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        key = digest(payload)
        if self._mode == "replay":
            return self._replay(key, payload)
        return self._record(key, payload, messages, system, tools, max_tokens, temperature)

    def _replay(self, key: str, payload: Mapping[str, Any]) -> LLMResponse:
        interaction = self._cassette.find(key)
        if interaction is None:
            raise ProviderError(
                f"no recorded interaction in {self._cassette.path} matches request "
                f"{key[:12]}: {describe_request(payload)}. "
                f"The cassette holds {len(self._cassette.interactions)} interaction(s): "
                f"{', '.join(i.digest[:12] for i in self._cassette.interactions) or 'none'}. "
                f"Re-record it, or fix the request so it matches."
            )
        response = response_from_dict(interaction.response)
        with self._lock:
            self._meter_total = self._meter_total + response.usage
        return response

    def _record(
        self,
        key: str,
        payload: Mapping[str, Any],
        messages: Sequence[LLMMessage],
        system: str | None,
        tools: Sequence[ToolSpec] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> LLMResponse:
        assert self._inner is not None  # guarded in __init__
        response = self._inner.complete(
            messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        with self._lock:
            self._cassette.model = self._inner.name()
            self._cassette.append(Interaction(key, payload, scrub(response_to_dict(response))))
            self._cassette.save()
        log.info("cassette.record", path=str(self._cassette.path), digest=key[:12])
        return response

    def total_usage(self) -> Usage:
        """In ``record`` mode the inner client's total; in ``replay`` mode the sum of
        the usage of the interactions served so far."""
        if self._inner is not None:
            return self._inner.total_usage()
        with self._lock:
            return self._meter_total

    def name(self) -> str:
        return self._inner.name() if self._inner is not None else self._cassette.model


def describe_request(payload: Mapping[str, Any]) -> str:
    """A one-line, already-scrubbed summary of a request, for an error message."""
    messages: Iterable[Mapping[str, Any]] = payload.get("messages") or ()
    parts = []
    for message in messages:
        text = str(message.get("text", ""))
        snippet = text if len(text) <= 40 else text[:37] + "..."
        extras = []
        if message.get("images"):
            extras.append(f"{len(message['images'])} image(s)")
        if message.get("tool_calls"):
            extras.append(f"{len(message['tool_calls'])} tool call(s)")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        parts.append(f"{message.get('role')}={snippet!r}{suffix}")
    tools = payload.get("tools")
    tool_names = "" if not tools else " tools=" + ",".join(str(t.get("name")) for t in tools)
    return (
        f"model={payload.get('model')!r} max_tokens={payload.get('max_tokens')}"
        f"{tool_names} messages=[{'; '.join(parts)}]"
    )
