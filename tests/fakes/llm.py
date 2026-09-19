"""Fake reasoning: ``FakeLLM`` (scripted replies), ``FakeCritic`` (scripted verdicts)
and ``FakeEmbedder`` (deterministic hash vectors). None of them touch a network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from skillweaver.contracts import (
    Fingerprint,
    LLMMessage,
    LLMResponse,
    Observation,
    ToolSpec,
    Usage,
    Verdict,
)


class ScriptExhausted(AssertionError):
    """A scripted fake was called more times than its script allows.

    An ``AssertionError`` on purpose: it is a test failure, and code under test that
    catches ``SkillWeaverError`` or ``ProviderError`` must not swallow it.
    """


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """The arguments of one recorded ``FakeLLM.complete`` call."""

    messages: tuple[LLMMessage, ...]
    system: str | None
    tools: tuple[ToolSpec, ...] | None
    max_tokens: int
    temperature: float | None


class FakeLLM:
    """A ``contracts.LLMClient`` that replays a script.

    Args:
        responses: The replies to return, in order. A plain ``str`` is shorthand for
            ``LLMResponse(text=...)``. Default: an empty script, i.e. a model that
            must never be called.
        model: What :meth:`name` reports.

    Attributes:
        calls: EXACTLY the number of times ``complete`` has been invoked, starting
            at ``0``. It counts every invocation, including one that raised
            :class:`ScriptExhausted`. ``assert fake_llm.calls == 0`` proves no model
            was consulted.
        requests: One :class:`LLMRequest` per invocation, in order.

    Raises:
        ScriptExhausted: from ``complete`` once the script has run out.
    """

    def __init__(self, responses: Sequence[LLMResponse | str] = (), model: str = "fake-llm"):
        self._script = [LLMResponse(text=r) if isinstance(r, str) else r for r in responses]
        self._model = model
        self._usage = Usage()
        self.calls: int = 0
        self.requests: list[LLMRequest] = []

    @property
    def remaining(self) -> int:
        """How many scripted responses have not been served yet."""
        return max(len(self._script) - self.calls, 0)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.requests.append(
            LLMRequest(
                tuple(messages),
                system,
                None if tools is None else tuple(tools),
                max_tokens,
                temperature,
            )
        )
        if self.calls > len(self._script):
            raise ScriptExhausted(
                f"FakeLLM script exhausted: call #{self.calls} but only "
                f"{len(self._script)} response(s) were scripted"
            )
        response = self._script[self.calls - 1]
        self._usage = self._usage + response.usage
        return response

    def total_usage(self) -> Usage:
        """Sum of the scripted responses' usage; ``calls`` is the number served."""
        served = min(self.calls, len(self._script))
        u = self._usage
        return Usage(u.input_tokens, u.output_tokens, served, u.cost_usd)

    def name(self) -> str:
        return self._model


@dataclass(frozen=True, slots=True)
class CriticCall:
    """The arguments of one recorded ``FakeCritic.judge`` call."""

    goal: str
    before: Observation
    after: Observation
    expectation: str | None


class FakeCritic:
    """A ``contracts.Critic`` with three modes, tried in this order:

    1. ``verdicts``: scripted verdicts, served in order until they run out.
    2. ``goal``: a fingerprint; the verdict is ``ok`` exactly when
       ``after.fingerprint == goal`` (``source="programmatic"``).
    3. ``default``: a fixed verdict.

    With none of the later modes configured, running out of script raises
    :class:`ScriptExhausted`. ``calls`` records every ``judge`` invocation.
    """

    def __init__(
        self,
        verdicts: Sequence[Verdict] = (),
        *,
        goal: Fingerprint | None = None,
        default: Verdict | None = None,
    ) -> None:
        self._script = list(verdicts)
        self._goal = goal
        self._default = default
        self.calls: list[CriticCall] = []

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> Verdict:
        self.calls.append(CriticCall(goal, before, after, expectation))
        if len(self.calls) <= len(self._script):
            return self._script[len(self.calls) - 1]
        if self._goal is not None:
            ok = after.fingerprint == self._goal
            return Verdict(ok, "reached goal state" if ok else "not at goal state", 1.0)
        if self._default is not None:
            return self._default
        raise ScriptExhausted(
            f"FakeCritic script exhausted: call #{len(self.calls)} but only "
            f"{len(self._script)} verdict(s) were scripted"
        )


class FakeEmbedder:
    """A ``contracts.Embedder`` built from hashed bags of words.

    Each lower-cased word is hashed (SHA-256, so stable across processes) to one of
    ``dim`` signed buckets; the vector is L2-normalized. Texts sharing words are
    therefore genuinely closer, which is enough to test retrieval ranking. A text
    with no words maps to the first unit vector. ``calls`` counts ``embed`` calls.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return [1.0] + [0.0] * (self.dim - 1)
        return [v / norm for v in vector]
