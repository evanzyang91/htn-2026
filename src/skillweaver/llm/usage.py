"""Token and money accounting for LLM calls.

Every adapter builds its per-call :class:`~skillweaver.contracts.Usage` here, so
there is exactly one pricing table and exactly one rule for an unknown model:
**report zero cost and warn, never guess.** A wrong price is worse than a missing
one because it silently corrupts the budget a run is checked against
(``Spend.check``), and a zero that logged a warning is at least findable.

Accumulation lives in :class:`UsageMeter`, which every adapter uses to answer
``LLMClient.total_usage()``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass

from skillweaver.contracts import Usage
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

_TOKENS_PER_UNIT = 1_000_000.0

# A trailing dated snapshot: "claude-opus-4-5-20251101", "claude-opus-4-5@20251101".
_SNAPSHOT_SUFFIX = re.compile(r"[-@]\d{8}$")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """US dollars per one million tokens, input and output."""

    input_per_mtok: float
    output_per_mtok: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Dollars for one call of this size."""
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / _TOKENS_PER_UNIT


# ---------------------------------------------------------------------------
# The pricing table.
#
# Anthropic rows: platform.claude.com pricing, as carried by the bundled
# `claude-api` skill's model table (cached 2026-06-24), read 2026-09-19.
# Google rows: ai.google.dev/gemini-api/docs/pricing, paid tier, read 2026-09-19.
#
# Two deliberate simplifications, both of which make us over-report cost rather
# than under-report it (the safe direction for a budget):
#   * Cache reads and writes are billed at the plain input rate. skillweaver does
#     not use prompt caching yet; when it does, this table needs a third column.
#   * Gemini 2.5 Pro's >200k-token tier and the promotional Gemini prices that
#     step up on 2027-01-01 are not modeled. The 2027 prices are the ones listed
#     here, so a run priced today is priced at the higher of the two rates.
#
# Add a model by adding a row. Do NOT add prefix matching: an unpriced model must
# fall through to the zero-and-warn path rather than borrow a neighbour's price.
# ---------------------------------------------------------------------------
PRICING: Mapping[str, ModelPrice] = {
    # Anthropic
    "claude-fable-5-1": ModelPrice(10.00, 50.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-mythos-5-1": ModelPrice(10.00, 50.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    # Google
    "gemini-2.5-computer-use-preview-10-2025": ModelPrice(2.00, 10.00),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
}

_warned: set[str] = set()
_warn_lock = threading.Lock()


def normalize_model(model: str) -> str:
    """Strip a trailing dated snapshot so ``claude-opus-4-5-20251101`` prices as
    ``claude-opus-4-5``. Any other name is returned unchanged."""
    return _SNAPSHOT_SUFFIX.sub("", model)


def price_for(model: str) -> ModelPrice | None:
    """The price of ``model``, or ``None`` when it is not in :data:`PRICING`."""
    return PRICING.get(model) or PRICING.get(normalize_model(model))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollars for one call. An unpriced model costs ``0.0`` and logs
    ``usage.unknown_model`` at WARNING - once per model per process, so a long run
    does not drown in it."""
    price = price_for(model)
    if price is None:
        _warn_once(model)
        return 0.0
    return price.cost_usd(input_tokens, output_tokens)


def usage_for(model: str, input_tokens: int, output_tokens: int, *, calls: int = 1) -> Usage:
    """Build the :class:`Usage` of one call, priced by :func:`cost_usd`."""
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=calls,
        cost_usd=cost_usd(model, input_tokens, output_tokens),
    )


def _warn_once(model: str) -> None:
    with _warn_lock:
        if model in _warned:
            return
        _warned.add(model)
    log.warning(
        "usage.unknown_model",
        model=model,
        cost_usd=0.0,
        hint="add it to skillweaver.llm.usage.PRICING",
    )


def reset_unknown_model_warnings() -> None:
    """Forget which models have been warned about. For tests only."""
    with _warn_lock:
        _warned.clear()


class UsageMeter:
    """A running total of :class:`Usage`, safe to share between threads.

    ``Usage`` itself is immutable and adds with ``+``; this only owns the mutable
    running sum so every adapter answers ``total_usage()`` the same way.
    """

    __slots__ = ("_lock", "_total")

    def __init__(self, total: Usage | None = None) -> None:
        self._total = total or Usage()
        self._lock = threading.Lock()

    def add(self, usage: Usage) -> Usage:
        """Add ``usage`` to the total and return the new total."""
        with self._lock:
            self._total = self._total + usage
            return self._total

    def record(self, model: str, input_tokens: int, output_tokens: int) -> Usage:
        """Price one call, add it, and return **that call's** usage (not the total)."""
        usage = usage_for(model, input_tokens, output_tokens)
        self.add(usage)
        return usage

    def total(self) -> Usage:
        """The sum of everything added so far."""
        with self._lock:
            return self._total
