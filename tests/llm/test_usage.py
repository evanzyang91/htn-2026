"""Usage arithmetic and cost, asserted against fixed numbers."""

from __future__ import annotations

import logging

import pytest

from skillweaver.contracts import Usage
from skillweaver.llm.usage import (
    PRICING,
    ModelPrice,
    UsageMeter,
    cost_usd,
    normalize_model,
    price_for,
    reset_unknown_model_warnings,
    usage_for,
)


@pytest.fixture(autouse=True)
def _fresh_warnings() -> None:
    reset_unknown_model_warnings()


class TestPricing:
    def test_known_model_costs_the_table_price(self) -> None:
        # claude-opus-5 is $5.00 in / $25.00 out per million tokens.
        # 1_000_000 in + 200_000 out = 5.00 + 5.00 = 10.00 exactly.
        assert cost_usd("claude-opus-5", 1_000_000, 200_000) == pytest.approx(10.00)

    def test_a_small_call_is_priced_proportionally(self) -> None:
        # 1234 in, 567 out on Opus 5.
        expected = (1234 * 5.00 + 567 * 25.00) / 1_000_000
        assert cost_usd("claude-opus-5", 1234, 567) == pytest.approx(expected)
        assert expected == pytest.approx(0.020345)

    def test_gemini_computer_use_is_priced(self) -> None:
        assert cost_usd("gemini-2.5-computer-use-preview-10-2025", 1_000_000, 1_000_000) == (
            pytest.approx(12.00)
        )

    def test_a_dated_snapshot_prices_as_its_base_model(self) -> None:
        assert normalize_model("claude-opus-4-5-20251101") == "claude-opus-4-5"
        assert normalize_model("claude-opus-5") == "claude-opus-5"
        # claude-opus-4-8-20260101 is a snapshot of a priced model.
        assert price_for("claude-opus-4-8-20260101") == PRICING["claude-opus-4-8"]

    def test_model_price_computes_directly(self) -> None:
        assert ModelPrice(3.0, 15.0).cost_usd(2_000_000, 100_000) == pytest.approx(7.5)

    def test_zero_tokens_cost_nothing(self) -> None:
        assert cost_usd("claude-opus-5", 0, 0) == 0.0


class TestUnknownModel:
    def test_reports_zero_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="skillweaver"):
            assert cost_usd("totally-made-up-model", 1_000_000, 1_000_000) == 0.0
        assert "usage.unknown_model" in caplog.text
        assert "totally-made-up-model" in caplog.text

    def test_price_for_returns_none(self) -> None:
        assert price_for("totally-made-up-model") is None

    def test_warns_once_per_model(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="skillweaver"):
            for _ in range(5):
                cost_usd("totally-made-up-model", 10, 10)
        assert caplog.text.count("usage.unknown_model") == 1

    def test_usage_still_records_the_tokens(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="skillweaver"):
            usage = usage_for("totally-made-up-model", 100, 20)
        assert usage == Usage(input_tokens=100, output_tokens=20, calls=1, cost_usd=0.0)


class TestArithmetic:
    def test_usage_adds_field_by_field(self) -> None:
        a = Usage(100, 20, 1, 0.5)
        b = Usage(300, 80, 2, 1.5)
        assert a + b == Usage(400, 100, 3, 2.0)

    def test_meter_accumulates_across_calls(self) -> None:
        meter = UsageMeter()
        assert meter.total() == Usage()
        first = meter.record("claude-opus-5", 1_000_000, 200_000)
        second = meter.record("claude-opus-5", 1_000_000, 200_000)
        # record() returns THIS call, not the running total.
        assert first == second == Usage(1_000_000, 200_000, 1, pytest.approx(10.00))
        assert meter.total() == Usage(2_000_000, 400_000, 2, pytest.approx(20.00))

    def test_meter_mixes_models(self) -> None:
        meter = UsageMeter()
        meter.record("claude-opus-5", 1_000_000, 0)  # $5.00
        meter.record("claude-haiku-4-5", 1_000_000, 0)  # $1.00
        total = meter.total()
        assert total.calls == 2
        assert total.input_tokens == 2_000_000
        assert total.cost_usd == pytest.approx(6.00)

    def test_meter_can_start_from_a_total(self) -> None:
        meter = UsageMeter(Usage(5, 5, 1, 0.25))
        meter.add(Usage(5, 5, 1, 0.25))
        assert meter.total() == Usage(10, 10, 2, 0.5)
