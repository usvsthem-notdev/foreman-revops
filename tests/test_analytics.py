"""Tests for burn map analytics and spend intelligence."""

import pandas as pd
import pytest

from src.analytics.burn_map import (
    burn_by_class,
    burn_rate_projection,
    daily_burn,
    key_metrics,
)
from src.analytics.intelligence import (
    _detect_batch_opportunity,
    _detect_cache_degradation,
    _detect_cache_opportunity,
    _detect_concentration,
    _detect_drift,
    _detect_reasoning_waste,
    _detect_untagged,
    _estimate_model_savings,
    _find_cheaper_alternative,
    detect,
    generate_report,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = df["cost_usd"].astype(float)
    df["is_local"] = df["is_local"].astype(bool)
    df["input_tokens"] = df.get("input_tokens", 0)
    df["output_tokens"] = df.get("output_tokens", 0)
    df["reasoning_tokens"] = df.get("reasoning_tokens", 0)
    df["cache_read_tokens"] = df.get("cache_read_tokens", 0)
    df["cache_creation_tokens"] = df.get("cache_creation_tokens", 0)
    return df


SAMPLE_ROWS = [
    {"timestamp": "2026-06-01", "provider": "anthropic", "model": "claude-opus-4",
     "workload_class": "reason", "cost_usd": 1.50, "is_local": False,
     "input_tokens": 10000, "output_tokens": 2000, "reasoning_tokens": 500, "team": "eng"},
    {"timestamp": "2026-06-01", "provider": "anthropic", "model": "claude-haiku-4-5",
     "workload_class": "extract", "cost_usd": 0.05, "is_local": False,
     "input_tokens": 5000, "output_tokens": 800, "reasoning_tokens": 0, "team": "eng"},
    {"timestamp": "2026-06-02", "provider": "openai", "model": "gpt-4o",
     "workload_class": "agents", "cost_usd": 0.80, "is_local": False,
     "input_tokens": 8000, "output_tokens": 1500, "reasoning_tokens": 0, "team": "product"},
    {"timestamp": "2026-06-02", "provider": "anthropic", "model": "llama-local",
     "workload_class": "rag", "cost_usd": 0.00, "is_local": True,
     "input_tokens": 20000, "output_tokens": 3000, "reasoning_tokens": 0, "team": "eng"},
]


class TestKeyMetrics:
    def test_total_cost(self):
        df = _make_df(SAMPLE_ROWS)
        m = key_metrics(df)
        assert abs(m["total_cost_usd"] - 2.35) < 0.01

    def test_local_pct(self):
        df = _make_df(SAMPLE_ROWS)
        m = key_metrics(df)
        # absorbed = 0.00, so local_pct = 0%
        assert m["local_pct"] == 0.0

    def test_empty_df_returns_zeros(self):
        from src.analytics.burn_map import _empty_df
        m = key_metrics(_empty_df())
        assert m["total_cost_usd"] == 0.0
        assert m["entry_count"] == 0


class TestDailyBurn:
    def test_two_dates(self):
        df = _make_df(SAMPLE_ROWS)
        daily = daily_burn(df)
        assert len(daily) == 2
        assert "frontier_usd" in daily.columns
        assert "absorbed_usd" in daily.columns

    def test_frontier_absorbed_split(self):
        df = _make_df(SAMPLE_ROWS)
        daily = daily_burn(df)
        # Day 2 has one local entry (cost 0.00) and one frontier (0.80)
        day2 = daily[daily["date"].astype(str) == "2026-06-02"].iloc[0]
        assert day2["frontier_usd"] == pytest.approx(0.80)
        assert day2["absorbed_usd"] == pytest.approx(0.00)


class TestBurnByClass:
    def test_workload_classes_present(self):
        df = _make_df(SAMPLE_ROWS)
        result = burn_by_class(df)
        classes = set(result["workload_class"])
        assert "reason" in classes
        assert "extract" in classes


class TestBurnRateProjection:
    def test_projects_30_days(self):
        df = _make_df(SAMPLE_ROWS)
        proj = burn_rate_projection(df, days_ahead=30)
        assert proj["daily_avg"] > 0
        assert proj["projected_total"] == pytest.approx(proj["daily_avg"] * 30)

    def test_empty_df(self):
        from src.analytics.burn_map import _empty_df
        proj = burn_rate_projection(_empty_df())
        assert proj["daily_avg"] == 0.0


class TestIntelligence:
    def test_detect_concentration(self):
        # Make opus dominate >70% of spend
        rows = SAMPLE_ROWS * 5  # amplify the opus entries
        df = _make_df(rows)
        findings = detect(df)
        categories = [f.category for f in findings]
        assert "concentration" in categories

    def test_detect_reasoning_waste(self):
        heavy_reason = [
            {"timestamp": "2026-06-01", "provider": "anthropic", "model": "claude-opus-4",
             "workload_class": "reason", "cost_usd": 2.0, "is_local": False,
             "input_tokens": 1000, "output_tokens": 500, "reasoning_tokens": 5000, "team": "eng"}
        ] * 10
        df = _make_df(heavy_reason)
        findings = detect(df)
        categories = [f.category for f in findings]
        assert "waste" in categories

    def test_generate_report_returns_proposals(self):
        df = _make_df(SAMPLE_ROWS)
        report = generate_report(df)
        assert isinstance(report.findings, list)
        assert isinstance(report.proposals, list)
        assert isinstance(report.workload_library, dict)

    def test_empty_df_no_crash(self):
        from src.analytics.burn_map import _empty_df
        report = generate_report(_empty_df())
        assert report.findings == []
        assert report.proposals == []


# ── Intelligence edge cases ───────────────────────────────────────────────────

class TestIntelligenceEdgeCases:
    def _row(self, date: str, model: str, cost: float, wc: str = "extract",
             input_tok: int = 100, output_tok: int = 20, reasoning_tok: int = 0):
        return {
            "timestamp": date, "provider": "anthropic", "model": model,
            "workload_class": wc, "cost_usd": cost, "is_local": False,
            "input_tokens": input_tok, "output_tokens": output_tok,
            "reasoning_tokens": reasoning_tok, "team": "eng",
        }

    def test_concentration_zero_cost_returns_empty(self):
        """Line 101: _detect_concentration short-circuits when total cost is 0."""
        df = _make_df([self._row("2026-06-01", "claude-opus-4", 0.0)])
        assert _detect_concentration(df) == []

    def test_reasoning_waste_zero_tokens_returns_empty(self):
        """Line 123: _detect_reasoning_waste short-circuits when total tokens are 0."""
        df = _make_df([self._row("2026-06-01", "claude-opus-4", 1.0,
                                 input_tok=0, output_tok=0, reasoning_tok=0)])
        assert _detect_reasoning_waste(df) == []

    def test_drift_empty_df_returns_empty(self):
        """Line 146: _detect_drift returns [] for empty DataFrame."""
        from src.analytics.burn_map import _empty_df
        assert _detect_drift(_empty_df()) == []

    def test_untagged_empty_df_returns_empty(self):
        """Line 181: _detect_untagged returns [] for empty DataFrame."""
        from src.analytics.burn_map import _empty_df
        assert _detect_untagged(_empty_df()) == []

    def test_drift_spend_up_detected(self):
        """Lines 157-168: drift finding raised when recent spend >> prior spend."""
        rows = []
        # Prior period (days 1-6 relative to max=day14)
        for i in range(1, 7):
            rows.append(self._row(f"2026-07-{i:02d}", "claude-haiku-4-5", 0.10))
        # Recent period (days 7-14)
        for i in range(7, 15):
            rows.append(self._row(f"2026-07-{i:02d}", "claude-haiku-4-5", 2.00))
        df = _make_df(rows)
        findings = _detect_drift(df)
        assert any(f.category == "drift" and "up" in f.title for f in findings)

    def test_drift_spend_down_detected(self):
        """Lines 169-174: elif branch when spend drops >30%."""
        rows = []
        # Prior period: high spend
        for i in range(1, 7):
            rows.append(self._row(f"2026-07-{i:02d}", "claude-haiku-4-5", 2.00))
        # Recent period: low spend
        for i in range(7, 15):
            rows.append(self._row(f"2026-07-{i:02d}", "claude-haiku-4-5", 0.10))
        df = _make_df(rows)
        findings = _detect_drift(df)
        assert any(f.category == "drift" and "down" in f.title for f in findings)

    def test_untagged_above_threshold_detected(self):
        """Lines 185-197: _detect_untagged creates a Finding when >20% unknown."""
        rows = [
            self._row("2026-06-01", "claude-haiku-4-5", 1.0, wc="unknown"),
            self._row("2026-06-01", "claude-haiku-4-5", 1.0, wc="unknown"),
            self._row("2026-06-01", "claude-haiku-4-5", 0.1, wc="extract"),
        ]
        df = _make_df(rows)
        findings = _detect_untagged(df)
        assert any(f.category == "untagged" for f in findings)

    def test_estimate_model_savings_unknown_model_returns_zero(self):
        """Line 262: _estimate_model_savings returns 0.0 when no alt found."""
        assert _estimate_model_savings("some-totally-unknown-xyz-llm", 100.0) == 0.0

    def test_concentration_high_severity(self):
        """Line 107 (severity='high'): model >70% of total spend."""
        rows = [
            self._row("2026-06-01", "claude-opus-4", 80.0, wc="reason"),
            self._row("2026-06-01", "claude-haiku-4-5", 5.0),
        ]
        df = _make_df(rows)
        findings = _detect_concentration(df)
        assert any(f.severity == "high" for f in findings)


class TestModernCostLevers:
    def _row(self, model: str, wc: str, cost: float, input_tok: int,
             cache_read: int = 0, is_local: bool = False):
        return {
            "timestamp": "2026-06-15", "provider": "anthropic", "model": model,
            "workload_class": wc, "cost_usd": cost, "is_local": is_local,
            "input_tokens": input_tok, "output_tokens": 500,
            "reasoning_tokens": 0, "cache_read_tokens": cache_read, "team": "eng",
        }

    def test_cache_opportunity_flagged_when_hit_rate_low(self):
        # Heavy uncached agent traffic on a cache-capable model.
        rows = [self._row("claude-sonnet-4-6", "agents", 5.0, 500_000)] * 10
        findings = _detect_cache_opportunity(_make_df(rows))
        assert len(findings) == 1
        assert findings[0].category == "caching"
        assert findings[0].estimated_savings_usd > 1.0

    def test_cache_opportunity_silent_when_hit_rate_healthy(self):
        rows = [
            self._row("claude-sonnet-4-6", "agents", 5.0, 500_000,
                      cache_read=400_000)
        ] * 10
        assert _detect_cache_opportunity(_make_df(rows)) == []

    def test_cache_opportunity_ignores_local_and_one_shot_classes(self):
        rows = [
            self._row("qwen3-32b-local", "agents", 0.0, 500_000, is_local=True),
            self._row("claude-sonnet-4-6", "extract", 5.0, 500_000),
        ] * 10
        assert _detect_cache_opportunity(_make_df(rows)) == []

    def test_batch_opportunity_flagged_for_latency_tolerant_spend(self):
        rows = [self._row("claude-haiku-4-5", "extract", 4.0, 10_000)] * 10
        findings = _detect_batch_opportunity(_make_df(rows))
        assert len(findings) == 1
        assert findings[0].category == "batch"
        # 50% discount x 80% batchable share of $40
        assert findings[0].estimated_savings_usd == pytest.approx(16.0)

    def test_batch_opportunity_silent_for_interactive_spend(self):
        rows = [self._row("claude-sonnet-4-6", "coding", 4.0, 10_000)] * 10
        assert _detect_batch_opportunity(_make_df(rows)) == []

    def test_generate_report_includes_new_proposals(self):
        rows = (
            [self._row("claude-sonnet-4-6", "agents", 5.0, 500_000)] * 10
            + [self._row("claude-haiku-4-5", "extract", 4.0, 10_000)] * 10
        )
        report = generate_report(_make_df(rows))
        titles = " ".join(p.title for p in report.proposals)
        assert "prompt caching" in titles.lower()
        assert "batch" in titles.lower()

    def test_cheaper_alternative_uses_current_models(self):
        alt = _find_cheaper_alternative("claude-opus-4-8")
        assert alt is not None and alt[0] == "claude-haiku-4-5"
        alt = _find_cheaper_alternative("gpt-5.5")
        assert alt is not None and alt[0] == "gpt-5.4-mini"

    def test_no_downgrade_suggested_for_already_cheap_model(self):
        # gpt-5.4-mini contains "gpt-5.4" but is already the suggested
        # alternative — routing it to itself would be nonsense.
        assert _find_cheaper_alternative("gpt-5.4-mini") is None
        assert _find_cheaper_alternative("claude-haiku-4-5") is None


class TestCacheDegradation:
    def _row(self, date: str, input_tok: int, cache_read: int):
        return {
            "timestamp": date, "provider": "anthropic",
            "model": "claude-sonnet-4-6", "workload_class": "agents",
            "cost_usd": 1.0, "is_local": False, "input_tokens": input_tok,
            "output_tokens": 500, "reasoning_tokens": 0,
            "cache_read_tokens": cache_read, "team": "eng",
        }

    def test_silent_invalidation_flagged(self):
        # Prior week: 60% hit rate. Recent week: ~0% — a prefix change
        # silently killed the cache.
        rows = (
            [self._row(f"2026-06-{d:02d}", 100_000, 60_000) for d in range(1, 8)]
            + [self._row(f"2026-06-{d:02d}", 100_000, 0) for d in range(8, 15)]
        )
        findings = _detect_cache_degradation(_make_df(rows))
        assert len(findings) == 1
        assert findings[0].category == "cache_degradation"
        assert findings[0].severity == "high"
        assert findings[0].estimated_savings_usd > 0

    def test_stable_hit_rate_not_flagged(self):
        rows = [self._row(f"2026-06-{d:02d}", 100_000, 60_000) for d in range(1, 15)]
        assert _detect_cache_degradation(_make_df(rows)) == []

    def test_never_cached_is_opportunity_not_degradation(self):
        # A workload that never had a meaningful hit rate can't "degrade" —
        # that's _detect_cache_opportunity's territory, not a regression.
        rows = [self._row(f"2026-06-{d:02d}", 100_000, 0) for d in range(1, 15)]
        assert _detect_cache_degradation(_make_df(rows)) == []
