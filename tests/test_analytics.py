"""Tests for burn map analytics and spend intelligence."""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.analytics.burn_map import (
    burn_by_class,
    burn_by_model,
    burn_rate_projection,
    daily_burn,
    key_metrics,
)
from src.analytics.intelligence import detect, generate_report, propose


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = df["cost_usd"].astype(float)
    df["is_local"] = df["is_local"].astype(bool)
    df["input_tokens"] = df.get("input_tokens", 0)
    df["output_tokens"] = df.get("output_tokens", 0)
    df["reasoning_tokens"] = df.get("reasoning_tokens", 0)
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
