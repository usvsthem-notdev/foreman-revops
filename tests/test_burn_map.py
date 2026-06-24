"""Tests for src/analytics/burn_map.py data-layer functions."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.analytics.burn_map import (
    _empty_df,
    budget_status,
    burn_by_class,
    burn_by_model,
    burn_by_provider,
    burn_rate_projection,
    cumulative_burn,
    daily_burn,
    key_metrics,
    load_dataframe,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(
    date: str = "2026-06-01",
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5",
    workload_class: str = "extract",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    reasoning_tokens: int = 0,
    cost_usd: float = 1.0,
    is_local: bool = False,
    team: str | None = "eng",
    feature: str | None = "chat",
) -> dict:
    return {
        "id": "test-id",
        "timestamp": pd.Timestamp(date),
        "provider": provider,
        "model": model,
        "workload_class": workload_class,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": cost_usd,
        "is_local": is_local,
        "team": team,
        "feature": feature,
        "notes": None,
        "source": "manual",
    }


def _make_df(*rows: dict) -> pd.DataFrame:
    if not rows:
        return _empty_df()
    df = pd.DataFrame(list(rows))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = pd.to_numeric(df["cost_usd"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["reasoning_tokens"] = df["reasoning_tokens"].astype(int)
    df["is_local"] = df["is_local"].astype(bool)
    return df


# ── load_dataframe ────────────────────────────────────────────────────────────

class TestLoadDataframe:
    def test_empty_db_returns_empty_df(self):
        with patch("src.analytics.burn_map.fetch_entries", return_value=[]):
            df = load_dataframe()
        assert df.empty

    def test_non_empty_db_returns_df(self):
        fake_row = _row()
        fake_row["timestamp"] = "2026-06-01T12:00:00"
        fake_row["cost_usd"] = "1.0"
        fake_row["input_tokens"] = "1000"
        fake_row["output_tokens"] = "200"
        fake_row["reasoning_tokens"] = "0"
        fake_row["is_local"] = 0
        with patch("src.analytics.burn_map.fetch_entries", return_value=[fake_row]):
            df = load_dataframe()
        assert len(df) == 1
        assert df["cost_usd"].iloc[0] == pytest.approx(1.0)
        assert df["is_local"].dtype == bool


# ── daily_burn ────────────────────────────────────────────────────────────────

class TestDailyBurn:
    def test_empty_returns_empty(self):
        result = daily_burn(_make_df())
        assert result.empty

    def test_frontier_only_adds_absorbed_column(self):
        df = _make_df(_row(is_local=False, cost_usd=2.0))
        result = daily_burn(df)
        assert "absorbed_usd" in result.columns
        assert result["absorbed_usd"].iloc[0] == 0.0
        assert result["frontier_usd"].iloc[0] == pytest.approx(2.0)

    def test_local_only_adds_frontier_column(self):
        df = _make_df(_row(is_local=True, cost_usd=0.5))
        result = daily_burn(df)
        assert "frontier_usd" in result.columns
        assert result["frontier_usd"].iloc[0] == 0.0
        assert result["absorbed_usd"].iloc[0] == pytest.approx(0.5)

    def test_mixed_totals_correctly(self):
        df = _make_df(
            _row(date="2026-06-01", is_local=False, cost_usd=3.0),
            _row(date="2026-06-01", is_local=True,  cost_usd=1.0),
        )
        result = daily_burn(df)
        assert result["total_usd"].iloc[0] == pytest.approx(4.0)

    def test_sorted_by_date(self):
        df = _make_df(
            _row(date="2026-06-03", cost_usd=1.0),
            _row(date="2026-06-01", cost_usd=2.0),
        )
        result = daily_burn(df)
        assert list(result["date"]) == sorted(result["date"].tolist())


# ── burn_by_class ─────────────────────────────────────────────────────────────

class TestBurnByClass:
    def test_empty_returns_empty(self):
        result = burn_by_class(_make_df())
        assert result.empty

    def test_adds_missing_absorbed_column(self):
        df = _make_df(_row(is_local=False, workload_class="reason", cost_usd=5.0))
        result = burn_by_class(df)
        assert "absorbed_usd" in result.columns
        assert result["absorbed_usd"].iloc[0] == 0.0

    def test_adds_missing_frontier_column(self):
        df = _make_df(_row(is_local=True, workload_class="coding", cost_usd=0.1))
        result = burn_by_class(df)
        assert "frontier_usd" in result.columns
        assert result["frontier_usd"].iloc[0] == 0.0

    def test_groups_by_class(self):
        df = _make_df(
            _row(workload_class="reason",  is_local=False, cost_usd=10.0),
            _row(workload_class="extract", is_local=False, cost_usd=2.0),
        )
        result = burn_by_class(df)
        assert len(result) == 2


# ── burn_by_model ─────────────────────────────────────────────────────────────

class TestBurnByModel:
    def test_empty_returns_empty(self):
        result = burn_by_model(_make_df())
        assert result.empty

    def test_returns_top_n(self):
        rows = [_row(model=f"model-{i}", cost_usd=float(i)) for i in range(15)]
        df = _make_df(*rows)
        result = burn_by_model(df, top_n=5)
        assert len(result) == 5
        assert result["total_usd"].iloc[0] == pytest.approx(14.0)

    def test_renamed_to_total_usd(self):
        df = _make_df(_row(model="gpt-4o", cost_usd=7.0))
        result = burn_by_model(df)
        assert "total_usd" in result.columns


# ── burn_by_provider ──────────────────────────────────────────────────────────

class TestBurnByProvider:
    def test_empty_returns_empty(self):
        result = burn_by_provider(_make_df())
        assert result.empty

    def test_sorted_descending(self):
        df = _make_df(
            _row(provider="openai",    cost_usd=1.0),
            _row(provider="anthropic", cost_usd=5.0),
        )
        result = burn_by_provider(df)
        assert result["provider"].iloc[0] == "anthropic"

    def test_total_usd_column(self):
        df = _make_df(_row(provider="cursor", cost_usd=3.0))
        result = burn_by_provider(df)
        assert "total_usd" in result.columns
        assert result["total_usd"].iloc[0] == pytest.approx(3.0)


# ── cumulative_burn ───────────────────────────────────────────────────────────

class TestCumulativeBurn:
    def test_empty_returns_empty(self):
        result = cumulative_burn(_make_df())
        assert result.empty

    def test_cumulative_column_added(self):
        df = _make_df(
            _row(date="2026-06-01", cost_usd=1.0),
            _row(date="2026-06-02", cost_usd=2.0),
        )
        result = cumulative_burn(df)
        assert "cumulative_usd" in result.columns
        assert result["cumulative_usd"].iloc[-1] == pytest.approx(3.0)


# ── key_metrics ───────────────────────────────────────────────────────────────

class TestKeyMetrics:
    def test_empty_returns_zeros(self):
        m = key_metrics(_make_df())
        assert m["total_cost_usd"] == 0.0
        assert m["entry_count"] == 0
        assert m["local_pct"] == 0.0

    def test_computes_totals(self):
        df = _make_df(
            _row(cost_usd=4.0, is_local=False, input_tokens=1000, output_tokens=200),
            _row(cost_usd=1.0, is_local=True,  input_tokens=500,  output_tokens=100),
        )
        m = key_metrics(df)
        assert m["total_cost_usd"] == pytest.approx(5.0)
        assert m["frontier_cost_usd"] == pytest.approx(4.0)
        assert m["absorbed_cost_usd"] == pytest.approx(1.0)
        assert m["local_pct"] == pytest.approx(20.0)
        assert m["entry_count"] == 2
        assert m["total_input_tokens"] == 1500
        assert m["total_output_tokens"] == 300

    def test_cost_per_1k_tokens(self):
        df = _make_df(_row(cost_usd=1.0, input_tokens=1000, output_tokens=0))
        m = key_metrics(df)
        assert m["cost_per_1k_tokens"] == pytest.approx(1.0)

    def test_zero_tokens_no_divide_by_zero(self):
        df = _make_df(_row(cost_usd=1.0, input_tokens=0, output_tokens=0))
        m = key_metrics(df)
        assert m["cost_per_1k_tokens"] == 0.0


# ── burn_rate_projection ──────────────────────────────────────────────────────

class TestBurnRateProjection:
    def test_empty_returns_zeros(self):
        result = burn_rate_projection(_make_df())
        assert result["projected_total"] == 0.0
        assert result["daily_avg"] == 0.0
        assert result["days_of_data"] == 0

    def test_single_day_returns_zeros(self):
        df = _make_df(_row(date="2026-06-01", cost_usd=5.0))
        result = burn_rate_projection(df)
        assert result["projected_total"] == 0.0

    def test_two_days_projects_correctly(self):
        df = _make_df(
            _row(date="2026-06-01", cost_usd=2.0),
            _row(date="2026-06-02", cost_usd=4.0),
        )
        result = burn_rate_projection(df, days_ahead=10)
        assert result["daily_avg"] == pytest.approx(3.0)
        assert result["projected_total"] == pytest.approx(30.0)
        assert result["days_of_data"] == 2


# ── budget_status ─────────────────────────────────────────────────────────────

class TestBudgetStatus:
    def _budget(self, period="monthly", amount=100.0, team=None, provider=None):
        b = {
            "id": "bud-1",
            "name": "Test budget",
            "amount_usd": amount,
            "period": period,
            "alert_threshold": 0.8,
        }
        if team:
            b["team"] = team
        if provider:
            b["provider"] = provider
        return b

    def test_empty_df_returns_zero_spent(self):
        result = budget_status(_make_df(), [self._budget()])
        assert result[0]["spent_usd"] == 0.0
        assert result[0]["remaining_usd"] == pytest.approx(100.0)
        assert result[0]["pct_used"] == 0.0

    def test_daily_period_filters_today(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(_row(date=now, cost_usd=10.0))
        result = budget_status(df, [self._budget(period="daily", amount=50.0)])
        assert result[0]["spent_usd"] == pytest.approx(10.0)

    def test_weekly_period(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(_row(date=now, cost_usd=5.0))
        result = budget_status(df, [self._budget(period="weekly", amount=50.0)])
        assert result[0]["spent_usd"] == pytest.approx(5.0)

    def test_monthly_period(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(_row(date=now, cost_usd=20.0))
        result = budget_status(df, [self._budget(period="monthly", amount=100.0)])
        assert result[0]["spent_usd"] == pytest.approx(20.0)

    def test_team_filter(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(
            _row(date=now, cost_usd=10.0, team="eng"),
            _row(date=now, cost_usd=5.0,  team="product"),
        )
        result = budget_status(df, [self._budget(team="eng")])
        assert result[0]["spent_usd"] == pytest.approx(10.0)

    def test_over_threshold_flagged(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(_row(date=now, cost_usd=90.0))
        result = budget_status(df, [self._budget(amount=100.0)])
        assert result[0]["over_threshold"] == True  # noqa: E712 (np.True_ != True via `is`)

    def test_pct_used_capped_at_1(self):
        now = datetime.utcnow().strftime("%Y-%m-%d")
        df = _make_df(_row(date=now, cost_usd=200.0))
        result = budget_status(df, [self._budget(amount=100.0)])
        assert result[0]["pct_used"] == 1.0
        assert result[0]["remaining_usd"] == 0.0
