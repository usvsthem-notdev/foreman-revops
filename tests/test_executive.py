"""Tests for the Executive Brief analytics (src/analytics/executive.py)."""

import pandas as pd

from src.analytics.executive import (
    budget_health,
    build_brief,
    top_cost_center,
    week_over_week,
)
from src.analytics.intelligence import generate_report


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = df["cost_usd"].astype(float)
    df["is_local"] = df["is_local"].astype(bool)
    for col in ("input_tokens", "output_tokens", "reasoning_tokens",
                "cache_read_tokens", "cache_creation_tokens"):
        df[col] = df.get(col, 0)
    return df


def _row(date: str, model: str, cost: float, wc: str = "reason",
         is_local: bool = False):
    return {
        "timestamp": date, "provider": "anthropic", "model": model,
        "workload_class": wc, "cost_usd": cost, "is_local": is_local,
        "input_tokens": 10_000, "output_tokens": 1_000,
        "reasoning_tokens": 0, "team": "eng",
    }


def _empty_df() -> pd.DataFrame:
    from src.analytics.burn_map import _empty_df as make
    return make()


class TestWeekOverWeek:
    def test_delta_computed_against_data_anchor_not_wall_clock(self):
        rows = (
            [_row(f"2026-05-{d:02d}", "claude-sonnet-4-6", 1.0) for d in range(1, 8)]
            + [_row(f"2026-05-{d:02d}", "claude-sonnet-4-6", 2.0) for d in range(8, 15)]
        )
        wow = week_over_week(_make_df(rows))
        assert wow["recent_usd"] == 14.0
        assert wow["prior_usd"] == 7.0
        assert wow["delta_pct"] == 100.0

    def test_no_prior_period_yields_none_delta(self):
        rows = [_row("2026-06-30", "claude-sonnet-4-6", 3.0)]
        wow = week_over_week(_make_df(rows))
        assert wow["delta_pct"] is None

    def test_empty_df(self):
        wow = week_over_week(_empty_df())
        assert wow == {"recent_usd": 0.0, "prior_usd": 0.0, "delta_pct": None}


class TestTopCostCenter:
    def test_largest_model_wins(self):
        rows = [
            _row("2026-06-01", "claude-opus-4-8", 10.0),
            _row("2026-06-01", "claude-haiku-4-5", 1.0),
        ]
        assert top_cost_center(_make_df(rows)) == ("claude-opus-4-8", 10.0)

    def test_empty_or_zero_cost_returns_none(self):
        assert top_cost_center(_empty_df()) is None
        assert top_cost_center(_make_df([_row("2026-06-01", "m", 0.0)])) is None


class TestBudgetHealth:
    def _budget(self, spent: float, amount: float, over_threshold: bool):
        return {
            "name": "b", "spent_usd": spent, "amount_usd": amount,
            "pct_used": min(spent / amount, 1.0) if amount else 0.0,
            "over_threshold": over_threshold,
        }

    def test_over_at_risk_and_on_track_are_disjoint(self):
        budgets = [
            self._budget(120, 100, True),   # over
            self._budget(90, 100, True),    # at risk (threshold crossed, not over)
            self._budget(10, 100, False),   # on track
        ]
        health = budget_health(budgets)
        assert health["total"] == 3
        assert health["over"] == 1
        assert health["at_risk"] == 1
        assert health["worst"]["spent_usd"] == 120

    def test_no_budgets(self):
        health = budget_health([])
        assert health == {"total": 0, "over": 0, "at_risk": 0, "worst": None}


class TestBuildBrief:
    def test_brief_shape_and_narrative(self):
        rows = [
            _row(f"2026-06-{d:02d}", "claude-opus-4-8", 5.0) for d in range(1, 15)
        ]
        df = _make_df(rows)
        brief = build_brief(df, [], generate_report(df))
        assert brief["metrics"]["total_cost_usd"] == 70.0
        assert brief["projection"]["projected_total"] > 0
        assert len(brief["top_actions"]) <= 3
        # First narrative line always states spend + run-rate + forecast.
        assert "run-rate" in brief["narrative"][0]

    def test_brief_handles_empty_budgets_and_savings(self):
        df = _make_df([_row("2026-06-01", "some-unknown-model", 0.5)])
        brief = build_brief(df, [], generate_report(df))
        assert brief["budgets"]["total"] == 0
        assert isinstance(brief["narrative"], list) and brief["narrative"]
