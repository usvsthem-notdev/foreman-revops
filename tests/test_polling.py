"""Tests for the live API polling layer."""
from __future__ import annotations

import stat
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models import EntrySource, Provider
from src.polling.anthropic import _check_status as anthropic_check_status
from src.polling.anthropic import _parse_ts
from src.polling.anthropic import _to_entry as anthropic_to_entry
from src.polling.base import mask_key, validate_key_format
from src.polling.cursor import _check_status as cursor_check_status
from src.polling.cursor import _ms
from src.polling.cursor import _parse_ts as cursor_parse_ts
from src.polling.cursor import _to_entry as cursor_to_entry
from src.polling.openai import _check_status as openai_check_status
from src.polling.openai import _to_entry as openai_to_entry

# ── Key format validation ────────────────────────────────────────────────────

class TestValidateKeyFormat:
    def test_empty_key_returns_error(self):
        assert validate_key_format("anthropic", "") is not None
        assert validate_key_format("openai", "") is not None

    def test_valid_anthropic_key(self):
        key = "sk-ant-api03-" + "A" * 90
        assert validate_key_format("anthropic", key) is None

    def test_invalid_anthropic_prefix(self):
        err = validate_key_format("anthropic", "sk-badprefix-" + "A" * 80)
        assert err is not None

    def test_valid_openai_key(self):
        assert validate_key_format("openai", "sk-" + "A" * 30) is None

    def test_valid_openai_proj_key(self):
        assert validate_key_format("openai", "sk-proj-" + "A" * 30) is None

    def test_invalid_openai_key_too_short(self):
        err = validate_key_format("openai", "sk-abc")
        assert err is not None

    def test_valid_cursor_key(self):
        assert validate_key_format("cursor", "crsr_" + "A" * 40) is None

    def test_invalid_cursor_key_missing_prefix(self):
        err = validate_key_format("cursor", "A" * 40)
        assert err is not None

    def test_valid_gemini_key(self):
        assert validate_key_format("gemini", "AIzaSy" + "A" * 33) is None

    def test_invalid_gemini_key_wrong_prefix(self):
        err = validate_key_format("gemini", "sk-" + "A" * 36)
        assert err is not None


# ── Key masking ──────────────────────────────────────────────────────────────

class TestMaskKey:
    def test_long_key_masked(self):
        key = "sk-ant-api03-" + "X" * 90
        result = mask_key(key)
        assert key not in result
        assert "…" in result
        assert result.startswith("sk-ant-api")

    def test_short_key_returns_stars(self):
        assert mask_key("tiny") == "****"

    def test_empty_key(self):
        assert mask_key("") == "****"


# ── Key store ────────────────────────────────────────────────────────────────

class TestKeyStore:
    def test_set_and_get_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.polling.key_store import get_key, set_key
        key = "sk-ant-api03-" + "T" * 90
        set_key("anthropic", key)
        assert get_key("anthropic") == key

    def test_env_var_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        env_key = "sk-ant-api03-" + "E" * 90
        file_key = "sk-ant-api03-" + "F" * 90
        monkeypatch.setenv("ANTHROPIC_API_KEY", env_key)

        from src.polling.key_store import get_key, set_key
        set_key("anthropic", file_key)
        assert get_key("anthropic") == env_key  # env wins

    def test_file_permissions(self, tmp_path, monkeypatch):
        env_local = tmp_path / ".env.local"
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", env_local)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.polling.key_store import set_key
        set_key("anthropic", "sk-ant-api03-" + "P" * 90)

        file_stat = env_local.stat()
        assert not (file_stat.st_mode & stat.S_IRGRP)
        assert not (file_stat.st_mode & stat.S_IWGRP)
        assert not (file_stat.st_mode & stat.S_IROTH)

    def test_clear_key(self, tmp_path, monkeypatch):
        env_local = tmp_path / ".env.local"
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", env_local)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.polling.key_store import clear_key, get_key, set_key
        set_key("anthropic", "sk-ant-api03-" + "C" * 90)
        assert get_key("anthropic") is not None
        clear_key("anthropic")
        assert get_key("anthropic") is None

    def test_has_key(self, tmp_path, monkeypatch):
        env_local = tmp_path / ".env.local"
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", env_local)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from src.polling.key_store import clear_key, has_key, set_key
        assert not has_key("openai")
        set_key("openai", "sk-" + "H" * 40)
        assert has_key("openai")
        clear_key("openai")
        assert not has_key("openai")


# ── SSRF guard ───────────────────────────────────────────────────────────────

class TestSafeGet:
    def test_disallowed_host_raises(self):
        from src.polling.base import safe_get
        with pytest.raises(ValueError, match="disallowed host"):
            safe_get("https://evil.example.com/steal", headers={})

    def test_allowed_hosts_in_allowlist(self):
        from src.polling.base import _ALLOWED_HOSTS
        assert "api.anthropic.com" in _ALLOWED_HOSTS
        assert "api.openai.com" in _ALLOWED_HOSTS
        assert "api.cursor.com" in _ALLOWED_HOSTS


# ── Entry parsing — Anthropic ────────────────────────────────────────────────

class TestAnthropicEntryParsing:
    def _item(self, **overrides):
        base = {
            "model": "claude-opus-4",
            "input_tokens": 10000,
            "output_tokens": 2000,
            "timestamp": "2026-06-01T12:00:00Z",
            "cost_usd": 0.18,
        }
        return {**base, **overrides}

    def test_basic_entry(self):
        entry = anthropic_to_entry(self._item())
        assert entry is not None
        assert entry.provider == Provider.anthropic
        assert entry.input_tokens == 10000
        assert entry.output_tokens == 2000
        assert entry.cost_usd == pytest.approx(0.18)
        assert entry.source == EntrySource.api

    def test_missing_model_returns_none(self):
        assert anthropic_to_entry({"input_tokens": 100, "output_tokens": 50}) is None

    def test_zero_tokens_returns_none(self):
        item = self._item(input_tokens=0, output_tokens=0)
        assert anthropic_to_entry(item) is None

    def test_unix_timestamp_parsed(self):
        item = self._item(timestamp=None, aggregation_timestamp=1748793600)
        entry = anthropic_to_entry(item)
        assert entry is not None
        assert entry.timestamp == datetime.utcfromtimestamp(1748793600)

    def test_workload_class_inferred(self):
        entry = anthropic_to_entry(self._item(model="claude-haiku-4-5"))
        assert entry is not None
        assert entry.workload_class.value == "extract"


class TestParseTs:
    def test_unix_int(self):
        ts = _parse_ts(1748793600)
        assert isinstance(ts, datetime)

    def test_iso_string(self):
        ts = _parse_ts("2026-06-01T12:00:00Z")
        assert ts is not None
        assert ts.year == 2026

    def test_date_only_string(self):
        ts = _parse_ts("2026-06-01")
        assert ts is not None

    def test_none_returns_none(self):
        assert _parse_ts(None) is None


# ── Entry parsing — OpenAI ───────────────────────────────────────────────────

class TestOpenAIEntryParsing:
    def _item(self, **overrides):
        base = {
            "snapshot_id": "gpt-4o",
            "n_context_tokens_total": 8000,
            "n_generated_tokens_total": 1500,
            "aggregation_timestamp": 1748793600,
        }
        return {**base, **overrides}

    def test_basic_entry(self):
        entry = openai_to_entry(self._item(), date(2026, 6, 1))
        assert entry is not None
        assert entry.provider == Provider.openai
        assert entry.input_tokens == 8000
        assert entry.output_tokens == 1500
        assert entry.source == EntrySource.api

    def test_missing_model_returns_none(self):
        item = {"n_context_tokens_total": 100, "n_generated_tokens_total": 50}
        assert openai_to_entry(item, date.today()) is None

    def test_zero_tokens_returns_none(self):
        item = self._item(n_context_tokens_total=0, n_generated_tokens_total=0)
        assert openai_to_entry(item, date.today()) is None

    def test_cached_tokens_stored_as_reasoning(self):
        item = self._item(n_cached_context_tokens_total=500)
        entry = openai_to_entry(item, date(2026, 6, 1))
        assert entry is not None
        assert entry.reasoning_tokens == 500

    def test_fallback_to_day_when_no_timestamp(self):
        item = {
            "snapshot_id": "gpt-4o",
            "n_context_tokens_total": 100,
            "n_generated_tokens_total": 50,
        }
        day = date(2026, 6, 15)
        entry = openai_to_entry(item, day)
        assert entry is not None
        assert entry.timestamp.date() == day


# ── Entry parsing — Cursor ───────────────────────────────────────────────────

class TestCursorEntryParsing:
    def _event(self, **overrides):
        base = {
            "model": "claude-3.5-sonnet",
            "kind": "chat",
            "isTokenBasedCall": True,
            "isChargeable": True,
            "timestamp": 1748793600000,  # epoch ms
            "tokenUsage": {
                "inputTokens": 5000,
                "outputTokens": 1000,
                "cacheReadTokens": 200,
                "cacheWriteTokens": 0,
                "totalCents": 0.09,
            },
            "chargedCents": 9,
        }
        return {**base, **overrides}

    def test_basic_entry(self):
        entry = cursor_to_entry(self._event())
        assert entry is not None
        assert entry.provider == Provider.cursor
        assert entry.input_tokens == 5000
        assert entry.output_tokens == 1000
        assert entry.source == EntrySource.cursor_api

    def test_cost_from_charged_cents(self):
        entry = cursor_to_entry(self._event(chargedCents=50))
        assert entry is not None
        assert entry.cost_usd == pytest.approx(0.50)

    def test_cache_tokens_stored_as_reasoning(self):
        entry = cursor_to_entry(self._event())
        assert entry is not None
        assert entry.reasoning_tokens == 200

    def test_non_token_call_returns_none(self):
        assert cursor_to_entry(self._event(isTokenBasedCall=False)) is None

    def test_missing_model_returns_none(self):
        assert cursor_to_entry(self._event(model="")) is None

    def test_zero_tokens_returns_none(self):
        event = self._event()
        event["tokenUsage"]["inputTokens"] = 0
        event["tokenUsage"]["outputTokens"] = 0
        assert cursor_to_entry(event) is None

    def test_ms_timestamp_parsed(self):
        entry = cursor_to_entry(self._event(timestamp=1748793600000))
        assert entry is not None
        assert entry.timestamp == datetime.utcfromtimestamp(1748793600)

    def test_team_set_from_email(self):
        entry = cursor_to_entry(self._event(userEmail="alice@example.com"))
        assert entry is not None
        assert entry.team == "alice@example.com"

    def test_feature_set_from_kind(self):
        entry = cursor_to_entry(self._event(kind="agent"))
        assert entry is not None
        assert entry.feature == "agent"


class TestCursorParseTs:
    def test_ms_epoch(self):
        ts = cursor_parse_ts(1748793600000)
        assert ts == datetime.utcfromtimestamp(1748793600)

    def test_iso_string(self):
        ts = cursor_parse_ts("2026-06-01T12:00:00Z")
        assert ts is not None
        assert ts.year == 2026

    def test_none_returns_none(self):
        assert cursor_parse_ts(None) is None

    def test_date_only_string(self):
        ts = cursor_parse_ts("2026-06-01")
        assert ts is not None
        assert ts.year == 2026


class TestCursorMs:
    def test_returns_epoch_ms(self):
        ms = _ms(date(2026, 1, 1))
        assert isinstance(ms, int)
        assert ms > 0

    def test_ms_is_midnight_utc(self):
        # 2026-01-01 00:00:00 UTC = 1767225600 seconds
        ms = _ms(date(2026, 1, 1))
        assert ms == 1767225600000


class TestCursorCheckStatus:
    def _resp(self, code: int, text: str = "") -> MagicMock:
        r = MagicMock()
        r.is_success = code < 400
        r.status_code = code
        r.text = text
        return r

    def test_200_returns_empty(self):
        assert cursor_check_status(self._resp(200)) == []

    def test_401_returns_error(self):
        errs = cursor_check_status(self._resp(401))
        assert len(errs) == 1
        assert "401" in errs[0]

    def test_403_returns_error(self):
        errs = cursor_check_status(self._resp(403))
        assert len(errs) == 1
        assert "403" in errs[0]
        assert "Team" in errs[0]

    def test_429_returns_error(self):
        errs = cursor_check_status(self._resp(429))
        assert len(errs) == 1
        assert "429" in errs[0]

    def test_500_returns_generic_error(self):
        errs = cursor_check_status(self._resp(500, "internal error"))
        assert len(errs) == 1
        assert "500" in errs[0]


class TestCursorPoll:
    def _fake_event(self):
        return {
            "model": "gpt-4o",
            "kind": "chat",
            "isTokenBasedCall": True,
            "timestamp": 1748793600000,
            "tokenUsage": {
                "inputTokens": 1000,
                "outputTokens": 200,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "totalCents": 0.02,
            },
            "chargedCents": 2,
        }

    def test_poll_returns_entries_on_success(self):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"usageEvents": [self._fake_event()]}

        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))

        assert errors == []
        assert len(entries) == 1
        assert entries[0].provider == Provider.cursor

    def test_poll_stops_on_401(self):
        mock_resp = MagicMock()
        mock_resp.is_success = False
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll(
                "crsr_bad", since=date(2026, 6, 1), until=date(2026, 6, 2)
            )

        assert entries == []
        assert any("401" in e for e in errors)

    def test_poll_timeout_returns_error(self):
        import httpx
        key = "crsr_" + "A" * 40
        with patch(
            "src.polling.cursor.safe_post",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))

        assert entries == []
        assert any("timed out" in e for e in errors)


# ── Gemini stub ──────────────────────────────────────────────────────────────

class TestGeminiPoll:
    def test_poll_returns_no_entries(self):
        from src.polling.gemini import poll
        entries, errors = poll("AIzaSy" + "A" * 33)
        assert entries == []

    def test_poll_returns_explanation(self):
        from src.polling.gemini import poll
        _, errors = poll("AIzaSy" + "A" * 33)
        assert len(errors) == 1
        assert "Google" in errors[0]
        assert "BigQuery" in errors[0]


# ── Anthropic poll() ──────────────────────────────────────────────────────────

class TestAnthropicCheckStatus:
    def _resp(self, code: int, text: str = "") -> MagicMock:
        r = MagicMock()
        r.is_success = code < 400
        r.status_code = code
        r.text = text
        return r

    def test_200_ok(self):
        assert anthropic_check_status(self._resp(200)) == []

    def test_401(self):
        errs = anthropic_check_status(self._resp(401))
        assert any("401" in e for e in errs)

    def test_403(self):
        errs = anthropic_check_status(self._resp(403))
        assert any("403" in e for e in errs)
        assert any("usage" in e.lower() for e in errs)

    def test_404(self):
        errs = anthropic_check_status(self._resp(404))
        assert any("404" in e for e in errs)

    def test_429(self):
        errs = anthropic_check_status(self._resp(429))
        assert any("429" in e for e in errs)

    def test_generic_error(self):
        errs = anthropic_check_status(self._resp(500, "server error"))
        assert any("500" in e for e in errs)


class TestAnthropicPoll:
    def _mock_resp(self, body: dict, status: int = 200) -> MagicMock:
        r = MagicMock()
        r.is_success = status < 400
        r.status_code = status
        r.json.return_value = body
        r.text = ""
        return r

    def _valid_item(self) -> dict:
        return {
            "model": "claude-opus-4",
            "input_tokens": 10000,
            "output_tokens": 2000,
            "timestamp": "2026-06-01T12:00:00Z",
            "cost_usd": 0.18,
        }

    def test_successful_poll_returns_entries(self):
        resp = self._mock_resp({"data": [self._valid_item()]})
        with patch("src.polling.anthropic.safe_get", return_value=resp):
            from src.polling.anthropic import poll
            entries, errors = poll(
                "sk-ant-api03-" + "A" * 90,
                since=date(2026, 6, 1),
                until=date(2026, 6, 1),
            )
        assert errors == []
        assert len(entries) == 1
        assert entries[0].provider == Provider.anthropic

    def test_poll_with_models_key(self):
        resp = self._mock_resp({"models": [self._valid_item()]})
        with patch("src.polling.anthropic.safe_get", return_value=resp):
            from src.polling.anthropic import poll
            entries, errors = poll("sk-ant-api03-" + "A" * 90)
        assert len(entries) == 1

    def test_poll_401_returns_error(self):
        resp = self._mock_resp({}, status=401)
        with patch("src.polling.anthropic.safe_get", return_value=resp):
            from src.polling.anthropic import poll
            entries, errors = poll("sk-ant-api03-" + "A" * 90)
        assert entries == []
        assert any("401" in e for e in errors)

    def test_poll_timeout_returns_error(self):
        import httpx
        with patch(
            "src.polling.anthropic.safe_get",
            side_effect=httpx.TimeoutException("t"),
        ):
            from src.polling.anthropic import poll
            _, errors = poll("sk-ant-api03-" + "A" * 90)
        assert any("timed out" in e for e in errors)

    def test_poll_network_error(self):
        import httpx
        with patch(
            "src.polling.anthropic.safe_get",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            from src.polling.anthropic import poll
            _, errors = poll("sk-ant-api03-" + "A" * 90)
        assert any("Network error" in e for e in errors)

    def test_poll_invalid_json_returns_error(self):
        resp = MagicMock()
        resp.is_success = True
        resp.status_code = 200
        resp.json.side_effect = ValueError("bad json")
        with patch("src.polling.anthropic.safe_get", return_value=resp):
            from src.polling.anthropic import poll
            _, errors = poll("sk-ant-api03-" + "A" * 90)
        assert any("parse" in e for e in errors)

    def test_poll_non_list_data_returns_error(self):
        resp = self._mock_resp({"data": "not-a-list"})
        with patch("src.polling.anthropic.safe_get", return_value=resp):
            from src.polling.anthropic import poll
            _, errors = poll("sk-ant-api03-" + "A" * 90)
        assert any("Unexpected" in e for e in errors)

    def test_poll_ssrf_blocked(self):
        with patch(
            "src.polling.anthropic.safe_get",
            side_effect=ValueError("Blocked request"),
        ):
            from src.polling.anthropic import poll
            _, errors = poll("sk-ant-api03-" + "A" * 90)
        assert any("Blocked" in e for e in errors)

    def test_poll_defaults_date_range(self):
        resp = self._mock_resp({"data": []})
        with patch("src.polling.anthropic.safe_get", return_value=resp) as mock_get:
            from src.polling.anthropic import poll
            poll("sk-ant-api03-" + "A" * 90)
        mock_get.assert_called_once()


# ── OpenAI poll() ─────────────────────────────────────────────────────────────

class TestOpenAICheckStatus:
    def _resp(self, code: int, text: str = "") -> MagicMock:
        r = MagicMock()
        r.is_success = code < 400
        r.status_code = code
        r.text = text
        return r

    def test_200_ok(self):
        assert openai_check_status(self._resp(200), date.today()) == []

    def test_401(self):
        errs = openai_check_status(self._resp(401), date.today())
        assert any("401" in e for e in errs)

    def test_403(self):
        errs = openai_check_status(self._resp(403), date.today())
        assert any("403" in e for e in errs)

    def test_429(self):
        errs = openai_check_status(self._resp(429), date.today())
        assert any("429" in e for e in errs)

    def test_generic_error(self):
        errs = openai_check_status(self._resp(500, "server error"), date.today())
        assert any("500" in e for e in errs)


class TestOpenAIPoll:
    def _mock_resp(self, body: dict, status: int = 200) -> MagicMock:
        r = MagicMock()
        r.is_success = status < 400
        r.status_code = status
        r.json.return_value = body
        r.text = ""
        return r

    def _valid_item(self) -> dict:
        return {
            "snapshot_id": "gpt-4o",
            "n_context_tokens_total": 8000,
            "n_generated_tokens_total": 1500,
            "aggregation_timestamp": 1748793600,
        }

    def test_successful_poll_returns_entries(self):
        resp = self._mock_resp({"data": [self._valid_item()]})
        with patch("src.polling.openai.safe_get", return_value=resp):
            from src.polling.openai import poll
            entries, errors = poll(
                "sk-" + "A" * 40,
                since=date(2026, 6, 1),
                until=date(2026, 6, 1),
            )
        assert errors == []
        assert len(entries) == 1

    def test_poll_stops_on_401(self):
        resp = self._mock_resp({}, status=401)
        with patch("src.polling.openai.safe_get", return_value=resp):
            from src.polling.openai import poll
            entries, errors = poll(
                "sk-bad", since=date(2026, 6, 1), until=date(2026, 6, 3)
            )
        assert entries == []
        assert any("401" in e for e in errors)

    def test_poll_stops_on_403(self):
        resp = self._mock_resp({}, status=403)
        with patch("src.polling.openai.safe_get", return_value=resp):
            from src.polling.openai import poll
            _, errors = poll(
                "sk-bad", since=date(2026, 6, 1), until=date(2026, 6, 3)
            )
        assert any("403" in e for e in errors)

    def test_poll_day_timeout(self):
        import httpx
        with patch(
            "src.polling.openai.safe_get",
            side_effect=httpx.TimeoutException("t"),
        ):
            from src.polling.openai import _poll_day
            _, errors = _poll_day("sk-" + "A" * 40, date(2026, 6, 1))
        assert any("Timeout" in e for e in errors)

    def test_poll_day_network_error(self):
        import httpx
        with patch(
            "src.polling.openai.safe_get",
            side_effect=httpx.ConnectError("conn"),
        ):
            from src.polling.openai import _poll_day
            _, errors = _poll_day("sk-" + "A" * 40, date(2026, 6, 1))
        assert any("Network error" in e for e in errors)

    def test_poll_day_invalid_json(self):
        resp = MagicMock()
        resp.is_success = True
        resp.status_code = 200
        resp.json.side_effect = ValueError("bad")
        with patch("src.polling.openai.safe_get", return_value=resp):
            from src.polling.openai import _poll_day
            _, errors = _poll_day("sk-" + "A" * 40, date(2026, 6, 1))
        assert any("parse" in e for e in errors)

    def test_poll_day_ssrf_blocked(self):
        with patch(
            "src.polling.openai.safe_get",
            side_effect=ValueError("Blocked"),
        ):
            from src.polling.openai import _poll_day
            _, errors = _poll_day("sk-" + "A" * 40, date(2026, 6, 1))
        assert any("Blocked" in e for e in errors)

    def test_poll_defaults_date_range(self):
        resp = self._mock_resp({"data": []})
        with patch("src.polling.openai.safe_get", return_value=resp):
            from src.polling.openai import poll
            poll("sk-" + "A" * 40)


# ── safe_post SSRF ────────────────────────────────────────────────────────────

class TestSafePost:
    def test_disallowed_host_raises(self):
        from src.polling.base import safe_post
        with pytest.raises(ValueError, match="disallowed host"):
            safe_post("https://evil.example.com/steal", headers={})

    def test_cursor_host_allowed_in_allowlist(self):
        from src.polling.base import _ALLOWED_HOSTS
        assert "api.cursor.com" in _ALLOWED_HOSTS


# ── Cursor poll additional branch coverage ────────────────────────────────────

class TestCursorPollCoverage:
    """Targets missing branches in src/polling/cursor.py."""

    def test_poll_default_dates(self):
        """Lines 55-58: since/until default correctly when not supplied."""
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"usageEvents": []}
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll("crsr_" + "A" * 40)
        assert errors == []

    def test_poll_pagination(self):
        """Line 73: page += 1 when a full page is returned."""
        import src.polling.cursor as cursor_mod
        page_size = cursor_mod._PAGE_SIZE
        full_event = {
            "model": "gpt-4o", "kind": "chat", "isTokenBasedCall": True,
            "timestamp": 1748793600000,
            "tokenUsage": {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 0},
            "chargedCents": 1,
        }
        full_page = MagicMock()
        full_page.is_success = True
        full_page.json.return_value = {"usageEvents": [full_event] * page_size}

        empty_page = MagicMock()
        empty_page.is_success = True
        empty_page.json.return_value = {"usageEvents": []}

        with patch("src.polling.cursor.safe_post", side_effect=[full_page, empty_page]):
            from src.polling.cursor import poll
            entries, errors = poll(
                "crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 30)
            )
        assert len(entries) == page_size

    def test_poll_value_error_from_safe_post(self):
        """Line 104: ValueError raised by safe_post (e.g. SSRF guard)."""
        with patch("src.polling.cursor.safe_post", side_effect=ValueError("Blocked")):
            from src.polling.cursor import poll
            entries, errors = poll("crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("Blocked" in e for e in errors)

    def test_poll_request_error(self):
        """Lines 107-108: httpx.RequestError (e.g. ConnectError)."""
        import httpx
        with patch("src.polling.cursor.safe_post", side_effect=httpx.ConnectError("refused")):
            from src.polling.cursor import poll
            entries, errors = poll("crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("Network error" in e for e in errors)

    def test_poll_invalid_json_response(self):
        """Lines 116-117: resp.json() raises, returns error message."""
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.side_effect = ValueError("not json")
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            _, errors = poll("crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert any("JSON" in e for e in errors)

    def test_poll_non_list_events(self):
        """Line 121: usageEvents is not a list → error returned."""
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"usageEvents": "not-a-list"}
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            _, errors = poll("crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert any("Unexpected" in e for e in errors)

    def test_to_entry_exception_skips_event(self):
        """Lines 129-130: exception in _to_entry is swallowed; event is skipped."""
        # inputTokens="abc" → int("abc") raises ValueError inside _to_entry
        bad_event = {
            "model": "gpt-4o", "isTokenBasedCall": True, "timestamp": 1748793600000,
            "tokenUsage": {"inputTokens": "abc", "outputTokens": 200},
            "chargedCents": 1,
        }
        good_event = {
            "model": "gpt-4o", "kind": "chat", "isTokenBasedCall": True,
            "timestamp": 1748793600000,
            "tokenUsage": {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 0},
            "chargedCents": 1,
        }
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"usageEvents": [bad_event, good_event]}
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll("crsr_" + "A" * 40, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert len(entries) == 1
        assert errors == []


class TestCursorToEntryEdgeCases:
    """Targets lines 179-182 and 187 in _to_entry."""

    def _base(self):
        return {
            "model": "claude-3.5-sonnet", "kind": "chat",
            "isTokenBasedCall": True, "timestamp": 1748793600000,
            "tokenUsage": {"inputTokens": 5000, "outputTokens": 1000, "cacheReadTokens": 0},
        }

    def test_total_cents_used_when_no_charged_cents(self):
        """Lines 179-180: totalCents path (chargedCents absent)."""
        event = self._base()
        event["tokenUsage"]["totalCents"] = 5.0
        entry = cursor_to_entry(event)
        assert entry is not None
        assert entry.cost_usd == pytest.approx(0.05)

    def test_no_cost_fields_yields_zero_cost(self):
        """Lines 181-182: else branch — neither chargedCents nor totalCents."""
        event = self._base()
        entry = cursor_to_entry(event)
        assert entry is not None
        assert entry.cost_usd == 0.0

    def test_null_timestamp_returns_none(self):
        """Line 187: timestamp=None → _parse_ts returns None → _to_entry returns None."""
        event = self._base()
        event["timestamp"] = None
        assert cursor_to_entry(event) is None


class TestCursorParseTsGarbage:
    def test_unparseable_string_returns_none(self):
        """Line 220: _parse_ts returns None for a string matching no format."""
        from src.polling.cursor import _parse_ts
        assert _parse_ts("!!!not-a-date-at-all!!!") is None


# ── Key store coverage ───────────────────────────────────────────────────────

class TestKeyStore:
    """Targets missing branches in src/polling/key_store.py (lines 39-41, 80, 105, 118-123)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        import src.polling.key_store as ks
        env_file = tmp_path / ".env.local"
        monkeypatch.setattr(ks, "_ENV_LOCAL", env_file)
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CURSOR_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        yield env_file

    def test_read_oserror_returns_empty(self, _isolate, monkeypatch):
        """Lines 39-41: OSError during file read returns {}."""
        import pathlib
        import src.polling.key_store as ks
        _isolate.write_text("ANTHROPIC_API_KEY=sk-test\n")
        original = pathlib.Path.read_text

        def raises_for_env_file(self, *args, **kwargs):
            if self == _isolate:
                raise OSError("Permission denied")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", raises_for_env_file)
        assert ks._read() == {}

    def test_set_key_unknown_provider_raises(self, _isolate):
        """Line 80: set_key with unrecognised provider → ValueError."""
        from src.polling.key_store import set_key
        with pytest.raises(ValueError, match="Unknown provider"):
            set_key("totally_bogus", "some-key")

    def test_clear_key_writes_remaining_keys(self, _isolate):
        """Line 105: _write(local) is called when sibling keys remain after pop."""
        import src.polling.key_store as ks
        _isolate.write_text("ANTHROPIC_API_KEY=sk-ant\nOPENAI_API_KEY=sk-oai\n")
        ks.clear_key("anthropic")
        data = ks._read()
        assert "OPENAI_API_KEY" in data
        assert "ANTHROPIC_API_KEY" not in data

    def test_key_source_env(self, _isolate, monkeypatch):
        """Line 120: key_source returns 'env' when env var is set."""
        from src.polling.key_store import key_source
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "A" * 90)
        assert key_source("anthropic") == "env"

    def test_key_source_file(self, _isolate):
        """Line 122: key_source returns 'file' when key is only in .env.local."""
        import src.polling.key_store as ks
        _isolate.write_text("ANTHROPIC_API_KEY=sk-ant-test\n")
        assert ks.key_source("anthropic") == "file"

    def test_key_source_none(self, _isolate):
        """Line 123: key_source returns 'none' when no key is configured."""
        from src.polling.key_store import key_source
        assert key_source("anthropic") == "none"

    def test_set_key_writes_file_and_env(self, _isolate):
        """Lines 82-83, 92-96: set_key writes to .env.local when env var absent."""
        import src.polling.key_store as ks
        key = "sk-ant-api03-" + "A" * 90
        ks.set_key("anthropic", key)
        data = ks._read()
        assert data.get("ANTHROPIC_API_KEY") == key

    def test_set_key_skips_when_env_var_set(self, _isolate, monkeypatch):
        """Lines 83-90: set_key returns early when env var already set."""
        import src.polling.key_store as ks
        monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")
        ks.set_key("anthropic", "new-key")
        assert not _isolate.exists()

    def test_clear_key_removes_file_when_last_key(self, _isolate):
        """Lines 106-107: clear_key unlinks .env.local when no keys remain."""
        import src.polling.key_store as ks
        _isolate.write_text("ANTHROPIC_API_KEY=sk-test\n")
        ks.clear_key("anthropic")
        assert not _isolate.exists()

    def test_has_key_returns_bool(self, _isolate):
        """Line 113: has_key returns False when absent, True when present."""
        from src.polling.key_store import has_key
        assert has_key("anthropic") is False
        _isolate.write_text("ANTHROPIC_API_KEY=sk-test\n")
        assert has_key("anthropic") is True
