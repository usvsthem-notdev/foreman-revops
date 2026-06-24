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

    def test_cost_from_total_cents_when_no_charged(self):
        # chargedCents absent → fall back to totalCents
        event = self._event()
        del event["chargedCents"]
        event["tokenUsage"]["totalCents"] = 0.05
        entry = cursor_to_entry(event)
        assert entry is not None
        assert entry.cost_usd == pytest.approx(0.0005)

    def test_cost_estimated_when_both_absent(self):
        # Neither chargedCents nor totalCents → estimation fallback
        event = self._event()
        del event["chargedCents"]
        del event["tokenUsage"]["totalCents"]
        entry = cursor_to_entry(event)
        assert entry is not None
        assert entry.cost_usd > 0.0

    def test_bad_timestamp_returns_none(self):
        entry = cursor_to_entry(self._event(timestamp=None))
        assert entry is None

    def test_cache_write_tokens_included_in_reasoning(self):
        event = self._event()
        event["tokenUsage"]["cacheReadTokens"] = 100
        event["tokenUsage"]["cacheWriteTokens"] = 50
        entry = cursor_to_entry(event)
        assert entry is not None
        assert entry.reasoning_tokens == 150   # read + write


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

    def test_unparseable_string_returns_none(self):
        assert cursor_parse_ts("not-a-date-at-all!!") is None


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

    def test_poll_value_error_from_safe_post(self):
        # safe_post raises ValueError (e.g. SSRF block)
        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", side_effect=ValueError("Blocked")):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("Blocked" in e for e in errors)

    def test_poll_request_error_from_safe_post(self):
        import httpx
        key = "crsr_" + "A" * 40
        with patch(
            "src.polling.cursor.safe_post",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("Network error" in e for e in errors)

    def test_poll_invalid_json_returns_error(self):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")
        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("JSON" in e for e in errors)

    def test_poll_usage_events_not_list(self):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"usageEvents": "not-a-list"}
        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert any("Unexpected" in e for e in errors)

    def test_poll_skips_bad_event_silently(self):
        # chargedCents="abc" causes int() to raise inside _to_entry
        bad_event = {
            "model": "gpt-4o",
            "isTokenBasedCall": True,
            "timestamp": 1748793600000,
            "tokenUsage": {"inputTokens": 1000, "outputTokens": 200},
            "chargedCents": "abc",
        }
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"usageEvents": [bad_event]}
        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", return_value=mock_resp):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert entries == []
        assert errors == []   # logged at DEBUG, not surfaced as error

    def test_poll_default_date_range(self):
        # Covers the since=None and until=None branches (lines 56, 58)
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"usageEvents": []}
        key = "crsr_" + "A" * 40
        with patch("src.polling.cursor.safe_post", return_value=mock_resp) as mock_post:
            from src.polling.cursor import poll
            poll(key)   # no since / until → defaults applied
        mock_post.assert_called_once()

    def test_poll_paginates_full_page(self):
        # First response: exactly _PAGE_SIZE (500) events → triggers page += 1
        # Second response: 0 events → stops
        from src.polling.cursor import _PAGE_SIZE
        full_page = [self._fake_event() for _ in range(_PAGE_SIZE)]
        resp_full = MagicMock()
        resp_full.is_success = True
        resp_full.status_code = 200
        resp_full.json.return_value = {"usageEvents": full_page}
        resp_empty = MagicMock()
        resp_empty.is_success = True
        resp_empty.status_code = 200
        resp_empty.json.return_value = {"usageEvents": []}
        key = "crsr_" + "A" * 40
        with patch(
            "src.polling.cursor.safe_post",
            side_effect=[resp_full, resp_empty],
        ):
            from src.polling.cursor import poll
            entries, errors = poll(key, since=date(2026, 6, 1), until=date(2026, 6, 1))
        assert len(entries) == _PAGE_SIZE
        assert errors == []


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


# ── Cursor cost estimation ─────────────────────────────────────────────────────

class TestCursorEstimation:
    def _cost(self, model, input_tok, cache_read=0, cache_write=0, output_tok=0):
        from src.polling.cursor import _estimate_cursor_cost
        return _estimate_cursor_cost(model, input_tok, cache_read, cache_write, output_tok)

    def test_claude_cache_read_at_10pct(self):
        cost_input  = self._cost("claude-sonnet", 1_000_000, 0, 0, 0)
        cost_cached = self._cost("claude-sonnet", 0, 1_000_000, 0, 0)
        assert cost_cached == pytest.approx(cost_input * 0.10, rel=1e-6)

    def test_claude_cache_write_at_125pct(self):
        cost_input = self._cost("claude-sonnet", 1_000_000, 0, 0, 0)
        cost_write = self._cost("claude-sonnet", 0, 0, 1_000_000, 0)
        assert cost_write == pytest.approx(cost_input * 1.25, rel=1e-6)

    def test_gpt_cache_read_at_50pct(self):
        cost_input  = self._cost("gpt-4o", 1_000_000, 0, 0, 0)
        cost_cached = self._cost("gpt-4o", 0, 1_000_000, 0, 0)
        assert cost_cached == pytest.approx(cost_input * 0.50, rel=1e-6)

    def test_unknown_model_uses_fallback(self):
        cost = self._cost("some-unknown-llm", 1_000_000, 0, 0, 0)
        # fallback: $2.5/M input
        assert cost == pytest.approx(2.5, rel=1e-6)

    def test_output_tokens_priced_correctly(self):
        # claude-sonnet output: $15/M
        cost = self._cost("claude-sonnet", 0, 0, 0, 1_000_000)
        assert cost == pytest.approx(15.0, rel=1e-4)


# ── Gemini pricing ────────────────────────────────────────────────────────────

class TestGeminiPricing:
    def _cost(self, model, input_tok, cache_read=0, thinking_tok=0, output_tok=0):
        from src.parsers.gemini import _estimate_gemini_cost
        return _estimate_gemini_cost(model, input_tok, cache_read, thinking_tok, output_tok)

    def test_flash_cache_read_cheaper_than_input(self):
        cost_input  = self._cost("gemini-2.5-flash", 1_000_000, 0, 0, 0)
        cost_cached = self._cost("gemini-2.5-flash", 0, 1_000_000, 0, 0)
        # Flash cache_read = $0.0375/M vs input = $0.15/M → 25%
        assert cost_cached == pytest.approx(cost_input * 0.25, rel=1e-6)

    def test_flash_thinking_tokens_priced_higher_than_output(self):
        cost_output  = self._cost("gemini-2.5-flash", 0, 0, 0, 1_000_000)
        cost_thinking = self._cost("gemini-2.5-flash", 0, 0, 1_000_000, 0)
        # Flash thinking = $3.50/M vs output = $0.60/M
        assert cost_thinking > cost_output * 5

    def test_pro_input_price(self):
        cost = self._cost("gemini-2.5-pro", 1_000_000, 0, 0, 0)
        assert cost == pytest.approx(1.25, rel=1e-4)

    def test_unknown_gemini_model_uses_flash_fallback(self):
        cost = self._cost("gemini-future-model", 1_000_000, 0, 0, 0)
        assert cost == pytest.approx(0.15, rel=1e-4)

    def test_parse_gemini_csv_returns_warning(self):
        from src.parsers.gemini import parse_gemini_csv
        bill = parse_gemini_csv(b"date,model,cost\n2026-06-01,gemini-2.5-pro,1.0\n")
        assert bill.provider.value == "gemini"
        assert len(bill.entries) == 0
        assert any("not yet" in w for w in bill.parse_warnings)
