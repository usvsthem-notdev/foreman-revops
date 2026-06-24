"""Tests for the auto-poll scheduler."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.polling.scheduler import (
    _due_for_poll,
    _heartbeat_path,
    load_config,
    read_heartbeat,
    write_heartbeat,
)

# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def _set_env(self, monkeypatch, anthropic_key="", openai_key="", **kwargs):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        if anthropic_key:
            monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
        if openai_key:
            monkeypatch.setenv("OPENAI_API_KEY", openai_key)
        for k, v in kwargs.items():
            monkeypatch.setenv(k, str(v))

    def test_raises_when_no_keys(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(monkeypatch)
        with pytest.raises(ValueError, match="No providers configured"):
            load_config()

    def test_loads_anthropic_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic",
        )
        config = load_config()
        assert "anthropic" in config["providers"]
        assert "openai" not in config["providers"]

    def test_loads_both_providers(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            openai_key="sk-" + "K" * 40,
            FOREMAN_POLL_PROVIDERS="anthropic,openai",
        )
        config = load_config()
        assert "anthropic" in config["providers"]
        assert "openai" in config["providers"]

    def test_interval_minimum_enforced(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic",
            FOREMAN_POLL_INTERVAL_HOURS="0",
        )
        config = load_config()
        assert config["interval_hours"] >= 1

    def test_lookback_capped_at_7(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic",
            FOREMAN_POLL_LOOKBACK_DAYS="999",
        )
        config = load_config()
        assert config["lookback_days"] <= 7

    def test_unknown_provider_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic,nonexistent",
        )
        config = load_config()
        assert "nonexistent" not in config["providers"]
        assert "anthropic" in config["providers"]

    def test_non_integer_interval_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            anthropic_key="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic",
            FOREMAN_POLL_INTERVAL_HOURS="not-a-number",
        )
        config = load_config()
        assert config["interval_hours"] == 6  # default


# ── Due-for-poll check ────────────────────────────────────────────────────────

class TestDueForPoll:
    def test_no_cursor_means_due(self):
        with patch("src.polling.scheduler.get_poll_cursor", return_value=None):
            assert _due_for_poll("anthropic", interval_hours=6) is True

    def test_recent_poll_not_due(self):
        last = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        with patch(
            "src.polling.scheduler.get_poll_cursor",
            return_value={"last_polled": last},
        ):
            assert _due_for_poll("anthropic", interval_hours=6) is False

    def test_old_poll_is_due(self):
        last = (datetime.utcnow() - timedelta(hours=8)).isoformat()
        with patch(
            "src.polling.scheduler.get_poll_cursor",
            return_value={"last_polled": last},
        ):
            assert _due_for_poll("anthropic", interval_hours=6) is True

    def test_malformed_cursor_treated_as_due(self):
        with patch(
            "src.polling.scheduler.get_poll_cursor",
            return_value={"last_polled": "not-a-date"},
        ):
            assert _due_for_poll("anthropic", interval_hours=6) is True


# ── Heartbeat read/write ──────────────────────────────────────────────────────

class TestHeartbeat:
    def test_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))

        write_heartbeat(["anthropic", "openai"], errors=0)
        hb = read_heartbeat()

        assert hb is not None
        assert hb["providers"] == "anthropic,openai"
        assert hb["errors"] == "0"
        assert "timestamp" in hb

    def test_read_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        assert read_heartbeat() is None

    def test_write_errors_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        write_heartbeat(["openai"], errors=3)
        hb = read_heartbeat()
        assert hb is not None
        assert hb["errors"] == "3"

    def test_heartbeat_path_is_beside_db(self, tmp_path, monkeypatch):
        db = tmp_path / "data" / "foreman.db"
        db.parent.mkdir()
        monkeypatch.setenv("FOREMAN_DB_PATH", str(db))
        path = _heartbeat_path()
        assert path.parent == db.parent
        assert path.name == ".scheduler_heartbeat"
