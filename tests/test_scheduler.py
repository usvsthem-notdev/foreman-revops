"""Tests for the auto-poll scheduler."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.polling.scheduler import (
    _due_for_poll,
    _heartbeat_path,
    load_config,
    read_heartbeat,
    run_once,
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

    def test_write_heartbeat_oserror_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        with patch("src.polling.scheduler.Path.write_text", side_effect=OSError("no disk")):
            write_heartbeat(["anthropic"], errors=0)  # must not raise

    def test_read_heartbeat_oserror_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        # Create the file so it exists, then make read fail
        hb_path = tmp_path / ".scheduler_heartbeat"
        hb_path.write_text("timestamp=2026-06-01\n")
        with patch("src.polling.scheduler.Path.read_text", side_effect=OSError("perm")):
            result = read_heartbeat()
        assert result is None


# ── run_once ──────────────────────────────────────────────────────────────────

class TestRunOnce:
    def _config(self, poller=None):
        if poller is None:
            poller = lambda key, since, until: ([], [])  # noqa: E731
        return {
            "interval_hours": 6,
            "lookback_days": 2,
            "providers": {"anthropic": poller},
        }

    def test_skips_provider_not_due(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=False),
            patch("src.polling.scheduler.get_key", return_value="sk-ant-api03-" + "A" * 90),
        ):
            polled, errors = run_once(self._config())
        assert polled == []
        assert errors == 0

    def test_skips_when_key_disappears(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=True),
            patch("src.polling.scheduler.get_key", return_value=None),
        ):
            polled, errors = run_once(self._config())
        assert polled == []

    def test_polls_and_inserts(self, tmp_path, monkeypatch):
        from src.models import EntrySource, Provider, SpendEntry
        from datetime import datetime as dt
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        fake_entry = SpendEntry(
            timestamp=dt(2026, 6, 1),
            provider=Provider.anthropic,
            model="claude-opus-4",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.1,
            source=EntrySource.api,
        )
        mock_poller = lambda key, since, until: ([fake_entry], [])  # noqa: E731
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=True),
            patch("src.polling.scheduler.get_key", return_value="sk-ant-api03-" + "A" * 90),
            patch("src.polling.scheduler.insert_entries_bulk", return_value=1),
            patch("src.polling.scheduler.set_poll_cursor"),
        ):
            polled, total_errors = run_once(self._config(poller=mock_poller))
        assert "anthropic" in polled
        assert total_errors == 0

    def test_counts_poller_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        mock_poller = lambda key, since, until: ([], ["err1", "err2"])  # noqa: E731
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=True),
            patch("src.polling.scheduler.get_key", return_value="sk-ant-api03-" + "A" * 90),
            patch("src.polling.scheduler.insert_entries_bulk", return_value=0),
            patch("src.polling.scheduler.set_poll_cursor"),
        ):
            polled, total_errors = run_once(self._config(poller=mock_poller))
        assert total_errors == 2

    def test_handles_poller_exception(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        def bad_poller(key, since, until):
            raise RuntimeError("unexpected!")
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=True),
            patch("src.polling.scheduler.get_key", return_value="sk-ant-api03-" + "A" * 90),
        ):
            polled, total_errors = run_once(self._config(poller=bad_poller))
        assert polled == []
        assert total_errors == 1

    def test_no_entries_returned_still_sets_cursor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))
        mock_poller = lambda key, since, until: ([], [])  # noqa: E731
        with (
            patch("src.polling.scheduler._due_for_poll", return_value=True),
            patch("src.polling.scheduler.get_key", return_value="sk-ant-api03-" + "A" * 90),
            patch("src.polling.scheduler.insert_entries_bulk", return_value=0) as mock_insert,
            patch("src.polling.scheduler.set_poll_cursor") as mock_cursor,
        ):
            polled, _ = run_once(self._config(poller=mock_poller))
        mock_insert.assert_not_called()
        mock_cursor.assert_called_once()
        assert "anthropic" in polled


# ── load_config edge cases ────────────────────────────────────────────────────

class TestLoadConfigEdgeCases:
    def _set_env(self, monkeypatch, **kwargs):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY",    raising=False)
        monkeypatch.delenv("CURSOR_API_KEY",    raising=False)
        monkeypatch.delenv("GEMINI_API_KEY",    raising=False)
        for k, v in kwargs.items():
            monkeypatch.setenv(k, str(v))

    def test_bad_lookback_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.polling.key_store._ENV_LOCAL", tmp_path / ".env.local")
        self._set_env(
            monkeypatch,
            ANTHROPIC_API_KEY="sk-ant-api03-" + "K" * 90,
            FOREMAN_POLL_PROVIDERS="anthropic",
            FOREMAN_POLL_LOOKBACK_DAYS="not-a-number",
        )
        config = load_config()
        assert config["lookback_days"] == 2  # default


# ── _handle_signal and run() loop ─────────────────────────────────────────────

class TestSchedulerRunAndSignal:
    def test_handle_signal_sets_shutdown(self, monkeypatch):
        """Lines 255-256: _handle_signal sets _SHUTDOWN = True."""
        import src.polling.scheduler as sched
        monkeypatch.setattr(sched, "_SHUTDOWN", False)
        sched._handle_signal(15, None)
        assert sched._SHUTDOWN is True

    def test_run_exits_after_single_cycle(self, tmp_path, monkeypatch):
        """Lines 264-278, 282-285, 288: run() loop with polled providers."""
        import src.polling.scheduler as sched
        monkeypatch.setattr(sched, "_SHUTDOWN", False)
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))

        def fake_run_once(cfg):
            sched._SHUTDOWN = True
            return (["anthropic"], 0)

        with (
            patch("src.polling.scheduler.run_once", side_effect=fake_run_once),
            patch("src.polling.scheduler.write_heartbeat"),
            patch("signal.signal"),
        ):
            sched.run({"interval_hours": 6, "lookback_days": 2, "providers": {}}, tick_seconds=1)

    def test_run_catches_run_once_exception(self, tmp_path, monkeypatch):
        """Lines 279-280: exception in run_once is caught and logged."""
        import src.polling.scheduler as sched
        monkeypatch.setattr(sched, "_SHUTDOWN", False)
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))

        call_count = [0]

        def bad_then_shutdown(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            sched._SHUTDOWN = True
            return ([], 0)

        with (
            patch("src.polling.scheduler.run_once", side_effect=bad_then_shutdown),
            patch("signal.signal"),
        ):
            sched.run({"interval_hours": 6, "lookback_days": 2, "providers": {}}, tick_seconds=0)

        assert call_count[0] == 2

    def test_run_calls_time_sleep(self, tmp_path, monkeypatch):
        """Line 286: time.sleep is called when _SHUTDOWN is False during tick."""
        import src.polling.scheduler as sched
        monkeypatch.setattr(sched, "_SHUTDOWN", False)
        monkeypatch.setenv("FOREMAN_DB_PATH", str(tmp_path / "foreman.db"))

        slept = [False]

        def fake_sleep(n):
            slept[0] = True
            sched._SHUTDOWN = True

        with (
            patch("src.polling.scheduler.run_once", return_value=([], 0)),
            patch("time.sleep", side_effect=fake_sleep),
            patch("signal.signal"),
        ):
            sched.run({"interval_hours": 6, "lookback_days": 2, "providers": {}}, tick_seconds=2)

        assert slept[0]
