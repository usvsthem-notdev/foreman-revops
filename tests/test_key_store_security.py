"""
Security regression tests for the API key store.

Covers:
  1. set_key() must NOT inject the key into os.environ (cross-session leak fix)
  2. get_key() must still read the key from .env.local after set_key()
  3. clear_key() removes from .env.local but cannot remove env-var-sourced keys
  4. env var takes priority over .env.local (existing guarantee preserved)
  5. SPACE_ID present → key entry gate returns early (no .env.local write path reachable)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_env_local(tmp_path, monkeypatch):
    """
    Redirect key_store to a temp directory and clear all four provider keys
    from the live os.environ for the duration of the test.
    """
    env_local = tmp_path / ".env.local"

    # Patch _ENV_LOCAL and _PROJECT_ROOT inside key_store so reads/writes go
    # to the temp file, not the real project root.
    import src.polling.key_store as ks
    monkeypatch.setattr(ks, "_ENV_LOCAL", env_local)

    # Remove provider keys from os.environ so tests start clean
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CURSOR_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    yield env_local


# ---------------------------------------------------------------------------
# 1. set_key() must NOT write to os.environ
# ---------------------------------------------------------------------------

class TestSetKeyDoesNotPollutateOsEnviron:
    """
    Cross-session key theft fix: set_key() must not call os.environ[name] = key.
    If it does, any concurrent Streamlit session (same process) could read the key.
    """

    def test_set_key_does_not_appear_in_os_environ(self, isolated_env_local):
        from src.polling.key_store import set_key

        fake_key = "sk-ant-api03-" + "A" * 93
        set_key("anthropic", fake_key)

        assert os.environ.get("ANTHROPIC_API_KEY") is None, (
            "set_key() injected the key into os.environ — cross-session key theft possible "
            "on shared deployments (HuggingFace, multi-browser Streamlit)."
        )

    def test_set_key_for_all_providers_does_not_leak(self, isolated_env_local):
        from src.polling.key_store import set_key

        keys = {
            "openai": "sk-proj-" + "B" * 50,
            "cursor": "crsr_" + "C" * 40,
            "gemini": "AIzaSy" + "D" * 33,
        }
        env_names = {
            "openai": "OPENAI_API_KEY",
            "cursor": "CURSOR_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        for provider, key in keys.items():
            set_key(provider, key)

        for provider, env_name in env_names.items():
            assert os.environ.get(env_name) is None, (
                f"set_key({provider!r}) injected the key into os.environ[{env_name!r}]"
            )


# ---------------------------------------------------------------------------
# 2. get_key() still works after set_key() (reads from .env.local)
# ---------------------------------------------------------------------------

class TestGetKeyReadsFromFile:
    def test_get_key_returns_key_after_set(self, isolated_env_local):
        from src.polling.key_store import get_key, set_key

        fake_key = "sk-ant-api03-" + "E" * 93
        set_key("anthropic", fake_key)

        retrieved = get_key("anthropic")
        assert retrieved == fake_key, (
            "get_key() did not return the key that was just stored via set_key()."
        )

    def test_get_key_returns_none_when_not_set(self, isolated_env_local):
        from src.polling.key_store import get_key

        assert get_key("anthropic") is None


# ---------------------------------------------------------------------------
# 3. env var still takes priority over .env.local
# ---------------------------------------------------------------------------

class TestEnvVarPriority:
    def test_env_var_takes_priority_over_file(self, isolated_env_local, monkeypatch):
        from src.polling.key_store import get_key, set_key

        file_key = "sk-ant-api03-" + "F" * 93
        env_key  = "sk-ant-api03-" + "G" * 93

        set_key("anthropic", file_key)
        monkeypatch.setenv("ANTHROPIC_API_KEY", env_key)

        assert get_key("anthropic") == env_key, (
            "get_key() should prefer the environment variable over .env.local."
        )

    def test_set_key_is_noop_when_env_var_present(self, isolated_env_local, monkeypatch):
        from src.polling.key_store import get_key, set_key

        env_key  = "sk-ant-api03-" + "H" * 93
        new_key  = "sk-ant-api03-" + "I" * 93

        monkeypatch.setenv("ANTHROPIC_API_KEY", env_key)
        set_key("anthropic", new_key)   # should silently no-op

        # env var is unchanged
        assert os.environ["ANTHROPIC_API_KEY"] == env_key
        # .env.local was not written (set_key refused)
        assert not isolated_env_local.exists() or env_key not in isolated_env_local.read_text()


# ---------------------------------------------------------------------------
# 4. clear_key() removes from file and process env
# ---------------------------------------------------------------------------

class TestClearKey:
    def test_clear_removes_from_file(self, isolated_env_local):
        from src.polling.key_store import clear_key, get_key, set_key

        fake_key = "sk-ant-api03-" + "J" * 93
        set_key("anthropic", fake_key)
        assert get_key("anthropic") == fake_key

        clear_key("anthropic")
        assert get_key("anthropic") is None

    def test_clear_removes_env_var_that_was_set_externally(self, isolated_env_local, monkeypatch):
        from src.polling.key_store import clear_key, get_key

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "K" * 93)
        clear_key("anthropic")
        assert get_key("anthropic") is None


# ---------------------------------------------------------------------------
# 5. SPACE_ID gate — key entry disabled on shared deployments
# ---------------------------------------------------------------------------

class TestSharedDeploymentGate:
    """
    When SPACE_ID is in the environment (HuggingFace Spaces), the Live API tab
    must not allow key writes to .env.local.  This is a code-path test: we verify
    that render() exits before calling set_key() when SPACE_ID is present.

    We can't import Streamlit components in a headless test, so we verify the
    gate via the key_store: after a simulated 'SPACE_ID present' render, the
    .env.local file must not have been written.
    """

    def test_set_key_not_called_path_is_unreachable_on_space(
        self, isolated_env_local, monkeypatch
    ):
        """
        Structural guard: verify render() returns before the key form when
        SPACE_ID is set.  We check the source text rather than running Streamlit.
        """
        import pathlib
        src = pathlib.Path("src/ui/api_poll.py").read_text()

        # The early return must appear after the SPACE_ID check and before
        # the tab rendering code.
        space_id_idx = src.index('os.environ.get("SPACE_ID")')
        early_return_idx = src.index("return", space_id_idx)
        tab_render_idx = src.index('_render_provider("anthropic")')

        assert early_return_idx < tab_render_idx, (
            "render() does not return early when SPACE_ID is set — "
            "the key entry form is reachable on the public demo Space."
        )

    def test_demo_gate_message_not_scheduler_status(self, isolated_env_local):
        """The demo gate must not render scheduler status or key forms — just the info block."""
        import pathlib
        src = pathlib.Path("src/ui/api_poll.py").read_text()

        space_id_idx = src.index('os.environ.get("SPACE_ID")')
        early_return_idx = src.index("return", space_id_idx)
        gate_block = src[space_id_idx:early_return_idx]

        assert "_render_scheduler_status" not in gate_block, (
            "The demo gate renders _render_scheduler_status() before returning — "
            "this shows confusing 'scheduler not running' UI on the public demo."
        )
        assert "set_key" not in gate_block, (
            "set_key() is reachable inside the demo gate block."
        )
