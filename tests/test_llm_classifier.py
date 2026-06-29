"""
Tests for the LLM-assisted classifier.

Uses monkeypatching to avoid real API calls — the unit under test is the
parsing and orchestration logic, not the network.
"""
from __future__ import annotations

import json

import pytest

from src.analytics.llm_classifier import _parse_response, classify_entry_with_llm
from src.models import AICategory


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_clean_json(self):
        raw = '{"category": "code_gen", "confidence": 0.85, "reasoning": "Cursor is a coding tool."}'
        cat, conf, reason = _parse_response(raw)
        assert cat == AICategory.code_gen
        assert conf == pytest.approx(0.85)
        assert "coding" in reason.lower()

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"category\": \"research\", \"confidence\": 0.9, \"reasoning\": \"RAG pipeline.\"}\n```"
        cat, conf, _ = _parse_response(raw)
        assert cat == AICategory.research
        assert conf == pytest.approx(0.9)

    def test_unknown_category_falls_back(self):
        raw = '{"category": "completely_made_up", "confidence": 0.7, "reasoning": "?"}'
        cat, conf, _ = _parse_response(raw)
        assert cat == AICategory.unknown

    def test_confidence_capped_at_0_95(self):
        raw = '{"category": "code_gen", "confidence": 1.0, "reasoning": "Certain."}'
        _, conf, _ = _parse_response(raw)
        assert conf <= 0.95

    def test_confidence_floored_at_0(self):
        raw = '{"category": "code_gen", "confidence": -0.5, "reasoning": "Odd."}'
        _, conf, _ = _parse_response(raw)
        assert conf >= 0.0

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_response("Sorry, I cannot classify this.")

    def test_raises_on_malformed_json(self):
        with pytest.raises(ValueError):
            _parse_response("{category: code_gen}")  # unquoted key

    def test_all_valid_categories_accepted(self):
        for cat in ("code_gen", "research", "document_office", "unknown"):
            raw = json.dumps({"category": cat, "confidence": 0.8, "reasoning": "ok"})
            result_cat, _, _ = _parse_response(raw)
            assert result_cat.value == cat


# ---------------------------------------------------------------------------
# classify_entry_with_llm — mocked API
# ---------------------------------------------------------------------------

class TestClassifyEntryWithLlm:
    def _fake_entry(self):
        return {
            "id": "test-id-001",
            "provider": "openai",
            "model": "gpt-4o",
            "workload_class": "agents",
            "feature": "legal document review",
            "notes": "batch processing contracts for compliance team",
            "team": "legal",
        }

    def test_anthropic_path(self, monkeypatch):
        import src.analytics.llm_classifier as mod

        def fake_post(url, headers, json=None, **kwargs):
            class FakeResp:
                status_code = 200
                def is_success(self): return True
                def json(self):
                    return {"content": [{"text": '{"category":"document_office","confidence":0.88,"reasoning":"Legal doc review."}'}]}
            r = FakeResp()
            r.is_success = True
            return r

        monkeypatch.setattr("src.polling.base.safe_post", fake_post)

        cat, conf, reason = classify_entry_with_llm(
            self._fake_entry(), api_key="sk-ant-api03-" + "A" * 93, llm_provider="anthropic"
        )
        assert cat == AICategory.document_office
        assert conf == pytest.approx(0.88)
        assert reason

    def test_openai_path(self, monkeypatch):
        import src.analytics.llm_classifier as mod

        def fake_post(url, headers, json=None, **kwargs):
            class FakeResp:
                status_code = 200
                is_success = True
                def json(self):
                    return {"choices": [{"message": {"content": '{"category":"research","confidence":0.75,"reasoning":"RAG pipeline."}'}}]}
            return FakeResp()

        monkeypatch.setattr("src.polling.base.safe_post", fake_post)

        cat, conf, _ = classify_entry_with_llm(
            self._fake_entry(), api_key="sk-proj-" + "B" * 50, llm_provider="openai"
        )
        assert cat == AICategory.research

    def test_api_error_raises(self, monkeypatch):
        def fake_post(url, headers, json=None, **kwargs):
            class FakeResp:
                status_code = 401
                is_success = False
                text = "Unauthorized"
            return FakeResp()

        monkeypatch.setattr("src.polling.base.safe_post", fake_post)

        with pytest.raises(ValueError, match="401"):
            classify_entry_with_llm(
                self._fake_entry(), api_key="bad-key", llm_provider="anthropic"
            )

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            classify_entry_with_llm(self._fake_entry(), api_key="key", llm_provider="mistral")
