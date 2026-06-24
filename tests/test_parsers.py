"""Tests for billing CSV parsers."""
import io
from datetime import datetime

import pytest

from src.parsers.anthropic import parse_anthropic_csv
from src.parsers.generic import detect_provider, parse_auto
from src.parsers.openai import parse_openai_csv
from src.models import Provider, WorkloadClass


ANTHROPIC_CSV = b"""Date,Organization,Project,Model,Input tokens,Output tokens,Cache read tokens,Cache write tokens,Cost (USD)
2026-06-01,ACME,default,claude-opus-4,10000,2000,500,100,0.18
2026-06-02,ACME,search,claude-haiku-4-5,5000,800,0,0,0.002
2026-06-03,ACME,rag,claude-3-5-haiku-20251022,20000,3000,1000,200,0.025
"""

OPENAI_CSV = b"""date,model,input_tokens,output_tokens,reasoning_tokens,cost
2026-06-01,gpt-4o,8000,1500,0,0.0235
2026-06-02,o1-mini,3000,400,2100,0.015
2026-06-03,text-embedding-3-small,100000,0,0,0.002
"""

EMPTY_CSV = b"""Date,Model\n"""

MALFORMED_CSV = b"not,a,valid,billing,file\n1,2,3,4,5\n"


class TestAnthropicParser:
    def test_parses_valid_csv(self):
        bill = parse_anthropic_csv(ANTHROPIC_CSV)
        assert bill.provider == Provider.anthropic
        assert len(bill.entries) == 3
        assert bill.total_cost_usd > 0

    def test_model_names_preserved(self):
        bill = parse_anthropic_csv(ANTHROPIC_CSV)
        models = [e.model for e in bill.entries]
        assert "claude-opus-4" in models
        assert "claude-haiku-4-5" in models

    def test_workload_class_inferred(self):
        bill = parse_anthropic_csv(ANTHROPIC_CSV)
        opus_entry = next(e for e in bill.entries if "opus" in e.model)
        assert opus_entry.workload_class == WorkloadClass.reason

    def test_cache_tokens_added_to_input(self):
        bill = parse_anthropic_csv(ANTHROPIC_CSV)
        opus_entry = next(e for e in bill.entries if "opus" in e.model)
        # input(10000) + cache_read(500) + cache_write(100) = 10600
        assert opus_entry.input_tokens == 10600

    def test_empty_csv_returns_empty_bill(self):
        bill = parse_anthropic_csv(EMPTY_CSV)
        assert len(bill.entries) == 0

    def test_file_too_large_raises(self):
        big_data = b"a" * (51 * 1024 * 1024)
        with pytest.raises(ValueError, match="too large"):
            parse_anthropic_csv(big_data)

    def test_total_cost_summed(self):
        bill = parse_anthropic_csv(ANTHROPIC_CSV)
        expected = sum(e.cost_usd for e in bill.entries)
        assert abs(bill.total_cost_usd - expected) < 1e-9


class TestOpenAIParser:
    def test_parses_valid_csv(self):
        bill = parse_openai_csv(OPENAI_CSV)
        assert bill.provider == Provider.openai
        assert len(bill.entries) == 3

    def test_reasoning_tokens_captured(self):
        bill = parse_openai_csv(OPENAI_CSV)
        o1_entry = next(e for e in bill.entries if "o1" in e.model)
        assert o1_entry.reasoning_tokens == 2100

    def test_embedding_workload_class(self):
        bill = parse_openai_csv(OPENAI_CSV)
        embed_entry = next(e for e in bill.entries if "embed" in e.model)
        assert embed_entry.workload_class == WorkloadClass.rag


class TestAutoDetect:
    def test_detects_anthropic(self):
        assert detect_provider(ANTHROPIC_CSV) == Provider.anthropic

    def test_detects_openai(self):
        assert detect_provider(OPENAI_CSV) == Provider.openai

    def test_unknown_returns_other(self):
        assert detect_provider(MALFORMED_CSV) == Provider.other

    def test_auto_routes_to_anthropic(self):
        bill = parse_auto(ANTHROPIC_CSV)
        assert bill.provider == Provider.anthropic

    def test_auto_routes_to_openai(self):
        bill = parse_auto(OPENAI_CSV)
        assert bill.provider == Provider.openai
