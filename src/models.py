from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Provider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    google = "google"
    mistral = "mistral"
    together = "together"
    other = "other"


class WorkloadClass(str, Enum):
    extract = "extract"
    rag = "rag"
    reason = "reason"
    agents = "agents"
    coding = "coding"
    unknown = "unknown"


class BudgetPeriod(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class EntrySource(str, Enum):
    manual = "manual"
    anthropic_csv = "anthropic_csv"
    openai_csv = "openai_csv"
    google_csv = "google_csv"
    api = "api"


_SAFE_TEXT = re.compile(r"^[\w\s\-\.\,\:\(\)\/\#\@]+$")


def _sanitize(value: str, max_len: int = 128) -> str:
    value = value.strip()[:max_len]
    return value


class SpendEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    provider: Provider
    model: str = Field(min_length=1, max_length=128)
    workload_class: WorkloadClass = WorkloadClass.unknown
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0, default=0)
    cost_usd: float = Field(ge=0.0)
    is_local: bool = False  # sage = True (absorbed), clay = False (frontier)
    team: Optional[str] = Field(default=None, max_length=64)
    feature: Optional[str] = Field(default=None, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=512)
    source: EntrySource = EntrySource.manual

    @field_validator("model", "team", "feature", "notes", mode="before")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return str(v).strip()

    @model_validator(mode="after")
    def total_tokens_positive(self) -> "SpendEntry":
        total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        if total == 0 and self.cost_usd > 0:
            pass  # allow cost-only entries (e.g., from invoices)
        return self


class Budget(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=64)
    amount_usd: float = Field(gt=0)
    period: BudgetPeriod
    provider: Optional[Provider] = None
    team: Optional[str] = Field(default=None, max_length=64)
    alert_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("name", "team", mode="before")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return str(v).strip()


class ParsedBill(BaseModel):
    source_file: str = Field(max_length=256)
    provider: Provider
    entries: list[SpendEntry]
    parse_warnings: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0

    def summarize(self) -> dict:
        return {
            "provider": self.provider,
            "entry_count": len(self.entries),
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "warnings": len(self.parse_warnings),
        }
