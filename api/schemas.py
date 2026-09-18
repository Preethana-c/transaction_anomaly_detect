from __future__ import annotations

from pydantic import BaseModel, Field


class StreamStartRequest(BaseModel):
    delay_seconds: float = Field(default=0.3, ge=0.05, le=5.0)


class StreamStatus(BaseModel):
    running: bool
    total_processed: int
    total_flagged: int
    started_at: float | None
    elapsed_seconds: float | None


class Transaction(BaseModel):
    id: int | str | None = None
    ingested_at: str | None = None
    original_time: float
    amount: float
    anomaly_score: float
    threshold: float
    flagged: bool
    is_synthetic: bool
    true_class: int | None = None
    model_name: str
