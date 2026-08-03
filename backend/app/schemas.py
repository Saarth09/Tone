from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProbeOut(BaseModel):
    id: str
    category: str
    prompt: str
    description: str


class SampleOut(BaseModel):
    id: int
    probe_id: str
    category: str
    prompt: str
    response: str
    is_baseline: bool
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    fact_ok: Optional[bool] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DriftScoreOut(BaseModel):
    id: int
    category: str
    mmd_score: float
    kl_score: float
    cosine_score: float
    combined_score: float
    threshold: float
    is_alert: bool
    sample_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class StatusOut(BaseModel):
    baseline_ready: bool
    demo_mode: bool
    sample_interval_minutes: int
    llm_model: str
    llm_base_url: str
    total_samples: int
    baseline_samples: int
    live_samples: int
    latest_overall_score: Optional[float] = None
    drifted: Optional[bool] = None


class BaselineRequest(BaseModel):
    runs: Optional[int] = Field(default=None, description="Total probe cycles to run for baseline")


class SampleRequest(BaseModel):
    probe_ids: Optional[list[str]] = None


class CompareOut(BaseModel):
    probe_id: str
    category: str
    prompt: str
    baseline_responses: list[str]
    current_response: Optional[str] = None
    cosine_similarities: list[float] = []


class HeatmapCell(BaseModel):
    category: str
    probe_id: str
    score: float


class AlertOut(BaseModel):
    id: int
    category: str
    score: float
    threshold: float
    message: str
    delivered: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DriftSimulateRequest(BaseModel):
    enable: bool = True
