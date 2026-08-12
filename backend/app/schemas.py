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


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    message: str = ""
    error: Optional[str] = None
    result: Optional[dict] = None
    created_at: str
    updated_at: str

    model_config = {"extra": "ignore"}


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
    system_prompt_poisoned: bool = False
    llm_connected: bool = False
    llm_connection_name: Optional[str] = None
    llm_provider: Optional[str] = None
    active_job: Optional[JobOut] = None


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


class SystemPromptRequest(BaseModel):
    system_prompt: str = ""


class LLMConnectionIn(BaseModel):
    name: str = "Production LLM"
    provider: str = Field(default="custom", description="openai|ollama|openrouter|custom")
    base_url: str
    api_key: str = ""
    model: str
    system_prompt: str = ""
    keep_existing_key: bool = False


class LLMConnectionOut(BaseModel):
    id: Optional[int] = None
    name: str
    provider: str
    base_url: str
    model: str
    system_prompt: str = ""
    api_key_masked: str = ""
    has_api_key: bool = False
    is_active: bool = False
    connected: bool = False
    last_tested_at: Optional[datetime] = None
    last_test_ok: Optional[bool] = None
    last_test_message: Optional[str] = None


class LLMConnectionTestIn(BaseModel):
    base_url: str
    api_key: str = ""
    model: str
    keep_existing_key: bool = False


class LLMConnectionTestOut(BaseModel):
    ok: bool
    message: str
    latency_ms: float = 0.0


class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: Optional[str] = None
    auth_provider: str = "password"

    model_config = {"from_attributes": True}


class ChatReviewIn(BaseModel):
    transcript: str = Field(..., min_length=20, description="Plain transcript or ChatGPT export JSON")
    goal: Optional[str] = Field(
        default=None,
        description="Optional original goal; defaults to the first user message",
    )
    threshold: float = Field(default=0.45, ge=0.05, le=0.95)
    use_llm_tips: bool = Field(
        default=True,
        description="If true and an LLM is connected, refine tips with that model",
    )


class ChatReviewPoint(BaseModel):
    turn_index: int
    window_start: int
    window_end: int
    label: str
    drift_score: float
    similarity: float
    is_alert: bool
    excerpt: str


class ChatReviewOut(BaseModel):
    goal: str
    message_count: int
    assistant_turns: int
    user_turns: int
    threshold: float
    overall_drift: float
    is_alert: bool
    peak: Optional[ChatReviewPoint] = None
    first_alert: Optional[ChatReviewPoint] = None
    timeline: list[ChatReviewPoint]
    tips: list[str]
    tips_source: str = "rules"


class ChatReviewGenerateTestIn(BaseModel):
    goal: str
    overall_drift: float = 0.0
    tips: list[str] = Field(default_factory=list)
    peak: Optional[ChatReviewPoint] = None
    first_alert: Optional[ChatReviewPoint] = None


class ChatReviewGenerateTestOut(BaseModel):
    code: str
    filename_hint: str = "test_tone_generated.py"


