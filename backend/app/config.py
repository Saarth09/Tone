from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Tone — Behavioral Drift Detector"
    database_url: str = "sqlite+aiosqlite:///./data/tone.db"
    chroma_path: str = "./data/chroma"

    # Target LLM (OpenAI-compatible)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3.2"
    llm_timeout_seconds: float = 60.0

    # Sampling
    sample_interval_minutes: int = 5
    baseline_runs: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Drift detection
    recent_window_size: int = 15
    mmd_weight: float = 0.4
    kl_weight: float = 0.3
    cosine_weight: float = 0.3
    drift_threshold: float = 0.40
    cosine_k: int = 5
    mmd_gamma: float | None = None  # None = median heuristic

    # Category-specific thresholds (override global if set)
    tone_threshold: float = 0.40
    fact_threshold: float = 0.35
    persona_threshold: float = 0.38

    # Alerting
    slack_webhook_url: str = ""
    alert_cooldown_minutes: int = 30

    # Demo mode generates synthetic drift without a live LLM
    demo_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
