"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["openai", "gemini"] = "openai"
    llm_timeout_s: float = 60.0  # free tiers are slow under load; 30 s produced 504 DEADLINE_EXCEEDED
    # Client-side throttle, e.g. 8 to stay under a free tier's per-minute limit. Unset = no limit.
    llm_max_requests_per_minute: float | None = None

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"

    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/banking_support"

    # auto = LangGraph when the selected provider's API key is set, otherwise the keyword mock.
    support_backend: Literal["auto", "langgraph", "mock"] = "auto"

    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def llm_api_key(self) -> SecretStr | None:
        return self.google_api_key if self.llm_provider == "gemini" else self.openai_api_key

    @property
    def llm_model(self) -> str:
        return self.gemini_model if self.llm_provider == "gemini" else self.openai_model

    @property
    def resolved_backend(self) -> Literal["langgraph", "mock"]:
        if self.support_backend == "auto":
            return "langgraph" if self.llm_api_key else "mock"
        return self.support_backend


@lru_cache
def get_settings() -> Settings:
    return Settings()
