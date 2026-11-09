"""Provider switch: settings pick the right key/model and make_llm builds the right client (no network)."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from banking_customer_support_ai.config.settings import Settings
from banking_customer_support_ai.services.langgraph_backend import make_llm


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_gemini_provider():
    s = _settings(llm_provider="gemini", google_api_key="g-key", llm_max_requests_per_minute=8)
    assert s.resolved_backend == "langgraph"
    assert s.llm_model == "gemini-3.5-flash"
    llm = make_llm(s)
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.rate_limiter is not None


def test_openai_provider():
    llm = make_llm(_settings(llm_provider="openai", openai_api_key="sk-test"))
    assert isinstance(llm, ChatOpenAI)
    assert llm.rate_limiter is None


def test_auto_falls_back_to_mock_without_the_selected_providers_key():
    assert _settings(llm_provider="gemini", openai_api_key="sk-test").resolved_backend == "mock"
