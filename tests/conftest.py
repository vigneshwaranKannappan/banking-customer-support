"""Shared fixtures. Tests never call OpenAI: agents get a fake chat model, the DB is in-memory SQLite."""

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from banking_customer_support_ai.agents.classifier import ClassifierOutput
from banking_customer_support_ai.database.repository import SupportRepository
from banking_customer_support_ai.database.session import init_db, make_engine
from banking_customer_support_ai.evaluation.scenarios import SCENARIOS
from banking_customer_support_ai.models.schemas import Classification
from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend

# Scenario labels double as the fake classifier's answer key.
LABELS = {s.message: s.expected for s in SCENARIOS}


class FakeLLM:
    """Duck-typed stand-in for a LangChain chat model."""

    def __init__(
        self,
        classify: Callable[[str], Classification] | None = None,
        reply: Callable[[str], str] | str = "",
        fail_classifier: bool = False,
        fail_reply: bool = False,
    ) -> None:
        self.classify = classify or (lambda msg: LABELS.get(msg, Classification.QUERY))
        self.reply = reply
        self.fail_classifier = fail_classifier
        self.fail_reply = fail_reply
        self.calls: list[str] = []

    @staticmethod
    def _customer_message(messages) -> str:
        text = messages[-1].text
        return text.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]

    def with_structured_output(self, schema):
        fake = self

        class _Structured:
            def invoke(self, messages):
                fake.calls.append("classifier")
                if fake.fail_classifier:
                    raise TimeoutError("LLM timed out")
                return ClassifierOutput(
                    classification=fake.classify(fake._customer_message(messages)),
                    confidence=0.93,
                    reasoning="fake",
                )

        assert schema is ClassifierOutput
        return _Structured()

    def invoke(self, messages):
        self.calls.append("reply")
        if self.fail_reply:
            raise ConnectionError("LLM unavailable")
        system = messages[0].text
        name = system.split("Thank you for your kind words, ", 1)[1].split("!", 1)[0]
        text = self.reply(name) if callable(self.reply) else self.reply
        return AIMessage(content=text)


def default_reply(name: str) -> str:
    return f"Thank you for your kind words, {name}! We’re delighted to assist you. Glad your issue is sorted."


@pytest.fixture
def repo() -> SupportRepository:
    engine = make_engine("sqlite://")
    init_db(engine)
    return SupportRepository(engine)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(reply=default_reply)


@pytest.fixture
def backend(fake_llm, repo) -> LangGraphBackend:
    return LangGraphBackend(fake_llm, repo, model_label="fake")
