"""Classifier Agent: labels a message as positive feedback, negative feedback or query."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from banking_customer_support_ai.agents.base import render_prompt
from banking_customer_support_ai.models.schemas import Classification
from banking_customer_support_ai.prompts.agent_prompts import CLASSIFIER_SYSTEM, CLASSIFIER_USER
from banking_customer_support_ai.services.backend import CLASSIFIER_NAME


class ClassifierOutput(BaseModel):
    classification: Classification
    confidence: float = Field(description="Probability 0.0-1.0 that the label is correct.")
    reasoning: str = Field(description="One short sentence explaining the label.")


class ClassifierAgent:
    name = CLASSIFIER_NAME

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = llm.with_structured_output(ClassifierOutput)

    def run(self, message: str) -> tuple[ClassifierOutput, str]:
        messages = [SystemMessage(CLASSIFIER_SYSTEM), HumanMessage(CLASSIFIER_USER.format(message=message))]
        output = self._chain.invoke(messages)
        output.confidence = min(max(output.confidence, 0.0), 1.0)
        return output, render_prompt(messages)
