"""Optional LLM-as-judge scoring of reply quality (empathy, clarity, accuracy). One LLM call per reply."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

JUDGE_SYSTEM = """\
You grade replies written by a bank's customer support assistant. Score each from 1 (poor) to 5 (excellent):
- empathy: warmth and acknowledgement of the customer's feelings, appropriate to the situation.
- clarity: the reply is easy to understand; ticket numbers and statuses are stated plainly.
- accuracy: the reply fits the customer's message and the stated category, with no invented facts.
Be strict. Fixed templated wording is acceptable if it fits the situation.
"""

JUDGE_USER = "Category: {category}\nCustomer message: {message}\nAssistant reply: {reply}"


class JudgeScore(BaseModel):
    empathy: int = Field(description="1-5")
    clarity: int = Field(description="1-5")
    accuracy: int = Field(description="1-5")
    rationale: str = Field(description="One short sentence.")


class ResponseJudge:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = llm.with_structured_output(JudgeScore)

    def score(self, category: str, message: str, reply: str) -> JudgeScore:
        out = self._chain.invoke(
            [SystemMessage(JUDGE_SYSTEM), HumanMessage(JUDGE_USER.format(category=category, message=message, reply=reply))]
        )
        for field in ("empathy", "clarity", "accuracy"):
            setattr(out, field, min(max(getattr(out, field), 1), 5))
        return out
