from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage

from banking_customer_support_ai.models.schemas import TicketAction


@dataclass
class HandlerOutcome:
    response: str
    ticket_id: str | None = None
    ticket_action: TicketAction = TicketAction.NONE
    prompt: str | None = None
    # Set when the handler recovered from a problem (e.g. LLM fell back to the template).
    warning: str | None = None


class HandlerAgent(Protocol):
    name: str

    def run(self, message: str, customer_name: str) -> HandlerOutcome: ...


def render_prompt(messages: list[BaseMessage]) -> str:
    """Readable prompt trace for the logs view."""
    return "\n\n".join(f"[{m.type}]\n{m.text}" for m in messages)
