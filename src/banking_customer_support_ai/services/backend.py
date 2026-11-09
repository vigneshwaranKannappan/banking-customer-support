"""Contract between the UI and whatever processes messages (keyword mock or LangGraph + Postgres)."""

import logging
from typing import Protocol

from banking_customer_support_ai.models.schemas import AgentResult, Ticket, TicketStatus

HANDLER_NAMES = {
    "positive_feedback": "Positive Feedback Handler",
    "negative_feedback": "Negative Feedback Handler",
    "query": "Query Handler",
}
CLASSIFIER_NAME = "Classifier"


class SupportBackend(Protocol):
    name: str
    uses_llm: bool

    def handle_message(self, message: str, customer_name: str, session_id: str) -> AgentResult: ...

    def list_tickets(self) -> list[Ticket]: ...

    def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Ticket | None: ...

    def list_logs(self, limit: int = 500) -> list[AgentResult]: ...

    def rate_response(self, log_id: int, rating: int) -> None: ...

    def reset(self) -> None: ...


def get_backend() -> SupportBackend:
    """Backend chosen by SUPPORT_BACKEND (auto | langgraph | mock)."""
    from banking_customer_support_ai.config.settings import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.resolved_backend == "mock":
        from banking_customer_support_ai.services.mock_backend import MockBackend

        return MockBackend()

    from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend

    return LangGraphBackend.from_settings(settings)
