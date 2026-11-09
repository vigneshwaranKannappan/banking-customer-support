from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Classification(StrEnum):
    POSITIVE_FEEDBACK = "positive_feedback"
    NEGATIVE_FEEDBACK = "negative_feedback"
    QUERY = "query"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class TicketStatus(StrEnum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class TicketAction(StrEnum):
    CREATED = "created"
    LOOKED_UP = "looked_up"
    NOT_FOUND = "not_found"
    MISSING_ID = "missing_id"
    NONE = "none"


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    message: str
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TraceStep(BaseModel):
    """One agent hop: what it was asked and what it produced."""

    agent: str
    prompt: str | None = None
    output: str
    latency_ms: float = 0.0


class AgentResult(BaseModel):
    """Outcome of processing one customer message through the agent graph."""

    log_id: int
    session_id: str
    customer_name: str
    user_message: str
    classification: Classification | None = None
    confidence: float | None = None
    agent_path: list[str] = Field(default_factory=list)
    response: str
    ticket_id: str | None = None
    ticket_action: TicketAction = TicketAction.NONE
    trace: list[TraceStep] = Field(default_factory=list)
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    user_rating: int | None = None  # 1 = helpful, 0 = not helpful
    created_at: datetime = Field(default_factory=utc_now)
