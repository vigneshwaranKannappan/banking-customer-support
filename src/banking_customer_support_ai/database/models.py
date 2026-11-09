"""SQLAlchemy tables: support_tickets (spec) and interaction_logs (LLMOps logging)."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from banking_customer_support_ai.models.schemas import TicketAction, TicketStatus, utc_now


class Base(DeclarativeBase):
    pass


class SupportTicketRow(Base):
    __tablename__ = "support_tickets"

    ticket_id: Mapped[str] = mapped_column(String(6), primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default=TicketStatus.OPEN.value, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class InteractionLogRow(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    user_message: Mapped[str] = mapped_column(Text)
    classification: Mapped[str | None] = mapped_column(String(32), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    agent_path: Mapped[list] = mapped_column(JSON, default=list)
    response: Mapped[str] = mapped_column(Text)
    ticket_id: Mapped[str | None] = mapped_column(String(6), index=True)
    ticket_action: Mapped[str] = mapped_column(String(20), default=TicketAction.NONE.value)
    trace: Mapped[list] = mapped_column(JSON, default=list)
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    user_rating: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
