"""Data access for tickets and interaction logs. Returns domain models, never ORM rows."""

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from banking_customer_support_ai.database.models import InteractionLogRow, SupportTicketRow
from banking_customer_support_ai.models.schemas import AgentResult, Ticket, TicketStatus, utc_now
from banking_customer_support_ai.utils.tickets import generate_ticket_id


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo; everything is stored as UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _to_ticket(row: SupportTicketRow) -> Ticket:
    return Ticket(
        ticket_id=row.ticket_id,
        customer_name=row.customer_name,
        message=row.message,
        status=TicketStatus(row.status),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
    )


def _to_result(row: InteractionLogRow) -> AgentResult:
    return AgentResult(
        log_id=row.id,
        session_id=row.session_id,
        customer_name=row.customer_name,
        user_message=row.user_message,
        classification=row.classification,
        confidence=row.confidence,
        agent_path=row.agent_path or [],
        response=row.response,
        ticket_id=row.ticket_id,
        ticket_action=row.ticket_action,
        trace=row.trace or [],
        success=row.success,
        error=row.error,
        latency_ms=row.latency_ms,
        user_rating=row.user_rating,
        created_at=_aware(row.created_at),
    )


class _ExistingTicketIds:
    """Container view over support_tickets so generate_ticket_id can check uniqueness."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def __contains__(self, ticket_id: object) -> bool:
        return self._session.get(SupportTicketRow, ticket_id) is not None


class SupportRepository:
    def __init__(self, engine: Engine) -> None:
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    # --- support_tickets ---

    def create_ticket(self, customer_name: str, message: str) -> Ticket:
        with self._sessions.begin() as s:
            row = SupportTicketRow(
                ticket_id=generate_ticket_id(_ExistingTicketIds(s)),
                customer_name=customer_name,
                message=message,
                status=TicketStatus.OPEN.value,
            )
            s.add(row)
            s.flush()
            return _to_ticket(row)

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        with self._sessions() as s:
            row = s.get(SupportTicketRow, ticket_id)
            return _to_ticket(row) if row else None

    def list_tickets(self) -> list[Ticket]:
        with self._sessions() as s:
            rows = s.scalars(select(SupportTicketRow).order_by(SupportTicketRow.created_at.desc()))
            return [_to_ticket(r) for r in rows]

    def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Ticket | None:
        with self._sessions.begin() as s:
            row = s.get(SupportTicketRow, ticket_id)
            if row is None:
                return None
            row.status = status.value
            row.updated_at = utc_now()
            s.flush()
            return _to_ticket(row)

    def seed_tickets(self, seeds: Iterable[tuple[str, str, str, TicketStatus]]) -> None:
        """Insert demo tickets, only into an empty table."""
        with self._sessions.begin() as s:
            if s.scalar(select(func.count()).select_from(SupportTicketRow)):
                return
            s.add_all(
                SupportTicketRow(ticket_id=tid, customer_name=name, message=msg, status=status.value)
                for tid, name, msg, status in seeds
            )

    # --- interaction_logs ---

    def add_log(self, result: AgentResult) -> int:
        data = result.model_dump(mode="json", exclude={"log_id", "created_at"})
        with self._sessions.begin() as s:
            row = InteractionLogRow(**data, created_at=result.created_at)
            s.add(row)
            s.flush()
            return row.id

    def list_logs(self, limit: int = 500) -> list[AgentResult]:
        with self._sessions() as s:
            rows = s.scalars(select(InteractionLogRow).order_by(InteractionLogRow.id.desc()).limit(limit))
            return [_to_result(r) for r in rows]

    def rate_log(self, log_id: int, rating: int) -> None:
        with self._sessions.begin() as s:
            if (row := s.get(InteractionLogRow, log_id)) is not None:
                row.user_rating = rating

    def clear(self) -> None:
        with self._sessions.begin() as s:
            s.execute(delete(InteractionLogRow))
            s.execute(delete(SupportTicketRow))
