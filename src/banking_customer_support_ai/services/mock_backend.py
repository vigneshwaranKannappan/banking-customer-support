"""In-memory backend with a keyword classifier. No LLM or database calls — for UI development."""

import re
import threading
import time

from banking_customer_support_ai.models.schemas import (
    AgentResult,
    Classification,
    Ticket,
    TicketAction,
    TicketStatus,
    TraceStep,
    utc_now,
)
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.security.guardrails import clean_name
from banking_customer_support_ai.services.backend import CLASSIFIER_NAME, HANDLER_NAMES
from banking_customer_support_ai.services.seed import SEED_TICKETS
from banking_customer_support_ai.utils.tickets import extract_ticket_id, generate_ticket_id

_POSITIVE_RE = re.compile(
    r"\b(thanks?|thank you|grateful|appreciate|great|excellent|awesome|helpful|resolved|sorted)\b",
    re.IGNORECASE,
)
_QUERY_RE = re.compile(r"\b(status|check|update on|track|where is|what is|how do)\b", re.IGNORECASE)


def _classify(message: str) -> tuple[Classification, float]:
    if extract_ticket_id(message) or _QUERY_RE.search(message):
        return Classification.QUERY, 0.9
    if _POSITIVE_RE.search(message) and not re.search(r"\b(not|never|still)\b", message, re.I):
        return Classification.POSITIVE_FEEDBACK, 0.85
    if message.rstrip().endswith("?"):
        return Classification.QUERY, 0.6
    return Classification.NEGATIVE_FEEDBACK, 0.75


class MockBackend:
    name = "Mock (keyword rules, in-memory)"
    uses_llm = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._tickets: dict[str, Ticket] = {
                tid: Ticket(ticket_id=tid, customer_name=name, message=msg, status=status)
                for tid, name, msg, status in SEED_TICKETS
            }
            self._logs: list[AgentResult] = []

    def handle_message(self, message: str, customer_name: str, session_id: str) -> AgentResult:
        started = time.perf_counter()
        name = clean_name(customer_name)

        classification, confidence = _classify(message)
        trace = [
            TraceStep(
                agent=CLASSIFIER_NAME,
                prompt=f"[mock keyword rules] classify: {message!r}",
                output=f"{classification.value} (confidence {confidence:.2f})",
            )
        ]
        handler = HANDLER_NAMES[classification.value]
        ticket_id: str | None = None
        action = TicketAction.NONE

        with self._lock:
            match classification:
                case Classification.POSITIVE_FEEDBACK:
                    response = templates.POSITIVE_REPLY.format(customer_name=name)
                case Classification.NEGATIVE_FEEDBACK:
                    ticket_id = generate_ticket_id(self._tickets)
                    self._tickets[ticket_id] = Ticket(
                        ticket_id=ticket_id, customer_name=name, message=message
                    )
                    action = TicketAction.CREATED
                    response = templates.NEGATIVE_REPLY.format(ticket_id=ticket_id)
                case Classification.QUERY:
                    ticket_id = extract_ticket_id(message)
                    if ticket_id is None:
                        action = TicketAction.MISSING_ID
                        response = templates.MISSING_TICKET_ID_REPLY
                    elif (ticket := self._tickets.get(ticket_id)) is None:
                        action = TicketAction.NOT_FOUND
                        response = templates.TICKET_NOT_FOUND_REPLY.format(ticket_id=ticket_id)
                    else:
                        action = TicketAction.LOOKED_UP
                        response = templates.STATUS_REPLY.format(
                            ticket_id=ticket_id, status=ticket.status.value
                        )

            trace.append(TraceStep(agent=handler, output=f"ticket_action={action.value}; {response}"))
            result = AgentResult(
                log_id=len(self._logs) + 1,
                session_id=session_id,
                customer_name=name,
                user_message=message,
                classification=classification,
                confidence=confidence,
                agent_path=[CLASSIFIER_NAME, handler],
                response=response,
                ticket_id=ticket_id,
                ticket_action=action,
                trace=trace,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self._logs.append(result)
        return result

    def list_tickets(self) -> list[Ticket]:
        with self._lock:
            return sorted(self._tickets.values(), key=lambda t: t.created_at, reverse=True)

    def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Ticket | None:
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is not None:
                ticket.status = status
                ticket.updated_at = utc_now()
            return ticket

    def list_logs(self, limit: int = 500) -> list[AgentResult]:
        with self._lock:
            return list(reversed(self._logs[-limit:]))

    def rate_response(self, log_id: int, rating: int) -> None:
        with self._lock:
            for log in self._logs:
                if log.log_id == log_id:
                    log.user_rating = rating
                    return
