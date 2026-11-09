"""Rule-based QA checks for one agent result: classification, routing, reply format and DB consistency."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from banking_customer_support_ai.models.schemas import AgentResult, Classification, Ticket, TicketAction, TicketStatus
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.services.backend import HANDLER_NAMES
from banking_customer_support_ai.utils.tickets import extract_ticket_id

_NEGATIVE_RE = re.compile(re.escape(templates.NEGATIVE_REPLY).replace(r"\{ticket_id\}", r"(\d{6})"))


@dataclass(frozen=True)
class QACheck:
    name: str
    passed: bool
    detail: str = ""


def _norm(text: str) -> str:
    return text.replace("’", "'")


def qa_checks(
    result: AgentResult, expected: Classification, lookup_ticket: Callable[[str], Ticket | None]
) -> list[QACheck]:
    predicted = result.classification
    expected_handler = HANDLER_NAMES[expected.value]
    checks = [
        QACheck("classification", predicted == expected, f"expected {expected.value}, got {predicted}"),
        QACheck(
            "routing",
            bool(result.agent_path) and result.agent_path[-1] == expected_handler,
            f"expected → {expected_handler}, got {' → '.join(result.agent_path) or 'nothing'}",
        ),
        QACheck("agent_success", result.success, result.error or ""),
    ]
    if predicted != expected or not result.success:
        return checks  # reply checks only make sense on the intended path

    response = _norm(result.response)
    match expected:
        case Classification.POSITIVE_FEEDBACK:
            opening = f"Thank you for your kind words, {result.customer_name}!"
            checks.append(QACheck("reply_format", response.startswith(opening), f"must start with {opening!r}"))
        case Classification.NEGATIVE_FEEDBACK:
            m = _NEGATIVE_RE.fullmatch(response)
            checks.append(QACheck("reply_format", m is not None, "matches negative reply template"))
            ticket = lookup_ticket(m.group(1)) if m else None
            checks.append(
                QACheck(
                    "ticket_persisted",
                    ticket is not None and ticket.status == TicketStatus.OPEN and result.ticket_id == ticket.ticket_id,
                    "new Open ticket stored in support_tickets with the quoted ID",
                )
            )
        case Classification.QUERY:
            ticket_id = extract_ticket_id(result.user_message)
            if ticket_id is None:
                ok = result.ticket_action == TicketAction.MISSING_ID and response == templates.MISSING_TICKET_ID_REPLY
                checks.append(QACheck("status_reply", ok, "asks for the ticket number"))
            elif (ticket := lookup_ticket(ticket_id)) is None:
                ok = response == templates.TICKET_NOT_FOUND_REPLY.format(ticket_id=ticket_id)
                checks.append(QACheck("status_reply", ok, f"reports #{ticket_id} not found"))
            else:
                want = templates.STATUS_REPLY.format(ticket_id=ticket_id, status=ticket.status.value)
                checks.append(QACheck("status_reply", response == want, f"expected {want!r}"))
    return checks
