"""Query Handler Agent: extracts a ticket number and reports its status from support_tickets."""

from banking_customer_support_ai.agents.base import HandlerOutcome
from banking_customer_support_ai.database.repository import SupportRepository
from banking_customer_support_ai.models.schemas import TicketAction
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.services.backend import HANDLER_NAMES
from banking_customer_support_ai.utils.tickets import extract_ticket_id


class QueryHandlerAgent:
    name = HANDLER_NAMES["query"]

    def __init__(self, repository: SupportRepository) -> None:
        self._repo = repository

    def run(self, message: str, customer_name: str) -> HandlerOutcome:
        ticket_id = extract_ticket_id(message)
        if ticket_id is None:
            return HandlerOutcome(templates.MISSING_TICKET_ID_REPLY, ticket_action=TicketAction.MISSING_ID)

        sql = f"SELECT status FROM support_tickets WHERE ticket_id = '{ticket_id}'"
        ticket = self._repo.get_ticket(ticket_id)
        if ticket is None:
            return HandlerOutcome(
                templates.TICKET_NOT_FOUND_REPLY.format(ticket_id=ticket_id),
                ticket_id=ticket_id,
                ticket_action=TicketAction.NOT_FOUND,
                prompt=sql,
            )
        return HandlerOutcome(
            templates.STATUS_REPLY.format(ticket_id=ticket_id, status=ticket.status.value),
            ticket_id=ticket_id,
            ticket_action=TicketAction.LOOKED_UP,
            prompt=sql,
        )
