"""Feedback Handler Agents: LLM thank-you for praise, ticket creation for complaints."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from banking_customer_support_ai.agents.base import HandlerOutcome, render_prompt
from banking_customer_support_ai.database.repository import SupportRepository
from banking_customer_support_ai.models.schemas import TicketAction
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.prompts.agent_prompts import POSITIVE_SYSTEM, POSITIVE_USER
from banking_customer_support_ai.services.backend import HANDLER_NAMES

logger = logging.getLogger(__name__)

MAX_REPLY_CHARS = 400


def _normalize_quotes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


class PositiveFeedbackAgent:
    name = HANDLER_NAMES["positive_feedback"]

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def run(self, message: str, customer_name: str) -> HandlerOutcome:
        fallback = templates.POSITIVE_REPLY.format(customer_name=customer_name)
        messages = [
            SystemMessage(POSITIVE_SYSTEM.format(customer_name=customer_name)),
            HumanMessage(POSITIVE_USER.format(message=message)),
        ]
        prompt = render_prompt(messages)
        try:
            reply = _normalize_quotes(self._llm.invoke(messages).text.strip())
        except Exception as exc:
            logger.warning("Positive feedback LLM call failed: %s", exc)
            return HandlerOutcome(fallback, prompt=prompt, warning=f"LLM call failed, used template: {exc}")

        # Guard the spec's required format; the LLM only adds the personalised follow-up sentence.
        required_opening = f"Thank you for your kind words, {customer_name}!"
        if not reply.startswith(required_opening) or len(reply) > MAX_REPLY_CHARS:
            return HandlerOutcome(fallback, prompt=prompt, warning=f"LLM reply failed format check: {reply!r}")
        return HandlerOutcome(reply, prompt=prompt)


class NegativeFeedbackAgent:
    name = HANDLER_NAMES["negative_feedback"]

    def __init__(self, repository: SupportRepository) -> None:
        self._repo = repository

    def run(self, message: str, customer_name: str) -> HandlerOutcome:
        ticket = self._repo.create_ticket(customer_name=customer_name, message=message)
        return HandlerOutcome(
            templates.NEGATIVE_REPLY.format(ticket_id=ticket.ticket_id),
            ticket_id=ticket.ticket_id,
            ticket_action=TicketAction.CREATED,
            prompt=f"INSERT INTO support_tickets (ticket_id, status) VALUES ('{ticket.ticket_id}', 'Open')",
        )
