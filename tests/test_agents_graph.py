"""Spec Part 1: classifier routing and each handler's behaviour, through the LangGraph workflow."""

import re

from banking_customer_support_ai.models.schemas import Classification, TicketAction, TicketStatus
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.services.backend import CLASSIFIER_NAME, HANDLER_NAMES
from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend

from .conftest import FakeLLM


def test_example_1_positive_feedback_is_llm_generated_and_personalised(backend, fake_llm):
    r = backend.handle_message("Thanks for sorting out my net banking login issue.", "Asha", "s")
    assert r.classification == Classification.POSITIVE_FEEDBACK
    assert r.agent_path == [CLASSIFIER_NAME, HANDLER_NAMES["positive_feedback"]]
    assert r.response.startswith("Thank you for your kind words, Asha! We're delighted to assist you.")
    assert fake_llm.calls == ["classifier", "reply"]
    assert r.success and r.error is None


def test_example_2_negative_feedback_creates_open_ticket(backend, repo):
    r = backend.handle_message("My debit card replacement still hasn't arrived.", "Asha", "s")
    assert r.agent_path[-1] == HANDLER_NAMES["negative_feedback"]
    assert r.ticket_action == TicketAction.CREATED
    assert re.fullmatch(r"\d{6}", r.ticket_id)
    assert r.response == templates.NEGATIVE_REPLY.format(ticket_id=r.ticket_id)
    ticket = repo.get_ticket(r.ticket_id)
    assert ticket.status == TicketStatus.OPEN
    assert ticket.customer_name == "Asha"


def test_example_3_query_returns_status_from_db(backend):
    r = backend.handle_message("Could you check the status of ticket 650932?", "Asha", "s")
    assert r.agent_path[-1] == HANDLER_NAMES["query"]
    assert r.response == "Your ticket #650932 is currently marked as: Resolved."
    assert r.ticket_action == TicketAction.LOOKED_UP


def test_query_reflects_status_updates(backend):
    created = backend.handle_message("I was charged twice for the same transaction.", "Asha", "s")
    backend.update_ticket_status(created.ticket_id, TicketStatus.IN_PROGRESS)
    r = backend.handle_message(f"Any update on {created.ticket_id}?", "Asha", "s")
    assert r.response == f"Your ticket #{created.ticket_id} is currently marked as: In Progress."


def test_query_unknown_and_missing_ticket(backend):
    assert backend.handle_message("Where is ticket 999999?", "A", "s").ticket_action == TicketAction.NOT_FOUND
    r = backend.handle_message("What is the status of my complaint?", "A", "s")
    assert r.ticket_action == TicketAction.MISSING_ID
    assert r.response == templates.MISSING_TICKET_ID_REPLY


def test_positive_reply_falls_back_to_template_when_llm_breaks_format(repo):
    llm = FakeLLM(reply="Hey! Cheers for that 🎉")
    be = LangGraphBackend(llm, repo)
    r = be.handle_message("Thanks for resolving my credit card issue.", "Asha", "s")
    assert r.response == templates.POSITIVE_REPLY.format(customer_name="Asha")
    assert r.success and "format check" in r.error


def test_positive_reply_falls_back_when_llm_errors(repo):
    be = LangGraphBackend(FakeLLM(fail_reply=True), repo)
    r = be.handle_message("Thanks for resolving my credit card issue.", "Asha", "s")
    assert r.response == templates.POSITIVE_REPLY.format(customer_name="Asha")
    assert r.success


def test_classifier_failure_is_logged_as_failure_without_routing(repo):
    be = LangGraphBackend(FakeLLM(fail_classifier=True), repo)
    r = be.handle_message("My card was blocked.", "Asha", "s")
    assert not r.success
    assert r.response == templates.ERROR_REPLY
    assert r.agent_path == [CLASSIFIER_NAME]
    assert "TimeoutError" in r.error
    assert repo.list_logs()[0].success is False
