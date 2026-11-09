"""Spec Part 2 §9: logs, prompt traces, ratings, persistence."""

from banking_customer_support_ai.models.schemas import Classification, TicketStatus
from banking_customer_support_ai.security.guardrails import MAX_MESSAGE_CHARS
from banking_customer_support_ai.services.langgraph_backend import LangGraphBackend
from banking_customer_support_ai.services.seed import SEED_TICKETS

from .conftest import FakeLLM


def test_every_interaction_is_logged_with_trace(backend):
    r = backend.handle_message("My debit card replacement still hasn't arrived.", "Asha", "sess-1")
    logs = backend.list_logs()
    assert [l.log_id for l in logs] == [r.log_id]
    log = logs[0]
    assert log.session_id == "sess-1"
    assert log.classification == Classification.NEGATIVE_FEEDBACK
    assert log.ticket_id == r.ticket_id
    assert [s.agent for s in log.trace] == r.agent_path
    assert "Classifier Agent" in log.trace[0].prompt


def test_rating_is_persisted(backend):
    r = backend.handle_message("Thanks for resolving my credit card issue.", "Asha", "s")
    backend.rate_response(r.log_id, 0)
    assert backend.list_logs()[0].user_rating == 0


def test_pii_is_masked_before_llm_db_and_logs(repo):
    seen: list[str] = []

    def classify(message: str) -> Classification:
        seen.append(message)
        return Classification.NEGATIVE_FEEDBACK

    be = LangGraphBackend(FakeLLM(classify=classify), repo)
    r = be.handle_message("Charged twice on card 4111 1111 1111 1111, mail me at a@b.com", "Asha", "s")
    for text in (seen[0], r.user_message, repo.get_ticket(r.ticket_id).message, repo.list_logs()[0].user_message):
        assert "4111" not in text and "a@b.com" not in text
        assert "[CARD]" in text and "[EMAIL]" in text
    assert r.trace[0].agent == "Input Guardrail"


def test_oversized_message_rejected_and_logged(backend, fake_llm):
    r = backend.handle_message("x" * (MAX_MESSAGE_CHARS + 1), "Asha", "s")
    assert not r.success and "too long" in r.response
    assert fake_llm.calls == []
    assert len(backend.list_logs()) == 1


def test_reset_restores_seed_data(backend):
    backend.handle_message("I was charged twice for the same transaction.", "Asha", "s")
    backend.update_ticket_status("650932", TicketStatus.OPEN)
    backend.reset()
    assert backend.list_logs() == []
    assert {t.ticket_id: t.status for t in backend.list_tickets()} == {tid: st for tid, _, _, st in SEED_TICKETS}
