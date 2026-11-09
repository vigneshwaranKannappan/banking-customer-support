"""Production backend: guardrails → LangGraph multi-agent workflow → PostgreSQL tickets and logs."""

import logging
import time

from langchain_core.language_models import BaseChatModel

from banking_customer_support_ai.agents.classifier import ClassifierAgent
from banking_customer_support_ai.agents.feedback import NegativeFeedbackAgent, PositiveFeedbackAgent
from banking_customer_support_ai.agents.query import QueryHandlerAgent
from banking_customer_support_ai.config.settings import Settings
from banking_customer_support_ai.database.repository import SupportRepository
from banking_customer_support_ai.database.session import init_db, make_engine
from banking_customer_support_ai.graph.workflow import build_graph
from banking_customer_support_ai.models.schemas import AgentResult, Ticket, TicketAction, TicketStatus, TraceStep
from banking_customer_support_ai.prompts import templates
from banking_customer_support_ai.security.guardrails import (
    InputRejected,
    clean_message,
    clean_name,
    looks_like_injection,
    mask_pii,
)
from banking_customer_support_ai.services.seed import SEED_TICKETS

logger = logging.getLogger(__name__)

GUARDRAIL_NAME = "Input Guardrail"


def make_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_api_key is None:
        key = "GOOGLE_API_KEY" if settings.llm_provider == "gemini" else "OPENAI_API_KEY"
        raise RuntimeError(f"{key} is not set; use SUPPORT_BACKEND=mock or add the key to .env")

    rate_limiter = None
    if settings.llm_max_requests_per_minute:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.llm_max_requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=2,  # one chat message (classifier + reply) never waits
        )

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.google_api_key,
            request_timeout=settings.llm_timeout_s,
            retries=2,
            rate_limiter=rate_limiter,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        reasoning_effort=settings.openai_reasoning_effort,
        timeout=settings.llm_timeout_s,
        max_retries=2,
        rate_limiter=rate_limiter,
    )


class LangGraphBackend:
    uses_llm = True

    def __init__(self, llm: BaseChatModel, repository: SupportRepository, model_label: str = "LLM") -> None:
        self.name = f"LangGraph multi-agent ({model_label})"
        self._repo = repository
        self._graph = build_graph(
            ClassifierAgent(llm),
            PositiveFeedbackAgent(llm),
            NegativeFeedbackAgent(repository),
            QueryHandlerAgent(repository),
        )
        repository.seed_tickets(SEED_TICKETS)

    @classmethod
    def from_settings(cls, settings: Settings, database_url: str | None = None) -> "LangGraphBackend":
        engine = make_engine(database_url or settings.database_url)
        init_db(engine)
        return cls(make_llm(settings), SupportRepository(engine), model_label=settings.llm_model)

    def handle_message(self, message: str, customer_name: str, session_id: str) -> AgentResult:
        started = time.perf_counter()
        name = clean_name(customer_name)
        try:
            text = clean_message(message)
        except InputRejected as exc:
            result = AgentResult(
                log_id=0,
                session_id=session_id,
                customer_name=name,
                user_message=mask_pii(message or "")[:200],
                agent_path=[GUARDRAIL_NAME],
                response=str(exc),
                trace=[TraceStep(agent=GUARDRAIL_NAME, output=f"Rejected: {exc}")],
                success=False,
                error=f"input rejected: {exc}",
            )
        else:
            result = self._run_graph(text, name, session_id)
        result.latency_ms = (time.perf_counter() - started) * 1000

        result.log_id = self._repo.add_log(result)
        logger.info(
            "log=%s class=%s path=%s ticket=%s/%s success=%s %.0fms",
            result.log_id, result.classification, " > ".join(result.agent_path),
            result.ticket_id, result.ticket_action.value, result.success, result.latency_ms,
        )
        return result

    def _run_graph(self, text: str, name: str, session_id: str) -> AgentResult:
        # PII never reaches the LLM, the tickets table or the logs.
        masked = mask_pii(text)
        pre_trace: list[TraceStep] = []
        notes = []
        if masked != text:
            notes.append("masked PII (card/account/phone/email) before processing")
        if looks_like_injection(text):
            notes.append("possible prompt-injection phrasing detected; message treated as data only")
        if notes:
            pre_trace.append(TraceStep(agent=GUARDRAIL_NAME, output="; ".join(notes)))

        try:
            state = self._graph.invoke(
                {"message": masked, "customer_name": name, "agent_path": [], "trace": pre_trace}
            )
        except Exception as exc:  # graph-level failure (node errors are captured inside the graph)
            logger.exception("Graph invocation failed")
            state = {"error": f"workflow failed: {type(exc).__name__}: {exc}", "agent_path": [], "trace": pre_trace}

        error = state.get("error")
        return AgentResult(
            log_id=0,
            session_id=session_id,
            customer_name=name,
            user_message=masked,
            classification=state.get("classification"),
            confidence=state.get("confidence"),
            agent_path=state.get("agent_path", []),
            response=templates.ERROR_REPLY if error else state["response"],
            ticket_id=state.get("ticket_id"),
            ticket_action=state.get("ticket_action", TicketAction.NONE),
            trace=state.get("trace", []),
            success=error is None,
            error=error or state.get("warning"),
        )

    def list_tickets(self) -> list[Ticket]:
        return self._repo.list_tickets()

    def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Ticket | None:
        return self._repo.update_ticket_status(ticket_id, status)

    def list_logs(self, limit: int = 500) -> list[AgentResult]:
        return self._repo.list_logs(limit)

    def rate_response(self, log_id: int, rating: int) -> None:
        self._repo.rate_log(log_id, rating)

    def reset(self) -> None:
        self._repo.clear()
        self._repo.seed_tickets(SEED_TICKETS)
