"""LangGraph workflow: Classifier → (Positive Feedback | Negative Feedback | Query) Handler."""

import logging
import operator
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from banking_customer_support_ai.agents.base import HandlerAgent
from banking_customer_support_ai.agents.classifier import ClassifierAgent
from banking_customer_support_ai.models.schemas import Classification, TicketAction, TraceStep

logger = logging.getLogger(__name__)


class SupportState(TypedDict, total=False):
    message: str  # already validated and PII-masked
    customer_name: str
    classification: Classification
    confidence: float
    response: str
    ticket_id: str | None
    ticket_action: TicketAction
    warning: str | None
    error: str | None
    agent_path: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceStep], operator.add]


Node = Callable[[SupportState], dict]


def _guarded(agent_name: str, node: Node) -> Node:
    """Turn an agent exception into an error entry in state instead of crashing the graph."""

    def wrapper(state: SupportState) -> dict:
        started = time.perf_counter()
        try:
            return node(state)
        except Exception as exc:
            logger.exception("%s failed", agent_name)
            return {
                "error": f"{agent_name} failed: {type(exc).__name__}: {exc}",
                "agent_path": [agent_name],
                "trace": [
                    TraceStep(
                        agent=agent_name,
                        output=f"FAILED: {type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                ],
            }

    return wrapper


def _classifier_node(agent: ClassifierAgent) -> Node:
    def node(state: SupportState) -> dict:
        started = time.perf_counter()
        output, prompt = agent.run(state["message"])
        return {
            "classification": output.classification,
            "confidence": output.confidence,
            "agent_path": [agent.name],
            "trace": [
                TraceStep(
                    agent=agent.name,
                    prompt=prompt,
                    output=f"{output.classification.value} (confidence {output.confidence:.2f}): {output.reasoning}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            ],
        }

    return node


def _handler_node(agent: HandlerAgent) -> Node:
    def node(state: SupportState) -> dict:
        started = time.perf_counter()
        outcome = agent.run(state["message"], state["customer_name"])
        output = f"ticket_action={outcome.ticket_action.value}; {outcome.response}"
        if outcome.warning:
            output += f"\nWARNING: {outcome.warning}"
        return {
            "response": outcome.response,
            "ticket_id": outcome.ticket_id,
            "ticket_action": outcome.ticket_action,
            "warning": outcome.warning,
            "agent_path": [agent.name],
            "trace": [
                TraceStep(
                    agent=agent.name,
                    prompt=outcome.prompt,
                    output=output,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            ],
        }

    return node


def _route(state: SupportState) -> str:
    if state.get("error"):
        return END
    return state["classification"].value


def build_graph(
    classifier: ClassifierAgent,
    positive: HandlerAgent,
    negative: HandlerAgent,
    query: HandlerAgent,
) -> CompiledStateGraph:
    handlers = {
        Classification.POSITIVE_FEEDBACK.value: positive,
        Classification.NEGATIVE_FEEDBACK.value: negative,
        Classification.QUERY.value: query,
    }
    graph = StateGraph(SupportState)
    graph.add_node("classifier", _guarded(classifier.name, _classifier_node(classifier)))
    for key, agent in handlers.items():
        graph.add_node(key, _guarded(agent.name, _handler_node(agent)))
        graph.add_edge(key, END)
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", _route, {**{k: k for k in handlers}, END: END})
    return graph.compile()
