# Banking Customer Support AI Agent

Multi-agent GenAI assistant for banking support (Simplilearn Applied GenAI capstone).
It classifies a customer's message, routes it to the right agent, writes tickets to PostgreSQL,
and logs every interaction for evaluation and debugging.

```
                    ┌──────────────────────┐
 message ─► Input ─►│   Classifier Agent   │  LLM, structured output
          Guardrail └──────────┬───────────┘
        (validate, mask PII)   │ conditional edge (LangGraph)
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
 Positive Feedback     Negative Feedback        Query Handler
 LLM thank-you reply   new 6-digit ticket →     extract ticket # →
 (format-checked)      support_tickets          look up status
          └────────────────────┴─────────────────────┘
                               ▼
                 interaction_logs (trace, ticket action, success, rating)
```

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14. PostgreSQL is optional (see backends below).

```powershell
uv sync
copy .env.example .env      # then set LLM_PROVIDER, its API key, and DATABASE_URL
createdb -U postgres banking_support   # tables and demo tickets are created on first start
uv run banking-customer-support-ai     # opens http://localhost:8501
```

| `SUPPORT_BACKEND` | What runs | Costs |
|---|---|---|
| `auto` (default) | `langgraph` if the provider's API key is set, else `mock` | — |
| `langgraph` | Real agents: Gemini or OpenAI (`LLM_PROVIDER`) + PostgreSQL | 2 LLM calls per positive-feedback message, 1 otherwise |
| `mock` | Keyword classifier, in-memory data | none (good for UI work) |

Demo tickets are seeded on first start: `650932` Resolved, `123456` Open, `784521` In Progress.

## How the spec maps to the code

| Spec item | Implementation |
|---|---|
| 1. Classifier Agent | [agents/classifier.py](src/banking_customer_support_ai/agents/classifier.py): LLM with a Pydantic structured output (label, confidence, reasoning) |
| Routing | [graph/workflow.py](src/banking_customer_support_ai/graph/workflow.py): LangGraph `StateGraph` with a conditional edge per class |
| 2. Feedback Handler (positive) | [agents/feedback.py](src/banking_customer_support_ai/agents/feedback.py): LLM writes a personalised reply that must start with the required sentence, otherwise it falls back to the template |
| 2. Feedback Handler (negative) | Same file: unique 6-digit ID → `support_tickets` (status Open) → apology reply |
| 3. Query Handler | [agents/query.py](src/banking_customer_support_ai/agents/query.py): regex extracts the ticket number → DB lookup → status reply (also covers unknown or missing ticket) |
| Support database | [database/](src/banking_customer_support_ai/database/): SQLAlchemy tables `support_tickets` and `interaction_logs` |
| 7. Model evaluation | [evaluation/](src/banking_customer_support_ai/evaluation/): rule-based QA checks (classification, routing, reply format, DB consistency) plus an optional LLM judge (empathy, clarity, accuracy, scored 1-5) |
| 8. Streamlit UI | [ui/app.py](src/banking_customer_support_ai/ui/app.py): Chat, Scenarios (accuracy, routing, QA), Tickets (view and update status), Logs, Metrics |
| 9. Logs and debugging | Every message is stored with its prompt trace, per-agent latency, ticket action, success flag and error. The Logs tab shows the trace of any row |
| 9. Feedback loop (optional) | 👍/👎 on each reply is stored. Metrics → *Needs review* lists 👎-rated and failed interactions to add as scenarios |
| Security | [security/guardrails.py](src/banking_customer_support_ai/security/guardrails.py): length and control-character checks, masking of card, phone, account and email before the LLM, DB or logs see them, and prompt-injection flagging |

## Evaluation

```powershell
uv run banking-eval                                     # mock backend, no LLM calls
uv run banking-eval --backend llm --limit 3             # real agents, small run
uv run banking-eval --backend llm --judge --output eval.json
```

LLM runs use a throwaway in-memory SQLite DB, so they never write to PostgreSQL.
They cost about 2 LLM calls per scenario, plus 1 more per scenario with `--judge`. There are 10 scenarios.

## Tests

```powershell
uv run pytest
```

The tests use a fake chat model and in-memory SQLite, so they make no OpenAI calls and need no database.
