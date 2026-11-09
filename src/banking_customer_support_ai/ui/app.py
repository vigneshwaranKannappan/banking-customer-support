"""Streamlit dashboard: chat, scenario tester, tickets, logs and metrics.

Run: uv run banking-customer-support-ai
"""

import uuid

import pandas as pd
import streamlit as st
from sqlalchemy.exc import OperationalError

from banking_customer_support_ai.evaluation.qa import qa_checks
from banking_customer_support_ai.evaluation.scenarios import SCENARIOS
from banking_customer_support_ai.models.schemas import AgentResult, Classification, TicketStatus
from banking_customer_support_ai.services.backend import SupportBackend, get_backend

CLASS_BADGES = {
    Classification.POSITIVE_FEEDBACK: ":green-badge[Positive feedback]",
    Classification.NEGATIVE_FEEDBACK: ":red-badge[Negative feedback]",
    Classification.QUERY: ":blue-badge[Query]",
}


@st.cache_resource
def _cached_backend() -> SupportBackend:
    return get_backend()


def backend() -> SupportBackend:
    try:
        return _cached_backend()
    except OperationalError as exc:
        st.error(
            "Cannot connect to the database. Check `DATABASE_URL` in `.env` (user, password, database name), "
            "or set `SUPPORT_BACKEND=mock` to run without a database, then restart the app.",
            icon=":material/database:",
        )
        st.code(str(exc.orig).strip(), language=None, wrap_lines=True)
        st.stop()


def init_session() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
    st.session_state.setdefault("chat", [])  # list[AgentResult] for this browser session


def badge(result: AgentResult) -> str:
    return CLASS_BADGES.get(result.classification, ":gray-badge[Unclassified]")


def render_details(result: AgentResult) -> None:
    cols = st.columns(4)
    cols[0].markdown(f"**Class**  \n{badge(result)}")
    cols[1].markdown(f"**Confidence**  \n{result.confidence:.0%}" if result.confidence else "**Confidence**  \n—")
    cols[2].markdown(f"**Ticket**  \n{('#' + result.ticket_id) if result.ticket_id else '—'} ({result.ticket_action.value})")
    cols[3].markdown(f"**Latency**  \n{result.latency_ms:.0f} ms")
    st.markdown("**Agent path:** " + " → ".join(result.agent_path))
    for step in result.trace:
        with st.container(border=True):
            st.markdown(f"**{step.agent}**")
            if step.prompt:
                st.code(step.prompt, language=None, wrap_lines=True)
            st.text(step.output)
    if result.error:
        st.error(result.error)


def rate(log_id: int) -> None:
    value = st.session_state.get(f"fb_{log_id}")
    if value is not None:
        backend().rate_response(log_id, value)


def sidebar() -> str:
    with st.sidebar:
        st.header("Customer")
        name = st.text_input("Customer name", value="Vignesh", help="Used to personalise replies.")
        st.caption(f"Session `{st.session_state.session_id}`")
        st.divider()
        st.header("Backend")
        be = backend()
        st.markdown(f"**{be.name}**")
        if be.uses_llm:
            st.warning("Live LLM — each message uses API quota.", icon=":material/paid:")
        else:
            st.info("No LLM or DB calls.", icon=":material/science:")
        if st.button("Reset demo data", icon=":material/restart_alt:", width="stretch"):
            be.reset()
            st.session_state.chat = []
            st.rerun()
    return name


def chat_tab(customer_name: str) -> None:
    history = st.container()
    prompt = st.chat_input("Type a message, e.g. 'Could you check the status of ticket 650932?'")
    if prompt:
        result = backend().handle_message(prompt, customer_name, st.session_state.session_id)
        st.session_state.chat.append(result)

    with history:
        if not st.session_state.chat:
            st.caption("No messages yet. Try praise, a complaint, or a ticket status question.")
        for result in st.session_state.chat:
            with st.chat_message("user"):
                st.write(result.user_message)
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                st.markdown(f"{badge(result)} {' → '.join(result.agent_path)}")
                st.write(result.response)
                with st.expander("Agent details & trace"):
                    render_details(result)
                st.feedback("thumbs", key=f"fb_{result.log_id}", on_change=rate, args=(result.log_id,))


def scenarios_tab(customer_name: str) -> None:
    be = backend()
    st.markdown("Run labeled messages through the agents and compare classification against the expected label.")
    if be.uses_llm:
        st.warning(f"Running all scenarios makes up to {len(SCENARIOS) * 2} LLM calls.")

    left, right = st.columns([3, 1])
    options = [f"{s.expected.label}: {s.message}" for s in SCENARIOS]
    picked = left.selectbox("Single scenario", range(len(SCENARIOS)), format_func=lambda i: options[i])
    run_one = right.button("Run selected", width="stretch")
    run_all = right.button("Run all", type="primary", width="stretch")

    to_run = list(range(len(SCENARIOS))) if run_all else [picked] if run_one else []
    if to_run:
        rows = []
        for i in to_run:
            sc = SCENARIOS[i]
            r = be.handle_message(sc.message, customer_name, st.session_state.session_id)
            tickets = {t.ticket_id: t for t in be.list_tickets()}
            checks = qa_checks(r, sc.expected, tickets.get)
            failed = [c.name for c in checks if not c.passed]
            rows.append(
                {
                    "message": sc.message,
                    "expected": sc.expected.label,
                    "predicted": r.classification.label if r.classification else "—",
                    "pass": r.classification == sc.expected,
                    "routed": next(c.passed for c in checks if c.name == "routing"),
                    "QA": not failed,
                    "failed checks": ", ".join(failed),
                    "agent path": " → ".join(r.agent_path),
                    "ticket": r.ticket_id or "",
                    "response": r.response,
                }
            )
        st.session_state.scenario_rows = rows

    rows = st.session_state.get("scenario_rows")
    if rows:
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        for col, key, label in ((c1, "pass", "Classification accuracy"), (c2, "routed", "Routing success"), (c3, "QA", "QA pass rate")):
            passed = int(df[key].sum())
            col.metric(label, f"{passed / len(df):.0%}", f"{passed}/{len(df)} passed", delta_color="off")
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={k: st.column_config.CheckboxColumn() for k in ("pass", "routed", "QA")},
        )
        st.caption("Full report with LLM-judge empathy/clarity scores: `uv run banking-eval --backend llm --judge`")


def tickets_tab() -> None:
    be = backend()
    tickets = be.list_tickets()
    status_filter = st.pills("Status", [s.value for s in TicketStatus], selection_mode="multi")
    df = pd.DataFrame([t.model_dump() for t in tickets])
    if not df.empty:
        if status_filter:
            df = df[df["status"].isin(status_filter)]
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "created_at": st.column_config.DatetimeColumn("created", format="YYYY-MM-DD HH:mm"),
                "updated_at": st.column_config.DatetimeColumn("updated", format="YYYY-MM-DD HH:mm"),
            },
        )

    st.subheader("Update ticket status")
    with st.form("update_ticket", border=True):
        c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        ticket_id = c1.selectbox("Ticket", [t.ticket_id for t in tickets])
        status = c2.selectbox("New status", list(TicketStatus), format_func=lambda s: s.value)
        if c3.form_submit_button("Update", width="stretch") and ticket_id:
            be.update_ticket_status(ticket_id, status)
            st.toast(f"Ticket #{ticket_id} → {status.value}")
            st.rerun()


def logs_tab() -> None:
    logs = backend().list_logs()
    if not logs:
        st.caption("No interactions logged yet.")
        return
    df = pd.DataFrame(
        {
            "id": l.log_id,
            "time": l.created_at,
            "customer": l.customer_name,
            "message": l.user_message,
            "classification": l.classification.label if l.classification else "—",
            "agent path": " → ".join(l.agent_path),
            "ticket": l.ticket_id or "",
            "ticket action": l.ticket_action.value,
            "success": l.success,
            "rating": {1: "👍", 0: "👎"}.get(l.user_rating, ""),
            "latency ms": round(l.latency_ms, 1),
        }
        for l in logs
    )
    c1, c2 = st.columns(2)
    cls = c1.multiselect("Classification", sorted(df["classification"].unique()))
    only_failures = c2.toggle("Failures only")
    if cls:
        df = df[df["classification"].isin(cls)]
    if only_failures:
        df = df[~df["success"]]

    st.caption("Select a row to see its prompt trace.")
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={"time": st.column_config.DatetimeColumn(format="HH:mm:ss")},
    )
    if event.selection.rows:
        log_id = int(df.iloc[event.selection.rows[0]]["id"])
        selected = next(l for l in logs if l.log_id == log_id)
        st.subheader(f"Trace for interaction #{log_id}")
        render_details(selected)


def metrics_tab() -> None:
    be = backend()
    logs = be.list_logs()
    if not logs:
        st.caption("No data yet.")
        return
    total = len(logs)
    successes = sum(l.success for l in logs)
    rated = [l.user_rating for l in logs if l.user_rating is not None]
    created = sum(l.ticket_action.value == "created" for l in logs)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Interactions", total)
    c2.metric("Agent success rate", f"{successes / total:.0%}")
    c3.metric("Tickets created", created)
    c4.metric("Avg latency", f"{sum(l.latency_ms for l in logs) / total:.0f} ms")
    c5.metric("Helpful (👍)", f"{sum(rated) / len(rated):.0%}" if rated else "—", f"{len(rated)} rated", delta_color="off")

    df = pd.DataFrame(
        {
            "classification": l.classification.label if l.classification else "Unclassified",
            "handler": l.agent_path[-1] if l.agent_path else "—",
            "success": l.success,
        }
        for l in logs
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Classification mix")
        st.bar_chart(df["classification"].value_counts(), horizontal=True, x_label="messages", y_label="")
    with right:
        st.subheader("Success / failure by agent")
        by_agent = (
            df.assign(outcome=df["success"].map({True: "success", False: "failure"}))
            .groupby(["handler", "outcome"])
            .size()
            .unstack(fill_value=0)
        )
        st.bar_chart(by_agent, horizontal=True, stack=True, x_label="messages", y_label="")

    # Improvement loop: thumbs-down replies and failures are the review queue for prompt/scenario updates.
    st.subheader("Needs review")
    review = [l for l in logs if l.user_rating == 0 or not l.success]
    if not review:
        st.caption("No 👎-rated or failed interactions.")
    else:
        st.caption("Add these to evaluation scenarios or adjust prompts, then re-run the evaluation.")
        st.dataframe(
            pd.DataFrame(
                {
                    "id": l.log_id,
                    "message": l.user_message,
                    "classification": l.classification.label if l.classification else "—",
                    "response": l.response,
                    "reason": "👎" if l.user_rating == 0 else (l.error or "failed"),
                }
                for l in review
            ),
            hide_index=True,
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="Banking Support AI", page_icon=":material/account_balance:", layout="wide")
    init_session()
    customer_name = sidebar()

    st.title("Banking Customer Support AI")
    st.caption("Multi-agent assistant: Classifier → Feedback Handler / Query Handler")

    chat, scenarios, tickets, logs, metrics = st.tabs(
        [":material/chat: Chat", ":material/science: Scenarios", ":material/confirmation_number: Tickets",
         ":material/receipt_long: Logs", ":material/monitoring: Metrics"]
    )
    with chat:
        chat_tab(customer_name)
    with scenarios:
        scenarios_tab(customer_name)
    with tickets:
        tickets_tab()
    with logs:
        logs_tab()
    with metrics:
        metrics_tab()


main()
