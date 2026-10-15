# PROCESS 1 – IMPORTS

import streamlit as st

from src.agent import ask_agent
from src.tools.agent_context import build_context, check_numbers, route_sections
from src.tools.analytics import analyze_dataframe
from src.tools.longitudinal import analyze_change, survey_waves
from src.tools.waves import model_sample
from src.ui import page_header, require_data, run_statistical_analysis


# PROCESS 2 – HEADER

page_header(
    "Ask AI",
    ":material/chat:",
    "Ask in plain language. Answers combine every verified result in the app: "
    "statistics, trends, team risk, recommendations and costs.",
)

SECTION_NAMES = {
    "overview": "Overview",
    "trends": "Trends",
    "teams": "Teams",
    "predictions": "Predictions",
    "recommendations": "Recommendations",
    "financials": "Financials",
}

EXAMPLES = [
    "Which teams should I worry about, and what will it cost?",
    "Is engagement getting better over time?",
    "What should managers do to improve productivity?",
    "Which departments are most likely to lose people next year?",
    "Give me a summary of everything.",
]


# PROCESS 3 – DATA AND EVIDENCE

df = st.session_state.get("uploaded_data")

if df is not None:

    # Stops the page when the data health check blocked every analysis.
    require_data()

    # The agent needs the analyses behind every page; run any that are missing once.
    missing = st.session_state.get("uploaded_analysis") is None or (
        len(survey_waves(df)) >= 2 and st.session_state.get("change_analysis") is None)

    if missing:
        with st.spinner("Preparing the analyses the agent answers from..."):
            if st.session_state.get("uploaded_analysis") is None:
                run_statistical_analysis()
            if len(survey_waves(df)) >= 2 and st.session_state.get("change_analysis") is None:
                try:
                    st.session_state["change_analysis"] = analyze_change(df)
                except ValueError:
                    pass

    analysis = st.session_state["uploaded_analysis"]
    st.caption(
        f":material/check_circle: Answering from **{st.session_state.get('uploaded_name')}** "
        f"· {analysis['sample_size']:,} employees in the statistical models."
    )

else:

    analysis = None
    st.caption(
        ":material/info: No data uploaded, so the agent answers from the development "
        "dataset (statistics, trends, team risk, recommendations and costs). Upload your "
        "own survey data to get answers about your organization."
    )


def context_for(question):
    """Verified dashboard facts for this question (cached per upload and sections)."""
    if df is None:
        return None, []
    sections = route_sections(question)
    key = (
        st.session_state.get("upload_signature"),
        tuple(sections),
        repr(st.session_state.get("assumptions")),
        st.session_state.get("change_analysis") is not None,
    )
    cache = st.session_state.setdefault("agent_context_cache", {})
    if key not in cache:
        cache[key] = build_context(
            df,
            analysis,
            st.session_state.get("change_analysis"),
            st.session_state.get("assumptions"),
            sections,
        )
    return cache[key], sections


# PROCESS 4 – CHAT CONTROLS

WELCOME = (
    "Hello. Ask me anything about engagement in your organization: which teams need "
    "support, what is changing, what managers should do, or what it is worth. I only use "
    "verified results and never discuss individual employees."
)

AVATARS = {"assistant": ":material/smart_toy:", "user": ":material/person:"}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]


def ask_example():
    """Pill clicked: queue its question and clear the pill."""
    st.session_state["pending_question"] = st.session_state["example_pills"]
    st.session_state["example_pills"] = None


def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]


# Examples wrap in a wide column; the clear button sits on the right.
examples_col, clear_col = st.columns([5, 1], vertical_alignment="top")

examples_col.pills(
    "Try a question",
    options=EXAMPLES,
    key="example_pills",
    on_change=ask_example,
)

clear_col.button(
    "Clear",
    icon=":material/restart_alt:",
    on_click=clear_chat,
    disabled=len(st.session_state.messages) <= 1,
    width="stretch",
    help="Start a new conversation.",
)


# PROCESS 5 – CHAT HISTORY

def safe(text):
    """Escape '$' so amounts like $5.5M and $465K are not read as a math formula."""
    return text.replace("$", "\\$")


def readable(value):
    """One section of evidence as short markdown bullets."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        lines = []
        for v in value:
            lines += readable(v) if not isinstance(v, dict) else [
                "; ".join(f"{k.replace('_', ' ')}: {x}" for k, x in v.items()
                          if not isinstance(x, (list, dict)))]
        return lines
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (str, int, float)) or v is None:
                lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")
            else:
                lines += readable(v)
        return lines
    return [str(value)]


def show_sources(message):

    if message.get("unverified"):
        st.caption(
            ":material/warning: Could not trace these figures to the verified results: "
            + ", ".join(f"{n:g}" for n in message["unverified"])
            + ". Check them on the relevant page before sharing."
        )

    if not message.get("sections"):
        return

    with st.expander(
        "Based on: " + ", ".join(SECTION_NAMES[s] for s in message["sections"]),
        icon=":material/fact_check:",
    ):
        evidence = message.get("evidence") or {}
        if "statistical_findings" in evidence:
            st.caption("Plus the detailed statistical findings for the question asked.")
        for name in message["sections"]:
            if name in evidence:
                st.markdown(f"**{SECTION_NAMES[name]}**")
                st.markdown("\n".join(f"- {line}" for line in readable(evidence[name])[:12])
                            .replace("$", "\\$"))


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(safe(message["content"]))
        show_sources(message)


# PROCESS 6 – ASK

typed = st.chat_input("Ask a question about employee engagement...")
user_question = typed or st.session_state.pop("pending_question", None)

if user_question:

    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):

        with st.spinner("Combining the verified results..."):

            try:
                context, sections = context_for(user_question)

                result = ask_agent(
                    user_question,
                    analysis_override=analysis,
                    context=context,
                    return_debug=True,
                )

                answer = result["answer"]
                evidence = result.get("evidence") or {}
                message = {
                    "role": "assistant",
                    "content": answer,
                    # Uploaded data: sections from the cached context. No upload:
                    # the agent builds the development dataset's facts itself.
                    "sections": sections or result.get("sections", []),
                    "evidence": evidence if isinstance(evidence, dict) else {},
                    "unverified": check_numbers(answer, evidence, user_question)
                    if evidence else [],
                }

            except Exception as exc:

                answer = ("I could not complete the analysis. Please check the application "
                          "configuration or try again.")
                message = {"role": "assistant", "content": answer}
                st.error(answer)
                # Development-only diagnostic
                st.caption(f"Technical detail: {exc}")

        if "sections" in message:
            st.markdown(safe(answer))
            show_sources(message)

    st.session_state.messages.append(message)
    st.rerun()
