# PROCESS 1 – IMPORTS
#
# Entry point: page config, the shared sidebar (data upload), and top
# navigation. Each report page lives in app_pages/.

import streamlit as st

from src.tools.data_ingestion import (
    normalize_columns,
    read_uploaded_file,
    validate_uploaded_data,
)
from src.tools.health_check import OUTCOMES, check_upload
from src.tools.longitudinal import derive_wave


# PROCESS 2 – PAGE CONFIGURATION

st.set_page_config(
    page_title="Engagement insights",
    page_icon=":material/insights:",
    layout="wide",
)


# PROCESS 3 – SIDEBAR: DATA SOURCE

SESSION_DATA_KEYS = [
    "uploaded_data",
    "uploaded_raw",
    "health_report",
    "uploaded_validation",
    "uploaded_analysis",
    "change_analysis",
    "trap_checks",
    "audit_result",
    "audit_sign_off",
    "agent_context_cache",
    "uploaded_name",
    "upload_signature",
    "coach_notes",
]

with st.sidebar:

    st.markdown(
        "### :material/insights: Engagement insights"
    )

    st.caption(
        "Gallup Q12 engagement and business outcomes"
    )

    uploaded_file = st.file_uploader(
        "Survey data",
        type=["csv", "xlsx"],
        help="CSV or Excel, one row per employee, with Q01–Q12 answers (1–5).",
        key="data_file",
    )

    if uploaded_file is None:

        # File removed: drop its data so pages show the empty state again.
        for key in SESSION_DATA_KEYS:
            st.session_state.pop(key, None)

    else:

        try:

            upload_signature = (
                f"{uploaded_file.name}:"
                f"{getattr(uploaded_file, 'size', 0)}"
            )

            # Read and validate only when the file changes.
            if st.session_state.get("upload_signature") != upload_signature:

                # Files with survey dates but no wave column get one wave
                # per survey year, so trends and models agree on periods.
                uploaded_df = derive_wave(
                    normalize_columns(read_uploaded_file(uploaded_file))
                )

                for key in SESSION_DATA_KEYS:
                    st.session_state.pop(key, None)

                # Data health: errors stop the analysis; an outcome whose data
                # is wrong (e.g. measured before the survey) is removed from
                # every page, while the raw upload stays for the Data page.
                report = check_upload(uploaded_df, uploaded_file.name)
                blocked = report["summary"]["blocked_outcomes"]
                usable = uploaded_df.drop(columns=[
                    column
                    for outcome, (column, _) in OUTCOMES.items()
                    if outcome in blocked and column in uploaded_df.columns
                ])

                st.session_state["uploaded_raw"] = uploaded_df
                st.session_state["health_report"] = report
                st.session_state["uploaded_data"] = usable
                st.session_state["uploaded_validation"] = (
                    validate_uploaded_data(usable)
                )
                st.session_state["uploaded_name"] = uploaded_file.name
                st.session_state["upload_signature"] = upload_signature

            validation = st.session_state["uploaded_validation"]
            health = st.session_state["health_report"]["summary"]

            if "all" in health["blocked_outcomes"]:

                st.badge(
                    "Analysis stopped: data errors",
                    icon=":material/block:",
                    color="red",
                )

            elif health["blocked_outcomes"]:

                st.badge(
                    "Some outcomes blocked",
                    icon=":material/warning:",
                    color="orange",
                )

            elif validation["is_valid"]:

                st.badge(
                    "Data loaded",
                    icon=":material/check_circle:",
                    color="green",
                )

            else:

                st.badge(
                    "Check data quality",
                    icon=":material/warning:",
                    color="orange",
                )

        except Exception as exc:

            for key in SESSION_DATA_KEYS:
                st.session_state.pop(key, None)

            st.error("The uploaded file could not be processed.")
            st.caption(f"Technical detail: {exc}")

    with st.expander(
        "About this report",
        icon=":material/info:",
    ):

        st.markdown(
            "This AI agent helps HR and business leaders understand how "
            "Gallup Q12 engagement factors relate to **voluntary turnover**, "
            "**absenteeism** and **performance**.\n\n"
            "1. Statistical models calculate the evidence.\n"
            "2. Data-health and robustness checks validate it.\n"
            "3. The AI explains verified results in business language."
        )

        st.caption(
            "The agent reports associations, not causal relationships."
        )


# PROCESS 4 – NAVIGATION

pages = {
    "Report": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/ceo_brief.py", title="CEO brief", icon=":material/summarize:"),
        st.Page("app_pages/engagement.py", title="Engagement", icon=":material/favorite:"),
        st.Page("app_pages/change.py", title="Trends", icon=":material/timeline:"),
        st.Page("app_pages/outcomes.py", title="Outcomes", icon=":material/query_stats:"),
        st.Page("app_pages/teams.py", title="Teams", icon=":material/groups:"),
    ],
    "Act": [
        st.Page("app_pages/action_story.py", title="Action story", icon=":material/auto_stories:"),
        st.Page("app_pages/predictions.py", title="Predictions", icon=":material/online_prediction:"),
        st.Page("app_pages/recommendations.py", title="Recommendations", icon=":material/tips_and_updates:"),
        st.Page("app_pages/financials.py", title="Financials", icon=":material/payments:"),
    ],
    "Tools": [
        st.Page("app_pages/agent.py", title="Ask AI", icon=":material/chat:"),
        st.Page("app_pages/data_quality.py", title="Data", icon=":material/fact_check:"),
        st.Page("app_pages/audit.py", title="Audit", icon=":material/verified:"),
    ],
}

# Sidebar navigation, grouped: ten report pages do not fit in a top bar.
st.navigation(
    pages,
    position="sidebar",
    expanded=True,
).run()
