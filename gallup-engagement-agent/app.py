# PROCESS 1 – IMPORTS
#
# Entry point: page config, the shared sidebar (data upload), and top
# navigation. Each report page lives in app_pages/.

import streamlit as st

from src.tools.data_ingestion import (
    normalize_columns,
    read_uploaded_file,
    upload_fingerprint,
    validate_uploaded_data,
)
from src.tools.agent_context import fact_sheet
from src.tools.financial import data_assumptions
from src.tools.health_check import OUTCOMES, check_upload
from src.tools.longitudinal import derive_wave
from src.hub import build_pages, reset_view_state
from src.layout import APP_NAME, LOGO, LOGO_ICON
from src.ui import OUTCOME_LEVEL, run_statistical_analysis, sidebar_help


# PROCESS 2 – PAGE CONFIGURATION

st.set_page_config(
    page_title=APP_NAME,
    page_icon=LOGO_ICON,
    layout="wide",
)

# Brand: full logo in the sidebar, the icon when the sidebar is collapsed.
st.logo(LOGO, size="large", icon_image=LOGO_ICON)


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
    "analysis_error",
    "audit_sign_off",
    "saved_scenarios",
    "survival_results",
    "strengths_cache",
    "active_strengths",
    "agent_context_cache",
    "uploaded_name",
    "upload_signature",
    "coach_notes",
    "assumptions",
    "salary_source",
    "agent_fact_sheet",
    "as_salary",
    "as_replace",
    "as_point",
    "as_headcount",
    "as_currency",
]

with st.sidebar:

    st.markdown(
        "### EngagAI"
    )

    st.caption(
        "Engagement intelligence for leaders: Gallup Q12 and business outcomes"
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
                f"{getattr(uploaded_file, 'size', 0)}:"
                f"{upload_fingerprint(uploaded_file)}"
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

                # Cost assumptions start from the data (average salary when the
                # file has a salary column), so every page uses the same numbers.
                assumptions, salary_source = data_assumptions(usable)
                st.session_state["assumptions"] = assumptions
                st.session_state["salary_source"] = salary_source

                # Run the analysis once now (well under a second), so every
                # page opens with results instead of a "run" step.
                if (st.session_state["uploaded_validation"]["capabilities"]["level"]
                        == OUTCOME_LEVEL and "all" not in blocked):
                    with st.spinner("Finding what drives your results..."):
                        try:
                            run_statistical_analysis()
                        except Exception as exc:          # shown on the pages
                            st.session_state["analysis_error"] = str(exc)

                # Ask AI: every verified result, built once now and reused for
                # every question (src/tools/agent_context.fact_sheet).
                with st.spinner("Preparing the verified results for Ask AI..."):
                    try:
                        st.session_state["agent_fact_sheet"] = {
                            "key": repr(st.session_state.get("assumptions")),
                            "sheet": fact_sheet(
                                usable, st.session_state.get("uploaded_analysis"),
                                None, st.session_state.get("assumptions")),
                        }
                    except Exception:                     # Ask AI builds it on demand
                        st.session_state.pop("agent_fact_sheet", None)

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
#
# The layout (which pages, in which group, which views are combined) lives in
# src/layout.py. Edit that file to move, merge or hide pages.

pages, pages_by_file = build_pages()

navigation = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

sidebar_help(pages_by_file)   # needs the pages registered above
reset_view_state()            # every page starts with its own title
navigation.run()
