# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.data_health import (
    column_guide,
    headline_numbers,
    wave_coverage,
)
from src.tools.health_check import required_adjustments
from src.ui import (
    OUTCOME_LEVEL,
    page_header,
    require_data,
)


# PROCESS 2 – CONFIGURATION

LEVELS = {
    "SURVEY_ONLY": ("Survey only",
                    "Engagement pages work. Add business outcomes to unlock Outcomes, "
                    "Predictions, Recommendations and Financials."),
    "SEGMENT_ANALYSIS": ("Survey + workforce groups",
                         "Engagement can be compared across groups. Add business outcomes to "
                         "unlock Outcomes, Predictions, Recommendations and Financials."),
    OUTCOME_LEVEL: ("Survey + business outcomes", "Every page in the app is available."),
}

OUTCOME_NAMES = {"turnover": "Voluntary turnover", "absence": "Absenteeism",
                 "performance": "Performance"}


# PROCESS 3 – HEADER

page_header(
    "Data & quality",
    ":material/fact_check:",
    "What was uploaded, whether it can be trusted, and which pages it unlocks.",
)

df, validation = require_data(stop_when_blocked=False)

# The raw upload, before blocked outcomes were removed for the other pages.
raw = st.session_state.get("uploaded_raw", df)
report = st.session_state.get("health_report")


# PROCESS 4 – TILES

tiles = headline_numbers(raw)

for (label, value, help_text), column in zip(tiles, st.columns(len(tiles))):
    column.metric(label, value, help=help_text or None, border=True)


# PROCESS 5 – DATA HEALTH: ERRORS STOP THE ANALYSIS, BIASES GET ADJUSTMENTS

STEP_STATUS = {"PASS": ":green[:material/check_circle:]",
               "WARN": ":orange[:material/warning:]",
               "FAIL": ":red[:material/cancel:]"}

if report is not None:

    summary = report["summary"]
    blocked = summary["blocked_outcomes"]
    adjustments = required_adjustments(report)

    with st.container(border=True):

        st.subheader("Data health", anchor=False)

        if "all" in blocked:
            st.error(
                f"Analysis stopped: {summary['fail']} error"
                f"{'s' if summary['fail'] != 1 else ''} must be fixed before any result "
                "can be trusted. Every page except this one is paused.",
                icon=":material/block:",
            )
        elif blocked:
            st.warning(
                "Analysis continues, but "
                + " and ".join(b.capitalize() for b in blocked)
                + " results are stopped because their data has errors.",
                icon=":material/warning:",
            )
        elif summary["warn"]:
            st.success(
                f"No data errors. {summary['warn']} known biases were found; the analysis "
                "corrects for them with the adjustments below.",
                icon=":material/verified:",
            )
        else:
            st.success("Your data passed every check.", icon=":material/verified:")

        with st.container(horizontal=True):
            st.metric("Checks run", summary["checks_run"], border=True)
            st.metric("Passed", summary["pass"], border=True)
            st.metric("Biases to adjust for", summary["warn"], border=True)
            st.metric("Errors", summary["fail"], border=True)

        errors = [c for c in report["checks"] if c["status"] == "FAIL"]
        if errors:
            st.markdown("**Errors to fix in the source file**")
            st.markdown("\n".join(f"- **{c['check']}:** {c['detail']}" for c in errors))

        fixes = [a for a in adjustments if not a["adjustment"].endswith("fixed")]
        if fixes:
            st.markdown("**Adjustments applied automatically**")
            st.table(
                [{"What the app does": a["adjustment"], "Because": a["because"],
                  "What was found": a["detail"]} for a in fixes],
                border="horizontal",
                hide_index=True,
                alt="Adjustments the analysis applies for known biases",
            )

        analysis = st.session_state.get("uploaded_analysis")
        if analysis is not None and analysis.get("adjustments"):
            added = {o: c for o, c in analysis["adjustments"]["model_controls_added"].items() if c}
            if added:
                st.caption(":material/functions: Extra model controls in the current analysis: "
                           + "; ".join(f"{o} + {', '.join(c)}" for o, c in added.items()) + ".")

        with st.expander("All checks, by step", icon=":material/checklist:"):
            steps = {}
            for check in report["checks"]:
                steps.setdefault(check["step"], []).append(check)
            for step, checks in steps.items():
                worst = ("FAIL" if any(c["status"] == "FAIL" for c in checks) else
                         "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
                st.markdown(f"{STEP_STATUS[worst]} **{step[2:] if step[:1].isdigit() else step}**")
                st.markdown("\n".join(
                    f"- {STEP_STATUS[c['status']]} {c['check']}: {c['detail']}" for c in checks))


# PROCESS 6 – WHAT THE DATA UNLOCKS

capabilities = validation["capabilities"]
level_name, level_text = LEVELS.get(capabilities["level"], (capabilities["level"], ""))

with st.container(border=True):

    st.subheader("What this data unlocks", anchor=False)

    st.markdown(f"**{level_name}.** {level_text}")

    if capabilities.get("outcomes"):
        st.markdown(" ".join(
            f":green-badge[:material/check: {OUTCOME_NAMES.get(o, o)}]"
            for o in capabilities["outcomes"]))


# PROCESS 7 – COVERAGE BY SURVEY

coverage = wave_coverage(raw)

if not coverage.empty:

    with st.container(border=True):

        st.subheader("Coverage by survey", anchor=False)

        config = {
            "wave": st.column_config.NumberColumn("Survey", format="Wave %d"),
            "survey_date": "Date",
            "responses": st.column_config.NumberColumn("Responses", format="localized"),
            "people": st.column_config.NumberColumn("People", format="localized"),
            "teams": st.column_config.NumberColumn("Teams", format="localized"),
        }
        for name in ["turnover", "absence", "performance"]:
            if f"{name}_known" in coverage.columns:
                config[f"{name}_known"] = st.column_config.ProgressColumn(
                    f"{OUTCOME_NAMES[name]} known", min_value=0, max_value=100,
                    format="%.0f%%")

        st.dataframe(
            coverage,
            hide_index=True,
            width="stretch",
            height="content",
            column_config=config,
            alt="Responses, people, teams and outcome coverage in each survey",
        )


# PROCESS 8 – COLUMNS THE APP RECOGNIZED

with st.container(border=True):

    st.subheader("Columns the app recognized", anchor=False)

    st.table(
        column_guide(raw),
        border="horizontal",
        hide_header=True,
        alt="How the app uses each uploaded column",
    )

    with st.expander("Preview the raw data (first 20 rows)", icon=":material/table_view:"):
        st.dataframe(
            raw.head(20),
            width="stretch",
            alt="First 20 rows of the uploaded data",
        )
