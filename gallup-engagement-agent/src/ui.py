"""Shared building blocks for the report pages (app_pages/*.py).

Pages stay short, direct scripts; anything two pages need lives here.
Styling comes from .streamlit/config.toml, not CSS.
"""

# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.analytics import analyze_dataframe
from src.tools.insights import analysis_frame
from src.tools.traps import run_trap_checks
from src.tools.waves import model_sample


# PROCESS 2 – CONFIGURATION

OUTCOME_LEVEL = "OUTCOME_ANALYSIS"

OUTCOME_TILES = {
    "left_voluntarily_12m": (
        "Voluntary turnover",
        lambda s: f"{s.mean() * 100:.1f}%",
        "Share of employees who left voluntarily within 12 months.",
    ),
    "unplanned_absence_days_6m": (
        "Unplanned absence",
        lambda s: f"{s.mean():.1f} days",
        "Average unplanned absence days per employee over 6 months.",
    ),
    "performance_rating": (
        "Performance rating",
        lambda s: f"{s.mean():.2f}",
        "Average performance rating (1–5).",
    ),
}


# PROCESS 3 – DATA ACCESS

def get_data():
    """Return (dataframe, validation) for the uploaded file, or (None, None)."""

    return (
        st.session_state.get("uploaded_data"),
        st.session_state.get("uploaded_validation"),
    )


def require_data(stop_when_blocked=True):
    """Return (dataframe, validation), or show the empty state and stop the page.

    When the data health check found errors that block every analysis, the page
    stops with the reasons (the Data page passes stop_when_blocked=False)."""

    df, validation = get_data()

    if df is None:
        empty_state()
        st.stop()

    report = st.session_state.get("health_report")

    if stop_when_blocked and report and "all" in report["summary"]["blocked_outcomes"]:

        with st.container(border=True):
            st.error(
                "Analysis stopped: the data health check found errors that would make "
                "every result unreliable.",
                icon=":material/block:",
            )
            st.markdown("\n".join(
                f"- **{c['check']}:** {c['detail']}"
                for c in report["checks"] if c["status"] == "FAIL"))
            st.caption("Fix these in the source file and upload it again. The **Data** page "
                       "lists every check.")
        st.stop()

    return df, validation


def trap_checks(df):
    """Analytical-trap checks, run once per upload (and again once the
    statistical analysis exists, so trapped reported findings are named)."""

    analysis = st.session_state.get("uploaded_analysis")
    key = "with_findings" if analysis is not None else "data_only"
    cache = st.session_state.setdefault("trap_checks", {})

    if key not in cache:
        with st.spinner("Checking for analytical traps..."):
            cache[key] = run_trap_checks(
                df, analysis["findings"] if analysis is not None else None
            )

    return cache[key]


def run_statistical_analysis():
    """Run the outcome models on the raw upload, obeying its health report:
    errors stop the analysis, blocked outcomes are left out, and required
    controls are added. Returns the analysis (also stored in session state)."""

    raw = st.session_state.get("uploaded_raw", st.session_state.get("uploaded_data"))

    analysis = analyze_dataframe(
        model_sample(raw),
        dataset_name=st.session_state.get("uploaded_name", "uploaded_dataset"),
        health_report=st.session_state.get("health_report"),
    )

    st.session_state["uploaded_analysis"] = analysis
    return analysis


def empty_state():

    with st.container(border=True):

        st.subheader(
            "Upload survey data to build your report",
            anchor=False,
        )

        st.markdown(
            "1. Use **Survey data** in the sidebar to upload a CSV or Excel file "
            "with one row per employee and the 12 Q12 answers (1–5).\n"
            "2. Add business outcomes (turnover, absence, performance) to unlock "
            "the **Outcomes** page.\n"
            "3. Run the statistical analysis to see which questions are linked "
            "to results."
        )

        st.caption(
            "Without an upload, **Ask AI** uses the built-in "
            "development dataset."
        )


# PROCESS 4 – PAGE HEADER

def page_header(title, icon, description):

    df, validation = get_data()

    st.title(title, icon=icon, anchor=False)

    st.caption(description)

    if df is None:
        return

    parts = [
        st.session_state.get("uploaded_name", "Uploaded data"),
        f"{len(df):,} survey responses",
    ]

    if "wave" in df.columns and df["wave"].nunique() > 1:
        parts.append(f"{df['wave'].nunique()} survey waves")

    analysis_badge = (
        ":green-badge[:material/check_circle: Analysis complete]"
        if st.session_state.get("uploaded_analysis") is not None
        else ":orange-badge[:material/pending: Analysis not run]"
        if validation["capabilities"]["level"] == OUTCOME_LEVEL
        else ":gray-badge[Survey only]"
    )

    st.markdown(
        " · ".join(parts) + "  " + analysis_badge
    )


# PROCESS 5 – EXPLANATION BLOCKS

def story_box(story):
    """Bordered takeaway box: bold headline plus bullet points."""

    with st.container(border=True):

        if story.get("headline"):
            st.markdown(f"**{story['headline']}**")

        if story.get("points"):
            st.markdown(
                "\n".join(f"- {point}" for point in story["points"])
            )


def how_to_read(text, label="How to read this chart"):

    with st.expander(label, icon=":material/help:"):
        st.write(text)


# PROCESS 6 – OUTCOME TILES

def outcome_tiles(df):
    """Bordered metric tiles for each business outcome in the model sample."""

    frame = analysis_frame(df)

    tiles = [
        (label, fmt(frame[column]), help_text)
        for column, (label, fmt, help_text) in OUTCOME_TILES.items()
        if column in frame.columns
    ]

    for label, value, help_text in tiles:
        st.metric(label, value, help=help_text, border=True)

    return len(tiles)


# PROCESS 7 – STATISTICAL ANALYSIS RUNNER

def analysis_runner(df, key):
    """
    Return the stored analysis. If outcome data exists but the analysis
    has not run yet, show a call-to-action card with a run button.
    """

    analysis = st.session_state.get("uploaded_analysis")

    if analysis is not None:
        return analysis

    _, validation = get_data()

    if validation["capabilities"]["level"] != OUTCOME_LEVEL:
        return None

    with st.container(border=True):

        st.markdown(
            "**:material/query_stats: Find what drives your results**"
        )

        st.caption(
            "Runs the statistical models and robustness checks that link "
            "each Q12 question to turnover, absence and performance. "
            "It takes a few seconds."
        )

        if st.button(
            "Run statistical analysis",
            type="primary",
            icon=":material/play_arrow:",
            key=key,
        ):

            with st.spinner(
                "Running statistical models and robustness checks..."
            ):

                try:

                    run_statistical_analysis()

                except Exception as exc:

                    st.error("Statistical analysis could not be completed.")
                    st.caption(f"Technical detail: {exc}")
                    return None

            st.toast(
                "Statistical analysis complete.",
                icon=":material/check_circle:",
            )
            st.rerun()

    return None
