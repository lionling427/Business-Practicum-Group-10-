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
            "3. The analysis runs automatically: start with **Overview** or the "
            "**CEO brief**."
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
    Return the statistical analysis, running it automatically the first time a
    page needs it (it takes well under a second). If it fails, show why with a
    retry button instead of retrying on every rerun.
    """

    analysis = st.session_state.get("uploaded_analysis")

    if analysis is not None:
        return analysis

    _, validation = get_data()

    if validation["capabilities"]["level"] != OUTCOME_LEVEL:
        return None

    error = st.session_state.get("analysis_error")

    if error is None:

        with st.spinner("Finding what drives your results..."):

            try:
                return run_statistical_analysis()
            except Exception as exc:
                st.session_state["analysis_error"] = str(exc)
                error = str(exc)

    with st.container(border=True):

        st.error("The statistical analysis could not be completed.",
                 icon=":material/error:")
        st.caption(f"Technical detail: {error}")

        if st.button("Try again", icon=":material/refresh:", key=key):
            st.session_state.pop("analysis_error", None)
            st.rerun()

    return None


# PROCESS – GUIDE AND GLOSSARY (SIDEBAR)

PAGE_GUIDE = [
    ("How are we doing overall?", "app_pages/overview.py", ":material/dashboard:"),
    ("I need a one-page summary for leadership", "app_pages/ceo_brief.py", ":material/summarize:"),
    ("Which questions score high or low?", "app_pages/engagement.py", ":material/favorite:"),
    ("Is engagement getting better or worse?", "app_pages/change.py", ":material/timeline:"),
    ("What drives turnover, absence and performance?", "app_pages/outcomes.py", ":material/query_stats:"),
    ("Which teams need help first?", "app_pages/teams.py", ":material/groups:"),
    ("What should we do, and where do we start?", "app_pages/action_story.py", ":material/auto_stories:"),
    ("Who is likely to leave next year?", "app_pages/predictions.py", ":material/online_prediction:"),
    ("What should each team's manager do?", "app_pages/recommendations.py", ":material/tips_and_updates:"),
    ("What does it cost, and what is it worth?", "app_pages/financials.py", ":material/payments:"),
    ("What if we improve something?", "app_pages/scenarios.py", ":material/tune:"),
    ("I have my own question", "app_pages/agent.py", ":material/chat:"),
    ("Is my data OK?", "app_pages/data_quality.py", ":material/fact_check:"),
    ("Can we trust these numbers?", "app_pages/audit.py", ":material/verified:"),
]

GLOSSARY = [
    ("Q12", "Gallup's 12 engagement questions, each answered from 1 (strongly disagree) "
            "to 5 (strongly agree)."),
    ("Engaged", "An employee whose average answer is 4.0 or higher."),
    ("Agree", "Answering 4 or 5 on a question."),
    ("Wave", "One round of the survey (wave 1 = the first survey, wave 2 = the next)."),
    ("Linked to", "People who score higher on a question also tend to have better results, "
                  "with other factors held equal. A link, not proof of cause."),
    ("Passed all checks", "The link is strong enough, unlikely to be chance, and survives "
                          "the robustness checks. Only these drive advice and money figures."),
    ("Lower odds of leaving", "How much less likely someone is to leave for each 1-point "
                              "higher score. 34% lower odds means clearly less likely."),
    ("Likely range", "Where the true number probably lies (95% confidence)."),
    ("High / Elevated risk", "The riskiest 20% of teams / the next 20%, from a model tested "
                             "on data it never saw."),
    ("Priority team", "Low on its focus question and worse than the company on turnover "
                      "or absence."),
    ("Privacy", "Only groups of 5 or more people are ever shown; never individual scores."),
]


def sidebar_help():
    """Two small popovers: which page answers my question, and what the words mean."""

    with st.sidebar:

        with st.popover("Which page do I need?", icon=":material/explore:",
                        width="stretch"):
            st.markdown("**Pick your question**")
            for question, page, icon in PAGE_GUIDE:
                st.page_link(page, label=question, icon=icon)

        if st.button("What do the words mean?", icon=":material/menu_book:",
                     width="stretch", key="open_glossary"):
            glossary_dialog()


@st.dialog("What the words mean", width="large")
def glossary_dialog():
    """Shown in the main (light) area so it stays readable."""
    for term, meaning in GLOSSARY:
        st.markdown(f"**{term}**: {meaning}")


# PROCESS – TIME-SERIES READINESS (DATA AND PREDICTIONS PAGES)

READY_BADGES = {"pass": ":green-badge[:material/check: Ready]",
                "warn": ":orange-badge[:material/warning: Improve]",
                "fail": ":red-badge[:material/close: Missing]"}


def readiness_panel(ready, show_checks=True):
    """Verdict, what can run, and every check with how to fix it."""

    show = st.success if ready["ready_for_survival"] and not ready["counts"]["fail"] else \
        st.error if ready["counts"]["fail"] else st.warning
    show(ready["verdict"], icon=":material/schedule:")

    st.markdown("  ".join(
        f":{'green' if ok else 'gray'}-badge[:material/{'check' if ok else 'block'}: {name}]"
        for name, (ok, _) in ready["can_run"].items()))
    st.caption(" · ".join(f"{name}: {need}" for name, (ok, need) in ready["can_run"].items()
                          if not ok))

    if show_checks:
        for check in ready["checks"]:
            st.markdown(f"{READY_BADGES[check['status']]} **{check['check']}:** {check['result']}"
                        + (f"  \n:material/build: *How to fix:* {check['fix']}" if check["fix"] else ""))
