# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.ceo_brief import brief_for_screen, brief_markdown, build_ceo_brief
from src.tools.financial import Assumptions
from src.tools.insights import analysis_frame
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    page_header,
    require_data,
    trap_checks,
)


# PROCESS 2 – HEADER AND DATA

page_header(
    "CEO brief",
    ":material/summarize:",
    "The statistical results on one page: where we stand, what drives results, what it "
    "is worth and what to decide.",
)

df, validation = require_data()

has_outcomes = validation["capabilities"]["level"] == OUTCOME_LEVEL
analysis = analysis_runner(df, key="run_analysis_ceo") if has_outcomes else None
assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))

brief = build_ceo_brief(df, analysis, assumptions,
                        trap_checks(df) if has_outcomes else None)
shown = brief_for_screen(brief)


# PROCESS 3 – BOTTOM LINE AND NUMBERS

with st.container(border=True):

    st.caption(f"BOTTOM LINE · {brief['survey']}")
    st.subheader(shown["bottom_line"], anchor=False)

for tile, column in zip(shown["scorecard"], st.columns(len(shown["scorecard"]))):
    column.metric(tile["label"], tile["value"], delta=tile["change"], delta_color="off",
                  delta_arrow="off", border=True)


# PROCESS 4 – THE STORY IN THREE BOXES

boxes = [
    (":material/insights: What drives results", shown["what_matters"]),
    (":material/payments: What it is worth", shown["money"]),
    (":material/flag: Where to act", shown["where_to_act"]),
]

for (title, points), column in zip(boxes, st.columns(3)):

    with column.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        if points:
            st.markdown("\n".join(f"- {p}" for p in points))
        else:
            st.caption("Available once the statistical analysis has run.")


# PROCESS 5 – DECISIONS AND CONFIDENCE

with st.container(border=True):

    st.markdown("**:material/gavel: Decisions for you**")
    st.markdown("\n".join(f"{i}. {d}" for i, d in enumerate(shown["decisions"], 1)))

with st.expander("How sure are we", icon=":material/verified:", expanded=True):
    st.markdown("\n".join(f"- {p}" for p in shown["confidence"]))

st.download_button(
    "Download the brief (Markdown)",
    data=brief_markdown(brief).encode(),
    file_name="ceo_engagement_brief.md",
    mime="text/markdown",
    icon=":material/download:",
    help="Paste into an email or document. Every number comes from the app's verified results.",
)
