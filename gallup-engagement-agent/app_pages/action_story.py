# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.action_story import CHAPTERS, build_action_story, for_screen, story_markdown
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
    "Action story",
    ":material/auto_stories:",
    "Predictions, recommendations and money as one story: what is likely to happen, what "
    "to do about it, what it is worth and where to start.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info("The action story needs business outcomes (turnover, absence or performance) "
            "in the uploaded data.", icon=":material/info:")
    st.stop()

analysis = analysis_runner(df, key="run_analysis_story")
assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))

with st.spinner("Putting the story together..."):
    story = build_action_story(df, analysis, st.session_state.get("change_analysis"),
                               assumptions, trap_checks(df))


# PROCESS 3 – THE STORY IN ONE SENTENCE

with st.container(border=True):
    st.caption("THE STORY IN ONE SENTENCE")
    st.subheader(for_screen(story["headline"]), anchor=False)


# PROCESS 4 – CHAPTERS 1–3

for number, (key, title, icon) in enumerate(CHAPTERS, 1):

    chapter = story["chapters"][key]

    with st.container(border=True):

        st.markdown(f"**{icon} {number}. {title}**")
        st.markdown(f"*{for_screen(chapter['lead'])}*")

        if chapter["points"]:
            st.markdown("\n".join(f"- {for_screen(p)}" for p in chapter["points"]))


# PROCESS 5 – CHAPTER 4: WHERE TO START

start = story["start_here"]

with st.container(border=True):

    st.markdown("**:material/flag: 4. Where to start**")

    if start.empty:
        st.caption("No team shows up on two or more lists.")
    else:
        st.markdown("*These teams show up on two or more lists: high risk, biggest "
                    "opportunity or highest extra cost.*")
        st.dataframe(
            start,
            hide_index=True,
            width="stretch",
            column_config={
                "team_id": "Team",
                "department": "Department",
                "lists": st.column_config.NumberColumn("Lists", format="%d"),
                "why": "Why it is here",
                "start_with": "Team's first step",
                "extra_cost": st.column_config.NumberColumn(
                    "Extra yearly cost", format="compact"),
            },
            alt="Teams that show up on several lists, with their first step",
        )


# PROCESS 6 – NEXT STEPS

with st.container(border=True):
    st.markdown("**:material/checklist: Next steps**")
    st.markdown("\n".join(f"{i}. {for_screen(s)}" for i, s in enumerate(story["next_steps"], 1)))

st.caption(story["caution"])

st.download_button(
    "Download the story (Markdown)",
    data=story_markdown(story).encode(),
    file_name="action_story.md",
    mime="text/markdown",
    icon=":material/download:",
)
