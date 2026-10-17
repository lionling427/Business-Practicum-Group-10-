# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.action_tools import DISCLAIMER, links_markdown, resources_for
from src.tools.dashboard import team_watchlist
from src.tools.data_ingestion import Q12_ITEMS
from src.tools.insights import (
    MIN_TEAM_SIZE,
    analysis_frame,
    brief_markdown,
    manager_brief,
    team_scorecard,
)
from src.tools.manager_coach import coach_brief
from src.tools.strengths import strength_play, team_profiles
from src.tools.prediction import (
    TARGETS,
    explain_team_risk,
    team_risk,
)
from src.tools.visualization import (
    explain_team_heatmap,
    team_heatmap_chart,
)
from src.tools.waves import model_wave, wave_label
from src.ui import (
    how_to_read,
    page_header,
    require_data,
    story_box,
)


RISK_MARKS = {"High": "🔴 High", "Elevated": "🟠 Elevated", "Typical": "⚪ Typical"}


# PROCESS 2 – HEADER AND DATA

page_header(
    "Teams",
    ":material/groups:",
    "Which teams need support first, and a practical plan for each manager.",
)

df, validation = require_data()

analysis = st.session_state.get("uploaded_analysis")

scorecard = team_scorecard(df, analysis) if validation["is_valid"] else None

if scorecard is None or scorecard.empty:
    st.info(
        f"Team views need a `team_id` column and teams with at least {MIN_TEAM_SIZE} people.",
        icon=":material/info:",
    )
    st.stop()

risk_results = {}
for outcome, (column, _) in TARGETS.items():
    if column in df.columns:
        try:
            risk_results[outcome] = team_risk(df, outcome)
        except ValueError:
            risk_results[outcome] = None

watchlist = team_watchlist(scorecard, risk_results)
risk_ok = {o: r is not None and r["status"] == "useful" for o, r in risk_results.items()}


# PROCESS 3 – TILES

tiles = [
    ("Teams", f"{len(watchlist):,}", f"Teams with {MIN_TEAM_SIZE}+ people in {wave_label(model_wave(df))}."),
    ("Priority teams", f"{int(watchlist['priority'].sum()):,}",
     "Bottom quarter on their focus question and worse turnover or absence than the company."),
]
for outcome in ["turnover", "absence"]:
    if risk_ok.get(outcome):
        tiles.append((
            f"High {outcome} risk",
            f"{int((watchlist[f'{outcome}_risk'] == 'High').sum()):,}",
            f"Riskiest 20% of teams for above-normal {outcome}, from a model that passed its test.",
        ))

for (label, value, help_text), column in zip(tiles, st.columns(len(tiles))):
    column.metric(label, value, help=help_text, border=True)

if analysis is None:
    st.caption(
        ":material/info: Add turnover, absence or performance data to flag priority "
        "teams and link focus questions to business results."
    )


# PROCESS 4 – WATCHLIST · HEATMAP · RISK MODEL

watch_tab, heat_tab, model_tab = st.tabs([
    ":material/flag: Watchlist",
    ":material/grid_on: Heatmap",
    ":material/verified: How the risk model was tested",
])

with watch_tab:

    filter_col, toggle_col = st.columns([1, 2], vertical_alignment="bottom")

    departments = sorted(watchlist["department"].dropna().unique()) if "department" in watchlist else []
    department = filter_col.selectbox(
        "Department",
        options=[None] + departments,
        format_func=lambda d: "All departments" if d is None else d,
        key="watch_department",
    )
    attention_only = toggle_col.toggle(
        "Only teams that need attention",
        value=True,
        help="Priority teams and teams with high or elevated predicted risk.",
        key="watch_attention",
    )

    shown = watchlist
    if department is not None:
        shown = shown[shown["department"] == department]
    if attention_only:
        shown = shown[shown["needs_attention"]]

    if shown.empty:
        st.info("No teams match these filters.")
        selected_team = None
    else:
        st.caption(":material/touch_app: Click a row to open that team's coaching plan below.")

        event = st.dataframe(
            shown[["team_id", "department", "employees", "q12_average", "turnover_risk",
                   "absence_risk", "focus_label", "priority"]].replace(
                {"turnover_risk": RISK_MARKS, "absence_risk": RISK_MARKS}),
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="watchlist_table",
            column_config={
                "team_id": "Team",
                "department": "Department",
                "employees": st.column_config.NumberColumn("People", format="%d"),
                "q12_average": st.column_config.ProgressColumn(
                    "Average Q12", min_value=1, max_value=5, format="%.2f"),
                "turnover_risk": "Turnover risk",
                "absence_risk": "Absence risk",
                "focus_label": "Focus question",
                "priority": st.column_config.CheckboxColumn("Priority"),
            },
            alt="Teams ranked by how urgently they need support",
        )

        rows = event.selection.rows
        selected_team = shown.iloc[rows[0]]["team_id"] if rows else shown.iloc[0]["team_id"]

        st.download_button(
            "Download watchlist (CSV)",
            data=watchlist.to_csv(index=False).encode(),
            file_name="team_watchlist.csv",
            mime="text/csv",
            icon=":material/download:",
        )

with heat_tab:

    items = [item for item in Q12_ITEMS if item in watchlist.columns]
    company_means = analysis_frame(df)[items].mean()

    heatmap_story = explain_team_heatmap(watchlist, company_means)
    story_box(heatmap_story)

    st.altair_chart(
        team_heatmap_chart(watchlist, company_means),
        width="stretch",
        alt="Top 25 teams on the watchlist: each question vs the company average",
    )

    how_to_read(heatmap_story["how_to_read"])

with model_tab:

    tested = [o for o, r in risk_results.items() if r is not None]

    if not tested:
        st.info("Risk predictions need turnover or absence data.")
    else:
        for outcome, column in zip(tested, st.columns(len(tested), gap="large")):
            with column:
                st.markdown(f"**{outcome.capitalize()} risk**")
                story_box(explain_team_risk(risk_results[outcome]))
                if not risk_ok[outcome]:
                    st.caption(":material/visibility_off: Not shown in the watchlist.")

        how_to_read(explain_team_risk(risk_results[tested[0]])["how_to_read"],
                    "What the risk bands mean")


# PROCESS 5 – MANAGER COACH FOR THE SELECTED TEAM

if selected_team is None:
    st.stop()

brief = manager_brief(df, selected_team, analysis=analysis, risk_results=risk_results)

with st.container(border=True):

    title_col, button_col = st.columns([3, 1], vertical_alignment="center")

    title_col.subheader(f"Manager coach · Team {selected_team}", anchor=False)

    coach_key = (st.session_state.get("upload_signature"), selected_team, analysis is not None)
    coach_notes = st.session_state.setdefault("coach_notes", {})

    if button_col.button("Write coaching note", icon=":material/auto_awesome:",
                         width="stretch"):
        with st.spinner("Writing a coaching note from verified facts..."):
            coach_notes[coach_key] = coach_brief(brief)

    note = coach_notes.get(coach_key)

    if note is not None and note["source"] == "ai":

        ai_tab, brief_tab = st.tabs(["AI coaching note", "Verified brief"])

        with ai_tab:
            st.markdown(note["text"])
            st.caption("Written by AI from the verified brief; every number was checked "
                       "against it.")

        with brief_tab:
            st.markdown(brief_markdown(brief))

    else:

        st.markdown(brief_markdown(brief))

        if note is not None:
            st.caption(f"Showing the verified brief instead. {note['reason']}")

    strengths = st.session_state.get("active_strengths")
    if strengths is not None:
        profile = team_profiles(strengths, df, scorecard)
        row = profile[profile["team_id"] == selected_team] if not profile.empty else profile
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**Use the team's strengths.** This team leans towards "
                        f"**{r['lead_domain']}** ({r['top_themes']}).")
            st.success(f"**Try this for {brief['focus']['label']}:** "
                       f"{strength_play(brief['focus']['item'], r['lead_domain'])}",
                       icon=":material/workspace_premium:")
    else:
        st.caption(":material/workspace_premium: Load strengths data on the **Strengths** page "
                   "to tailor this plan to what the team does best.")

    with st.expander(f"Manager learning: {brief['focus']['label']}",
                     icon=":material/school:"):
        st.markdown(links_markdown(resources_for(brief["focus"]["item"])))
        st.caption(DISCLAIMER)
