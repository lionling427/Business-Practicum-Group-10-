# PROCESS 1 – IMPORTS

import pandas as pd
import streamlit as st

from src.tools.insights import analysis_frame, team_scorecard
from src.tools.strengths import (
    DOMAIN_MEANING,
    DOMAINS,
    company_profile,
    demo_strengths,
    domain_chart,
    explain_themes,
    has_strengths,
    pilot_effect,
    read_strengths,
    team_profiles,
    themes_chart,
)
from src.ui import how_to_read, page_header, require_data, story_box


# PROCESS 2 – HEADER AND DATA

page_header(
    "Strengths",
    ":material/workspace_premium:",
    "What each team is naturally good at (CliftonStrengths), and how to use it to lift "
    "the team's weakest engagement question.",
)

df, validation = require_data()
raw = st.session_state.get("uploaded_raw", df)
analysis = st.session_state.get("uploaded_analysis")


# PROCESS 3 – WHERE THE STRENGTHS DATA COMES FROM

sources = (["upload"] if has_strengths(raw) else []) + ["file", "demo"]

with st.container(border=True):

    title_col, source_col = st.columns([1.2, 2], vertical_alignment="bottom")
    title_col.markdown("**:material/upload_file: Strengths data**")
    source = source_col.segmented_control(
        "Source", options=sources, default=sources[0], key="strengths_source",
        format_func={"upload": "My upload", "file": "Strengths file",
                     "demo": "Demo data"}.get,
        label_visibility="collapsed") or sources[0]

    table, key = None, None
    if source == "upload":
        table, key = raw, ("upload", st.session_state.get("upload_signature"))
    elif source == "demo":
        table, key = demo_strengths(df), ("demo", st.session_state.get("upload_signature"))
        st.caption(":material/science: Demo data: random Top 5 themes for every employee in "
                   "your upload, and 'coached' for about 30% of teams. Random, so it shows the "
                   "page working, not real findings.")
    else:
        file = st.file_uploader(
            "Strengths file (CSV)", type=["csv"], key="strengths_file",
            help="One row per person: employee_id, top1–top5 (theme names), and optionally "
            "coached (1 = took part in strengths coaching). Export it from Gallup Access or "
            "enter it by hand.")
        if file is not None:
            table, key = pd.read_csv(file), ("file", file.name, file.size)

    example = pd.DataFrame({"employee_id": ["E0001", "E0002"],
                            "top1": ["Achiever", "Woo"], "top2": ["Learner", "Empathy"],
                            "top3": ["Relator", "Strategic"], "top4": ["Focus", "Activator"],
                            "top5": ["Input", "Harmony"], "coached": [1, 0]})
    st.download_button("Download the file format (CSV)", example.to_csv(index=False).encode(),
                       "strengths_format.csv", "text/csv", icon=":material/download:")

if table is None:
    st.info("Choose **Demo data** to see how the page works, or upload a strengths file.",
            icon=":material/touch_app:")
    st.stop()

cache = st.session_state.setdefault("strengths_cache", {})
if key not in cache:
    try:
        cache[key] = read_strengths(table)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()
strengths = cache[key]
st.session_state["active_strengths"] = strengths          # the Teams page coach uses it too

if strengths["unknown"]:
    st.warning("Not recognised as theme names (left out): " + ", ".join(strengths["unknown"][:10]),
               icon=":material/warning:")

scorecard = team_scorecard(df, analysis)
teams = team_profiles(strengths, df, scorecard)
company = company_profile(strengths)
mix = company["mix"]


# PROCESS 4 – THE COMPANY'S STRENGTHS

st.subheader(f"Company strengths · {strengths['people']:,} people", anchor=False)

for domain, column in zip(DOMAINS, st.columns(4)):
    column.metric(domain, f"{mix[domain]:.0%}", help=f"Share of all Top 5 themes. This domain "
                  f"{DOMAIN_MEANING[domain]}.", border=True)

chart_col, story_col = st.columns([3, 2], gap="large")

with chart_col:
    st.markdown("**Most common themes**")
    st.caption("Hover over a bar to see what the theme means.")
    st.altair_chart(themes_chart(company["top_themes"]), width="stretch",
                    alt="Most common CliftonStrengths themes in the company")

with story_col:
    lead, low = mix.idxmax(), mix.idxmin()
    story_box({
        "headline": f"The company leans towards {lead} ({mix[lead]:.0%} of Top 5 themes).",
        "points": [
            f"**{lead}:** {DOMAIN_MEANING[lead]}.",
            f"**Least common: {low}** ({mix[low]:.0%}): {DOMAIN_MEANING[low]}. Teams light "
            "here can borrow it from colleagues or partners.",
            f"{len(teams)} teams have strengths data for 5+ people.",
        ]})


# PROCESS 5 – EACH TEAM: STRENGTHS + FOCUS QUESTION = A TAILORED ACTION

with st.container(border=True):

    st.subheader("Use each team's strengths on its weakest question", anchor=False)

    if teams.empty:
        st.info("No team has strengths data for 5 or more people yet.", icon=":material/info:")
    else:
        department = st.selectbox("Department", ["All departments"] +
                                  sorted(teams["department"].dropna().unique()),
                                  key="strengths_department")
        shown = teams if department == "All departments" else teams[teams["department"] == department]
        st.caption(":material/touch_app: Click a row to see that team's plan below.")
        event = st.dataframe(
            shown[["team_id", "department", "people_with_strengths", "lead_domain",
                   "top_themes", "focus_label"]],
            hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            key="strengths_teams",
            column_config={"team_id": "Team", "department": "Department",
                           "people_with_strengths": st.column_config.NumberColumn("People", format="%d"),
                           "lead_domain": "Strongest domain", "top_themes": "Most common themes",
                           "focus_label": "Focus question"})

        rows = event.selection.rows
        team = shown.iloc[rows[0] if rows else 0]

        st.markdown(f"#### Team {team['team_id']} · {team['department']}")
        team_mix = pd.Series({d: team[d] for d in DOMAINS})
        plan_col, chart_col = st.columns([3, 2], gap="large")
        with chart_col:
            st.altair_chart(domain_chart(team_mix), width="stretch",
                            alt=f"Team {team['team_id']} strengths by domain")
            st.caption(f"Strengths data for {team['people_with_strengths']} people "
                       f"({team['coverage']:.0%} of the team).")
        with plan_col:
            if isinstance(team.get("focus_label"), str):
                st.markdown(f"**Focus question:** {team['focus_label']} ({team['focus_item']}).")
                st.markdown(f"**Strongest domain:** {team['lead_domain']}: the team "
                            f"{DOMAIN_MEANING[team['lead_domain']]}.")
                st.success(f"**Try this:** {team['play']}", icon=":material/lightbulb:")
            st.markdown("**Most common themes in this team**")
            st.markdown("\n".join(f"- {line}" for line in explain_themes(team["top_themes"])))

        how_to_read(
            "Each person's Top 5 themes are grouped into four domains. A team's strongest "
            "domain is where most of its Top 5 themes fall. The suggested action tackles the "
            "team's focus question (its weakest verified question, from the Teams page) in a "
            "way that suits that domain. Only teams with 5+ people with strengths data are "
            "shown; individual themes are never displayed.",
            "How the plans are built",
        )


# PROCESS 6 – PILOT: DID STRENGTHS COACHING WORK?

with st.container(border=True):

    title_col, measure_col = st.columns([1.5, 2], vertical_alignment="bottom")
    title_col.subheader("Did strengths coaching work?", anchor=False)
    measure = measure_col.segmented_control(
        "Measure", options=["Q03", "engagement"], default="Q03", key="pilot_measure",
        format_func={"Q03": "Doing what I do best (Q03)", "engagement": "Overall engagement"}.get,
        label_visibility="collapsed") or "Q03"

    st.caption("Compares how much coached teams changed between the last two surveys with how "
               "much other teams changed (a fair before-and-after comparison).")

    pilot = pilot_effect(df, strengths, measure)
    if pilot["status"] == "no_pilot":
        st.info(pilot["text"], icon=":material/info:")
    else:
        low, high = pilot["range"]
        tiles = [("Difference", f"{pilot['effect']:+.2f} points",
                  f"likely {low:+.2f} to {high:+.2f}"),
                 ("Coached teams", f"{pilot['coached_teams']}", "compared"),
                 ("Other teams", f"{pilot['other_teams']}", "compared")]
        for (label, value, note), column in zip(tiles, st.columns(3)):
            column.metric(label, value, delta=note, delta_color="off", delta_arrow="off",
                          border=True)
        (st.success if pilot["status"] == "clear" and pilot["effect"] > 0 else st.info)(
            pilot["text"], icon=":material/insights:")

st.caption("CliftonStrengths and its theme names are trademarks of Gallup. This app is "
           "independent and not endorsed by Gallup.")
