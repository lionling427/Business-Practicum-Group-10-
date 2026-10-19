# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.action_tools import (
    DISCLAIMER,
    action_plan_rows,
    links_markdown,
    resources_for,
)
from src.tools.analytics import analyze_dataframe
from src.tools.insights import ITEM_LABELS
from src.tools.longitudinal import analyze_change, survey_waves
from src.tools.prediction import TARGETS, team_risk
from src.tools.recommendations import (
    PERFORMANCE,
    TIER_ORDER,
    company_summary,
    productivity_levers,
    starting_points_chart,
    team_gap_chart,
    team_plan,
    team_plans,
)
from src.tools.waves import model_sample
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    how_to_read,
    page_header,
    require_data,
    story_box,
)


# PROCESS 2 – HEADER

page_header(
    "Recommendations",
    ":material/tips_and_updates:",
    "What each team's leader can do next to lift productivity (performance rating), "
    "based on the questions verified to go with higher performance.",
)

df, validation = require_data()

if PERFORMANCE not in df.columns or validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info(
        "Recommendations need performance ratings (`performance_rating`) in the uploaded data.",
        icon=":material/info:",
    )
    st.stop()


# PROCESS 3 – EVIDENCE (RUN ANY MISSING ANALYSIS)

analysis = analysis_runner(df, key="run_analysis_recommendations")
if analysis is None:
    st.stop()

change_analysis = st.session_state.get("change_analysis")
if change_analysis is None and len(survey_waves(df)) >= 2:
    with st.spinner("Comparing people with themselves over time..."):
        try:
            change_analysis = analyze_change(df)
            st.session_state["change_analysis"] = change_analysis
        except ValueError:
            change_analysis = None          # single-survey evidence only

levers = productivity_levers(df, analysis, change_analysis)

if levers.empty:
    st.info(
        "No engagement question has a verified link to performance in this data, so "
        "there is nothing evidence-based to recommend yet.",
        icon=":material/info:",
    )
    st.stop()

risk_results = {}
for outcome, (column, _) in TARGETS.items():
    if column in df.columns:
        try:
            risk_results[outcome] = team_risk(
                df, outcome, health_report=st.session_state.get("health_report"))
        except ValueError:
            pass

plans = team_plans(df, levers, risk_results)
summary = company_summary(plans, levers)


# PROCESS 4 – TILES

strongest = levers.iloc[0]
top_start = summary["starts"].iloc[0] if not summary["starts"].empty else None

tiles = [
    ("Biggest opportunity", f"{int((plans['tier'] == TIER_ORDER[0]).sum())} teams", None,
     "Top 20% of teams by expected performance gain from their first step."),
    ("Most common first step", top_start["item"] if top_start is not None else "—",
     f"{top_start['label']} · {int(top_start['teams'])} teams" if top_start is not None else None,
     f"{top_start['label']}: the question most teams should start with."
     if top_start is not None else None),
    ("Strongest lever", strongest["item"],
     f"{strongest['label']} · {strongest['points_per_point']:+.2f} pts per point",
     f"{strongest['label']}: rating points per 1-point higher score "
     f"({strongest['evidence'].lower()} evidence)."),
    ("Already strong", f"{int((plans['tier'] == TIER_ORDER[2]).sum())} teams", None,
     "At or above the company on every productivity lever: protect and share what works."),
]

for (label, value, delta, help_text), column in zip(tiles, st.columns(len(tiles))):
    column.metric(label, value, delta=delta, delta_color="off", delta_arrow="off",
                  help=help_text, border=True)


# PROCESS 5 – WHERE TO START

with st.container(border=True):

    st.subheader("Where to start", anchor=False)

    chart_col, story_col = st.columns([3, 2], gap="large")

    with chart_col:
        if not summary["starts"].empty:
            st.altair_chart(
                starting_points_chart(summary["starts"]),
                width="stretch",
                alt="Number of teams that should start with each engagement question",
            )

    with story_col:
        story_box(summary)

    how_to_read(
        "Each bar is an engagement question verified to go with higher performance, and "
        "how many teams should start there because that is where they have the most room "
        "to improve. Dark green = strong evidence (two analyses agree); light green = good "
        "evidence (one analysis)."
    )


# PROCESS 6 – PLAN FOR EACH TEAM

with st.container(border=True):

    st.subheader("Plan for each team", anchor=False)

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1.4, 1], vertical_alignment="bottom")

    departments = sorted(plans["department"].dropna().unique()) if "department" in plans else []
    department = filter_col1.selectbox(
        "Department",
        options=[None] + departments,
        format_func=lambda d: "All departments" if d is None else d,
        key="rec_department",
    )
    tiers = filter_col2.pills(
        "Opportunity",
        options=TIER_ORDER,
        selection_mode="multi",
        default=TIER_ORDER[:2],
        key="rec_tiers",
    )
    below_only = filter_col3.toggle(
        "Below company performance only",
        key="rec_below_only",
        help="Show only teams whose performance rating is below the company average.",
    )

    shown = plans
    if department is not None:
        shown = shown[shown["department"] == department]
    if tiers:
        shown = shown[shown["tier"].isin(tiers)]
    if below_only:
        shown = shown[shown["performance"] < shown["company_performance"]]

    if shown.empty:
        st.info("No teams match these filters.")
        st.stop()

    st.caption(":material/touch_app: Click a row to open that team's plan below.")

    table = shown.assign(
        start_with=shown["start_with"].map(
            lambda i: f"{i} · {ITEM_LABELS.get(i, i)}" if isinstance(i, str) else "Keep strengths"),
        risk=shown["high_risk"].map(
            lambda r: "⚠ Both" if "&" in r else f"⚠ {r.capitalize()}" if r else ""),
    )

    event = st.dataframe(
        table[["team_id", "department", "people", "performance", "start_with",
               "estimated_gain", "tier", "risk"]],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="plans_table",
        column_config={
            "team_id": "Team",
            "department": "Department",
            "people": st.column_config.NumberColumn("People", format="%d"),
            "performance": st.column_config.NumberColumn("Performance", format="%.2f"),
            "start_with": st.column_config.TextColumn("Start with", width="medium"),
            "estimated_gain": st.column_config.ProgressColumn(
                "Expected gain", min_value=0,
                max_value=max(0.01, float(plans["estimated_gain"].max())), format="%+.2f"),
            "tier": "Opportunity",
            "risk": "High risk",
        },
        alt="Recommended first step and expected gain for each team",
    )

    with st.container(horizontal=True):
        st.download_button(
            "Download all team plans (CSV)",
            data=plans.to_csv(index=False).encode(),
            file_name="team_productivity_plans.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Download action plans for Gallup Access (CSV)",
            data=action_plan_rows(plans).to_csv(index=False).encode(),
            file_name="gallup_action_plans.csv",
            mime="text/csv",
            icon=":material/assignment:",
            help="One row per step with goal, owner, due date, success indicator and "
            "follow-up date: the fields of a Gallup Access action plan, ready to copy in.",
        )


# PROCESS 7 – THE SELECTED TEAM'S PLAN

rows = event.selection.rows
selected = shown.iloc[rows[0]]["team_id"] if rows else shown.iloc[0]["team_id"]
plan_row = plans.set_index("team_id").loc[selected].copy()
plan_row["team_id"] = selected
plan = team_plan(plan_row)

tier_badge = {
    TIER_ORDER[0]: ":red-badge[Biggest opportunity]",
    TIER_ORDER[1]: ":orange-badge[Solid opportunity]",
    TIER_ORDER[2]: ":green-badge[Keep strengths]",
}[plan_row["tier"]]
risk_badge = (f" :orange-badge[:material/warning: High {plan_row['high_risk']} risk]"
              if plan_row["high_risk"] else "")

with st.container(border=True):

    st.subheader(f"Plan · Team {selected}", anchor=False)
    st.markdown(f"{plan_row['department']} · {plan_row['people']} people  {tier_badge}{risk_badge}")

    plan_col, chart_col = st.columns([1, 1], gap="large")

    with plan_col:

        if plan_row["start_with"]:
            gain = float(plan_row["estimated_gain"])
            st.metric(
                "Expected performance if the gap closes",
                f"{plan_row['performance']:.2f} → {plan_row['performance'] + gain:.2f}",
                delta=f"{gain:+.2f} rating points",
                delta_arrow="off",
                help="If the team reaches the company average on its first question. "
                "An estimate from links in your data, not a guarantee.",
            )

        st.markdown(f"**{plan['headline']}**")
        st.markdown(plan["why"])

        st.markdown("\n".join(f"- **{when}:** {step}" for when, step in plan["steps"]))

        if plan_row["high_risk"]:
            st.caption(":material/groups: This team also has a risk plan on the **Teams** page.")

    with chart_col:

        st.markdown("**Team vs company on each productivity question**")
        st.altair_chart(
            team_gap_chart(plan_row, df, levers),
            width="stretch",
            alt=f"Team {selected} vs company on each productivity question",
        )
        st.caption("Dot = this team (red = where to start); gray tick = company average. "
                   "Biggest gaps at the top.")

    st.caption(plan["caution"])

    with st.expander("Take it to Gallup action planning", icon=":material/assignment:"):

        focus_item = plan_row["start_with"] or plan_row["keep_strength"]
        team_actions = action_plan_rows(plans[plans["team_id"] == selected])

        if not team_actions.empty:
            first = team_actions.iloc[0]
            st.markdown(f"**Goal:** {first['goal']}  \n**Success indicator:** "
                        f"{first['success_indicator']}  \n**Follow up by:** {first['follow_up']}")
            st.dataframe(
                team_actions[["step", "next_step", "owner", "due_date"]],
                hide_index=True, width="stretch",
                column_config={"step": "When", "next_step": "Next step", "owner": "Owner",
                               "due_date": "Due"},
            )
            st.download_button(
                f"Download Team {selected}'s action plan (CSV)",
                data=team_actions.to_csv(index=False).encode(),
                file_name=f"action_plan_team_{selected}.csv",
                mime="text/csv",
                icon=":material/download:",
                key="team_action_plan",
            )

        st.markdown(links_markdown(resources_for(focus_item)))
        st.caption(DISCLAIMER)
