# PROCESS 1 – IMPORTS

import pandas as pd
import streamlit as st

from src.tools.financial import Assumptions, md, money
from src.tools.insights import ITEM_LABELS, analysis_frame
from src.tools.priorities import factor_value
from src.tools.scenario import (
    WHO,
    contributions,
    explain_scenario,
    path_chart,
    simulate,
    verified_links,
)
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    how_to_read,
    page_header,
    require_data,
    story_box,
)


# PROCESS 2 – HEADER AND DATA

page_header(
    "Scenarios",
    ":material/tune:",
    "What if some engagement questions improve? See the expected change in leavers, "
    "absence, performance and money, quarter by quarter.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info("Scenarios need business outcomes (turnover, absence or performance) in the "
            "uploaded data.", icon=":material/info:")
    st.stop()

analysis = analysis_runner(df, key="run_analysis_scenarios")
if analysis is None:
    st.stop()

assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))
c = assumptions.currency
links = verified_links(analysis)
verified_items = list(dict.fromkeys(links["item"]))


# PROCESS 3 – CHOOSE THE SCENARIO

def item_name(item: str) -> str:
    mark = " ✓" if item in verified_items else ""
    return f"{item} · {ITEM_LABELS[item]}{mark}"


factors = factor_value(df, analysis, assumptions)
default_items = list(factors["item"].head(1)) or verified_items[:1]
departments = sorted(analysis_frame(df)["department"].dropna().unique()) \
    if "department" in df.columns else []

with st.container(border=True):

    st.markdown("**:material/tune: Build a scenario**")
    st.caption("✓ = the question has a verified link to at least one outcome. Questions "
               "without one can be chosen but count as zero.")

    items = st.multiselect("Questions that improve", options=list(ITEM_LABELS),
                           default=default_items, format_func=item_name, max_selections=4,
                           key="scenario_items")

    lifts = {}
    if items:
        for item, column in zip(items, st.columns(len(items))):
            lifts[item] = column.slider(
                f"{ITEM_LABELS[item]}: raise by", min_value=0.1, max_value=1.0, value=0.3,
                step=0.1, key=f"lift_{item}",
                help="Points on the 1–5 answer scale. 0.3 is a solid improvement for one "
                "year; 1.0 is a big change.")

    who_col, where_col, ramp_col, horizon_col = st.columns(4)
    who = who_col.segmented_control("Who improves", options=list(WHO), default="everyone",
                                    format_func=lambda w: "Everyone" if w == "everyone"
                                    else "Low scorers (1–3)", key="scenario_who") or "everyone"
    where = where_col.selectbox("Where", ["Whole company"] + departments, key="scenario_where")
    ramp = ramp_col.slider("Months to build up", 0, 12, 6, key="scenario_ramp",
                           help="How long until the improvement is fully in place.")
    quarters = horizon_col.segmented_control("Show", options=[4, 8], default=4,
                                             format_func=lambda q: f"{q} quarters",
                                             key="scenario_quarters") or 4

if not lifts:
    st.info("Pick at least one question to see a scenario.", icon=":material/touch_app:")
    st.stop()

result = simulate(df, analysis, lifts, assumptions, who,
                  None if where == "Whole company" else where, ramp, quarters)
totals = result["totals"]["estimate"]
low, high = result["value_range"]


# PROCESS 4 – RESULTS

st.subheader("What the models expect", anchor=False)

cautious, optimistic = result["totals"]["cautious"], result["totals"]["optimistic"]
tiles = []
if totals["leavers"] > 0:
    tiles.append(("Fewer leavers a year", f"{totals['leavers']:,.0f}",
                  f"{cautious['leavers']:,.0f}–{optimistic['leavers']:,.0f} likely"))
if totals["absence_days"] > 0:
    tiles.append(("Fewer absence days", f"{totals['absence_days']:,.0f}",
                  f"{cautious['absence_days']:,.0f}–{optimistic['absence_days']:,.0f} likely"))
if totals["rating_points"] > 0:
    tiles.append(("Performance rating", f"{totals['rating_points']:+.2f}", "points on average"))
tiles += [
    ("Yearly value", money(totals["value"], c), md(f"{money(low, c)}–{money(high, c)} likely")),
    ("First year" if quarters == 4 else "First two years",
     money(float(result["path"]["value"].sum()), c), f"{ramp}-month build-up"),
]
for (label, value, note), column in zip(tiles, st.columns(len(tiles))):
    column.metric(label, value, delta=note, delta_color="off", delta_arrow="off", border=True)

chart_col, story_col = st.columns([3, 2], gap="large")

with chart_col:
    st.markdown("**Value building up, quarter by quarter**")
    st.altair_chart(path_chart(result["path"], c), width="stretch",
                    alt="Running total of the scenario's value by quarter, with likely range")
    st.caption("Line = expected running total; shaded band = likely range.")

with story_col:
    story = explain_scenario(result, assumptions)
    story_box({"headline": md(story["headline"]), "points": [md(p) for p in story["points"]]})

if len(lifts) > 1:
    with st.expander("Each question on its own", icon=":material/stacked_bar_chart:"):
        parts = contributions(df, analysis, lifts, assumptions, who,
                              None if where == "Whole company" else where)
        st.dataframe(parts.drop(columns=["item"]), hide_index=True, width="stretch",
                     column_config={
                         "question": "Question", "lift": st.column_config.NumberColumn(
                             "Raise by", format="+%.1f"),
                         "leavers_avoided": st.column_config.NumberColumn(
                             "Fewer leavers", format="%.0f"),
                         "absence_days_avoided": st.column_config.NumberColumn(
                             "Fewer absence days", format="%.0f"),
                         "rating_points": st.column_config.NumberColumn(
                             "Rating change", format="%+.2f"),
                         "yearly_value": st.column_config.NumberColumn(
                             "Yearly value", format="compact"),
                         "verified_for": "Verified link to"})
        st.caption("Questions overlap (the same people answer all of them), so these do not "
                   "add up exactly to the combined scenario.")

how_to_read(
    "Each employee's turnover risk, absence and rating are predicted twice with the app's "
    "verified models: with today's answers and with the improved ones. The difference, scaled "
    "to your headcount, is the effect. Only links that passed every check count; everything "
    "else (tenure, department, other questions) stays as it is today. The likely range uses "
    "each link's margin of error. Numbers can differ a little from the Financials page, "
    "which uses the company's average turnover rate; this page uses each person's own risk. "
    "These are associations in your data, not guarantees.",
    "How these scenarios work",
)


# PROCESS 5 – COMPARE SCENARIOS

saved = st.session_state.setdefault("saved_scenarios", [])

with st.container(border=True):

    title_col, button_col = st.columns([3, 1], vertical_alignment="center")
    title_col.markdown("**:material/compare_arrows: Compare scenarios**")

    if button_col.button("Save this scenario", icon=":material/bookmark_add:", width="stretch"):
        saved.append({
            "Scenario": ", ".join(f"{ITEM_LABELS[i]} +{v:.1f}" for i, v in lifts.items()),
            "Who": "Everyone" if who == "everyone" else "People answering 1–3",
            "Where": where,
            "Build-up": f"{ramp} months",
            "Fewer leavers a year": round(totals["leavers"]),
            "Fewer absence days": round(totals["absence_days"]),
            "Yearly value": money(totals["value"], c),
            "Likely range": f"{money(low, c)}–{money(high, c)}",
            "First year": money(float(result["path"]["value"].head(4).sum()), c),
        })

    if saved:
        table = pd.DataFrame(saved)
        st.dataframe(table, hide_index=True, width="stretch")
        with st.container(horizontal=True):
            st.download_button("Download comparison (CSV)", data=table.to_csv(index=False).encode(),
                               file_name="engagement_scenarios.csv", mime="text/csv",
                               icon=":material/download:")
            if st.button("Clear", icon=":material/delete:"):
                st.session_state["saved_scenarios"] = []
                st.rerun()
    else:
        st.caption("Save a few scenarios to compare them side by side, e.g. a small lift for "
                   "everyone vs a big lift for low scorers only.")
