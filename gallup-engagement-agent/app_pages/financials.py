# PROCESS 1 – IMPORTS

import altair as alt
import streamlit as st

from src.tools.financial import (
    Assumptions,
    current_costs,
    department_costs,
    excess_cost_chart,
    explain_where_money,
    md,
    money,
    team_excess_cost,
)
from src.tools.insights import analysis_frame
from src.tools.priorities import (
    PRIORITY_MEANING,
    explain_priorities,
    factor_value,
    factor_value_range,
    priority_chart,
    priority_matrix,
    recommended_program,
)
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    how_to_read,
    page_header,
    require_data,
    story_box,
    trap_checks,
)


# PROCESS 2 – HEADER

page_header(
    "Financial impact",
    ":material/payments:",
    "What turnover and absence cost today, and what engagement improvements backed by "
    "your data could be worth.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info(
        "Financial impact needs business outcomes (turnover, absence or performance) "
        "in the uploaded data.",
        icon=":material/info:",
    )
    st.stop()

sample = len(analysis_frame(df))
assumptions = st.session_state.get("assumptions") or Assumptions(headcount=sample)
c = assumptions.currency


# PROCESS 3 – ASSUMPTIONS: ONE LINE + EDIT POPOVER
#
# Changes apply at once (no Apply button), so the absence-day cost follows the
# salary while you type. Every page reads the same st.session_state["assumptions"].

_, edit_col = st.columns([4, 1], vertical_alignment="center")

with edit_col.popover("Edit assumptions", icon=":material/edit:", width="stretch"):

    avg_salary = st.number_input(
        f"Average salary per year ({c})", min_value=0.0,
        value=float(assumptions.avg_salary), step=1000.0, format="%.0f", key="as_salary")
    st.caption(md(st.session_state.get("salary_source")
                  or "Placeholder: replace it with your organization's average salary."))

    replacement_pct = st.slider(
        "Cost to replace one leaver (% of salary)", min_value=10, max_value=200,
        value=int(round(assumptions.replacement_cost_pct * 100)), step=5, key="as_replace",
        help="Recruiting, onboarding and lost output. Studies often put it at 50–200%.")

    # The slider is rebuilt whenever the salary changes (its key includes the
    # salary), so its default and its range move with the salary at once.
    auto_day = max(1, int(round(avg_salary / 260)))
    same_salary = abs(avg_salary - assumptions.avg_salary) < 0.5
    day_cost = st.slider(
        f"Cost of one unplanned absence day ({c})",
        min_value=1, max_value=max(1000, auto_day * 3),
        value=int(round(assumptions.day_cost)) if same_salary else auto_day,
        step=1, key=f"as_day_{int(avg_salary)}",
        help="Use the arrow keys to move it one dollar at a time.")
    st.caption(md(f"Default: average salary ÷ 260 workdays = {money(auto_day, c)} a day."))

    point_value = st.number_input(
        "Value of +1 performance point, per person per year (0 = leave out)",
        min_value=0.0, value=float(assumptions.value_per_rating_point),
        step=500.0, format="%.0f", key="as_point")
    headcount = st.number_input(
        "Company headcount", min_value=1,
        value=int(assumptions.headcount or sample), step=100, key="as_headcount",
        help=f"The survey sample has {sample:,} people; results scale to this number.")
    currency = st.selectbox(
        "Currency symbol", options=["$", "€", "£", "Rp", "¥"],
        index=["$", "€", "£", "Rp", "¥"].index(assumptions.currency), key="as_currency")

    updated = Assumptions(
        avg_salary=avg_salary,
        replacement_cost_pct=replacement_pct / 100,
        absence_day_cost=None if day_cost == auto_day else float(day_cost),
        value_per_rating_point=point_value,
        headcount=int(headcount),
        currency=currency,
    )
    if updated != assumptions:
        st.session_state["assumptions"] = assumptions = updated
        c = assumptions.currency



# PROCESS 4 – WHAT IT COSTS TODAY

costs = current_costs(df, assumptions)

st.subheader("What it costs today", anchor=False)

today = [("People cost at stake", money(costs["total_cost"], c),
          "Voluntary turnover + unplanned absence, per year.")]
if "turnover_cost" in costs:
    today.append(("Voluntary turnover", money(costs["turnover_cost"], c),
                  md(f"About {costs['leavers']:,.0f} leavers a year × "
                     f"{money(assumptions.cost_per_leaver, c)} each.")))
if "absence_cost" in costs:
    today.append(("Unplanned absence", money(costs["absence_cost"], c),
                  md(f"About {costs['absence_days']:,.0f} days a year × "
                     f"{money(assumptions.day_cost, c)} a day.")))

for (label, value, help_text), column in zip(today, st.columns(len(today))):
    column.metric(label, value, help=help_text, border=True)

# Which department carries the biggest share of each cost.
by_dept = department_costs(df, assumptions)

if not by_dept.empty and by_dept["total_cost"].sum() > 0:

    with st.container(border=True):

        st.markdown("**Which department carries the biggest share of the cost**")

        # One bar per department, largest first: turnover and absence side by side
        # in the same bar, the total and its share at the end.
        order = list(by_dept.sort_values("total_cost", ascending=False)["department"])
        parts = [(col, name) for col, name in (("turnover_cost", "Voluntary turnover"),
                                                ("absence_cost", "Unplanned absence"))
                 if by_dept[col].sum() > 0]
        long = by_dept.melt(id_vars="department", value_vars=[col for col, _ in parts],
                            var_name="kind", value_name="value")
        long["kind"] = long["kind"].map(dict(parts))
        long["label"] = long["value"].map(lambda v: money(v, c))
        totals = by_dept[["department", "total_cost"]].copy()
        totals["label"] = [f"{money(v, c)} · {v / totals['total_cost'].sum():.0%}"
                           for v in totals["total_cost"]]
        top = totals.sort_values("total_cost", ascending=False).iloc[0]
        st.caption(md(f"Largest: **{top['department']}**, {top['label'].split(' · ')[1]} of the "
                      f"total ({money(top['total_cost'], c)} a year)."))

        y = alt.Y("department:N", sort=order, title=None, axis=alt.Axis(labelLimit=200))
        bars = alt.Chart(long).mark_bar(height=18).encode(
            y=y,
            x=alt.X("sum(value):Q", title=f"Cost per year ({c})", axis=alt.Axis(format="~s"),
                    scale=alt.Scale(domain=[0, float(totals["total_cost"].max()) * 1.25])),
            color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=[name for _, name in parts],
                                            range=["#8C2A3A", "#C9A227"][:len(parts)])),
            order=alt.Order("kind:N"),
            tooltip=[alt.Tooltip("department:N", title="Department"),
                     alt.Tooltip("kind:N", title="Cost"),
                     alt.Tooltip("label:N", title="Per year")])
        labels = alt.Chart(totals).mark_text(align="left", dx=6, fontWeight="bold").encode(
            y=y, x="total_cost:Q", text="label:N")
        st.altair_chart((bars + labels).properties(height=alt.Step(30)), width="stretch",
                        alt="Turnover and absence cost by department")


# PROCESS 5 – THE RECOMMENDED PROGRAM: WHAT IT IS WORTH
#
# One program, valued once, with a likely range. The same figure is used on the
# Executive summary, in the action story and report pack, and as the starting
# point of the Scenarios tab (priorities.recommended_program).

analysis = analysis_runner(df, key="run_analysis_financials")

if analysis is not None:

    # Every priced question, in the same order as "Where to act" on the Strategy
    # page. #1 is the recommended program; the others can be picked to compare.
    ranked = factor_value(df, analysis, assumptions, trap_checks(df))
    ranked = ranked[ranked["yearly_value"] > 0].reset_index(drop=True)
    ranges = factor_value_range(df, analysis, assumptions, trap_checks(df))

    with st.container(border=True):

        if ranked.empty:
            st.subheader("Fix first", anchor=False)
            st.caption("No verified driver is available to value yet.")
        else:
            # The question picker sits inside the title:
            #   If we fix [Q04 Recognition ▾], what is it worth?
            # Options follow the "Where to act" order on the Strategy page.
            title_style = "font-size:1.2rem;font-weight:600;white-space:nowrap"
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"<span style='{title_style}'>If we fix</span>",
                            unsafe_allow_html=True, width="content")
                pick = st.selectbox(
                    "Question to fix", list(ranked.index), key="fin_fix_pick",
                    format_func=lambda i: f"{ranked.loc[i, 'item']} {ranked.loc[i, 'label']}",
                    label_visibility="collapsed", width=260)
                st.markdown(f"<span style='{title_style}'>, what is it worth?</span>",
                            unsafe_allow_html=True, width="content")
            row = ranked.loc[pick]
            value = float(row["yearly_value"])
            low, high = ranges.get(row["item"], (value, value))

            if pick == 0:
                lead = (f"Of the 12 engagement questions, improving **{row['label']}** is worth "
                        f"the most: if the {row['people_below']:,.0f} employees who score it "
                        "1–3 move up one point.")
            else:
                lead = (f"**{row['label']}** ranks #{pick + 1} of the questions with a verified "
                        f"link ({row['priority']}): this is its value if the "
                        f"{row['people_below']:,.0f} employees who score it 1–3 move up one "
                        "point.")
            st.markdown(md(lead))

            share = value / costs["total_cost"] if costs["total_cost"] else 0
            tiles = [
                ("We can save per year", money(value, c),
                 md(f"{money(low, c)}–{money(high, c)} likely")),
                ("Cuts today's people cost by", f"{share:.0%}",
                 md(f"of {money(costs['total_cost'], c)} a year")),
            ]
            if row["avoided_leavers"] > 0:
                tiles.append(("Since fewer people leave each year",
                              f"{row['avoided_leavers']:,.0f}",
                              md(f"× {money(assumptions.cost_per_leaver, c)} each")))
            if row["avoided_absence_days"] > 0:
                tiles.append(("Fewer absence days a year", f"{row['avoided_absence_days']:,.0f}",
                              md(f"× {money(assumptions.day_cost, c)} a day")))
            if row["performance_value"] > 0:
                tiles.append(("From higher performance", money(row["performance_value"], c),
                              f"{row['rating_points']:,.0f} rating points"))

            for (label, val, detail), column in zip(tiles, st.columns(len(tiles))):
                column.metric(label, val, delta=detail, delta_color="off",
                              delta_arrow="off", border=True)



# PROCESS 6 – ALL TEAMS (SEARCH BY TEAM OR DEPARTMENT)

teams = team_excess_cost(df, assumptions)

if not teams.empty:

    with st.container(border=True):

        st.subheader("All teams", anchor=False)
        st.caption("Each team's extra yearly cost: what its turnover and absence cost above "
                   "what the company rate would give for a team of its size. Descriptive "
                   "only: it shows where the extra cost is, not why.")

        search_col, dept_col = st.columns([2, 2], vertical_alignment="bottom")
        search = search_col.text_input(
            "Search a team or department", placeholder="e.g. T002530 or Sales",
            key="fin_team_search", icon=":material/search:")
        departments = sorted(teams["department"].dropna().unique()) \
            if "department" in teams.columns else []
        chosen = dept_col.multiselect("Departments", departments, key="fin_team_depts",
                                      placeholder="All departments")

        shown = teams
        if chosen:
            shown = shown[shown["department"].isin(chosen)]
        if search.strip():
            text = search.strip().lower()
            match = shown["team_id"].astype(str).str.lower().str.contains(text, regex=False)
            if "department" in shown.columns:
                match |= shown["department"].astype(str).str.lower().str.contains(text,
                                                                                 regex=False)
            shown = shown[match]

        st.caption(f"{len(shown):,} of {len(teams):,} teams")
        if shown.empty:
            st.info("No team matches this search.", icon=":material/search_off:")
        else:
            st.dataframe(
                shown.round({col: 0 for col in shown.columns if "cost" in col}),
                hide_index=True,
                width="stretch",
                column_config={
                    "team_id": "Team",
                    "department": "Department",
                    "people": st.column_config.NumberColumn("People", format="%d"),
                    "turnover_rate": st.column_config.NumberColumn("Turnover", format="percent"),
                    "absence_days": st.column_config.NumberColumn(
                        "Absence days (6 mo.)", format="%.1f"),
                    "excess_turnover_cost": st.column_config.NumberColumn(
                        f"Extra turnover cost ({c})", format="compact"),
                    "excess_absence_cost": st.column_config.NumberColumn(
                        f"Extra absence cost ({c})", format="compact"),
                    "excess_cost": st.column_config.NumberColumn(
                        f"Extra cost, total ({c})", format="compact"),
                },
                alt="Extra yearly turnover and absence cost for every team",
            )
