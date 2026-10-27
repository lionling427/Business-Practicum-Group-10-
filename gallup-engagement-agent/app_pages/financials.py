# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.financial_updated import (
    SOURCE_OPTIONS,
    Assumptions,
    assumption_register,
    assumption_sensitivity,
    assumptions_line,
    break_even_lift,
    break_even_per_person,
    calculation_receipts,
    current_costs,
    default_assumptions,
    department_excess_cost,
    engagement_value,
    excess_cost_chart,
    explain_departments,
    explain_where_money,
    exposure_vs_opportunity,
    gallup_benchmark_check,
    md,
    money,
    money_warning,
    outcome_frame,
    program_economics,
    range_text,
    receipt_markdown,
    team_cost_persistence,
    team_excess_cost,
)
from src.tools.priorities import (
    PRIORITY_MEANING,
    explain_priorities,
    factor_value,
    priority_chart,
    priority_matrix,
)
from src.tools.recommendations import (
    PERFORMANCE,
    TIER_ORDER,
    productivity_levers,
    team_plans,
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

sample = len(outcome_frame(df))           # every employee in the outcome file, not only respondents
assumptions = st.session_state.get("assumptions") or default_assumptions(df)
c = assumptions.currency


# PROCESS 3 – ASSUMPTIONS: ONE LINE + EDIT POPOVER

line_col, edit_col = st.columns([4, 1], vertical_alignment="center")

line_col.caption(
    f":material/tune: **Using:** {assumptions_line(assumptions, sample)}. "
    "Defaults are placeholders; replace them with your organization's numbers."
)

with edit_col.popover("Edit assumptions", icon=":material/edit:", width="stretch"):

    with st.form("assumptions_form", border=False):

        avg_salary = st.number_input(
            "Average salary per year", min_value=0.0,
            value=float(assumptions.avg_salary), step=1000.0, format="%.0f")
        replacement_pct = st.slider(
            "Cost to replace one leaver (% of salary)", min_value=10, max_value=200,
            value=int(round(assumptions.replacement_cost_pct * 100)), step=5,
            help="Recruiting, onboarding and lost output. Studies often put it at 50–200%.")
        day_cost = st.number_input(
            "Cost of one unplanned absence day (0 = salary ÷ 260)", min_value=0.0,
            value=float(assumptions.absence_day_cost or 0.0), step=10.0, format="%.0f")
        point_value = st.number_input(
            "Value of +1 performance point, per person per year (0 = leave out)",
            min_value=0.0, value=float(assumptions.value_per_rating_point),
            step=500.0, format="%.0f")
        headcount = st.number_input(
            "Company headcount", min_value=1,
            value=int(assumptions.headcount or sample), step=100,
            help=f"The data has {sample:,} people; results scale to this number.")
        sd_pct = st.number_input(
            "SD of productivity value (% of salary, 0 = off)", min_value=0.0, max_value=200.0,
            value=float(assumptions.sd_productivity_pct * 100), step=5.0, format="%.0f",
            help="Prices a performance point with the utility-analysis formula (the method "
                 "Gallup cites for productivity): value of a point = this share of salary ÷ the "
                 "spread of ratings. Ignored if you enter a value per point above. A rule of "
                 "thumb often cited in the utility-analysis literature is about 40%; check it "
                 "with your finance team before relying on it.")
        program_cost = st.number_input(
            "Yearly cost of the program you are considering (0 = not entered)", min_value=0.0,
            value=float(assumptions.program_cost), step=10000.0, format="%.0f",
            help="Used for return on investment and break-even. It is your assumption, not "
                 "something the data can tell us.")
        realization = st.slider(
            "Share of the modeled effect the program delivers (%)", min_value=10, max_value=100,
            value=int(round(assumptions.effect_realization * 100)), step=5,
            help="100% assumes the program moves scores as far as the scenario. Lower it to "
                 "test a program that falls short.")
        currency = st.selectbox(
            "Currency symbol", options=["$", "€", "£", "Rp", "¥"],
            index=["$", "€", "£", "Rp", "¥"].index(assumptions.currency))
        source = st.selectbox(
            "Where did the numbers you changed come from?", options=SOURCE_OPTIONS,
            help="Recorded next to each changed number in the assumptions list.")

        if st.form_submit_button("Apply", type="primary", icon=":material/check:",
                                 width="stretch"):
            new = Assumptions(
                avg_salary=avg_salary,
                replacement_cost_pct=replacement_pct / 100,
                absence_day_cost=day_cost or None,
                value_per_rating_point=point_value,
                headcount=int(headcount),
                currency=currency,
                program_cost=program_cost,
                effect_realization=realization / 100,
                sd_productivity_pct=sd_pct / 100,
            )
            placeholder = Assumptions()
            labels = {"avg_salary": new.avg_salary, "replacement_cost_pct": new.replacement_cost_pct,
                      "absence_day_cost": new.absence_day_cost,
                      "value_per_rating_point": new.value_per_rating_point,
                      "sd_productivity_pct": new.sd_productivity_pct,
                      "program_cost": new.program_cost,
                      "effect_realization": new.effect_realization}
            kept = assumptions.sources
            new.sources = {
                key: (source if value != getattr(assumptions, key) or key not in kept else kept[key])
                for key, value in labels.items() if value != getattr(placeholder, key)}
            if int(headcount) != sample:
                new.sources["headcount"] = (source if int(headcount) != assumptions.headcount
                                            else kept.get("headcount", source))
            st.session_state["assumptions"] = new
            st.rerun()


# PROCESS 3B – SURVEY-RESPONSE NOTE (leavers answer less often, so savings are understated)

response_note = money_warning(df, st.session_state.get("health_report"))
if response_note:
    st.warning(response_note, icon=":material/info:")


# PROCESS 4 – WHAT IT COSTS TODAY

costs = current_costs(df, assumptions)

st.subheader("What it costs today", anchor=False)

today = [("People cost at stake", money(costs["total_cost"], c),
          "Voluntary turnover + unplanned absence, per year.")]
if "turnover_cost" in costs:
    today.append(("Voluntary turnover", money(costs["turnover_cost"], c),
                  md(f"About {costs['leavers']:,.0f} leavers a year (all employees) × "
                     f"{money(assumptions.cost_per_leaver, c)} each.")))
if "absence_cost" in costs:
    today.append(("Unplanned absence", money(costs["absence_cost"], c),
                  md(f"About {costs['absence_days']:,.0f} days a year × "
                     f"{money(assumptions.day_cost, c)} a day.")))

for (label, value, help_text), column in zip(today, st.columns(len(today))):
    column.metric(label, value, help=help_text, border=True)


# PROCESS 5 – WHAT ENGAGEMENT COULD BE WORTH

analysis = analysis_runner(df, key="run_analysis_financials")

# One engine (the Scenarios simulation) prices every factor and the headline program.
factors = factor_value(df, analysis, assumptions, trap_checks(df)) if analysis is not None else None

if analysis is not None:

    plans = None
    if PERFORMANCE in df.columns:
        levers = productivity_levers(df, analysis, st.session_state.get("change_analysis"))
        if not levers.empty:
            plans = team_plans(df, levers)

    with st.container(border=True):

        title_col, tier_col = st.columns([2, 2], vertical_alignment="bottom")

        title_col.subheader("What engagement could be worth", anchor=False)

        tiers = TIER_ORDER[:1]
        if plans is not None and assumptions.value_per_rating_point > 0:
            tiers = tier_col.pills(
                "Productivity: teams that follow their plan",
                options=TIER_ORDER[:2],
                selection_mode="multi",
                default=TIER_ORDER[:1],
                key="fin_tiers",
            ) or TIER_ORDER[:1]

        value = engagement_value(df, analysis, assumptions, plans, tiers, factors=factors)
        scenarios = value["scenarios"]
        program = value["program"]

        if scenarios.empty:
            st.caption("No verified driver is available to value yet.")
        else:
            share = value["total"] / costs["total_cost"] if costs["total_cost"] else 0
            low, high = value["range"]
            tiles = [("Total, per year", money(value["total"], c),
                      f"{share:.0%} of today's people cost · {range_text(low, high, c)}")]
            tiles += [(row["source"], money(row["value"], c),
                       f"{row['units']:,.0f} {row['unit_label']}")
                      for _, row in scenarios.iterrows()]

            for (label, val, detail), column in zip(tiles, st.columns(len(tiles))):
                column.metric(label, val, delta=detail, delta_color="off",
                              delta_arrow="off", border=True)

            if program:
                st.markdown(md(
                    f"**The program:** everyone who answers 1–3 on **{program['names']}** moves "
                    f"up one point ({program['people']:,.0f} people), counted once even when a "
                    "person is helped by more than one question. The turnover and absence "
                    "figures above are this one program, not a sum of separate programs."))
                per_person = break_even_per_person(program["value"], program["people"])
                st.caption(md(
                    f":material/balance: Break-even: the program pays for itself if it costs "
                    f"less than about {money(per_person, c)} per person per year "
                    f"(low end of the range: "
                    f"{money(break_even_per_person(program['low'], program['people']), c)})."))
            st.markdown("\n".join(
                f"- **{row['source']}:** {md(row['what'])}."
                for _, row in scenarios.iterrows()))

            split = exposure_vs_opportunity(costs, value)
            st.caption(md(
                f":material/compare_arrows: Exposure vs opportunity: turnover and absence cost "
                f"{money(split['exposure'], c)} a year today (the exposure). The modeled "
                f"opportunity is {split['opportunity_share']:.0%} of that. An opportunity is not "
                "a realized return."))
            for warning in value["warnings"]:
                st.caption(f":material/warning: {warning}")

            economics = program_economics(value, assumptions) if program else None
            if economics:
                base = economics["cases"]["base"]
                low_case, high_case = economics["cases"]["downside"], economics["cases"]["upside"]
                roi_tiles = [
                    ("Net value, per year", money(base["net"], c),
                     f"{money(low_case['net'], c)} to {money(high_case['net'], c)}"),
                    ("Return on investment", f"{base['roi']:.0%}",
                     f"{low_case['roi']:.0%} to {high_case['roi']:.0%}"),
                    ("Payback", ("not reached" if base["payback_months"] == float("inf")
                                 else f"{base['payback_months']:.1f} mo."),
                     f"cost {money(economics['cost'], c)}/yr"),
                    ("Break-even cost", money(economics["break_even_cost"], c),
                     f"{money(economics['break_even_cost_per_person'], c)} per person"),
                ]
                for (label, val, detail), column in zip(roi_tiles, st.columns(len(roi_tiles))):
                    column.metric(label, val, delta=detail, delta_color="off",
                                  delta_arrow="off", border=True)
                lift = break_even_lift(df, analysis, assumptions, program["items"])
                need = (f"must move the people who answer 1–3 up by at least "
                        f"{lift['lift']:.2f} points on average" if lift["status"] == "ok" else
                        "cannot break even within a 3-point rise" if lift["status"] == "unreachable"
                        else "has no modeled effect to break even on")
                st.markdown(md(
                    f"**What would have to be true:** at {economics['realization']:.0%} of the "
                    f"modeled effect, the program {need}, or deliver at least "
                    f"{base['required_realization']:.0%} of the modeled effect. The data cannot "
                    "say whether a real program will do that."))
            elif program:
                st.caption(":material/savings: Enter a yearly program cost under **Edit assumptions** "
                           "to see return on investment, payback and break-even.")

        if plans is not None and assumptions.value_per_rating_point <= 0:
            st.caption(":material/info: Add a value for +1 performance point under **Edit "
                       "assumptions** to include the productivity plans from Recommendations.")

        how_to_read(
            "The headline is one named program, simulated with each person's own risk from "
            "the verified models (the same engine as the Scenarios page), so the two pages "
            "agree. The range is the 95% margin of error of those models. It covers sampling "
            "error only, and the survey-response note above says which way the real value is "
            "likely to differ. Productivity assumes the chosen teams close the gap on their "
            "first recommended question. Values are yearly once the change is complete. They "
            "describe associations in your data, not guaranteed returns.",
            "How these figures work",
        )

        if program:
            with st.expander("Sensitivity to the economic assumptions", icon=":material/tune:"):
                sensitivity = assumption_sensitivity(value, assumptions)
                st.dataframe(
                    sensitivity, hide_index=True, width="stretch", height="content",
                    column_config={
                        "assumption": "Assumption",
                        "setting": "If it were",
                        "value": st.column_config.NumberColumn("Program value, per year",
                                                               format="compact"),
                        "change": st.column_config.NumberColumn("Change", format="percent"),
                    },
                    alt="Program value under other economic assumptions")
                st.caption("Each row changes one assumption and holds the statistical result "
                           "fixed. The 95% range above is statistical uncertainty only; this "
                           "table is the other kind. Data limits are listed in the note at the top.")

        if scenarios is not None and not scenarios.empty:
            with st.expander("Show exactly how these figures were calculated",
                             icon=":material/receipt_long:"):
                for receipt in calculation_receipts(
                        df, analysis, assumptions, costs, value, economics,
                        st.session_state.get("health_report")):
                    st.markdown(receipt_markdown(receipt))
                    st.divider()


# PROCESS 6 – WHERE TO INVEST: FACTORS RANKED BY DOLLARS

if analysis is not None:

    if not factors.empty:

        with st.container(border=True):

            st.subheader("Where to invest: factors ranked by value", anchor=False)
            st.caption(
                "Every question that passed all checks, priced with your assumptions and "
                "ranked by yearly value: each person's own risk change from the verified "
                "models, for the people who score 1–3 today, with a 95% range in the table."
            )

            chart_col, story_col = st.columns([3, 2], gap="large")

            with chart_col:
                st.altair_chart(
                    priority_chart(factors, c),
                    width="stretch",
                    alt="Engagement questions ranked by estimated yearly value",
                )

            with story_col:
                story_box(explain_priorities(factors, assumptions))

            st.markdown("  ".join(
                f"**{name}:** {meaning}." for name, meaning in PRIORITY_MEANING.items()))

            with st.expander("Impact vs current score", icon=":material/scatter_plot:"):
                st.altair_chart(
                    priority_matrix(factors, c),
                    width="stretch",
                    alt="Each question's value per person against the share who agree today",
                )
                st.caption(
                    "Dashed lines: the median share agreeing across all 12 questions, and half "
                    "the strongest factor's value per person. Top-left = Fix first."
                )

            with st.expander("All factors as a table", icon=":material/table_chart:"):
                st.dataframe(
                    factors.drop(columns=["mean_score"]),
                    hide_index=True,
                    width="stretch",
                    height="content",
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "item": "Question",
                        "label": "Topic",
                        "priority": "Priority",
                        "yearly_value": st.column_config.NumberColumn(
                            "Yearly value", format="compact"),
                        "value_low": st.column_config.NumberColumn(
                            "Low end (95%)", format="compact"),
                        "value_high": st.column_config.NumberColumn(
                            "High end (95%)", format="compact"),
                        "value_per_person": st.column_config.NumberColumn(
                            "Per person moved", format="compact"),
                        "agree_pct": st.column_config.NumberColumn(
                            "Agree today", format="percent"),
                        "people_below": st.column_config.NumberColumn(
                            "People answering 1–3", format="%,.0f"),
                        "avoided_leavers": st.column_config.NumberColumn(
                            "Leavers avoided", format="%.1f"),
                        "turnover_value": st.column_config.NumberColumn(
                            "Turnover value", format="compact"),
                        "avoided_absence_days": st.column_config.NumberColumn(
                            "Absence days avoided", format="%,.0f"),
                        "absence_value": st.column_config.NumberColumn(
                            "Absence value", format="compact"),
                        "rating_points": st.column_config.NumberColumn(
                            "Performance points", format="%,.0f"),
                        "performance_value": st.column_config.NumberColumn(
                            "Performance value", format="compact"),
                        "outcomes": "Linked to",
                        "caution": "Note",
                    },
                    alt="Every valued engagement factor with avoided turnover, absence and value",
                )

            how_to_read(
                "Scenario: everyone who answers 1–3 on the question moves up one point, with "
                "everything else unchanged. Each person's chance of leaving and expected absence "
                "days are predicted twice by the verified models, today and after the lift, and "
                "the difference is added up. Each factor is valued on its own: the same people "
                "answer every question, so don't add the values together (the program above "
                "counts them jointly).",
                "How these values work",
            )


# PROCESS 7 – WHERE THE MONEY IS

departments = department_excess_cost(df, assumptions)
teams = team_excess_cost(df, assumptions)

if not departments.empty and (departments["excess_cost"] > 0).any():

    with st.container(border=True):

        st.subheader("Where the money is: departments", anchor=False)

        dept_col, dept_story_col = st.columns([3, 2], gap="large")

        with dept_col:
            st.dataframe(
                departments,
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    "department": "Department",
                    "people": st.column_config.NumberColumn("People", format="%,.0f"),
                    "turnover_rate": st.column_config.NumberColumn("Turnover", format="percent"),
                    "absence_days": st.column_config.NumberColumn(
                        "Absence days (6 mo.)", format="%.1f"),
                    "excess_turnover_cost": st.column_config.NumberColumn(
                        "Extra turnover cost", format="compact"),
                    "excess_absence_cost": st.column_config.NumberColumn(
                        "Extra absence cost", format="compact"),
                    "excess_cost": st.column_config.NumberColumn(
                        "Extra cost, total", format="compact"),
                },
                alt="Extra yearly turnover and absence cost for every department",
            )

        with dept_story_col:
            story_box(explain_departments(departments, assumptions))
            st.caption("Counted on all employees. Descriptive only: it shows where extra "
                       "costs are, not why.")

if not teams.empty and (teams["excess_cost"] > 0).any():

    persistence = team_cost_persistence(df, assumptions)

    with st.container(border=True):

        st.subheader("Where the money is: teams", anchor=False)

        if persistence["status"] == "failed":
            st.warning(
                persistence["text"] + " Team rankings are hidden because the costliest teams "
                "do not stay costly: most of a team's extra cost one year is chance. Use the "
                "department view above.", icon=":material/visibility_off:")
        else:
            if persistence["status"] == "passed":
                st.caption(":material/fact_check: " + persistence["text"])
            else:
                st.caption(":material/hourglass_empty: " + persistence["text"])

            basis_label = st.segmented_control(
                "Compare each team with",
                options=["The company rate", "Its own department"],
                default="The company rate",
                key="fin_team_basis",
            ) or "The company rate"
            basis = "department" if basis_label == "Its own department" else "company"
            ranked = teams.sort_values("excess_cost_dept" if basis == "department"
                                       else "excess_cost", ascending=False)

            chart_col, story_col = st.columns([3, 2], gap="large")

            with chart_col:
                st.altair_chart(
                    excess_cost_chart(ranked, c, basis=basis),
                    width="stretch",
                    alt="Teams with the highest extra yearly cost from turnover and absence",
                )

            with story_col:
                story_box(explain_where_money(teams, assumptions))
                st.caption("Descriptive only. Small teams are pulled toward the company rate "
                           "so one leaver does not decide the ranking.")

            with st.expander("All teams", icon=":material/table_chart:"):
                st.dataframe(
                    ranked.drop(columns=["excess_turnover_cost_dept", "excess_absence_cost_dept"]),
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
                            "Extra turnover cost", format="compact"),
                        "excess_absence_cost": st.column_config.NumberColumn(
                            "Extra absence cost", format="compact"),
                        "excess_cost": st.column_config.NumberColumn(
                            "Extra cost vs company", format="compact"),
                        "excess_cost_dept": st.column_config.NumberColumn(
                            "Extra cost vs own department", format="compact"),
                    },
                    alt="Extra yearly turnover and absence cost for every team",
                )


# PROCESS 8 – ASSUMPTIONS AND LIMITS

with st.expander("Assumptions and limits", icon=":material/rule:"):

    st.dataframe(assumption_register(assumptions, sample), hide_index=True, width="stretch",
                 height="content", column_config={"assumption": "Assumption", "value": "Value",
                                                  "source": "Source"},
                 alt="Every economic assumption with its value and source")
    st.markdown("\n".join(f"- {line}" for line in assumptions.describe(sample)))
    st.caption(
        "Costs today count every employee in the outcome file. Savings come from models fitted "
        "on survey respondents and scaled to the headcount above. "
        "Your data has no revenue or profit, so these are cost-based estimates, not a "
        "measured link to profit. To measure that link, add team-level financial results "
        "(for example profit or revenue per team per period) to the data."
    )


# PROCESS 9 – PLAUSIBILITY CHECK AGAINST GALLUP'S PUBLISHED FINDINGS

with st.expander("Does your data look like Gallup's research?", icon=":material/fact_check:"):

    bench = gallup_benchmark_check(df)
    if bench.empty:
        st.caption("Not enough teams with both survey answers and outcomes to compare.")
    else:
        st.dataframe(
            bench, hide_index=True, width="stretch", height="content",
            column_config={
                "outcome": "Outcome",
                "teams": st.column_config.NumberColumn("Teams", format="%d"),
                "this_data_r": st.column_config.NumberColumn("Your data (r)", format="%.2f"),
                "gallup_r": st.column_config.NumberColumn("Gallup average (r)", format="%.2f"),
                "gallup_low": st.column_config.NumberColumn("Gallup range, low", format="%.2f"),
                "gallup_high": st.column_config.NumberColumn("Gallup range, high", format="%.2f"),
                "verdict": "Compared with Gallup",
            },
            alt="Team-level correlation of engagement with outcomes, against Gallup's research")
    st.caption(
        "Gallup's 2024 Q12 meta-analysis reports business-unit-level correlations between "
        "engagement and outcomes. This table compares them with the same team-level "
        "correlations in your data; the range is the middle 80% of Gallup's studies. It is a "
        "plausibility check only. No dollar figure above uses Gallup's numbers: they describe "
        "groups, not individual employees, and Gallup's own cost-of-disengagement estimate "
        "relies on a proprietary classification we cannot reproduce.")
