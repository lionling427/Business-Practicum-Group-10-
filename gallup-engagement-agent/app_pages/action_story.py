# PROCESS 1 – IMPORTS

import altair as alt
import html
import pandas as pd
import re
import streamlit as st

from src.tools.action_story import MAX_START_TEAMS, build_action_story, for_screen
from src.tools.ceo_brief import brief_for_screen, build_ceo_brief
from src.tools.dashboard import driver_rows
from src.tools.financial import Assumptions, current_costs, md, money
from src.tools.insights import (
    ITEM_LABELS,
    OUTCOME_INFO,
    Q12_ITEMS,
    analysis_frame,
    team_scorecard,
)
from src.tools.pdf_export import barh, document_pdf, pdf_text
from src.tools.priorities import (
    PRIORITY_COLORS,
    PRIORITY_MEANING,
    PRIORITY_ORDER,
    factor_value,
    recommended_program,
)
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
    "The story in one sentence, then what drives results, what it is worth, where to act "
    "and which teams to start with.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info("The action story needs business outcomes (turnover, absence or performance) "
            "in the uploaded data.", icon=":material/info:")
    st.stop()

analysis = analysis_runner(df, key="run_analysis_story")
assumptions = (st.session_state.get("assumptions")
               or Assumptions(headcount=len(analysis_frame(df))))
c = assumptions.currency
traps = trap_checks(df)

with st.spinner("Putting the story together..."):
    story = build_action_story(df, analysis, st.session_state.get("change_analysis"),
                               assumptions, traps)
    brief = build_ceo_brief(df, analysis, assumptions, traps)
shown = brief_for_screen(brief)

GREEN, BURGUNDY, BUTTER = "#2F6B3F", "#8C2A3A", "#C9A227"


# PROCESS 3 – CHART DATA (shared by the screen and the PDF)

drivers = pd.DataFrame()
worth = pd.DataFrame()
factors = pd.DataFrame()
priority_teams = pd.DataFrame()
program = None
ranking = pd.DataFrame()
NO_LINK = "No verified link"
NOT_PRICED = "Linked, not priced"

if analysis is not None:

    # 1. What drives results: verified links, top 5 per outcome, in plain units
    #    ("per 100 employees") instead of odds or rate ratios.
    rows = driver_rows(analysis, df)
    drivers = (rows[(rows["reliability"] == "Passed all checks") & (rows["size"] > 0)]
               .sort_values("size", ascending=False).groupby("outcome").head(5).copy())
    frame = analysis_frame(df)
    plain = []
    for _, r in drivers.iterrows():
        if r["outcome"] == "turnover":
            base = float(frame[OUTCOME_INFO["turnover"]["column"]].mean())
            odds = base / (1 - base) * (1 - r["size"] / 100)
            value = (base - odds / (1 + odds)) * 100          # fewer leavers per 100 people
            plain.append((value, f"{value:.1f} fewer leavers"))
        elif r["outcome"] == "absence":
            base = float(frame[OUTCOME_INFO["absence"]["column"]].mean()) * 2   # per year
            value = base * r["size"]                           # fewer days per 100 people
            plain.append((value, f"{value:,.0f} fewer days"))
        else:
            plain.append((r["size"], f"+{r['size']:.2f} points"))
    drivers["plain"] = [v for v, _ in plain]
    drivers["plain_label"] = [t for _, t in plain]
    drivers["label"] = drivers["item"].map(ITEM_LABELS)

    def explain(r):
        if r["outcome"] == "turnover":
            return (f"When 100 employees score {r['label']} one point higher, about "
                    f"{r['plain']:.1f} fewer of them leave in a year.")
        if r["outcome"] == "absence":
            return (f"When 100 employees score {r['label']} one point higher, they take about "
                    f"{r['plain']:,.0f} fewer unplanned absence days a year.")
        return (f"Employees who score {r['label']} one point higher get about "
                f"{r['plain']:+.2f} points on their performance rating.")

    drivers["explain"] = drivers.apply(explain, axis=1)

    # 2. What it is worth: for turnover and for absence, the cost today next to
    #    what the recommended program (the #1 lever) would save.
    costs = current_costs(df, assumptions)
    program = recommended_program(df, analysis, assumptions, traps)
    rows_worth = []
    lever = program["label"] if program is not None else ""
    saved_kind = f"Saved if {lever} improves" if program is not None else "Saved"
    ratio = ((program["low"] / program["value"], program["high"] / program["value"])
             if program is not None and program["value"] else (1.0, 1.0))
    who = (f"the {program['people']:,.0f} employees who answer 1–3 on {lever} move up one point"
           if program is not None else "")
    if costs.get("turnover_cost"):
        rows_worth.append(("Voluntary turnover", "Cost today", costs["turnover_cost"], None, None,
                           f"{costs['leavers']:,.0f} people left in the last 12 months × "
                           f"{money(assumptions.cost_per_leaver, c)} to replace each one."))
        if program is not None and program["row"]["turnover_value"] > 0:
            v = float(program["row"]["turnover_value"])
            rows_worth.append(("Voluntary turnover", saved_kind, v, v * ratio[0], v * ratio[1],
                               f"If {who}, about {program['leavers']:,.0f} fewer people leave "
                               f"a year × {money(assumptions.cost_per_leaver, c)} each."))
    if costs.get("absence_cost"):
        rows_worth.append(("Unplanned absence", "Cost today", costs["absence_cost"], None, None,
                           f"{costs['absence_days']:,.0f} unplanned absence days a year × "
                           f"{money(assumptions.day_cost, c)} a day."))
        if program is not None and program["row"]["absence_value"] > 0:
            v = float(program["row"]["absence_value"])
            rows_worth.append(("Unplanned absence", saved_kind, v, v * ratio[0], v * ratio[1],
                               f"If {who}, about {program['absence_days']:,.0f} fewer absence "
                               f"days a year × {money(assumptions.day_cost, c)} a day."))
    worth = pd.DataFrame(rows_worth, columns=["outcome", "kind", "value", "low", "high",
                                              "explain"])
    if not worth.empty:
        worth["label"] = worth["value"].map(lambda v: money(v, c))

    # 3. Where to act: factor priorities, and priority teams per department.
    all_factors = factor_value(df, analysis, assumptions, traps)
    factors = all_factors[all_factors["yearly_value"] > 0] if not all_factors.empty \
        else all_factors

    # All 12 questions in one ranking: priced questions by yearly value, then the
    # ones without a verified link, lowest agreement first.
    frame12 = analysis_frame(df)
    ranking = pd.DataFrame({"item": [i for i in Q12_ITEMS if i in frame12.columns]})
    ranking["label"] = ranking["item"].map(ITEM_LABELS)
    ranking["agree_pct"] = [float((frame12[i] >= 4).mean()) for i in ranking["item"]]
    ranking["people_below"] = [float((frame12[i] <= 3).sum()) * assumptions.scale(len(frame12))
                               for i in ranking["item"]]
    priced = factors.set_index("item") if not factors.empty else pd.DataFrame()
    ranking["yearly_value"] = [float(priced.loc[i, "yearly_value"]) if i in priced.index
                               else 0.0 for i in ranking["item"]]
    linked = set(all_factors["item"]) if not all_factors.empty else set()
    ranking["priority"] = [priced.loc[i, "priority"] if i in priced.index
                           else NOT_PRICED if i in linked else NO_LINK
                           for i in ranking["item"]]
    ranking = ranking.sort_values(["yearly_value", "agree_pct"], ascending=[False, True])
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking["name"] = ranking["item"] + " " + ranking["label"]
    ranking["value_label"] = [money(v, c) if v > 0 else "–" for v in ranking["yearly_value"]]
    # Unpriced questions get a short stub so they still show up in the ranking.
    stub = float(ranking["yearly_value"].max()) * 0.02 if ranking["yearly_value"].max() > 0 else 1
    ranking["bar"] = ranking["yearly_value"].where(ranking["yearly_value"] > 0, stub)
    ranking["explain"] = [
        (f"{r['priority']}: {PRIORITY_MEANING[r['priority']]}. {r['agree_pct']:.0%} agree today; "
         f"lifting the {r['people_below']:,.0f} people who answer 1–3 by one point is worth "
         f"about {r['value_label']} a year.") if r["priority"] in PRIORITY_MEANING else
        (f"Linked to higher performance ratings, but not priced: add the value of one "
         f"rating point under Financial impact > Edit assumptions. {r['agree_pct']:.0%} agree "
         "today.") if r["priority"] == NOT_PRICED else
        (f"No verified link to turnover, absence or performance yet, so it is not priced. "
         f"{r['agree_pct']:.0%} agree today.")
        for _, r in ranking.iterrows()]
    teams = team_scorecard(df, analysis)
    if not teams.empty and "priority" in teams and "department" in teams and teams["priority"].any():
        priority_teams = (teams[teams["priority"]]["department"].value_counts()
                          .rename_axis("department").reset_index(name="teams"))
        priority_teams["explain"] = priority_teams.apply(
            lambda r: f"{r['teams']} teams in {r['department']} score low on their focus "
                      "question and have worse turnover or absence than the company.", axis=1)


# PROCESS 4 – RECOMMENDATION SUMMARY

top_factor = factors.iloc[0] if not factors.empty and factors.iloc[0]["yearly_value"] > 0 else None
start_count = len(story["start_here"])
team_word = "team" if start_count == 1 else "teams"
factor_recommendation = (f'Company-wide: Prioritize “{top_factor["label"]}”'
                         if top_factor is not None else
                         "Company-wide: No verified factor is currently available.")
team_recommendation = (f"Team review: Review the {start_count} {team_word} with the "
                       "strongest combined signals." if start_count else
                       "Team review: No cross-signal teams are currently available for review.")

st.html(
    "<style>"
    ".strategy-summary-lead{display:flex;flex-direction:column;gap:.42rem;margin-bottom:1.1rem}"
    ".strategy-summary-row{display:flex;align-items:center;gap:.6rem;min-width:0;text-align:left}"
    ".strategy-summary-marker{display:inline-flex;align-items:center;justify-content:center;"
    "flex:none;width:1.35rem;height:1.35rem;border-radius:50%;background:#1F4A2E;"
    "color:white;font-size:.9rem;font-weight:600}"
    ".strategy-summary-title{color:#1F4A2E;font-size:1.1rem;font-weight:600;"
    "white-space:nowrap;line-height:1.35;min-width:0}"
    "@media(max-width:1000px){.strategy-summary-title{white-space:normal}}"
    "</style>"
    '<div class="strategy-summary-lead">'
    '<div class="strategy-summary-row"><span class="strategy-summary-marker" '
    'aria-hidden="true">1</span>'
    f'<div class="strategy-summary-title">{html.escape(factor_recommendation)}</div></div>'
    '<div class="strategy-summary-row"><span class="strategy-summary-marker" '
    'aria-hidden="true">2</span>'
    f'<div class="strategy-summary-title">{html.escape(team_recommendation)}</div></div>'
    '</div>'
)

with st.container(border=True):

    leavers_match = re.search(r"about ([\d,]+) people are likely to leave next year",
                               story["headline"])
    if leavers_match is None:
        leavers_match = re.search(
            r"About \*\*([\d,]+) people\*\* are likely to leave",
            " ".join(story["chapters"]["predict"]["points"]),
        )
    risk_display = leavers_match.group(1) if leavers_match else "Unavailable"
    value_display = money(top_factor["yearly_value"], c) if top_factor is not None else "Unavailable"
    teams_display = f"{start_count:,}"
    metrics = (
        ("Risk if nothing changes", risk_display, "modeled likely leavers next year", "#8C2A3A"),
        ("Potential yearly value", value_display, "modeled from observed associations", "#1F4A2E"),
        ("Teams to review", teams_display, "highest combined signals shown", "#1F4A2E"),
    )
    team_help = ("Teams appear here when they qualify on at least two independent concern "
                 f"or opportunity lists. The page displays up to {MAX_START_TEAMS} teams, "
                 "so additional teams may qualify.")
    metric_html = "".join(
        f'<div class="strategy-summary-metric"><div class="strategy-summary-label" style="color:{color}">'
        f'{html.escape(label)}'
        + (f' <span class="strategy-summary-help" title="{html.escape(team_help, quote=True)}" '
           'aria-label="About teams to review" tabindex="0">?</span>'
           if label == "Teams to review" else "")
        + '</div>'
        f'<div class="strategy-summary-value" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="strategy-summary-detail">{html.escape(detail)}</div></div>'
        for label, value, detail, color in metrics
    )
    st.html(
        "<style>"
        ".strategy-summary-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));"
        "margin-top:1rem}"
        ".strategy-summary-metric{min-width:0;padding:0 1rem;text-align:center}"
        ".strategy-summary-metric+.strategy-summary-metric{border-left:1px solid #E8DDB5}"
        ".strategy-summary-label{font-size:.84rem;font-weight:600;min-height:1.3rem}"
        ".strategy-summary-value{font-size:1.45rem;font-weight:600;line-height:1.3}"
        ".strategy-summary-detail{font-size:.8rem;color:#6E6A5E;min-height:1.3rem}"
        ".strategy-summary-help{cursor:help;border-bottom:1px dotted #6E6A5E}"
        ".strategy-summary-caveat{font-size:.72rem;font-style:italic;color:#6E6A5E;"
        "margin-top:1.5rem;text-align:left}"
        "@media(max-width:640px){.strategy-summary-metrics{grid-template-columns:1fr;gap:.7rem}"
        ".strategy-summary-metric{padding:0;border-left:0!important}}"
        "</style>"
        f'<div class="strategy-summary-metrics">{metric_html}</div>'
        '<div class="strategy-summary-caveat">Predictions are likelihoods, not certainties; '
        'potential values are modeled from observed associations, not guarantees, and depend '
        'on the cost assumptions in Financials.</div>'
    )

with st.container(border=True):
    st.markdown("**How to read this recommendation**")
    active_lists = sorted({name for why in story["start_here"].get("why", [])
                           for name in str(why).split(" · ") if name})
    lists_text = (" The lists represented here include "
                  + ", ".join(name.lower() for name in active_lists) + ".") if active_lists else ""
    factor_text = (f"{top_factor['label']} is the current company-wide priority based on "
                   "the verified factor analysis. " if top_factor is not None else
                   "No verified company-wide factor is currently available. ")
    team_text = (f"Separately, the {start_count} {team_word} shown here appeared on at least "
                 "two independent concern or opportunity lists. " if start_count else
                 "Separately, no team currently appears on at least two available concern "
                 "or opportunity lists. ")
    explanation = (factor_text + team_text + lists_text + " Teams are ordered by how many "
                   "lists they appear on, then by estimated extra annual cost. The page "
                   f"displays up to {MAX_START_TEAMS} teams, so others may qualify. Each "
                   "team's own first step is calculated separately and may focus on a "
                   "different engagement factor.")
    st.markdown(explanation)


# PROCESS 5 – 1-3: WHAT DRIVES RESULTS, WHAT IT IS WORTH, WHERE TO ACT
#
# Each chapter: charts on the left, the plain-language points on the right.

CHAPTERS = [
    (":material/insights:", "What drives results", "what_matters"),
    (":material/payments:", "What it is worth", "money"),
    (":material/flag:", "Where to act", "where_to_act"),
]


PLAIN_AXIS = {
    "turnover": "Fewer leavers a year, per 100 employees",
    "absence": "Fewer absence days a year, per 100 employees",
    "performance": "Higher rating, points per person",
}
PLAIN_COLOR = {"turnover": GREEN, "absence": GREEN, "performance": BUTTER}


def plain_chart(data, outcome):
    data = data.sort_values("plain", ascending=False)
    base = alt.Chart(data).encode(
        y=alt.Y("question:N", sort=list(data["question"]), title=None,
                axis=alt.Axis(labelLimit=170)))
    bars = base.mark_bar(cornerRadiusEnd=3, color=PLAIN_COLOR[outcome]).encode(
        x=alt.X("plain:Q", title=PLAIN_AXIS[outcome], axis=alt.Axis(tickCount=4),
                scale=alt.Scale(domain=[0, float(data["plain"].max()) * 1.9])),
        tooltip=[alt.Tooltip("question:N", title="Question"),
                 alt.Tooltip("explain:N", title="What it means")])
    text = base.mark_text(align="left", dx=4).encode(x="plain:Q", text="plain_label:N")
    return (bars + text).properties(height=alt.Step(34))


def driver_points():
    """One plain sentence per outcome, for its strongest verified question."""
    out = []
    for outcome in ("turnover", "absence", "performance"):
        d = drivers[drivers["outcome"] == outcome]
        if d.empty:
            continue
        top = d.sort_values("plain", ascending=False).iloc[0]
        if outcome == "turnover":
            out.append(f"When 100 employees score **{top['label']}** one point higher, about "
                       f"**{top['plain']:.1f} fewer** of them leave in a year.")
        elif outcome == "absence":
            out.append(f"When 100 employees score **{top['label']}** one point higher, they "
                       f"take about **{top['plain']:,.0f} fewer** unplanned absence days a year.")
        else:
            out.append(f"Employees who score higher on **{top['label']}** tend to get higher "
                       f"performance ratings: about **+{top['plain']:.2f} points** for each "
                       "1-point higher score.")
    if out:
        out.append("These are links in your data, not proof of cause: they show where action "
                   "is most likely to pay off.")
    return out


def drivers_screen():
    if drivers.empty:
        st.caption("No engagement question passed every check yet.")
        return
    outcomes = [o for o in ("turnover", "absence", "performance")
                if (drivers["outcome"] == o).any()]
    for outcome, column in zip(outcomes, st.columns(len(outcomes))):
        with column:
            st.markdown(f"**{OUTCOME_INFO[outcome]['label']}**")
            st.altair_chart(plain_chart(drivers[drivers["outcome"] == outcome], outcome),
                            width="stretch", alt=f"Top engagement questions linked to {outcome}")
    st.caption(":material/touch_app: Hover a bar to see what it means. Only links that passed "
               "every check; links in your data, not proof of cause.")


def worth_screen():
    if worth.empty:
        st.caption("Add turnover or absence data to put a value on engagement.")
        return
    kinds = list(dict.fromkeys(worth["kind"]))
    base = alt.Chart(worth).encode(
        y=alt.Y("outcome:N", title=None, sort=list(dict.fromkeys(worth["outcome"])),
                axis=alt.Axis(labelLimit=200, labelFontWeight="bold")),
        yOffset=alt.YOffset("kind:N", sort=kinds))
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title=f"Per year ({c})", axis=alt.Axis(format="~s"),
                scale=alt.Scale(domain=[0, float(worth[["value", "high"]].max().max()) * 1.2])),
        color=alt.Color("kind:N", title=None, sort=kinds,
                        legend=alt.Legend(orient="bottom", labelLimit=400),
                        scale=alt.Scale(domain=kinds, range=[BURGUNDY, GREEN][:len(kinds)])),
        tooltip=[alt.Tooltip("outcome:N", title=" "), alt.Tooltip("kind:N", title=" "),
                 alt.Tooltip("label:N", title="Per year"),
                 alt.Tooltip("explain:N", title="How it is worked out")])
    whisker = base.mark_rule(strokeWidth=3, color=BUTTER).encode(x="low:Q", x2="high:Q")
    text = base.mark_text(align="left", dx=6, fontWeight="bold").encode(
        x=alt.X("value:Q"), text="label:N")
    st.altair_chart((bars + whisker + text).properties(height=alt.Step(30)),
                    width="stretch", alt="Cost today and what the program would save")
    if program is not None:
        share = program["value"] / costs["total_cost"] if costs.get("total_cost") else 0
        st.markdown(md(
            f"**In total, improving {lever} is worth about {money(program['value'], c)} a year** "
            f"({money(program['low'], c)}–{money(program['high'], c)} likely, gold line), "
            f"{share:.0%} of today's {money(costs['total_cost'], c)} people cost."))
    missing = [o for o in worth["outcome"].unique()
               if not ((worth["outcome"] == o) & (worth["kind"] != "Cost today")).any()]
    if program is not None and missing:
        st.caption(f"{lever} has no verified link to {' or '.join(m.lower() for m in missing)}, "
                   "so the program saves nothing there.")
    st.caption(":material/touch_app: Hover a bar to see how it is worked out. Cost today counts "
               "people who actually left in the last 12 months; the number in the story above "
               "is a forecast for next year.")


def act_screen():
    st.markdown("**All 12 questions, ranked by what improving them is worth**")
    if ranking.empty:
        st.caption("No question has a price yet.")
    else:
        order = PRIORITY_ORDER + [NOT_PRICED, NO_LINK]
        colors = PRIORITY_COLORS + ["#EAD58A", "#D9D4C7"]
        present = [p for p in order if (ranking["priority"] == p).any()]
        base = alt.Chart(ranking).encode(
            x=alt.X("name:N", sort=list(ranking["name"]), title=None,
                    axis=alt.Axis(labelAngle=-35, labelLimit=160, labelOverlap=False)))
        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            y=alt.Y("bar:Q", title=f"Yearly value if improved ({c})",
                    axis=alt.Axis(format="~s"),
                    scale=alt.Scale(domain=[0, float(ranking["bar"].max()) * 1.18])),
            color=alt.Color("priority:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=present,
                                            range=[colors[order.index(p)] for p in present])),
            tooltip=[alt.Tooltip("rank:Q", title="Rank"),
                     alt.Tooltip("name:N", title="Question"),
                     alt.Tooltip("value_label:N", title="Yearly value"),
                     alt.Tooltip("agree_pct:Q", title="Agree today", format=".0%"),
                     alt.Tooltip("explain:N", title="What it means")])
        text = base.mark_text(dy=-8, fontWeight="bold", fontSize=11).encode(
            y="bar:Q", text="value_label:N")
        st.altair_chart((bars + text).properties(height=360), width="stretch",
                        alt="All 12 engagement questions ranked by yearly value")
        st.caption(":material/touch_app: Hover a bar for what it means. **Fix first** = big "
                   "payoff and many people score low; **Protect** = big payoff, already strong; "
                   "**Improve next** = smaller payoff, many score low; **Linked, not priced** = "
                   "goes with performance, which has no dollar value yet; gray = no verified "
                   "link yet.")

    st.markdown("**Teams that need support first, by department**")
    if priority_teams.empty:
        st.caption("No team needs support first.")
    else:
        bars = alt.Chart(priority_teams).mark_bar(cornerRadiusEnd=4, color=BURGUNDY).encode(
            y=alt.Y("department:N", sort="-x", title=None),
            x=alt.X("teams:Q", title="Priority teams", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("department:N", title="Department"),
                     alt.Tooltip("explain:N", title="What it means")])
        labels = bars.mark_text(align="left", dx=4, color="#1F2A22").encode(text="teams:Q")
        st.altair_chart((bars + labels).properties(height=alt.Step(26)),
                        width="stretch", alt="Priority teams per department")
        st.caption("Priority team = low on its focus question and worse than the company "
                   "on turnover or absence.")


SCREEN = {"what_matters": drivers_screen, "money": worth_screen, "where_to_act": act_screen}

for number, (icon, title, key) in enumerate(CHAPTERS, 1):
    with st.container(border=True):
        st.markdown(f"#### {icon} {number}. {title}")
        # Charts use the full width; hover a bar for what it means.
        if analysis is not None:
            SCREEN[key]()
        else:
            st.caption("Charts appear once the statistical analysis has run.")


# PROCESS 6 – 4: WHERE TO START

start = story["start_here"].copy()
if "extra_cost" in start.columns:
    # Written out (e.g. $1.2M) so the browser's language never changes the units.
    start["extra_cost"] = start["extra_cost"].map(lambda v: money(v, c))

with st.container(border=True):

    st.markdown("#### :material/target: 4. Where to start")

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
                "extra_cost": "Extra yearly cost",
            },
            alt="Teams that show up on several lists, with their first step",
        )

st.caption(story["caution"])


# PROCESS 7 – DOWNLOAD (PDF, WITH THE SAME CHARTS)

def pdf_charts(key):
    charts = []
    if key == "what_matters" and not drivers.empty:
        for outcome in ("turnover", "absence", "performance"):
            d = drivers[drivers["outcome"] == outcome].sort_values("plain", ascending=False)
            if not d.empty:
                charts.append({
                    "title": f"{OUTCOME_INFO[outcome]['label']}: {PLAIN_AXIS[outcome].lower()}, "
                             "per 1-point higher score",
                    "height": 0.55 + 0.32 * len(d),
                    "draw": barh(list(d["question"]), list(d["plain"]), PLAIN_COLOR[outcome],
                                 list(d["plain_label"]))})
    if key == "money" and not worth.empty:
        charts.append({
            "title": f"Cost today and what improving {lever} would save (gold = likely range)",
            "height": 0.55 + 0.42 * len(worth),
            "draw": barh([f"{o}: {'cost today' if k == 'Cost today' else 'saved'}"
                          for o, k in zip(worth["outcome"], worth["kind"])],
                         list(worth["value"]),
                         [BURGUNDY if k == "Cost today" else GREEN for k in worth["kind"]],
                         list(worth["label"]),
                         ranges=list(zip(worth["low"], worth["high"])),
                         xlabel=f"Per year ({c})")})
    if key == "where_to_act":
        if not ranking.empty:
            palette = dict(zip(PRIORITY_ORDER + [NOT_PRICED, NO_LINK],
                               PRIORITY_COLORS + ["#EAD58A", "#D9D4C7"]))

            def ranked(ax):
                xs = range(len(ranking))
                ax.bar(xs, ranking["bar"], color=[palette[p] for p in ranking["priority"]],
                       width=0.65)
                ax.set_xticks(list(xs), [pdf_text(n) for n in ranking["name"]], rotation=40,
                              ha="right", fontsize=7)
                top = float(ranking["bar"].max())
                ax.set_ylim(0, top * 1.2)
                for x, v, t in zip(xs, ranking["bar"], ranking["value_label"]):
                    ax.text(x, v + top * 0.02, t, ha="center", fontsize=7, fontweight="bold",
                            parse_math=False)
                ax.yaxis.set_major_formatter(
                    lambda y, _: pdf_text(money(y, c)) if y else "0")
                ax.set_ylabel(pdf_text(f"Yearly value if improved ({c})"), fontsize=8)
                for name in [p for p in palette if (ranking["priority"] == p).any()]:
                    ax.bar([0], [0], color=palette[name], label=name)
                ax.legend(fontsize=7, frameon=False, loc="upper right")

            charts.append({"title": "All 12 questions, ranked by what improving them is worth",
                           "height": 3.6, "draw": ranked})
        if not priority_teams.empty:
            charts.append({
                "title": "Teams that need support first, by department",
                "height": 0.55 + 0.32 * len(priority_teams),
                "draw": barh(list(priority_teams["department"]),
                             list(priority_teams["teams"]), BURGUNDY,
                             [str(v) for v in priority_teams["teams"]], xlabel="Priority teams")})
    return charts


pdf_table = start.rename(columns={
    "team_id": "Team", "department": "Department", "lists": "Lists",
    "why": "Why it is here", "start_with": "First step", "extra_cost": "Extra yearly cost"})

sections = [{"heading": "The story in one sentence", "lead": story["headline"]}]
def pdf_points(key):
    if key == "what_matters" and analysis is not None and not drivers.empty:
        return driver_points()
    return brief[key]


sections += [{"heading": f"{n}. {title}", "charts": pdf_charts(key), "bullets": pdf_points(key)}
             for n, (_, title, key) in enumerate(CHAPTERS, 1)]
sections += [
    {"heading": "4. Where to start",
     "paragraphs": ["These teams show up on two or more lists: high risk, biggest opportunity "
                    "or highest extra cost." if not start.empty
                    else "No team shows up on two or more lists."],
     "table": pdf_table if not start.empty else None},
    {"heading": "How sure are we", "bullets": brief.get("confidence", []) + [story["caution"]],
     "color": "#6E6A5E"},
]

st.download_button(
    "Download the action story (PDF)",
    data=lambda: document_pdf("Action story", sections, subtitle=brief.get("survey", "")),
    file_name="engagai_action_story.pdf",
    mime="application/pdf",
    icon=":material/picture_as_pdf:",
    type="primary",
)
