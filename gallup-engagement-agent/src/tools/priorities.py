"""Dollars and priorities: what each engagement factor is worth, and where to start.

For every Q12 question (factor) with a finding that passed all checks, in the
direction that helps the business:

  avoided turnover   leavers avoided a year if the employees who answer 1-3 on
                     the question move up 1 point (the model's odds ratio applied
                     to the company turnover rate), x cost per leaver
  avoided absence    unplanned absence days avoided a year in the same scenario
                     (rate ratio x company average days), x cost per day
  performance        rating points gained (standardized beta converted to rating
                     points), x the value of a rating point, if one is entered

This is the same scenario and arithmetic as the Financials page
(insights.dollar_impact), applied to every verified factor, not just the top
one, and scaled to company headcount.

Ranking combines impact and current score:
  value_per_person  dollars a year for each person who moves up (impact)
  agree_pct         share answering 4-5 today (current score)
  yearly_value      value_per_person x people answering 1-3 (room to improve):
                    the ranking order
  priority          Fix first     high impact, low score
                    Protect       high impact, high score (keep it strong)
                    Improve next  lower impact, low score
                    Maintain      lower impact, high score
High impact = at least half the value per person of the strongest factor.
Low score = fewer agree than the median of all 12 questions.

Factors overlap (the same people, related questions), so values are not added
across factors. Associations, not guarantees.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .financial import ABSENCE, TURNOVER, Assumptions, md, money
from .insights import ITEM_LABELS, Q12_ITEMS, _robust_favorable, analysis_frame


# PROCESS 2 – CONFIGURATION

PERFORMANCE = "performance_rating"
HIGH_IMPACT_SHARE = 0.5    # high impact = at least half the strongest factor's value per person
PRIORITY_ORDER = ["Fix first", "Protect", "Improve next", "Maintain"]
PRIORITY_COLORS = ["#B5473F", "#1E7B34", "#E0A030", "#9AA5B1"]
PRIORITY_MEANING = {
    "Fix first": "big payoff and many people score low: start here",
    "Protect": "big payoff but already strong: keep it that way",
    "Improve next": "smaller payoff, many people score low",
    "Maintain": "smaller payoff, already strong",
}
COLUMNS = ["rank", "item", "label", "priority", "yearly_value", "value_per_person",
           "agree_pct", "mean_score", "people_below", "avoided_leavers", "turnover_value",
           "avoided_absence_days", "absence_value", "rating_points", "performance_value",
           "outcomes", "caution"]


# PROCESS 3 – PER-FACTOR DOLLARS

def avoided_leavers(people_below: float, base_rate: float, odds_ratio: float) -> float:
    """Leavers avoided a year when people_below move up 1 point (company turnover rate)."""
    odds = base_rate / (1 - base_rate) * odds_ratio
    return people_below * (base_rate - odds / (1 + odds))


def avoided_absence_days(people_below: float, base_days_6m: float, rate_ratio: float) -> float:
    """Unplanned absence days avoided a year (the 6-month average x 2)."""
    return people_below * base_days_6m * (1 - rate_ratio) * 2


def factor_value(df: pd.DataFrame, analysis: dict | None, a: Assumptions,
                 traps: dict | None = None) -> pd.DataFrame:
    """One row per verified factor: avoided turnover and absence, dollars, priority.
    Empty if the analysis has not run or nothing passed all checks."""
    if analysis is None or analysis["findings"].empty:
        return pd.DataFrame(columns=COLUMNS)
    findings = _robust_favorable(analysis["findings"])
    if findings.empty:
        return pd.DataFrame(columns=COLUMNS)

    data = analysis_frame(df)
    scale = a.scale(len(data))
    base_rate = float(data[TURNOVER].mean()) if TURNOVER in data.columns else None
    base_days = float(data[ABSENCE].mean()) if ABSENCE in data.columns else None
    sd_perf = float(data[PERFORMANCE].std()) if PERFORMANCE in data.columns else None
    trapped = set((traps or {}).get("manager", {}).get("affected", []))

    rows = []
    for item, group in findings.groupby("item", sort=False):
        effects = dict(zip(group["outcome"], group["effect"].astype(float)))
        below = int((data[item] <= 3).sum())
        row = {"item": item, "label": ITEM_LABELS.get(item, item),
               "agree_pct": float((data[item] >= 4).mean()),
               "mean_score": float(data[item].mean()),
               "people_below": below * scale,
               "avoided_leavers": 0.0, "avoided_absence_days": 0.0, "rating_points": 0.0}

        if "turnover" in effects and base_rate:
            row["avoided_leavers"] = avoided_leavers(below, base_rate, effects["turnover"]) * scale
        if "absence" in effects and base_days is not None:
            row["avoided_absence_days"] = avoided_absence_days(
                below, base_days, effects["absence"]) * scale
        if "performance" in effects and sd_perf:
            sd_item = float(data[item].std())
            if sd_item > 0:
                row["rating_points"] = below * effects["performance"] * sd_perf / sd_item * scale

        row["turnover_value"] = row["avoided_leavers"] * a.cost_per_leaver
        row["absence_value"] = row["avoided_absence_days"] * a.day_cost
        row["performance_value"] = row["rating_points"] * a.value_per_rating_point
        row["yearly_value"] = row["turnover_value"] + row["absence_value"] + row["performance_value"]
        row["value_per_person"] = row["yearly_value"] / row["people_below"] if below else 0.0
        row["outcomes"] = ", ".join(o for o in ("turnover", "absence", "performance") if o in effects)
        notes = []
        if any((o, item) in trapped for o in effects):
            notes.append("Mostly a team (manager) effect: act through managers.")
        if row["rating_points"] > 0 and row["performance_value"] == 0 and row["yearly_value"] == 0:
            notes.append("Performance gain only: enter a value per rating point to price it.")
        row["caution"] = " ".join(notes)
        rows.append(row)

    table = pd.DataFrame(rows)
    benchmark = float(np.median([(data[i] >= 4).mean() for i in Q12_ITEMS if i in data.columns]))
    valued = table.loc[table["yearly_value"] > 0, "value_per_person"]
    impact_cut = HIGH_IMPACT_SHARE * float(valued.max()) if len(valued) else np.inf
    table.attrs["impact_cut"] = impact_cut
    high_impact = (table["value_per_person"] >= impact_cut) & (table["yearly_value"] > 0)
    low_score = table["agree_pct"] < benchmark
    table["priority"] = np.select(
        [high_impact & low_score, high_impact, low_score],
        ["Fix first", "Protect", "Improve next"], "Maintain")

    table = table.sort_values(["yearly_value", "rating_points", "agree_pct"],
                              ascending=[False, False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table.attrs["agree_benchmark"] = benchmark
    return table[COLUMNS]


# PROCESS 4 – PLAIN LANGUAGE

def _gains(row: pd.Series, c: str) -> str:
    parts = []
    if row["avoided_leavers"] > 0:
        parts.append(f"{row['avoided_leavers']:,.1f} fewer leavers")
    if row["avoided_absence_days"] > 0:
        parts.append(f"{row['avoided_absence_days']:,.0f} fewer absence days")
    if row["rating_points"] > 0:
        parts.append(f"{row['rating_points']:,.0f} performance points"
                     + (f" ({money(row['performance_value'], c)})"
                        if row["performance_value"] > 0 else ""))
    return " and ".join(parts) if parts else "no costed gain"


def factor_sentence(row: pd.Series, a: Assumptions) -> str:
    """'#1 Recognition (Q04), Fix first: lifting the 1,230 people who answer 1-3 ...'"""
    c = a.currency
    return (f"#{row['rank']} {row['label']} ({row['item']}), {row['priority']}: lifting the "
            f"{row['people_below']:,.0f} people who answer 1–3 by one point is linked to "
            f"{_gains(row, c)} a year, worth about {money(row['yearly_value'], c)} "
            f"({money(row['value_per_person'], c)} per person). Today {row['agree_pct']:.0%} "
            "agree." + (f" {row['caution']}" if row["caution"] else ""))


def explain_priorities(table: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    if table.empty:
        return {"headline": "No engagement factor passed all checks, so none is valued yet.",
                "points": []}
    c = a.currency
    top = table.iloc[0]
    fix = table[table["priority"] == "Fix first"]
    protect = table[table["priority"] == "Protect"]
    detail = factor_sentence(top, a).split(": ", 1)[1]
    points = [md(f"**Start with {top['label']} ({top['item']}).** "
                 + detail[0].upper() + detail[1:])]
    if len(fix) > (1 if top["priority"] == "Fix first" else 0):
        others = fix[fix["item"] != top["item"]]
        points.append("**Also fix first:** " + ", ".join(
            md(f"{r['label']} ({money(r['yearly_value'], c)})") for _, r in others.iterrows()) + ".")
    if not protect.empty:
        points.append("**Protect:** " + ", ".join(
            f"{r['label']} ({r['agree_pct']:.0%} agree)" for _, r in protect.iterrows())
            + ". These pay off strongly and are already strong; don't let them slip.")
    if (table["rating_points"] > 0).any() and a.value_per_rating_point <= 0:
        points.append("Performance gains are counted in rating points only. Enter a value per "
                      "rating point under Edit assumptions to turn them into money.")
    points.append("Values overlap across factors (the same people), so don't add them up.")
    return {"headline": md(f"{len(table)} engagement factors are worth money. The biggest, "
                           f"{top['label']}, is worth about {money(top['yearly_value'], c)} a year."),
            "points": points}


# PROCESS 5 – CHARTS

def priority_chart(table: pd.DataFrame, currency: str = "$") -> alt.Chart:
    """Factors ranked by yearly value, colored by priority."""
    data = table.assign(
        name=table["item"] + " · " + table["label"],
        value_label=table["yearly_value"].map(
            lambda v: money(v, currency) if v > 0 else "not priced"),
        agree=table["agree_pct"].map("{:.0%} agree".format))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("name:N", sort=list(data["name"]), title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=260)),
        x=alt.X("yearly_value:Q", title=f"Estimated yearly value ({currency})",
                axis=alt.Axis(format="~s")),
        color=alt.Color("priority:N", title=None,
                        scale=alt.Scale(domain=PRIORITY_ORDER, range=PRIORITY_COLORS),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("name:N", title="Question"), alt.Tooltip("priority:N"),
                 alt.Tooltip("value_label:N", title="Yearly value"),
                 alt.Tooltip("agree:N", title="Today"),
                 alt.Tooltip("outcomes:N", title="Linked to")])
    text = bars.mark_text(align="left", dx=6, color="#333").encode(
        text="value_label:N", color=alt.value("#333"))
    return (bars + text).properties(height=alt.Step(30), width="container")


def priority_matrix(table: pd.DataFrame, currency: str = "$") -> alt.Chart:
    """Impact (value per person) against current score (share agreeing)."""
    valued = table[table["yearly_value"] > 0]
    data = valued.assign(value_label=valued["yearly_value"].map(lambda v: money(v, currency)))
    base = alt.Chart(data).encode(
        x=alt.X("agree_pct:Q", title="Agree today (4–5)", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y("value_per_person:Q", title=f"Value per person who moves up ({currency})",
                axis=alt.Axis(format="~s"), scale=alt.Scale(zero=False, padding=20)))
    dots = base.mark_circle(opacity=0.85).encode(
        size=alt.Size("yearly_value:Q", legend=None, scale=alt.Scale(range=[120, 1200])),
        color=alt.Color("priority:N", title=None,
                        scale=alt.Scale(domain=PRIORITY_ORDER, range=PRIORITY_COLORS),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Question"), alt.Tooltip("priority:N"),
                 alt.Tooltip("agree_pct:Q", title="Agree today", format=".0%"),
                 alt.Tooltip("value_per_person:Q", title="Per person", format=",.0f"),
                 alt.Tooltip("value_label:N", title="Yearly value")])
    labels = base.mark_text(dy=-14, fontSize=11, color="#333").encode(text="item:N")
    rules = []
    if "agree_benchmark" in table.attrs:
        rules.append(alt.Chart(pd.DataFrame({"x": [table.attrs["agree_benchmark"]]}))
                     .mark_rule(strokeDash=[4, 4], color="#999").encode(x="x:Q"))
    if "impact_cut" in table.attrs and np.isfinite(table.attrs["impact_cut"]):
        rules.append(alt.Chart(pd.DataFrame({"y": [table.attrs["impact_cut"]]}))
                     .mark_rule(strokeDash=[4, 4], color="#999").encode(y="y:Q"))
    return alt.layer(*rules, dots, labels).properties(height=340, width="container")
