"""Financial impact: translate verified engagement findings into money.

The data has no revenue or profit, so every figure here comes from the user's
own cost assumptions applied to verified results:
- What it costs today: voluntary turnover and unplanned absence, per year.
- What engagement could be worth (scenarios, from verified findings only):
    turnover    -> insights.dollar_impact(): the top turnover driver rises 1 point
                   for employees who answer 1-3 on it
    absence     -> the same for the top absence driver
    productivity-> recommendations.team_plans(): each team closes the gap on its
                   first lever, valued at the user's 'value of one rating point'
- Where the money is: teams whose turnover and absence cost more than the
  company rate would (descriptive, no model needed).

All results scale from the survey sample to the company headcount the user
enters. They are illustrative scenarios built on associations, not forecasts.

If team-level financial results (e.g. profit per team) are added to the data
later, they can be modeled directly as an outcome instead of estimated here.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import altair as alt
import pandas as pd

from .insights import analysis_frame, dollar_impact
from .visualization import HIGH_COLOR, MUTED_COLOR


# PROCESS 2 – ASSUMPTIONS

TURNOVER = "left_voluntarily_12m"
ABSENCE = "unplanned_absence_days_6m"
WORKDAYS_PER_YEAR = 260
MIN_TEAM_SIZE = 5


@dataclass
class Assumptions:
    """Cost assumptions a leader enters. Defaults are placeholders, not facts."""

    avg_salary: float = 60_000.0
    replacement_cost_pct: float = 0.5          # cost to replace one leaver, share of salary
    absence_day_cost: float | None = None      # None -> salary / 260 workdays
    value_per_rating_point: float = 0.0        # yearly value of +1 performance point per person
    headcount: int | None = None               # None -> people in the survey sample
    currency: str = "$"

    @property
    def cost_per_leaver(self) -> float:
        return self.avg_salary * self.replacement_cost_pct

    @property
    def day_cost(self) -> float:
        return self.absence_day_cost if self.absence_day_cost else self.avg_salary / WORKDAYS_PER_YEAR

    def describe(self, sample: int) -> list[str]:
        c = self.currency
        lines = [
            f"Average salary {c}{self.avg_salary:,.0f}.",
            f"Replacing one leaver costs {self.replacement_cost_pct:.0%} of salary "
            f"({c}{self.cost_per_leaver:,.0f}).",
            f"One unplanned absence day costs {c}{self.day_cost:,.0f}"
            + (" (salary ÷ 260 workdays)." if not self.absence_day_cost else "."),
            f"Headcount {self._headcount(sample):,} (survey sample: {sample:,}); results scale "
            "up by that ratio.",
        ]
        if self.value_per_rating_point > 0:
            lines.append(f"One extra performance-rating point is worth "
                         f"{c}{self.value_per_rating_point:,.0f} per person per year.")
        else:
            lines.append("Productivity value is not counted (value of a rating point = 0).")
        return lines

    def _headcount(self, sample: int) -> int:
        return int(self.headcount) if self.headcount else sample

    def scale(self, sample: int) -> float:
        return self._headcount(sample) / sample if sample else 1.0


# PROCESS 3 – WHAT IT COSTS TODAY

def current_costs(df: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    """Yearly cost of voluntary turnover and unplanned absence, company-wide."""
    data = analysis_frame(df)
    n = len(data)
    scale = a.scale(n)
    out: dict[str, Any] = {"sample": n, "headcount": a._headcount(n)}
    if TURNOVER in data.columns:
        leavers = float(data[TURNOVER].mean()) * n * scale
        out.update(leavers=leavers, turnover_cost=leavers * a.cost_per_leaver)
    if ABSENCE in data.columns:
        days = float(data[ABSENCE].mean()) * 2 * n * scale      # 6-month window -> year
        out.update(absence_days=days, absence_cost=days * a.day_cost)
    out["total_cost"] = out.get("turnover_cost", 0.0) + out.get("absence_cost", 0.0)
    return out


# PROCESS 4 – WHAT ENGAGEMENT COULD BE WORTH

def engagement_value(df: pd.DataFrame, analysis: dict | None, a: Assumptions,
                     plans: pd.DataFrame | None = None,
                     tiers: list[str] | None = None) -> dict[str, Any]:
    """Yearly value of three verified scenarios, scaled to headcount."""
    data = analysis_frame(df)
    scale = a.scale(len(data))
    scenarios: list[dict[str, Any]] = []

    impact = dollar_impact(analysis, df, a.avg_salary, a.replacement_cost_pct) if analysis else None

    if impact is not None:
        scenarios.append({
            "source": "Turnover",
            "lever": f"{impact['item']} {impact['label']}",
            "what": (f"The {impact['employees_not_agreeing'] * scale:,.0f} employees who answer "
                     f"1–3 on {impact['label']} move up 1 point"),
            "units": impact["fewer_leavers"] * scale,
            "unit_label": "fewer leavers a year",
            "value": impact["savings"] * scale,
        })
        if "absence_days_saved" in impact:
            scenarios.append({
                "source": "Absence",
                "lever": f"{impact['absence_item']} {impact['absence_label']}",
                "what": f"Employees who answer 1–3 on {impact['absence_label']} move up 1 point",
                "units": impact["absence_days_saved"] * scale,
                "unit_label": "fewer absence days a year",
                "value": impact["absence_days_saved"] * scale * a.day_cost,
            })

    if plans is not None and a.value_per_rating_point > 0:
        chosen = plans if not tiers else plans[plans["tier"].isin(tiers)]
        chosen = chosen[chosen["estimated_gain"] > 0]
        if not chosen.empty:
            point_years = float((chosen["estimated_gain"] * chosen["people"]).sum()) * scale
            scenarios.append({
                "source": "Productivity",
                "lever": "Each team's first recommended step",
                "what": f"{len(chosen)} teams close the gap on their starting question",
                "units": point_years,
                "unit_label": "rating points × people",
                "value": point_years * a.value_per_rating_point,
            })

    table = pd.DataFrame(scenarios, columns=["source", "lever", "what", "units",
                                             "unit_label", "value"])
    return {"scenarios": table, "total": float(table["value"].sum()) if len(table) else 0.0}


# PROCESS 5 – WHERE THE MONEY IS

def team_excess_cost(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Per team: yearly turnover + absence cost above what the company rate
    would give for a team of that size. Descriptive only (no model)."""
    data = analysis_frame(df)
    if "team_id" not in data.columns:
        return pd.DataFrame()
    scale = a.scale(len(data))
    agg = {"people": ("team_id", "size")}
    if TURNOVER in data.columns:
        agg["turnover_rate"] = (TURNOVER, "mean")
    if ABSENCE in data.columns:
        agg["absence_days"] = (ABSENCE, "mean")
    if "department" in data.columns:
        agg["department"] = ("department", lambda s: s.mode().iat[0])
    teams = data.groupby("team_id").agg(**agg).reset_index()
    teams = teams[teams["people"] >= MIN_TEAM_SIZE].copy()

    teams["excess_turnover_cost"] = 0.0
    teams["excess_absence_cost"] = 0.0
    if "turnover_rate" in teams:
        extra = (teams["turnover_rate"] - data[TURNOVER].mean()) * teams["people"]
        teams["excess_turnover_cost"] = extra * a.cost_per_leaver * scale
    if "absence_days" in teams:
        extra = (teams["absence_days"] - data[ABSENCE].mean()) * 2 * teams["people"]
        teams["excess_absence_cost"] = extra * a.day_cost * scale
    teams["excess_cost"] = teams["excess_turnover_cost"] + teams["excess_absence_cost"]
    return teams.sort_values("excess_cost", ascending=False).reset_index(drop=True)


# PROCESS 6 – PLAIN LANGUAGE

def money(value: float, currency: str = "$") -> str:
    """1_234_567 -> '$1.2M', 45_300 -> '$45K', 900 -> '$900'."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e9:
        return f"{sign}{currency}{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{sign}{currency}{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{sign}{currency}{v / 1e3:.0f}K"
    return f"{sign}{currency}{v:,.0f}"


def explain_value(costs: dict[str, Any], value: dict[str, Any], teams: pd.DataFrame,
                  a: Assumptions) -> dict[str, Any]:
    c = a.currency
    points = []
    if costs.get("total_cost"):
        points.append(f"Turnover and unplanned absence cost about "
                      f"**{money(costs['total_cost'], c)}** a year today.")
    for _, s in value["scenarios"].iterrows():
        points.append(f"**{s['source']}:** {s['what']} → about {s['units']:,.0f} "
                      f"{s['unit_label']}, worth **{money(s['value'], c)}** a year.")
    if not teams.empty:
        above = teams[teams["excess_cost"] > 0]
        if not above.empty:
            top = above.head(10)
            share = top["excess_cost"].sum() / above["excess_cost"].sum()
            points.append(f"{len(above)} teams cost more than the company rate would; the "
                          f"top 10 account for {share:.0%} of that extra cost.")
    headline = (f"Engagement improvements backed by your data could be worth about "
                f"{money(value['total'], c)} a year."
                if value["total"] > 0 else
                "Run the statistical analysis to see what engagement could be worth.")
    return {"headline": headline, "points": points}


# PROCESS 7 – CHARTS

def value_chart(value: dict[str, Any], currency: str = "$") -> alt.Chart:
    data = value["scenarios"].assign(label=lambda d: d["value"].map(lambda v: money(v, currency)))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=22, color=HIGH_COLOR).encode(
        y=alt.Y("source:N", sort="-x", title=None),
        x=alt.X("value:Q", title=f"Estimated yearly value ({currency})", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("source:N", title="Scenario"),
                 alt.Tooltip("lever:N", title="Lever"),
                 alt.Tooltip("what:N", title="Assumes"),
                 alt.Tooltip("label:N", title="Value")])
    text = bars.mark_text(align="left", dx=6, color="#333").encode(text="label:N")
    return (bars + text).properties(height=60 * max(1, len(data)), width=560)


def excess_cost_chart(teams: pd.DataFrame, currency: str = "$", n: int = 15) -> alt.Chart:
    top = teams[teams["excess_cost"] > 0].head(n)
    long = top.melt(id_vars=["team_id"], value_vars=["excess_turnover_cost", "excess_absence_cost"],
                    var_name="kind", value_name="cost")
    long["kind"] = long["kind"].map({"excess_turnover_cost": "Turnover",
                                     "excess_absence_cost": "Absence"})
    long = long[long["cost"] > 0]
    return alt.Chart(long).mark_bar(height=14).encode(
        y=alt.Y("team_id:N", sort=list(top["team_id"]), title=None,
                axis=alt.Axis(labelOverlap=False)),
        x=alt.X("sum(cost):Q", title=f"Extra yearly cost vs company rate ({currency})",
                axis=alt.Axis(format="~s")),
        color=alt.Color("kind:N", title=None,
                        scale=alt.Scale(domain=["Turnover", "Absence"],
                                        range=["#B5473F", MUTED_COLOR]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("team_id:N", title="Team"), alt.Tooltip("kind:N"),
                 alt.Tooltip("cost:Q", title="Extra cost", format=",.0f")],
    ).properties(height=alt.Step(26), width="container")



# PROCESS 8 – PAGE HELPERS: ASSUMPTION SUMMARY AND WHERE-THE-MONEY STORY

def md(text: str) -> str:
    """Escape '$' so Streamlit markdown does not read '$...$' as a math formula."""
    return text.replace("$", "\\$")


def assumptions_line(a: Assumptions, sample: int) -> str:
    """One compact line with the assumptions behind every figure."""
    c = a.currency
    return md(f"{c}{a.avg_salary:,.0f} salary · {a.replacement_cost_pct:.0%} of salary to replace "
            f"a leaver · {c}{a.day_cost:,.0f} per absence day · {a._headcount(sample):,} people"
            + (f" · {c}{a.value_per_rating_point:,.0f} per performance point"
               if a.value_per_rating_point > 0 else ""))


def explain_where_money(teams: pd.DataFrame, a: Assumptions) -> dict[str, Any]:
    above = teams[teams["excess_cost"] > 0]
    if above.empty:
        return {"headline": "No team costs more than the company rate would.", "points": []}
    c = a.currency
    total = above["excess_cost"].sum()
    top10 = above.head(10)["excess_cost"].sum()
    points = [f"The 10 costliest teams account for {top10 / total:.0%} of that extra cost "
              f"({money(top10, c)})."]
    if "department" in above.columns:
        by_dept = above.groupby("department")["excess_cost"].sum().sort_values(ascending=False)
        points.append("Most of it sits in " + " and ".join(
            f"**{d}** ({money(v, c)})" for d, v in by_dept.head(2).items()) + ".")
    turnover = above["excess_turnover_cost"].clip(lower=0).sum()
    absence = above["excess_absence_cost"].clip(lower=0).sum()
    points.append(f"About {turnover / max(turnover + absence, 1):.0%} of the extra cost comes "
                  "from turnover, the rest from absence.")
    return {"headline": md(f"{len(above)} teams cost {money(total, c)} a year more than "
                            "the company rate would for teams their size."),
            "points": [md(p) for p in points]}
