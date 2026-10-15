"""Productivity recommendations: what each team's leader can do to lift performance.

Productivity here = the performance rating. For each team the module finds the
Q12 questions that are verified to go with higher performance ("levers"), picks
the lever where the team has the most room to improve, estimates the rating
gain from closing that gap, and turns it into a dated, plain-language plan.

Evidence for a lever (rating points per 1-point higher score):
- Same employee over time (longitudinal.analyze_change, employee fixed effects):
  clear links only. Preferred, because each person is compared with themselves.
- Company-wide model (analytics.analyze_dataframe): findings that passed all
  checks, converted from standardized units to rating points.
Questions that only reach 'Use with caution' are not used.

Estimates are illustrative: they assume the verified link holds for this team,
and they are associations, not guarantees. Gains are NOT added across questions,
because the questions move together and the effects overlap.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .data_ingestion import Q12_ITEMS
from .insights import ACTION_LIBRARY, ITEM_LABELS, analysis_frame
from .longitudinal import INDEX, prepare
from .visualization import HIGH_COLOR, MUTED_COLOR


# PROCESS 2 – CONFIGURATION

PERFORMANCE = "performance_rating"
MIN_TEAM_SIZE = 5
TOP_SHARE = 0.20            # top 20% of teams by estimated gain = "Biggest opportunity"

TIER_ORDER = ["Biggest opportunity", "Solid opportunity", "Keep strengths"]

PLAN_TIMELINE = ["Weeks 1–2", "Weeks 3–6", "Week 8"]


# PROCESS 3 – PRODUCTIVITY LEVERS

def productivity_levers(df: pd.DataFrame, analysis: dict | None = None,
                        change_analysis: dict | None = None) -> pd.DataFrame:
    """Q12 questions verified to go with higher performance, with their strength
    in rating points per 1-point higher score and the evidence behind them."""
    within: dict[str, float] = {}
    if change_analysis is not None and not change_analysis["results"].empty:
        r = change_analysis["results"]
        r = r[(r["model"] == "employee_fe") & (r["outcome"] == "performance")
              & (r["measure"] != INDEX) & r["clear_link"] & (r["effect"] > 0)]
        within = dict(zip(r["measure"], r["effect"]))

    company: dict[str, float] = {}
    if analysis is not None and not analysis["findings"].empty:
        f = analysis["findings"]
        f = f[(f["outcome"] == "performance") & (f["robustness_status"] == "ROBUST")
              & (f["effect"] > 0)]
        frame = analysis_frame(df)
        sd_perf = float(frame[PERFORMANCE].std())
        for _, row in f.iterrows():
            sd_item = float(frame[row["item"]].std())
            if sd_item > 0:
                # Standardized beta -> rating points per 1-point higher score.
                company[row["item"]] = float(row["effect"]) * sd_perf / sd_item

    rows = []
    for item in Q12_ITEMS:
        if item not in within and item not in company:
            continue
        both = item in within and item in company
        rows.append({
            "item": item,
            "label": ITEM_LABELS.get(item, item),
            "points_per_point": within.get(item, company.get(item)),
            "evidence": "Strong" if both else "Good",
            "source": ("Same employees over time + company-wide model" if both
                       else "Same employees over time" if item in within
                       else "Company-wide model"),
        })
    columns = ["item", "label", "points_per_point", "evidence", "source"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("points_per_point", ascending=False).reset_index(drop=True))


# PROCESS 4 – TEAM PLANS

def _latest(data: pd.DataFrame) -> pd.DataFrame:
    if "wave" in data.columns and data["wave"].notna().any():
        return data[data["wave"] == data["wave"].max()]
    return data


def team_plans(df: pd.DataFrame, levers: pd.DataFrame,
               risk_results: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per team (5+ respondents, latest survey) with its starting lever,
    second lever, estimated rating gain from the first lever, and tier."""
    data = _latest(prepare(df))
    if "team_id" not in data.columns or PERFORMANCE not in data.columns:
        raise ValueError("Team plans need team_id and performance_rating.")

    items = [i for i in Q12_ITEMS if i in data.columns]
    agg = {"people": (INDEX, "size"), "q12_average": (INDEX, "mean"),
           "performance": (PERFORMANCE, "mean"), **{i: (i, "mean") for i in items}}
    if "department" in data.columns:
        agg["department"] = ("department", lambda s: s.mode().iat[0])
    teams = data.groupby("team_id").agg(**agg).reset_index()
    teams = teams[teams["people"] >= MIN_TEAM_SIZE].copy()
    company = data[items].mean()
    company_performance = float(data[PERFORMANCE].mean())

    strength = dict(zip(levers["item"], levers["points_per_point"]))
    evidence = dict(zip(levers["item"], levers["evidence"]))
    source = dict(zip(levers["item"], levers["source"]))

    high_risk: dict[str, list[str]] = {}
    for outcome, result in (risk_results or {}).items():
        if result is not None and result.get("status") == "useful":
            for team in result["teams"].loc[result["teams"]["risk_band"] == "High", "team_id"]:
                high_risk.setdefault(team, []).append(outcome)

    rows = []
    for _, t in teams.iterrows():
        gains = {item: (company[item] - t[item]) * strength[item]
                 for item in strength if t[item] < company[item]}
        ranked = sorted(gains, key=gains.get, reverse=True)
        first = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        # Strength to protect: the lever where the team is furthest above the company.
        above = {i: t[i] - company[i] for i in strength if t[i] >= company[i]}
        keep = max(above, key=above.get) if above else None
        rows.append({
            "team_id": t["team_id"],
            "department": t.get("department", ""),
            "people": int(t["people"]),
            "q12_average": float(t["q12_average"]),
            "performance": float(t["performance"]),
            "company_performance": company_performance,
            "start_with": first,
            "then": second,
            "keep_strength": keep,
            "team_score": float(t[first]) if first else np.nan,
            "company_score": float(company[first]) if first else np.nan,
            "estimated_gain": float(gains[first]) if first else 0.0,
            "evidence": evidence.get(first, ""),
            "source": source.get(first, ""),
            "high_risk": " & ".join(high_risk.get(t["team_id"], [])),
        })

    plans = pd.DataFrame(rows).sort_values("estimated_gain", ascending=False)
    has_gap = plans["estimated_gain"] > 0
    cutoff = plans.loc[has_gap, "estimated_gain"].quantile(1 - TOP_SHARE) if has_gap.any() else np.inf
    plans["tier"] = np.select(
        [has_gap & (plans["estimated_gain"] >= cutoff), has_gap],
        TIER_ORDER[:2], TIER_ORDER[2])
    return plans.reset_index(drop=True)


def _item_name(item: str | None) -> str:
    return f"{ITEM_LABELS.get(item, item)} ({item})" if item else ""


def team_plan(plan: pd.Series) -> dict[str, Any]:
    """Plain-language plan for one team's leader."""
    perf, company_perf = plan["performance"], plan["company_performance"]
    position = ("above" if perf > company_perf + 0.05
                else "below" if perf < company_perf - 0.05 else "about level with")
    summary = (f"Team {plan['team_id']} averages **{perf:.2f}** on performance, "
               f"{position} the company ({company_perf:.2f}).")

    if not plan["start_with"]:
        keep = _item_name(plan["keep_strength"])
        return {
            "summary": summary,
            "headline": "This team is at or above the company on every productivity lever.",
            "why": f"Protect what works, especially {keep}." if keep else "Protect what works.",
            "steps": [
                ("This month", f"Thank the team for what works on {keep or 'engagement'}, "
                               "and ask what helps them most."),
                ("This quarter", "Share one practice from this team with a team that is struggling."),
                ("Next survey", "Check that the strengths held."),
            ],
            "estimate": "",
            "caution": _CAUTION,
        }

    first, second = plan["start_with"], plan["then"]
    headline = f"Start with {_item_name(first)}."
    why = (f"It is one of the questions most closely linked to performance in your data, "
           f"and this team scores {plan['team_score']:.2f} vs {plan['company_score']:.2f} "
           f"for the company. Evidence: {plan['evidence'].lower()} "
           f"({plan['source'].lower()}).")
    actions = ACTION_LIBRARY[first]["actions"]
    steps = [
        (PLAN_TIMELINE[0], actions[0]),
        (PLAN_TIMELINE[1], actions[1] + (f" Then add: {ACTION_LIBRARY[second]['actions'][0]}"
                                         if second else "")),
        (PLAN_TIMELINE[2], f"Ask the team: \"{ACTION_LIBRARY[first]['questions'][0]}\" "
                           "and compare with the next survey."),
    ]
    estimate = (f"If the team closes the gap to the company average on "
                f"{ITEM_LABELS.get(first, first)}, its performance rating is expected to be "
                f"about **{plan['estimated_gain']:+.2f}** points higher "
                f"({perf:.2f} → {perf + plan['estimated_gain']:.2f}).")
    if plan["high_risk"]:
        estimate += (f" Note: this team is also at high {plan['high_risk']} risk; see the "
                     "Teams page for its risk plan.")
    if plan["keep_strength"]:
        why += f" Keep doing what works on {_item_name(plan['keep_strength'])}."
    return {"summary": summary, "headline": headline, "why": why, "steps": steps,
            "estimate": estimate, "caution": _CAUTION}


_CAUTION = ("Estimates assume the link found in your data holds for this team. They are "
            "associations, not guarantees, and they are not added across questions because "
            "the questions move together.")


# PROCESS 5 – COMPANY-WIDE SUMMARY

def company_summary(plans: pd.DataFrame, levers: pd.DataFrame) -> dict[str, Any]:
    starts = (plans[plans["start_with"].notna()]
              .groupby("start_with").agg(teams=("team_id", "size"),
                                         average_gain=("estimated_gain", "mean"))
              .reset_index().rename(columns={"start_with": "item"})
              .sort_values("teams", ascending=False))
    starts["label"] = starts["item"].map(ITEM_LABELS)
    starts["evidence"] = starts["item"].map(dict(zip(levers["item"], levers["evidence"])))

    points = []
    if not starts.empty:
        top = starts.iloc[0]
        headline = (f"{top['label']} ({top['item']}) is the most common starting point: "
                    f"{top['teams']} teams should begin there.")
        strongest = levers.iloc[0]
        points.append(f"The strongest productivity lever overall is **{strongest['label']}** "
                      f"({strongest['item']}): about {strongest['points_per_point']:+.2f} rating "
                      f"points per 1-point higher score ({strongest['evidence'].lower()} evidence).")
        biggest = plans[plans["tier"] == TIER_ORDER[0]]
        if not biggest.empty:
            points.append(f"{len(biggest)} teams have the biggest opportunity, with an estimated "
                          f"gain of {biggest['estimated_gain'].mean():+.2f} rating points each "
                          "from their first step.")
        keep = int((plans["tier"] == TIER_ORDER[2]).sum())
        if keep:
            points.append(f"{keep} teams are already at or above the company on every lever; "
                          "their leaders should protect what works and share it.")
    else:
        headline = "No team trails the company on a verified productivity lever."
    return {"headline": headline, "points": points, "starts": starts}


# PROCESS 6 – CHARTS

def starting_points_chart(starts: pd.DataFrame) -> alt.Chart:
    """How many teams should start with each lever."""
    data = starts.assign(name=starts["item"] + " · " + starts["label"])
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("name:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("teams:Q", title="Teams that should start here", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("evidence:N", title="Evidence",
                        scale=alt.Scale(domain=["Strong", "Good"], range=[HIGH_COLOR, "#9CCB9F"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("name:N", title="Question"),
                 alt.Tooltip("teams:Q", title="Teams"),
                 alt.Tooltip("average_gain:Q", title="Avg. estimated gain", format="+.2f"),
                 alt.Tooltip("evidence:N", title="Evidence")],
    ).properties(height=alt.Step(28), width="container")


def team_gap_chart(plan: pd.Series, df: pd.DataFrame, levers: pd.DataFrame) -> alt.LayerChart:
    """Team vs company on each productivity lever (dot plot)."""
    data = _latest(prepare(df))
    team = data[data["team_id"] == plan["team_id"]]
    rows = []
    for item in levers["item"]:
        rows.append({"name": f"{item} · {ITEM_LABELS.get(item, item)}",
                     "team": float(team[item].mean()), "company": float(data[item].mean()),
                     "start": item == plan["start_with"]})
    long = pd.DataFrame(rows)
    # Biggest gap first, so the starting point is at the top.
    long["gap"] = long["company"] - long["team"]
    y = alt.Y("name:N", title=None, sort=list(long.sort_values("gap", ascending=False)["name"]),
              axis=alt.Axis(labelLimit=260, labelOverlap=False))
    x = alt.X("company:Q", title="Average score (1–5)", scale=alt.Scale(domain=[1, 5]))
    rule = alt.Chart(long).mark_rule(color=MUTED_COLOR).encode(y=y, x=x, x2="team:Q")
    company = alt.Chart(long).mark_tick(color=MUTED_COLOR, thickness=3, size=18).encode(
        y=y, x=x, tooltip=[alt.Tooltip("company:Q", title="Company", format=".2f")])
    dots = alt.Chart(long).mark_circle(size=110, opacity=1).encode(
        y=y, x=alt.X("team:Q"),
        color=alt.condition("datum.start", alt.value("#B5473F"), alt.value(HIGH_COLOR)),
        tooltip=[alt.Tooltip("name:N", title="Question"),
                 alt.Tooltip("team:Q", title="Team", format=".2f"),
                 alt.Tooltip("company:Q", title="Company", format=".2f")])
    return (rule + company + dots).properties(height=alt.Step(28), width="container")
