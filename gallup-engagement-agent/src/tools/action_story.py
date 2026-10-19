"""Action story: predictions, recommendations and money told as one story.

Four short chapters, each built from results the other pages already verify:

  1. What is likely to happen   risk models that passed their test, expected
                                leavers, engagement forecast
  2. What we can do about it    the warning signs in high-risk teams, matched with the
                                verified levers and each team's first step
  3. What it is worth           what the expected leavers cost, what acting on
                                the top lever is worth, where costs sit today
  4. Where to start             teams that show up on several lists (high risk,
                                biggest opportunity, highest extra cost)

No AI writes it, so every number traces to a calculation. Chapters that cannot
be built (no outcomes, a model that failed its test) say so in one line.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .financial import Assumptions, current_costs, md, money, team_excess_cost
from .insights import ACTION_LIBRARY, ITEM_LABELS, analysis_frame
from .prediction import TARGETS, employee_risk_by_group, explain_forecast, forecast_engagement, team_risk
from .priorities import factor_value
from .recommendations import PERFORMANCE, productivity_levers, team_plans


# PROCESS 2 – CONFIGURATION

TOP_COST_TEAMS = 20        # "highest extra cost" = the 20 costliest teams
MAX_START_TEAMS = 10
CHAPTERS = [
    ("predict", "What is likely to happen", ":material/online_prediction:"),
    ("act", "What we can do about it", ":material/tips_and_updates:"),
    ("money", "What it is worth", ":material/payments:"),
]


def _chapter(lead: str, points: list[str], available: bool = True) -> dict[str, Any]:
    return {"lead": lead, "points": points, "available": available}


def _items_in(drivers: pd.Series) -> Counter:
    """Count Q12 items named in 'Q12 Learn and grow, Q04 Recognition' strings."""
    counts: Counter = Counter()
    for text in drivers.dropna():
        for part in str(text).split(","):
            code = part.strip().split(" ")[0]
            if code in ITEM_LABELS:
                counts[code] += 1
    return counts


def _purpose(outcomes: str) -> str:
    """'for keeping people' / 'for attendance' / 'for performance' from 'turnover, absence'."""
    words = {"turnover": "keeping people", "absence": "attendance", "performance": "performance"}
    named = [words[o.strip()] for o in outcomes.split(",") if o.strip() in words]
    return "for " + " and ".join(named) if named else ""


def _risks(df: pd.DataFrame, health_report: dict | None = None) -> dict[str, dict | None]:
    out = {}
    for outcome, (column, _) in TARGETS.items():
        if column in df.columns:
            try:
                out[outcome] = team_risk(df, outcome, health_report=health_report)
            except ValueError:
                out[outcome] = None
    return out


# PROCESS 3 – CHAPTER 1: PREDICTIONS

def _predict(df, risks, a) -> tuple[dict[str, Any], dict[str, Any]]:
    facts: dict[str, Any] = {"high_teams": {}, "risk_items": Counter(), "expected_leavers": None}
    points = []
    for outcome, result in risks.items():
        if result is None or result["status"] != "useful":
            if result is not None:
                points.append(f"The {outcome} risk model did not pass its test on unseen data, "
                              "so no team ranking is used for it.")
            continue
        teams = result["teams"]
        high = teams[teams["risk_band"] == "High"]
        facts["high_teams"][outcome] = set(high["team_id"])
        facts["risk_items"] += _items_in(high["risk_drivers"])
        where = ""
        if "department" in high.columns and not high.empty:
            top = high["department"].value_counts()
            where = f", most in {top.index[0]} ({top.iloc[0]})"
        points.append(f"**{len(high)} teams** are at high risk of {result['label']}{where}. "
                      f"The model picks the riskier of two teams correctly "
                      f"{result['auc']:.0%} of the time.")

    x7 = next((r["survey_response_warning"] for r in risks.values()
               if r is not None and r.get("survey_response_warning")), None)
    if x7 is not None:
        points.append(x7["text"])

    if "department" in df.columns and TARGETS["turnover"][0] in df.columns:
        try:
            risk = employee_risk_by_group(df, "department")
            if risk["status"] == "useful":
                leavers = float(risk["table"]["expected_leavers"].sum())
                facts["expected_leavers"] = leavers * a.scale(len(analysis_frame(df)))
                points.insert(0, f"About **{facts['expected_leavers']:,.0f} people** are likely to "
                                 "leave voluntarily in the next 12 months if nothing changes.")
        except ValueError:
            pass

    forecast = forecast_engagement(df)
    points.append(explain_forecast(forecast) if forecast["available"] else
                  "No engagement forecast yet: it needs at least 4 surveys.")

    high_any = set().union(*facts["high_teams"].values()) if facts["high_teams"] else set()
    lead = (f"If nothing changes, {len(high_any)} teams are likely to see above-normal "
            "turnover or absence." if high_any else
            "No risk model passed its test yet, so there is no reliable look ahead.")
    return _chapter(lead, points, bool(high_any or facts["expected_leavers"])), facts


# PROCESS 4 – CHAPTER 2: RECOMMENDATIONS

def _act(df, analysis, change, facts, factors, risks) -> tuple[dict[str, Any], pd.DataFrame | None]:
    points = []
    plans = None
    risk_items = facts["risk_items"]
    if risk_items:
        named = [f"{ITEM_LABELS[i]} ({i})" for i, _ in risk_items.most_common(3)]
        points.append("Warning signs seen most often in high-risk teams (signals to listen to, "
                      "not proven causes): " + ", ".join(named[:-1]) + " and " + named[-1] + "."
                      if len(named) > 1 else
                      f"The warning sign seen most often in high-risk teams: {named[0]} "
                      "(a signal to listen to, not a proven cause).")

    if not factors.empty:
        top = factors.iloc[0]
        overlap = top["item"] in dict(risk_items.most_common(3))
        purpose = _purpose(top["outcomes"])
        points.append(f"Company-wide, the strongest verified lever {purpose} is **{top['label']}** "
                      f"({top['agree_pct']:.0%} agree today)"
                      + (", which is also one of the questions driving risk." if overlap else ".")
                      + f" First step for managers: {ACTION_LIBRARY[top['item']]['actions'][0]}")

    if analysis is not None and PERFORMANCE in df.columns:
        levers = productivity_levers(df, analysis, change)
        if not levers.empty:
            plans = team_plans(df, levers, risks)
            starts = plans["start_with"].dropna().value_counts()
            if not starts.empty:
                same = not factors.empty and starts.index[0] == factors.iloc[0]["item"]
                points.append(f"Team by team, to lift performance, most teams should start with "
                              f"{ITEM_LABELS[starts.index[0]]} ({int(starts.iloc[0])} teams)"
                              + (". " if same else
                                 f": a different lever from {factors.iloc[0]['label']} because it "
                                 "targets performance, so do both. " if not factors.empty else ". ")
                              + f"{int((plans['tier'] == 'Biggest opportunity').sum())} teams have "
                              "the biggest opportunity; each has a ready plan on Recommendations.")

    if not points:
        return _chapter("Add turnover, absence or performance data to see what to do.", [], False), plans
    lead = (f"Focus on {factors.iloc[0]['label']} first: it is the lever with the biggest verified "
            "payoff." if not factors.empty else
            "Use each team's plan on the Recommendations page.")
    return _chapter(lead, points), plans


# PROCESS 5 – CHAPTER 3: MONEY

def _money(df, a, facts, factors, excess) -> dict[str, Any]:
    c = a.currency
    costs = current_costs(df, a)
    points = []
    if costs.get("total_cost"):
        points.append(f"Voluntary turnover and unplanned absence cost about "
                      f"**{money(costs['total_cost'], c)} a year** today.")
    if facts["expected_leavers"]:
        points.append(f"The {facts['expected_leavers']:,.0f} likely leavers would cost about "
                      f"{money(facts['expected_leavers'] * a.cost_per_leaver, c)} to replace.")
    if not factors.empty and factors.iloc[0]["yearly_value"] > 0:
        top = factors.iloc[0]
        points.append(f"Lifting {top['label']} for the {top['people_below']:,.0f} people who "
                      f"score it low is linked to {top['avoided_leavers']:,.0f} fewer leavers, "
                      f"worth about **{money(top['yearly_value'], c)} a year**.")
    above = excess[excess["excess_cost"] > 0] if not excess.empty else excess
    if not above.empty and "department" in above.columns:
        by_dept = above.groupby("department")["excess_cost"].sum().sort_values(ascending=False)
        points.append(f"Extra costs sit mostly in {by_dept.index[0]} "
                      f"({money(by_dept.iloc[0], c)} above the company rate).")
    if not points:
        return _chapter("Money needs turnover or absence data.", [], False)
    value = factors.iloc[0]["yearly_value"] if not factors.empty else 0
    lead = (f"Acting on the top lever is worth about {money(value, c)} a year, out of "
            f"{money(costs['total_cost'], c)} at stake." if value > 0 and costs.get("total_cost")
            else "Here is what is at stake today.")
    return _chapter(lead, points)


# PROCESS 6 – CHAPTER 4: WHERE TO START

def where_to_start(facts, plans, excess, a) -> pd.DataFrame:
    """Teams on two or more lists, most lists first, then highest extra cost."""
    lists: dict[str, set] = {
        "High turnover risk": facts["high_teams"].get("turnover", set()),
        "High absence risk": facts["high_teams"].get("absence", set()),
    }
    if plans is not None:
        lists["Biggest opportunity"] = set(plans.loc[plans["tier"] == "Biggest opportunity", "team_id"])
    if not excess.empty:
        lists["Highest extra cost"] = set(excess[excess["excess_cost"] > 0]
                                          .head(TOP_COST_TEAMS)["team_id"])
    teams = set().union(*lists.values()) if lists else set()
    rows = []
    cost = excess.set_index("team_id")["excess_cost"] if not excess.empty else pd.Series(dtype=float)
    dept = excess.set_index("team_id")["department"] if "department" in excess.columns else {}
    first = plans.set_index("team_id")["start_with"] if plans is not None else {}
    for team in teams:
        on = [name for name, members in lists.items() if team in members]
        if len(on) < 2:
            continue
        step = first.get(team) if hasattr(first, "get") else None
        rows.append({"team_id": team, "department": dept.get(team, "") if hasattr(dept, "get") else "",
                     "lists": len(on), "why": " · ".join(on),
                     "start_with": f"{ITEM_LABELS[step]} ({step})" if isinstance(step, str) else "",
                     "extra_cost": float(cost.get(team, 0.0))})
    columns = ["team_id", "department", "lists", "why", "start_with", "extra_cost"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(["lists", "extra_cost"], ascending=False)
            .head(MAX_START_TEAMS).reset_index(drop=True))


# PROCESS 7 – THE WHOLE STORY

def build_action_story(df: pd.DataFrame, analysis: dict | None = None,
                       change_analysis: dict | None = None,
                       assumptions: Assumptions | None = None,
                       traps: dict | None = None) -> dict[str, Any]:
    a = assumptions or Assumptions(headcount=len(analysis_frame(df)))
    risks = _risks(df, (analysis or {}).get("health_report"))
    factors = factor_value(df, analysis, a, traps)
    excess = team_excess_cost(df, a)

    predict, facts = _predict(df, risks, a)
    act, plans = _act(df, analysis, change_analysis, facts, factors, risks)
    worth = _money(df, a, facts, factors, excess)
    start = where_to_start(facts, plans, excess, a)

    parts = []
    if facts["expected_leavers"]:
        parts.append(f"about {facts['expected_leavers']:,.0f} people are likely to leave next year")
    if not factors.empty and factors.iloc[0]["yearly_value"] > 0:
        top = factors.iloc[0]
        parts.append(f"improving {top['label']} is the strongest lever, worth about "
                     f"{money(top['yearly_value'], a.currency)} a year")
    if not start.empty:
        parts.append(f"start with the {len(start)} teams that show up on several lists")
    headline = ("If nothing changes, " + "; ".join(parts) + "."
                if parts else "Upload outcomes and run the analysis to build the story.")

    next_steps = []
    if not factors.empty:
        next_steps.append(f"Launch a company-wide push on {factors.iloc[0]['label']} "
                          f"{_purpose(factors.iloc[0]['outcomes'])}.")
    if not start.empty:
        next_steps.append(f"Meet the managers of the {len(start)} start-here teams this month and "
                          "agree each team's first step (Recommendations page).")
    next_steps.append("Check progress at the next survey: risk, the lever's score and the cost.")

    return {"headline": headline,
            "chapters": {"predict": predict, "act": act, "money": worth},
            "start_here": start, "next_steps": next_steps,
            "caution": ("Predictions are likelihoods, not certainties; values are links in your "
                        "data, not guarantees, and use the cost assumptions on Financials.")}


def story_markdown(story: dict[str, Any]) -> str:
    lines = ["# Action story", "", f"**{story['headline']}**"]
    for number, (key, title, _) in enumerate(CHAPTERS, 1):
        chapter = story["chapters"][key]
        lines += ["", f"## {number}. {title}", f"*{chapter['lead']}*"]
        lines += [f"- {p.replace('**', '')}" for p in chapter["points"]]
    start = story["start_here"]
    if not start.empty:
        lines += ["", "## 4. Where to start", "| Team | Department | Why | Team's first step |",
                  "|---|---|---|---|"]
        lines += [f"| {r['team_id']} | {r['department']} | {r['why']} | {r['start_with']} |"
                  for _, r in start.iterrows()]
    lines += ["", "## Next steps"] + [f"{i}. {s}" for i, s in enumerate(story["next_steps"], 1)]
    lines += ["", f"_{story['caution']}_"]
    return "\n".join(lines) + "\n"


def for_screen(text: str) -> str:
    return md(text)
