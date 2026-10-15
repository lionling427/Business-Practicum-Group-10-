"""CEO brief: the statistical results on one page, in plain words.

Built only from results the app has already verified; no AI writes it, so
every number traces to a calculation:

  bottom_line   one sentence: where engagement stands and the biggest lever
  scorecard     the headline numbers with change since the last survey
  what_matters  the strongest finding per outcome that passed all checks
  money         what turnover and absence cost today; what the top factors are worth
  where_to_act  factors to fix first, teams that need support first
  decisions     three asks for the CEO
  confidence    how sure we are, and the analytical traps to keep in mind

brief_markdown() turns it into a document to download or paste into an email.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import pandas as pd

from .dashboard import kpi_tiles
from .financial import Assumptions, current_costs, md, money
from .insights import analysis_frame, explain_finding, team_scorecard, top_drivers
from .priorities import factor_value
from .waves import latest_wave, model_wave, wave_label


# PROCESS 2 – CONFIGURATION

SMALL_POINTS = 1.0      # changes under 1 percentage point read "about the same"
SMALL_SCORE = 0.05      # ... and under 0.05 on scores, days and ratings

# Only traps that are not already handled by the models (department is).
TRAP_PLAIN = {
    "manager": "Some links mostly reflect how teams are managed, so act through managers there.",
    "nonresponse": "The people who did not answer the survey look different from those who did, "
                   "so results may not speak for everyone.",
}


def _change(tile: dict[str, Any], judge: bool = True) -> str:
    """'up 2.1 pts (better)' style text; judge=False leaves out the verdict."""
    delta = tile["delta"]
    if delta is None:
        return "first survey"
    percent = "%" in tile["value"]
    if abs(delta) < (SMALL_POINTS if percent else SMALL_SCORE):
        return "about the same"
    unit = " pts" if percent else ""
    better = (delta < 0) if tile["inverse"] else (delta > 0)
    text = f"{'up' if delta > 0 else 'down'} {abs(delta):g}{unit}"
    return f"{text} ({'better' if better else 'worse'})" if judge else text


# PROCESS 3 – BUILD THE BRIEF

def build_ceo_brief(df: pd.DataFrame, analysis: dict | None = None,
                    assumptions: Assumptions | None = None,
                    traps: dict | None = None) -> dict[str, Any]:
    a = assumptions or Assumptions(headcount=len(analysis_frame(df)))
    c = a.currency
    tiles = kpi_tiles(df)
    by_label = {t["label"]: t for t in tiles}
    brief: dict[str, Any] = {
        "survey": wave_label(latest_wave(df)),
        "model_survey": wave_label(model_wave(df)),
        "scorecard": [{"label": t["label"], "value": t["value"], "change": _change(t)}
                      for t in tiles],
    }

    # Engagement in one clause.
    engaged = by_label.get("Employees engaged")
    if not engaged:
        engagement = "Engagement was measured"
    elif engaged["delta"] is None:
        engagement = f"{engaged['value']} of employees are engaged"
    elif _change(engaged) == "about the same":
        engagement = f"{engaged['value']} of employees are engaged, about the same as the last survey"
    else:
        engagement = (f"{engaged['value']} of employees are engaged, "
                      f"{_change(engaged, judge=False)} since the last survey")

    # What matters: strongest verified finding per outcome.
    drivers = top_drivers(analysis) if analysis is not None else {}
    brief["what_matters"] = [explain_finding(row, df)["headline"]
                             for row in drivers.values()]

    # Money.
    costs = current_costs(df, a)
    factors = factor_value(df, analysis, a, traps)
    money_points = []
    if costs.get("total_cost"):
        money_points.append(f"Voluntary turnover and unplanned absence cost about "
                            f"{money(costs['total_cost'], c)} a year today.")
    top = factors.iloc[0] if not factors.empty and factors.iloc[0]["yearly_value"] > 0 else None
    if top is not None:
        gains = []
        if top["avoided_leavers"] > 0:
            gains.append(f"about {top['avoided_leavers']:,.0f} fewer people leaving")
        if top["avoided_absence_days"] > 0:
            gains.append(f"{top['avoided_absence_days']:,.0f} fewer absence days")
        money_points.append(
            f"Improving {top['label']} for the {top['people_below']:,.0f} people who score it "
            f"low is linked to {' and '.join(gains)} a year, worth about "
            f"{money(top['yearly_value'], c)}.")
    brief["money"] = money_points

    # Where to act.
    fix = factors[factors["priority"] == "Fix first"] if not factors.empty else factors
    protect = factors[factors["priority"] == "Protect"] if not factors.empty else factors
    act = []
    if not fix.empty:
        act.append("Fix first: " + ", ".join(
            f"{r['label']} ({r['agree_pct']:.0%} agree today)" for _, r in fix.iterrows()) + ".")
    if not protect.empty:
        act.append("Protect: " + ", ".join(
            f"{r['label']} ({r['agree_pct']:.0%} agree)" for _, r in protect.iterrows())
            + ". It pays off and is already strong.")
    teams = team_scorecard(df, analysis)
    priority = teams[teams["priority"]] if not teams.empty and "priority" in teams else pd.DataFrame()
    if not priority.empty:
        line = f"{len(priority)} of {len(teams)} teams need support first"
        if "department" in priority.columns:
            depts = priority["department"].value_counts()
            line += " (most in " + " and ".join(depts.index[:2]) + ")"
        act.append(line + ": low on their focus question and worse turnover or absence.")
    brief["where_to_act"] = act

    # Bottom line.
    if top is not None:
        brief["bottom_line"] = (f"{engagement}. The biggest lever is {top['label']}: "
                                f"lifting it is worth about {money(top['yearly_value'], c)} a year.")
    else:
        brief["bottom_line"] = engagement + "." + (
            "" if analysis is not None else " Run the statistical analysis to see what drives "
            "results and what it is worth.")

    # Decisions for the CEO.
    decisions = []
    if top is not None:
        decisions.append(f"Back a company-wide push on {top['label']} this quarter, with one "
                         "executive owner.")
    if not priority.empty:
        decisions.append(f"Ask the managers of the {len(priority)} priority teams for an action "
                         "plan within 30 days (ready-made on the Recommendations page).")
    if traps and traps.get("nonresponse", {}).get("status") == "caution":
        decisions.append("Raise survey participation next time: the people who did not answer "
                         "look different from those who did.")
    elif not protect.empty:
        decisions.append(f"Protect {protect.iloc[0]['label']}: keep what already works as you change.")
    decisions.append("Review progress at the next survey against these numbers.")
    brief["decisions"] = decisions[:3]

    # Confidence.
    confidence = []
    if analysis is not None:
        f = analysis["findings"]
        confidence.append(f"{int((f['robustness_status'] == 'ROBUST').sum())} findings passed every "
                          f"statistical check ({analysis['sample_size']:,} employees, "
                          f"{brief['model_survey']}).")
    for name, sentence in TRAP_PLAIN.items():
        trap = (traps or {}).get(name)
        if trap and trap["status"] in ("trap", "caution"):
            confidence.append(sentence)
    confidence.append("These are links found in your data, not proof of cause; money figures "
                      "use cost assumptions you can change on the Financials page.")
    brief["confidence"] = confidence
    return brief


# PROCESS 4 – DOCUMENT

def brief_markdown(brief: dict[str, Any], title: str = "Engagement brief for the CEO") -> str:
    """The brief as a markdown document ('$' escaped for Streamlit when shown there)."""
    lines = [f"# {title}", f"*{brief['survey']}*", "", f"**{brief['bottom_line']}**", "",
             "## The numbers", "| Measure | Now | Since last survey |", "|---|---|---|"]
    lines += [f"| {s['label']} | {s['value']} | {s['change']} |" for s in brief["scorecard"]]
    sections = [("What drives results", brief["what_matters"]),
                ("What it is worth", brief["money"]),
                ("Where to act", brief["where_to_act"])]
    for heading, points in sections:
        if points:
            lines += ["", f"## {heading}"] + [f"- {p}" for p in points]
    lines += ["", "## Decisions for you"] + [f"{i}. {d}" for i, d in enumerate(brief["decisions"], 1)]
    lines += ["", "## How sure are we"] + [f"- {p}" for p in brief["confidence"]]
    return "\n".join(lines) + "\n"


def brief_for_screen(brief: dict[str, Any]) -> dict[str, Any]:
    """Same brief with '$' escaped so Streamlit markdown shows money correctly."""
    def esc(value):
        if isinstance(value, str):
            return md(value)
        if isinstance(value, list):
            return [esc(v) for v in value]
        if isinstance(value, dict):
            return {k: esc(v) for k, v in value.items()}
        return value
    return esc(brief)
