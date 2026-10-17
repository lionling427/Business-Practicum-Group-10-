# PROCESS 1 – IMPORTS
"""Turn verified statistical findings into decisions a CEO or manager can use.

Everything here is computed from the verified findings produced by
analytics.analyze_dataframe() plus simple descriptive summaries of the data.
No new statistical claims are made: effects come only from the models, and
every number shown in the app can be traced back to a calculation ID.

Main functions
- analysis_frame(df):        the same rows the models used
- explain_finding(row, df):  one plain-English sentence per finding
- findings_table(analysis, df): business-friendly results table
- top_drivers(analysis):     strongest verified factor per outcome
- dollar_impact(...):        illustrative cost of the gap, with assumptions shown
- team_scorecard(df, ...):   per-team scores, outcomes, focus item, priority flag
- manager_brief(...):        strengths, focus item, actions, conversation guide
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .data_ingestion import Q12_ITEMS
from .waves import model_wave


# PROCESS 2 – LABELS AND SETTINGS

ITEM_LABELS = {
    "Q01": "Know what is expected", "Q02": "Materials and equipment",
    "Q03": "Opportunity to do best", "Q04": "Recognition",
    "Q05": "Someone cares", "Q06": "Development encouraged",
    "Q07": "Opinions count", "Q08": "Mission and purpose",
    "Q09": "Co-workers committed to quality", "Q10": "Best friend at work",
    "Q11": "Progress conversations", "Q12": "Learn and grow",
}

OUTCOME_INFO = {
    "turnover": {"column": "left_voluntarily_12m", "label": "Voluntary turnover",
                 "good": "lower", "unit": "of employees left within 12 months"},
    "absence": {"column": "unplanned_absence_days_6m", "label": "Unplanned absence",
                "good": "lower", "unit": "unplanned absence days per employee (6 months)"},
    "performance": {"column": "performance_rating", "label": "Performance",
                    "good": "higher", "unit": "average performance rating (1-5)"},
}

STATUS_BADGES = {
    "ROBUST": "✅ Passed all checks",
    "CAUTION": "⚠️ Use with caution",
    "NOT_REPORTABLE": "Not reportable",
}

MIN_TEAM_SIZE = 5          # privacy rule: never show smaller teams
PRIORITY_SHARE = 0.05      # flag the top 5% of teams (at least 5) as "act first"
WORKDAYS_PER_YEAR = 260

# Practical actions per Q12 item. These are suggested management practices
# written for this project; they are not Gallup's official guidance.
ACTION_LIBRARY = {
    "Q01": {
        "actions": [
            "Write down the top 3 outcomes you expect from each person this quarter and review them 1:1.",
            "Start weekly team meetings by restating this week's priorities.",
            "When priorities change, tell the team the same day and say what drops off the list.",
        ],
        "questions": [
            "What part of your role feels least clear right now?",
            "If you could only get three things done this month, what should they be?",
        ],
    },
    "Q02": {
        "actions": [
            "Ask each person to name one tool, resource or access issue that slows them down.",
            "Fix the top two issues within 30 days and report back on the rest.",
            "Add a 'blockers' check to every 1:1.",
        ],
        "questions": [
            "What slows you down most in a normal week?",
            "What one resource would make the biggest difference to your work?",
        ],
    },
    "Q03": {
        "actions": [
            "Ask each person what tasks energize them and which drain them.",
            "Shift at least one task per person toward what they do best.",
            "Match new projects to people's strengths, not just availability.",
        ],
        "questions": [
            "When did you last feel you were doing your best work? What were you doing?",
            "Which part of your job would you hand off if you could?",
        ],
    },
    "Q04": {
        "actions": [
            "Recognize one specific contribution per person every week, naming what they did and why it mattered.",
            "Ask each person how they prefer to be recognized (public, private, written).",
            "Open team meetings with a two-minute shout-out round.",
        ],
        "questions": [
            "What recent work are you proud of that may have gone unnoticed?",
            "How do you like to receive recognition?",
        ],
    },
    "Q05": {
        "actions": [
            "Hold a short 1:1 with each person that is about them, not about tasks.",
            "Follow up on personal things people have shared (a move, a family event).",
            "Check workload and wellbeing before deadlines, not after.",
        ],
        "questions": [
            "How are you doing, really, outside of the work itself?",
            "What could I do to make your week more manageable?",
        ],
    },
    "Q06": {
        "actions": [
            "Agree one development goal with each person and review it monthly.",
            "Pair people with a mentor or stretch assignment.",
            "Share open roles and projects that could grow their skills.",
        ],
        "questions": [
            "Where would you like to be in two years?",
            "Which skill do you most want to build this year?",
        ],
    },
    "Q07": {
        "actions": [
            "Ask for input before decisions, not after.",
            "Close the loop: tell people which ideas were used and why others were not.",
            "Rotate who leads a part of the team meeting.",
        ],
        "questions": [
            "What is one thing we should change about how this team works?",
            "When did you last feel your opinion was ignored?",
        ],
    },
    "Q08": {
        "actions": [
            "Connect each project to the customer or community it serves.",
            "Share a real customer story or result at least once a month.",
            "Explain the 'why' behind every new priority.",
        ],
        "questions": [
            "How does your work help our customers?",
            "What part of our mission matters most to you?",
        ],
    },
    "Q09": {
        "actions": [
            "Agree as a team what 'good quality' looks like for your main outputs.",
            "Address repeated quality issues directly and early.",
            "Recognize quality work publicly, not just speed.",
        ],
        "questions": [
            "Where do you see quality slipping on this team?",
            "What standard should we hold each other to?",
        ],
    },
    "Q10": {
        "actions": [
            "Create regular time for the team to connect beyond tasks.",
            "Pair people on projects across usual working groups.",
            "Welcome new hires with a peer buddy for their first 90 days.",
        ],
        "questions": [
            "Who on the team do you rely on most?",
            "How could we make it easier to get to know each other?",
        ],
    },
    "Q11": {
        "actions": [
            "Hold a progress conversation with each person at least every 6 months, ideally quarterly.",
            "Focus the conversation on progress and next steps, not just ratings.",
            "Write down agreed next steps and revisit them.",
        ],
        "questions": [
            "What progress are you most proud of in the last six months?",
            "What should we focus on for your next six months?",
        ],
    },
    "Q12": {
        "actions": [
            "Give each person one new learning opportunity this quarter.",
            "Share lessons learned after each project as a team.",
            "Budget time for learning, not only money.",
        ],
        "questions": [
            "What have you learned at work this year?",
            "What would you like to learn next?",
        ],
    },
}


# PROCESS 3 – DATA HELPERS

def analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows the models use: respondents only, newest wave with outcomes
    (waves.model_wave), so model results describe the current workforce."""
    out = df.copy()
    if "responded" in out.columns:
        out = out[pd.to_numeric(out["responded"], errors="coerce") == 1]
    wave = model_wave(out)
    if wave is not None:
        out = out[out["wave"] == wave]
    for item in [i for i in Q12_ITEMS if i in out.columns]:
        out[item] = pd.to_numeric(out[item], errors="coerce")
    return out.copy()


def _rate(df: pd.DataFrame, outcome: str) -> float | None:
    col = OUTCOME_INFO[outcome]["column"]
    if col not in df.columns:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _fmt_outcome(outcome: str, value: float) -> str:
    if outcome == "turnover":
        return f"{value * 100:.1f}%"
    if outcome == "absence":
        return f"{value:.1f} days"
    return f"{value:.2f}"


# PROCESS 4 – PLAIN-LANGUAGE FINDINGS

def explain_finding(row: pd.Series | dict, df: pd.DataFrame) -> dict[str, Any]:
    """Translate one model result into a headline a nontechnical reader understands.

    Model-based numbers hold the other factors in the model constant and describe
    an association, not cause and effect.
    """
    item, outcome = row["item"], row["outcome"]
    label = ITEM_LABELS.get(item, item)
    effect = float(row["effect"])
    data = analysis_frame(df)
    base = _rate(data, outcome)

    if outcome == "turnover" and base:
        odds = base / (1 - base) * effect
        new = odds / (1 + odds)
        change = (effect - 1) * 100
        headline = (f"Each 1-point higher score on {label} is linked to "
                    f"{abs(change):.0f}% {'lower' if change < 0 else 'higher'} odds of leaving.")
        example = (f"For a typical employee ({base * 100:.1f}% chance of leaving), one point higher "
                   f"is linked to about {new * 100:.1f}%.")
    elif outcome == "absence" and base is not None:
        new = base * effect
        change = (effect - 1) * 100
        headline = (f"Each 1-point higher score on {label} is linked to "
                    f"{abs(change):.0f}% {'fewer' if change < 0 else 'more'} unplanned absence days.")
        example = f"For a typical employee ({base:.1f} days in 6 months), one point higher is linked to about {new:.1f} days."
    elif outcome == "performance":
        sd_perf = float(data["performance_rating"].std())
        sd_item = float(data[item].std()) if item in data else float("nan")
        per_point = effect * sd_perf / sd_item if sd_item else float("nan")
        size = "small" if abs(effect) < 0.10 else "moderate" if abs(effect) < 0.30 else "large"
        headline = (f"Higher {label} scores are linked to {'higher' if effect > 0 else 'lower'} "
                    f"performance ratings ({size} link).")
        example = (f"Each 1-point higher score is linked to about {per_point:+.2f} rating points, "
                   f"after accounting for prior performance.")
    else:
        headline, example = f"{label} is linked to {outcome}.", ""

    reasons = row.get("diagnostic_reasons", "")
    if isinstance(reasons, (list, tuple)):
        reasons = "; ".join(str(r) for r in reasons if r)
    if not isinstance(reasons, str) or reasons.lower() in ("nan", "none", "[]"):
        reasons = ""

    return {"item": item, "label": label, "outcome": outcome,
            "outcome_label": OUTCOME_INFO[outcome]["label"],
            "headline": headline, "example": example,
            "status": STATUS_BADGES.get(row.get("robustness_status"), str(row.get("robustness_status"))),
            "is_robust": row.get("robustness_status") == "ROBUST",
            "caution_reason": reasons}


def findings_table(analysis: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Business-friendly version of the findings table."""
    findings = analysis["findings"]
    if findings.empty:
        return pd.DataFrame()
    rows = []
    for _, r in findings.iterrows():
        e = explain_finding(r, df)
        rows.append({"Outcome": e["outcome_label"],
                     "Engagement factor": f"{e['item']} · {e['label']}",
                     "What the data shows": e["headline"],
                     "Example": e["example"],
                     "Reliability": e["status"],
                     "Why caution": e["caution_reason"] if not e["is_robust"] else ""})
    order = {"Voluntary turnover": 0, "Unplanned absence": 1, "Performance": 2}
    out = pd.DataFrame(rows)
    out["_o"] = out["Outcome"].map(order)
    out["_r"] = out["Reliability"].str.startswith("✅").map({True: 0, False: 1})
    return out.sort_values(["_o", "_r"]).drop(columns=["_o", "_r"]).reset_index(drop=True)


def _robust_favorable(findings: pd.DataFrame) -> pd.DataFrame:
    """Findings that passed all checks AND point the good way (a higher score
    goes with lower turnover/absence or higher performance)."""
    robust = findings[findings["robustness_status"] == "ROBUST"]
    good_lower = robust["outcome"].map(lambda o: OUTCOME_INFO[o]["good"] == "lower")
    favorable = (good_lower & (robust["effect"] < 1)) | (~good_lower & (robust["effect"] > 0))
    return robust[favorable]


def top_drivers(analysis: dict) -> dict[str, dict]:
    """Strongest favorable finding per outcome among those that passed all checks."""
    f = _robust_favorable(analysis["findings"])
    out = {}
    for outcome in ["turnover", "absence", "performance"]:
        sub = f[f["outcome"] == outcome].copy()
        if sub.empty:
            continue
        if outcome == "performance":
            sub["strength"] = sub["effect"].abs()
        else:
            sub["strength"] = sub["effect"].apply(lambda e: abs(math.log(e)))
        out[outcome] = sub.sort_values("strength", ascending=False).iloc[0].to_dict()
    return out


def verified_items(analysis: dict, outcome: str | None = None) -> list[str]:
    f = _robust_favorable(analysis["findings"])
    if outcome:
        f = f[f["outcome"] == outcome]
    return list(dict.fromkeys(f["item"]))


# PROCESS 5 – DOLLAR IMPACT (ILLUSTRATIVE)

def dollar_impact(
    analysis: dict,
    df: pd.DataFrame,
    avg_salary: float = 60000.0,
    replacement_cost_pct: float = 0.5,
) -> dict[str, Any] | None:
    """Illustrative value of lifting the top turnover factor by 1 point for
    employees who currently answer 1-3 (not agreeing).

    Uses the model's odds ratio and the company's own turnover rate. This is an
    association-based scenario, not a forecast or a guarantee.
    """
    drivers = top_drivers(analysis)
    if "turnover" not in drivers:
        return None
    row = drivers["turnover"]
    item, odds_ratio = row["item"], float(row["effect"])
    data = analysis_frame(df)
    base = _rate(data, "turnover")
    if not base:
        return None

    not_agreeing = int((data[item] <= 3).sum())
    odds = base / (1 - base) * odds_ratio
    new_rate = odds / (1 + odds)
    fewer_leavers = not_agreeing * (base - new_rate)
    cost_per_leaver = avg_salary * replacement_cost_pct
    current_leavers = base * len(data)

    result = {
        "item": item, "label": ITEM_LABELS.get(item, item),
        "employees": len(data), "employees_not_agreeing": not_agreeing,
        "turnover_rate": base, "current_leavers": current_leavers,
        "current_turnover_cost": current_leavers * cost_per_leaver,
        "fewer_leavers": fewer_leavers,
        "savings": fewer_leavers * cost_per_leaver,
        "cost_per_leaver": cost_per_leaver,
        "assumptions": (f"Average salary ${avg_salary:,.0f}; replacing one employee costs "
                        f"{replacement_cost_pct:.0%} of salary; the {not_agreeing:,} employees who "
                        f"answer 1-3 on {item} move up 1 point; other factors stay the same."),
    }

    drivers_abs = drivers.get("absence")
    if drivers_abs is not None and "unplanned_absence_days_6m" in data.columns:
        a_item, irr = drivers_abs["item"], float(drivers_abs["effect"])
        avg_days = float(data["unplanned_absence_days_6m"].mean())
        n_low = int((data[a_item] <= 3).sum())
        days_saved_year = n_low * avg_days * (1 - irr) * 2       # 6-month window -> 1 year
        result.update({"absence_item": a_item,
                       "absence_label": ITEM_LABELS.get(a_item, a_item),
                       "absence_days_saved": days_saved_year,
                       "absence_savings": days_saved_year * avg_salary / WORKDAYS_PER_YEAR})
    return result


# PROCESS 6 – TEAM SCORECARD

def team_scorecard(df: pd.DataFrame, analysis: dict | None = None) -> pd.DataFrame:
    """One row per team (5+ people): Q12 averages, outcomes, focus item, priority flag.

    Focus item = the verified factor where the team is furthest below the company.
    Priority = team is in the bottom quarter on its focus item AND has worse
    turnover or absence than the company average.
    """
    data = analysis_frame(df)
    if "team_id" not in data.columns:
        return pd.DataFrame()
    items = [i for i in Q12_ITEMS if i in data.columns]
    # Count people by employee_id when present; otherwise count rows.
    headcount = ("employee_id", "nunique") if "employee_id" in data.columns else ("team_id", "size")
    agg = {"employees": headcount, **{i: (i, "mean") for i in items}}
    if "department" in data.columns:
        agg["department"] = ("department", lambda s: s.mode().iat[0] if not s.mode().empty else "")
    for outcome, info in OUTCOME_INFO.items():
        if info["column"] in data.columns:
            agg[outcome] = (info["column"], "mean")
    teams = data.groupby("team_id").agg(**agg).reset_index()
    teams = teams[teams["employees"] >= MIN_TEAM_SIZE].copy()
    teams["q12_average"] = teams[items].mean(axis=1)

    company = data[items].mean()
    focus_pool = verified_items(analysis) if analysis else []
    focus_pool = focus_pool or items
    gaps = teams[focus_pool] - company[focus_pool]
    teams["focus_item"] = gaps.idxmin(axis=1)
    teams["focus_gap"] = gaps.min(axis=1)

    cut = {i: teams[i].quantile(0.25) for i in focus_pool}
    low_on_focus = teams.apply(lambda r: r[r["focus_item"]] <= cut[r["focus_item"]], axis=1)

    # Priority score: how far below the company on the focus item, plus how much
    # worse turnover / absence are (all in standard-deviation units).
    score = -teams["focus_gap"] / (teams[focus_pool].std().mean() or 1)
    worse = pd.Series(False, index=teams.index)
    for outcome in ["turnover", "absence"]:
        if outcome in teams.columns:
            excess = teams[outcome] - _rate(data, outcome)
            worse |= excess > 0
            score = score + excess.clip(lower=0) / (teams[outcome].std() or 1)
    teams["meets_priority_rules"] = low_on_focus & worse
    teams["priority_score"] = score.where(teams["meets_priority_rules"], np.nan)
    n_flag = max(5, int(round(len(teams) * PRIORITY_SHARE)))
    flagged = teams["priority_score"].nlargest(n_flag).index
    teams["priority"] = teams.index.isin(flagged)
    teams["focus_label"] = teams["focus_item"].map(ITEM_LABELS)
    return (teams.sort_values(["priority", "priority_score", "focus_gap"],
                              ascending=[False, False, True]).reset_index(drop=True))


# PROCESS 7A – RISK OUTLOOK FOR MANAGERS
#
# Turns a team's prediction (from prediction.team_risk) into plain language:
# how urgent it is, how likely, what pushes the risk up, and what to do.
# Only predictions whose model passed its test are used.

RISK_URGENCY = {
    "High": ("Act this month", "red"),
    "Elevated": ("Keep a close eye this quarter", "orange"),
    "Typical": ("Keep doing what works", "green"),
}

RISK_WHAT = {
    "turnover": ("above-normal voluntary turnover", "in the next 12 months"),
    "absence": ("above-normal unplanned absence", "in the next 6 months"),
}

RISK_STEPS = {
    "turnover": {
        "High": [
            "Hold a stay conversation with each person in the next 2 weeks: ask what "
            "keeps them here and what might make them leave.",
            "Ask HR to review workload, pay fit and growth options for this team this month.",
            "Check in again within 30 days and note what has changed.",
        ],
        "Elevated": [
            "Add one stay question to your regular 1:1s this month: "
            "'What would make you want to stay another year?'",
            "Watch for early signs: fewer ideas in meetings, people going quiet, more sick days.",
        ],
    },
    "absence": {
        "High": [
            "Review workload and schedules with the team in the next 2 weeks.",
            "Ask each person privately whether anything is making it hard to be at work.",
            "Agree one change that reduces pressure and check in again within 30 days.",
        ],
        "Elevated": [
            "Ask about workload and wellbeing in every 1:1 this month.",
            "Look at the team's absence pattern (not individuals) again next month.",
        ],
    },
}

RISK_CAUTION = (
    "This is a likelihood based on patterns across the company, not a certainty and "
    "not a judgment of anyone on the team. Keep the risk label private: share the "
    "actions, not the score."
)


def _in_ten(probability: float) -> str:
    """0.87 -> '9 in 10' (kept between 1 and 9 so it never reads as certain)."""
    return f"{max(1, min(9, round(probability * 10)))} in 10"


def risk_outlook(team_id: str, risk_results: dict[str, dict] | None,
                 focus_item: str | None = None) -> dict[str, Any] | None:
    """Plain-language risk outlook and recommended steps for one team.

    risk_results maps outcome -> prediction.team_risk() result. Results whose
    model did not pass its test are ignored. Returns None when nothing usable.
    """
    if not risk_results:
        return None
    order = {"High": 0, "Elevated": 1, "Typical": 2}
    outcomes = []
    for outcome, result in risk_results.items():
        if result is None or result.get("status") != "useful":
            continue
        teams = result["teams"]
        row = teams[teams["team_id"] == team_id]
        if row.empty:
            continue
        row = row.iloc[0]
        what, when = RISK_WHAT[outcome]
        drivers = [d.split(" ", 1)[0] for d in str(row["risk_drivers"]).split(", ") if d]
        outcomes.append({
            "outcome": outcome, "what": what, "when": when, "band": row["risk_band"],
            "risk_pct": int(round(float(row["risk"]) * 100)),
            "typical_pct": int(round(float(teams["risk"].median()) * 100)),
            "chance": _in_ten(float(row["risk"])),
            "typical_chance": _in_ten(float(teams["risk"].median())),
            "drivers": [{"item": d, "label": ITEM_LABELS.get(d, d)} for d in drivers],
        })
    if not outcomes:
        return None

    outcomes.sort(key=lambda o: order[o["band"]])
    worst = outcomes[0]
    urgency, color = RISK_URGENCY[worst["band"]]

    headline = (f"This team has about a {worst['chance']} chance of {worst['what']} "
                f"{worst['when']}, compared with about {worst['typical_chance']} for a "
                "typical team.")
    for other in outcomes[1:]:
        if other["band"] != "Typical":
            headline += (f" It also has about a {other['chance']} chance of {other['what']} "
                         f"(typical: about {other['typical_chance']}).")

    steps: list[str] = []
    for o in outcomes:
        for step in RISK_STEPS[o["outcome"]].get(o["band"], []):
            if step not in steps:
                steps.append(step)

    driver_note = ""
    if worst["drivers"]:
        top = worst["drivers"][0]
        if top["item"] == focus_item:
            driver_note = (f"Your focus question, {top['label']} ({top['item']}), is also what "
                           "pushes this risk up most, so working on it covers both.")
        else:
            driver_note = (f"The answers pushing this risk up most are on {top['label']} "
                           f"({top['item']}). Also try: "
                           f"{ACTION_LIBRARY[top['item']]['actions'][0]}")
    if worst["band"] == "Typical":
        steps = ["Keep doing what works and look at this again after the next survey."]
        driver_note = ""

    return {"urgency": urgency, "color": color, "band": worst["band"], "headline": headline,
            "outcomes": outcomes, "steps": steps[:3], "driver_note": driver_note,
            "caution": RISK_CAUTION}


# PROCESS 7 – MANAGER BRIEF

def manager_brief(df: pd.DataFrame, team_id: str, analysis: dict | None = None,
                  risk_results: dict[str, dict] | None = None) -> dict[str, Any]:
    """Facts for one team: strengths, focus item, outcomes vs company, actions."""
    data = analysis_frame(df)
    teams = team_scorecard(df, analysis)
    if teams.empty or team_id not in set(teams["team_id"]):
        raise ValueError(f"Team {team_id} not found or has fewer than {MIN_TEAM_SIZE} people.")
    t = teams.set_index("team_id").loc[team_id]
    items = [i for i in Q12_ITEMS if i in data.columns]
    company = data[items].mean()
    diff = (t[items].astype(float) - company).sort_values(ascending=False)

    strengths = [{"item": i, "label": ITEM_LABELS[i], "team": round(float(t[i]), 2),
                  "company": round(float(company[i]), 2)} for i in diff.index[:2]]
    focus = t["focus_item"]
    focus_outcomes = []
    if analysis:
        f = _robust_favorable(analysis["findings"])
        focus_outcomes = f[f["item"] == focus]["outcome"].tolist()

    outcomes = {}
    for outcome in ["turnover", "absence", "performance"]:
        if outcome in t.index and pd.notna(t[outcome]):
            outcomes[outcome] = {"label": OUTCOME_INFO[outcome]["label"],
                                 "team": _fmt_outcome(outcome, float(t[outcome])),
                                 "company": _fmt_outcome(outcome, _rate(data, outcome))}

    return {"team_id": team_id, "department": t.get("department", ""),
            "employees": int(t["employees"]),
            "q12_average": round(float(t["q12_average"]), 2),
            "company_q12_average": round(float(company.mean()), 2),
            "strengths": strengths,
            "focus": {"item": focus, "label": ITEM_LABELS[focus],
                      "team": round(float(t[focus]), 2), "company": round(float(company[focus]), 2),
                      "linked_outcomes": [f"{OUTCOME_INFO[o]['good']} {OUTCOME_INFO[o]['label'].lower()}"
                                          for o in focus_outcomes]},
            "outcomes": outcomes, "priority": bool(t["priority"]),
            "strengths_heading": ("What is going well"
                                  if strengths and all(x["team"] >= x["company"] for x in strengths)
                                  else "Relative strengths (the team's best questions; "
                                       "some are still below the company average)"),
            "risk": risk_outlook(team_id, risk_results, focus),
            "actions": ACTION_LIBRARY[focus]["actions"],
            "questions": ACTION_LIBRARY[focus]["questions"]}


def brief_markdown(brief: dict[str, Any]) -> str:
    """Deterministic manager brief (works without the AI service)."""
    f = brief["focus"]
    lines = [f"### Team {brief['team_id']} · {brief['department']} · {brief['employees']} people",
             f"Average Q12 score **{brief['q12_average']:.2f}** (company {brief['company_q12_average']:.2f}).", ""]
    risk = brief.get("risk")
    if risk:
        lines += [f"**Risk outlook** :{risk['color']}-badge[{risk['urgency']}]", "", risk["headline"]]
        if risk["driver_note"]:
            lines += ["", risk["driver_note"]]
        lines += ["", "**Recommended next steps**"]
        lines += [f"{i}. {step}" for i, step in enumerate(risk["steps"], 1)]
        lines += ["", f"_{risk['caution']}_", ""]
    lines += [
             "**What is going well**" if brief["strengths"][0]["team"] >= brief["strengths"][0]["company"]
             else "**Relative strengths** (the team's best items, still below the company average)"]
    lines += [f"- {s['label']} ({s['item']}): {s['team']:.2f} vs company {s['company']:.2f}" for s in brief["strengths"]]
    linked = f", which is linked to {' and '.join(f['linked_outcomes'])} in your data" if f["linked_outcomes"] else ""
    lines += ["", f"**This month's focus: {f['label']} ({f['item']})**",
              f"Team score {f['team']:.2f} vs company {f['company']:.2f}{linked}.", "",
              "**Three actions to try**"]
    lines += [f"{i}. {a}" for i, a in enumerate(brief["actions"], 1)]
    lines += ["", "**Questions for a 15-minute team conversation**"]
    lines += [f"- {q}" for q in brief["questions"]]
    if brief["outcomes"]:
        lines += ["", "**Team results vs company**"]
        lines += [f"- {o['label']}: {o['team']} (company {o['company']})" for o in brief["outcomes"].values()]
    lines += ["", "_These are associations in your data, not proof of cause and effect. "
                  "Suggested actions are common management practices, not Gallup's official guidance._"]
    return "\n".join(lines)
