"""Verified facts from every page, packaged for the Ask AI agent.

The agent already answers from the statistical findings (analytics). This
module adds the results the other pages compute, as plain, display-ready
facts (percent as 0-100, money as "$5.5M", scores rounded), so the AI can
combine them into one consistent answer without recalculating anything:

  overview         latest survey: scores, engagement, outcomes
  trends           what changed between surveys + change-over-time verdicts
  teams            watchlist: teams that need support first, risk bands
  predictions      turnover risk by department, expected leavers, forecast
  recommendations  productivity levers and each team's first step
  financials       cost today, what engagement could be worth, where costs sit
  traps            misleading department effects, hidden manager effects,
                   nonresponse bias (traps.run_trap_checks)

Privacy: no employee IDs or individual scores; teams of 5+ people only.

route_sections(question) picks the sections a question needs, and
check_numbers(answer, evidence) lists any figure in the answer that does not
appear in the evidence.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd

from .action_tools import DISCLAIMER, plan_resources_summary
from .dashboard import kpi_tiles, team_watchlist, where_we_stand
from .financial import (
    Assumptions,
    current_costs,
    engagement_value,
    money,
    team_excess_cost,
)
from .insights import ACTION_LIBRARY, ITEM_LABELS, analysis_frame, explain_finding, team_scorecard
from .longitudinal import (
    INDEX,
    explain_item_change,
    item_change,
    outcome_summary,
    survey_waves,
)
from .prediction import (
    TARGETS,
    employee_risk_by_group,
    explain_forecast,
    forecast_engagement,
    team_risk,
)
from .priorities import factor_sentence, factor_value
from .recommendations import company_summary, productivity_levers, team_plans
from .traps import run_trap_checks, trap_summary
from .waves import latest_wave, model_wave, wave_label


# PROCESS 2 – SECTION ROUTING

SECTION_KEYWORDS = {
    "teams": ["team", "manager", "worry", "watchlist", "priority", "support first",
              "struggling", "which group", "where to focus", "who needs"],
    "predictions": ["predict", "risk", "forecast", "next year", "next survey", "future",
                    "likely", "expected", "will leave", "going to", "heading"],
    "financials": ["cost", "money", "$", "dollar", "saving", "worth", "roi", "budget",
                   "profit", "financial", "spend", "expensive", "value",
                   "invest", "avoid", "prioriti", "biggest impact", "pay off"],
    "trends": ["trend", "change", "over time", "improving", "improved", "declin", "wave", "since",
               "last survey", "last year", "getting better", "getting worse", "history"],
    "recommendations": ["recommend", "what should", "plan", "action", "productivity",
                        "lever", "do next", "improve performance", "first step", "how to",
                        "gallup", "resource", "learning", "training", "course", "article",
                        "action plan", "read more", "learn more"],
    "traps": ["trap", "bias", "mislead", "simpson", "nonresponse", "non-response",
              "response rate", "confound", "trust", "reliable", "department effect",
              "manager effect", "who answered", "didn't answer", "did not answer"],
}

BROAD_WORDS = ["summary", "summarize", "overall", "everything", "big picture",
               "dashboard", "report", "brief me", "key takeaways", "ceo",
               "executive", "board"]


def route_sections(question: str) -> list[str]:
    """Sections this question needs. Overview is always included."""
    q = question.lower()
    if any(word in q for word in BROAD_WORDS):
        return ["overview"] + list(SECTION_KEYWORDS)
    chosen = [name for name, words in SECTION_KEYWORDS.items() if any(w in q for w in words)]
    return ["overview"] + chosen


# PROCESS 3 – SECTION BUILDERS (each returns plain, JSON-safe facts)

def _pct(x: float) -> float:
    return round(float(x) * 100, 1)


def key_findings(analysis: dict | None, df) -> list[str]:
    """The statistical findings as plain sentences (for broad questions)."""
    if analysis is None:
        return ["Statistical analysis not run yet."]
    sentences = []
    for _, row in analysis["findings"].iterrows():
        e = explain_finding(row, df)
        sentences.append(f"{e['outcome_label']} · {e['label']} ({e['item']}): {e['headline']} "
                         + ("Passed all checks." if e["is_robust"] else
                            "Use with caution, not a priority."))
    return sentences or ["No engagement question met the reporting rules."]


def _overview(df, analysis, change, a) -> dict:
    tiles = {t["label"]: {"value": t["value"], "change_vs_last_survey": t["delta"]}
             for t in kpi_tiles(df)}
    return {"latest_survey": wave_label(latest_wave(df)),
            "headline_numbers": tiles,
            "where_we_stand": [p.replace("**", "") for p in where_we_stand(df)],
            "key_findings": key_findings(analysis, df)}


def _trends(df, analysis, change, a) -> dict:
    if len(survey_waves(df)) < 2:
        return {"note": "Only one survey wave, so there is no trend yet."}
    same = "employee_id" in df.columns
    moved = item_change(df, same_employees=same)
    table = moved["table"]
    section = {
        "compared": f"{wave_label(moved['before'])} to {wave_label(moved['after'])}",
        "people_compared": int(moved["n_employees"]),
        "summary": explain_item_change(moved)["headline"],
        "questions_that_clearly_changed": [
            {"question": r["name"], "change_in_average": round(float(r["change"]), 2)}
            for _, r in table[(table["measure"] != INDEX) & table["clear_change"]].iterrows()],
    }
    if change is not None and not change["results"].empty:
        section["is_change_linked_to_results"] = [
            {"outcome": c["outcome"], "verdict": c["verdict"],
             "evidence": [f"{l['model']}: {l['text']} ({'clear' if l['clear'] else 'not clear'})"
                          for l in c["lines"]]}
            for c in outcome_summary(change["results"])]
    else:
        section["is_change_linked_to_results"] = "Change-over-time analysis not run yet."
    return section


def _risk_results(df) -> dict:
    results = {}
    for outcome, (column, _) in TARGETS.items():
        if column in df.columns:
            try:
                results[outcome] = team_risk(df, outcome)
            except ValueError:
                results[outcome] = None
    return results


def _teams(df, analysis, change, a, risk_results=None) -> dict:
    scorecard = team_scorecard(df, analysis)
    if scorecard.empty:
        return {"note": "No teams with 5+ people."}
    risk_results = risk_results if risk_results is not None else _risk_results(df)
    watch = team_watchlist(scorecard, risk_results)
    plain = {"useful": "passed its test on unseen data",
             "weak": "did not pass its test, so its risk bands are not used",
             "untested": "could not be tested yet"}
    models = {f"{o} risk model": plain.get(r["status"], r["status"]) if r else "unavailable"
              for o, r in risk_results.items()}
    top = watch[watch["needs_attention"]].head(5)

    def describe(r):
        risks = [f"{o} risk {r[f'{o}_risk'].lower()}" for o in ("turnover", "absence")
                 if r[f"{o}_risk"] in ("High", "Elevated")]
        return (f"{r['team_id']} ({r.get('department', '')}, {int(r['employees'])} people): "
                f"average Q12 {r['q12_average']:.2f}"
                + (", " + " and ".join(risks) if risks else "")
                + f"; focus question {r['focus_label']}"
                + ("; priority team" if r["priority"] else "") + ".")

    return {
        "teams_reported": int(len(watch)),
        "priority_teams": int(watch["priority"].sum()),
        "high_turnover_risk_teams": int((watch["turnover_risk"] == "High").sum()),
        "high_absence_risk_teams": int((watch["absence_risk"] == "High").sum()),
        "risk_model_test_result": models,
        "most_urgent_teams": [describe(r) for _, r in top.iterrows()],
        "how_urgency_is_ranked": "Priority flag first, then high or elevated predicted risk.",
    }


def _predictions(df, analysis, change, a) -> dict:
    section: dict[str, Any] = {}
    if "department" in df.columns:
        try:
            risk = employee_risk_by_group(df, "department")
            if risk["status"] == "useful":
                t = risk["table"]
                section["turnover_risk_by_department"] = [
                    {"department": r["department"],
                     "percent_in_riskiest_fifth": _pct(r["high_risk_share"]),
                     "expected_leavers_next_12_months": int(r["expected_leavers"])}
                    for _, r in t.iterrows()]
                section["expected_leavers_total"] = int(t["expected_leavers"].sum())
                section["model_accuracy_percent"] = _pct(risk["auc"])
                section["fairness_check"] = ("pending" if t["actual_rate"].isna().all()
                                             else "passed" if not t["fairness_flag"].any()
                                             else "flagged: " + ", ".join(
                                                 t.loc[t["fairness_flag"], "department"]))
                gap = t["calibration_gap"].dropna()
                if len(gap) and (gap > 0).all():
                    section["calibration_note"] = (
                        "When tested, the model predicted slightly more leavers than happened, "
                        "so expected leavers are a ceiling.")
            else:
                section["turnover_risk_by_department"] = f"Model did not pass its test: {risk['status_text']}"
        except ValueError as exc:
            section["turnover_risk_by_department"] = str(exc)
    forecast = forecast_engagement(df)
    section["engagement_forecast"] = explain_forecast(forecast).replace("**", "")
    if forecast["available"]:
        section["forecast_numbers"] = {"next_survey": round(forecast["forecast"], 2),
                                       "likely_low": round(forecast["low"], 2),
                                       "likely_high": round(forecast["high"], 2)}
    return section


def _recommendations(df, analysis, change, a, risk_results=None) -> dict:
    if analysis is None or "performance_rating" not in df.columns:
        return {"note": "Statistical analysis not run, or no performance data."}
    levers = productivity_levers(df, analysis, change)
    if levers.empty:
        return {"note": "No question has a verified link to performance yet."}
    plans = team_plans(df, levers, risk_results)
    summary = company_summary(plans, levers)

    # Questions whose single-survey link weakens once past performance is
    # considered: the same-people-over-time evidence is the better guide.
    findings = analysis["findings"]
    caution = set(findings.loc[(findings["outcome"] == "performance")
                               & (findings["robustness_status"] != "ROBUST"), "item"])

    def lever_sentence(r):
        actions = ACTION_LIBRARY.get(r["item"], {}).get("actions", [])[:2]
        text = (f"{r['label']} ({r['item']}): about {r['points_per_point']:+.2f} rating points "
                f"per 1-point higher score. Evidence: {r['evidence'].lower()} "
                f"({r['source'].lower()}).")
        if r["item"] in caution and "Same employees" in r["source"]:
            text += (" In the single-survey model this link weakens once past performance "
                     "is considered; following the same people over time is the better guide.")
        if actions:
            text += " What managers can do: " + " ".join(actions)
        return text

    biggest = plans[plans["tier"] == "Biggest opportunity"]
    keep = int((plans["tier"] == "Keep strengths").sum())
    starts = summary["starts"]
    facts = [f"Productivity is measured by the performance rating (1–5)."]
    if not starts.empty:
        top = starts.iloc[0]
        facts.append(f"Most common first step: {top['label']} ({top['item']}); "
                     f"{int(top['teams'])} teams should start there.")
    if not biggest.empty:
        facts.append(f"{len(biggest)} teams form the biggest-opportunity group (the top 20% "
                     f"by expected gain); their first step is worth about "
                     f"{biggest['estimated_gain'].mean():+.2f} rating points each on average.")
    if keep:
        facts.append(f"{keep} teams are already at or above the company on every lever; "
                     "their managers should protect and share what works.")

    return {
        "key_facts": facts,
        "strongest_levers_with_actions": [lever_sentence(r) for _, r in levers.head(4).iterrows()],
        "biggest_opportunity_teams": [
            f"{r['team_id']} ({r['department']}): start with "
            f"{ITEM_LABELS.get(r['start_with'], r['start_with'])} ({r['start_with']}); "
            f"performance {r['performance']:.2f}, expected about {r['estimated_gain']:+.2f} "
            "rating points if it closes the gap"
            for _, r in biggest.head(5).iterrows()],
        "note": "Gains are estimates from links in the data, for the first step only; "
                "they are not added across questions.",
        "gallup_resources": {
            "links": plan_resources_summary(list(levers["item"].head(4))),
            "how_to_use": "Teams can turn their first step into a Gallup Access action plan "
                          "(goal, next steps, owner, due date, success indicator, follow-up); "
                          "the Recommendations page downloads these plans. " + DISCLAIMER,
        },
    }


def _financials(df, analysis, change, a, risk_results=None) -> dict:
    c = a.currency
    costs = current_costs(df, a)
    section: dict[str, Any] = {
        "assumptions_used": {
            "average_salary": money(a.avg_salary, c),
            "replacement_cost_percent_of_salary": round(a.replacement_cost_pct * 100),
            "cost_per_absence_day": money(a.day_cost, c),
            "headcount": int(costs["headcount"]),
            "note": "Cost assumptions entered on the Financials page (defaults are placeholders).",
        },
        "cost_today_per_year": {
            "total_people_cost": money(costs["total_cost"], c),
            "voluntary_turnover": money(costs.get("turnover_cost", 0), c),
            "unplanned_absence": money(costs.get("absence_cost", 0), c),
        },
    }
    if analysis is not None:
        value = engagement_value(df, analysis, a)
        section["what_engagement_could_be_worth_per_year"] = {
            "total": money(value["total"], c),
            "scenarios": [{"area": r["source"], "assumes": r["what"],
                           "result": f"{r['units']:,.0f} {r['unit_label']}",
                           "value": money(r["value"], c)}
                          for _, r in value["scenarios"].iterrows()],
        }
    if analysis is not None:
        factors = factor_value(df, analysis, a)
        if not factors.empty:
            section["factors_ranked_by_value"] = {
                "ranking": [factor_sentence(r, a) for _, r in factors.iterrows()],
                "how_ranked": ("Yearly value = value per person who moves up one point × "
                               "people answering 1-3 today. Fix first = high impact and low "
                               "score; Protect = high impact, already strong. Values overlap "
                               "across factors, so never add them together."),
            }
    teams = team_excess_cost(df, a)
    above = teams[teams["excess_cost"] > 0]
    if not above.empty:
        section["costliest_teams_vs_company_rate"] = [
            f"{r['team_id']} ({r.get('department', '')}): {money(r['excess_cost'], c)} a year "
            "above what the company rate would cost"
            for _, r in above.head(5).iterrows()]
        section["where_costs_sit"] = (
            f"{len(above)} teams cost {money(above['excess_cost'].sum(), c)} a year more in "
            "turnover and absence than the company rate would for teams their size. This is "
            "where today's costs are concentrated, not a saving or a forecast.")
    section["limit"] = ("Estimates from the user's cost assumptions, not measured profit; "
                        "the data has no revenue or profit.")
    return section


def _traps(df, analysis, change, a) -> dict:
    findings = analysis["findings"] if analysis is not None else None
    traps = run_trap_checks(df, findings)
    return {"survey": traps["wave"],
            "checks": trap_summary(traps),
            "note": "The reported findings already compare people within departments. "
                    "Team (manager) effects and nonresponse are not adjusted for in the "
                    "findings, so mention them when they apply."}


BUILDERS: dict[str, Callable] = {
    "overview": _overview,
    "trends": _trends,
    "teams": _teams,
    "predictions": _predictions,
    "recommendations": _recommendations,
    "traps": _traps,
    "financials": _financials,
}


# PROCESS 4 – BUILD THE CONTEXT

def build_context(df: pd.DataFrame, analysis: dict | None = None,
                  change_analysis: dict | None = None,
                  assumptions: Assumptions | None = None,
                  sections: list[str] | None = None) -> dict[str, Any]:
    """Facts for the requested sections (all by default). A section that cannot
    be built is replaced by a short note instead of failing the whole answer."""
    a = assumptions or Assumptions(headcount=len(analysis_frame(df)))
    wanted = sections or list(BUILDERS)
    needs_risk = {"teams", "recommendations"} & set(wanted)
    risk_results = _risk_results(df) if needs_risk else None
    context: dict[str, Any] = {"model_survey": wave_label(model_wave(df))}
    for name in wanted:
        builder = BUILDERS[name]
        try:
            if name in ("teams", "recommendations"):
                context[name] = builder(df, analysis, change_analysis, a, risk_results)
            else:
                context[name] = builder(df, analysis, change_analysis, a)
        except Exception as exc:          # one failed section must not block the answer
            context[name] = {"note": f"Not available: {exc}"}
    return context


# PROCESS 5 – NUMBER CHECK

NUMBER = re.compile(r"(?<![A-Za-z\d.])\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in NUMBER.findall(text)}


def check_numbers(answer: str, evidence: Any, question: str = "") -> list[float]:
    """Figures in the answer that do not appear (at any rounding) in the evidence
    or the question. Small counting words like 1-12 are ignored."""
    source = json.dumps(evidence, default=str) + " " + question
    allowed: set[float] = set()
    for value in _numbers(source):
        for digits in (0, 1, 2):
            allowed.add(round(value, digits))
            allowed.add(round(value * 100, digits))     # 0.38 written as 38%
    unverified = []
    for value in _numbers(answer):
        if value <= 12 and float(value).is_integer():
            continue                                     # list numbers, "Q04", "3 teams"
        if not any(abs(value - x) < 0.051 for x in allowed):
            unverified.append(value)
    return sorted(set(unverified))
