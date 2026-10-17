"""Scenario simulator: "if these questions improve, what happens to people and cost?"

The user picks:
  lifts        questions and how many points each rises (on the 1-5 scale)
  who          'everyone' (scores capped at 5) or 'low' (only people answering 1-3)
  department   whole company or one department
  ramp_months  how long the improvement takes to build up (0 = immediately)
  quarters     how far ahead to show the path (4 or 8 quarters)

How it works (no new statistics, only the verified models):
  1. Each outcome's fixed model (model_spec) is fitted on the model survey.
  2. Each employee's outcome is predicted twice: with today's answers and with
     the improved answers. The difference, summed and scaled to headcount, is
     the effect: fewer leavers, fewer absence days, higher ratings.
  3. Only links that passed every check move the numbers. A question without a
     verified link to an outcome counts as zero for that outcome (and says so).
  4. The range repeats step 2 with each link at the edges of its 95% margin
     of error.
  5. The yearly effect is spread over quarters, building up over ramp_months.

Everything else (tenure, department, the other questions) is held as it is
today. These are associations from your data, not guarantees.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .audit import audit_sample
from .financial import Assumptions, money
from .insights import ITEM_LABELS, _robust_favorable
from .model_spec import SPECS, design_matrix, fit_outcome_model
from .visualization import HIGH_COLOR


# PROCESS 2 – CONFIGURATION

WHO = {"everyone": "Everyone (scores capped at 5)", "low": "Only people answering 1–3"}
BOUNDS = ("estimate", "cautious", "optimistic")
MAX_SCORE = 5.0


def _extras(analysis: dict, outcome: str) -> tuple[str, ...]:
    return tuple((analysis.get("adjustments") or {}).get("model_controls_added", {}).get(outcome, []))


def verified_links(analysis: dict) -> pd.DataFrame:
    """Findings that passed every check and point the helpful way."""
    return _robust_favorable(analysis["findings"])


def improved_scores(scores: pd.Series, lift: float, who: str) -> pd.Series:
    """New answers after the lift: everyone or only people answering 1-3, capped at 5."""
    target = scores <= 3 if who == "low" else pd.Series(True, index=scores.index)
    return scores.where(~target, np.minimum(scores + lift, MAX_SCORE))


# PROCESS 3 – ONE SIMULATION

def _coefficient(link: pd.Series, bound: str, ratio: bool) -> float:
    """The link's coefficient at the estimate or one edge of its margin of error.
    'cautious' = the edge closest to no effect."""
    values = [link["effect"], link["ci_low"], link["ci_high"]]
    if ratio:
        values = [np.log(v) for v in values]
    estimate, edges = values[0], sorted(values[1:], key=abs)
    return {"estimate": estimate, "cautious": edges[0], "optimistic": edges[1]}[bound]


def _simulate_outcome(sample, scope, outcome, analysis, lifts, who, links) -> dict[str, Any]:
    """Yearly change in one outcome for the people in scope, per bound."""
    spec = SPECS[outcome]
    extra = _extras(analysis, outcome)
    fit = fit_outcome_model(sample, outcome, extra)
    X0 = design_matrix(sample, outcome, extra).loc[scope.index]
    ratio = spec.method in ("logit", "negbin")
    standardized = set(spec.standardize)

    moved, unverified = [], []
    X1 = X0.copy()
    for item, lift in lifts.items():
        if item not in links.index:
            unverified.append(item)
            continue
        change = improved_scores(scope[item], lift, who) - scope[item]
        if item in standardized:                        # performance model uses z-scores
            change = change / float(np.nanstd(sample[item]))
        X1[item] = X1[item] + change
        moved.append(item)

    out = {"moved": moved, "unverified": unverified}
    # Today's predictions stay fixed at the fitted model; each bound only changes
    # how strongly the improvement moves them.
    eta0 = X0.to_numpy(float) @ fit.params.to_numpy()
    for bound in BOUNDS:
        step = np.zeros(len(X0))
        for item in moved:
            step += (X1[item] - X0[item]).to_numpy(float) * _coefficient(links.loc[item], bound, ratio)
        eta1 = eta0 + step
        if spec.method == "logit":
            p0, p1 = 1 / (1 + np.exp(-eta0)), 1 / (1 + np.exp(-eta1))
            out[bound] = {"baseline": float(p0.sum()), "change": float((p0 - p1).sum())}
        elif spec.method == "negbin":
            m0, m1 = np.exp(eta0), np.exp(eta1)
            out[bound] = {"baseline": float(m0.sum() * 2), "change": float((m0 - m1).sum() * 2)}
        else:                                              # rating points, per person on average
            sd = float(np.nanstd(sample[spec.outcome_column]))
            out[bound] = {"baseline": float(scope[spec.outcome_column].mean()),
                          "change": float((eta1 - eta0).mean() * sd)}
    return out


def simulate(df: pd.DataFrame, analysis: dict, lifts: dict[str, float],
             assumptions: Assumptions | None = None, who: str = "everyone",
             department: str | None = None, ramp_months: int = 6,
             quarters: int = 4) -> dict[str, Any]:
    """Run one scenario. lifts = {'Q04': 0.3, ...} in points on the 1-5 scale."""
    sample = audit_sample(df)
    a = assumptions or Assumptions(headcount=len(sample))
    scale = a.scale(len(sample))
    scope = sample if not department else sample[sample["department"] == department]
    if scope.empty:
        raise ValueError(f"No employees in {department}.")
    lifts = {i: float(v) for i, v in lifts.items() if v and i in sample.columns}

    links = verified_links(analysis)
    outcomes = [o for o in SPECS if SPECS[o].outcome_column in sample.columns
                and o in set(analysis["results"]["outcome"])]
    per_outcome = {o: _simulate_outcome(sample, scope, o, analysis, lifts, who,
                                        links[links["outcome"] == o].set_index("item"))
                   for o in outcomes}

    totals = {}
    for bound in BOUNDS:
        leavers = per_outcome.get("turnover", {}).get(bound, {}).get("change", 0.0) * scale
        days = per_outcome.get("absence", {}).get(bound, {}).get("change", 0.0) * scale
        points = per_outcome.get("performance", {}).get(bound, {}).get("change", 0.0)
        value = (leavers * a.cost_per_leaver + days * a.day_cost
                 + points * len(scope) * scale * a.value_per_rating_point)
        totals[bound] = {"leavers": leavers, "absence_days": days, "rating_points": points,
                         "value": value}
    low, high = sorted([totals["cautious"]["value"], totals["optimistic"]["value"]])

    people = len(scope) * scale
    baseline_leavers = per_outcome.get("turnover", {}).get("estimate", {}).get("baseline", 0.0) * scale
    actual_lift = {i: float((improved_scores(scope[i], v, who) - scope[i]).mean())
                   for i, v in lifts.items()}
    result = {
        "inputs": {"lifts": lifts, "who": who, "department": department or "Whole company",
                   "ramp_months": ramp_months, "quarters": quarters},
        "people": people,
        "actual_lift": actual_lift,
        "totals": totals,
        "value_range": (low, high),
        "baseline_leavers": baseline_leavers,
        "unverified": {o: r["unverified"] for o, r in per_outcome.items() if r["unverified"]},
        "moved": {o: r["moved"] for o, r in per_outcome.items() if r["moved"]},
        "path": quarterly_path(totals, (low, high), ramp_months, quarters),
    }
    result["first_year_value"] = float(result["path"]["value"].head(4).sum())
    return result


# PROCESS 4 – OVER TIME

def ramp_share(month: float, ramp_months: int) -> float:
    """Share of the full effect reached by a given month (linear build-up)."""
    return 1.0 if ramp_months <= 0 else min(month / ramp_months, 1.0)


def quarterly_path(totals, value_range, ramp_months: int, quarters: int) -> pd.DataFrame:
    """Per quarter: leavers avoided, value, and running totals with the range."""
    rows, cum, cum_low, cum_high, cum_leavers = [], 0.0, 0.0, 0.0, 0.0
    for q in range(1, quarters + 1):
        share = ramp_share(3 * q - 1.5, ramp_months)        # middle of the quarter
        value = totals["estimate"]["value"] / 4 * share
        leavers = totals["estimate"]["leavers"] / 4 * share
        cum += value
        cum_low += value_range[0] / 4 * share
        cum_high += value_range[1] / 4 * share
        cum_leavers += leavers
        rows.append({"quarter": f"Q{q}", "quarter_number": q, "effect_reached": share,
                     "leavers_avoided": leavers, "value": value,
                     "cumulative_value": cum, "cumulative_low": cum_low,
                     "cumulative_high": cum_high, "cumulative_leavers": cum_leavers})
    return pd.DataFrame(rows)


# PROCESS 5 – EACH QUESTION ON ITS OWN

def contributions(df, analysis, lifts, assumptions=None, who="everyone",
                  department=None) -> pd.DataFrame:
    """Value of each question's lift on its own (they overlap, so do not add them)."""
    rows = []
    for item, lift in lifts.items():
        r = simulate(df, analysis, {item: lift}, assumptions, who, department, 0, 4)
        t = r["totals"]["estimate"]
        rows.append({"item": item, "question": ITEM_LABELS.get(item, item), "lift": lift,
                     "leavers_avoided": t["leavers"], "absence_days_avoided": t["absence_days"],
                     "rating_points": t["rating_points"], "yearly_value": t["value"],
                     "verified_for": ", ".join(o for o, items in r["moved"].items() if item in items)
                                     or "no verified link"})
    return pd.DataFrame(rows).sort_values("yearly_value", ascending=False).reset_index(drop=True)


# PROCESS 6 – PLAIN LANGUAGE AND CHART

def explain_scenario(result: dict[str, Any], a: Assumptions) -> dict[str, Any]:
    c = a.currency
    t = result["totals"]["estimate"]
    lifts = result["inputs"]["lifts"]
    named = [f"{ITEM_LABELS.get(i, i)} (+{v:.1f})" for i, v in lifts.items()]
    names = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
    who = ("everyone's answers" if result["inputs"]["who"] == "everyone"
           else "answers of people who score 1–3")
    where = result["inputs"]["department"]
    low, high = result["value_range"]

    if not lifts:
        return {"headline": "Pick at least one question and how much it improves.", "points": []}
    if t["value"] <= 0 and t["rating_points"] <= 0:
        return {"headline": f"Raising {names} has no verified link to turnover, absence or "
                            "performance in your data, so the models expect no change.",
                "points": []}

    gains = []
    if t["leavers"] > 0:
        gains.append(f"about **{t['leavers']:,.0f} fewer leavers** a year "
                     f"(of {result['baseline_leavers']:,.0f} expected)")
    if t["absence_days"] > 0:
        gains.append(f"**{t['absence_days']:,.0f} fewer absence days**")
    if t["rating_points"] > 0:
        gains.append(f"performance ratings **{t['rating_points']:+.2f} points** on average")
    points = [f"Worth about **{money(t['value'], c)} a year** once the change is complete, "
              f"likely between {money(low, c)} and {money(high, c)}."]
    ramp = result["inputs"]["ramp_months"]
    if ramp > 0:
        points.append(f"Change takes time: with a {ramp}-month build-up, the first year is worth "
                      f"about {money(result['first_year_value'], c)}.")
    for item in lifts:
        counts_for = [o for o, items in result["moved"].items() if item in items]
        if len(counts_for) < len(result["unverified"]) + len(result["moved"]) or not counts_for:
            points.append(f"{ITEM_LABELS.get(item, item)} "
                          + (f"counts only for {' and '.join(counts_for)}: it has no verified "
                             "link to the other outcomes." if counts_for else
                             "has no verified link to any outcome, so it counts as zero."))
    capped = [i for i, v in lifts.items() if result["actual_lift"][i] < v - 0.05]
    if capped and result["inputs"]["who"] == "everyone":
        points.append("Scores cannot go above 5, so the average rise is smaller than chosen for "
                      + ", ".join(f"{ITEM_LABELS[i]} (+{result['actual_lift'][i]:.2f})" for i in capped) + ".")
    where = "across the whole company" if where == "Whole company" else f"in {where}"
    return {"headline": f"If {who} on {names} rise {where}, the models expect "
                        + ", ".join(gains) + ".",
            "points": points}


def path_chart(path: pd.DataFrame, currency: str = "$") -> alt.LayerChart:
    """Running total of value by quarter, with the likely range as a band."""
    data = path.assign(label=path["cumulative_value"].map(lambda v: money(v, currency)))
    base = alt.Chart(data).encode(
        x=alt.X("quarter:N", sort=list(data["quarter"]), title="Quarter from today",
                axis=alt.Axis(labelAngle=0)))
    band = base.mark_area(opacity=0.18, color=HIGH_COLOR).encode(
        y=alt.Y("cumulative_low:Q", title=f"Value so far ({currency})", axis=alt.Axis(format="~s")),
        y2="cumulative_high:Q")
    line = base.mark_line(point=True, color=HIGH_COLOR, strokeWidth=3).encode(
        y="cumulative_value:Q",
        tooltip=[alt.Tooltip("quarter:N", title="Quarter"),
                 alt.Tooltip("label:N", title="Value so far"),
                 alt.Tooltip("cumulative_leavers:Q", title="Leavers avoided so far", format=",.0f"),
                 alt.Tooltip("effect_reached:Q", title="Effect reached", format=".0%")])
    text = line.mark_text(dy=-12, fontSize=11, color="#333").encode(text="label:N")
    return (band + line + text).properties(height=300, width="container")
