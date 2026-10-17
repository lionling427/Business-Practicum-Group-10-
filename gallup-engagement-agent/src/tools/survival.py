"""Monthly leaving risk from person-month data (discrete-time survival model).

Input: one row per employee per month (see ts_readiness and the demo file):
  employee_id, period (e.g. 2026-03), exited (1 in the month they leave) or
  exit_date, and optionally Q01-Q12 (latest pulse), tenure_years, team_id,
  department, manager_id, absence_days.

Model: for every month someone is still employed, the chance they leave that
month (a ridge logistic "hazard" model). Features use only the past:
  engagement now, change in engagement over 3 months, tenure, absence in the
  last 3 months, exits in the team in the last 3 months, manager change in the
  last 6 months.

Testing (rolling backtests): at several cut-off months the model is trained
only on earlier months, then asked who leaves in the next 6 months. It must
beat chance and a "team's recent exit rate" baseline, or nothing is shown
(the same rule as the rest of the app).

Output: expected leavers per month for the next 6 months, and the chance of
leaving within 6 months by department and team (groups of 5+ only). No
individual scores ever leave this module.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .data_ingestion import Q12_ITEMS
from .prediction import (
    LAMBDAS,
    MIN_AUC,
    MIN_LIFT_OVER_BASELINE,
    _fit_ridge_logit,
    _predict,
    auc,
)
from .ts_readiness import exit_flags, period_index, readiness
from .visualization import HIGH_COLOR, MUTED_COLOR


# PROCESS 2 – CONFIGURATION

HORIZON = 6                 # months ahead
MIN_GROUP = 5               # privacy: smallest group shown
MIN_TRAIN_MONTHS = 6
FOLDS = 3                   # rolling backtest cut-offs
FOLD_STEP = 3               # months between cut-offs
FEATURE_LABELS = {
    "engagement": "Engagement now",
    "engagement_change": "Change in engagement (3 months)",
    "tenure": "Tenure",
    "absence_recent": "Absence in the last 3 months",
    "team_exits_recent": "Team exits in the last 3 months",
    "manager_change": "New manager (last 6 months)",
}


# PROCESS 3 – PREPARE PERSON-MONTH ROWS

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean person-month rows with past-only features and a month number t."""
    data = df.copy()
    tcol = next(c for c in ["period", "month", "date", "snapshot_date"] if c in data.columns)
    data["_period"] = period_index(data[tcol])
    flags = exit_flags(data, data["_period"])
    if flags is None:
        raise ValueError("Needs an 'exited' flag per month or an 'exit_date'.")
    data["exited"] = flags.to_numpy()
    months = sorted(data["_period"].unique())
    data["t"] = data["_period"].map({p: i for i, p in enumerate(months)}).astype(int)
    data = data.sort_values(["employee_id", "t"])

    # Nothing after the exit month.
    first_exit = data[data["exited"] == 1].groupby("employee_id")["t"].min()
    after = data["employee_id"].map(first_exit)
    data = data[after.isna() | (data["t"] <= after)].copy()

    g = data.groupby("employee_id")
    items = [i for i in Q12_ITEMS if i in data.columns]
    if items:
        data[items] = g[items].ffill()                      # latest pulse answers carry forward
        data["engagement"] = data[items].mean(axis=1)
        data["engagement"] = data["engagement"].fillna(data["engagement"].mean())
        data["engagement_change"] = (data["engagement"]
                                     - data.groupby("employee_id")["engagement"].shift(3)).fillna(0.0)
    if "tenure_years" in data.columns:
        data["tenure"] = pd.to_numeric(data["tenure_years"], errors="coerce")
    else:
        data["tenure"] = g.cumcount() / 12.0
    data["tenure"] = data["tenure"].fillna(data["tenure"].median())
    if "absence_days" in data.columns:
        data["absence_recent"] = (g["absence_days"]
                                  .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
                                  .fillna(0.0))
    if "team_id" in data.columns:
        team_month = data.groupby(["team_id", "t"])["exited"].sum().rename("team_exits")
        team_month = team_month.reset_index().sort_values("t")
        team_month["team_exits_recent"] = (team_month.groupby("team_id")["team_exits"]
                                           .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
                                           .fillna(0.0))
        data = data.merge(team_month[["team_id", "t", "team_exits_recent"]],
                          on=["team_id", "t"], how="left")
    if "manager_id" in data.columns:
        changed = (data.groupby("employee_id")["manager_id"].shift(1) != data["manager_id"]) \
            & data.groupby("employee_id")["manager_id"].shift(1).notna()
        data["manager_change"] = (changed.astype(float).groupby(data["employee_id"])
                                  .transform(lambda s: s.rolling(6, min_periods=1).max()))
    return data.reset_index(drop=True)


def features(data: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_LABELS if f in data.columns]


def _standardize(train: pd.DataFrame, cols: list[str]):
    mean, sd = train[cols].mean(), train[cols].std().replace(0, 1.0)
    return lambda frame: ((frame[cols] - mean) / sd).fillna(0.0).to_numpy(float)


def _fit(train: pd.DataFrame, cols: list[str], lam: float = LAMBDAS[1]):
    scale = _standardize(train, cols)
    beta = _fit_ridge_logit(scale(train), train["exited"].to_numpy(float), lam)
    return beta, scale


def _leave_within(beta, scale, at_risk: pd.DataFrame, horizon: int) -> np.ndarray:
    """Chance of leaving within `horizon` months, features frozen except tenure."""
    survive = np.ones(len(at_risk))
    frame = at_risk.copy()
    monthly = []
    for k in range(horizon):
        hazard = _predict(beta, scale(frame))
        monthly.append(survive * hazard)                  # expected leaving in month k
        survive = survive * (1 - hazard)
        frame["tenure"] = frame["tenure"] + 1 / 12
    return 1 - survive, np.array(monthly)


# PROCESS 4 – ROLLING BACKTESTS

def backtest(data: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Train on months before each cut-off, test who leaves in the next `horizon` months."""
    cols = features(data)
    last = int(data["t"].max())
    cutoffs = [last - horizon + 1 - FOLD_STEP * k for k in range(FOLDS)]
    cutoffs = sorted(c for c in cutoffs if c >= MIN_TRAIN_MONTHS)
    rows = []
    for c in cutoffs:
        train = data[data["t"] < c]
        at_risk = data[data["t"] == c]
        if train["exited"].sum() < 10 or at_risk.empty:
            continue
        beta, scale = _fit(train, cols)
        predicted, _ = _leave_within(beta, scale, at_risk, horizon)
        window = data[(data["t"] >= c) & (data["t"] < c + horizon)]
        left = window.groupby("employee_id")["exited"].max()
        actual = at_risk["employee_id"].map(left).fillna(0).to_numpy(int)
        if "team_id" in data.columns:
            recent = data[(data["t"] < c) & (data["t"] >= c - 12)]
            team_rate = recent.groupby("team_id")["exited"].mean()
            baseline = at_risk["team_id"].map(team_rate).fillna(recent["exited"].mean()).to_numpy()
        else:
            baseline = np.full(len(at_risk), 0.5)
        rows.append({"cutoff_month": c, "people": len(at_risk), "left": int(actual.sum()),
                     "expected": float(predicted.sum()), "model_auc": auc(actual, predicted),
                     "baseline_auc": auc(actual, baseline)})
    return pd.DataFrame(rows)


# PROCESS 5 – THE WHOLE RUN

def monthly_risk(df: pd.DataFrame, horizon: int = HORIZON) -> dict[str, Any]:
    ready = readiness(df)
    if not ready["ready_for_survival"]:
        return {"status": "not_ready", "readiness": ready,
                "status_text": ready["verdict"]}
    data = prepare(df)
    tests = backtest(data, horizon)
    if tests.empty:
        return {"status": "untested", "readiness": ready, "status_text":
                "Not enough history to test the model on months it never saw."}
    model_auc = float(tests["model_auc"].mean())
    baseline_auc = float(tests["baseline_auc"].mean())
    useful = model_auc >= MIN_AUC and model_auc >= max(0.5, baseline_auc) + MIN_LIFT_OVER_BASELINE
    result = {
        "status": "useful" if useful else "weak", "readiness": ready, "tests": tests,
        "auc": model_auc, "baseline_auc": baseline_auc, "horizon": horizon,
        "months": int(data["t"].max()) + 1,
        "status_text": ("The model beat chance and the team's recent exit rate in every-month "
                        "backtests it never saw." if useful else
                        "The model did not clearly beat chance or the team's recent exit rate, "
                        "so its numbers should not guide decisions."),
    }
    if not useful:
        return result

    cols = features(data)
    beta, scale = _fit(data, cols)
    last = int(data["t"].max())
    at_risk = data[(data["t"] == last) & (data["exited"] == 0)].copy()
    chance, monthly = _leave_within(beta, scale, at_risk, horizon)
    at_risk["chance"] = chance
    result.update({
        "people_at_risk": len(at_risk),
        "expected_leavers": float(chance.sum()),
        "by_month": pd.DataFrame({"month": [f"+{k + 1}" for k in range(horizon)],
                                  "expected_leavers": monthly.sum(axis=1)}),
        "by_department": _groups(at_risk, "department"),
        "by_team": _groups(at_risk, "team_id"),
        "drivers": pd.DataFrame({"feature": [FEATURE_LABELS[c] for c in cols],
                                 "odds_change_per_sd": np.exp(beta[1:]) - 1})
        .sort_values("odds_change_per_sd", key=abs, ascending=False).reset_index(drop=True),
        "calibration": float(tests["expected"].sum() / max(tests["left"].sum(), 1)),
    })
    return result


def _groups(at_risk: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in at_risk.columns:
        return pd.DataFrame()
    table = (at_risk.groupby(column)
             .agg(people=("chance", "size"), average_chance=("chance", "mean"),
                  expected_leavers=("chance", "sum"))
             .reset_index())
    return (table[table["people"] >= MIN_GROUP]
            .sort_values("average_chance", ascending=False).reset_index(drop=True))


# PROCESS 6 – PLAIN LANGUAGE AND CHART

def explain_monthly_risk(result: dict[str, Any]) -> dict[str, Any]:
    if result["status"] != "useful":
        return {"headline": result["status_text"], "points": []}
    h = result["horizon"]
    points = [f"Tested on {len(result['tests'])} cut-off months it never saw: it picks who "
              f"leaves {result['auc']:.0%} of the time (the team's recent exit rate gets "
              f"{result['baseline_auc']:.0%}).",
              f"In those tests it predicted {result['calibration']:.0%} of the leavers who "
              "actually left" + (" (close)." if 0.8 <= result["calibration"] <= 1.2 else ".")]
    if not result["by_department"].empty:
        top = result["by_department"].iloc[0]
        points.append(f"Highest risk: **{top.iloc[0]}**, where on average {top['average_chance']:.0%} "
                      f"of people may leave within {h} months.")
    if not result["drivers"].empty:
        d = result["drivers"].iloc[0]
        points.append(f"Strongest signal: {d['feature'].lower()} "
                      f"({'raises' if d['odds_change_per_sd'] > 0 else 'lowers'} the monthly "
                      f"odds of leaving by {abs(d['odds_change_per_sd']):.0%} per typical step).")
    return {"headline": f"About **{result['expected_leavers']:,.0f} of {result['people_at_risk']:,} "
                        f"people** are likely to leave in the next {h} months if nothing changes.",
            "points": points}


def monthly_chart(result: dict[str, Any]) -> alt.LayerChart:
    data = result["by_month"].assign(cumulative=lambda d: d["expected_leavers"].cumsum())
    base = alt.Chart(data).encode(x=alt.X("month:N", sort=list(data["month"]),
                                          title="Months from now", axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(color=HIGH_COLOR, opacity=0.85, cornerRadiusEnd=3).encode(
        y=alt.Y("expected_leavers:Q", title="Expected leavers"),
        tooltip=[alt.Tooltip("month:N", title="Month"),
                 alt.Tooltip("expected_leavers:Q", title="Expected leavers", format=",.0f"),
                 alt.Tooltip("cumulative:Q", title="So far", format=",.0f")])
    line = base.mark_line(color=MUTED_COLOR, point=True, strokeDash=[4, 3]).encode(
        y=alt.Y("cumulative:Q", title="Expected leavers"))
    return (bars + line).properties(height=260, width="container")


# PROCESS 7 – DEMO PERSON-MONTH DATA

def demo_person_month(employees: int = 1500, months: int = 24, seed: int = 7) -> pd.DataFrame:
    """Synthetic person-month file with a real engagement → leaving link, for trying the
    feature and as a template. Leavers are replaced by new hires."""
    rng = np.random.default_rng(seed)
    departments = ["Sales", "Operations", "Engineering", "Finance", "HR", "Customer Service"]
    teams = [f"T{n:03d}" for n in range(60)]
    team_effect = dict(zip(teams, rng.normal(0, 0.35, len(teams))))
    team_dept = {t: departments[i % len(departments)] for i, t in enumerate(teams)}
    managers = {t: f"M{n:03d}" for n, t in enumerate(teams)}

    people = [{"id": f"E{n:05d}", "team": rng.choice(teams), "tenure": rng.gamma(2, 2),
               "eng": rng.normal(3.6, 0.6)} for n in range(employees)]
    next_id, rows = employees, []
    start = pd.Period("2025-01", "M")
    for m in range(months):
        if m % 6 == 3:                                      # a reorganisation: a few manager changes
            for t in rng.choice(teams, 5, replace=False):
                managers[t] = f"M{rng.integers(100, 999)}"
        survivors = []
        for p in people:
            p["eng"] = float(np.clip(p["eng"] + rng.normal(0, 0.08) + 0.15 * (team_effect[p["team"]] < -0.3) * -0.1, 1, 5))
            if rng.random() < 0.01:
                p["team"] = rng.choice(teams)
            hazard = 1 / (1 + np.exp(-(-4.6 - 1.1 * (p["eng"] - 3.6) + team_effect[p["team"]]
                                       - 0.08 * p["tenure"])))
            left = rng.random() < hazard
            row = {"employee_id": p["id"], "period": str(start + m), "team_id": p["team"],
                   "department": team_dept[p["team"]], "manager_id": managers[p["team"]],
                   "tenure_years": round(p["tenure"], 2),
                   "absence_days": int(rng.poisson(0.4 + 0.3 * (p["eng"] < 3))),
                   "exited": int(left), "responded": 1}
            if m % 3 == 0:                                   # quarterly pulse survey
                for i, item in enumerate(Q12_ITEMS):
                    row[item] = int(np.clip(round(p["eng"] + rng.normal(0, 0.7)), 1, 5))
            rows.append(row)
            p["tenure"] += 1 / 12
            if not left:
                survivors.append(p)
        hires = employees - len(survivors)
        for _ in range(hires):
            survivors.append({"id": f"E{next_id:05d}", "team": rng.choice(teams), "tenure": 0.0,
                              "eng": rng.normal(3.7, 0.5)})
            next_id += 1
        people = survivors
    return pd.DataFrame(rows)
