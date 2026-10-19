"""Predictions: team risk, employee risk (aggregate only), and engagement forecast.

Every prediction is tested before it is shown (v4 adds the five Prediction
Review fixes, see prediction_checks.py):
- Risk models are trained on earlier survey waves and scored on the latest wave
  (out-of-time validation). They are compared with chance and with a
  "last year's rate" baseline. If a model does not beat them, it is marked
  'weak' and the app shows no ranking.
- The engagement forecast needs 4+ waves and is backtested against a
  "same as last time" baseline.
- v4 team risk: the target is the team's SMOOTHED rate from ALL employees
  (respondents and non-respondents) 3+ points above the company rate; Q12 must
  beat a within-department permutation test to be listed as a warning sign;
  teams that changed must be predicted better than "last year's rate";
  percentages are shown only when calibrated; blocked data is never predicted.

Privacy: the employee model never returns individual scores. Only totals for
groups of at least MIN_GROUP people leave this module.

Main functions
- team_risk(df, outcome):            ranked team risk + model check
- employee_risk_by_group(df, group):  aggregate employee risk + fairness check
- forecast_engagement(df, group, value): next-survey forecast (4+ waves)
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from .data_ingestion import Q12_ITEMS
from .longitudinal import INDEX, derive_wave, prepare
from .prediction_checks import (
    CHANGE_INDEX_DELTA,
    N_PERMUTATIONS,
    SMOOTHING_K,
    TURNOVER_MARGIN,
    RELATIVE_MARGIN,
    above_normal,
    attribution_text,
    calibration_check,
    calibration_pending,
    change_gate,
    permutation_test,
    permutations_for,
    prediction_health_gate,
    smooth_rate,
    survey_response_warning,
)
from .visualization import HIGH_COLOR, ITEM_LABELS, MUTED_COLOR, OVERALL_COLOR


# PROCESS 2 – CONFIGURATION

TARGETS = {
    "turnover": ("left_voluntarily_12m", "above-normal voluntary turnover"),
    "absence": ("unplanned_absence_days_6m", "above-normal unplanned absence"),
}

MIN_TEAM_SIZE = 5
MIN_GROUP = 5                 # privacy: smallest group shown for employee risk
MIN_TRAIN_ROWS = 100
LAMBDAS = [0.1, 1.0, 10.0]    # ridge strengths tried by cross-validation

# A model is "useful" only if it clearly beats chance and the persistence baseline.
MIN_AUC = 0.60
MIN_LIFT_OVER_BASELINE = 0.02

HIGH_SHARE, ELEVATED_SHARE = 0.20, 0.40   # top 20% = High, next 20% = Elevated

MIN_FORECAST_WAVES = 4


# PROCESS 3 – MODEL HELPERS (numpy only)

def _fit_ridge_logit(X: np.ndarray, y: np.ndarray, lam: float,
                     max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """L2-penalized logistic regression on standardized X (intercept unpenalized)."""
    Xi = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xi.shape[1])
    penalty = np.full(Xi.shape[1], lam)
    penalty[0] = 0.0
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(Xi @ beta, -30, 30)))
        grad = Xi.T @ (y - p) - penalty * beta
        H = (Xi * (p * (1 - p))[:, None]).T @ Xi + np.diag(penalty)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(beta[0] + X @ beta[1:], -30, 30)))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Chance that a random positive outranks a random negative (ties = half)."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _choose_lambda(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Pick the ridge strength with the best cross-validated AUC."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(y))
    best, best_auc = LAMBDAS[1], -1.0
    for lam in LAMBDAS:
        scores = np.zeros(len(y))
        for k in range(folds):
            train, test = fold != k, fold == k
            if test.sum() == 0 or len(np.unique(y[train])) < 2:
                continue
            scores[test] = _predict(_fit_ridge_logit(X[train], y[train], lam), X[test])
        value = auc(y, scores)
        if value > best_auc:
            best, best_auc = lam, value
    return best


class _Model:
    """Standardize -> ridge logistic, with feature names kept for explanations."""

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "_Model":
        X = frame[self.features].to_numpy(float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = (X - self.mean) / self.std
        self.lam = _choose_lambda(Z, y)
        self.beta = _fit_ridge_logit(Z, y, self.lam)
        return self

    def z(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame[self.features].to_numpy(float) - self.mean) / self.std

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return _predict(self.beta, self.z(frame))

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """How much each feature pushes each row's risk up (+) or down (-)."""
        return pd.DataFrame(self.z(frame) * self.beta[1:], columns=self.features,
                            index=frame.index)


def _verdict(model_auc: float, baseline_auc: float | None) -> tuple[str, str]:
    reference = max(0.5, baseline_auc if baseline_auc is not None and not np.isnan(baseline_auc) else 0.5)
    if np.isnan(model_auc):
        return "untested", "The latest survey has no outcomes to test against yet."
    if model_auc >= MIN_AUC and model_auc >= reference + MIN_LIFT_OVER_BASELINE:
        return "useful", "The model beat chance and the 'last year's rate' baseline on a survey it never saw."
    return "weak", ("The model did not clearly beat chance or the 'last year's rate' "
                    "baseline, so its rankings should not guide decisions.")


def _bands(prob: pd.Series) -> pd.Series:
    rank = prob.rank(ascending=False, pct=True)
    return pd.Series(np.select([rank <= HIGH_SHARE, rank <= ELEVATED_SHARE],
                               ["High", "Elevated"], "Typical"), index=prob.index)


def _cross_validated_auc(model_features: list[str], frame: pd.DataFrame, y: np.ndarray,
                         folds: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(y))
    scores = np.full(len(y), np.nan)
    for k in range(folds):
        train, test = fold != k, fold == k
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        model = _Model(model_features).fit(frame[train], y[train])
        scores[test] = model.predict(frame[test])
    ok = ~np.isnan(scores)
    return auc(y[ok], scores[ok])


def _validate_and_predict(table: pd.DataFrame, features: list[str], target: str,
                          baseline: str | None = None) -> dict[str, Any]:
    """Test on the latest wave that has known outcomes (training only on waves
    before it), then refit on every wave with known outcomes and score the
    latest survey. With a single labeled wave, falls back to cross-validation.

    In live use the newest survey's outcomes are not known yet; this is the
    case it is built for.
    """
    labeled = table[table[target].notna()]
    labeled_waves = sorted(labeled["wave"].unique())
    if not labeled_waves:
        raise ValueError("No survey wave has known outcomes to learn from yet.")
    latest = table["wave"].max()
    test_wave = labeled_waves[-1]
    train = labeled[labeled["wave"] < test_wave]
    test = labeled[labeled["wave"] == test_wave]

    baseline_auc = None
    if len(train) >= MIN_TRAIN_ROWS // 5 and train[target].nunique() >= 2:
        method = "out_of_time"
        check = _Model(features).fit(train, train[target].to_numpy(int))
        model_auc = auc(test[target], check.predict(test))
        if baseline is not None:
            has_prev = test[baseline].notna()
            if has_prev.sum() >= 20:
                baseline_auc = auc(test.loc[has_prev, target], test.loc[has_prev, baseline])
        validation = test.assign(_score=check.predict(test))
    else:
        method = "cross_validated"
        if len(test) < MIN_TRAIN_ROWS // 5 or test[target].nunique() < 2:
            raise ValueError("Not enough data with known outcomes to build a risk model.")
        model_auc = _cross_validated_auc(features, test, test[target].to_numpy(int))
        validation = None

    model = _Model(features).fit(labeled, labeled[target].to_numpy(int))
    scored = table[table["wave"] == latest].copy()
    scored["risk"] = model.predict(scored)
    status, status_text = _verdict(model_auc, baseline_auc)
    if method == "cross_validated" and status == "useful":
        status_text = ("Checked by cross-validation within one survey only; it has not "
                       "yet been tested on a later survey, so treat rankings as a first look.")
    return {"model": model, "scored": scored, "validation": validation, "method": method,
            "auc": model_auc, "baseline_auc": baseline_auc, "status": status,
            "status_text": status_text, "test_wave": test_wave, "predict_wave": latest,
            "train_waves": sorted(train["wave"].unique().tolist()) if method == "out_of_time" else [],
            "n_train": len(train) if method == "out_of_time" else len(test),
            "n_test": len(test), "outcomes_known": bool(scored[target].notna().any())}


# PROCESS 4 – TEAM RISK

def _team_table(data: pd.DataFrame) -> pd.DataFrame:
    """Respondents' Q12 profile per team and wave (teams of MIN_TEAM_SIZE+ respondents)."""
    items = [i for i in Q12_ITEMS if i in data.columns]
    agg = {"size": (INDEX, "size"), INDEX: (INDEX, "mean"), **{i: (i, "mean") for i in items}}
    if "department" in data.columns:
        agg["department"] = ("department", lambda s: s.mode().iat[0])
    if "manager_id" in data.columns:
        agg["manager_id"] = ("manager_id", lambda s: s.mode().iat[0] if len(s.mode()) else None)
    teams = data.groupby(["team_id", "wave"]).agg(**agg).reset_index()
    teams = teams[teams["size"] >= MIN_TEAM_SIZE].copy()
    return teams


def _team_outcomes(everyone: pd.DataFrame, column: str, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    """Outcome per team and wave from ALL employees with a known outcome (fix 1/5: the
    respondents-only rate misses the leavers who did not answer)."""
    known = everyone[everyone[column].notna()]
    company = known.groupby("wave")[column].mean()
    g = everyone.groupby(["team_id", "wave"])
    table = pd.DataFrame({
        "headcount": g.size(),
        "outcome_total": g[column].sum(min_count=1),
        "outcome_count": g[column].count(),
    }).reset_index()
    rate = table["wave"].map(company)
    table[outcome] = (table["outcome_total"] / table["outcome_count"]).where(table["outcome_count"] > 0)
    table["smoothed_rate"] = smooth_rate(table["outcome_total"].fillna(0), table["outcome_count"], rate)
    table["target"] = above_normal(table["smoothed_rate"], rate, outcome)
    return table, company


def _out_of_time_auc(table: pd.DataFrame, features: list[str], target: str, test_wave,
                     lam: float | None = None) -> tuple[float, np.ndarray | None]:
    labeled = table[table[target].notna()]
    train = labeled[labeled["wave"] < test_wave]
    test = labeled[labeled["wave"] == test_wave]
    if len(train) < MIN_TRAIN_ROWS // 5 or train[target].nunique() < 2 or test[target].nunique() < 2:
        return float("nan"), None
    model = _Model(features)
    X = train[features].to_numpy(float)
    model.mean, model.std = X.mean(axis=0), X.std(axis=0)
    model.std[model.std == 0] = 1.0
    model.lam = lam if lam is not None else _choose_lambda((X - model.mean) / model.std,
                                                           train[target].to_numpy(int))
    model.beta = _fit_ridge_logit((X - model.mean) / model.std, train[target].to_numpy(int), model.lam)
    score = model.predict(test)
    return auc(test[target], score), score


_TEAM_RISK_CACHE: dict = {}


def team_risk(df: pd.DataFrame, outcome: str = "turnover", health_report: dict | None = None,
              n_permutations: int = N_PERMUTATIONS, seed: int = 0) -> dict[str, Any]:
    """Each team's predicted chance of above-normal turnover/absence after the
    latest survey, from its Q12 profile, recent change, size and department.

    Raises PredictionBlocked (a ValueError) when the health report blocks the
    outcome or everything (fix 4). Results are cached per data content, because
    the permutation test refits the model 200 times and pages rerun often."""
    prediction_health_gate(health_report, outcome)
    key = (outcome, n_permutations, seed, len(df), tuple(df.columns),
           int(pd.util.hash_pandas_object(df, index=False).sum()))
    if key not in _TEAM_RISK_CACHE:
        if len(_TEAM_RISK_CACHE) >= 8:
            _TEAM_RISK_CACHE.pop(next(iter(_TEAM_RISK_CACHE)))
        _TEAM_RISK_CACHE[key] = _team_risk(df, outcome, health_report, n_permutations, seed)
    result = _TEAM_RISK_CACHE[key]
    return {**result, "teams": result["teams"].copy()}


def _team_risk(df, outcome, health_report, n_permutations, seed):
    column, label = TARGETS[outcome]
    everyone = derive_wave(df)
    if "team_id" not in everyone.columns or column not in everyone.columns:
        raise ValueError(f"Team risk needs team_id and {column}.")
    if "wave" not in everyone.columns:
        everyone = everyone.assign(wave=1)
    everyone = everyone.assign(**{column: pd.to_numeric(everyone[column], errors="coerce")})
    data = prepare(everyone)

    teams = _team_table(data)
    outcomes, company_rate = _team_outcomes(everyone, column, outcome)
    teams = teams.merge(outcomes, on=["team_id", "wave"], how="left")
    teams["log_size"] = np.log(teams["headcount"].fillna(teams["size"]))

    items = [i for i in Q12_ITEMS if i in teams.columns]

    # Change since the team's previous survey (fix 2): item and index deltas, new manager.
    teams = teams.sort_values(["team_id", "wave"]).reset_index(drop=True)
    prev = teams.groupby("team_id")
    has_prev = prev["wave"].shift().notna()
    teams["has_previous"] = has_prev.astype(float)
    delta_cols = [f"d_{i}" for i in items] + ["d_index"]
    for i in items:
        teams[f"d_{i}"] = (teams[i] - prev[i].shift()).fillna(0.0)
    teams["d_index"] = (teams[INDEX] - prev[INDEX].shift()).fillna(0.0)
    change_features = delta_cols + ["has_previous"]
    if "manager_id" in teams.columns:
        teams["manager_changed"] = ((teams["manager_id"] != prev["manager_id"].shift())
                                    & has_prev).astype(float)
        change_features.append("manager_changed")
    teams["changed"] = has_prev & ((teams["d_index"].abs() >= CHANGE_INDEX_DELTA)
                                   | (teams.get("manager_changed", 0) == 1))

    features = items + change_features + ["log_size"]
    base_features = ["log_size"]
    if "department" in teams.columns:
        dummies = pd.get_dummies(teams["department"].astype(str), prefix="dept",
                                 drop_first=True, dtype=float)
        teams = pd.concat([teams, dummies], axis=1)
        features += list(dummies.columns)
        base_features += list(dummies.columns)

    # Persistence baseline: the team's own SMOOTHED rate in its previous wave (fix 5).
    teams["previous_rate"] = prev["smoothed_rate"].shift()
    teams["previous_target"] = prev["target"].shift()

    fit = _validate_and_predict(teams, features, "target", baseline="previous_rate")
    scored = fit["scored"]

    # Fix 1: does Q12 add anything beyond department and size?
    q12_columns = items + delta_cols
    base_auc = float("nan")
    survey_signal: dict[str, Any] = {"status": "untested", "adds_signal": False}
    if fit["method"] == "out_of_time":
        base_auc, _ = _out_of_time_auc(teams, base_features, "target", fit["test_wave"])
        lam = fit["model"].lam
        labeled = teams[teams["target"].notna()]
        observed, _ = _out_of_time_auc(labeled, features, "target", fit["test_wave"], lam)
        survey_signal = permutation_test(
            lambda shuffled: _out_of_time_auc(shuffled, features, "target", fit["test_wave"], lam)[0],
            labeled, q12_columns, ["department", "wave"], observed,
            n=permutations_for(len(labeled), n_permutations), seed=seed)
    survey_signal["department_size_auc"] = None if np.isnan(base_auc) else round(base_auc, 4)
    survey_signal["text"] = attribution_text(survey_signal, fit["auc"], base_auc)

    contrib = fit["model"].contributions(scored)[items]
    if survey_signal.get("adds_signal"):
        scored["risk_drivers"] = contrib.apply(
            lambda r: ", ".join(f"{i} {ITEM_LABELS.get(i, i)}"
                                for i in r[r > 0].sort_values(ascending=False).index[:2]), axis=1)
    else:
        scored["risk_drivers"] = ""
    scored["risk_band"] = _bands(scored["risk"])

    # How the High band did, calibration (fix 3) and the change gate (fix 2): validation wave only.
    observed_high, base_rate = float("nan"), float("nan")
    calibration = calibration_pending()
    change = {"status": "pending", "text": "Change check pending: needs a later year with known outcomes."}
    if fit["validation"] is not None:
        v = fit["validation"].copy()
        v["_band"] = _bands(v["_score"])
        high = v["_band"] == "High"
        observed_high = float(v.loc[high, "target"].mean())
        base_rate = float(v["target"].mean())
        calibration = calibration_check(v["_score"], v["target"])
        change = change_gate(v, auc)
    scored["recent_change"] = scored["changed"].astype(bool)
    scored["change_note"] = np.where(
        scored["recent_change"] & (change["status"] != "passed"),
        "Recent change: band unconfirmed", "")

    warning = survey_response_warning(everyone, column, health_report) if outcome == "turnover" else None

    columns = ["team_id"] + (["department"] if "department" in scored.columns else []) + [
        "size", INDEX, "risk", "risk_band", "risk_drivers", "previous_rate",
        "recent_change", "change_note"]
    return {**{k: v for k, v in fit.items() if k not in ("model", "scored", "validation")},
            "outcome": outcome, "label": label,
            "teams": scored.sort_values("risk", ascending=False)[columns].reset_index(drop=True),
            "base_rate": base_rate, "observed_high_rate": observed_high,
            "survey_signal": survey_signal, "change_check": change, "calibration": calibration,
            "show_percentages": bool(calibration["show_percentages"]),
            "survey_response_warning": warning,
            "target_definition": {
                "smoothing_k": SMOOTHING_K,
                "margin": (f"{TURNOVER_MARGIN * 100:.0f} points above the company rate" if outcome == "turnover"
                           else f"{RELATIVE_MARGIN:.0%} above the company average"),
                "company_rate_by_wave": {str(k): round(float(v), 4) for k, v in company_rate.items()},
                "text": (f"A team counts as {label} when its rate, pulled toward the company rate as "
                         f"if {SMOOTHING_K} typical employees were added, is "
                         + (f"{TURNOVER_MARGIN * 100:.0f}+ points above the company rate."
                            if outcome == "turnover" else
                            f"{RELATIVE_MARGIN:.0%}+ above the company average.")
                         + " Rates use every employee, including those who did not answer the survey."),
            }}


# PROCESS 5 – EMPLOYEE RISK (AGGREGATE ONLY)

def _employee_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Internal: per-employee risk for the latest survey. Never returned directly."""
    column = TARGETS["turnover"][0]
    data = prepare(df)
    if column not in data.columns:
        raise ValueError("Employee risk needs the voluntary turnover column.")
    if "wave" not in data.columns:
        data = data.assign(wave=1)
    # Fairness: only engagement answers and tenure; no demographic or pay fields.
    features = [i for i in Q12_ITEMS if i in data.columns]
    if "tenure_years" in data.columns:
        features.append("tenure_years")
    data = data.dropna(subset=features)
    fit = _validate_and_predict(data, features, column)
    scored = fit["scored"]
    scored["high_risk"] = _bands(scored["risk"]) == "High"
    validation = fit["validation"]
    if validation is not None:
        validation = validation.rename(columns={"_score": "risk"})
    info = {k: v for k, v in fit.items() if k not in ("model", "scored", "validation")}
    return scored, validation, {**info, "features": features}


def employee_risk_by_group(df: pd.DataFrame, group: str = "department",
                           health_report: dict | None = None) -> dict[str, Any]:
    """Aggregate employee turnover risk by group, plus a fairness check.

    Returns group totals only (groups under MIN_GROUP people are dropped).
    Risk columns describe the latest survey. The fairness check (accuracy and
    calibration per group) uses the validation wave, where outcomes are known
    and the model had not seen them.
    """
    prediction_health_gate(health_report, "turnover")
    scored, validation, info = _employee_model(df)
    column = TARGETS["turnover"][0]
    if group not in scored.columns:
        raise ValueError(f"{group} is not in the data.")
    rows = []
    for name, g in scored.groupby(group):
        if len(g) < MIN_GROUP:
            continue
        row = {group: name, "employees": len(g),
               "predicted_rate": float(g["risk"].mean()),
               "high_risk_share": float(g["high_risk"].mean()),
               "actual_rate": float("nan"), "auc": float("nan"),
               "check_predicted": float("nan")}
        if validation is not None and group in validation.columns:
            v = validation[validation[group] == name]
            if len(v) >= MIN_GROUP:
                row["actual_rate"] = float(v[column].mean())
                row["check_predicted"] = float(v["risk"].mean())
                if len(v) >= 50:
                    row["auc"] = auc(v[column], v["risk"])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("high_risk_share", ascending=False)
    # Calibration on the validation wave: what the model predicted vs what happened.
    table["calibration_gap"] = table.pop("check_predicted") - table["actual_rate"]
    table["expected_leavers"] = (table["predicted_rate"] * table["employees"]).round()
    # Flag groups where the model is much less accurate or clearly mis-calibrated.
    overall_auc = info["auc"]
    table["fairness_flag"] = (
        (table["auc"] < overall_auc - 0.05) | (table["calibration_gap"].abs() > 0.05))
    calibration = (calibration_check(validation["risk"], validation[column], unit="employees")
                   if validation is not None else calibration_pending())
    warning = survey_response_warning(derive_wave(df), column, health_report)
    return {**info, "group": group, "table": table.reset_index(drop=True),
            "calibration": calibration, "show_percentages": bool(calibration["show_percentages"]),
            "survey_response_warning": warning}


# PROCESS 6 – ENGAGEMENT FORECAST

def forecast_engagement(df: pd.DataFrame, group: str | None = None, value: Any = None) -> dict[str, Any]:
    """Forecast the next survey's average Q12 score (needs MIN_FORECAST_WAVES).

    Two candidate methods are backtested on past waves (fit on the first k
    waves, predict wave k+1): a straight-line trend and 'same as last time'.
    The one with the smaller backtest error is used; its interval comes from
    its backtest errors. Assumes surveys are evenly spaced.
    """
    data = prepare(df)
    if group is not None:
        data = data[data[group] == value]
    if "wave" not in data.columns:
        return {"available": False, "n_waves": 0,
                "reason": "No survey waves found, so there is nothing to forecast from."}
    history = (data.groupby("wave")[INDEX].agg(["mean", "size"]).reset_index()
               .query("size >= @MIN_TEAM_SIZE").reset_index(drop=True))
    n = len(history)
    if n < MIN_FORECAST_WAVES:
        return {"available": False, "n_waves": n, "history": history,
                "reason": (f"A forecast needs at least {MIN_FORECAST_WAVES} survey waves; "
                           f"this data has {n}. With so few points, any trend line would be "
                           "a guess.")}

    x, y = history["wave"].to_numpy(float), history["mean"].to_numpy(float)
    errors = {"trend": [], "naive": []}
    for k in range(2, n):
        slope, intercept = np.polyfit(x[:k], y[:k], 1)
        errors["trend"].append(y[k] - (intercept + slope * x[k]))
        errors["naive"].append(y[k] - y[k - 1])
    mae = {m: float(np.mean(np.abs(e))) for m, e in errors.items()}
    method = "trend" if mae["trend"] < mae["naive"] else "naive"

    next_wave = x[-1] + (x[-1] - x[-2])
    if method == "trend":
        slope, intercept = np.polyfit(x, y, 1)
        point = intercept + slope * next_wave
    else:
        point = y[-1]
    spread = float(np.sqrt(np.mean(np.square(errors[method]))))
    q = stats.t.ppf(0.9, df=max(len(errors[method]) - 1, 1))   # 80% interval

    return {"available": True, "n_waves": n, "history": history, "method": method,
            "next_wave": next_wave, "forecast": float(point),
            "low": float(point - q * spread), "high": float(point + q * spread),
            "backtest_mae": mae}


# PROCESS 7 – PLAIN-LANGUAGE EXPLANATIONS

def explain_team_risk(result: dict[str, Any]) -> dict[str, Any]:
    r = result
    wave = _w(r["test_wave"])
    if r["method"] == "out_of_time":
        train = ", ".join(_w(w) for w in r["train_waves"])
        points = [f"Trained on wave {train} teams ({r['n_train']}), then tested on wave {wave} "
                  f"({r['n_test']} teams), a survey it never saw."]
    else:
        points = [f"Only wave {wave} has known outcomes, so the model was checked by "
                  "cross-validation (hiding part of the teams, predicting them, repeating)."]
    if not r["outcomes_known"]:
        points.append(f"Risks are for wave {_w(r['predict_wave'])}, whose outcomes are not known "
                      "yet: this is a genuine look ahead.")
    if not np.isnan(r["auc"]):
        points.append(f"Accuracy: picks the higher-risk of two teams correctly "
                      f"{r['auc'] * 100:.0f}% of the time (chance = 50%).")
    if r["baseline_auc"] is not None:
        points.append(f"Using only each team's previous rate gets {r['baseline_auc'] * 100:.0f}%.")
    if r["status"] == "useful" and not np.isnan(r["observed_high_rate"]):
        points.append(f"Of the teams it rated High, {r['observed_high_rate'] * 100:.0f}% actually had "
                      f"{r['label']}, vs {r['base_rate'] * 100:.0f}% of all teams.")
    for key in ("survey_signal", "change_check", "calibration"):
        if r.get(key, {}).get("text"):
            points.append(r[key]["text"])
    if r.get("survey_response_warning"):
        points.append(r["survey_response_warning"]["text"])
    headline = {"useful": f"The {r['outcome']} risk model passed its test.",
                "weak": f"The {r['outcome']} risk model did not pass its test.",
                "untested": f"The {r['outcome']} risk model could not be tested yet."}[r["status"]]
    return {"headline": headline, "points": points + [r["status_text"]],
            "how_to_read": (
                "Risk is the model's estimated chance that a team has "
                f"{r['label']} (clearly worse than the company rate) after the latest survey. "
                "High = the riskiest 20% of teams, Elevated = the next 20%. 'Warning signs' "
                "lists the questions where the team's answers raise its predicted risk most; "
                "they are signals to listen to, not causes, and the Outcomes page decides what "
                "to improve. Percentages are shown only when they proved accurate on a year the "
                "model never saw. Use the ranking to decide where to listen first.")}


def explain_forecast(result: dict[str, Any]) -> str:
    if not result["available"]:
        return result["reason"]
    method = ("Your scores have followed a steady trend, so the forecast extends it."
              if result["method"] == "trend" else
              "Your scores have moved up and down rather than in a steady direction, so the "
              "best guess is the latest score: on past surveys that guess was more accurate "
              "than extending a trend line.")
    return (f"The next survey (wave {_w(result['next_wave'])}) will most likely land around "
            f"**{result['forecast']:.2f}**, somewhere between {result['low']:.2f} and "
            f"{result['high']:.2f}. {method}")


def _w(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(value)
    except (TypeError, ValueError):
        return str(value)


# PROCESS 8 – CHARTS

def team_risk_chart(result: dict[str, Any], n: int = 15) -> alt.Chart:
    """Top-n teams by predicted risk, colored by band."""
    show = result.get("show_percentages", True)
    ranked = result["teams"].assign(
        risk_pct=lambda d: d["risk"] * 100 if show else (1 - d["risk"].rank(ascending=False, pct=True)) * 100)
    top = ranked.head(n)
    title = "Predicted risk (%)" if show else "Risk rank (100 = highest; percentages not yet calibrated)"
    return alt.Chart(top).mark_bar(cornerRadiusEnd=3, height=14).encode(
        y=alt.Y("team_id:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("risk_pct:Q", title=title, scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("risk_band:N", title=None,
                        scale=alt.Scale(domain=["High", "Elevated", "Typical"],
                                        range=["#B5473F", "#C27C1E", "#B9B1A8"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("team_id:N", title="Team"),
                 alt.Tooltip("risk_pct:Q", title="Risk" if show else "Risk rank", format=".0f"),
                 alt.Tooltip("risk_band:N", title="Band"),
                 alt.Tooltip("risk_drivers:N", title="Warning signs")],
    ).properties(height=alt.Step(24), width=600)


def forecast_chart(result: dict[str, Any]) -> alt.LayerChart:
    """Past surveys as a line, the next survey as a labelled point with its likely range."""
    history = result["history"].assign(label=lambda d: "Wave " + d["wave"].map(_w))
    nxt = result["next_wave"]
    point = pd.DataFrame({"wave": [nxt], "mean": [result["forecast"]], "low": [result["low"]],
                          "high": [result["high"]], "label": [f"Wave {_w(nxt)} (forecast)"],
                          "text": [f"{result['forecast']:.2f}"]})
    bridge = pd.concat([history.tail(1), point])[["wave", "mean"]]
    span = [float(min(history["mean"].min(), result["low"])) - 0.1,
            float(max(history["mean"].max(), result["high"])) + 0.1]
    x = alt.X("wave:Q", title=None, scale=alt.Scale(domain=[history["wave"].min() - 0.3, nxt + 0.3]),
              axis=alt.Axis(values=list(history["wave"]) + [nxt], format="d",
                            labelExpr=f"datum.value == {nxt} ? 'Wave ' + datum.label + ' (next)' "
                                      ": 'Wave ' + datum.label"))
    y = alt.Y("mean:Q", title="Average Q12 score (1–5)", scale=alt.Scale(domain=span))
    tooltip = [alt.Tooltip("label:N", title="Survey"), alt.Tooltip("mean:Q", title="Score", format=".2f")]
    line = alt.Chart(history).mark_line(color=OVERALL_COLOR, strokeWidth=2).encode(x=x, y=y)
    dots = alt.Chart(history).mark_circle(color=OVERALL_COLOR, size=70, opacity=1).encode(
        x=x, y=y, tooltip=tooltip)
    dashed = alt.Chart(bridge).mark_line(color=MUTED_COLOR, strokeDash=[4, 4]).encode(x=x, y=y)
    whisker = alt.Chart(point).mark_rule(color=HIGH_COLOR, strokeWidth=2).encode(
        x=x, y="low:Q", y2="high:Q")
    caps = alt.Chart(point[["wave", "low", "high"]].melt(id_vars=["wave"], value_name="mean")
                     ).mark_tick(color=HIGH_COLOR, thickness=2, size=18,
                                 orient="horizontal").encode(x=x, y=y)
    fc = alt.Chart(point).mark_circle(color=HIGH_COLOR, size=120, opacity=1).encode(
        x=x, y=y, tooltip=tooltip + [alt.Tooltip("low:Q", title="Likely from", format=".2f"),
                                     alt.Tooltip("high:Q", title="Likely to", format=".2f")])
    label = alt.Chart(point).mark_text(align="left", dx=10, fontWeight="bold",
                                       color=HIGH_COLOR).encode(x=x, y=y, text="text:N")
    return (line + dots + dashed + whisker + caps + fc + label).properties(
        height=280, width="container")


# PROCESS 9 – GROUP RISK: STORY AND CHART

def explain_group_risk(result: dict[str, Any]) -> dict[str, Any]:
    t, group = result["table"], result["group"]
    name = group.replace("_", " ")
    top = t.iloc[0]
    above = int((t["high_risk_share"] > HIGH_SHARE).sum())
    headline = (f"{top[group]} carries the most turnover risk: {top['high_risk_share'] * 100:.0f}% "
                f"of its people are among the company's riskiest fifth (20% would be average).")
    points = [f"{above} of {len(t)} {name}s have more than their share of high-risk people."]
    points.append(f"About **{t['expected_leavers'].sum():,.0f}** people are expected to leave "
                  "voluntarily in the next 12 months across these groups.")
    gap = t["calibration_gap"].dropna()
    if len(gap) and (gap > 0).all() and gap.mean() >= 1 / 100:
        points.append(f"When tested, the model predicted slightly more leavers than happened "
                      f"(about {gap.mean() * 100:+.1f} points in every {name}), so treat the "
                      "expected numbers as a ceiling.")
    elif len(gap) and (gap < 0).all() and gap.mean() <= -1 / 100:
        points.append(f"When tested, the model predicted slightly fewer leavers than happened "
                      f"(about {gap.mean() * 100:+.1f} points), so treat the numbers as a floor.")
    flagged = t[t["fairness_flag"]]
    if t["actual_rate"].isna().all():
        points.append("Fairness check pending: it needs a survey the model was not trained on.")
    elif flagged.empty:
        points.append(f"Fairness check passed: the model is about equally accurate for every {name}.")
    else:
        points.append("Fairness check flagged " + ", ".join(f"**{n}**" for n in flagged[group])
                      + ": the model is less accurate there, so treat those numbers with care.")
    return {"headline": headline, "points": points}


def group_risk_chart(result: dict[str, Any]) -> alt.LayerChart:
    t = result["table"].copy()
    group = result["group"]
    t["share"] = t["high_risk_share"] * 100
    t["label"] = t["share"].round().astype(int).astype(str) + "%  ·  ~" + \
        t["expected_leavers"].astype(int).astype(str) + " leavers"
    order = list(t.sort_values("share", ascending=False)[group])
    y = alt.Y(f"{group}:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False))
    color = alt.condition(f"datum.share > {HIGH_SHARE * 100}", alt.value("#B5473F"), alt.value("#B9B1A8"))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=y, x=alt.X("share:Q", title="% of people in the company's riskiest fifth",
                     scale=alt.Scale(domain=[0, max(40.0, float(t["share"].max()) * 1.45)])),
        color=color,
        tooltip=[alt.Tooltip(f"{group}:N", title=group.replace("_", " ").capitalize()),
                 alt.Tooltip("share:Q", title="% at high risk", format=".0f"),
                 alt.Tooltip("employees:Q", title="People", format=","),
                 alt.Tooltip("expected_leavers:Q", title="Expected leavers", format=",.0f")])
    text = alt.Chart(t).mark_text(align="left", dx=5, fontSize=12).encode(
        y=y, x="share:Q", text="label:N")
    rule = alt.Chart(pd.DataFrame({"x": [HIGH_SHARE * 100]})).mark_rule(
        strokeDash=[4, 4], color=OVERALL_COLOR).encode(x="x:Q")
    return (bars + text + rule).properties(height=alt.Step(30), width="container")
