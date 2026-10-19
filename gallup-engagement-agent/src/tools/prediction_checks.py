"""Checks shared by every prediction in the app (v4, Prediction Review fixes 1-5).

1. Q12 attribution: does the survey add anything beyond department and size?
   A within-department permutation test (shuffle the Q12 answers between teams
   of the same department, refit, repeat 200 times). Q12 "adds signal" only if
   the real model beats the 95th percentile of the shuffled models. A fixed AUC
   gap is not used: on null data with nonresponse bias (X7) the gap averaged
   0.034 across 21 seeds, so any fixed cut-off is either too loose or too strict.
   Even when Q12 adds signal, predictions are warning signs, not causes; the
   Outcomes page alone decides what to improve.
2. Change gate: teams whose scores or manager changed are where "last year's
   rate" is most likely wrong, so the model must beat that baseline on them.
3. Calibration: predicted percentages are shown only when, on a year the model
   never saw, teams it called "30%" really had about 30%.
4. Health gate: blocked data never produces predictions.
5. Smoothing: a team's rate is pulled toward the company rate in proportion to
   how small the team is, so a 6-person team with 2 leavers is not "above
   normal" by chance alone.

Everything here is plain numpy/pandas and is unit-tested in tests/test_prediction_checks.py.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


# PROCESS 2 – CONFIGURATION (fixed in advance)

N_PERMUTATIONS = 200


def permutations_for(rows: int, requested: int = N_PERMUTATIONS) -> int:
    """Fewer shuffles for very large files (the shuffled AUCs barely vary there).
    40 shuffles still resolve p < 0.05 (smallest possible p = 1/41)."""
    if rows > 50_000:
        return min(requested, 40)
    if rows > 20_000:
        return min(requested, 100)
    return requested
PERMUTATION_LEVEL = 0.95          # real model must beat the 95th percentile

CHANGE_MIN_TEAMS = 30             # fewer changed teams -> "insufficient"
CHANGE_MIN_LIFT = 0.05            # AUC on changed teams vs "last year's rate"
CHANGE_MAX_FALSE_TURNAROUND = 0.20
CHANGE_INDEX_DELTA = 0.30         # a team "changed" if its Q12 average moved 0.3+ points
N_BOOTSTRAP = 200

CALIBRATION_MIN_PER_BIN = 100
CALIBRATION_MAX_GAP = 0.10        # 10 percentage points ...
CALIBRATION_SE_MULTIPLIER = 2.0   # ... and more than 2 standard errors

SMOOTHING_K = 20                  # "virtual" people at the company rate added to each team
TURNOVER_MARGIN = 0.03            # above normal = 3+ points above the company rate
RELATIVE_MARGIN = 0.10            # for counts (absence days): 10% above the company mean

RESPONSE_GAP_WARN = 0.05


class PredictionBlocked(ValueError):
    """The data health check blocks this prediction."""


# PROCESS 3 – HEALTH GATE (fix 4)

def prediction_health_gate(health_report: dict | None, outcome: str | None) -> None:
    """Raise PredictionBlocked when the report blocks everything or this outcome."""
    if not health_report:
        return
    blocked = health_report.get("summary", {}).get("blocked_outcomes", [])
    if "all" in blocked:
        raise PredictionBlocked(
            "Predictions are stopped: the data health check found errors that must be fixed "
            "first. See the Data page.")
    if outcome is not None and outcome in blocked:
        raise PredictionBlocked(
            f"{outcome.capitalize()} predictions are stopped: the data health check blocked "
            f"{outcome} for this file. See the Data page.")


def allowed_prediction_outcomes(health_report: dict | None, outcomes) -> list[str]:
    if not health_report:
        return list(outcomes)
    blocked = health_report.get("summary", {}).get("blocked_outcomes", [])
    if "all" in blocked:
        return []
    return [o for o in outcomes if o not in blocked]


# PROCESS 4 – SMOOTHING (fix 5)

def smooth_rate(total: pd.Series, count: pd.Series, company: pd.Series | float,
                k: float = SMOOTHING_K) -> pd.Series:
    """(team total + k x company rate) / (team count + k). NaN where count is 0."""
    total, count = pd.Series(total, dtype=float), pd.Series(count, dtype=float)
    out = (total + k * company) / (count + k)
    return out.where(count > 0)


def above_normal(smoothed: pd.Series, company: pd.Series | float, outcome: str) -> pd.Series:
    margin = TURNOVER_MARGIN if outcome == "turnover" else RELATIVE_MARGIN * company
    return (smoothed > company + margin).astype(float).where(smoothed.notna())


# PROCESS 5 – SURVEY-RESPONSE WARNING (X7, plain language)

def survey_response_warning(data: pd.DataFrame, outcome_column: str,
                            health_report: dict | None = None) -> dict | None:
    """Leavers answer less often -> teams about to lose people look better than they are."""
    leaver, stayer = None, None
    if {"responded", outcome_column} <= set(data.columns):
        known = data[data[outcome_column].notna()]
        if outcome_column == "left_voluntarily_12m" and known["responded"].nunique() > 1:
            rates = known.groupby(known[outcome_column].astype(int))["responded"].mean()
            if 0 in rates.index and 1 in rates.index:
                leaver, stayer = float(rates[1]), float(rates[0])
    if leaver is None and health_report:
        for check in health_report.get("checks", []):
            if check.get("code") == "NR_LEAVERS" and check.get("status") == "WARN":
                m = check.get("metrics", {})
                leaver, stayer = m.get("leaver_rate"), m.get("stayer_rate")
    if leaver is None or stayer is None or stayer - leaver <= RESPONSE_GAP_WARN:
        return None
    return {
        "leaver_response_rate": round(leaver, 3),
        "stayer_response_rate": round(stayer, 3),
        "text": (f"People who later left answered the survey less often ({leaver:.0%} vs "
                 f"{stayer:.0%} of those who stayed). So a team that is about to lose people "
                 "can look better on the survey than it really is. Treat team scores here as "
                 "warning signs, not causes, and use the Outcomes page to decide what to improve."),
    }


# PROCESS 6 – Q12 ATTRIBUTION: WITHIN-DEPARTMENT PERMUTATION (fix 1)

def permute_within(frame: pd.DataFrame, columns: list[str], groups: list[str],
                   rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle whole rows of `columns` among rows that share the same `groups` values."""
    out = frame.copy()
    keys = [g for g in groups if g in frame.columns]
    if not keys:
        order = rng.permutation(len(frame))
        out[columns] = frame[columns].to_numpy()[order]
        return out
    values = frame[columns].to_numpy()
    new = values.copy()
    for _, idx in frame.groupby(keys).indices.items():
        new[idx] = values[rng.permutation(idx)]
    out[columns] = new
    return out


def permutation_test(score_fn: Callable[[pd.DataFrame], float], frame: pd.DataFrame,
                     columns: list[str], groups: list[str], observed: float,
                     n: int = N_PERMUTATIONS, seed: int = 0) -> dict[str, Any]:
    """score_fn(frame) -> AUC on the test year after refitting on `frame`."""
    rng = np.random.default_rng(seed)
    null = np.array([score_fn(permute_within(frame, columns, groups, rng)) for _ in range(n)])
    null = null[~np.isnan(null)]
    if not len(null) or np.isnan(observed):
        return {"status": "untested", "adds_signal": False, "n_permutations": int(len(null))}
    cut = float(np.quantile(null, PERMUTATION_LEVEL))
    p = float((1 + (null >= observed).sum()) / (1 + len(null)))
    return {"status": "passed" if observed > cut else "failed",
            "adds_signal": bool(observed > cut), "observed_auc": round(float(observed), 4),
            "shuffled_auc_mean": round(float(null.mean()), 4),
            "shuffled_auc_95th": round(cut, 4), "p_value": round(p, 4),
            "n_permutations": int(len(null))}


def attribution_text(result: dict[str, Any], model_auc: float, base_auc: float | None) -> str:
    gap = "" if base_auc is None or np.isnan(base_auc) else (
        f" With Q12 the model scores {model_auc:.0%}; department and team size alone score "
        f"{base_auc:.0%}.")
    if result.get("adds_signal"):
        return ("Q12 answers add warning signs beyond department and team size: the real model "
                "beat the same model with answers shuffled between teams of the same "
                f"department ({result['n_permutations']} shuffles)." + gap +
                " These are warning signs, not causes.")
    if result.get("status") == "untested":
        return "The Q12 contribution could not be tested yet."
    return ("Q12 answers did not add warning signs beyond department and team size: shuffling "
            "the answers between teams of the same department predicted about as well." + gap +
            " The ranking reflects department, size and history, so no question is listed as "
            "pushing risk up.")


# PROCESS 7 – CHANGE GATE (fix 2)

def _bootstrap_lift(y, model, base, n=N_BOOTSTRAP, seed=0, auc_fn=None):
    rng = np.random.default_rng(seed)
    y, model, base = np.asarray(y), np.asarray(model), np.asarray(base)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(auc_fn(y[i], model[i]) - auc_fn(y[i], base[i]))
    if not diffs:
        return float("nan"), float("nan")
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def change_gate(validation: pd.DataFrame, auc_fn, score: str = "_score",
                target: str = "target", baseline: str = "previous_rate",
                previous_target: str = "previous_target", changed: str = "changed",
                band: str = "_band", unit: str = "teams",
                min_changed: int = CHANGE_MIN_TEAMS) -> dict[str, Any]:
    """Third gate: on teams that changed, the model must beat 'last year's rate'."""
    v = validation[validation[changed].astype(bool) & validation[baseline].notna()
                   & validation[target].notna()]
    n = int(len(v))
    if n < min_changed or v[target].nunique() < 2:
        return {"status": "insufficient", "changed_teams": n,
                "text": (f"Only {n} {unit} changed clearly (scores moved or a new manager) in the "
                         f"test period, too few to test; predictions for {unit} that just changed "
                         "lean on their history.")}
    m_auc = auc_fn(v[target], v[score])
    b_auc = auc_fn(v[target], v[baseline])
    low, high = _bootstrap_lift(v[target], v[score], v[baseline], auc_fn=auc_fn)
    was_high = v[v[previous_target] == 1] if previous_target in v.columns else v.iloc[0:0]
    turned = was_high[was_high[target] == 0]
    recall = float((turned[band] != "High").mean()) if len(turned) else float("nan")
    called_better = was_high[was_high[band] == "Typical"]
    false_rate = float(called_better[target].mean()) if len(called_better) else float("nan")
    lift = m_auc - b_auc
    passed = (lift >= CHANGE_MIN_LIFT and not np.isnan(low) and low > 0
              and (np.isnan(false_rate) or false_rate <= CHANGE_MAX_FALSE_TURNAROUND))
    result = {
        "status": "passed" if passed else "failed", "changed_teams": n,
        "model_auc": round(float(m_auc), 4), "baseline_auc": round(float(b_auc), 4),
        "lift": round(float(lift), 4), "lift_95ci": [round(low, 4), round(high, 4)],
        "turnarounds": int(len(turned)),
        "turnaround_recall": None if np.isnan(recall) else round(recall, 3),
        "predicted_turnarounds": int(len(called_better)),
        "false_turnaround_rate": None if np.isnan(false_rate) else round(false_rate, 3),
    }
    if passed:
        result["text"] = (f"On the {n} {unit} that changed, the model beat the history-only "
                          f"baseline ({m_auc:.0%} vs {b_auc:.0%}), so it can be trusted to spot "
                          f"{unit} that are turning around or slipping.")
    else:
        reasons = []
        if lift < CHANGE_MIN_LIFT or np.isnan(low) or low <= 0:
            reasons.append(f"it did not clearly beat the history-only baseline ({m_auc:.0%} vs {b_auc:.0%})")
        if not np.isnan(false_rate) and false_rate > CHANGE_MAX_FALSE_TURNAROUND:
            reasons.append(f"{false_rate:.0%} of the teams it called improved were still above normal")
        result["text"] = (f"On the {n} {unit} that changed, " + " and ".join(reasons)
                          + f". For {unit} with a recent change, treat the prediction as unconfirmed.")
    return result


# PROCESS 8 – CALIBRATION (fix 3)

def calibration_check(predicted, observed, min_per_bin: int = CALIBRATION_MIN_PER_BIN,
                      unit: str = "teams") -> dict[str, Any]:
    """Bins by predicted risk; a bin is off if its gap is > 10 points AND > 2 standard
    errors (SE = sqrt(p(1-p)/n), p = predicted). Any bin off -> hide percentages."""
    p = np.asarray(predicted, float)
    o = np.asarray(observed, float)
    ok = ~(np.isnan(p) | np.isnan(o))
    p, o = p[ok], o[ok]
    n = len(p)
    if n >= 5 * min_per_bin:
        bins = 5
    elif n >= 3 * min_per_bin:
        bins = 3
    else:
        return {"status": "insufficient", "show_percentages": False, "n": int(n),
                "text": (f"Only {n} {unit} could be checked ({3 * min_per_bin} needed), so "
                         "percentages are hidden and only risk bands are shown.")}
    order = np.argsort(p, kind="mergesort")
    rows = []
    for k, idx in enumerate(np.array_split(order, bins)):
        pb, ob, nb = float(p[idx].mean()), float(o[idx].mean()), int(len(idx))
        se = float(np.sqrt(max(pb * (1 - pb), 1e-12) / nb))
        gap = ob - pb
        off = abs(gap) > CALIBRATION_MAX_GAP and abs(gap) > CALIBRATION_SE_MULTIPLIER * se
        rows.append({"bin": k + 1, unit: nb, "predicted": round(pb, 4), "observed": round(ob, 4),
                     "gap": round(gap, 4), "standard_error": round(se, 4), "off": bool(off)})
    table = pd.DataFrame(rows)
    off = bool(table["off"].any())
    worst = table.loc[table["gap"].abs().idxmax()]
    return {"status": "off" if off else "passed", "show_percentages": not off, "n": int(n),
            "bins": table,
            "text": ("On data it never saw, the predicted percentages matched what happened "
                     f"(largest gap {abs(worst['gap']) * 100:.0f} points), so they are shown."
                     if not off else
                     f"On data it never saw, {int(table['off'].sum())} of {bins} risk groups were "
                     f"off by more than 10 points (worst: predicted {worst['predicted']:.0%}, "
                     f"actual {worst['observed']:.0%}). Percentages are hidden; the High / "
                     "Elevated / Typical bands still rank teams.")}


def calibration_pending() -> dict[str, Any]:
    return {"status": "pending", "show_percentages": False,
            "text": ("Calibration check pending: it needs a later year with known outcomes that "
                     "the model did not learn from. Until then only risk bands are shown.")}
