"""Statistical audit: can these results be trusted, and can anyone check them?

Every check returns the same plain record, so a reader never has to guess:

  area     which question it answers (see AREAS)
  check    what was checked, in a few words
  status   pass | warn | fail | skip
  result   what we found, in one plain sentence
  why      why it matters, in one plain sentence
  details  the numbers behind it (a small table), for anyone who wants them

The five areas:
  math          Is the math right?        independent recomputation (statsmodels),
                                          every reported number recomputed,
                                          multiple-testing control redone
  stability     Are the results stable?   repeat in another survey, leave one
                                          department out, overlapping questions,
                                          smallest effect the data can detect
  fit           Do the models fit?        absence spread, turnover calibration
  money         How sure are the money    ranges from confidence intervals,
                figures?                  cost-assumption sensitivity
  reproducible  Can it be reproduced?     data and method fingerprints, code
                                          version, packages, health check

run_audit() returns the checks, a verdict and an audit record (JSON-safe) that
can be downloaded with a reviewer's sign-off.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats_tools as st
from .financial import Assumptions, money
from .insights import ITEM_LABELS
from .model_spec import SPEC_VERSION, SPECS, design_matrix, fit_outcome_model, spec_fingerprint
from .priorities import avoided_absence_days, avoided_leavers, factor_value
from .waves import model_sample, wave_label


# PROCESS 2 – CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]
AREAS = {
    "math": ("Is the math right?", ":material/calculate:"),
    "stability": ("Are the results stable?", ":material/balance:"),
    "fit": ("Do the models fit the data?", ":material/straighten:"),
    "money": ("How sure are the money figures?", ":material/payments:"),
    "reproducible": ("Can it be reproduced?", ":material/history:"),
}
STATUS_ORDER = ["fail", "warn", "pass", "skip"]
STATUS_LABEL = {"pass": "Pass", "warn": "Warning", "fail": "Fail", "skip": "Not checked"}

COEF_TOL = 1e-6          # coefficients must match the independent library this closely
SE_TOL = 0.01            # standard errors within 1%
CONVENTION_WARN = 0.05   # expected- vs observed-information gap worth mentioning
VIF_WARN = 5.0           # overlap between questions that makes single-question effects shaky
CALIBRATION_GAP = 0.03   # predicted vs actual turnover within 3 points per group
POWER_Z = 2.8            # 80% power at 5% two-sided significance
COST_SCENARIOS = [0.25, 0.5, 1.0, 2.0]
MIN_EFFECT = {"turnover": np.log(1.10), "absence": np.log(1.10), "performance": 0.05}
PACKAGES = ["pandas", "numpy", "scipy", "statsmodels", "streamlit", "altair"]


def _check(area, check, status, result, why, details=None) -> dict[str, Any]:
    return {"area": area, "check": check, "status": status, "result": result, "why": why,
            "details": details if details is not None else pd.DataFrame()}


def _skip(area, check, reason, why) -> dict[str, Any]:
    return _check(area, check, "skip", reason, why)


def audit_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the rows analyze_dataframe() fits: respondents in the model wave."""
    rows = model_sample(df)
    if "responded" in rows.columns:
        rows = rows[rows["responded"] == 1]
    if "wave" in rows.columns and rows["wave"].notna().any():
        rows = rows[rows["wave"] == rows["wave"].dropna().min()]
    return rows.copy()


def _extras(analysis: dict, outcome: str) -> tuple[str, ...]:
    return tuple((analysis.get("adjustments") or {}).get("model_controls_added", {}).get(outcome, []))


def _outcomes(analysis: dict, sample: pd.DataFrame) -> list[str]:
    reported = set(analysis["results"]["outcome"])
    return [o for o in SPECS if o in reported and SPECS[o].outcome_column in sample.columns]


# PROCESS 3 – IS THE MATH RIGHT?

def _statsmodels_fit(sample, outcome, extra, alpha):
    import statsmodels.api as sm

    spec = SPECS[outcome]
    X = design_matrix(sample, outcome, extra)
    y = sample[spec.outcome_column].astype(float)
    if spec.outcome_column in spec.standardize:
        y = pd.Series(st.zscore(y), index=sample.index)
    groups = pd.factorize(sample[spec.cluster])[0]
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
    if spec.method == "logit":
        return sm.Logit(y, X).fit(disp=0, **kw)
    if spec.method == "negbin":
        family = sm.families.NegativeBinomial(alpha=alpha)
        return sm.GLM(y, X, family=family).fit(**kw)
    return sm.OLS(y, X).fit(**kw)


def check_independent_recompute(sample, analysis) -> dict[str, Any]:
    why = ("The models run on the app's own code; a second, widely used statistics library "
           "must give the same answers.")
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        return _skip("math", "Same answers from an independent statistics library",
                     "statsmodels is not installed (pip install statsmodels).", why)
    rows = []
    for outcome in _outcomes(analysis, sample):
        extra = _extras(analysis, outcome)
        ours = fit_outcome_model(sample, outcome, extra)
        theirs = _statsmodels_fit(sample, outcome, extra, ours.extra.get("alpha"))
        coef = float(max(abs(ours.params[i] - theirs.params[i]) for i in st.ITEMS))
        convention = 0.0
        if SPECS[outcome].method == "negbin":
            # statsmodels uses observed information for this model, the app expected
            # information (both standard). Compare like with like, report the gap.
            se_like = _negbin_se_observed(sample, outcome, extra, ours)
            convention = float(max(abs(ours.se[i] / se_like[i] - 1) for i in st.ITEMS))
        else:
            se_like = ours.se
        se = float(max(abs(se_like[i] / theirs.bse[i] - 1) for i in st.ITEMS))
        rows.append({"outcome": outcome, "method": SPECS[outcome].method_name,
                     "largest_coefficient_difference": coef, "largest_se_difference": se,
                     "convention_gap": convention,
                     "match": coef < COEF_TOL and se < SE_TOL})
    table = pd.DataFrame(rows)
    ok = bool(table["match"].all())
    gap = float(table["convention_gap"].max())
    note = (f" For absence, the app's margins of error use a different standard formula "
            f"(expected information) than statsmodels; they differ by up to {gap:.1%}."
            if gap > 0 else "")
    return _check(
        "math", "Same answers from an independent statistics library",
        "fail" if not ok else "warn" if gap > CONVENTION_WARN else "pass",
        (f"All {len(table)} models give the same effects and margins of error in statsmodels."
         + note if ok else
         "Some models give different answers in statsmodels: "
         + ", ".join(table.loc[~table["match"], "outcome"]) + ". Do not share until fixed."),
        why, table)


def _negbin_se_observed(sample, outcome, extra, fit) -> pd.Series:
    """The app's clustered standard errors rebuilt with observed information."""
    spec = SPECS[outcome]
    X = design_matrix(sample, outcome, extra).to_numpy(float)
    y = sample[spec.outcome_column].to_numpy(float)
    alpha = fit.extra["alpha"]
    mu = np.exp(X @ fit.params.to_numpy())
    bread = np.linalg.inv((X * (mu * (1 + alpha * y) / (1 + alpha * mu) ** 2)[:, None]).T @ X)
    cov = st._sandwich(X, X * ((y - mu) / (1 + alpha * mu))[:, None], bread, sample[spec.cluster])
    return pd.Series(np.sqrt(np.diag(cov)), index=fit.params.index)


def check_reported_numbers(sample, analysis) -> dict[str, Any]:
    results = analysis["results"]
    rows = []
    for outcome in _outcomes(analysis, sample):
        fit = fit_outcome_model(sample, outcome, _extras(analysis, outcome))
        ratio = SPECS[outcome].method in ("logit", "negbin")
        for _, r in results[results["outcome"] == outcome].iterrows():
            recomputed = float(np.exp(fit.params[r["item"]]) if ratio else fit.params[r["item"]])
            rows.append({"outcome": outcome, "item": r["item"], "reported": float(r["effect"]),
                         "recomputed": recomputed, "difference": abs(recomputed - r["effect"])})
    table = pd.DataFrame(rows)
    ok = bool((table["difference"] < 1e-9).all()) and len(table) == len(results)
    return _check(
        "math", "Every reported number recomputed from the fixed method",
        "pass" if ok else "fail",
        (f"All {len(table)} effects shown in the app were recomputed from scratch and match "
         f"exactly (method version {SPEC_VERSION}, fingerprint {spec_fingerprint()})." if ok else
         "Some effects shown in the app differ from a fresh calculation. Re-run the analysis."),
        "Shows the numbers on screen came from the pre-defined method, not an old or edited run.",
        table)


def check_multiple_testing(analysis) -> dict[str, Any]:
    results = analysis["results"]
    why = ("Testing 36 links at once produces some false alarms by chance; the app keeps the "
           "expected false-alarm share at 5% or less.")
    if len(results) != 3 * len(st.ITEMS):
        return _skip("math", "False-alarm control redone",
                     "Some outcomes were blocked by the health check, so the 36-test set "
                     "cannot be rebuilt here.", why)
    redo = st.bh_fdr(results["p_value"].to_numpy(), q=0.05)
    same = bool((redo == results["fdr_significant"].to_numpy()).all())
    passed = int(redo.sum())
    return _check("math", "False-alarm control redone", "pass" if same else "fail",
                  (f"Redone independently: the same {passed} of 36 links pass the 5% "
                   "false-discovery rule." if same else
                   "The false-alarm control gives a different answer when redone."), why,
                  results[["outcome", "item", "p_value", "fdr_significant"]].assign(redone=redo))


# PROCESS 4 – ARE THE RESULTS STABLE?

def check_replication(df, analysis) -> dict[str, Any]:
    why = "A link that shows up again in a different survey is far less likely to be a fluke."
    findings = analysis["findings"]
    findings = findings[findings["robustness_status"] == "ROBUST"]
    if "wave" not in df.columns or findings.empty:
        return _skip("stability", "Same finding in another survey",
                     "Needs findings that passed all checks and more than one survey.", why)
    model = audit_sample(df)["wave"].iloc[0]
    others = []
    for wave in sorted(df["wave"].dropna().unique()):
        rows = df[df["wave"] == wave]
        rows = rows[rows["responded"] == 1] if "responded" in rows.columns else rows
        if wave != model and all(rows[SPECS[o].outcome_column].notna().any()
                                 for o in set(findings["outcome"])):
            others.append((wave, rows))
    if not others:
        return _skip("stability", "Same finding in another survey",
                     "Only one survey has business outcomes, so there is nothing to repeat in.", why)
    table_rows = []
    for wave, rows in others:
        fits = {o: fit_outcome_model(rows, o, _extras(analysis, o)) for o in set(findings["outcome"])}
        for _, f in findings.iterrows():
            fit = fits[f["outcome"]]
            ratio = SPECS[f["outcome"]].method in ("logit", "negbin")
            same_way = np.sign(fit.params[f["item"]]) == np.sign(np.log(f["effect"]) if ratio else f["effect"])
            p = float(fit.pvalues[f["item"]])
            verdict = ("Repeated" if same_way and p < 0.10 else
                       "Same direction, weaker" if same_way else "Did not repeat")
            table_rows.append({"finding": f"{ITEM_LABELS[f['item']]} → {f['outcome']}",
                               "survey": wave_label(wave),
                               "effect_there": float(np.exp(fit.params[f["item"]]) if ratio else fit.params[f["item"]]),
                               "p_value": p, "verdict": verdict})
    table = pd.DataFrame(table_rows)
    repeated = int((table["verdict"] == "Repeated").sum())
    failed = int((table["verdict"] == "Did not repeat").sum())
    status = "pass" if failed == 0 and repeated == len(table) else "warn"
    return _check("stability", "Same finding in another survey", status,
                  f"{repeated} of {len(table)} headline findings show up again in "
                  + " and ".join(wave_label(w) for w, _ in others)
                  + (f"; {failed} point the other way there." if failed else "."),
                  why, table)


def check_leave_one_department_out(sample, analysis) -> dict[str, Any]:
    why = "A headline should not depend on one department; if it does, say so or drop it."
    findings = analysis["findings"]
    findings = findings[findings["robustness_status"] == "ROBUST"]
    if findings.empty or "department" not in sample.columns:
        return _skip("stability", "Holds without any single department",
                     "Needs findings that passed all checks and a department column.", why)
    rows = []
    for dept in sorted(sample["department"].dropna().unique()):
        rest = sample[sample["department"] != dept]
        for outcome in set(findings["outcome"]):
            fit = fit_outcome_model(rest, outcome, _extras(analysis, outcome))
            ratio = SPECS[outcome].method in ("logit", "negbin")
            for _, f in findings[findings["outcome"] == outcome].iterrows():
                full = np.log(f["effect"]) if ratio else f["effect"]
                now = float(fit.params[f["item"]])
                rows.append({"finding": f"{ITEM_LABELS[f['item']]} → {outcome}",
                             "left_out": dept, "share_of_effect": now / full if full else np.nan})
    table = pd.DataFrame(rows)
    worst = table.groupby("finding")["share_of_effect"].min().reset_index()
    fragile = worst[worst["share_of_effect"] < 0.5]
    status = "pass" if fragile.empty else "warn"
    return _check(
        "stability", "Holds without any single department", status,
        ("Every headline finding keeps at least half its size when any one department is "
         "left out." if fragile.empty else
         f"{len(fragile)} finding(s) shrink by more than half when one department is left out: "
         + "; ".join(fragile["finding"]) + "."),
        why, table.pivot(index="finding", columns="left_out", values="share_of_effect").reset_index())


def check_overlap(sample, analysis) -> dict[str, Any]:
    why = ("The 12 questions move together. When overlap is high, the model struggles to tell "
           "which single question matters, so lean on the overall pattern, not one question.")
    rows = []
    for outcome in _outcomes(analysis, sample):
        X = design_matrix(sample, outcome, _extras(analysis, outcome)).drop(columns="Intercept")
        X = X.loc[:, X.std() > 0]
        inv = np.linalg.pinv(np.corrcoef(X.to_numpy(float), rowvar=False))
        vif = pd.Series(np.diag(inv), index=X.columns)
        rows += [{"outcome": outcome, "item": i, "question": ITEM_LABELS[i], "vif": float(vif[i])}
                 for i in st.ITEMS]
    table = pd.DataFrame(rows)
    by_item = table.groupby(["item", "question"])["vif"].max().reset_index()
    high = by_item[by_item["vif"] >= VIF_WARN]
    return _check(
        "stability", "Overlap between questions", "pass" if high.empty else "warn",
        (f"Overlap is acceptable for every question (highest {by_item['vif'].max():.1f}; "
         f"concern starts at {VIF_WARN:.0f})." if high.empty else
         f"{len(high)} question(s) overlap strongly with others: "
         + ", ".join(high["question"]) + ". Read their single-question effects with care."),
        why, by_item.sort_values("vif", ascending=False))


def check_detectable_effect(sample, analysis) -> dict[str, Any]:
    rows = []
    for outcome in _outcomes(analysis, sample):
        fit = fit_outcome_model(sample, outcome, _extras(analysis, outcome))
        mde = POWER_Z * float(np.median([fit.se[i] for i in st.ITEMS]))
        ratio = SPECS[outcome].method in ("logit", "negbin")
        shown = (f"ratio below {np.exp(-mde):.2f} or above {np.exp(mde):.2f}" if ratio
                 else f"beta of {mde:.3f} or more")
        rows.append({"outcome": outcome, "people": int(fit.n), "smallest_detectable": shown,
                     "reporting_rule": SPECS[outcome].min_effect,
                     "enough_data": mde <= MIN_EFFECT[outcome]})
    table = pd.DataFrame(rows)
    short = table[~table["enough_data"]]
    return _check(
        "stability", "Enough data to see small effects", "pass" if short.empty else "warn",
        ("The data is large enough to detect every effect big enough to report." if short.empty else
         "For " + ", ".join(short["outcome"]) + ", small-but-real effects may be missed: "
         "'not significant' there does not mean 'no effect'."),
        "Without enough people, a real link can look like no link.", table)


# PROCESS 5 – DO THE MODELS FIT?

def check_absence_spread(sample, analysis) -> dict[str, Any]:
    why = ("Absence days vary more than a simple count model allows; the app's model must "
           "handle that or its margins of error are too narrow.")
    column = SPECS["absence"].outcome_column
    if "absence" not in _outcomes(analysis, sample):
        return _skip("fit", "Absence model handles uneven absence", "No absence data.", why)
    y = sample[column].dropna()
    ratio = float(y.var() / y.mean()) if y.mean() else np.nan
    alpha = float(fit_outcome_model(sample, "absence", _extras(analysis, "absence")).extra["alpha"])
    return _check("fit", "Absence model handles uneven absence", "pass",
                  (f"Absence is {ratio:.1f}× more spread out than a simple count; the "
                   f"negative binomial model allows for this (dispersion {alpha:.2f})."
                   if ratio > 1.2 else
                   "Absence is not unusually spread out; the model is still appropriate."),
                  why, pd.DataFrame([{"variance_to_mean": ratio, "dispersion_alpha": alpha}]))


def check_turnover_calibration(sample, analysis) -> dict[str, Any]:
    why = "If the model says 20% will leave, about 20% should actually have left."
    if "turnover" not in _outcomes(analysis, sample):
        return _skip("fit", "Turnover model's predictions match reality", "No turnover data.", why)
    extra = _extras(analysis, "turnover")
    fit = fit_outcome_model(sample, "turnover", extra)
    X = design_matrix(sample, "turnover", extra)
    predicted = 1 / (1 + np.exp(-(X.to_numpy(float) @ fit.params.to_numpy())))
    data = pd.DataFrame({"predicted": predicted,
                         "actual": sample[SPECS["turnover"].outcome_column].to_numpy(float)})
    data["group"] = pd.qcut(data["predicted"].rank(method="first"), 10, labels=range(1, 11))
    table = data.groupby("group", observed=True).agg(people=("actual", "size"),
                                                     predicted=("predicted", "mean"),
                                                     actual=("actual", "mean")).reset_index()
    gap = float((table["predicted"] - table["actual"]).abs().max())
    return _check("fit", "Turnover model's predictions match reality",
                  "pass" if gap <= CALIBRATION_GAP else "warn",
                  f"Across 10 groups from lowest to highest risk, predicted and actual turnover "
                  f"differ by at most {gap:.1%}"
                  + (" (good)." if gap <= CALIBRATION_GAP else
                     f", above the {CALIBRATION_GAP:.0%} comfort level."),
                  why, table)


# PROCESS 6 – HOW SURE ARE THE MONEY FIGURES?

def check_money_range(df, analysis, a) -> dict[str, Any]:
    why = "Each effect has a margin of error, so each dollar figure is a range, not one number."
    factors = factor_value(df, analysis, a)
    factors = factors[factors["yearly_value"] > 0].head(3)
    if factors.empty:
        return _skip("money", "Range around each dollar figure", "No factor has a dollar value.", why)
    sample = audit_sample(df)
    scale = a.scale(len(sample))
    base_rate = float(sample[SPECS["turnover"].outcome_column].mean())
    base_days = float(sample[SPECS["absence"].outcome_column].mean())
    found = analysis["findings"].set_index(["item", "outcome"])
    rows = []
    for _, f in factors.iterrows():
        below = f["people_below"] / scale
        lows, highs = [], []
        for bound in ("ci_low", "ci_high"):
            value = f["performance_value"]
            if (f["item"], "turnover") in found.index:
                value += avoided_leavers(below, base_rate, found.loc[(f["item"], "turnover"), bound]) \
                    * scale * a.cost_per_leaver
            if (f["item"], "absence") in found.index:
                value += avoided_absence_days(below, base_days, found.loc[(f["item"], "absence"), bound]) \
                    * scale * a.day_cost
            (lows if bound == "ci_low" else highs).append(value)
        low, high = sorted([lows[0], highs[0]])
        rows.append({"factor": f["label"], "estimate": f["yearly_value"], "cautious": low,
                     "optimistic": high})
    table = pd.DataFrame(rows)
    c = a.currency
    top = table.iloc[0]
    return _check("money", "Range around each dollar figure",
                  "pass" if (table["cautious"] > 0).all() else "warn",
                  f"{top['factor']}: about {money(top['estimate'], c)} a year, likely between "
                  f"{money(top['cautious'], c)} and {money(top['optimistic'], c)}. "
                  + ("Even the cautious end is a real saving." if (table["cautious"] > 0).all()
                     else "The cautious end is close to zero for some factors."),
                  why, table)


def check_cost_sensitivity(df, analysis, a) -> dict[str, Any]:
    why = ("Replacement cost is an assumption. The advice should not flip when the "
           "assumption changes.")
    rows = []
    for pct in COST_SCENARIOS:
        scenario = Assumptions(avg_salary=a.avg_salary, replacement_cost_pct=pct,
                               absence_day_cost=a.absence_day_cost,
                               value_per_rating_point=a.value_per_rating_point,
                               headcount=a.headcount, currency=a.currency)
        f = factor_value(df, analysis, scenario)
        f = f[f["yearly_value"] > 0]
        if f.empty:
            continue
        rows.append({"replacement_cost": f"{pct:.0%} of salary", "top_factor": f.iloc[0]["label"],
                     "top_value": float(f.iloc[0]["yearly_value"])})
    if not rows:
        return _skip("money", "Advice holds under other cost assumptions",
                     "No factor has a dollar value.", why)
    table = pd.DataFrame(rows)
    same = table["top_factor"].nunique() == 1
    c = a.currency
    return _check("money", "Advice holds under other cost assumptions", "pass" if same else "warn",
                  (f"{table['top_factor'].iloc[0]} stays the top lever whether replacing a leaver "
                   f"costs 25% or 200% of salary (worth {money(table['top_value'].min(), c)} to "
                   f"{money(table['top_value'].max(), c)} a year)." if same else
                   "The top lever changes with the cost assumption: "
                   + "; ".join(f"{r['replacement_cost']} → {r['top_factor']}"
                               for _, r in table.iterrows()) + "."),
                  why, table)


# PROCESS 7 – CAN IT BE REPRODUCED?

def data_fingerprint(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df.reindex(sorted(df.columns), axis=1), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()[:12].upper()


def code_version() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "app_pages", "app.py"],
                               cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() + (" + uncommitted changes" if dirty else "")) \
            if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def check_reproducible(df, analysis, health_report) -> list[dict[str, Any]]:
    checks = []
    commit = code_version()
    checks.append(_check(
        "reproducible", "Code version recorded",
        "pass" if commit and "uncommitted" not in commit else "warn",
        (f"Code version {commit}." if commit and "uncommitted" not in commit else
         f"Code version {commit}: commit the changes so this run matches saved code." if commit else
         "The code is not saved in version control (git has no commits), so this run cannot be "
         "tied to an exact code version. Commit the project to fix this."),
        "Lets anyone rerun the exact same code later and get the same numbers."))
    checks.append(_check(
        "reproducible", "Data and method fingerprints", "pass",
        f"Data fingerprint {data_fingerprint(df)}; method version {SPEC_VERSION}, fingerprint "
        f"{spec_fingerprint()}; {analysis['sample_size']:,} employees in the models.",
        "If either fingerprint changes, the numbers may change too; matching fingerprints mean "
        "the same data and method."))
    if health_report:
        summary = health_report.get("summary", {})
        blocked = [b for b in summary.get("blocked_outcomes", []) if b != "all"]
        added = {o: c for o, c in (analysis.get("adjustments") or {}).get("model_controls_added", {}).items() if c}
        checks.append(_check(
            "reproducible", "Data health check applied",
            "warn" if blocked else "pass",
            ("Health check passed and was applied"
             + (": blocked " + ", ".join(blocked) if blocked else "")
             + ("; controls added: " + "; ".join(f"{o}: {', '.join(c)}" for o, c in added.items())
                if added else "; no extra controls needed") + "."),
            "Errors in the data stop the analysis, and known biases get the controls registered "
            "for them."))
    return checks


# PROCESS 8 – RUN THE AUDIT

def run_audit(df: pd.DataFrame, analysis: dict, health_report: dict | None = None,
              assumptions: Assumptions | None = None) -> dict[str, Any]:
    """Every check, a verdict and a downloadable record."""
    sample = audit_sample(df)
    a = assumptions or Assumptions(headcount=len(sample))
    runners = [
        lambda: check_independent_recompute(sample, analysis),
        lambda: check_reported_numbers(sample, analysis),
        lambda: check_multiple_testing(analysis),
        lambda: check_replication(df, analysis),
        lambda: check_leave_one_department_out(sample, analysis),
        lambda: check_overlap(sample, analysis),
        lambda: check_detectable_effect(sample, analysis),
        lambda: check_absence_spread(sample, analysis),
        lambda: check_turnover_calibration(sample, analysis),
        lambda: check_money_range(df, analysis, a),
        lambda: check_cost_sensitivity(df, analysis, a),
    ]
    checks = []
    for runner in runners:
        try:
            checks.append(runner())
        except Exception as exc:                          # one broken check must not hide the rest
            checks.append(_check("math", "Audit check could not run", "fail",
                                 f"An audit check stopped with an error: {exc}",
                                 "Every check must run before the results are shared."))
    checks += check_reproducible(df, analysis, health_report)

    counts = {s: sum(c["status"] == s for c in checks) for s in STATUS_ORDER}
    if counts["fail"]:
        verdict = ("fail", "Do not share yet",
                   f"{counts['fail']} check(s) failed. Fix them before sharing the results.")
    elif counts["warn"]:
        verdict = ("warn", "Share with the caveats below",
                   f"The math checks out. {counts['warn']} warning(s) to mention when sharing.")
    else:
        verdict = ("pass", "Ready to share", "Every check passed.")
    return {"checks": checks, "counts": counts, "verdict": verdict,
            "record": audit_record(df, analysis, checks, verdict)}


def audit_record(df, analysis, checks, verdict, sign_off: dict | None = None) -> dict[str, Any]:
    """JSON-safe record of this run, for the audit pack."""
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": {"status": verdict[0], "summary": verdict[1], "detail": verdict[2]},
        "data_fingerprint": data_fingerprint(df),
        "rows_in_upload": int(len(df)),
        "employees_in_models": int(analysis["sample_size"]),
        "method_version": SPEC_VERSION,
        "method_fingerprint": spec_fingerprint(),
        "code_version": code_version() or "not in version control",
        "packages": package_versions(),
        "controls_added": (analysis.get("adjustments") or {}).get("model_controls_added", {}),
        "checks": [{k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in c.items()} for c in checks],
        "sign_off": sign_off or {},
    }


def audit_markdown(audit: dict[str, Any], sign_off: dict | None = None) -> str:
    status, summary, detail = audit["verdict"]
    lines = ["# Statistical audit", "", f"**{summary}.** {detail}", ""]
    for area, (question, _) in AREAS.items():
        area_checks = [c for c in audit["checks"] if c["area"] == area]
        if not area_checks:
            continue
        lines += [f"## {question}", "| Status | Check | Result |", "|---|---|---|"]
        lines += [f"| {STATUS_LABEL[c['status']]} | {c['check']} | {c['result']} |"
                  for c in area_checks]
        lines.append("")
    record = audit["record"]
    lines += ["## Record", f"- Audited: {record['audited_at']}",
              f"- Data fingerprint: {record['data_fingerprint']}",
              f"- Method: version {record['method_version']}, fingerprint {record['method_fingerprint']}",
              f"- Code version: {record['code_version']}",
              "- Packages: " + ", ".join(f"{k} {v}" for k, v in record["packages"].items())]
    if sign_off:
        lines += ["", "## Reviewer sign-off", f"- Reviewer: {sign_off.get('name', '')} "
                  f"({sign_off.get('role', '')})", f"- Decision: {sign_off.get('decision', '')}",
                  f"- Note: {sign_off.get('note', '') or '—'}", f"- Date: {sign_off.get('date', '')}"]
    return "\n".join(lines) + "\n"
