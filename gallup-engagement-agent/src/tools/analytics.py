# PROCESS 1 – IMPORTS

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats_tools as st
from .model_spec import SPECS, describe, fit_outcome_model, method_id, resolve_adjustments


# PROCESS 2 – CONFIGURATION

ITEM_LABELS = {
    "Q01": "Expectations",
    "Q02": "Materials and Equipment",
    "Q03": "Opportunity to Do Best",
    "Q04": "Recognition",
    "Q05": "Someone Cares",
    "Q06": "Development",
    "Q07": "Opinions Count",
    "Q08": "Mission and Purpose",
    "Q09": "Commitment to Quality",
    "Q10": "Best Friend at Work",
    "Q11": "Progress",
    "Q12": "Learn and Grow",
}

OUTCOME_LABELS = {
    "turnover": "voluntary turnover",
    "absence": "absenteeism",
    "performance": "performance",
}

# Derived from the fixed specifications (model_spec.SPECS): one source of truth.
MODEL_METADATA = {
    outcome: {
        "method": SPECS[outcome].method_name,
        "effect_type": SPECS[outcome].effect_type,
        "controls": describe(outcome)["controls"],
        "cluster": "team",
        "method_id": method_id(outcome),
    }
    for outcome in SPECS
}


PRIMARY_CAVEATS = {
    "turnover": {
        "NR_LEAVERS",
    },

    "absence": {
        "NR_SMD_unplanned_absence_days_6m",
        "MS_DIF",
    },

    "performance": {
        "MS_DIF",
    },
}


# PROCESS 3 – LOAD DATA

def load_agent_data(dataset_path):
    dataset_path = Path(dataset_path)
    data_path = dataset_path / "agent_data"

    survey = pd.read_csv(
        data_path / "survey_responses.csv"
    )

    individual = pd.read_csv(
        data_path / "individual_outcomes.csv"
    )

    team = pd.read_csv(
        data_path / "team_outcomes.csv"
    )

    return survey, individual, team


# PROCESS 4 – BUILD ANALYTICAL SAMPLE

def prepare_analysis_data(
    dataset_path,
    wave=1,
):
    survey, individual, team = load_agent_data(
        dataset_path
    )

    merged = survey.merge(
        individual,
        on=["wave", "employee_id"],
    )

    wave_all = merged[
        merged["wave"] == wave
    ].copy()

    respondents = wave_all[
        wave_all["responded"] == 1
    ].copy()

    return {
        "merged": merged,
        "wave_all": wave_all,
        "respondents": respondents,
        "team": team,
    }


# PROCESS 5 – HEALTH REPORT

def load_health_report(dataset_path):
    dataset_path = Path(dataset_path)

    report_path = (
        dataset_path
        / "reports"
        / "health_report.json"
    )

    if not report_path.exists():

        health_script = (
            Path(__file__)
            .with_name("health_check.py")
        )

        subprocess.run(
            [
                sys.executable,
                str(health_script),
                str(dataset_path),
            ],
            check=True,
        )

    with open(
        report_path,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    return report


# PROCESS 6 – RUN STANDARD MODELS

class HealthCheckBlocked(ValueError):
    """The data health check found errors that stop every analysis."""


def required_model_controls(health_report, df):
    """Extra controls per outcome: what the health report asks for, resolved
    against each spec's pre-registered adjustments. Returns (applied, decisions);
    decisions record what was applied, already in the model, refused or missing."""
    from .health_check import model_controls

    applied, decisions = {}, {}
    for outcome in SPECS:
        decision = resolve_adjustments(outcome, model_controls(health_report, outcome), df)
        applied[outcome] = decision["applied"]
        decisions[outcome] = decision
    return applied, decisions


def run_standard_models(df, extra_controls=None, outcomes=None):
    """Fit each allowed outcome's pre-defined model (model_spec.SPECS).
    Outcomes blocked by the health check are never fitted (v3 fix 2)."""
    extra = extra_controls or {}
    outcomes = list(SPECS) if outcomes is None else list(outcomes)
    return {
        outcome: fit_outcome_model(df, outcome, tuple(extra.get(outcome, ())))
        for outcome in SPECS
        if outcome in outcomes
    }


def enforce_health(health_report):
    """Stop when the health check blocks everything; return blocked outcomes."""
    blocked = health_report.get("summary", {}).get("blocked_outcomes", [])
    if "all" in blocked:
        failed = [c["check"] for c in health_report.get("checks", []) if c["status"] == "FAIL"]
        raise HealthCheckBlocked(
            "Analysis stopped: the data health check found errors that must be fixed first ("
            + "; ".join(failed) + "). See the Data page."
        )
    partial = [b for b in blocked if b != "all"]
    if partial and not [o for o in SPECS if o not in partial]:
        raise HealthCheckBlocked(
            "Analysis stopped: the data health check blocked every outcome ("
            + "; ".join(blocking_reasons(health_report)) + "). See the Data page."
        )
    return partial


ALL_OUTCOMES = tuple(SPECS)


def allowed_outcomes(health_report):
    """Outcomes the health check allows: the single source for the agent's gate."""
    blocked = health_report.get("summary", {}).get("blocked_outcomes", [])
    if "all" in blocked:
        return []
    return [o for o in ALL_OUTCOMES if o not in blocked]


def blocking_reasons(health_report):
    """Plain-language reasons for every FAIL check, for refusal messages."""
    reasons = []
    for c in health_report.get("checks", []):
        if c.get("status") != "FAIL":
            continue
        violations = (c.get("metrics") or {}).get("violations")
        reasons.append(
            f"{c['check']} failed"
            + (f" ({violations:,} problem rows or values)"
               if isinstance(violations, int) and violations else "")
        )
    return reasons


# PROCESS 7 – COLLECT MODEL RESULTS

def collect_model_results(models):
    """One row per item for every fitted outcome (blocked outcomes are absent)."""
    rows = []

    for outcome, fit in models.items():

        spec = SPECS[outcome]

        for item in st.ITEMS:

            if spec.method == "ols":
                ci_low, ci_high = fit.ci(item)
                effect = float(fit.params[item])
            else:
                effect, ci_low, ci_high = fit.ratio(item)

            rows.append({
                "item": item,
                "outcome": outcome,
                "effect_type": spec.effect_type,
                "effect": effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": fit.pvalues[item],
                "n": fit.n,
            })

    return pd.DataFrame(rows)


# PROCESS 8 – FDR AND MEANINGFUL EFFECT

def apply_statistical_guardrails(results):
    results = results.copy()

    results["fdr_significant"] = (
        st.bh_fdr(
            results[
                "p_value"
            ].to_numpy(),
            q=0.05,
        )
    )

    def meaningful(row):

        if row["outcome"] in [
            "turnover",
            "absence",
        ]:
            return (
                row["effect"] < 0.91
                or row["effect"] > 1.10
            )

        if row["outcome"] == "performance":
            return (
                abs(row["effect"])
                >= 0.05
            )

        return False

    results["meaningful_effect"] = (
        results.apply(
            meaningful,
            axis=1,
        )
    )

    results["reportable"] = (
        results["fdr_significant"]
        & results["meaningful_effect"]
    )

    return results


# PROCESS 9 – PRIOR-PERFORMANCE DIAGNOSTICS

def prior_performance_diagnostics(
    df,
    models,
):
    controlled_fit = models["performance"]

    diagnostics = {}

    for item in st.ITEMS:

        z = pd.DataFrame(
            {
                "performance_rating":
                    st.zscore(
                        df[
                            "performance_rating"
                        ]
                    ),

                item:
                    st.zscore(
                        df[item]
                    ),
            },
            index=df.index,
        )

        raw_fit = st.ols(
            z["performance_rating"],
            st.design(
                z,
                [item],
            ),
        )

        raw_beta = float(
            raw_fit.params[item]
        )

        controlled_beta = float(
            controlled_fit.params[item]
        )

        if abs(raw_beta) < 1e-12:
            reduction = 0.0
        else:
            reduction = (
                1
                - abs(
                    controlled_beta
                    / raw_beta
                )
            )

        if reduction >= 0.50:
            diagnostics[
                ("performance", item)
            ] = {
                "code":
                    "PRIOR_OUTCOME_SENSITIVE",

                "reason": (
                    f"The estimated association "
                    f"decreased by "
                    f"{reduction:.1%} after "
                    f"accounting for prior "
                    f"performance."
                ),
            }

    return diagnostics


# PROCESS 10 – TEAM CONFOUNDING DIAGNOSTICS

def team_confounding_diagnostics(df):
    diagnostics = {}

    work = df.assign(
        y=np.log1p(
            df[
                "unplanned_absence_days_6m"
            ]
        ),

        q01_low=(
            df["Q01"] < 3
        ).astype(float),

        night=(
            df["shift_type"] == "Night"
        ).astype(float),
    )

    xs = (
        list(st.ITEMS)
        + [
            "q01_low",
            "tenure_years",
            "night",
        ]
    )

    department_fit = st.ols(
        work["y"],

        st.design(
            work,
            xs,
            ["department"],
        ),

        work["team_id"],
    )

    within_team = st.demean(
        work,
        ["y"] + xs,
        "team_id",
    )

    team_fit = st.ols(
        within_team["y"],
        st.design(
            within_team,
            xs,
        ),
        work["team_id"],
    )

    icc, icc_p = st.icc_oneway(
        work["y"],
        work["team_id"],
    )

    for item in st.ITEMS:

        department_beta = float(
            department_fit.params[item]
        )

        team_beta = float(
            team_fit.params[item]
        )

        if abs(department_beta) < 1e-12:
            continue

        shrinkage = (
            1
            - abs(
                team_beta
                / department_beta
            )
        )

        if (
            shrinkage >= 0.50
            and icc > 0.01
            and icc_p < 0.001
        ):
            diagnostics[
                ("absence", item)
            ] = {
                "code":
                    "TEAM_LEVEL_CONFOUNDING",

                "reason": (
                    f"The estimated association "
                    f"decreased by "
                    f"{shrinkage:.1%} when "
                    f"employees within the same "
                    f"teams were compared. "
                    f"This suggests that team-level "
                    f"factors may explain much of "
                    f"the apparent relationship."
                ),
            }

    return diagnostics


# PROCESS 11 – SIMPSON DIAGNOSTICS

def simpson_diagnostics(df):
    diagnostics = {}

    for item in st.ITEMS:

        z = pd.DataFrame(
            {
                "performance_rating":
                    st.zscore(
                        df[
                            "performance_rating"
                        ]
                    ),

                item:
                    st.zscore(
                        df[item]
                    ),
            },
            index=df.index,
        )

        z["department"] = (
            df["department"]
        )

        pooled_fit = st.ols(
            z["performance_rating"],
            st.design(
                z,
                [item],
            ),
        )

        adjusted_fit = st.ols(
            z["performance_rating"],
            st.design(
                z,
                [item],
                ["department"],
            ),
        )

        pooled = float(
            pooled_fit.params[item]
        )

        adjusted = float(
            adjusted_fit.params[item]
        )

        direction_flip = (
            np.sign(pooled)
            != np.sign(adjusted)
            and abs(pooled) > 1e-6
            and abs(adjusted) > 1e-6
        )

        if direction_flip:
            diagnostics[
                ("performance", item)
            ] = {
                "code":
                    "DIRECTION_FLIP",

                "reason": (
                    "The direction of the "
                    "association changes after "
                    "department differences are "
                    "accounted for, so department "
                    "composition materially affects "
                    "the interpretation."
                ),
            }

    return diagnostics


# PROCESS 12 – COMBINE DIAGNOSTICS

def run_robustness_diagnostics(
    df,
    models,
):
    combined = {}

    # Only for outcomes that were fitted (blocked outcomes have no model).
    diagnostics_sets = []

    if "performance" in models:
        diagnostics_sets += [
            prior_performance_diagnostics(
                df,
                models,
            ),

            simpson_diagnostics(
                df
            ),
        ]

    if "absence" in models:
        diagnostics_sets.append(
            team_confounding_diagnostics(
                df
            )
        )

    for diagnostic_set in diagnostics_sets:

        for key, diagnostic in (
            diagnostic_set.items()
        ):

            combined.setdefault(
                key,
                []
            ).append(diagnostic)

    return combined


# PROCESS 12A – THRESHOLD AND SUBGROUP PATTERNS (v3 fixes 4 and 5: X4, X5)
#
# A straight-line model can hide two patterns HR needs to hear about:
# - a threshold: absence jumps only below a score (Q01 below 3). This is the
#   pre-registered derived control q01_low in the absence spec, so its term is
#   read from the absence model, not fitted again;
# - a subgroup difference: an item matters more for new hires (tenure under
#   2 years). Screened for all 12 items in one turnover model, FDR-corrected.

THRESHOLD_TESTS = [
    # (item, outcome, model term, cut-off)
    ("Q01", "absence", "q01_low", 3),
]

NEW_HIRE_YEARS = 2


def threshold_diagnostics(models, dataset_name):
    findings = []

    for item, outcome, term, cut in THRESHOLD_TESTS:

        fit = models.get(outcome)

        if fit is None or term not in fit.params.index:
            continue

        ratio, low, high = fit.ratio(term)
        p_value = float(fit.pvalues[term])

        if p_value >= 0.05 or 0.91 <= ratio <= 1.10:
            continue

        findings.append({
            "type": "threshold",
            "item": item,
            "item_label": ITEM_LABELS.get(item, item),
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
            "effect_type": "Rate Ratio (score below threshold vs at or above)",
            "threshold": f"score below {cut}",
            "effect": round(ratio, 4),
            "confidence_interval": {
                "low": round(low, 4),
                "high": round(high, 4),
                "level": 0.95,
            },
            "p_value": round(p_value, 6),
            "sample_size": int(fit.n),
            "summary": (
                f"Employees who score {item} ({ITEM_LABELS.get(item, item)}) "
                f"below {cut} have about {abs(ratio - 1):.0%} "
                f"{'more' if ratio > 1 else 'fewer'} "
                f"{OUTCOME_LABELS.get(outcome, outcome)} days than the "
                "straight-line pattern alone would suggest. The link is "
                "concentrated at low scores, so a single average effect "
                "understates it for that group."
            ),
            "calculation_id": make_calculation_id(
                dataset_name, outcome, item, f"Threshold below {cut}|{method_id(outcome)}"
            ),
        })

    return findings


def subgroup_diagnostics(df, dataset_name, extra_controls=()):
    """New hires (tenure under 2 years) vs everyone else, every item, turnover."""
    if "tenure_years" not in df.columns:
        return []

    work = df.assign(
        new_hire=(df["tenure_years"] < NEW_HIRE_YEARS).astype(float)
    )

    if work["new_hire"].nunique() < 2:
        return []

    interactions = [(item, "new_hire") for item in st.ITEMS]

    fit = st.turnover_model(
        work,
        extra_numeric=["new_hire"],
        interactions=interactions,
        extra_categorical=tuple(extra_controls),
    )

    terms = [f"{item}:new_hire" for item in st.ITEMS]
    discoveries = st.bh_fdr(
        fit.pvalues[terms].to_numpy(),
        q=0.05,
    )

    findings = []

    for item, term, significant in zip(st.ITEMS, terms, discoveries):

        ratio_of_ors, low, high = fit.ratio(term)

        if not significant or 0.91 <= ratio_of_ors <= 1.10:
            continue

        or_others = float(np.exp(fit.params[item]))
        or_new = float(np.exp(fit.params[item] + fit.params[term]))

        findings.append({
            "type": "subgroup_difference",
            "item": item,
            "item_label": ITEM_LABELS.get(item, item),
            "outcome": "turnover",
            "outcome_label": OUTCOME_LABELS["turnover"],
            "subgroup": f"tenure under {NEW_HIRE_YEARS} years",
            "effect_type": "Odds Ratio by subgroup",
            "odds_ratio_new_hires": round(or_new, 4),
            "odds_ratio_others": round(or_others, 4),
            "ratio_of_odds_ratios": round(ratio_of_ors, 4),
            "confidence_interval": {
                "low": round(low, 4),
                "high": round(high, 4),
                "level": 0.95,
                "applies_to": "ratio_of_odds_ratios",
            },
            "p_value": round(float(fit.pvalues[term]), 6),
            "sample_size": int(fit.n),
            "summary": (
                f"{ITEM_LABELS.get(item, item)} ({item}) is linked to turnover "
                f"{'more strongly' if abs(np.log(or_new)) > abs(np.log(or_others)) else 'less strongly'} "
                f"for employees with under {NEW_HIRE_YEARS} years of tenure "
                f"(odds ratio {or_new:.2f} per point) than for others "
                f"({or_others:.2f}). Report the two groups separately; the "
                "overall average hides the difference."
            ),
            "calculation_id": make_calculation_id(
                dataset_name, "turnover", item, "New-hire interaction"
            ),
        })

    return findings


def special_patterns(df, models, dataset_name, extra_controls=None):
    """Threshold and subgroup patterns for the outcomes that were fitted."""
    extra = extra_controls or {}
    found = threshold_diagnostics(models, dataset_name)
    if "turnover" in models:
        found += subgroup_diagnostics(df, dataset_name, extra.get("turnover", ()))
    return found


def attach_patterns(payloads, special_findings):
    """Put each pattern on the matching finding as related_patterns."""
    for payload in payloads:
        related = [
            {k: v for k, v in sf.items()
             if k not in ("item", "item_label", "outcome", "outcome_label")}
            for sf in special_findings
            if sf["item"] == payload["item"] and sf["outcome"] == payload["outcome"]
        ]
        if related:
            payload["related_patterns"] = related
    return payloads


# PROCESS 13 – CLASSIFY ROBUSTNESS

def classify_robustness(
    results,
    diagnostics,
):
    output = results.copy()

    statuses = []
    eligible = []
    codes = []
    reasons = []

    for _, row in output.iterrows():

        if not row["reportable"]:

            statuses.append(
                "NOT_REPORTABLE"
            )

            eligible.append(False)
            codes.append([])
            reasons.append([])

            continue

        key = (
            row["outcome"],
            row["item"],
        )

        row_diagnostics = (
            diagnostics.get(
                key,
                []
            )
        )

        if row_diagnostics:

            statuses.append("CAUTION")
            eligible.append(False)

            codes.append([
                d["code"]
                for d in row_diagnostics
            ])

            reasons.append([
                d["reason"]
                for d in row_diagnostics
            ])

        else:

            statuses.append("ROBUST")
            eligible.append(True)
            codes.append([])
            reasons.append([])

    output[
        "robustness_status"
    ] = statuses

    output[
        "priority_eligible"
    ] = eligible

    output[
        "diagnostic_codes"
    ] = codes

    output[
        "diagnostic_reasons"
    ] = reasons

    return output


# PROCESS 14 – BUSINESS CAVEATS

def select_caveats(
    health_report,
    outcome,
    item,
    question_scope="overall",
):
    selected = []

    primary_codes = (
        PRIMARY_CAVEATS.get(
            outcome,
            set(),
        )
    )

    for check in health_report["checks"]:

        if check["status"] != "WARN":
            continue

        affects = check.get(
            "affects",
            [],
        )

        if (
            "all" not in affects
            and outcome not in affects
        ):
            continue

        code = check["code"]

        if (
            question_scope == "overall"
            and code not in primary_codes
        ):
            continue

        if code == "MS_DIF":

            affected_items = (
                check
                .get("metrics", {})
                .get("items", [])
            )

            if item not in affected_items:
                continue

        metrics = check.get(
            "metrics",
            {},
        )

        if code == "NR_LEAVERS":

            leaver = (
                metrics.get(
                    "leaver_rate"
                )
            )

            stayer = (
                metrics.get(
                    "stayer_rate"
                )
            )

            detail = (
                "Survey response rates were "
                f"lower among employees who "
                f"later left ({leaver:.1%}) "
                f"than among stayers "
                f"({stayer:.1%}). "
                "This nonresponse pattern may "
                "cause the observed turnover "
                "association to be understated."
            )

        elif (
            code
            == "NR_SMD_unplanned_absence_days_6m"
        ):

            detail = (
                "Employees who responded to "
                "the survey differed in "
                "absenteeism from those who "
                "did not respond, so "
                "nonresponse may affect the "
                "estimated association."
            )

        elif code == "MS_DIF":

            detail = (
                "The item shows different "
                "response behavior across "
                "departments, so department "
                "context should be considered "
                "when interpreting the result."
            )

        else:

            detail = check["detail"]

        selected.append(detail)

    return selected


# PROCESS 15 – BUSINESS METRICS

def business_metrics(
    effect_type,
    effect,
):
    if effect_type == "Odds Ratio":

        return {
            "relative_odds_change_pct":
                round(
                    (effect - 1)
                    * 100,
                    1,
                ),

            "interpretation_unit":
                "odds, not probability",
        }

    if (
        effect_type
        == "Incidence Rate Ratio"
    ):

        return {
            "relative_rate_change_pct":
                round(
                    (effect - 1)
                    * 100,
                    1,
                )
        }

    if (
        effect_type
        == "Standardized Beta"
    ):

        return {
            "standardized_beta":
                round(
                    effect,
                    4,
                )
        }

    return {}


# PROCESS 16 – CALCULATION ID

def make_calculation_id(
    dataset,
    outcome,
    item,
    method,
):
    raw = (
        f"{dataset}|"
        f"{outcome}|"
        f"{item}|"
        f"{method}"
    )

    short_hash = (
        hashlib.sha256(
            raw.encode("utf-8")
        )
        .hexdigest()[:10]
        .upper()
    )

    return (
        f"{outcome.upper()}-"
        f"{item}-"
        f"{short_hash}"
    )


# PROCESS 17 – BUILD SAFE PAYLOAD

def build_finding_payload(
    row,
    health_report,
    dataset_name,
    question_scope="overall",
):
    outcome = row["outcome"]

    meta = MODEL_METADATA[
        outcome
    ]

    diagnostic_reasons = (
        row["diagnostic_reasons"]
        if isinstance(
            row["diagnostic_reasons"],
            list,
        )
        else []
    )

    payload = {
        "dataset":
            dataset_name,

        "item":
            row["item"],

        "item_label":
            ITEM_LABELS.get(
                row["item"],
                row["item"],
            ),

        "outcome":
            outcome,

        "outcome_label":
            OUTCOME_LABELS.get(
                outcome,
                outcome,
            ),

        "method":
            meta["method"],

        "effect_type":
            row["effect_type"],

        "effect":
            round(
                float(
                    row["effect"]
                ),
                4,
            ),

        "confidence_interval": {
            "low": round(
                float(
                    row["ci_low"]
                ),
                4,
            ),

            "high": round(
                float(
                    row["ci_high"]
                ),
                4,
            ),

            "level": 0.95,
        },

        "sample_size":
            int(row["n"]),

        "business_metrics":
            business_metrics(
                row["effect_type"],
                float(
                    row["effect"]
                ),
            ),

        "robustness_status":
            row[
                "robustness_status"
            ],

        "priority_recommendation_allowed":
            bool(
                row[
                    "priority_eligible"
                ]
            ),

        "robustness_reasons":
            diagnostic_reasons,

        "controls":
            meta["controls"],

        "required_caveats":
            select_caveats(
                health_report,
                outcome,
                row["item"],
                question_scope,
            ),
    }

    payload["calculation_id"] = (
        make_calculation_id(
            dataset_name,
            outcome,
            row["item"],
            meta["method_id"],
        )
    )

    payload["method_id"] = meta["method_id"]

    return payload


# PROCESS 18 – RUN FULL ANALYSIS

def analyze_dataset(
    dataset_path,
    wave=1,
    question_scope="overall",
):
    dataset_path = Path(
        dataset_path
    )

    data = prepare_analysis_data(
        dataset_path,
        wave=wave,
    )

    df = data["respondents"]

    health_report = (
        load_health_report(
            dataset_path
        )
    )

    blocked_outcomes = enforce_health(health_report)

    extra_controls, control_decisions = required_model_controls(health_report, df)

    fitted_outcomes = [o for o in SPECS if o not in blocked_outcomes]

    models = run_standard_models(
        df,
        extra_controls,
        fitted_outcomes,
    )

    results = collect_model_results(
        models
    )

    results = (
        apply_statistical_guardrails(
            results
        )
    )

    diagnostics = (
        run_robustness_diagnostics(
            df,
            models,
        )
    )

    results = classify_robustness(
        results,
        diagnostics,
    )

    # Outcomes blocked by the health check are never reported.
    results = results[
        ~results["outcome"].isin(blocked_outcomes)
    ].reset_index(drop=True)

    findings = (
        results[
            results["reportable"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    payloads = []

    dataset_name = (
        dataset_path.name
    )

    for _, row in findings.iterrows():

        payloads.append(
            build_finding_payload(
                row,
                health_report,
                dataset_name,
                question_scope,
            )
        )

    special_findings = special_patterns(df, models, dataset_name, extra_controls)
    attach_patterns(payloads, special_findings)

    return {
        "dataset_name":
            dataset_name,

        "status":
            "PARTIAL" if blocked_outcomes else "OK",

        "allowed_outcomes":
            fitted_outcomes,

        "blocked_outcomes":
            blocked_outcomes,

        "special_findings":
            special_findings,

        "health_report":
            health_report,

        "results":
            results,

        "findings":
            findings,

        "payloads":
            payloads,

        "sample_size":
            len(df),

        "adjustments": {
            "model_controls_added":
                extra_controls,

            "control_decisions":
                control_decisions,

            "blocked_outcomes":
                blocked_outcomes,
        },

        "methods": {
            outcome: describe(outcome, extra_controls.get(outcome))
            for outcome in fitted_outcomes
        },
    }

# PROCESS 18A – ANALYZE UPLOADED DATAFRAME

def analyze_dataframe(
    df,
    dataset_name="uploaded_dataset",
    health_report=None,
):
    """
    Run the analytical engine directly on an uploaded dataframe.

    Expected columns include:
    - Q01–Q12
    - workforce controls
    - turnover, absence, and performance outcomes
    """

    df = df.copy()

    # Health check on the rows as given (non-respondents included, so the
    # survey-response checks such as NR_LEAVERS can run). Callers with the
    # whole multi-wave upload should pass its report instead.
    if health_report is None:

        from .health_check import check_upload

        health_report = check_upload(
            df,
            dataset_name,
        )

    # -----------------------------------------
    # BASIC FILTERING
    # -----------------------------------------

    if "responded" in df.columns:

        df = df[
            df["responded"] == 1
        ].copy()

    if "wave" in df.columns:

        first_wave = (
            df["wave"]
            .dropna()
            .min()
        )

        df = df[
            df["wave"] == first_wave
        ].copy()

    # -----------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------

    required_columns = [
        "employee_id",
        "team_id",
        "department",
        "tenure_years",
        "role_level",
        "pay_band",
        "shift_type",
        "left_voluntarily_12m",
        "unplanned_absence_days_6m",
        "performance_rating",
        "prior_performance_rating",
    ]

    required_columns += [
        f"Q{i:02d}"
        for i in range(1, 13)
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Uploaded dataset is missing columns required "
            "for full outcome analysis: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------
    # DATA HEALTH: STOP ON ERRORS, ADJUST FOR BIASES
    # -----------------------------------------
    #
    # Best run on the whole upload (all waves) by the
    # caller; otherwise it runs here on these rows.
    # -----------------------------------------

    blocked_outcomes = enforce_health(
        health_report
    )

    extra_controls, control_decisions = required_model_controls(
        health_report,
        df,
    )

    # -----------------------------------------
    # STANDARD MODELS
    # -----------------------------------------

    fitted_outcomes = [o for o in SPECS if o not in blocked_outcomes]

    models = run_standard_models(
        df,
        extra_controls,
        fitted_outcomes,
    )

    results = collect_model_results(
        models
    )

    # -----------------------------------------
    # STATISTICAL GUARDRAILS
    # -----------------------------------------

    results = (
        apply_statistical_guardrails(
            results
        )
    )

    # -----------------------------------------
    # ROBUSTNESS DIAGNOSTICS
    # -----------------------------------------

    diagnostics = (
        run_robustness_diagnostics(
            df,
            models,
        )
    )

    results = classify_robustness(
        results,
        diagnostics,
    )

    # -----------------------------------------
    # REPORTABLE FINDINGS
    # -----------------------------------------

    # Outcomes blocked by the health check are never reported.
    results = results[
        ~results["outcome"].isin(blocked_outcomes)
    ].reset_index(drop=True)

    findings = (
        results[
            results["reportable"]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------
    # SAFE PAYLOADS
    # -----------------------------------------

    payloads = []

    for _, row in findings.iterrows():

        payload = build_finding_payload(
            row=row,
            health_report=health_report,
            dataset_name=dataset_name,
            question_scope="overall",
        )

        payloads.append(
            payload
        )

    special_findings = special_patterns(df, models, dataset_name, extra_controls)
    attach_patterns(payloads, special_findings)

    return {
        "dataset_name":
            dataset_name,

        "status":
            "PARTIAL" if blocked_outcomes else "OK",

        "allowed_outcomes":
            fitted_outcomes,

        "blocked_outcomes":
            blocked_outcomes,

        "special_findings":
            special_findings,

        "sample_size":
            len(df),

        "health_report":
            health_report,

        "results":
            results,

        "findings":
            findings,

        "payloads":
            payloads,

        "adjustments": {
            "model_controls_added":
                extra_controls,

            "control_decisions":
                control_decisions,

            "blocked_outcomes":
                blocked_outcomes,
        },

        "methods": {
            outcome: describe(outcome, extra_controls.get(outcome))
            for outcome in fitted_outcomes
        },
    }

# PROCESS 19 – FIND ONE RESULT

def get_finding(
    analysis,
    item,
    outcome,
):
    for payload in analysis["payloads"]:

        if (
            payload["item"] == item
            and payload["outcome"]
            == outcome
        ):
            return payload

    return None