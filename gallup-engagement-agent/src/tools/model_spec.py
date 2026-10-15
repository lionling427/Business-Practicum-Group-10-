"""Fixed statistical methods: one pre-defined specification per outcome.

Every outcome model in the app is fitted from the specification below, by
fit_outcome_model(). Nothing is chosen while looking at the data:

- the method, effect measure, controls, clustering and reporting rules are
  fixed here in advance;
- the data health check may switch on extra controls, but ONLY those listed in
  a spec's `allowed_adjustments` (pre-registered, each with its reason). Any
  other request is refused and recorded;
- SPEC_VERSION + a fingerprint of these specs is stamped on every finding
  (inside its calculation ID), so a number can always be traced to the exact
  method that produced it. Changing a spec changes the fingerprint.

The language model never computes a number: it only explains results produced
here and checked by the analytics pipeline.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from . import stats_tools as st


# PROCESS 2 – THE SPECIFICATIONS

SPEC_VERSION = "1.0"


@dataclass(frozen=True)
class ModelSpec:
    outcome: str
    label: str
    outcome_column: str
    method: str                      # family used to fit: logit | negbin | ols
    method_name: str                 # how it is described to people
    effect_type: str
    effect_unit: str
    numeric_controls: tuple[str, ...]
    categorical_controls: tuple[str, ...]
    derived_controls: tuple[str, ...] = ()     # built from the data by a fixed rule
    standardize: tuple[str, ...] = ()           # z-scored before fitting
    cluster: str = "team_id"
    allowed_adjustments: tuple[tuple[str, str], ...] = ()   # (column, reason)
    min_effect: str = ""
    control_labels: dict = field(default_factory=dict)


ITEMS = tuple(st.ITEMS)

SPECS: dict[str, ModelSpec] = {
    "turnover": ModelSpec(
        outcome="turnover",
        label="Voluntary turnover (left within 12 months)",
        outcome_column="left_voluntarily_12m",
        method="logit",
        method_name="Logistic Regression",
        effect_type="Odds Ratio",
        effect_unit="odds ratio per 1-point higher score",
        numeric_controls=("tenure_years",),
        categorical_controls=("department", "role_level", "pay_band"),
        allowed_adjustments=(
            ("shift_type", "turnover differs by shift"),
        ),
        min_effect="ratio below 0.91 or above 1.10",
        control_labels={"tenure_years": "tenure", "department": "department",
                        "role_level": "role level", "pay_band": "pay band",
                        "shift_type": "work shift"},
    ),
    "absence": ModelSpec(
        outcome="absence",
        label="Unplanned absence days (6 months)",
        outcome_column="unplanned_absence_days_6m",
        method="negbin",
        method_name="Negative Binomial Regression",
        effect_type="Incidence Rate Ratio",
        effect_unit="rate ratio per 1-point higher score",
        numeric_controls=("tenure_years",),
        categorical_controls=("department", "shift_type"),
        derived_controls=("q01_low",),
        allowed_adjustments=(
            ("role_level", "absence differs by role level"),
            ("pay_band", "absence differs by pay band"),
        ),
        min_effect="ratio below 0.91 or above 1.10",
        control_labels={"tenure_years": "tenure", "department": "department",
                        "shift_type": "work shift", "q01_low": "very low Q01 scores",
                        "role_level": "role level", "pay_band": "pay band"},
    ),
    "performance": ModelSpec(
        outcome="performance",
        label="Performance rating (1-5)",
        outcome_column="performance_rating",
        method="ols",
        method_name="Standardized Linear Regression",
        effect_type="Standardized Beta",
        effect_unit="standard deviations per 1 SD higher score",
        numeric_controls=("prior_performance_rating", "tenure_years"),
        categorical_controls=("department",),
        standardize=ITEMS + ("performance_rating", "prior_performance_rating"),
        allowed_adjustments=(
            ("role_level", "ratings differ by role level"),
            ("pay_band", "ratings differ by pay band"),
        ),
        min_effect="standardized beta of 0.05 or more",
        control_labels={"tenure_years": "tenure", "department": "department",
                        "prior_performance_rating": "prior performance",
                        "role_level": "role level", "pay_band": "pay band"},
    ),
}

# Rules shared by every outcome (fixed in advance, applied by analytics).
REPORTING_RULES = {
    "multiple_testing": "Benjamini-Hochberg, 5% false discovery rate across 36 tests",
    "standard_errors": "clustered by team",
    "robustness": "prior-performance, team-confounding and direction-flip diagnostics",
}


def spec_fingerprint() -> str:
    """Short hash of every spec + the reporting rules: changes when a method changes."""
    payload = json.dumps(
        {"version": SPEC_VERSION, "specs": {k: asdict(v) for k, v in SPECS.items()},
         "rules": REPORTING_RULES},
        sort_keys=True, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8].upper()


def method_id(outcome: str) -> str:
    """Method identity stamped into calculation IDs, e.g. 'Logistic Regression|v1.0-3F2A9C1E'."""
    return f"{SPECS[outcome].method_name}|v{SPEC_VERSION}-{spec_fingerprint()}"


# PROCESS 3 – PRE-REGISTERED ADJUSTMENTS

def resolve_adjustments(outcome: str, requested: list[str], df: pd.DataFrame) -> dict[str, list]:
    """Split requested extra controls into applied / already in the model /
    refused (not pre-registered) / unavailable (column missing)."""
    spec = SPECS[outcome]
    base = set(spec.numeric_controls) | set(spec.categorical_controls) | set(spec.derived_controls)
    allowed = {column for column, _ in spec.allowed_adjustments}
    out = {"applied": [], "already_in_model": [], "refused": [], "unavailable": []}
    for column in dict.fromkeys(requested):
        if column in base:
            out["already_in_model"].append(column)
        elif column not in allowed:
            out["refused"].append(column)
        elif column not in df.columns:
            out["unavailable"].append(column)
        else:
            out["applied"].append(column)
    return out


# PROCESS 4 – FIT FROM THE SPEC

def _derive(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "q01_low":                        # very low Q01 scores (below 3)
        return (df["Q01"] < 3).astype(float)
    raise KeyError(f"No rule for derived control {name}")


def design_matrix(df: pd.DataFrame, outcome: str, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """The model's X: intercept, 12 items, numeric + derived controls, dummies."""
    spec = SPECS[outcome]
    data = df.copy()
    for name in spec.derived_controls:
        data[name] = _derive(data, name)
    for column in spec.standardize:
        data[column] = st.zscore(data[column])
    if outcome == "absence":
        numeric = list(ITEMS) + list(spec.derived_controls) + list(spec.numeric_controls)
    else:
        numeric = list(ITEMS) + list(spec.numeric_controls) + list(spec.derived_controls)
    categorical = list(dict.fromkeys(list(spec.categorical_controls) + list(extra)))
    return st.design(data, numeric, categorical)


def fit_outcome_model(df: pd.DataFrame, outcome: str, extra: tuple[str, ...] = ()):
    """Fit one outcome's pre-defined model. `extra` must already be resolved
    through resolve_adjustments (pre-registered controls only)."""
    spec = SPECS[outcome]
    illegal = [c for c in extra if c not in {col for col, _ in spec.allowed_adjustments}]
    if illegal:
        raise ValueError(f"Controls not pre-registered for {outcome}: {illegal}")
    X = design_matrix(df, outcome, tuple(extra))
    y = df[spec.outcome_column]
    if spec.outcome_column in spec.standardize:
        y = pd.Series(st.zscore(y), index=df.index)
    cluster = df[spec.cluster] if spec.cluster in df.columns else None
    fit = {"logit": st.logit, "negbin": st.negbin, "ols": st.ols}[spec.method]
    return fit(y, X, cluster)


# PROCESS 5 – DESCRIPTIONS FOR PEOPLE AND THE AGENT

def describe(outcome: str, applied: list[str] | None = None) -> dict[str, Any]:
    spec = SPECS[outcome]
    labels = spec.control_labels
    controls = [labels.get(c, c) for c in
                list(spec.numeric_controls) + list(spec.categorical_controls) + list(spec.derived_controls)]
    added = [labels.get(c, c) for c in (applied or [])]
    return {
        "outcome": outcome,
        "label": spec.label,
        "method": spec.method_name,
        "effect_type": spec.effect_type,
        "effect_unit": spec.effect_unit,
        "controls": controls + added,
        "controls_added_by_health_check": added,
        "cluster": "team",
        "min_effect": spec.min_effect,
        "pre_registered_adjustments": [f"{labels.get(c, c)} ({why})" for c, why in spec.allowed_adjustments],
        "method_id": method_id(outcome),
    }
