"""Plain-language health checks for an uploaded survey file.

health_checks(df)   one row per check: status (pass / warn / fail) and what it means
wave_coverage(df)   responses, people, teams and outcome coverage per survey wave
column_guide(df)    which uploaded columns the app recognized, and how it uses them
headline_numbers(df) people, surveys, teams, departments for the page tiles
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_ingestion import OPTIONAL_COLUMNS, OUTCOME_COLUMNS, Q12_ITEMS
from .waves import model_wave, wave_label


# PROCESS 2 – CONFIGURATION

MIN_ITEMS_ANSWERED = 9            # same rule as the engagement calculations
MIN_TEAM_SIZE = 5                 # privacy rule used across the app
MIN_LEAVERS = 50                  # fewer leavers than this makes turnover models fragile

FRIENDLY = {column: name for name, column in OUTCOME_COLUMNS.items()}

STATUS_ICONS = {"pass": ":green[:material/check_circle:]",
                "warn": ":orange[:material/warning:]",
                "fail": ":red[:material/cancel:]"}

COLUMN_ROLES = [
    ("Survey answers", Q12_ITEMS),
    ("People and teams", ["employee_id", "team_id"]),
    ("Groups for comparison", ["department", "role_level", "pay_band", "shift_type", "tenure_years"]),
    ("Business outcomes", list(OUTCOME_COLUMNS.values()) + ["prior_performance_rating"]),
    ("Time", ["wave", "survey_date", "survey_year"]),
    ("Other recognized", [c for c in OPTIONAL_COLUMNS
                          if c not in ("prior_performance_rating", "wave", "survey_date")]),
]


# PROCESS 3 – HELPERS

def _items(df: pd.DataFrame) -> list[str]:
    return [i for i in Q12_ITEMS if i in df.columns]


def _respondents(df: pd.DataFrame) -> pd.DataFrame:
    if "responded" in df.columns:
        return df[pd.to_numeric(df["responded"], errors="coerce") == 1]
    return df


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


# PROCESS 4 – HEADLINE NUMBERS

def headline_numbers(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    tiles = [("Survey responses", f"{len(df):,}", "Rows in the uploaded file.")]
    if "employee_id" in df.columns:
        tiles.append(("People", f"{df['employee_id'].nunique():,}",
                      "Different employees across all surveys."))
    if "wave" in df.columns:
        tiles.append(("Surveys", f"{df['wave'].nunique():,}", "Survey waves in the file."))
    if "team_id" in df.columns:
        tiles.append(("Teams", f"{df['team_id'].nunique():,}", "Different teams in the file."))
    if "department" in df.columns:
        tiles.append(("Departments", f"{df['department'].nunique():,}", ""))
    return tiles


# PROCESS 5 – HEALTH CHECKS

def health_checks(df: pd.DataFrame) -> list[dict[str, str]]:
    checks = []
    items = _items(df)

    # 1. Required columns
    missing = [c for c in ["employee_id", *Q12_ITEMS] if c not in df.columns]
    checks.append(_check(
        "Required columns",
        "fail" if missing else "pass",
        ("Missing: " + ", ".join(missing) + ". The app needs employee_id and Q01–Q12.")
        if missing else "employee_id and all 12 survey questions are present."))

    # 2. One response per person per survey
    if "employee_id" in df.columns:
        keys = ["employee_id"] + (["wave"] if "wave" in df.columns else [])
        dup = int(df.duplicated(keys).sum())
        repeat_people = (int(df.groupby("employee_id")["wave"].nunique().gt(1).sum())
                         if "wave" in df.columns else 0)
        detail = (f"{dup:,} people answered more than once in the same survey; only one "
                  "answer per person per survey should be kept." if dup else
                  "Nobody answered twice in the same survey.")
        if repeat_people:
            detail += (f" {repeat_people:,} people took part in more than one survey, which "
                       "lets the app follow change over time.")
        checks.append(_check("One answer per person per survey", "fail" if dup else "pass", detail))

    if items:
        answers = df[items].apply(pd.to_numeric, errors="coerce")

        # 3. Answers in the 1–5 range
        out_of_range = int(((answers < 1) | (answers > 5)).sum().sum())
        checks.append(_check(
            "Answers on the 1–5 scale",
            "fail" if out_of_range else "pass",
            f"{out_of_range:,} answers fall outside 1–5 and should be fixed."
            if out_of_range else "Every answer is between 1 and 5."))

        # 4. Completeness
        respondents = _respondents(df)
        answered = respondents[items].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
        too_few = int((answered < MIN_ITEMS_ANSWERED).sum())
        blank_share = 1 - answers.notna().mean().mean()
        checks.append(_check(
            "Complete surveys",
            "warn" if too_few / max(len(respondents), 1) > 0.05 else "pass",
            "Every survey has all 12 answers." if blank_share == 0 else
            f"{blank_share:.1%} of answers are blank. {too_few:,} surveys have fewer than "
            f"{MIN_ITEMS_ANSWERED} of 12 answers and are left out of engagement scores."))

    # 5. Outcomes for the survey the models use
    outcomes = [c for c in OUTCOME_COLUMNS.values() if c in df.columns]
    if outcomes:
        wave = model_wave(df)
        sample = df if wave is None else df[df["wave"] == wave]
        coverage = sample[outcomes].notna().mean()
        low = coverage[coverage < 0.9]
        checks.append(_check(
            "Business outcomes",
            "warn" if len(low) else "pass",
            (f"In {wave_label(wave)} (the survey the models use), outcomes are filled in for "
             + ", ".join(f"{FRIENDLY.get(c, c)} {v:.0%}" for c, v in coverage.items()) + ".")))
    else:
        checks.append(_check("Business outcomes", "warn",
                             "No turnover, absence or performance columns, so only engagement "
                             "is analysed (no Outcomes, Predictions or Financials)."))

    # 6. Enough leavers to model turnover
    turnover = OUTCOME_COLUMNS["turnover"]
    if turnover in df.columns:
        wave = model_wave(df)
        sample = df if wave is None else df[df["wave"] == wave]
        leavers = int(pd.to_numeric(sample[turnover], errors="coerce").sum())
        checks.append(_check(
            "Enough leavers to model turnover",
            "pass" if leavers >= MIN_LEAVERS else "warn",
            f"{leavers:,} voluntary leavers in {wave_label(wave)}"
            + ("." if leavers >= MIN_LEAVERS else
               f"; fewer than {MIN_LEAVERS} makes turnover results fragile.")))

    # 7. Teams large enough to report
    if "team_id" in df.columns:
        wave = df["wave"].max() if "wave" in df.columns else None
        latest = df if wave is None else df[df["wave"] == wave]
        sizes = latest.groupby("team_id").size()
        small = int((sizes < MIN_TEAM_SIZE).sum())
        checks.append(_check(
            "Teams large enough to report",
            "warn" if small / max(len(sizes), 1) > 0.25 else "pass",
            f"{len(sizes) - small:,} of {len(sizes):,} teams have {MIN_TEAM_SIZE}+ people in "
            f"the latest survey. {small:,} smaller teams are hidden to protect anonymity."))

    return checks


def overall_status(checks: list[dict[str, str]]) -> tuple[str, str]:
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        n = sum(c["status"] == "fail" for c in checks)
        return "fail", f"{n} problem{'s' if n > 1 else ''} to fix before relying on the results."
    if "warn" in statuses:
        n = sum(c["status"] == "warn" for c in checks)
        return "warn", f"Usable, with {n} point{'s' if n > 1 else ''} to keep in mind."
    return "pass", "Your data passed every check."


# PROCESS 6 – COVERAGE BY SURVEY

def wave_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if "wave" not in df.columns:
        return pd.DataFrame()
    agg: dict[str, Any] = {"responses": ("wave", "size")}
    if "employee_id" in df.columns:
        agg["people"] = ("employee_id", "nunique")
    if "team_id" in df.columns:
        agg["teams"] = ("team_id", "nunique")
    if "survey_date" in df.columns:
        agg["survey_date"] = ("survey_date", "min")
    for name, column in OUTCOME_COLUMNS.items():
        if column in df.columns:
            agg[f"{name}_known"] = (column, lambda s: s.notna().mean() * 100)
    return df.groupby("wave").agg(**agg).reset_index()


# PROCESS 7 – COLUMN GUIDE

def column_guide(df: pd.DataFrame) -> list[dict[str, str]]:
    used: set[str] = set()
    guide = []
    for role, columns in COLUMN_ROLES:
        found = [c for c in columns if c in df.columns]
        used.update(found)
        if found:
            guide.append({"role": role, "columns": ", ".join(found)})
    other = [c for c in df.columns if c not in used]
    if other:
        guide.append({"role": "Not used by the app", "columns": ", ".join(other)})
    return guide
