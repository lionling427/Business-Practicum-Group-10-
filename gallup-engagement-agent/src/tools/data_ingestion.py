# PROCESS 1 – IMPORTS

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


# PROCESS 2 – SCHEMA CONFIGURATION

Q12_ITEMS = [
    f"Q{i:02d}"
    for i in range(1, 13)
]


CORE_COLUMNS = [
    "employee_id",
    *Q12_ITEMS,
]


PROFILE_COLUMNS = [
    "department",
    "team_id",
    "tenure_years",
    "role_level",
    "pay_band",
    "shift_type",
]


OUTCOME_COLUMNS = {
    "turnover": "left_voluntarily_12m",
    "absence": "unplanned_absence_days_6m",
    "performance": "performance_rating",
}


OPTIONAL_COLUMNS = [
    "prior_performance_rating",
    "responded",
    "wave",
    "survey_date",
    "turnover_window_end",
    "absence_window_end",
    "performance_review_date",
]


# PROCESS 3 – READ UPLOADED FILE

def read_uploaded_file(
    uploaded_file,
) -> pd.DataFrame:

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):

        return pd.read_csv(
            uploaded_file
        )

    if filename.endswith(
        (".xlsx", ".xls")
    ):

        return pd.read_excel(
            uploaded_file
        )

    raise ValueError(
        "Unsupported file type. "
        "Please upload CSV or Excel."
    )


# PROCESS 4 – NORMALIZE COLUMN NAMES

def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    rename_map = {}

    for column in df.columns:

        cleaned = str(
            column
        ).strip()

        upper = cleaned.upper()

        if upper in Q12_ITEMS:
            rename_map[column] = upper

        else:
            rename_map[column] = (
                cleaned
                .strip()
                .lower()
                .replace(" ", "_")
            )

    return df.rename(
        columns=rename_map
    )


# PROCESS 5 – DETECT DATA CAPABILITIES

def detect_capabilities(
    df: pd.DataFrame,
) -> dict[str, Any]:

    columns = set(
        df.columns
    )

    has_core = all(
        column in columns
        for column in CORE_COLUMNS
    )

    profile_available = [
        column
        for column in PROFILE_COLUMNS
        if column in columns
    ]

    outcomes_available = [
        outcome
        for outcome, column
        in OUTCOME_COLUMNS.items()
        if column in columns
    ]

    if not has_core:

        level = "INVALID"

    elif len(
        outcomes_available
    ) > 0:

        level = "OUTCOME_ANALYSIS"

    elif len(
        profile_available
    ) > 0:

        level = "SEGMENT_ANALYSIS"

    else:

        level = "SURVEY_ONLY"

    return {
        "level":
            level,

        "has_core":
            has_core,

        "profile_columns":
            profile_available,

        "outcomes":
            outcomes_available,
    }


# PROCESS 6 – VALIDATE Q12 VALUES

def validate_q12_values(
    df: pd.DataFrame,
) -> dict[str, Any]:

    problems = []

    available_items = [
        item
        for item in Q12_ITEMS
        if item in df.columns
    ]

    for item in available_items:

        numeric = pd.to_numeric(
            df[item],
            errors="coerce",
        )

        invalid = (
            numeric.notna()
            & (
                (numeric < 1)
                | (numeric > 5)
            )
        )

        invalid_count = int(
            invalid.sum()
        )

        if invalid_count > 0:

            problems.append({
                "column":
                    item,

                "issue":
                    "Values outside the expected 1–5 range.",

                "count":
                    invalid_count,
            })

    return {
        "valid":
            len(problems) == 0,

        "problems":
            problems,
    }


# PROCESS 7 – VALIDATE DATASET

def validate_uploaded_data(
    df: pd.DataFrame,
) -> dict[str, Any]:

    capabilities = (
        detect_capabilities(
            df
        )
    )

    missing_core = [
        column
        for column in CORE_COLUMNS
        if column not in df.columns
    ]

    q12_validation = (
        validate_q12_values(
            df
        )
    )

    duplicate_employee_ids = 0

    if "employee_id" in df.columns:

        duplicate_employee_ids = int(
            df["employee_id"]
            .duplicated()
            .sum()
        )

    missing_q12 = {}

    for item in Q12_ITEMS:

        if item in df.columns:

            missing_q12[item] = int(
                df[item]
                .isna()
                .sum()
            )

    return {
        "rows":
            len(df),

        "columns":
            len(df.columns),

        "missing_core_columns":
            missing_core,

        "duplicate_employee_ids":
            duplicate_employee_ids,

        "q12_validation":
            q12_validation,

        "missing_q12":
            missing_q12,

        "capabilities":
            capabilities,

        "is_valid":
            (
                len(missing_core) == 0
                and q12_validation[
                    "valid"
                ]
            ),
    }


# PROCESS 8 – OVERALL SURVEY SUMMARY

def build_survey_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:

    available_items = [
        item
        for item in Q12_ITEMS
        if item in df.columns
    ]

    if not available_items:

        return {
            "employee_count": len(df),
            "overall_mean": None,
            "item_summary": pd.DataFrame(),
        }

    survey = (
        df[available_items]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    item_summary = pd.DataFrame({
        "item":
            available_items,

        "mean_score": [
            survey[item].mean()
            for item in available_items
        ],

        "median_score": [
            survey[item].median()
            for item in available_items
        ],

        "agree_share": [
            (
                survey[item]
                >= 4
            ).mean()
            for item in available_items
        ],

        "response_count": [
            survey[item]
            .notna()
            .sum()
            for item in available_items
        ],
    })

    overall_mean = float(
        survey
        .mean(axis=1)
        .mean()
    )

    return {
        "employee_count":
            len(df),

        "overall_mean":
            overall_mean,

        "item_summary":
            item_summary,
    }


# PROCESS 9 – SEGMENT SUMMARY

def build_segment_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    if group_column not in df.columns:

        raise ValueError(
            f"{group_column} is not available."
        )

    survey = df.copy()

    survey["engagement_mean"] = (
        survey[Q12_ITEMS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .mean(axis=1)
    )

    summary = (
        survey
        .groupby(
            group_column,
            dropna=False,
        )
        .agg(
            employee_count=(
                "employee_id",
                "nunique",
            ),
            engagement_mean=(
                "engagement_mean",
                "mean",
            ),
        )
        .reset_index()
    )

    # Governance:
    # suppress groups with fewer than 5 employees.
    summary = summary[
        summary[
            "employee_count"
        ] >= 5
    ].copy()

    return summary