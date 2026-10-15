# PROCESS 1 – IMPORTS

import pandas as pd

from src.tools.data_ingestion import (
    build_survey_summary,
    detect_capabilities,
    normalize_columns,
    validate_uploaded_data,
)


# PROCESS 2 – SAMPLE SURVEY DATA

def make_sample_data():

    data = {
        "employee_id":
            range(1, 11),
    }

    for i in range(
        1,
        13,
    ):

        data[
            f"Q{i:02d}"
        ] = [4] * 10

    return pd.DataFrame(
        data
    )


# PROCESS 3 – SURVEY-ONLY CAPABILITY

def test_survey_only_dataset():

    df = make_sample_data()

    result = (
        detect_capabilities(
            df
        )
    )

    assert (
        result["level"]
        == "SURVEY_ONLY"
    )


# PROCESS 4 – OUTCOME CAPABILITY

def test_outcome_dataset():

    df = make_sample_data()

    df[
        "left_voluntarily_12m"
    ] = 0

    result = (
        detect_capabilities(
            df
        )
    )

    assert (
        result["level"]
        == "OUTCOME_ANALYSIS"
    )

    assert (
        "turnover"
        in result["outcomes"]
    )


# PROCESS 5 – VALID Q12 RANGE

def test_q12_validation():

    df = make_sample_data()

    result = (
        validate_uploaded_data(
            df
        )
    )

    assert (
        result["is_valid"]
        is True
    )


# PROCESS 6 – INVALID Q12 RANGE

def test_invalid_q12_value():

    df = make_sample_data()

    df.loc[
        0,
        "Q04",
    ] = 8

    result = (
        validate_uploaded_data(
            df
        )
    )

    assert (
        result["is_valid"]
        is False
    )


# PROCESS 7 – SURVEY SUMMARY

def test_survey_summary():

    df = make_sample_data()

    summary = (
        build_survey_summary(
            df
        )
    )

    assert (
        summary[
            "employee_count"
        ]
        == 10
    )

    assert (
        summary[
            "overall_mean"
        ]
        == 4.0
    )


# PROCESS 8 – COLUMN NORMALIZATION

def test_column_normalization():

    df = pd.DataFrame({
        " Employee ID ":
            [1, 2],

        "q01":
            [4, 5],

        "Department":
            ["HR", "Sales"],
    })

    normalized = (
        normalize_columns(
            df
        )
    )

    assert (
        "employee_id"
        in normalized.columns
    )

    assert (
        "Q01"
        in normalized.columns
    )

    assert (
        "department"
        in normalized.columns
    )