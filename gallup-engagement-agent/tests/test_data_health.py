# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.data_health import (
    column_guide,
    headline_numbers,
    health_checks,
    overall_status,
    wave_coverage,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


def _status(checks, name):
    return next(c["status"] for c in checks if c["check"] == name)


# PROCESS 3 – CHECKS

def test_clean_panel_passes_duplicates_check(data):

    checks = health_checks(data)

    assert _status(checks, "One answer per person per survey") == "pass"
    assert "more than one survey" in next(
        c["detail"] for c in checks if c["check"] == "One answer per person per survey")


def test_same_survey_duplicates_fail(data):

    broken = pd.concat([data, data.head(3)])

    assert _status(health_checks(broken), "One answer per person per survey") == "fail"
    assert overall_status(health_checks(broken))[0] == "fail"


def test_out_of_range_answers_fail(data):

    broken = data.copy()
    broken.loc[broken.index[:4], "Q01"] = 9

    checks = health_checks(broken)

    assert _status(checks, "Answers on the 1–5 scale") == "fail"


def test_missing_columns_fail(data):

    checks = health_checks(data.drop(columns=["Q05"]))

    assert _status(checks, "Required columns") == "fail"


def test_no_outcomes_warns(data):

    survey_only = data.drop(columns=["left_voluntarily_12m", "unplanned_absence_days_6m",
                                     "performance_rating"])

    assert _status(health_checks(survey_only), "Business outcomes") == "warn"


# PROCESS 4 – TILES, COVERAGE, COLUMN GUIDE

def test_headline_numbers(data):

    tiles = dict((label, value) for label, value, _ in headline_numbers(data))

    assert tiles["People"] == f"{data['employee_id'].nunique():,}"
    assert tiles["Surveys"] == "2"


def test_wave_coverage_one_row_per_wave(data):

    coverage = wave_coverage(data)

    assert list(coverage["wave"]) == [1, 2]
    assert coverage["responses"].sum() == len(data)


def test_column_guide_lists_every_column(data):

    guide = column_guide(data.assign(extra_note="x"))
    listed = ", ".join(row["columns"] for row in guide)

    assert all(column in listed for column in data.columns)
    assert guide[-1]["role"] == "Not used by the app"
