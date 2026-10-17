# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.audit import audit_sample
from src.tools.financial import Assumptions
from src.tools.health_check import check_upload
from src.tools.scenario import (
    contributions,
    explain_scenario,
    improved_scores,
    path_chart,
    quarterly_path,
    ramp_share,
    simulate,
    verified_links,
)
from src.tools.waves import model_sample


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


@pytest.fixture(scope="module")
def analysis(data):

    return analyze_dataframe(model_sample(data), dataset_name="dev",
                             health_report=check_upload(data, "dev"))


@pytest.fixture(scope="module")
def a(data):

    return Assumptions(headcount=len(audit_sample(data)))


@pytest.fixture(scope="module")
def lever(analysis):
    """A question with a verified turnover link."""
    return verified_links(analysis).query("outcome == 'turnover'")["item"].iloc[0]


# PROCESS 3 – THE SIMULATION BEHAVES SENSIBLY

def test_no_lift_means_no_change(data, analysis, a, lever):

    result = simulate(data, analysis, {lever: 0.0}, a)

    assert result["totals"]["estimate"]["value"] == 0


def test_bigger_lift_means_bigger_effect(data, analysis, a, lever):

    small = simulate(data, analysis, {lever: 0.2}, a)["totals"]["estimate"]["leavers"]
    big = simulate(data, analysis, {lever: 0.6}, a)["totals"]["estimate"]["leavers"]

    assert 0 < small < big


def test_range_brackets_the_estimate(data, analysis, a, lever):

    result = simulate(data, analysis, {lever: 0.5}, a)
    low, high = result["value_range"]

    assert low < result["totals"]["estimate"]["value"] < high


def test_unverified_question_counts_as_zero(data, analysis, a):

    verified = set(verified_links(analysis)["item"])
    unverified = next(i for i in [f"Q{n:02d}" for n in range(1, 13)] if i not in verified)

    result = simulate(data, analysis, {unverified: 1.0}, a)

    assert result["totals"]["estimate"]["value"] == 0
    assert "no verified link" in explain_scenario(result, a)["headline"]


def test_scores_never_go_above_five():

    scores = pd.Series([1.0, 3.0, 4.8, 5.0])

    assert improved_scores(scores, 0.5, "everyone").tolist() == [1.5, 3.5, 5.0, 5.0]
    assert improved_scores(scores, 0.5, "low").tolist() == [1.5, 3.5, 4.8, 5.0]


def test_one_department_is_part_of_the_company(data, analysis, a, lever):

    company = simulate(data, analysis, {lever: 0.3}, a)["totals"]["estimate"]["leavers"]
    dept = data["department"].mode().iat[0]
    part = simulate(data, analysis, {lever: 0.3}, a, department=dept)["totals"]["estimate"]["leavers"]

    assert 0 < part < company


# PROCESS 4 – OVER TIME

def test_ramp_builds_up_linearly():

    assert ramp_share(3, 6) == 0.5
    assert ramp_share(9, 6) == 1.0
    assert ramp_share(1, 0) == 1.0


def test_without_build_up_a_year_equals_the_yearly_value(data, analysis, a, lever):

    result = simulate(data, analysis, {lever: 0.3}, a, ramp_months=0, quarters=4)

    assert result["first_year_value"] == pytest.approx(result["totals"]["estimate"]["value"])


def test_build_up_lowers_the_first_year(data, analysis, a, lever):

    slow = simulate(data, analysis, {lever: 0.3}, a, ramp_months=12)

    assert slow["first_year_value"] < slow["totals"]["estimate"]["value"]
    assert slow["path"]["cumulative_value"].is_monotonic_increasing
    assert path_chart(slow["path"]).to_dict()


def test_quarterly_path_keeps_the_range_ordered():

    totals = {"estimate": {"value": 400.0, "leavers": 8.0}}
    path = quarterly_path(totals, (300.0, 500.0), 0, 8)

    assert len(path) == 8
    assert (path["cumulative_low"] <= path["cumulative_value"]).all()
    assert (path["cumulative_value"] <= path["cumulative_high"]).all()
    assert path["cumulative_value"].iloc[-1] == pytest.approx(800.0)


# PROCESS 5 – EXPLANATION AND PARTS

def test_explanation_names_the_change_and_range(data, analysis, a, lever):

    story = explain_scenario(simulate(data, analysis, {lever: 0.3}, a), a)

    assert "fewer leavers" in story["headline"]
    assert any("likely between" in p for p in story["points"])


def test_each_question_on_its_own(data, analysis, a, lever):

    parts = contributions(data, analysis, {lever: 0.3, "Q09": 0.3}, a)

    assert len(parts) == 2
    assert parts.iloc[0]["item"] == lever
