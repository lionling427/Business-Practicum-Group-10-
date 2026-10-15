# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.waves import model_sample
from src.tools.longitudinal import analyze_change
from src.tools.recommendations import (
    TIER_ORDER,
    company_summary,
    productivity_levers,
    starting_points_chart,
    team_gap_chart,
    team_plan,
    team_plans,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"

    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    return survey.merge(
        individual,
        on=[c for c in survey.columns if c in individual.columns],
    )


@pytest.fixture(scope="module")
def evidence(data):

    return analyze_dataframe(model_sample(data), dataset_name="dev"), analyze_change(data)


@pytest.fixture(scope="module")
def levers(data, evidence):

    return productivity_levers(data, *evidence)


@pytest.fixture(scope="module")
def plans(data, levers):

    return team_plans(data, levers)


# PROCESS 3 – LEVERS

def test_levers_are_positive_and_verified(levers):

    assert not levers.empty
    assert (levers["points_per_point"] > 0).all()
    assert levers["points_per_point"].is_monotonic_decreasing
    assert set(levers["evidence"]) <= {"Strong", "Good"}


def test_strong_evidence_needs_both_analyses(levers):

    strong = levers[levers["evidence"] == "Strong"]

    assert strong["source"].str.contains(r"\+").all()


def test_no_levers_without_evidence(data):

    assert productivity_levers(data, None, None).empty


def test_cross_sectional_only_uses_checked_findings(data, evidence):

    analysis, _ = evidence
    levers = productivity_levers(data, analysis, None)
    findings = analysis["findings"]
    robust = findings[(findings["outcome"] == "performance")
                      & (findings["robustness_status"] == "ROBUST")]

    assert set(levers["item"]) == set(robust["item"])


# PROCESS 4 – TEAM PLANS

def test_each_team_starts_where_it_trails_the_company(data, plans):

    with_lever = plans[plans["start_with"].notna()]

    assert (with_lever["team_score"] < with_lever["company_score"]).all()
    assert (with_lever["estimated_gain"] > 0).all()
    assert set(plans["tier"]) <= set(TIER_ORDER)


def test_teams_without_gaps_keep_strengths(plans):

    keep = plans[plans["tier"] == "Keep strengths"]

    assert keep["start_with"].isna().all()
    assert (keep["estimated_gain"] == 0).all()


@pytest.mark.parametrize("position", [0, -1])
def test_team_plan_is_plain_language(plans, position):

    plan = team_plan(plans.iloc[position])

    assert len(plan["steps"]) == 3
    assert "guarantee" in plan["caution"]
    for text in [plan["headline"], plan["why"], plan["estimate"]]:
        assert "regression" not in text and "coefficient" not in text


def test_company_summary_counts_match(plans, levers):

    summary = company_summary(plans, levers)

    assert summary["starts"]["teams"].sum() == plans["start_with"].notna().sum()
    assert summary["headline"]


def test_recommendation_charts_compile(data, plans, levers):

    summary = company_summary(plans, levers)

    assert starting_points_chart(summary["starts"]).to_dict()
    assert team_gap_chart(plans.iloc[0], data, levers).to_dict()
