# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.waves import model_sample
from src.tools.financial import (
    Assumptions,
    current_costs,
    engagement_value,
    excess_cost_chart,
    explain_value,
    money,
    team_excess_cost,
    value_chart,
)
from src.tools.insights import analysis_frame, dollar_impact
from src.tools.recommendations import productivity_levers, team_plans


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
def analysis(data):

    return analyze_dataframe(model_sample(data), dataset_name="dev")


@pytest.fixture(scope="module")
def plans(data, analysis):

    return team_plans(data, productivity_levers(data, analysis))


# PROCESS 3 – HELPERS

@pytest.mark.parametrize("value, text", [
    (1_234_567, "$1.2M"), (45_300, "$45K"), (900, "$900"), (-2_500_000, "-$2.5M"),
])
def test_money_formatting(value, text):

    assert money(value) == text


def test_default_day_cost_is_salary_over_workdays():

    assert Assumptions(avg_salary=52_000).day_cost == pytest.approx(200)


# PROCESS 4 – COSTS TODAY

def test_current_costs_follow_assumptions(data):

    a = Assumptions(avg_salary=50_000, replacement_cost_pct=1.0)
    costs = current_costs(data, a)
    frame = analysis_frame(data)

    expected_leavers = frame["left_voluntarily_12m"].mean() * len(frame)
    assert costs["leavers"] == pytest.approx(expected_leavers)
    assert costs["turnover_cost"] == pytest.approx(expected_leavers * 50_000)


def test_headcount_scales_costs_linearly(data):

    sample = len(analysis_frame(data))
    small = current_costs(data, Assumptions(headcount=sample))
    double = current_costs(data, Assumptions(headcount=2 * sample))

    assert double["total_cost"] == pytest.approx(2 * small["total_cost"])


# PROCESS 5 – SCENARIOS

def test_turnover_scenario_matches_dollar_impact(data, analysis):

    a = Assumptions()
    value = engagement_value(data, analysis, a)
    reference = dollar_impact(analysis, data, a.avg_salary, a.replacement_cost_pct)

    turnover = value["scenarios"].set_index("source").loc["Turnover", "value"]
    assert turnover == pytest.approx(reference["savings"])


def test_productivity_only_counted_with_a_point_value(data, analysis, plans):

    without = engagement_value(data, analysis, Assumptions(), plans)
    with_value = engagement_value(
        data, analysis, Assumptions(value_per_rating_point=1_000), plans
    )

    assert "Productivity" not in set(without["scenarios"]["source"])
    assert "Productivity" in set(with_value["scenarios"]["source"])
    assert with_value["total"] > without["total"]


def test_no_analysis_means_no_scenarios(data):

    value = engagement_value(data, None, Assumptions())

    assert value["scenarios"].empty and value["total"] == 0


# PROCESS 6 – WHERE THE MONEY IS

def test_excess_cost_sums_to_zero_at_company_rate(data):

    teams = team_excess_cost(data, Assumptions())
    frame = analysis_frame(data)
    kept = frame[frame["team_id"].isin(teams["team_id"])]

    # Extra costs are relative to the company rate, so they roughly cancel out.
    assert abs(teams["excess_cost"].sum()) < 0.05 * current_costs(kept, Assumptions(
        headcount=len(frame)))["total_cost"]
    assert teams["excess_cost"].is_monotonic_decreasing


def test_story_and_charts(data, analysis):

    a = Assumptions()
    costs = current_costs(data, a)
    value = engagement_value(data, analysis, a)
    teams = team_excess_cost(data, a)

    story = explain_value(costs, value, teams, a)

    assert "a year" in story["headline"]
    assert value_chart(value).to_dict()
    assert excess_cost_chart(teams).to_dict()


# PROCESS 7 – PAGE HELPERS

from src.tools.financial import assumptions_line, explain_where_money  # noqa: E402


def test_assumptions_line_shows_every_input(data):

    line = assumptions_line(Assumptions(value_per_rating_point=2_000, headcount=5_000), 4_000)

    for part in ["$60,000 salary", "50% of salary", "per absence day", "5,000 people",
                 "per performance point"]:
        assert part in line


def test_where_money_story_names_departments(data):

    story = explain_where_money(team_excess_cost(data, Assumptions()), Assumptions())

    assert "teams cost" in story["headline"]
    assert any("Most of it sits in" in p for p in story["points"])
