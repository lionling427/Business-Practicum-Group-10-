# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.agent_context import build_context, route_sections
from src.tools.analytics import analyze_dataframe
from src.tools.financial import Assumptions, engagement_value
from src.tools.health_check import check_upload
from src.tools.insights import analysis_frame, dollar_impact
from src.tools.priorities import (
    PRIORITY_ORDER,
    explain_priorities,
    factor_value,
    priority_chart,
    priority_matrix,
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
def sample(data):

    return len(analysis_frame(data))


# PROCESS 3 – AVOIDED TURNOVER AND COST

def test_every_verified_factor_is_valued(data, analysis, sample):

    table = factor_value(data, analysis, Assumptions(headcount=sample))
    robust = analysis["findings"].query("robustness_status == 'ROBUST'")

    assert set(table["item"]) <= set(robust["item"])
    assert not table.empty
    assert (table["yearly_value"] >= 0).all()
    assert table["rank"].tolist() == list(range(1, len(table) + 1))
    assert table["yearly_value"].is_monotonic_decreasing


def test_matches_the_financials_page_scenario(data, analysis, sample):
    """The top turnover factor's numbers equal the existing dollar_impact scenario."""
    a = Assumptions(headcount=sample)
    table = factor_value(data, analysis, a).set_index("item")
    impact = dollar_impact(analysis, data, a.avg_salary, a.replacement_cost_pct)

    row = table.loc[impact["item"]]
    assert row["avoided_leavers"] == pytest.approx(impact["fewer_leavers"])
    assert row["turnover_value"] == pytest.approx(impact["savings"])

    scenarios = engagement_value(data, analysis, a)["scenarios"].set_index("source")
    assert row["turnover_value"] == pytest.approx(scenarios.loc["Turnover", "value"])


def test_dollars_follow_the_assumptions(data, analysis, sample):

    base = factor_value(data, analysis, Assumptions(headcount=sample))
    double = factor_value(data, analysis, Assumptions(headcount=sample * 2, avg_salary=120_000))

    merged = base.merge(double, on="item", suffixes=("", "_x"))
    assert list(merged["avoided_leavers_x"]) == pytest.approx(list(merged["avoided_leavers"] * 2))
    assert list(merged["turnover_value_x"]) == pytest.approx(list(merged["turnover_value"] * 4))


def test_performance_is_priced_only_with_a_point_value(data, analysis, sample):

    free = factor_value(data, analysis, Assumptions(headcount=sample))
    priced = factor_value(data, analysis, Assumptions(headcount=sample,
                                                      value_per_rating_point=1000))

    perf = free[free["rating_points"] > 0]
    if perf.empty:
        pytest.skip("No verified performance factor in this data")
    assert (perf["performance_value"] == 0).all()
    assert (priced.loc[priced["rating_points"] > 0, "performance_value"] > 0).all()


# PROCESS 4 – PRIORITY BY IMPACT AND CURRENT SCORE

def test_priority_labels_follow_impact_and_score(data, analysis, sample):

    table = factor_value(data, analysis, Assumptions(headcount=sample))
    benchmark = table.attrs["agree_benchmark"]
    cut = 0.5 * table.loc[table["yearly_value"] > 0, "value_per_person"].max()

    assert set(table["priority"]) <= set(PRIORITY_ORDER)
    for _, r in table.iterrows():
        high = r["yearly_value"] > 0 and r["value_per_person"] >= cut
        low = r["agree_pct"] < benchmark
        expected = ("Fix first" if high and low else "Protect" if high
                    else "Improve next" if low else "Maintain")
        assert r["priority"] == expected


def test_no_analysis_means_no_values(data, sample):

    assert factor_value(data, None, Assumptions(headcount=sample)).empty
    assert "none is valued" in explain_priorities(
        factor_value(data, None, Assumptions()), Assumptions())["headline"]


# PROCESS 5 – STORY, CHARTS AND THE AGENT

def test_story_escapes_money_and_warns_against_adding(data, analysis, sample):

    story = explain_priorities(factor_value(data, analysis, Assumptions(headcount=sample)),
                               Assumptions(headcount=sample))

    assert "\\$" in story["headline"]
    assert any("don't add them up" in p for p in story["points"])


def test_charts_build(data, analysis, sample):

    table = factor_value(data, analysis, Assumptions(headcount=sample))

    assert priority_chart(table).to_dict()
    assert priority_matrix(table).to_dict()


def test_agent_gets_the_ranking(data, analysis, sample):

    assert "financials" in route_sections("Where should we invest to avoid turnover?")

    context = build_context(data, analysis, sections=["financials"],
                            assumptions=Assumptions(headcount=sample))
    ranking = context["financials"]["factors_ranked_by_value"]["ranking"]

    assert ranking[0].startswith("#1 ")
    assert "never add" in context["financials"]["factors_ranked_by_value"]["how_ranked"]
