# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.action_story import build_action_story, story_markdown, where_to_start
from src.tools.analytics import analyze_dataframe
from src.tools.financial import Assumptions, current_costs, money
from src.tools.health_check import check_upload
from src.tools.insights import analysis_frame
from src.tools.prediction import employee_risk_by_group
from src.tools.priorities import factor_value
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
def story(data, analysis):

    return build_action_story(data, analysis)


# PROCESS 3 – ONE STORY, SAME NUMBERS AS THE OTHER PAGES

def test_story_has_every_chapter(story):

    assert set(story["chapters"]) == {"predict", "act", "money"}
    for chapter in story["chapters"].values():
        assert chapter["lead"] and chapter["available"]
    assert story["headline"].startswith("If nothing changes")


def test_predictions_match_the_predictions_page(data, story):

    leavers = employee_risk_by_group(data, "department")["table"]["expected_leavers"].sum()

    assert f"{leavers:,.0f} people" in story["chapters"]["predict"]["points"][0]


def test_money_matches_the_financials_page(data, analysis, story):

    a = Assumptions(headcount=len(analysis_frame(data)))
    top = factor_value(data, analysis, a).iloc[0]

    text = " ".join(story["chapters"]["money"]["points"])
    assert money(current_costs(data, a)["total_cost"]) in text
    assert money(top["yearly_value"]) in text
    assert top["label"] in story["headline"]


def test_different_levers_are_reconciled(story):
    """The company-wide lever and the team performance lever are explained, not contradictory."""
    act = " ".join(story["chapters"]["act"]["points"])

    assert "Company-wide" in act and "Team by team" in act
    if "a different lever" in act:
        assert "so do both" in act


# PROCESS 4 – WHERE TO START

def test_start_here_teams_are_on_two_or_more_lists(story):

    start = story["start_here"]

    assert not start.empty and len(start) <= 10
    assert (start["lists"] >= 2).all()
    assert start["lists"].is_monotonic_decreasing


def test_where_to_start_without_plans_or_costs():

    facts = {"high_teams": {"turnover": {"A", "B"}, "absence": {"B", "C"}}}

    start = where_to_start(facts, None, pd.DataFrame(), Assumptions())

    assert start["team_id"].tolist() == ["B"]


# PROCESS 5 – READABLE DOCUMENT

def test_markdown_reads_top_to_bottom(story):

    text = story_markdown(story)

    order = [text.index(h) for h in ["## 1. What is likely", "## 2. What we can do",
                                     "## 3. What it is worth", "## 4. Where to start",
                                     "## Next steps"]]
    assert order == sorted(order)
    assert "**" not in text.split("## 1.")[1]           # no stray bold markers in points


def test_story_without_analysis_still_tells_predictions(data):

    story = build_action_story(data)

    assert story["chapters"]["predict"]["available"]
    assert "Add turnover, absence or performance data" in story["chapters"]["act"]["lead"] or \
        story["chapters"]["act"]["points"]
