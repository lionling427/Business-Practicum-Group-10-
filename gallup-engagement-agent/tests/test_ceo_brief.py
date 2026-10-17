# PROCESS 1 – IMPORTS

import re

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.ceo_brief import brief_for_screen, brief_markdown, build_ceo_brief
from src.tools.dashboard import kpi_tiles
from src.tools.financial import Assumptions, current_costs, money
from src.tools.health_check import check_upload
from src.tools.insights import analysis_frame
from src.tools.priorities import factor_value
from src.tools.traps import run_trap_checks
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
def brief(data, analysis):

    return build_ceo_brief(data, analysis, traps=run_trap_checks(data, analysis["findings"]))


# PROCESS 3 – EVERY NUMBER COMES FROM A VERIFIED RESULT

def test_scorecard_matches_the_overview_tiles(data, brief):

    assert [(s["label"], s["value"]) for s in brief["scorecard"]] == \
        [(t["label"], t["value"]) for t in kpi_tiles(data)]


def test_money_matches_the_financials_page(data, analysis, brief):

    a = Assumptions(headcount=len(analysis_frame(data)))
    top = factor_value(data, analysis, a).iloc[0]

    assert money(current_costs(data, a)["total_cost"]) in brief["money"][0]
    assert money(top["yearly_value"]) in brief["bottom_line"]
    assert top["label"] in brief["bottom_line"]


def test_what_matters_uses_only_findings_that_passed(analysis, brief):

    assert 1 <= len(brief["what_matters"]) <= 3
    robust = analysis["findings"].query("robustness_status == 'ROBUST'")
    from src.tools.insights import ITEM_LABELS
    labels = {ITEM_LABELS[i] for i in robust["item"]}
    for sentence in brief["what_matters"]:
        assert any(label in sentence for label in labels)


# PROCESS 4 – CEO-READY SHAPE

def test_brief_is_short_and_decisive(brief):

    assert len(brief["decisions"]) == 3
    assert brief["decisions"][0].startswith("Back a company-wide push")
    text = brief_markdown(brief)
    assert len(text.split()) < 600                       # one page
    for heading in ["The numbers", "What drives results", "What it is worth",
                    "Where to act", "Decisions for you", "How sure are we"]:
        assert f"## {heading}" in text


def test_confidence_names_the_traps_that_apply(brief):

    assert any("not proof of cause" in p for p in brief["confidence"])
    assert any("did not answer" in p for p in brief["confidence"])   # dev data: nonresponse caution


def test_screen_version_escapes_money(brief):

    shown = brief_for_screen(brief)

    assert "\\$" in shown["bottom_line"]
    assert not re.search(r"(?<!\\)\$", shown["bottom_line"])


def test_brief_without_analysis_still_works(data):

    brief = build_ceo_brief(data)

    assert "Add turnover, absence or performance data" in brief["bottom_line"]
    assert brief["scorecard"] and brief["what_matters"] == []
