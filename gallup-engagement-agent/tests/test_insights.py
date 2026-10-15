# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.waves import model_sample
from src.tools.insights import (
    MIN_TEAM_SIZE,
    analysis_frame,
    brief_markdown,
    dollar_impact,
    explain_finding,
    findings_table,
    manager_brief,
    team_scorecard,
    top_drivers,
    verified_items,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"

    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    keys = [
        column
        for column in survey.columns
        if column in individual.columns
    ]

    return survey.merge(individual, on=keys)


@pytest.fixture(scope="module")
def analysis(data):

    return analyze_dataframe(model_sample(data), dataset_name="dev")


@pytest.fixture(scope="module")
def scorecard(data, analysis):

    return team_scorecard(data, analysis)


# PROCESS 3 – ANALYSIS FRAME

def test_analysis_frame_matches_model_sample(data, analysis):

    assert len(analysis_frame(data)) == analysis["sample_size"]


# PROCESS 4 – FINDINGS

def test_explain_finding_uses_association_language(data, analysis):

    for _, row in analysis["findings"].iterrows():

        explained = explain_finding(row, data)

        assert "linked to" in explained["headline"]
        assert "cause" not in explained["headline"].lower()


def test_findings_table_has_one_row_per_finding(data, analysis):

    table = findings_table(analysis, data)

    assert len(table) == len(analysis["findings"])
    assert "What the data shows" in table.columns


def test_top_drivers_only_use_robust_findings(analysis):

    drivers = top_drivers(analysis)

    assert drivers

    for driver in drivers.values():
        assert driver["robustness_status"] == "ROBUST"


def test_verified_items_are_robust_and_favorable(analysis):

    findings = analysis["findings"]
    robust = findings[findings["robustness_status"] == "ROBUST"]
    good_lower = robust["outcome"].isin(["turnover", "absence"])
    favorable = (good_lower & (robust["effect"] < 1)) | (~good_lower & (robust["effect"] > 0))

    assert set(verified_items(analysis)) == set(robust.loc[favorable, "item"])


# PROCESS 5 – DOLLAR IMPACT

def test_dollar_impact_turnover_savings(data, analysis):

    impact = dollar_impact(analysis, data)

    assert impact is not None
    assert impact["savings"] >= 0
    assert impact["fewer_leavers"] <= impact["current_leavers"]
    assert impact["assumptions"]


# PROCESS 6 – TEAMS

def test_team_scorecard_respects_privacy_and_flags(scorecard):

    assert not scorecard.empty
    assert (scorecard["employees"] >= MIN_TEAM_SIZE).all()

    flagged = scorecard[scorecard["priority"]]

    assert len(flagged) >= 1
    assert flagged["meets_priority_rules"].all()


def test_manager_brief_and_markdown(data, analysis, scorecard):

    team_id = scorecard["team_id"].iloc[0]

    brief = manager_brief(data, team_id, analysis)

    assert brief["team_id"] == team_id
    assert len(brief["actions"]) == 3
    assert len(brief["questions"]) == 2

    markdown = brief_markdown(brief)

    assert brief["focus"]["label"] in markdown
    assert "not proof of cause and effect" in markdown


def test_manager_brief_rejects_unknown_team(data, analysis):

    with pytest.raises(ValueError):
        manager_brief(data, "NO_SUCH_TEAM", analysis)


# PROCESS 7 – DIRECTION AND MISSING-COLUMN GUARDS

def _with_unfavorable_turnover(analysis):
    """Copy of the analysis where the top turnover driver points the wrong way."""

    findings = analysis["findings"].copy()

    item = top_drivers(analysis)["turnover"]["item"]
    row = (findings["outcome"] == "turnover") & (findings["item"] == item)

    findings.loc[row, "effect"] = 1 / findings.loc[row, "effect"]

    return {**analysis, "findings": findings}, item


def test_top_drivers_skip_unfavorable_findings(analysis):

    flipped, item = _with_unfavorable_turnover(analysis)

    assert top_drivers(flipped)["turnover"]["item"] != item
    assert item not in verified_items(flipped, "turnover")


def test_dollar_impact_never_negative_with_unfavorable_finding(data, analysis):

    flipped, _ = _with_unfavorable_turnover(analysis)

    impact = dollar_impact(flipped, data)

    assert impact is None or impact["savings"] >= 0


def test_linked_outcomes_state_direction(data, analysis, scorecard):

    for team_id in scorecard["team_id"].head(20):

        brief = manager_brief(data, team_id, analysis)

        for phrase in brief["focus"]["linked_outcomes"]:
            assert phrase.split()[0] in ("lower", "higher")


def test_team_scorecard_without_employee_id(data, analysis):

    scorecard = team_scorecard(
        data.drop(columns=["employee_id"]),
        analysis,
    )

    assert not scorecard.empty
    assert (scorecard["employees"] >= MIN_TEAM_SIZE).all()


# PROCESS 8 – RISK OUTLOOK IN THE MANAGER BRIEF

from src.tools.insights import risk_outlook  # noqa: E402
from src.tools.manager_coach import find_unverified_numbers  # noqa: E402
from src.tools.prediction import team_risk  # noqa: E402


@pytest.fixture(scope="module")
def risk_results(data):

    return {outcome: team_risk(data, outcome) for outcome in ["turnover", "absence"]}


@pytest.mark.parametrize("band, urgency", [
    ("High", "Act this month"),
    ("Elevated", "Keep a close eye this quarter"),
])
def test_risk_outlook_urgency_follows_band(data, risk_results, band, urgency):

    teams = risk_results["turnover"]["teams"]
    absence = risk_results["absence"]["teams"].set_index("team_id")["risk_band"]

    # A team in this turnover band and no worse band for absence.
    order = {"High": 0, "Elevated": 1, "Typical": 2}
    candidates = teams[
        (teams["risk_band"] == band)
        & teams["team_id"].map(lambda t: order[absence.get(t, "Typical")] >= order[band])
    ]
    team_id = candidates["team_id"].iloc[0]

    outlook = risk_outlook(team_id, risk_results)

    assert outlook["urgency"] == urgency
    assert "chance" in outlook["headline"]
    assert 1 <= len(outlook["steps"]) <= 3
    assert "not a certainty" in outlook["caution"]


def test_risk_outlook_ignores_models_that_failed(risk_results):

    failed = {o: {**r, "status": "weak"} for o, r in risk_results.items()}
    team_id = risk_results["turnover"]["teams"]["team_id"].iloc[0]

    assert risk_outlook(team_id, failed) is None


def test_brief_with_risk_passes_number_check(data, risk_results):

    team_id = risk_results["turnover"]["teams"]["team_id"].iloc[0]

    brief = manager_brief(data, team_id, risk_results=risk_results)
    markdown = brief_markdown(brief)

    assert "Risk outlook" in markdown
    assert "Recommended next steps" in markdown
    assert find_unverified_numbers(markdown, brief) == []


def test_brief_without_risk_is_unchanged(data, scorecard):

    brief = manager_brief(data, scorecard["team_id"].iloc[0])

    assert brief["risk"] is None
    assert "Risk outlook" not in brief_markdown(brief)
