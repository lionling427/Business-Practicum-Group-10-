# PROCESS 1 – IMPORTS

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.agent import DEFAULT_DATASET, PROJECT_ROOT
from src.tools.analytics import HealthCheckBlocked, analyze_dataframe
from src.tools.health_check import (
    blocked_outcomes,
    build_report,
    check_upload,
    model_controls,
    required_adjustments,
)
from src.tools.waves import model_sample

APP = str(PROJECT_ROOT / "app.py")


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


def _with_absence_dated_before_survey(data):
    """Absence windows that end before the survey: absence must be blocked."""
    broken = data.copy()
    broken["absence_window_end"] = pd.to_datetime(broken["survey_date"]) + pd.Timedelta(days=180)
    broken.loc[broken.index[:10], "absence_window_end"] = (
        pd.to_datetime(broken.loc[broken.index[:10], "survey_date"]) - pd.Timedelta(days=1))
    return broken


# PROCESS 3 – THE CHECKER

def test_refactor_matches_the_original_report():

    d = DEFAULT_DATASET
    saved = json.load(open(d / "reports" / "health_report.json"))
    survey = pd.read_csv(d / "agent_data" / "survey_responses.csv",
                         dtype={"employee_id": str, "team_id": str})
    indiv = pd.read_csv(d / "agent_data" / "individual_outcomes.csv", dtype={"employee_id": str})
    team = pd.read_csv(d / "agent_data" / "team_outcomes.csv", dtype={"team_id": str})

    new = build_report(survey, indiv, team, d.name, survey_raw_columns=list(survey.columns))

    old_checks = {c["code"]: (c["status"], c["actions"]) for c in saved["checks"]}
    new_checks = {c["code"]: (c["status"], c["actions"]) for c in new["checks"]}
    assert new_checks == old_checks


@pytest.mark.parametrize("break_it", [
    lambda d: pd.concat([d, d.head(3)]),                      # same person twice in a survey
    lambda d: d.assign(Q01=d["Q01"].where(d.index > 4, 9)),   # scores outside 1-5
    lambda d: d.assign(tenure_years=-1.0),                    # negative tenure
])
def test_data_errors_block_everything(data, break_it):

    report = check_upload(break_it(data))

    assert blocked_outcomes(report) == ["all"]
    assert any("stopped" in a["adjustment"] for a in required_adjustments(report))


def test_outcome_dated_before_survey_blocks_only_that_outcome(data):

    report = check_upload(_with_absence_dated_before_survey(data))

    assert blocked_outcomes(report) == ["absence"]
    assert report["summary"]["analysis_allowed"] == {
        "turnover": True, "absence": False, "performance": True}


def test_known_biases_require_controls(data):

    report = check_upload(data)

    assert blocked_outcomes(report) == []
    for check in report["checks"]:
        for action in check["actions"]:
            if check["status"] != "PASS" and action.startswith("add_control:"):
                column = action.split(":", 1)[1]
                for outcome in (check["affects"] if check["affects"] != ["all"]
                                else ["turnover", "absence", "performance"]):
                    assert column in model_controls(report, outcome)


# PROCESS 4 – THE ANALYSIS OBEYS THE REPORT

def test_analysis_stops_on_errors(data):

    broken = pd.concat([data, data.head(3)])

    with pytest.raises(HealthCheckBlocked):
        analyze_dataframe(model_sample(broken), health_report=check_upload(broken))


def test_blocked_outcome_is_never_reported(data):

    broken = _with_absence_dated_before_survey(data)
    analysis = analyze_dataframe(model_sample(broken), health_report=check_upload(broken))

    assert "absence" not in set(analysis["findings"]["outcome"])
    assert analysis["adjustments"]["blocked_outcomes"] == ["absence"]


def test_required_controls_are_added_to_models(data):

    report = check_upload(data)
    analysis = analyze_dataframe(model_sample(data), health_report=report)
    added = analysis["adjustments"]["model_controls_added"]

    base = {"turnover": {"tenure_years", "department", "role_level", "pay_band"},
            "absence": {"tenure_years", "department", "shift_type"},
            "performance": {"tenure_years", "department", "prior_performance_rating"}}
    for outcome, controls in added.items():
        assert not set(controls) & base[outcome]              # never added twice
        expected = [c for c in model_controls(report, outcome)
                    if c in data.columns and c not in base[outcome]]
        assert controls == expected


# PROCESS 5 – THE APP

def _upload(frame, name="file.csv"):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.file_uploader(key="data_file").upload(name, frame.to_csv(index=False).encode(), "text/csv").run()
    return at


def test_app_pauses_pages_when_data_has_errors(data):

    at = _upload(pd.concat([data, data.head(3)]))

    for page in ["app_pages/overview.py", "app_pages/outcomes.py", "app_pages/agent.py"]:
        at.switch_page(page).run()
        assert not at.exception
        assert any("Analysis stopped" in e.value for e in at.error)

    at.switch_page("app_pages/data_quality.py").run()
    assert not at.exception
    assert any("Errors to fix" in m.value for m in at.markdown)


def test_app_removes_a_blocked_outcome(data):

    at = _upload(_with_absence_dated_before_survey(data))

    assert "unplanned_absence_days_6m" not in at.session_state["uploaded_data"].columns
    assert "unplanned_absence_days_6m" in at.session_state["uploaded_raw"].columns

    at.switch_page("app_pages/overview.py").run()
    assert not at.exception
