"""Tests for the five Prediction Review fixes (v4).
Run: python -m pytest tests/test_prediction_checks.py -v"""

import numpy as np
import pandas as pd
import pytest

import src.agent as agent
from src.agent import DEFAULT_DATASET
from src.tools import prediction, prediction_checks as pc
from src.tools.health_check import check_upload
from src.tools.survival import demo_person_month, monthly_risk


@pytest.fixture(scope="module")
def dev():
    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    indiv = pd.read_csv(path / "individual_outcomes.csv")
    return survey.merge(indiv, on=["wave", "employee_id"])


BLOCK_ALL = {"summary": {"blocked_outcomes": ["all"]}, "checks": []}
BLOCK_TURNOVER = {"summary": {"blocked_outcomes": ["turnover"]}, "checks": []}


# ---------------------------------------------------------------------------
# Fix 5: smoothing
# ---------------------------------------------------------------------------
def test_smoothing_pulls_small_teams_toward_company_rate():
    smoothed = pc.smooth_rate(pd.Series([2, 20]), pd.Series([6, 60]), 0.18)
    # 2 of 6 (33%) becomes (2 + 20 x 0.18) / 26 = 21.5%; a big team moves much less
    assert smoothed[0] == pytest.approx((2 + 20 * 0.18) / 26)
    assert abs(smoothed[1] - 20 / 60) < abs(smoothed[0] - 2 / 6)


def test_above_normal_needs_three_points():
    flags = pc.above_normal(pd.Series([0.20, 0.22, np.nan]), 0.18, "turnover")
    assert flags[0] == 0 and flags[1] == 1 and np.isnan(flags[2])


# ---------------------------------------------------------------------------
# Fix 3: calibration
# ---------------------------------------------------------------------------
def test_calibrated_predictions_are_shown():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 2000)
    y = rng.binomial(1, p)
    result = pc.calibration_check(p, y)
    assert result["status"] == "passed" and result["show_percentages"]
    assert len(result["bins"]) == 5


def test_miscalibrated_predictions_are_hidden():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 2000)
    y = rng.binomial(1, np.clip(p + 0.2, 0, 1))          # reality 20 points higher
    result = pc.calibration_check(p, y)
    assert result["status"] == "off" and not result["show_percentages"]


def test_calibration_bins_follow_the_size_rule():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.5, 350)
    assert len(pc.calibration_check(p, rng.binomial(1, p))["bins"]) == 3
    assert pc.calibration_check(p[:250], rng.binomial(1, p[:250]))["status"] == "insufficient"


# ---------------------------------------------------------------------------
# Fix 1: permutation within department
# ---------------------------------------------------------------------------
def test_permutation_keeps_departments_together():
    frame = pd.DataFrame({"department": list("AAABBB"), "wave": 1, "x": [1, 2, 3, 10, 20, 30]})
    out = pc.permute_within(frame, ["x"], ["department", "wave"], np.random.default_rng(0))
    assert sorted(out.loc[out.department == "A", "x"]) == [1, 2, 3]
    assert sorted(out.loc[out.department == "B", "x"]) == [10, 20, 30]


def test_no_drivers_listed_when_q12_adds_nothing(dev, monkeypatch):
    monkeypatch.setattr(prediction, "permutation_test",
                        lambda *a, **k: {"status": "failed", "adds_signal": False, "n_permutations": 5})
    prediction._TEAM_RISK_CACHE.clear()
    result = prediction.team_risk(dev, "turnover", n_permutations=5)
    prediction._TEAM_RISK_CACHE.clear()
    assert (result["teams"]["risk_drivers"] == "").all()
    assert "did not add" in result["survey_signal"]["text"]


def test_team_risk_reports_every_check(dev):
    result = prediction.team_risk(dev, "turnover", n_permutations=20)
    for key in ("survey_signal", "change_check", "calibration", "show_percentages",
                "target_definition", "survey_response_warning"):
        assert key in result
    assert result["survey_response_warning"] is not None          # X7 is planted in the dev data
    assert "answered the survey less often" in result["survey_response_warning"]["text"]
    assert {"recent_change", "change_note"} <= set(result["teams"].columns)


# ---------------------------------------------------------------------------
# Fix 2: change gate
# ---------------------------------------------------------------------------
def test_change_gate_needs_enough_changed_teams():
    v = pd.DataFrame({"changed": [True] * 10, "previous_rate": 0.2, "target": [0, 1] * 5,
                      "_score": 0.5, "previous_target": 0, "_band": "Typical"})
    assert pc.change_gate(v, prediction.auc)["status"] == "insufficient"


def test_change_gate_fails_when_history_predicts_as_well():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.3, 400)
    noise = rng.normal(0, 1, 400)
    v = pd.DataFrame({"changed": True, "target": y, "previous_target": 0, "_band": "Typical",
                      "previous_rate": y + noise, "_score": y + noise + rng.normal(0, 0.01, 400)})
    assert pc.change_gate(v, prediction.auc)["status"] == "failed"


# ---------------------------------------------------------------------------
# Fix 4: health gate
# ---------------------------------------------------------------------------
def test_blocked_data_gives_no_team_or_group_prediction(dev):
    with pytest.raises(pc.PredictionBlocked):
        prediction.team_risk(dev, "turnover", health_report=BLOCK_ALL)
    with pytest.raises(pc.PredictionBlocked):
        prediction.team_risk(dev, "turnover", health_report=BLOCK_TURNOVER)
    with pytest.raises(pc.PredictionBlocked):
        prediction.employee_risk_by_group(dev, "department", health_report=BLOCK_TURNOVER)
    with pytest.raises(pc.PredictionBlocked):
        monthly_risk(demo_person_month(employees=200, months=12), health_report=BLOCK_ALL)


def test_blocked_outcome_does_not_stop_the_other(dev):
    result = prediction.team_risk(dev, "absence", health_report=BLOCK_TURNOVER, n_permutations=10)
    assert result["status"] in {"useful", "weak", "untested"}


# ---------------------------------------------------------------------------
# Pending year: newest wave without outcomes (multi-year data)
# ---------------------------------------------------------------------------
def test_pending_wave_is_a_forecast_and_health_check_allows_it(dev):
    hidden = dev.copy()
    last = hidden["wave"] == hidden["wave"].max()
    for column in ["left_voluntarily_12m", "turnover_window_end", "unplanned_absence_days_6m",
                   "absence_window_end", "performance_rating", "performance_review_date"]:
        hidden[column] = hidden[column].astype(object)
        hidden.loc[last, column] = None
    report = check_upload(hidden, "pending")
    codes = {c["code"]: c["status"] for c in report["checks"]}
    assert codes["OUT_PENDING"] == "PASS" and report["summary"]["fail"] == 0
    result = prediction.team_risk(hidden, "turnover", n_permutations=10)
    assert result["predict_wave"] == hidden["wave"].max()
    assert not result["outcomes_known"]
    assert result["calibration"]["status"] == "pending" and not result["show_percentages"]


# ---------------------------------------------------------------------------
# Monthly model and agent prompt
# ---------------------------------------------------------------------------
def test_monthly_model_reports_new_checks():
    result = monthly_risk(demo_person_month(), n_permutations=10)
    assert result["status"] == "useful"
    for key in ("engagement_signal", "change_check", "calibration_check", "show_percentages"):
        assert key in result


def test_prompt_tells_the_agent_predictions_are_warning_signs():
    assert "warning signs, not causes" in agent.SYSTEM_PROMPT
    assert "survey-response warning" in agent.SYSTEM_PROMPT
