# PROCESS 1 – IMPORTS

import numpy as np
import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.data_ingestion import Q12_ITEMS
from src.tools.prediction import (
    MIN_GROUP,
    _fit_ridge_logit,
    auc,
    employee_risk_by_group,
    explain_forecast,
    explain_team_risk,
    forecast_chart,
    forecast_engagement,
    team_risk,
    team_risk_chart,
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


def _five_waves(data, slope=0.1):
    """Synthetic 5-wave copy with a steady upward trend in every answer."""

    frames = []
    base = data[data["wave"] == 1]

    for wave in range(1, 6):
        frame = base.assign(wave=wave)
        frame[Q12_ITEMS] = (frame[Q12_ITEMS] + slope * (wave - 1)).clip(1, 5)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# PROCESS 3 – MODEL HELPERS

def test_auc_known_values():

    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5


def test_ridge_logit_recovers_direction():

    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1])))).astype(float)

    beta = _fit_ridge_logit(X, y, lam=1.0)

    assert beta[1] > 1.0 and beta[2] < -0.5


# PROCESS 4 – TEAM RISK

@pytest.mark.parametrize("outcome", ["turnover", "absence"])
def test_team_risk_validated_on_latest_wave(data, outcome):

    result = team_risk(data, outcome)

    assert result["test_wave"] == data["wave"].max()
    assert result["train_waves"] == [1]
    assert result["teams"]["risk"].between(0, 1).all()
    assert result["teams"]["risk"].is_monotonic_decreasing
    assert set(result["teams"]["risk_band"]) == {"High", "Elevated", "Typical"}
    assert explain_team_risk(result)["headline"]
    assert team_risk_chart(result).to_dict()


def test_team_risk_marks_signal_free_model_weak(data):

    rng = np.random.default_rng(0)
    noise = data.copy()

    # Shuffle outcomes across teams: survey answers can no longer predict them.
    teams = noise["team_id"].unique()
    shuffled = dict(zip(teams, rng.permutation(teams)))
    rates = noise.groupby("team_id")["left_voluntarily_12m"].transform(
        lambda s: rng.random() < 0.15
    )
    noise["left_voluntarily_12m"] = rates.astype(int)
    noise["team_id"] = noise["team_id"].map(shuffled)
    noise[Q12_ITEMS] = rng.integers(1, 6, size=(len(noise), 12))

    result = team_risk(noise, "turnover")

    assert result["status"] == "weak"


def test_team_risk_with_one_wave_is_cross_validated(data):

    result = team_risk(data[data["wave"] == 1])

    assert result["method"] == "cross_validated"
    assert "cross-validation" in explain_team_risk(result)["points"][0]


# PROCESS 5 – EMPLOYEE RISK (AGGREGATE ONLY)

def test_employee_risk_returns_groups_never_people(data):

    result = employee_risk_by_group(data, "department")
    table = result["table"]

    assert "employee_id" not in table.columns
    assert (table["employees"] >= MIN_GROUP).all()
    assert table["predicted_rate"].between(0, 1).all()
    assert "fairness_flag" in table.columns


def test_employee_risk_uses_no_demographic_or_pay_fields(data):

    result = employee_risk_by_group(data, "department")

    assert set(result["features"]) <= set(Q12_ITEMS) | {"tenure_years"}


# PROCESS 6 – FORECAST

def test_forecast_refuses_with_two_waves(data):

    result = forecast_engagement(data)

    assert not result["available"]
    assert "at least 4" in explain_forecast(result)


def test_forecast_follows_a_steady_trend(data):

    result = forecast_engagement(_five_waves(data))

    assert result["available"]
    assert result["method"] == "trend"
    assert result["forecast"] > result["history"]["mean"].iloc[-1]
    assert result["low"] <= result["forecast"] <= result["high"]
    assert forecast_chart(result).to_dict()


# PROCESS 7 – LIVE USE: NEWEST SURVEY'S OUTCOMES NOT KNOWN YET

@pytest.fixture(scope="module")
def live(data):

    outcomes = ["left_voluntarily_12m", "unplanned_absence_days_6m", "performance_rating"]
    frame = data.copy()
    frame.loc[frame["wave"] == 2, outcomes] = np.nan

    return frame


def test_team_risk_looks_ahead_when_outcomes_unknown(live):

    result = team_risk(live, "turnover")

    assert result["predict_wave"] == 2
    assert result["method"] == "cross_validated"
    assert not result["outcomes_known"]
    assert result["teams"]["risk"].notna().all()
    assert any("look ahead" in p for p in explain_team_risk(result)["points"])


def test_employee_fairness_pending_without_tested_wave(live):

    result = employee_risk_by_group(live, "department")

    assert result["table"]["actual_rate"].isna().all()
    assert not result["table"]["fairness_flag"].any()


# PROCESS 8 – GROUP RISK STORY AND CHART

from src.tools.prediction import explain_group_risk, group_risk_chart  # noqa: E402


def test_group_risk_story_and_expected_leavers(data):

    result = employee_risk_by_group(data, "department")
    table = result["table"]

    assert (table["expected_leavers"] == (table["predicted_rate"] * table["employees"]).round()).all()

    story = explain_group_risk(result)
    assert str(table.iloc[0]["department"]) in story["headline"]
    assert any("expected to leave" in p for p in story["points"])
    assert group_risk_chart(result).to_dict()


def test_forecast_text_is_plain(data):

    text = explain_forecast(forecast_engagement(_five_waves(data)))

    assert "backtest" not in text.lower()
    assert "most likely" in text
