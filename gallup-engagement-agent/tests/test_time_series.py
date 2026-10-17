# PROCESS 1 – IMPORTS

import numpy as np
import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.survival import (
    MIN_GROUP,
    backtest,
    demo_person_month,
    explain_monthly_risk,
    monthly_chart,
    monthly_risk,
    prepare,
)
from src.tools.ts_readiness import readiness


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def demo():

    return demo_person_month()


@pytest.fixture(scope="module")
def result(demo):

    return monthly_risk(demo)


@pytest.fixture(scope="module")
def waves():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


def _status(ready, key):
    return next(c["status"] for c in ready["checks"] if c["key"] == key)


# PROCESS 3 – READINESS CHECK

def test_demo_person_month_file_is_ready(demo):

    ready = readiness(demo)

    assert ready["ready_for_survival"]
    assert ready["counts"]["fail"] == 0
    assert all(ok for ok, _ in ready["can_run"].values())


def test_survey_waves_are_not_ready_and_say_why(waves):

    ready = readiness(waves)

    assert not ready["ready_for_survival"]
    assert _status(ready, "points") == "fail"
    assert _status(ready, "exits") == "warn"
    assert ready["can_run"]["Team risk from survey waves"][0]
    for check in ready["checks"]:
        assert check["status"] == "pass" or check["fix"]       # every gap says how to fix it


def test_rows_after_an_exit_are_flagged_as_leakage(demo):

    leaver = demo.loc[demo["exited"] == 1].iloc[0]
    ghost = leaver.copy()
    ghost["period"] = str(pd.Period(leaver["period"], "M") + 1)
    ghost["exited"] = 0

    ready = readiness(pd.concat([demo, ghost.to_frame().T], ignore_index=True))

    assert _status(ready, "after_exit") == "fail"


def test_vanishing_people_without_exit_records_are_flagged(demo):

    hidden = demo.assign(exited=0)           # leavers vanish with no exit recorded
    hidden = hidden.drop(columns="exited").assign(exited=0)

    ready = readiness(hidden)

    assert _status(ready, "leavers") == "warn"
    assert not ready["ready_for_survival"] or _status(ready, "events") == "fail"


def test_duplicate_rows_fail_the_id_check(demo):

    ready = readiness(pd.concat([demo, demo.head(50)], ignore_index=True))

    assert _status(ready, "ids") == "fail"


# PROCESS 4 – MONTHLY LEAVING RISK

def test_features_use_only_the_past(demo):

    data = prepare(demo)
    team_month = data.groupby(["team_id", "t"])["exited"].sum()

    # Team exits "in the last 3 months" never include the current month.
    row = data[(data["t"] == 10)].iloc[0]
    past = sum(team_month.get((row["team_id"], t), 0) for t in range(7, 10))
    assert row["team_exits_recent"] == past


def test_model_is_tested_on_months_it_never_saw(result):

    tests = result["tests"]

    assert result["status"] == "useful"
    assert len(tests) >= 2
    assert (tests["model_auc"] > tests["baseline_auc"]).all()


def test_model_finds_the_planted_engagement_effect(result):

    drivers = result["drivers"].set_index("feature")["odds_change_per_sd"]

    assert drivers["Engagement now"] < -0.2                   # higher engagement, fewer exits


def test_expected_leavers_are_calibrated(result):

    assert 0.8 <= result["calibration"] <= 1.25
    assert result["by_month"]["expected_leavers"].sum() == pytest.approx(result["expected_leavers"])


def test_only_groups_of_five_or_more_are_shown(result):

    for table in (result["by_department"], result["by_team"]):
        assert (table["people"] >= MIN_GROUP).all()
        assert "employee_id" not in table.columns


def test_noise_is_not_reported_as_useful(demo):
    """Shuffle exits across people: no real signal, so the model must not pass."""
    rng = np.random.default_rng(0)
    noise = demo.copy()
    noise["exited"] = rng.permutation(noise["exited"].to_numpy())

    assert monthly_risk(noise)["status"] in {"weak", "not_ready", "untested"}


def test_survey_waves_do_not_run_the_monthly_model(waves):

    assert monthly_risk(waves)["status"] == "not_ready"


def test_explanation_and_chart(result):

    story = explain_monthly_risk(result)

    assert "likely to leave in the next 6 months" in story["headline"]
    assert monthly_chart(result).to_dict()
