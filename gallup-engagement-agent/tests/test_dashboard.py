# PROCESS 1 – IMPORTS

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.analytics import analyze_dataframe
from src.tools.dashboard import kpi_tiles, wave_summary, what_drives_results, where_we_stand
from src.tools.insights import ITEM_LABELS, analysis_frame
from src.tools.waves import latest_wave, model_sample, model_wave


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


# PROCESS 3 – WAVE RULES

def test_model_wave_is_newest_wave_with_outcomes(data):

    assert model_wave(data) == latest_wave(data) == 2

    unknown = data.copy()
    unknown.loc[unknown["wave"] == 2, ["left_voluntarily_12m",
                                        "unplanned_absence_days_6m",
                                        "performance_rating"]] = None

    assert model_wave(unknown) == 1
    assert latest_wave(unknown) == 2


def test_models_and_analysis_frame_use_the_same_rows(data):

    analysis = analyze_dataframe(model_sample(data), dataset_name="dev")

    assert analysis["sample_size"] == len(analysis_frame(data))


# PROCESS 4 – TILES

def test_tiles_describe_latest_wave_with_change(data):

    tiles = {tile["label"]: tile for tile in kpi_tiles(data)}
    table = wave_summary(data)

    engaged = tiles["Employees engaged"]
    assert engaged["value"] == f"{table['pct_engaged'].iloc[-1]:.0f}%"
    assert engaged["delta"] == pytest.approx(
        table["pct_engaged"].iloc[-1] - table["pct_engaged"].iloc[-2], abs=0.051)
    assert len(engaged["trend"]) == 2

    assert tiles["Voluntary turnover"]["inverse"]          # down is good
    assert not tiles["Performance rating"]["inverse"]


# PROCESS 5 – KEY FINDINGS

def test_where_we_stand_uses_app_labels(data):

    points = where_we_stand(data)

    assert len(points) == 3
    assert any(label in points[0] for label in ITEM_LABELS.values())


def test_drivers_are_plain_language(data):

    analysis = analyze_dataframe(model_sample(data), dataset_name="dev")
    drivers = what_drives_results(analysis, data)

    assert drivers
    for driver in drivers:
        assert "linked to" in driver["headline"]
        assert " SD" not in driver["headline"]

    assert what_drives_results(None, data) == []


# PROCESS 6 – OUTCOMES PAGE

from src.tools.dashboard import driver_chart, driver_rows  # noqa: E402


def test_driver_rows_plain_sizes(data):

    analysis = analyze_dataframe(model_sample(data), dataset_name="dev")
    rows = driver_rows(analysis, data)

    assert len(rows) == len(analysis["findings"])
    turnover = rows[rows["outcome"] == "turnover"]
    assert turnover["size_text"].str.contains("% (?:lower|higher)").all()
    absence = rows[rows["outcome"] == "absence"]
    assert absence["size_text"].str.contains("% (?:fewer|more)").all()
    caution = rows[rows["reliability"] == "Use with caution"]
    assert (caution["caution"] != "").all()
    for outcome in rows["outcome"].unique():
        assert driver_chart(rows, outcome).to_dict()


# PROCESS 7 – TEAMS WATCHLIST

from src.tools.dashboard import team_watchlist  # noqa: E402
from src.tools.insights import manager_brief, team_scorecard  # noqa: E402
from src.tools.prediction import team_risk  # noqa: E402


def test_watchlist_ranks_most_urgent_first(data):

    analysis = analyze_dataframe(model_sample(data), dataset_name="dev")
    risks = {o: team_risk(data, o) for o in ["turnover", "absence"]}
    watch = team_watchlist(team_scorecard(data, analysis), risks)

    assert watch["urgency"].is_monotonic_decreasing
    assert watch.loc[watch["priority"], "urgency"].min() >= 3
    assert set(watch["turnover_risk"]) <= {"High", "Elevated", "Typical", "—"}
    assert watch["needs_attention"].equals(watch["urgency"] > 0)


def test_watchlist_ignores_failed_models(data):

    risks = {o: {**team_risk(data, o), "status": "weak"} for o in ["turnover", "absence"]}
    watch = team_watchlist(team_scorecard(data), risks)

    assert (watch["turnover_risk"] == "—").all()


def test_brief_names_strengths_honestly(data):

    scorecard = team_scorecard(data)
    weakest = scorecard.sort_values("q12_average").iloc[0]["team_id"]

    brief = manager_brief(data, weakest)

    assert "still below the company" in brief["strengths_heading"]
