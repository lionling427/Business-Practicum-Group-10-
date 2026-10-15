# PROCESS 1 – IMPORTS

import numpy as np
import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools import stats_tools as st
from src.tools.longitudinal import (
    INDEX,
    _fixed_effects,
    analyze_change,
    derive_wave,
    explain_item_change,
    explain_team_change,
    item_change,
    item_change_chart,
    team_change,
    team_movers_chart,
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


@pytest.fixture(scope="module")
def change(data):

    return analyze_change(data)


# PROCESS 3 – WAVES FROM DATES

def test_derive_wave_from_survey_years(data):

    derived = derive_wave(data.drop(columns=["wave"]))

    assert (derived["wave"] == data["wave"]).all()


def test_derive_wave_leaves_single_year_alone(data):

    one_year = data[data["wave"] == 1].drop(columns=["wave"])

    assert "wave" not in derive_wave(one_year).columns


# PROCESS 4 – FIXED EFFECTS RECOVER A KNOWN WITHIN-PERSON EFFECT

def test_fixed_effects_removes_stable_confounder():

    rng = np.random.default_rng(0)
    people, waves, true_slope = 400, 3, 0.5

    person = np.repeat(np.arange(people), waves)
    wave = np.tile(np.arange(1, waves + 1), people)
    trait = np.repeat(rng.normal(0, 1, people), waves)      # stable confounder

    x = 3 + 0.8 * trait + rng.normal(0, 0.5, people * waves)
    y = 2.0 * trait + true_slope * x + 0.3 * wave + rng.normal(0, 0.3, people * waves)

    panel = pd.DataFrame({"id": person, "wave": wave, "x": x, "y": y})

    fit = _fixed_effects(panel, "y", "x", "id", "id")
    pooled = st.ols(panel["y"], st.design(panel, ["x"])).params["x"]

    assert fit["ci_low"] <= true_slope <= fit["ci_high"]
    assert abs(pooled - true_slope) > 1          # naive comparison is badly biased


# PROCESS 5 – DESCRIPTIVE CHANGE

def test_item_change_same_employees(data):

    result = item_change(data, same_employees=True)
    table = result["table"]

    assert len(table) == 13
    respondents = data[data["responded"] == 1]

    assert result["n_employees"] == int(
        respondents.groupby("employee_id")["wave"].nunique().eq(2).sum()
    )

    index = table[table["measure"] == INDEX].iloc[0]
    assert index["change"] == pytest.approx(index["after"] - index["before"], abs=0.011)
    assert explain_item_change(result)["headline"]


def test_item_change_needs_two_waves(data):

    with pytest.raises(ValueError):
        item_change(data[data["wave"] == 1])


def test_team_change_sorted_and_private(data):

    teams = team_change(data)

    assert teams["change"].is_monotonic_increasing
    assert explain_team_change(teams)["headline"]


# PROCESS 6 – PANEL MODELS

def test_analyze_change_runs_all_models(change):

    results = change["results"]

    assert set(results["model"]) == {"employee_fe", "team_fe", "trajectory"}
    assert results.groupby(["model", "outcome"]).size().eq(13).all()
    assert (results["ci_low"] <= results["ci_high"]).all()


def test_change_explanations_use_association_language(change):

    for text in change["results"]["explanation"]:
        assert "cause" not in text.lower()
        assert "rises 1 point" in text or "fell 1 point" in text


def test_analyze_change_needs_two_waves(data):

    with pytest.raises(ValueError):
        analyze_change(data[data["wave"] == 1])


# PROCESS 7 – CHARTS

def test_change_charts_compile(data):

    assert item_change_chart(item_change(data)).to_dict()
    assert team_movers_chart(team_change(data)).to_dict()


# PROCESS 8 – COMPARING CHOSEN WAVES AND LEADER SUMMARIES

from src.tools.longitudinal import (  # noqa: E402
    outcome_summary,
    question_matrix,
    short_effect,
)


def test_item_change_accepts_chosen_waves(data):

    result = item_change(data, before=2, after=1)     # order does not matter

    assert (result["before"], result["after"]) == (1, 2)

    with pytest.raises(ValueError):
        item_change(data, before=1, after=1)


def test_outcome_summary_one_card_per_outcome(change):

    cards = outcome_summary(change["results"])

    assert {c["key"] for c in cards} == {"turnover", "absence", "performance"}
    for card in cards:
        assert card["verdict"]
        assert all(line["text"] for line in card["lines"])


def test_short_effect_is_compact(change):

    for _, row in change["results"].iterrows():
        text = short_effect(row, typical_note=False)
        assert len(text) < 60
        assert "per +1 point" in text or "odds of leaving" in text


def test_question_matrix_has_a_row_per_question(change):

    matrix = question_matrix(change["results"], "performance")

    assert len(matrix) == 12
    assert "Same people" in matrix.columns
    assert matrix.iloc[:, 1:].apply(lambda c: c.str[0].isin(["✓", "·"])).all().all()
