# PROCESS 1 – IMPORTS

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.insights import team_scorecard
from src.tools.strengths import (
    DOMAINS,
    MIN_GROUP,
    PLAYS,
    THEME_MEANING,
    THEMES,
    explain_themes,
    company_profile,
    demo_strengths,
    domain_chart,
    normalize_theme,
    pilot_effect,
    read_strengths,
    team_profiles,
    themes_chart,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def survey():

    path = DEFAULT_DATASET / "agent_data"
    s = pd.read_csv(path / "survey_responses.csv")
    i = pd.read_csv(path / "individual_outcomes.csv")
    return s.merge(i, on=[c for c in s.columns if c in i.columns])


@pytest.fixture(scope="module")
def strengths(survey):

    return read_strengths(demo_strengths(survey))


# PROCESS 3 – THEMES AND DOMAINS

def test_34_themes_in_four_domains():

    assert len(THEMES) == 34
    assert Counter(THEMES.values()) == {"Executing": 9, "Influencing": 8,
                                        "Relationship Building": 9, "Strategic Thinking": 8}


def test_every_question_has_an_action_for_every_domain():

    assert set(PLAYS) == {f"Q{n:02d}" for n in range(1, 13)}
    for plays in PLAYS.values():
        assert set(plays) == set(DOMAINS) and all(p.endswith((".", "?")) for p in plays.values())


@pytest.mark.parametrize("raw, theme", [("self assurance", "Self-Assurance"), ("WOO", "Woo"),
                                         (" Learner ", "Learner"), ("Strategy", None)])
def test_theme_names_are_read_flexibly(raw, theme):

    assert normalize_theme(raw) == theme


def test_one_cell_lists_and_unknown_names():

    data = pd.DataFrame({"employee_id": [1], "top5": ["Achiever; WOO, Learner|Relator, Foo"]})

    result = read_strengths(data)

    assert result["long"]["theme"].tolist() == ["Achiever", "Woo", "Learner", "Relator"]
    assert result["unknown"] == ["Foo"]


# PROCESS 4 – PROFILES AND PRIVACY

def test_team_profiles_respect_privacy_and_add_up(survey, strengths):

    teams = team_profiles(strengths, survey, team_scorecard(survey))

    assert not teams.empty
    assert (teams["people_with_strengths"] >= MIN_GROUP).all()
    assert np.allclose(teams[DOMAINS].sum(axis=1), 1.0)
    assert "employee_id" not in teams.columns
    assert (teams["play"].str.len() > 0).all()


def test_small_teams_are_hidden(survey, strengths):

    four = strengths["long"]["employee_id"].drop_duplicates().head(4)
    few = {**strengths, "long": strengths["long"][strengths["long"]["employee_id"].isin(four)]}

    assert team_profiles(few, survey).empty                # 4 people: below the 5-person rule


def test_company_profile(strengths):

    company = company_profile(strengths)

    assert company["mix"].sum() == pytest.approx(1.0)
    assert len(company["top_themes"]) == 10
    assert domain_chart(company["mix"]).to_dict() and themes_chart(company["top_themes"]).to_dict()


# PROCESS 5 – PILOT: COACHED VS OTHER TEAMS

def _with_effect(survey, strengths, lift):
    """Planted effect: coached teams' Q03 rises by `lift` in the latest survey."""
    coached = strengths["coached"]
    latest = survey["wave"].max()
    out = survey.copy()
    mask = (out["wave"] == latest) & out["employee_id"].map(coached).fillna(0).astype(bool)
    out.loc[mask, "Q03"] = (out.loc[mask, "Q03"] + lift).clip(upper=5)
    return out


def test_random_coaching_shows_no_clear_difference(survey, strengths):

    pilot = pilot_effect(survey, strengths, "Q03")

    assert pilot["status"] == "unclear"
    assert pilot["range"][0] < 0 < pilot["range"][1]


def test_planted_coaching_effect_is_found(survey, strengths):

    pilot = pilot_effect(_with_effect(survey, strengths, 0.6), strengths, "Q03")

    assert pilot["status"] == "clear"
    assert pilot["effect"] > 0.2


def test_pilot_needs_coaching_records(survey, strengths):

    assert pilot_effect(survey, {**strengths, "coached": None})["status"] == "no_pilot"


def test_every_theme_has_a_plain_meaning():

    assert set(THEME_MEANING) == set(THEMES)
    assert all(m.endswith(".") and len(m) < 100 for m in THEME_MEANING.values())
    assert "Winning Others Over" in THEME_MEANING["Woo"]
    assert explain_themes("Woo, Learner, Nonsense")[0].startswith("**Woo:**")
    assert len(explain_themes("Woo, Learner, Nonsense")) == 2
