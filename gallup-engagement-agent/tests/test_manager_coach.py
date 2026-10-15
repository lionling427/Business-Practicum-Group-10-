# PROCESS 1 – IMPORTS

from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools.insights import (
    brief_markdown,
    manager_brief,
    team_scorecard,
)
from src.tools.manager_coach import (
    ACTION_WINDOW_DAYS,
    CONVERSATION_MINUTES,
    coach_brief,
    extract_numbers,
    find_unverified_numbers,
)


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def brief():

    path = DEFAULT_DATASET / "agent_data"

    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    data = survey.merge(
        individual,
        on=[c for c in survey.columns if c in individual.columns],
    )

    team_id = team_scorecard(data)["team_id"].iloc[0]

    return manager_brief(data, team_id)


class FakeClient:
    """Stands in for the Azure client and returns a fixed answer."""

    def __init__(self, answer=None, error=None):

        def create(**kwargs):

            if error:
                raise error

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=answer)
                    )
                ]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


# PROCESS 3 – NUMBER VERIFICATION TESTS

def test_deterministic_brief_has_no_unverified_numbers(brief):

    assert find_unverified_numbers(
        brief_markdown(brief),
        brief,
    ) == []


def test_extract_numbers_ignores_codes_and_list_markers():

    text = "1. Q04 Recognition scored 3.40\n## 2) Team T000123 had 1,200"

    assert extract_numbers(text) == {3.4, 1200}


# PROCESS 4 – AI COACH TESTS

def test_coach_accepts_answer_with_brief_numbers(brief):

    answer = (
        f"Your team sits at {brief['q12_average']} over the next "
        f"{ACTION_WINDOW_DAYS} days; start with a "
        f"{CONVERSATION_MINUTES}-minute conversation."
    )

    result = coach_brief(brief, client=FakeClient(answer))

    assert result["source"] == "ai"
    assert result["text"] == answer


def test_coach_falls_back_on_invented_number(brief):

    result = coach_brief(
        brief,
        client=FakeClient("Engagement rose by 47% this year."),
    )

    assert result["source"] == "deterministic"
    assert "47" in result["reason"]
    assert result["text"] == brief_markdown(brief)


def test_coach_falls_back_when_ai_fails(brief):

    result = coach_brief(
        brief,
        client=FakeClient(error=RuntimeError("no key")),
    )

    assert result["source"] == "deterministic"


def test_coach_falls_back_on_empty_answer(brief):

    result = coach_brief(brief, client=FakeClient(""))

    assert result["source"] == "deterministic"


def test_coach_prompt_covers_risk_section():

    from src.tools.manager_coach import COACH_PROMPT

    assert '"risk" section' in COACH_PROMPT
    assert "likelihood, never a certainty" in COACH_PROMPT
