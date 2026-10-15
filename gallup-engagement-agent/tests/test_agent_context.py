# PROCESS 1 – IMPORTS

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import src.agent as agent
from src.agent import DEFAULT_DATASET, ask_agent
from src.tools.agent_context import BUILDERS, build_context, check_numbers, route_sections
from src.tools.analytics import analyze_dataframe
from src.tools.longitudinal import analyze_change
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

    return analyze_dataframe(model_sample(data), dataset_name="dev")


@pytest.fixture(scope="module")
def context(data, analysis):

    return build_context(data, analysis, analyze_change(data))


# PROCESS 3 – ROUTING

@pytest.mark.parametrize("question, expected", [
    ("Which teams should I worry about, and what will it cost?", {"teams", "financials"}),
    ("Is engagement getting better over time?", {"trends"}),
    ("What should managers do to improve productivity?", {"recommendations"}),
    ("Which departments are most likely to lose people next year?", {"predictions"}),
])
def test_route_sections(question, expected):

    sections = route_sections(question)

    assert sections[0] == "overview"
    assert expected <= set(sections)


def test_broad_question_gets_everything():

    assert set(route_sections("Give me a summary of everything")) == {"overview", *BUILDERS}


# PROCESS 4 – CONTEXT

def test_every_section_builds(context):

    for name in BUILDERS:
        assert name in context
        assert not str(context[name]).startswith("{'note': 'Not available")


def test_context_never_contains_employee_ids(data, context):

    text = json.dumps(context, default=str)

    assert not any(employee in text for employee in data["employee_id"].head(500))


def test_context_is_compact(context):

    assert len(json.dumps(context, default=str)) < 20_000


# PROCESS 5 – NUMBER CHECK

def test_check_numbers_flags_only_invented_figures(context):

    teams = context["teams"]
    honest = (f"{teams['priority_teams']} priority teams. "
              + context["financials"]["where_costs_sit"])

    assert check_numbers(honest, context) == []
    assert check_numbers("About 987654 people will leave.", context) == [987654.0]


# PROCESS 6 – AGENT USES THE CONTEXT

def test_ask_agent_sends_dashboard_sections(monkeypatch, data, analysis):

    sent = {}

    def create(**kwargs):
        sent["user"] = kwargs["messages"][1]["content"]
        sent["system"] = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"),
                                     finish_reason="stop")],
            usage=None,
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "get_azure_client", lambda: fake)

    question = "Which teams should I worry about, and what will it cost?"
    context = build_context(data, analysis, None, None, route_sections(question))

    result = ask_agent(question, analysis_override=analysis, context=context, return_debug=True)

    assert result["answer"] == "ok"
    assert "most_urgent_teams" in sent["user"]
    assert "cost_today_per_year" in sent["user"]
    assert "key_findings" in sent["user"]
    assert "statistical_findings" not in sent["user"]     # broad question: plain sentences only
    assert "never present them as certain" in sent["system"]



def test_specific_question_keeps_full_evidence(monkeypatch, data, analysis):

    sent = {}

    def create(**kwargs):
        sent["user"] = kwargs["messages"][1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "get_azure_client", lambda: fake)

    question = "How is Q04 Recognition related to voluntary turnover?"
    context = build_context(data, analysis, None, None, route_sections(question))

    ask_agent(question, analysis_override=analysis, context=context)

    assert "statistical_findings" in sent["user"]


# PROCESS 7 – THE AGENT BUILDS DASHBOARD FACTS ITSELF

def _capture(monkeypatch):

    sent = {}

    def create(**kwargs):
        sent["user"] = kwargs["messages"][1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "get_azure_client", lambda: fake)
    return sent


def test_agent_without_upload_knows_teams_and_costs(monkeypatch):

    sent = _capture(monkeypatch)

    result = ask_agent("Which teams should I worry about, and what will it cost?",
                       return_debug=True)

    assert {"teams", "financials"} <= set(result["sections"])
    assert "most_urgent_teams" in sent["user"]
    assert "cost_today_per_year" in sent["user"]


def test_agent_with_data_builds_context(monkeypatch, data, analysis):

    sent = _capture(monkeypatch)

    result = ask_agent("Is engagement getting better over time?", analysis_override=analysis,
                       data=data, return_debug=True)

    assert "trends" in result["sections"]
    assert "questions_that_clearly_changed" in sent["user"]


def test_statistics_only_mode_still_available(monkeypatch):

    sent = _capture(monkeypatch)

    result = ask_agent("Which teams should I worry about, and what will it cost?",
                       include_dashboard=False, return_debug=True)

    assert result["sections"] == []
    assert "most_urgent_teams" not in sent["user"]
