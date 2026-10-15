# PROCESS 1 – IMPORTS

from datetime import date
from urllib.parse import urlparse

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools import action_tools
from src.tools.action_tools import (
    PLAN_COLUMNS,
    action_plan_rows,
    all_links,
    check_links,
    links_markdown,
    load_resources,
    resources_for,
)
from src.tools.agent_context import build_context, route_sections
from src.tools.analytics import analyze_dataframe
from src.tools.health_check import check_upload
from src.tools.insights import ITEM_LABELS
from src.tools.waves import model_sample


# PROCESS 2 – THE RESOURCE FILE

def test_every_q12_item_has_its_gallup_element_guide():

    elements = load_resources()["elements"]

    assert set(elements) == set(ITEM_LABELS)
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
                "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth"]
    for number, ordinal in enumerate(ordinals, start=1):
        entry = elements[f"Q{number:02d}"]
        assert entry["title"] == f"The {ordinal} Element of Great Managing"
        assert ordinal.lower() in entry["url"]


def test_links_are_https_gallup_pages():

    for link in all_links():
        url = urlparse(link["url"])
        assert url.scheme == "https"
        assert url.netloc.endswith("gallup.com")


def test_organization_can_add_its_gallup_access_link(monkeypatch):

    custom = dict(load_resources(), gallup_access_url="https://example.org/gallup-access")
    monkeypatch.setattr(action_tools, "load_resources", lambda path=None: custom)

    planning = action_tools.action_planning_links()

    assert planning[0]["url"] == "https://example.org/gallup-access"


# PROCESS 3 – RECOMMENDATION → RESOURCES

def test_resources_for_a_recommendation_are_grouped():

    resources = resources_for("Q04")

    kinds = [r["kind"] for r in resources]
    assert kinds[0] == "Element guide"
    assert "Action planning" in kinds and "Manager learning" in kinds
    assert "fourth-element" in resources[0]["url"]

    md = links_markdown(resources)
    assert "**Element guide**" in md and "](https://" in md


def test_no_item_still_gives_planning_and_learning():

    kinds = {r["kind"] for r in resources_for(None)}

    assert kinds == {"Action planning", "Manager learning"}


# PROCESS 4 – ACTION PLANS

@pytest.fixture
def plans():

    return pd.DataFrame([
        {"team_id": "T1", "department": "Sales", "start_with": "Q04",
         "team_score": 3.1, "company_score": 3.6},
        {"team_id": "T2", "department": "HR", "start_with": None,
         "team_score": float("nan"), "company_score": float("nan")},
    ])


def test_action_plan_has_gallup_fields_and_dates(plans):

    rows = action_plan_rows(plans, start=date(2026, 1, 5))

    assert list(rows.columns) == PLAN_COLUMNS
    assert set(rows["team_id"]) == {"T1"}                 # teams with no gap get no plan
    assert len(rows) == 3
    assert rows["due_date"].tolist() == ["2026-01-19", "2026-02-16", "2026-03-02"]
    assert rows["follow_up"].unique().tolist() == ["2026-04-05"]
    assert "from 3.10 to the company average of 3.60" in rows["goal"].iloc[0]
    assert "fourth-element" in rows["learn_more"].iloc[0]


# PROCESS 5 – LINK CHECK (NO NETWORK IN TESTS)

def test_link_check_flags_broken_links():

    broken = all_links()[1]["url"]
    report = check_links(fetch=lambda url: 404 if url == broken else 200)

    assert not report.set_index("url").loc[broken, "ok"]
    assert report["ok"].sum() == len(report) - 1


def test_script_blocked_help_page_is_not_a_failure():

    report = check_links(fetch=lambda url: 403 if "support.gallup.com" in url else 200)

    assert report["ok"].all()


# PROCESS 6 – THE AGENT SEES THE RESOURCES

def test_agent_gets_gallup_resources():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    data = survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])
    analysis = analyze_dataframe(model_sample(data), dataset_name="dev",
                                 health_report=check_upload(data, "dev"))

    assert "recommendations" in route_sections("Any Gallup learning resources for managers?")

    context = build_context(data, analysis, sections=["recommendations"])
    resources = context["recommendations"]["gallup_resources"]

    assert any("Element of Great Managing" in r["title"] for r in resources["links"])
    assert "not endorsed by Gallup" in resources["how_to_use"]
