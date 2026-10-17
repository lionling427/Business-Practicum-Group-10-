# PROCESS 1 – IMPORTS

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.agent import DEFAULT_DATASET, PROJECT_ROOT


# PROCESS 2 – CONFIGURATION

APP = str(PROJECT_ROOT / "app.py")

PAGES = [
    "app_pages/overview.py",
    "app_pages/ceo_brief.py",
    "app_pages/engagement.py",
    "app_pages/change.py",
    "app_pages/outcomes.py",
    "app_pages/teams.py",
    "app_pages/action_story.py",
    "app_pages/predictions.py",
    "app_pages/recommendations.py",
    "app_pages/strengths.py",
    "app_pages/financials.py",
    "app_pages/scenarios.py",
    "app_pages/agent.py",
    "app_pages/data_quality.py",
    "app_pages/audit.py",
    "app_pages/report_pack.py",
]


# PROCESS 3 – FIXTURES

@pytest.fixture(scope="module")
def csv_bytes():

    path = DEFAULT_DATASET / "agent_data"

    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    data = survey.merge(
        individual,
        on=[c for c in survey.columns if c in individual.columns],
    )

    return data.to_csv(index=False).encode()


def _app_with_upload(csv_bytes):

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.file_uploader(key="data_file").upload("dev.csv", csv_bytes, "text/csv").run()

    return at


def _assert_clean(at, page):

    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"

    technical = [c.value for c in at.caption if "Technical detail" in c.value]
    assert not technical, f"{page}: {technical}"


# PROCESS 4 – EMPTY STATE

@pytest.mark.parametrize("page", PAGES)
def test_pages_without_data_show_empty_state(page):

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page(page).run()

    assert not at.exception, [e.value for e in at.exception]


# PROCESS 5 – PAGES WITH DATA

@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_uploaded_data(csv_bytes, page):

    at = _app_with_upload(csv_bytes)
    at.switch_page(page).run()

    _assert_clean(at, page)


def test_analysis_runs_and_unlocks_findings(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/outcomes.py").run()


    _assert_clean(at, "outcomes after analysis")
    assert at.session_state["uploaded_analysis"] is not None
    assert any("Which questions matter" in h.value for h in at.subheader)
    assert any("Analytical traps" in h.value for h in at.subheader)
    assert "with_findings" in at.session_state["trap_checks"]

    # Other pages pick up the analysis.
    for page in ["app_pages/overview.py", "app_pages/teams.py"]:
        at.switch_page(page).run()
        _assert_clean(at, page)


def test_overview_scope_switch(csv_bytes):

    at = _app_with_upload(csv_bytes)

    at.segmented_control(key="overview_scope").set_value("team_id").run()

    _assert_clean(at, "overview team view")
    assert any("percentile" in m.value for m in at.markdown)


def test_change_analysis_runs(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/change.py").run()


    _assert_clean(at, "change after analysis")
    assert at.session_state["change_analysis"]["results"].shape[0] > 0
    assert any("Rising engagement goes with" in m.value for m in at.markdown)


def test_dates_without_wave_column_get_waves(csv_bytes):

    no_wave = pd.read_csv(__import__("io").BytesIO(csv_bytes)).drop(columns=["wave"])

    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.file_uploader(key="data_file").upload(
        "dates.csv", no_wave.to_csv(index=False).encode(), "text/csv"
    ).run()

    assert sorted(at.session_state["uploaded_data"]["wave"].unique()) == [1, 2]


def test_teams_page_shows_risk_ranking(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/teams.py").run()

    _assert_clean(at, "teams with risk")
    assert any("Manager coach" in h.value for h in at.subheader)
    assert any("passed its test" in m.value for m in at.markdown)
    assert len(at.dataframe) >= 1      # the watchlist


def test_predictions_page_never_lists_employees(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/predictions.py").run()

    _assert_clean(at, "predictions")

    for frame in at.dataframe:
        assert "employee_id" not in frame.value.columns


def test_recommendations_page_builds_plans(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/recommendations.py").run()


    _assert_clean(at, "recommendations")
    assert any("Plan for each team" in h.value for h in at.subheader)
    assert any("Start with" in m.value or "Protect what works" in m.value for m in at.markdown)
    assert any("Take it to Gallup action planning" in e.label for e in at.expander)
    assert any("element-great-managing" in m.value for m in at.markdown)


def test_financials_page_values_engagement(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/financials.py").run()


    _assert_clean(at, "financials")
    assert any("could be worth" in h.value for h in at.subheader)
    assert any(m.label == "Total, per year" for m in at.metric)
    assert any("Where to invest" in h.value for h in at.subheader)


FIVE_YEAR_FILE = __import__("pathlib").Path.home() / "Downloads" / "gallup_q12_survey_5yr.csv"


@pytest.mark.skipif(not FIVE_YEAR_FILE.exists(), reason="5-year sample file not on this machine")
@pytest.mark.parametrize("page", PAGES)
def test_pages_with_five_year_file(page):

    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.file_uploader(key="data_file").upload(
        "gallup_q12_survey_5yr.csv", FIVE_YEAR_FILE.read_bytes(), "text/csv"
    ).run()
    at.switch_page(page).run()

    _assert_clean(at, f"5-year {page}")


def test_audit_page_runs_the_audit(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/audit.py").run()

    at.button(key="run_audit").click().run()

    _assert_clean(at, "audit")
    assert at.session_state["audit_result"]["audit"]["checks"]
    assert any("Is the math right?" in h.value for h in at.subheader)


def test_scenarios_page_simulates(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/scenarios.py").run()


    _assert_clean(at, "scenarios")
    assert any("What the models expect" in h.value for h in at.subheader)
    assert any(m.label == "Yearly value" for m in at.metric)


def test_analysis_runs_automatically_without_a_button(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/outcomes.py").run()

    _assert_clean(at, "outcomes, automatic analysis")
    assert at.session_state["uploaded_analysis"] is not None
    assert not [b for b in at.button if b.label == "Run statistical analysis"]


def test_sidebar_guide_links_every_page():

    from src.ui import GLOSSARY, PAGE_GUIDE

    pages = {path for _, path, _ in PAGE_GUIDE}
    assert pages == set(PAGES)
    assert all(meaning.endswith(".") for _, meaning in GLOSSARY)


def test_predictions_monthly_risk_with_demo_data(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.session_state["survival_source"] = "demo"
    at.switch_page("app_pages/predictions.py").run()

    _assert_clean(at, "predictions monthly risk")
    assert any("Monthly leaving risk" in h.value for h in at.subheader)
    assert any(m.label.startswith("Likely leavers, next") for m in at.metric)


def test_data_page_shows_time_series_readiness(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.switch_page("app_pages/data_quality.py").run()

    _assert_clean(at, "data readiness")
    assert any("Ready for time-series prediction?" in h.value for h in at.subheader)


def test_strengths_page_with_demo_data(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.session_state["strengths_source"] = "demo"
    at.switch_page("app_pages/strengths.py").run()

    _assert_clean(at, "strengths demo")
    assert any("Use each team's strengths" in h.value for h in at.subheader)
    assert any("Did strengths coaching work?" in h.value for h in at.subheader)

    at.switch_page("app_pages/teams.py").run()           # the coach now uses strengths
    _assert_clean(at, "teams with strengths")
    assert any("Use the team's strengths" in m.value for m in at.markdown)
