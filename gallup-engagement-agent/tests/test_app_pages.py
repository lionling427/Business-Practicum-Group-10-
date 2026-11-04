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


def _open(at, path, view=None):
    """Open the sidebar page that shows a page file (src/layout.py), on its view."""
    from src.layout import locate

    spot = locate(path, view)
    if spot["view_key"]:
        at.session_state[spot["view_key"]] = spot["view"]
    # The sidebar pages are functions (src/hub.py), which AppTest.switch_page()
    # cannot open by file; pick the registered page by its URL name instead.
    for page_hash, info in at._registered_pages.items():
        if spot["page"] in (info.get("url_pathname"), info.get("page_name")):
            at._page_hash = page_hash
            return at
    raise ValueError(f"No sidebar page shows {path}")


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
    _open(at, page).run()

    assert not at.exception, [e.value for e in at.exception]


# PROCESS 5 – PAGES WITH DATA

@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_uploaded_data(csv_bytes, page):

    at = _app_with_upload(csv_bytes)
    _open(at, page).run()

    _assert_clean(at, page)


def test_analysis_runs_and_unlocks_findings(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/action_story.py").run()          # Strategy: where to act

    _assert_clean(at, "strategy after analysis")
    assert at.session_state["uploaded_analysis"] is not None
    assert any("What drives results" in m.value for m in at.markdown)

    _open(at, "app_pages/outcomes.py", view="Analytical checks").run()   # Trust
    _assert_clean(at, "analytical checks")
    assert any("Analytical traps" in h.value for h in at.subheader)
    assert "with_findings" in at.session_state["trap_checks"]

    # Other pages pick up the analysis.
    for page in ["app_pages/overview.py", "app_pages/teams.py"]:
        _open(at, page).run()
        _assert_clean(at, page)


def test_overview_scope_switch(csv_bytes):

    at = _app_with_upload(csv_bytes)

    at.segmented_control(key="overview_scope").set_value("team_id").run()

    _assert_clean(at, "overview team view")
    assert any("percentile" in m.value for m in at.markdown)


def test_change_analysis_runs(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/change.py").run()


    _assert_clean(at, "change after analysis")
    # "Is change linked to results?" is switched off in src/layout.py; the change
    # charts for the same people are shown.
    assert any("Everyone vs the same people" in h.value for h in at.subheader)
    assert any("What changed" in h.value for h in at.subheader)


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
    _open(at, "app_pages/teams.py").run()

    _assert_clean(at, "teams with risk")
    assert any("Manager coach" in h.value for h in at.subheader)
    assert any("never saw" in c.value for c in at.caption)      # test result, in short
    assert len(at.dataframe) >= 1      # the watchlist

    _open(at, "app_pages/teams.py", view="Prediction tests").run()   # Trust
    _assert_clean(at, "prediction tests")
    assert any("passed its test" in m.value for m in at.markdown)


def test_predictions_page_never_lists_employees(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/predictions.py").run()

    _assert_clean(at, "predictions")

    for frame in at.dataframe:
        assert "employee_id" not in frame.value.columns


def test_recommendations_page_builds_plans(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/recommendations.py").run()


    _assert_clean(at, "recommendations")
    assert any("Plan for each team" in h.value for h in at.subheader)
    assert any("Start with" in m.value or "Protect what works" in m.value for m in at.markdown)
    assert any("Take it to Gallup action planning" in e.label for e in at.expander)
    assert any("element-great-managing" in m.value for m in at.markdown)


def test_financials_page_values_engagement(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/financials.py").run()


    _assert_clean(at, "financials")
    assert any("What it costs today" in h.value for h in at.subheader)
    assert any("what is it worth" in m.value for m in at.markdown)
    assert any(m.label == "We can save per year" for m in at.metric)


FIVE_YEAR_FILE = __import__("pathlib").Path.home() / "Downloads" / "gallup_q12_survey_5yr.csv"


@pytest.mark.skipif(not FIVE_YEAR_FILE.exists(), reason="5-year sample file not on this machine")
@pytest.mark.parametrize("page", PAGES)
def test_pages_with_five_year_file(page):

    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.file_uploader(key="data_file").upload(
        "gallup_q12_survey_5yr.csv", FIVE_YEAR_FILE.read_bytes(), "text/csv"
    ).run()
    _open(at, page).run()

    _assert_clean(at, f"5-year {page}")


def test_audit_page_runs_the_audit(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/audit.py").run()

    at.button(key="run_audit").click().run()

    _assert_clean(at, "audit")
    assert at.session_state["audit_result"]["audit"]["checks"]
    assert any("Is the math right?" in h.value for h in at.subheader)


def test_scenarios_page_simulates(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/scenarios.py").run()


    _assert_clean(at, "scenarios")
    assert any("What the models expect" in h.value for h in at.subheader)
    assert any(m.label == "Yearly value" for m in at.metric)


def test_analysis_runs_automatically_without_a_button(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/outcomes.py").run()

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
    _open(at, "app_pages/predictions.py").run()

    _assert_clean(at, "predictions monthly risk")
    assert any("Monthly leaving risk" in h.value for h in at.subheader)
    assert any(m.label.startswith("Likely leavers, next") for m in at.metric)


def test_data_page_shows_time_series_readiness(csv_bytes):

    at = _app_with_upload(csv_bytes)
    _open(at, "app_pages/data_quality.py").run()

    _assert_clean(at, "data readiness")
    assert any("Ready for time-series prediction?" in h.value for h in at.subheader)


def test_strengths_page_with_demo_data(csv_bytes):

    at = _app_with_upload(csv_bytes)
    at.session_state["strengths_source"] = "demo"
    _open(at, "app_pages/strengths.py").run()

    _assert_clean(at, "strengths demo")
    assert any("Use each team's strengths" in h.value for h in at.subheader)
    assert any("Did strengths coaching work?" in h.value for h in at.subheader)

    _open(at, "app_pages/teams.py").run()           # the coach now uses strengths
    _assert_clean(at, "teams with strengths")
    assert any("Use the team's strengths" in m.value for m in at.markdown)
