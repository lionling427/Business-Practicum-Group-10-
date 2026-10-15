# PROCESS 1 – IMPORTS

import numpy as np
import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools import stats_tools as st
from src.tools.agent_context import build_context, route_sections
from src.tools.traps import (
    department_trap,
    manager_trap,
    nonresponse_trap,
    run_trap_checks,
    trap_summary,
)


# PROCESS 2 – SYNTHETIC WORKFORCE WITH PLANTED TRAPS

def _workforce(seed=7, n=4000, simpson=False, team_effect=False, leavers_skip=False):
    """Clean data by default. Each flag plants one trap:

    simpson       departments with higher Q05 have LOWER performance, while
                  within each department higher Q05 means HIGHER performance;
    team_effect   teams with higher Q06 have more absence, but within a team Q06
                  has no link (a manager effect, not a people effect);
    leavers_skip  leavers answer the survey far less often."""
    rng = np.random.default_rng(seed)
    departments = [f"D{k}" for k in range(6)]
    team = rng.integers(0, 80, n)
    dept_of_team = np.arange(80) % 6
    dept = dept_of_team[team]

    df = pd.DataFrame({
        "employee_id": np.arange(n), "wave": 1, "team_id": [f"T{t}" for t in team],
        "department": [departments[d] for d in dept],
        "role_level": rng.choice(["IC", "Lead", "Manager"], n),
        "pay_band": rng.choice(["B1", "B2", "B3"], n),
        "shift_type": rng.choice(["Day", "Night"], n),
        "tenure_years": rng.gamma(2.0, 2.0, n),
        "prior_performance_rating": rng.normal(3.5, 0.6, n),
    })
    for item in st.ITEMS:
        df[item] = rng.integers(1, 6, n).astype(float)

    dept_shift = np.linspace(-1.2, 1.2, 6)[dept]
    team_shift = rng.normal(0, 1, 80)[team]
    if simpson:
        df["Q05"] = np.clip(np.round(3 + dept_shift + rng.normal(0, 0.8, n)), 1, 5)
    if team_effect:
        df["Q06"] = np.clip(np.round(3 + team_shift + rng.normal(0, 0.7, n)), 1, 5)

    perf = 3.5 + 0.3 * (df["Q05"] - 3 - dept_shift) - (0.9 * dept_shift if simpson else 0)
    df["performance_rating"] = perf + rng.normal(0, 0.5, n)
    df["unplanned_absence_days_6m"] = rng.poisson(
        np.exp(1.0 + (0.5 * team_shift if team_effect else 0)))
    df["left_voluntarily_12m"] = (rng.random(n) < 0.15).astype(int)
    p_respond = np.where((df["left_voluntarily_12m"] == 1) & leavers_skip, 0.4, 0.85)
    df["responded"] = (rng.random(n) < p_respond).astype(int)
    return df


def _verdict(table, outcome, item):
    return table.set_index(["outcome", "item"]).loc[(outcome, item), "verdict"]


# PROCESS 3 – DEPARTMENT TRAP (SIMPSON'S PARADOX)

def test_planted_department_reversal_is_found():

    sample = _workforce(simpson=True).query("responded == 1")
    trap = department_trap(sample)

    assert trap["status"] == "trap"
    assert _verdict(trap["table"], "performance", "Q05") == "flip"
    assert ("performance", "Q05") in trap["affected"]
    assert any("Q05" in p and "reverses" in p for p in trap["points"])


def test_clean_link_is_not_flagged():

    sample = _workforce().query("responded == 1")
    trap = department_trap(sample)

    assert _verdict(trap["table"], "performance", "Q05") == "ok"
    assert not (trap["table"]["verdict"] == "flip").any()


# PROCESS 4 – MANAGER (TEAM) TRAP

def test_planted_manager_effect_is_found():

    sample = _workforce(team_effect=True).query("responded == 1")
    trap = manager_trap(sample)

    assert trap["status"] in {"caution", "trap"}
    assert _verdict(trap["table"], "absence", "Q06") in {"shrink", "flip"}
    assert trap["icc"]["absence"] > trap["icc"]["turnover"]


def test_manager_trap_needs_teams():

    sample = _workforce().query("responded == 1").drop(columns="team_id")

    assert manager_trap(sample)["status"] == "cannot_test"


# PROCESS 5 – NONRESPONSE TRAP

def test_leavers_skipping_the_survey_is_found():

    trap = nonresponse_trap(_workforce(leavers_skip=True))

    assert trap["status"] == "caution"
    assert any("Leavers answered less" in p for p in trap["points"])
    rates = trap["table"].set_index("measure")
    assert rates.loc["voluntary turnover", "non_respondents"] > rates.loc["voluntary turnover", "respondents"]


def test_balanced_response_is_clear():

    assert nonresponse_trap(_workforce())["status"] == "clear"


def test_respondents_only_file_cannot_be_tested():

    respondents = _workforce().query("responded == 1")

    trap = nonresponse_trap(respondents)

    assert trap["status"] == "cannot_test"
    assert "responded = 1 or 0" in trap["headline"]


# PROCESS 6 – ALL THREE ON THE REAL DEV FILE, AND THE AGENT SEES THEM

@pytest.fixture(scope="module")
def dev_data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


def test_dev_file_runs_every_trap(dev_data):

    traps = run_trap_checks(dev_data)

    for name in ("department", "manager", "nonresponse"):
        assert traps[name]["status"] in {"clear", "caution", "trap", "cannot_test"}
        assert traps[name]["headline"]
    assert len(trap_summary(traps)) == 3


def test_trapped_reported_findings_are_named(dev_data):

    traps = run_trap_checks(dev_data)
    affected = traps["manager"]["affected"][:1]
    findings = pd.DataFrame(affected, columns=["outcome", "item"])

    marked = run_trap_checks(dev_data, findings)

    assert marked["manager"]["reported_affected"] == affected
    if affected:
        assert any(p.startswith("Reported findings this affects") for p in marked["manager"]["points"])


def test_agent_gets_the_trap_checks(dev_data):

    assert "traps" in route_sections("Can I trust these results, or is there nonresponse bias?")

    context = build_context(dev_data, sections=["traps"])

    assert len(context["traps"]["checks"]) == 3
    assert "note" in context["traps"]
