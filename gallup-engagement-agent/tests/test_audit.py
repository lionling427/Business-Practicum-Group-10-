# PROCESS 1 – IMPORTS

import json

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools import audit as audit_module
from src.tools.analytics import analyze_dataframe
from src.tools.audit import (
    AREAS,
    audit_markdown,
    audit_sample,
    check_independent_recompute,
    check_reported_numbers,
    data_fingerprint,
    run_audit,
)
from src.tools.health_check import check_upload
from src.tools.waves import model_sample


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")
    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


@pytest.fixture(scope="module")
def health(data):

    return check_upload(data, "dev")


@pytest.fixture(scope="module")
def analysis(data, health):

    return analyze_dataframe(model_sample(data), dataset_name="dev", health_report=health)


@pytest.fixture(scope="module")
def audit(data, analysis, health):

    return run_audit(data, analysis, health)


# PROCESS 3 – EVERY CHECK IS READABLE WITHOUT GUESSING

def test_every_check_has_a_status_result_and_reason(audit):

    assert audit["checks"]
    for check in audit["checks"]:
        assert check["area"] in AREAS
        assert check["status"] in {"pass", "warn", "fail", "skip"}
        assert check["result"].endswith((".", ")"))
        assert check["why"]


def test_every_area_is_covered(audit):

    assert {c["area"] for c in audit["checks"]} == set(AREAS)


def test_audit_sample_is_exactly_the_model_sample(data, analysis):

    assert len(audit_sample(data)) == analysis["sample_size"]


# PROCESS 4 – IS THE MATH RIGHT?

def test_math_matches_an_independent_library(data, analysis):

    pytest.importorskip("statsmodels")
    check = check_independent_recompute(audit_sample(data), analysis)

    assert check["status"] in {"pass", "warn"}
    assert check["details"]["match"].all()


def test_a_wrong_number_is_caught(data, analysis):
    """Tamper with one reported effect: the audit must fail."""
    tampered = dict(analysis, results=analysis["results"].copy())
    tampered["results"].loc[0, "effect"] *= 1.05

    check = check_reported_numbers(audit_sample(data), tampered)

    assert check["status"] == "fail"


def test_broken_check_fails_the_verdict(data, analysis, health, monkeypatch):

    def broken(*args, **kwargs):
        raise RuntimeError("simulated")

    monkeypatch.setattr(audit_module, "check_overlap", broken)
    result = run_audit(data, analysis, health)

    assert result["verdict"][0] == "fail"
    assert any("simulated" in c["result"] for c in result["checks"])


# PROCESS 5 – STABILITY AND MONEY

def test_money_range_brackets_the_estimate(audit):

    money = next(c for c in audit["checks"] if c["check"] == "Range around each dollar figure")
    table = money["details"]

    assert (table["cautious"] <= table["estimate"]).all()
    assert (table["estimate"] <= table["optimistic"]).all()


def test_cost_sensitivity_covers_every_scenario(audit):

    check = next(c for c in audit["checks"] if c["check"].startswith("Advice holds"))

    assert len(check["details"]) == len(audit_module.COST_SCENARIOS)


def test_replication_uses_the_other_survey(audit):

    check = next(c for c in audit["checks"] if c["check"] == "Same finding in another survey")

    assert check["status"] != "skip"
    assert set(check["details"]["survey"]) == {"wave 1"}


# PROCESS 6 – REPRODUCIBLE RECORD

def test_fingerprint_changes_when_data_changes(data):

    changed = data.copy()
    changed.loc[changed.index[0], "Q01"] = 6 - changed.loc[changed.index[0], "Q01"]

    assert data_fingerprint(data) == data_fingerprint(data.copy())
    assert data_fingerprint(data) != data_fingerprint(changed)


def test_audit_pack_is_json_and_includes_sign_off(data, analysis, audit):

    sign_off = {"name": "Reviewer", "role": "Analyst", "decision": "Approved to share",
                "note": "", "date": "2026-10-14"}
    record = audit_module.audit_record(data, analysis, audit["checks"], audit["verdict"], sign_off)

    loaded = json.loads(json.dumps(record, default=str))
    assert loaded["sign_off"]["name"] == "Reviewer"
    assert loaded["method_fingerprint"] and loaded["data_fingerprint"]
    assert len(loaded["checks"]) == len(audit["checks"])

    report = audit_markdown(audit, sign_off)
    assert "## Reviewer sign-off" in report and "| Status | Check | Result |" in report
