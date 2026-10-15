# PROCESS 1 – IMPORTS

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.agent import DEFAULT_DATASET
from src.tools import model_spec, stats_tools as st
from src.tools.analytics import MODEL_METADATA, analyze_dataframe
from src.tools.health_check import check_upload
from src.tools.model_spec import (
    SPECS,
    fit_outcome_model,
    method_id,
    resolve_adjustments,
    spec_fingerprint,
)
from src.tools.waves import model_sample

GOLDEN = Path(__file__).parent / "fixtures" / "golden_findings.json"


# PROCESS 2 – FIXTURES

@pytest.fixture(scope="module")
def data():

    path = DEFAULT_DATASET / "agent_data"
    survey = pd.read_csv(path / "survey_responses.csv")
    individual = pd.read_csv(path / "individual_outcomes.csv")

    return survey.merge(individual, on=[c for c in survey.columns if c in individual.columns])


@pytest.fixture(scope="module")
def sample(data):

    rows = model_sample(data)
    return rows[rows["responded"] == 1]


@pytest.fixture(scope="module")
def analysis(data):

    return analyze_dataframe(model_sample(data), dataset_name="golden",
                             health_report=check_upload(data, "golden"))


# PROCESS 3 – EVERY OUTCOME HAS ONE FIXED METHOD

def test_every_outcome_has_a_complete_spec():

    assert set(SPECS) == {"turnover", "absence", "performance"}
    for spec in SPECS.values():
        assert spec.method in {"logit", "negbin", "ols"}
        assert spec.numeric_controls or spec.categorical_controls
        assert spec.cluster == "team_id"


@pytest.mark.parametrize("outcome, original", [
    ("turnover", st.turnover_model),
    ("absence", st.absence_model),
    ("performance", st.performance_model),
])
def test_spec_fit_equals_the_original_model(sample, outcome, original):

    new, old = fit_outcome_model(sample, outcome), original(sample)

    for item in st.ITEMS:
        assert new.params[item] == pytest.approx(old.params[item], abs=1e-12)
        assert new.se[item] == pytest.approx(old.se[item], abs=1e-12)


# PROCESS 4 – ONLY PRE-REGISTERED ADJUSTMENTS

def test_unregistered_controls_are_refused(sample):

    decision = resolve_adjustments("turnover", ["department", "shift_type", "Q99_made_up"], sample)

    assert decision["already_in_model"] == ["department"]
    assert decision["applied"] == ["shift_type"]
    assert decision["refused"] == ["Q99_made_up"]

    with pytest.raises(ValueError):
        fit_outcome_model(sample, "turnover", ("Q99_made_up",))


def test_analysis_records_every_control_decision(analysis):

    decisions = analysis["adjustments"]["control_decisions"]

    assert set(decisions) == set(SPECS)
    for outcome, decision in decisions.items():
        assert set(decision) == {"applied", "already_in_model", "refused", "unavailable"}
        assert analysis["adjustments"]["model_controls_added"][outcome] == decision["applied"]


# PROCESS 5 – EVERY NUMBER TRACES TO ITS METHOD

def test_fingerprint_changes_when_a_method_changes(monkeypatch):

    before = spec_fingerprint()
    changed = dict(SPECS, turnover=replace(SPECS["turnover"], categorical_controls=("department",)))
    monkeypatch.setattr(model_spec, "SPECS", changed)

    assert spec_fingerprint() != before


def test_calculation_ids_carry_the_method_version(analysis):

    for payload in analysis["payloads"]:
        assert payload["method_id"] == method_id(payload["outcome"])
        assert payload["method_id"] == MODEL_METADATA[payload["outcome"]]["method_id"]
        assert spec_fingerprint() in payload["method_id"]


def test_methods_are_described_for_people(analysis):

    for outcome, method in analysis["methods"].items():
        assert method["method"] == SPECS[outcome].method_name
        assert method["controls"]
        assert method["method_id"] == method_id(outcome)


# PROCESS 6 – VERIFIED, REPRODUCIBLE NUMBERS

def test_same_data_gives_identical_numbers(data, analysis):

    again = analyze_dataframe(model_sample(data), dataset_name="golden",
                              health_report=check_upload(data, "golden"))

    pd.testing.assert_frame_equal(analysis["results"].drop(columns=["diagnostic_codes", "diagnostic_reasons"]),
                                  again["results"].drop(columns=["diagnostic_codes", "diagnostic_reasons"]))


def test_numbers_match_the_golden_snapshot(analysis):
    """A change to a method or to the code that moves any number fails here.
    If the change is intended, regenerate tests/fixtures/golden_findings.json."""

    golden = json.loads(GOLDEN.read_text())

    assert golden["fingerprint"] == spec_fingerprint(), "method changed: regenerate the snapshot"
    assert golden["sample_size"] == analysis["sample_size"]

    results = analysis["results"].set_index(["outcome", "item"])
    for row in golden["results"]:
        current = results.loc[(row["outcome"], row["item"])]
        assert current["effect"] == pytest.approx(row["effect"], abs=1e-8)
        assert current["ci_low"] == pytest.approx(row["ci_low"], abs=1e-8)
        assert current["ci_high"] == pytest.approx(row["ci_high"], abs=1e-8)
        assert bool(current["reportable"]) == row["reportable"]
        assert current["robustness_status"] == row["robustness_status"]
