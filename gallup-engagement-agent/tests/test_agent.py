# PROCESS 1 – IMPORTS

import json

import pytest

from src.agent import (
    DEFAULT_DATASET,
    OUTCOME_LABELS,
    get_analysis,
    route_question,
    select_verified_evidence,
)


# PROCESS 2 – SHARED ANALYSIS FIXTURE

@pytest.fixture(scope="session")
def analysis():
    """
    Run the analytical backend once for the entire test session.
    """
    return get_analysis(
        str(DEFAULT_DATASET.resolve())
    )


# PROCESS 3 – ROUTING TESTS

@pytest.mark.parametrize(
    "question, expected_item, expected_outcome, expected_intent",
    [
        (
            "What does Q04 Recognition tell us about turnover?",
            "Q04",
            "turnover",
            "explain",
        ),
        (
            "Should HR prioritize Q07 for absenteeism?",
            "Q07",
            "absence",
            "priority",
        ),
        (
            "Which engagement factors are most strongly associated with turnover?",
            None,
            "turnover",
            "ranking",
        ),
        (
            "How is Learn and Grow related to performance?",
            "Q12",
            "performance",
            "explain",
        ),
    ],
)
def test_question_routing(
    question,
    expected_item,
    expected_outcome,
    expected_intent,
):
    route = route_question(question)

    assert route["item"] == expected_item
    assert route["outcome"] == expected_outcome
    assert route["intent"] == expected_intent


# PROCESS 4 – BUSINESS-FRIENDLY OUTCOME LABELS

def test_outcome_labels():
    assert (
        OUTCOME_LABELS["turnover"]
        == "voluntary turnover"
    )

    assert (
        OUTCOME_LABELS["absence"]
        == "absenteeism"
    )

    assert (
        OUTCOME_LABELS["performance"]
        == "performance"
    )


# PROCESS 5 – Q04 ROBUST FINDING

def test_q04_turnover_is_robust(analysis):

    route = route_question(
        "What does Q04 Recognition tell us about voluntary turnover?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    assert evidence["type"] == "single_finding"
    assert len(evidence["findings"]) == 1

    finding = evidence["findings"][0]

    assert finding["item"] == "Q04"
    assert finding["item_label"] == "Recognition"

    assert finding["outcome"] == "turnover"
    assert (
        finding["outcome_label"]
        == "voluntary turnover"
    )

    assert finding["robustness_status"] == "ROBUST"

    assert (
        finding[
            "priority_recommendation_allowed"
        ]
        is True
    )

    assert finding["sample_size"] == 4024

    assert finding["effect"] == pytest.approx(
        0.6913,
        abs=0.0001,
    )


# PROCESS 6 – Q07 CAUTION FINDING

def test_q07_absence_is_caution(analysis):

    route = route_question(
        "Should HR prioritize Q07 Opinions Count for absenteeism?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    assert evidence["type"] == "single_finding"

    finding = evidence["findings"][0]

    assert finding["item"] == "Q07"
    assert finding["item_label"] == "Opinions Count"

    assert finding["outcome"] == "absence"
    assert (
        finding["outcome_label"]
        == "absenteeism"
    )

    assert finding["robustness_status"] == "CAUTION"

    assert (
        finding[
            "priority_recommendation_allowed"
        ]
        is False
    )

    assert finding["effect"] == pytest.approx(
        0.8124,
        abs=0.0001,
    )

    reasons = " ".join(
        finding["robustness_reasons"]
    )

    assert "93.8%" in reasons


# PROCESS 7 – Q12 PERFORMANCE INSUFFICIENT EVIDENCE

def test_q12_performance_has_insufficient_evidence(
    analysis,
):

    route = route_question(
        "How is Learn and Grow related to performance?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    assert (
        evidence["type"]
        == "insufficient_evidence"
    )

    assert len(evidence["findings"]) == 1

    finding = evidence["findings"][0]

    assert finding["item"] == "Q12"
    assert finding["outcome"] == "performance"

    assert (
        "does not provide statistically reliable evidence"
        in finding["evidence_summary"]
    )


# PROCESS 8 – Q02 TURNOVER INSUFFICIENT EVIDENCE

def test_q02_turnover_has_insufficient_evidence(
    analysis,
):

    route = route_question(
        "Is Q02 associated with voluntary turnover?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    assert (
        evidence["type"]
        == "insufficient_evidence"
    )

    finding = evidence["findings"][0]

    assert finding["item"] == "Q02"

    assert (
        finding["outcome_label"]
        == "voluntary turnover"
    )

    assert (
        "should not currently"
        in finding["decision"].lower()
    )


# PROCESS 9 – TURNOVER RANKING

def test_turnover_ranking_returns_expected_findings(
    analysis,
):

    route = route_question(
        "Which engagement factors are most strongly associated with turnover?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    assert evidence["type"] == "outcome_summary"

    items = {
        finding["item"]
        for finding in evidence["findings"]
    }

    assert items == {
        "Q04",
        "Q05",
        "Q12",
    }

    for finding in evidence["findings"]:
        assert finding["outcome"] == "turnover"
        assert (
            finding["robustness_status"]
            == "ROBUST"
        )


# PROCESS 10 – NO INTERNAL DIAGNOSTIC CODE LEAKAGE

def test_q07_evidence_does_not_expose_internal_codes(
    analysis,
):

    route = route_question(
        "Should HR prioritize Q07 for absenteeism?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    evidence_text = json.dumps(
        evidence,
        default=str,
    )

    forbidden_terms = [
        "X3_TEAM_CONFOUNDING",
        "TEAM_LEVEL_CONFOUNDING",
        "MS_DIF",
        "NR_SMD_unplanned_absence_days_6m",
        "priority_eligible",
        "diagnostic_codes",
    ]

    for term in forbidden_terms:
        assert term not in evidence_text


# PROCESS 11 – NO INTERNAL CLASSIFICATION LEAKAGE

def test_insufficient_evidence_payload_is_business_safe(
    analysis,
):

    route = route_question(
        "Is Q02 associated with voluntary turnover?"
    )

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    finding_text = json.dumps(
        evidence["findings"],
        default=str,
    ).lower()

    assert "nonreportable" not in finding_text
    assert '"reportable"' not in finding_text
    assert "not_reportable" not in finding_text