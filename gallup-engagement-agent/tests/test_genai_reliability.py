"""Focused, offline checks for generated-answer acceptance."""

import pytest

from src.tools.agent_context import (
    UnverifiedAnswerError,
    check_high_risk_claims,
    check_numbers,
    generate_verified_reply,
    public_answer_failure,
)


def result(answer):
    return {"answer": answer, "evidence": {"financials": {"cost": 9_700_000,
            "share": 0.22, "employees": 850}}, "sections": ["financials"]}


def test_verified_answer_and_equivalent_formatting():
    replies = []
    def generate(prompt):
        replies.append(prompt)
        return result("Financials estimates $9.7M, 22%, and 850 employees.")
    assert generate_verified_reply("What is the cost?", generate, ["financials"])["answer"]
    assert len(replies) == 1
    assert check_numbers("$9.7M, 22%, 850 employees", result("")["evidence"]) == []


def test_one_correction_then_acceptance():
    prompts = []
    answers = iter(["The cost is $47M.", "The estimated cost is $9.7M."])
    def generate(prompt):
        prompts.append(prompt)
        return result(next(answers))
    assert "$9.7M" in generate_verified_reply("Cost?", generate, ["financials"])["answer"]
    assert len(prompts) == 2
    assert "CORRECTION" in prompts[1]


def test_second_failure_withholds_answer():
    prompts = []
    def generate(prompt):
        prompts.append(prompt)
        return result("The cost is $47M.")
    with pytest.raises(UnverifiedAnswerError):
        generate_verified_reply("Cost?", generate, ["financials"])
    assert len(prompts) == 2


def test_question_number_is_not_evidence():
    assert check_numbers("The result is 47%.", result("")["evidence"], "Is it 47%?") == [47.0]


@pytest.mark.parametrize("error", [TimeoutError("secret endpoint"),
                                    RuntimeError("secret provider detail")])
def test_provider_failure_is_non_technical(error):
    message = public_answer_failure(error)
    assert "temporarily unavailable" in message
    assert "secret" not in message


def test_verification_failure_is_non_technical():
    message = public_answer_failure(UnverifiedAnswerError("sensitive draft"))
    assert "could not be verified" in message
    assert "sensitive" not in message


@pytest.mark.parametrize("answer", [
    "Recognition causes turnover.",
    "This guarantees savings of $9.7M.",
    "The measured profit is $9.7M.",
    "Employee 123 will resign.",
    "The Overview page shows this.",
])
def test_high_risk_claims_rejected(answer):
    assert check_high_risk_claims(answer, ["financials"])
