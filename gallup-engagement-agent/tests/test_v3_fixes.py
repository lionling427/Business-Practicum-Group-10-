"""Regression tests for the v3 fixes, merged into the Oct 17 version (v4).
Run: python -m pytest tests/test_v3_fixes.py -v"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import openai
import pandas as pd
import pytest

import src.agent as agent
from src.tools import analytics
from src.tools.health_check import check_dataframe, check_folder

DEV = Path(__file__).resolve().parents[1] / "data" / "dev" / "gallup_v2_5K"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dev_analysis():
    return analytics.analyze_dataset(DEV)


def _copy_dev(tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(DEV / "agent_data", target / "agent_data")
    return target


def _merged_upload():
    data = DEV / "agent_data"
    survey = pd.read_csv(data / "survey_responses.csv")
    indiv = pd.read_csv(data / "individual_outcomes.csv")
    return survey.merge(indiv, on=["wave", "employee_id"])


def _no_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the language model must not be called")
    monkeypatch.setattr(agent, "generate_business_answer", fail)


def _blocked(analysis, outcomes):
    """Same analysis, with some outcomes blocked by the health report."""
    allowed = [o for o in analytics.ALL_OUTCOMES if o not in outcomes]
    report = {
        "summary": {"analysis_allowed": {o: o in allowed for o in analytics.ALL_OUTCOMES},
                    "blocked_outcomes": ["all"] if not allowed else list(outcomes)},
        "checks": [{"check": "Test failure", "detail": "planted", "status": "FAIL"}],
    }
    return {
        **analysis,
        "health_report": report,
        "allowed_outcomes": allowed,
        "blocked_outcomes": list(outcomes),
        "payloads": [p for p in analysis["payloads"] if p["outcome"] in allowed],
        "special_findings": [s for s in analysis["special_findings"] if s["outcome"] in allowed],
    }


# ---------------------------------------------------------------------------
# Fix 1: health gate also applies when the question names no outcome
# ---------------------------------------------------------------------------
def test_gate_refuses_question_without_outcome_when_all_blocked(dev_analysis, monkeypatch):
    _no_llm(monkeypatch)
    analysis = _blocked(dev_analysis, analytics.ALL_OUTCOMES)
    result = agent.ask_agent("What should we focus on?", analysis_override=analysis,
                             include_dashboard=False, return_debug=True)
    assert result["blocked"] is True
    assert result["evidence"] is None
    assert "can't answer" in result["answer"]


def test_gate_refuses_blocked_outcome_and_lists_allowed(dev_analysis, monkeypatch):
    _no_llm(monkeypatch)
    analysis = _blocked(dev_analysis, ["performance"])
    answer = agent.ask_agent("Which items are linked to performance?", analysis_override=analysis,
                             include_dashboard=False)
    assert "blocked" in answer and "voluntary turnover" in answer


def test_partial_block_excludes_outcome_and_says_so(dev_analysis, monkeypatch):
    captured = {}

    def fake_answer(question, route, evidence):
        captured["evidence"] = evidence
        return {"answer": "ok", "finish_reason": "stop", "usage": None}

    monkeypatch.setattr(agent, "generate_business_answer", fake_answer)
    analysis = _blocked(dev_analysis, ["performance"])
    agent.ask_agent("Give me a summary of everything.", analysis_override=analysis, include_dashboard=False)
    evidence = captured["evidence"]
    assert all(f["outcome"] != "performance" for f in evidence["findings"])
    assert evidence["excluded_by_data_health_checks"]["outcomes"] == ["performance"]


def test_outcome_is_allowed_no_longer_true_for_none(dev_analysis):
    assert agent.outcome_is_allowed(_blocked(dev_analysis, analytics.ALL_OUTCOMES), None) is False
    assert agent.outcome_is_allowed(dev_analysis, None) is True


# ---------------------------------------------------------------------------
# Fix 2: health check before models; no crash on dirty data
# ---------------------------------------------------------------------------
def test_outcome_column_in_survey_file_blocks_instead_of_crashing(tmp_path):
    folder = _copy_dev(tmp_path)
    survey_path = folder / "agent_data" / "survey_responses.csv"
    survey = pd.read_csv(survey_path)
    survey["performance_rating"] = 3          # dirty_A style leakage column
    survey.to_csv(survey_path, index=False)

    with pytest.raises(analytics.HealthCheckBlocked):   # v2: KeyError 'performance_rating'
        analytics.analyze_dataset(folder)


def test_date_error_blocks_only_its_outcome(tmp_path):
    folder = _copy_dev(tmp_path)
    path = folder / "agent_data" / "individual_outcomes.csv"
    indiv = pd.read_csv(path)
    indiv.loc[:9, "performance_review_date"] = "2000-01-01"
    indiv.to_csv(path, index=False)

    from src.tools.health_check import check_folder as run_folder
    run_folder(str(folder))                     # write the folder's health report
    analysis = analytics.analyze_dataset(folder)
    assert analysis["status"] == "PARTIAL"
    assert analysis["blocked_outcomes"] == ["performance"]
    assert {p["outcome"] for p in analysis["payloads"]} <= {"turnover", "absence"}
    assert set(analysis["results"]["outcome"]) == {"turnover", "absence"}


# ---------------------------------------------------------------------------
# Fix 3: uploads get the real health check, not a placeholder
# ---------------------------------------------------------------------------
def test_upload_health_check_matches_folder_check():
    upload = check_dataframe(_merged_upload(), "upload")
    folder = check_folder(str(DEV), write=False)
    assert upload["summary"]["checks_run"] >= 40
    assert upload["summary"]["fail"] == folder["summary"]["fail"] == 0
    codes = {c["code"]: c["status"] for c in upload["checks"]}
    assert codes["NR_LEAVERS"] == "WARN"            # X7 caveat now reaches uploads
    assert codes["COV_SMALL_TEAMS"] == "WARN"       # X8 small-team suppression


def test_upload_with_bad_scores_is_blocked():
    df = _merged_upload()
    df.loc[df.index[:5], "Q03"] = 9
    with pytest.raises(analytics.HealthCheckBlocked):
        analytics.analyze_dataframe(df, "bad_upload")


def test_dev_dataset_blocked_question_is_refused_without_llm(monkeypatch):
    _no_llm(monkeypatch)
    def stop(*args, **kwargs):
        raise analytics.HealthCheckBlocked("Analysis stopped: planted.")
    monkeypatch.setattr(agent, "get_analysis", stop)
    result = agent.ask_agent("What should we focus on?", include_dashboard=False, return_debug=True)
    assert result["blocked"] is True and "can't answer" in result["answer"]


def test_upload_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="missing columns"):
        analytics.analyze_dataframe(_merged_upload().drop(columns=["shift_type"]))


# ---------------------------------------------------------------------------
# Fix 4 and 5: threshold (X4) and new-hire subgroup (X5) are reported
# ---------------------------------------------------------------------------
def test_threshold_pattern_reported(dev_analysis):
    found = [s for s in dev_analysis["special_findings"] if s["type"] == "threshold"]
    assert found and found[0]["item"] == "Q01" and found[0]["outcome"] == "absence"
    assert found[0]["effect"] > 1.10


def test_new_hire_pattern_reported(dev_analysis):
    found = [s for s in dev_analysis["special_findings"] if s["type"] == "subgroup_difference"]
    q04 = [s for s in found if s["item"] == "Q04"]
    assert q04, "Q04 x new hire should be found"
    assert q04[0]["odds_ratio_new_hires"] < q04[0]["odds_ratio_others"] < 1


def test_patterns_reach_the_agent_evidence(dev_analysis):
    route = {"item": "Q04", "outcome": "turnover", "intent": "explain"}
    evidence = agent.add_patterns_and_exclusions(
        agent.select_verified_evidence(dev_analysis, route), dev_analysis, route)
    assert any(p["type"] == "subgroup_difference" for p in evidence["threshold_and_subgroup_patterns"])


# ---------------------------------------------------------------------------
# Fix 6: API retry
# ---------------------------------------------------------------------------
def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                    finish_reason="stop")], usage=None)


def _status_error(cls, code):
    """Build an openai status error the way the real SDK does (needs an httpx response)."""
    try:
        import httpx
        request = httpx.Request("POST", "https://example.invalid/chat/completions")
        return cls(f"HTTP {code}", response=httpx.Response(code, request=request), body=None)
    except ImportError:  # minimal stand-in package without httpx
        return cls(f"HTTP {code}")


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return _response(item)


def test_retry_on_rate_limit_then_succeed():
    waits = []
    client = FakeClient([_status_error(openai.RateLimitError, 429), _status_error(openai.RateLimitError, 429), "answer"])
    _, answer = agent.call_with_retry(client, {}, sleep=waits.append)
    assert answer == "answer" and client.calls == 3 and waits == [2.0, 4.0]


def test_retry_on_empty_answer():
    client = FakeClient(["", "answer"])
    _, answer = agent.call_with_retry(client, {}, sleep=lambda s: None)
    assert answer == "answer"


def test_gives_up_after_three_retries():
    client = FakeClient([_status_error(openai.RateLimitError, 429)] * 4)
    with pytest.raises(openai.RateLimitError):
        agent.call_with_retry(client, {}, sleep=lambda s: None)
    assert client.calls == 4


def test_no_retry_on_auth_error():
    client = FakeClient([_status_error(openai.AuthenticationError, 401), "answer"])
    with pytest.raises(openai.AuthenticationError):
        agent.call_with_retry(client, {}, sleep=lambda s: None)
    assert client.calls == 1
