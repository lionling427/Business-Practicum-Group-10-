# PROCESS 1 – IMPORTS

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import openai
from openai import AzureOpenAI

from src.tools.analytics import (
    ITEM_LABELS,
    HealthCheckBlocked,
    allowed_outcomes as health_allowed_outcomes,
    analyze_dataset,
    blocking_reasons,
)


# PROCESS 2 – CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "dev"
    / "gallup_v2_5K"
)


def load_env_file(path=PROJECT_ROOT / ".env"):
    """
    Load KEY=VALUE lines from the project's .env file (git-ignored).
    Variables already set in the environment take precedence.
    """

    if not path.exists():
        return

    for line in path.read_text().splitlines():

        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("\"'")

        if key and not os.environ.get(key):
            os.environ[key] = value


load_env_file()

AZURE_RESOURCE = os.getenv(
    "GW_AZURE_OPENAI_RESOURCE",
    "gw-sb-10",
)

AZURE_DEPLOYMENT = os.getenv(
    "GW_AZURE_OPENAI_DEPLOYMENT",
    "gpt-5-nano",
)

AZURE_API_VERSION = (
    "2024-12-01-preview"
)


# PROCESS 3 – OUTCOME ALIASES

OUTCOME_ALIASES = {
    "turnover": [
        "turnover",
        "attrition",
        "resignation",
        "resign",
        "leaving",
        "leave",
        "retention",
        "employee exit",
        "voluntary turnover",
    ],

    "absence": [
        "absence",
        "absenteeism",
        "absent",
        "unplanned absence",
        "absence rate",
    ],

    "performance": [
        "performance",
        "performance rating",
        "rating",
        "employee performance",
    ],
}

# PROCESS 3A – BUSINESS-FRIENDLY OUTCOME LABELS

OUTCOME_LABELS = {
    "turnover": "voluntary turnover",
    "absence": "absenteeism",
    "performance": "performance",
}

# PROCESS 4 – Q12 ITEM ALIASES

ITEM_ALIASES = {
    "Q01": [
        "expectations",
        "know what is expected",
        "what is expected",
    ],

    "Q02": [
        "materials and equipment",
        "materials",
        "equipment",
    ],

    "Q03": [
        "opportunity to do best",
        "do what i do best",
        "do best",
    ],

    "Q04": [
        "recognition",
        "recognition or praise",
        "praise",
    ],

    "Q05": [
        "someone cares",
        "cares about me",
        "care at work",
    ],

    "Q06": [
        "development",
        "encourages development",
        "encourage development",
    ],

    "Q07": [
        "opinions count",
        "opinion counts",
        "opinions seem to count",
    ],

    "Q08": [
        "mission and purpose",
        "mission",
        "purpose",
    ],

    "Q09": [
        "commitment to quality",
        "quality",
        "quality work",
    ],

    "Q10": [
        "best friend at work",
        "best friend",
    ],

    "Q11": [
        "progress",
        "talked about progress",
    ],

    "Q12": [
        "learn and grow",
        "learning and growth",
        "opportunities to learn",
        "grow",
    ],
}


# PROCESS 5 – QUESTION INTENT

def detect_intent(question: str) -> str:
    q = question.lower()

    priority_words = [
        "prioritize",
        "priority",
        "recommend",
        "recommendation",
        "focus on",
        "action",
        "what should we do",
        "should hr",
    ]

    ranking_words = [
        "strongest",
        "most strongly associated",
        "top",
        "most important",
        "which factors",
        "which engagement factors",
        "which items",
        "rank",
        "ranking",
    ]

    if any(
        word in q
        for word in priority_words
    ):
        return "priority"

    if any(
        word in q
        for word in ranking_words
    ):
        return "ranking"

    return "explain"


# PROCESS 6 – DETECT OUTCOME

def detect_outcome(
    question: str,
) -> str | None:

    q = question.lower()

    scores = {}

    for outcome, aliases in (
        OUTCOME_ALIASES.items()
    ):

        score = sum(
            1
            for alias in aliases
            if alias in q
        )

        if score > 0:
            scores[outcome] = score

    if not scores:
        return None

    return max(
        scores,
        key=scores.get,
    )


# PROCESS 7 – DETECT Q12 ITEM

def detect_item(
    question: str,
) -> str | None:

    q = question.lower()

    # First preference:
    # explicitly written Q1, Q01 ... Q12
    match = re.search(
        r"\bq\s*0?(1[0-2]|[1-9])\b",
        q,
    )

    if match:
        number = int(
            match.group(1)
        )

        return f"Q{number:02d}"

    # Otherwise detect item name
    scores = {}

    for item, aliases in (
        ITEM_ALIASES.items()
    ):

        score = sum(
            1
            for alias in aliases
            if alias in q
        )

        if score > 0:
            scores[item] = score

    if not scores:
        return None

    return max(
        scores,
        key=scores.get,
    )


# PROCESS 8 – ROUTE QUESTION

def route_question(
    question: str,
) -> dict:

    return {
        "question":
            question,

        "item":
            detect_item(
                question
            ),

        "outcome":
            detect_outcome(
                question
            ),

        "intent":
            detect_intent(
                question
            ),

        "question_scope":
            "overall",
    }


# PROCESS 9 – CACHE ANALYTICAL ENGINE

@lru_cache(maxsize=4)
def get_analysis(
    dataset_path: str,
):
    return analyze_dataset(
        dataset_path
    )


# PROCESS 9A – DASHBOARD DATA FOR THE DEVELOPMENT DATASET
#
# Without an upload, the agent still knows trends, team risk, recommendations
# and costs: it loads the development dataset and runs the same pipeline the
# app runs for uploads (model wave, change-over-time models).

@lru_cache(maxsize=2)
def load_dev_frame(dataset_path: str):
    import pandas as pd

    from src.tools.longitudinal import derive_wave

    data_path = Path(dataset_path) / "agent_data"
    survey = pd.read_csv(data_path / "survey_responses.csv")
    individual = pd.read_csv(data_path / "individual_outcomes.csv")
    keys = [c for c in survey.columns if c in individual.columns]

    return derive_wave(survey.merge(individual, on=keys))


@lru_cache(maxsize=2)
def get_dev_dashboard(dataset_path: str):
    """(frame, analysis, change_analysis) for the development dataset."""
    from src.tools.analytics import analyze_dataframe
    from src.tools.longitudinal import analyze_change, survey_waves
    from src.tools.waves import model_sample

    from src.tools.health_check import check_upload

    frame = load_dev_frame(dataset_path)
    analysis = analyze_dataframe(
        model_sample(frame),
        dataset_name=Path(dataset_path).name,
        health_report=check_upload(frame, Path(dataset_path).name),
    )
    change = None
    if len(survey_waves(frame)) >= 2:
        try:
            change = analyze_change(frame)
        except ValueError:
            change = None

    return frame, analysis, change


# PROCESS 10 – HEALTH GATE

# v3 fix 1, merged into the Oct 17 version:
# - every outcome blocked          -> refuse, whatever the question
# - the asked outcome is blocked   -> refuse that outcome, name what is still available
# - no outcome named, some blocked -> answer from allowed outcomes only, say which were excluded
# The analysis pipeline raises HealthCheckBlocked when everything is blocked;
# ask_agent turns that into the same polite refusal (no language-model call).

BLOCKED_ANALYSIS = {"allowed_outcomes": [], "blocked_outcomes": ["all"], "payloads": []}


def analysis_allowed_outcomes(analysis):
    if analysis.get("allowed_outcomes") is not None:
        return list(analysis["allowed_outcomes"])
    return health_allowed_outcomes(analysis["health_report"])


def analysis_blocked_outcomes(analysis):
    if analysis.get("blocked_outcomes") is not None:
        return list(analysis["blocked_outcomes"])
    return [o for o in OUTCOME_LABELS if o not in health_allowed_outcomes(analysis["health_report"])]


def outcome_is_allowed(
    analysis,
    outcome,
):
    """True when the outcome may be interpreted. With no outcome named, true only
    when at least one outcome is allowed (the old version returned True here)."""

    allowed = analysis_allowed_outcomes(analysis)

    if outcome is None:
        return bool(allowed)

    return outcome in allowed


def health_gate(analysis, outcome):
    """None when the question may continue, otherwise the refusal text."""

    blocked = analysis_blocked_outcomes(analysis)

    if not analysis_allowed_outcomes(analysis):
        reasons = blocking_reasons(analysis["health_report"])
        detail = (
            " Issues found: " + "; ".join(reasons[:4])
            + (f"; and {len(reasons) - 4} more" if len(reasons) > 4 else "") + "."
            if reasons else ""
        )
        return (
            "I can't answer this yet: the data-health checks found errors that block "
            "every outcome analysis (turnover, absenteeism and performance)." + detail
            + " Please fix the data and upload it again; see the Data page for the full list."
        )

    if outcome is not None and outcome in blocked:
        label = OUTCOME_LABELS.get(outcome, outcome)
        return (
            f"Analysis for {label} is currently blocked by the data-health checks, so I "
            "won't provide a business interpretation until the underlying data issue is "
            "resolved. I can still answer about: "
            + ", ".join(OUTCOME_LABELS.get(o, o) for o in analysis_allowed_outcomes(analysis))
            + "."
        )

    return None


# PROCESS 11 – BUILD NON-REPORTABLE EVIDENCE

def build_nonreportable_evidence(
    analysis,
    item,
    outcome,
):
    results = analysis["results"]

    row = results[
        (results["item"] == item)
        & (
            results["outcome"]
            == outcome
        )
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    if bool(row["reportable"]):
        return None

    if not bool(
        row["fdr_significant"]
    ):
        reason = (
            "The association did not "
            "remain statistically reliable "
            "after correction for multiple "
            "Q12-outcome tests."
        )

    elif not bool(
        row["meaningful_effect"]
    ):
        reason = (
            "The estimated association "
            "was too small to meet the "
            "predefined meaningful-effect "
            "threshold."
        )

    else:
        reason = (
            "The result does not meet "
            "the analytical reporting "
            "criteria."
        )

    return {
        "item":
            item,

        "item_label":
            ITEM_LABELS.get(
                item,
                item,
            ),

        "outcome":
            outcome,

        "outcome_label":
            OUTCOME_LABELS.get(
                outcome,
                outcome,
            ),

        "sample_size":
            int(row["n"]),

        "evidence_summary":
            (
                "The current analysis does not "
                "provide statistically reliable "
                "evidence of an association."
            ),

        "reason":
            reason,

        "decision":
            (
                "This item should not currently "
                "be presented as a supported "
                "finding or prioritized based "
                "on the available evidence."
            ),
    }


# PROCESS 12 – SELECT VERIFIED EVIDENCE

def select_verified_evidence(
    analysis,
    route,
):
    item = route["item"]
    outcome = route["outcome"]

    payloads = (
        analysis["payloads"]
    )

    # -------------------------------------------------
    # CASE 1:
    # specific item + specific outcome
    # -------------------------------------------------

    if (
        item is not None
        and outcome is not None
    ):

        matches = [
            p
            for p in payloads
            if (
                p["item"] == item
                and p["outcome"]
                == outcome
            )
        ]

        if matches:
            return {
                "type":
                    "single_finding",

                "findings":
                    matches,
            }

        nonreportable = (
            build_nonreportable_evidence(
                analysis,
                item,
                outcome,
            )
        )

        if nonreportable:
            return {
                "type":
                    "insufficient_evidence",

                "findings": [
                    nonreportable
                ],
            }

    # -------------------------------------------------
    # CASE 2:
    # specific item, all outcomes
    # -------------------------------------------------

    if (
        item is not None
        and outcome is None
    ):

        matches = [
            p
            for p in payloads
            if p["item"] == item
        ]

        if matches:
            return {
                "type":
                    "item_summary",

                "findings":
                    matches,
            }

    # -------------------------------------------------
    # CASE 3:
    # outcome specified, all Q12 items
    # -------------------------------------------------

    if (
        item is None
        and outcome is not None
    ):

        matches = [
            p
            for p in payloads
            if p["outcome"]
            == outcome
        ]

        return {
            "type":
                "outcome_summary",

            "findings":
                matches,
        }

    # -------------------------------------------------
    # CASE 4:
    # no specific item/outcome
    # -------------------------------------------------

    return {
        "type":
            "portfolio_summary",

        "findings":
            payloads,
    }


# PROCESS 13 – CREATE AZURE CLIENT

def add_patterns_and_exclusions(evidence, analysis, route):
    """Attach X4/X5-type patterns that match the question, and the list of
    outcomes the health check excluded."""

    item = route["item"]
    outcome = route["outcome"]

    patterns = [
        sf
        for sf in analysis.get("special_findings", [])
        if (item is None or sf["item"] == item)
        and (outcome is None or sf["outcome"] == outcome)
    ]

    if patterns:
        evidence["threshold_and_subgroup_patterns"] = patterns

    blocked = analysis_blocked_outcomes(analysis)

    if blocked:
        evidence["excluded_by_data_health_checks"] = {
            "outcomes": [OUTCOME_LABELS.get(o, o) for o in blocked],
            "note": (
                "These outcomes failed the data-health checks. Do not discuss "
                "them; say once that they were excluded until the data is fixed."
            ),
        }

    return evidence


# PROCESS 13 – CREATE AZURE CLIENT (v3 retry below)

def get_azure_client():

    # Re-read .env so a key added while the app is running is picked up.
    load_env_file()

    api_key = os.getenv(
        "GW_AZURE_OPENAI_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GW_AZURE_OPENAI_KEY "
            "was not found. "
            "Add it to the .env file in the "
            "project folder, or export it "
            "before running the agent."
        )

    endpoint = (
        f"https://"
        f"{AZURE_RESOURCE}"
        f".openai.azure.com/"
    )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=AZURE_API_VERSION,
    )


# PROCESS 14 – SYSTEM PROMPT

SYSTEM_PROMPT = """
You are an Employee Engagement Decision Intelligence Agent.

Your audience is nontechnical HR Leaders and Business Leaders.

You receive VERIFIED_EVIDENCE that was produced by deterministic
statistical tools. Your job is to explain that evidence clearly.

STRICT RULES:

1. Use only the evidence supplied in VERIFIED_EVIDENCE.

2. Never invent, estimate, recalculate, or change a supplied number.

3. Never claim causation.

4. Use terms such as:
   - associated with
   - linked with
   - shows an association

5. Avoid causal wording such as:
   - causes
   - drives
   - leads to

6. Distinguish odds from probability.

7. For a single verified finding, include:
   - effect size,
   - 95% confidence interval,
   - sample size,
   when those values are supplied.

8. Respect robustness status.

9. If robustness status is CAUTION:
   - explain the limitation,
   - do not present it as a priority recommendation.

10. If priority_recommendation_allowed is false,
    clearly state that the finding should not currently be prioritized.

11. Findings marked ROBUST may be discussed as credible areas
    for further business consideration, but not as proven causal levers.

12. Findings marked nonreportable must not be presented as
    statistically supported findings.

13. Do not expose Python field names, JSON keys, health-check codes,
    or diagnostic codes.

14. Translate technical metadata into natural business language.

15. Do not make recommendations about individual employees.

16. Do not rank or score individual employees.

17. If multiple findings are supplied, distinguish ROBUST findings
    from CAUTION findings.

18. When the user asks for priorities, prioritize only findings
    explicitly allowed for priority consideration.

19. Keep the response concise and understandable for HR and
    business decision-makers.

20. Display each supplied Calculation ID at most once.

21. If the evidence is insufficient to answer the question,
    state that clearly instead of guessing.

22.⁠ ⁠When an outcome_label is supplied, use that label in the
    user-facing response instead of the internal outcome name.

23. For findings that do not meet the reporting criteria, do not say
    "there is no association." Instead say:
    "The current analysis does not provide statistically reliable
    evidence of an association."

24. Do not expose internal classification terms such as
    "nonreportable" or "NOT_REPORTABLE" to the user.
    Explain the decision in plain business language instead.

25. For insufficient-evidence findings, do not use internal labels
    such as "nonreportable", "reportable", "insufficient_evidence",
    or similar implementation terminology.

26. State the conclusion as:
    "The current analysis does not provide statistically reliable
    evidence of an association."

27. Do not mention that an effect size or confidence interval was
    "not provided" unless the user specifically asks for those values.

28. Never expose internal field names or values such as:
    - priority_recommendation_allowed
    - robustness_status
    - evidence_summary
    - question_scope
    - reportable
    - calculation metadata fields
    Translate their meaning into natural business language.

29. Do not mention internal dataset names such as gallup_v2_5K
    unless the user explicitly asks about the dataset or data source.

30. Only give prioritization or action recommendations when:
    - the user explicitly asks for priority, recommendation, action,
      focus, or what to do next; or
    - the question is explicitly a ranking/prioritization question.
    Otherwise, explain the finding without introducing a recommendation.

31. When a finding should not be prioritized, say:
    "This finding should not currently be prioritized based on
    the available evidence."
    Never describe this using internal boolean fields.

32. When multiple findings share the same caveat, state the caveat
    once after the findings instead of repeating it for every item.  

33. VERIFIED_EVIDENCE may also contain sections from the dashboard:
    overview, trends, teams, predictions, recommendations and financials.
    Use every section that helps answer the question and combine them into
    one consistent answer. Quote their numbers exactly as given.

34. Predictions (risk bands, expected leavers, forecasts) are likelihoods
    from models that passed a test on data they had not seen. Say "likely",
    "expected" or "estimated"; never present them as certain.

35. Money figures are estimates based on the user's own cost assumptions,
    not measured profit. Say so once, briefly.

36. Talk about teams and groups only. Never single out, rank or guess about
    individual employees, even if asked.

37. If a section says a model did not pass its test, a check is pending, or
    an analysis was not run, say that plainly instead of using that section.

38. Answer the question first, in one or two plain sentences, then give only
    the supporting facts the question needs, under short bold headings.
    Keep it under 300 words unless the user asks for more detail.

39. Odds ratios, confidence intervals, sample sizes and robustness wording
    belong only in answers about a specific question-outcome relationship.
    For broad questions (teams, costs, trends, summaries) translate findings
    into plain effects, e.g. "linked to 24% lower odds of leaving".

40. When the user asks what to do (what should managers do, how to improve,
    actions, next steps), lead with up to three concrete actions taken from
    the recommendations section, each tied to its question and expected
    effect. Then name a few teams to start with if the evidence lists them.

41. When two analyses describe the same question differently, follow the
    explanation supplied in the evidence (for example, prefer the result that
    follows the same people over time) and say why in one short sentence.
    Never present the same question as both a top lever and "not a priority"
    without that explanation.

42. The traps section reports three checks: misleading department effects,
    hidden manager (team) effects and nonresponse bias. When a finding you
    mention is listed as affected by a manager trap, say the link mostly
    reflects differences between teams and suggest acting through managers.
    When nonresponse is "caution", add one sentence that the survey may not
    speak for everyone (for example, leavers answered less). Never call a
    result trustworthy without mentioning a trap that applies to it.

43. For "where should we invest", "what pays off most" or priority questions,
    use financials.factors_ranked_by_value: give the ranking with each
    factor's avoided leavers or absence days, its yearly value and its
    priority label, quoted exactly. Never add values across factors.

44. When recommending what to do, you may point to Gallup resources from
    recommendations.gallup_resources: the element guide for the question,
    action planning help, or manager learning. Use only the titles and URLs
    given there, written as markdown links; never invent a link. Say the
    Recommendations page can download each team's action plan for Gallup
    Access. Do not imply this app is made or endorsed by Gallup.

45. If VERIFIED_EVIDENCE lists outcomes excluded by the data-health checks,
    do not give any figure, trend, cost or recommendation for those outcomes.
    Say once, briefly, that they were excluded until the data is fixed.

46. If threshold or subgroup patterns are supplied, mention them when they
    concern the item or outcome asked about: a threshold means the link is
    concentrated below a score; a subgroup difference means the two groups
    should be reported separately, with both values.

47. Team risk predictions are warning signs, not causes. Never say a Q12
    item "drives" a team's predicted risk, and never choose what to improve
    from the predictions: the Outcomes findings decide that. If the
    predictions section carries a survey-response warning, explain it in one
    plain sentence: people who later leave answer the survey less often, so
    team scores can look better than the team really is. If the predictions
    section says percentages are hidden (calibration), give only the risk
    band (High, Elevated, Typical), never a percentage. If predictions are
    blocked or not validated, say so and do not name at-risk teams.
"""


# PROCESS 14A – API RETRY (fix #6)
#
# The Azure guide asks every client to retry HTTP 429 (rate limit) with a
# wait. Validation runs ask dozens of questions in a row, so one 429, timeout
# or empty answer must not stop the run.

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2.0
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _retry_wait(attempt, exc=None):
    """Wait time: the server's Retry-After header when given, else 2, 4, 8 s."""
    try:
        header = exc.response.headers.get("retry-after")  # type: ignore[union-attr]
        if header:
            return min(float(header), 60.0)
    except Exception:  # noqa: BLE001
        pass
    return RETRY_BASE_SECONDS * (2 ** attempt)


def _is_retryable(exc):
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRY_STATUS_CODES
    return False


def call_with_retry(client, request, max_retries=MAX_RETRIES, sleep=time.sleep):
    """Call chat.completions.create; retry 429 / 5xx / timeouts / empty answers
    up to max_retries times. Returns (response, answer_text)."""

    last_error = None

    for attempt in range(max_retries + 1):

        try:
            response = client.chat.completions.create(**request)

        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            last_error = exc
            sleep(_retry_wait(attempt, exc))
            continue

        answer = (response.choices[0].message.content or "")

        if answer.strip():
            return response, answer

        last_error = RuntimeError(
            "The language model returned no visible answer. "
            f"Finish reason: {response.choices[0].finish_reason}"
        )

        if attempt == max_retries:
            raise last_error

        sleep(_retry_wait(attempt))

    raise last_error  # pragma: no cover


# PROCESS 15 – GENERATE BUSINESS ANSWER

def generate_business_answer(
    question,
    route,
    evidence,
):
    client = get_azure_client()

    verified_json = json.dumps(
        evidence,
        indent=2,
    )

    routing_json = json.dumps(
        {
            "detected_item":
                route["item"],

            "detected_outcome":
                route["outcome"],

            "detected_intent":
                route["intent"],
        },
        indent=2,
    )

    request = dict(
            model=AZURE_DEPLOYMENT,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content": f"""
USER QUESTION:
{question}

ROUTING:
{routing_json}

VERIFIED_EVIDENCE:
{verified_json}

Answer the user's question for an HR Leader and Business Leader.
Use only VERIFIED_EVIDENCE.
""",
                },
            ],

            # Reasoning models spend tokens thinking before they write; with the
            # larger dashboard evidence, 4,000 was not enough to finish an answer.
            max_completion_tokens=16000,

            reasoning_effort="low",
    )

    # Retries 429 / 5xx / timeouts / empty answers (v3 fix 6).
    response, answer = call_with_retry(
        client,
        request,
    )

    return {
        "answer":
            answer,

        "finish_reason":
            response.choices[
                0
            ].finish_reason,

        "usage":
            response.usage,
    }


# PROCESS 16 – MAIN AGENT FUNCTION

def ask_agent(
    question: str,
    dataset_path=DEFAULT_DATASET,
    analysis_override=None,
    return_debug=False,
    context=None,
    data=None,
    change_analysis=None,
    assumptions=None,
    include_dashboard=True,
):
    """
    Answer a natural-language engagement question.

    The agent also knows the dashboard results (overview, trends, team risk,
    predictions, recommendations, financials):
    - context: facts already built by agent_context.build_context, or
    - data (+ change_analysis, assumptions): the agent builds the facts the
      question needs itself. Without data and without analysis_override, it
      uses the development dataset through the same pipeline as uploads.
    include_dashboard=False restores the statistics-only behaviour.

    If analysis_override is supplied, the agent uses the
    verified analysis from an uploaded dataset.

    Otherwise, the agent falls back to the development dataset.
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    # -----------------------------------------
    # ROUTE THE QUESTION
    # -----------------------------------------

    route = route_question(
        question
    )

    # -----------------------------------------
    # SELECT ANALYSIS SOURCE
    # -----------------------------------------

    frame = data

    stopped = None

    try:

        if analysis_override is not None:

            analysis = analysis_override

        elif include_dashboard and data is None:

            frame, analysis, dev_change = get_dev_dashboard(
                str(Path(dataset_path).resolve())
            )

            if change_analysis is None:
                change_analysis = dev_change

        else:

            dataset_path = str(
                Path(
                    dataset_path
                ).resolve()
            )

            analysis = get_analysis(
                dataset_path
            )

    except HealthCheckBlocked as exc:

        # Every outcome is blocked: refuse without calling the language model.
        stopped = str(exc)
        analysis = BLOCKED_ANALYSIS

    outcome = route["outcome"]

    # -----------------------------------------
    # HEALTH GATE
    # -----------------------------------------

    blocked_answer = (
        health_gate(analysis, outcome)
        if stopped is None
        else (
            "I can't answer this yet: the data-health checks found errors that block "
            "every outcome analysis (turnover, absenteeism and performance). "
            + stopped
        )
    )

    if blocked_answer is not None:

        if return_debug:

            return {
                "answer":
                    blocked_answer,

                "route":
                    route,

                "evidence":
                    None,

                "blocked":
                    True,
            }

        return blocked_answer

    # -----------------------------------------
    # SELECT VERIFIED EVIDENCE
    # -----------------------------------------

    evidence = select_verified_evidence(
        analysis,
        route,
    )

    evidence = add_patterns_and_exclusions(
        evidence,
        analysis,
        route,
    )

    # -----------------------------------------
    # DASHBOARD FACTS (TRENDS, RISK, COSTS ...)
    # -----------------------------------------

    sections = []

    if context is None and include_dashboard and frame is not None:

        from src.tools.agent_context import build_context, route_sections

        sections = route_sections(question)

        context = build_context(
            frame,
            analysis,
            change_analysis,
            assumptions,
            sections,
        )

    elif context:

        sections = [
            name
            for name in context
            if name != "model_survey"
        ]

    if context:

        # Broad questions get the findings as plain sentences (in the
        # overview's key_findings); full technical evidence is sent only
        # when the question names a specific Q12 item or outcome.
        specific = (
            route["item"] is not None
            or route["outcome"] is not None
        )

        exclusions = {
            key: evidence[key]
            for key in ("excluded_by_data_health_checks",)
            if key in evidence
        }

        evidence = {
            **(
                {"statistical_findings": evidence}
                if specific
                else {}
            ),

            **context,

            **exclusions,
        }

    # -----------------------------------------
    # GENERATE BUSINESS ANSWER
    # -----------------------------------------

    result = generate_business_answer(
        question,
        route,
        evidence,
    )

    # -----------------------------------------
    # OPTIONAL DEBUG OUTPUT
    # -----------------------------------------

    if return_debug:

        return {
            "answer":
                result["answer"],

            "route":
                route,

            "evidence":
                evidence,

            "sections":
                sections,

            "finish_reason":
                result[
                    "finish_reason"
                ],

            "usage":
                result["usage"],
        }

    return result["answer"]


# PROCESS 17 – COMMAND-LINE TEST

if __name__ == "__main__":

    if len(sys.argv) > 1:
        question = " ".join(
            sys.argv[1:]
        )

    else:
        question = input(
            "Ask the engagement agent: "
        )

    result = ask_agent(
        question,
        return_debug=True,
    )

    print()
    print(result["answer"])
    print()
    print(
        "Based on: statistical findings"
        + "".join(
            f", {name}"
            for name in result.get("sections", [])
        )
    )

    from src.tools.agent_context import check_numbers

    unverified = check_numbers(
        result["answer"],
        result.get("evidence"),
        question,
    )

    if unverified:
        print(
            "Could not trace these figures to the verified results: "
            + ", ".join(f"{n:g}" for n in unverified)
        )