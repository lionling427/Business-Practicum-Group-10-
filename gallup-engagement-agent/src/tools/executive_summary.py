# PROCESS 1 – IMPORTS

import math

from src.tools.analytics import (
    ITEM_LABELS,
    OUTCOME_LABELS,
)


# PROCESS 2 – CONFIGURATION

OUTCOME_COLUMNS = {
    "turnover": "left_voluntarily_12m",
    "absence": "unplanned_absence_days_6m",
    "performance": "performance_rating",
}


# PROCESS 3 – HELPERS

def _item_name(item):

    label = ITEM_LABELS.get(item)

    return f"{item} {label}" if label else item


def is_favorable(outcome, effect):

    # Lower turnover and absence are good; higher performance is good.
    if outcome == "performance":
        return effect > 0

    return effect < 1


def effect_strength(effect_type, effect):

    if effect_type == "Standardized Beta":
        return abs(effect)

    if effect <= 0:
        return 0.0

    return abs(math.log(effect))


def _describe_effect(outcome, effect_type, effect):

    outcome_label = OUTCOME_LABELS.get(outcome, outcome)

    if effect_type == "Odds Ratio":

        change = abs(effect - 1) * 100
        direction = "lower" if effect < 1 else "higher"

        return (
            f"{change:.0f}% {direction} odds of {outcome_label}"
        )

    if effect_type == "Incidence Rate Ratio":

        change = abs(effect - 1) * 100
        direction = "lower" if effect < 1 else "higher"

        return (
            f"{change:.0f}% {direction} rate of {outcome_label}"
        )

    direction = "higher" if effect > 0 else "lower"

    return (
        f"{abs(effect):.2f} SD {direction} {outcome_label}"
    )


# PROCESS 4 – BUILD EXECUTIVE SUMMARY

def build_executive_summary(
    df,
    survey_summary=None,
    analysis=None,
):
    """
    Build a short, plain-language executive summary.

    Returns a dict with:
    - headline: one-sentence overview
    - highlights: list of short markdown bullets
    - next_step: recommended action, or None
    """

    highlights = []

    employee_count = len(df)

    headline = (
        f"This dataset covers **{employee_count:,} employees**."
    )

    # -----------------------------------------
    # ENGAGEMENT LEVEL
    # -----------------------------------------

    if (
        survey_summary is not None
        and survey_summary.get("overall_mean") is not None
    ):

        overall_mean = survey_summary["overall_mean"]

        headline = (
            f"This dataset covers **{employee_count:,} employees** "
            f"with an average Q12 score of **{overall_mean:.2f}**."
        )

        item_summary = survey_summary["item_summary"]

        if not item_summary.empty:

            ranked = item_summary.sort_values(
                "mean_score",
                ascending=False,
            )

            top = ranked.iloc[0]
            bottom = ranked.iloc[-1]

            highlights.append(
                f"**Strongest engagement area:** "
                f"{_item_name(top['item'])} "
                f"({top['mean_score']:.2f})"
            )

            highlights.append(
                f"**Weakest engagement area:** "
                f"{_item_name(bottom['item'])} "
                f"({bottom['mean_score']:.2f})"
            )

    # -----------------------------------------
    # BUSINESS OUTCOMES
    # -----------------------------------------

    outcome_parts = []

    if OUTCOME_COLUMNS["turnover"] in df.columns:

        rate = df[OUTCOME_COLUMNS["turnover"]].mean() * 100

        outcome_parts.append(
            f"{rate:.1f}% voluntary turnover"
        )

    if OUTCOME_COLUMNS["absence"] in df.columns:

        avg_absence = df[OUTCOME_COLUMNS["absence"]].mean()

        outcome_parts.append(
            f"{avg_absence:.2f} average absence days"
        )

    if OUTCOME_COLUMNS["performance"] in df.columns:

        avg_performance = df[OUTCOME_COLUMNS["performance"]].mean()

        outcome_parts.append(
            f"{avg_performance:.2f} average performance rating"
        )

    if outcome_parts:

        highlights.append(
            "**Business outcomes:** " + ", ".join(outcome_parts)
        )

    # -----------------------------------------
    # STATISTICAL FINDINGS
    # -----------------------------------------

    if analysis is None:

        next_step = (
            "Run the statistical analysis to identify which "
            "engagement factors are linked to business outcomes."
            if outcome_parts
            else None
        )

        return {
            "headline": headline,
            "highlights": highlights,
            "next_step": next_step,
        }

    findings = analysis["findings"]

    if findings.empty:

        highlights.append(
            "**Key drivers:** no Q12-outcome relationships met "
            "the statistical reporting criteria."
        )

        return {
            "headline": headline,
            "highlights": highlights,
            "next_step": None,
        }

    robust = findings[
        findings["priority_eligible"]
    ]

    highlights.append(
        f"**Evidence quality:** {len(robust)} of "
        f"{len(findings)} reportable findings are robust."
    )

    # Strongest favorable robust driver for each outcome.
    for outcome in OUTCOME_LABELS:

        candidates = [
            row
            for _, row in robust[
                robust["outcome"] == outcome
            ].iterrows()
            if is_favorable(outcome, row["effect"])
        ]

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda row: effect_strength(
                row["effect_type"],
                row["effect"],
            ),
        )

        # Performance models use z-scored items; the others use raw 1–5 items.
        unit = (
            "1-SD"
            if best["effect_type"] == "Standardized Beta"
            else "1-point"
        )

        highlights.append(
            f"**Top driver of {OUTCOME_LABELS[outcome]}:** "
            f"{_item_name(best['item'])}, where a {unit} "
            f"higher score is associated with "
            + _describe_effect(
                outcome,
                best["effect_type"],
                best["effect"],
            )
            + "."
        )

    next_step = (
        "Prioritize the top drivers above in action planning, "
        "and review the full results table for caveats."
        if not robust.empty
        else "No finding passed robustness checks; treat results "
        "as exploratory before prioritizing action."
    )

    return {
        "headline": headline,
        "highlights": highlights,
        "next_step": next_step,
    }
