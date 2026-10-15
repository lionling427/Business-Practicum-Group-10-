# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.dashboard import (
    driver_chart,
    driver_rows,
)
from src.tools.insights import (
    OUTCOME_INFO,
    findings_table,
)
from src.tools.visualization import (
    OUTCOME_RULES,
    ITEM_LABELS,
    explain_sorting_power,
    rank_sorting_power,
    explain_sorting_distribution,
    sorting_distribution,
    sorting_distribution_chart,
    sorting_power_chart,
)
from src.tools.traps import STATUS_BADGES, TRAP_NAMES
from src.tools.model_spec import SPEC_VERSION, SPECS, describe, spec_fingerprint
from src.tools.waves import model_wave, wave_label
from src.ui import (
    OUTCOME_LEVEL,
    analysis_runner,
    how_to_read,
    outcome_tiles,
    page_header,
    require_data,
    story_box,
    trap_checks,
)


# PROCESS 2 – HEADER

page_header(
    "Business outcomes",
    ":material/query_stats:",
    "Which engagement questions are linked to turnover, absence and performance.",
)

df, validation = require_data()

if validation["capabilities"]["level"] != OUTCOME_LEVEL:
    st.info(
        "This page needs business outcomes in the uploaded data: "
        "`left_voluntarily_12m`, `unplanned_absence_days_6m` or `performance_rating`.",
        icon=":material/info:",
    )
    st.stop()

sample_wave = wave_label(model_wave(df))
analysis = analysis_runner(df, key="run_analysis_outcomes")


# PROCESS 3 – ONE ROW OF TILES

st.subheader(f"Outcomes · {sample_wave}", anchor=False)

with st.container(horizontal=True):

    outcome_tiles(df)

    if analysis is not None:
        findings = analysis["findings"]
        st.metric(
            "Findings that passed all checks",
            f"{int((findings['robustness_status'] == 'ROBUST').sum())} of {len(findings)}",
            help=f"Out of the findings that met the reporting rules, for "
            f"{analysis['sample_size']:,} employees in {sample_wave}.",
            border=True,
        )

# PROCESS 4 – WHICH QUESTIONS MATTER FOR EACH OUTCOME

# Shown once the statistical analysis has run (button above).
if analysis is not None:

    rows = driver_rows(analysis, df)

    with st.container(border=True):

        st.subheader("Which questions matter for each outcome", anchor=False)

        st.markdown(
            ":green-badge[Passed all checks] :gray-badge[Use with caution]  "
            "Bars show how much better the outcome is for each 1-point higher score "
        "on the question."
        )

        outcomes = ["turnover", "absence", "performance"]

        # Row 1: one chart per outcome. Row 2: its takeaway, so the text lines up.
        for outcome, column in zip(outcomes, st.columns(3, gap="large")):

            with column:

                st.markdown(f"**{OUTCOME_INFO[outcome]['label']}**")

                if (rows["outcome"] == outcome).any():
                    st.altair_chart(
                        driver_chart(rows, outcome),
                        width="stretch",
                        alt=f"Questions linked to {OUTCOME_INFO[outcome]['label'].lower()}",
                    )
                else:
                    st.caption("No question met the reporting rules for this outcome.")

        for outcome, column in zip(outcomes, st.columns(3, gap="large")):

            outcome_rows = rows[rows["outcome"] == outcome]

            if outcome_rows.empty:
                continue

            with column:

                top = outcome_rows.sort_values("size", ascending=False).iloc[0]
                st.caption(top["headline"])

                for caution in outcome_rows.loc[outcome_rows["caution"] != "", "caution"].unique():
                    st.caption(f":material/warning: {caution}")

        st.caption(
            f"From {sample_wave}. Each result holds the other questions and workforce "
            "factors constant. Associations, not proof of cause and effect."
        )

        with st.expander("All findings as a table", icon=":material/table_chart:"):
            st.table(
                findings_table(analysis, df),
                border="horizontal",
                hide_index=True,
                alt="Plain-language list of findings",
            )

        with st.expander("Technical details", icon=":material/functions:"):
            st.dataframe(
                analysis["findings"][[
                    "item", "outcome", "effect_type", "effect",
                    "ci_low", "ci_high", "robustness_status", "priority_eligible",
                ]],
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    "item": "Q12 item",
                    "outcome": "Outcome",
                    "effect_type": "Effect type",
                    "effect": st.column_config.NumberColumn("Effect", format="%.3f"),
                    "ci_low": st.column_config.NumberColumn("95% CI low", format="%.3f"),
                    "ci_high": st.column_config.NumberColumn("95% CI high", format="%.3f"),
                    "robustness_status": "Robustness",
                    "priority_eligible": "Priority eligible",
                },
                alt="Model effects with 95% confidence intervals",
            )

        with st.expander("How these numbers are computed", icon=":material/rule:"):

            st.markdown(
                "Each outcome uses a **pre-defined method and controls**, fixed before any "
                "data is seen. The data health check can only switch on extra controls that "
                "were registered in advance. Verified code computes every number; the AI "
                "only explains them."
            )

            methods = analysis.get("methods") or {
                o: describe(o) for o in SPECS
            }

            st.table(
                [{
                    "Outcome": SPECS[o].label,
                    "Method": m["method"],
                    "Effect reported": m["effect_unit"],
                    "Controls": ", ".join(m["controls"]),
                    "Added by health check": ", ".join(m["controls_added_by_health_check"]) or "none",
                    "Minimum effect to report": m["min_effect"],
                } for o, m in methods.items()],
                border="horizontal",
                hide_index=True,
                alt="Fixed statistical method for each outcome",
            )

            decisions = (analysis.get("adjustments") or {}).get("control_decisions", {})
            refused = {o: d["refused"] for o, d in decisions.items() if d["refused"]}
            if refused:
                st.caption(":material/block: Requested but not pre-registered, so not added: "
                           + "; ".join(f"{o}: {', '.join(c)}" for o, c in refused.items()) + ".")

            st.caption(
                f"Method version {SPEC_VERSION} · fingerprint {spec_fingerprint()} (part of "
                "every calculation ID). Standard errors clustered by team; multiple testing "
                "controlled with a 5% false discovery rate."
            )


# PROCESS 5 – ANALYTICAL TRAPS

TRAP_ICONS = {"department": ":material/domain:", "manager": ":material/supervisor_account:",
              "nonresponse": ":material/how_to_reg:"}

traps = trap_checks(df)

with st.container(border=True):

    st.subheader("Analytical traps checked", anchor=False)
    st.caption(
        f"Three common ways engagement results mislead, tested on {traps['wave']} with the "
        "same fixed models."
    )

    for name, column in zip(TRAP_NAMES, st.columns(3, gap="large")):

        trap = traps[name]

        with column:
            st.markdown(f"**{TRAP_ICONS[name]} {TRAP_NAMES[name]}**")
            st.markdown(STATUS_BADGES[trap["status"]])
            st.markdown(trap["headline"])
            for point in trap["points"]:
                st.markdown(f"- {point}")

    with st.expander("Trap details", icon=":material/table_chart:"):

        verdicts = {"flip": "Reverses", "shrink": "Mostly disappears", "ok": "Holds"}

        department = traps["department"]["table"]
        if not department.empty:
            st.markdown("**Department:** each link ignoring departments vs comparing within "
                        "departments (ratios for turnover and absence, beta for performance).")
            st.dataframe(
                department.assign(verdict=department["verdict"].map(verdicts))
                [["outcome", "question", "ignoring_department",
                  "comparing_within_department", "verdict"]],
                hide_index=True, width="stretch", height=280,
                column_config={"outcome": "Outcome", "question": "Question",
                               "ignoring_department": st.column_config.NumberColumn(
                                   "Ignoring departments", format="%.3f"),
                               "comparing_within_department": st.column_config.NumberColumn(
                                   "Within departments", format="%.3f"),
                               "verdict": "Verdict"},
            )

        manager = traps["manager"]["table"]
        if not manager.empty:
            st.markdown("**Manager (team):** each link across all teams vs within the same "
                        "team (linear model, team fixed effects).")
            st.dataframe(
                manager.assign(verdict=manager["verdict"].map(verdicts))
                [["outcome", "question", "across_teams", "within_same_team", "verdict"]],
                hide_index=True, width="stretch", height=280,
                column_config={"outcome": "Outcome", "question": "Question",
                               "across_teams": st.column_config.NumberColumn(
                                   "Across teams", format="%.3f"),
                               "within_same_team": st.column_config.NumberColumn(
                                   "Within the same team", format="%.3f"),
                               "verdict": "Verdict"},
            )

        nonresponse = traps["nonresponse"]["table"]
        if not nonresponse.empty:
            st.markdown("**Nonresponse:** people who answered vs people who did not.")
            st.dataframe(
                nonresponse.drop(columns="kind"),
                hide_index=True, width="stretch",
                column_config={"measure": "Measure",
                               "respondents": st.column_config.NumberColumn(
                                   "Answered", format="%.2f"),
                               "non_respondents": st.column_config.NumberColumn(
                                   "Did not answer", format="%.2f")},
            )

        st.caption("A link is flagged only when a naive comparison shows it as significant: "
                   "'Reverses' means it is significant in the other direction once compared "
                   "fairly; 'Mostly disappears' means half or more of it goes away.")


# PROCESS 6 – SORTING EFFECT

sorting_outcomes = [o for o, rule in OUTCOME_RULES.items() if rule["column"] in df.columns]

with st.container(border=True):

    title_col, outcome_col, level_col = st.columns([1.2, 1.6, 1], vertical_alignment="bottom")

    title_col.subheader("Sorting effect", anchor=False)

    sort_outcome = outcome_col.segmented_control(
        "Outcome",
        options=sorting_outcomes,
        format_func=lambda o: {"performance": "Performance", "turnover": "Turnover",
                               "absence": "Absence"}[o],
        default=sorting_outcomes[0],
        key="sort_outcome",
        label_visibility="collapsed",
    ) or sorting_outcomes[0]

    sort_level = level_col.segmented_control(
        "Compare",
        options=["team", "employee"],
        format_func=lambda x: "Teams" if x == "team" else "Employees",
        default="team",
        key="sort_level",
        label_visibility="collapsed",
    ) or "team"

    st.caption(
        "Does each question separate the best-performing groups from the rest? "
        "Descriptive only: raw scores, no controls."
    )

    try:
        ranking = rank_sorting_power(df, sort_outcome, sort_level)
    except ValueError as exc:
        st.info(str(exc))
        st.stop()

    sorting_story = explain_sorting_power(ranking, sort_outcome, sort_level)

    chart_col, story_col = st.columns([3, 2], gap="large")

    with chart_col:
        st.altair_chart(
            sorting_power_chart(ranking),
            width="stretch",
            alt="Q12 questions ranked by sorting power",
        )

    with story_col:
        story_box(sorting_story)

    how_to_read(sorting_story["how_to_read"])

    st.markdown("**Look closer at a few questions**")

    sort_items = st.multiselect(
        "Questions to compare",
        options=list(ranking["item"]),
        default=[ranking["item"].iloc[0], ranking["item"].iloc[-1]],
        max_selections=4,
        format_func=lambda i: f"{i} · {ITEM_LABELS.get(i, i)}",
        key=f"sort_items_{sort_outcome}_{sort_level}",
        label_visibility="collapsed",
    )

    for position, item in enumerate(sort_items):

        dist = sorting_distribution(df, item, sort_outcome, sort_level)

        text_col, chart_col = st.columns([2, 3], gap="large", vertical_alignment="center")

        with text_col:
            st.markdown(f"**{item} · {ITEM_LABELS.get(item, item)}**")
            st.markdown(explain_sorting_distribution(dist))

        with chart_col:
            st.altair_chart(
                # One legend, above the first question only.
                sorting_distribution_chart(dist, show_legend=position == 0),
                width="stretch",
                alt=f"Share who disagree, are neutral or agree on {item}, by performance group",
            )

    if sort_items:
        how_to_read(
            "Each bar splits one group into those who disagree (red), are neutral (gray) "
            "and agree (green). When the top bar is much greener than the bottom one, the "
            "question separates the stronger group from the weaker one. All "
            + ("employees" if sort_level == "employee" else "teams")
            + " in the two groups are counted. For teams, each team's average answer is "
            "used: below 3, 3 to 4, or 4 and above.",
            "How to read these bars",
        )
