# PROCESS 1 – IMPORTS

import streamlit as st

from src.agent import ask_agent

from src.tools.analytics import (
    analyze_dataframe,
)

from src.tools.executive_summary import (
    build_executive_summary,
)

from src.tools.insights import (
    MIN_TEAM_SIZE,
    analysis_frame,
    brief_markdown,
    manager_brief,
    team_scorecard,
)

from src.tools.visualization import (
    OUTCOME_RULES,
    VISUAL_CONFIG,
    SCOPE_COLUMNS,
    build_engagement_trend,
    build_overview,
    engagement_trend_chart,
    explain_overview,
    percentile_meter_chart,
    score_gauge_chart,
    trend_sparkline_chart,
    explain_engagement_trend,
    explain_sorting_items,
    explain_sorting_power,
    explain_team_heatmap,
    rank_sorting_power,
    sorting_effect_chart,
    sorting_power_chart,
    team_heatmap_chart,
)

from src.tools.manager_coach import (
    coach_brief,
)

from src.tools.data_ingestion import (
    build_segment_summary,
    build_survey_summary,
    normalize_columns,
    read_uploaded_file,
    validate_uploaded_data,
)


# PROCESS 2 – PAGE CONFIGURATION

st.set_page_config(
    page_title="Employee Engagement Intelligence Agent",
    page_icon="📊",
    layout="wide",
)


# PROCESS 3 – PAGE HEADER

st.title(
    "Employee Engagement Intelligence Agent"
)

st.caption(
    "Evidence-based insights for HR and Business Leaders"
)


# PROCESS 3A – WORKSPACE TABS

upload_tab, coach_tab, chat_tab = st.tabs(
    [
        "Data Workspace",
        "Manager Coach",
        "Ask the Agent",
    ]
)


# PROCESS 4 – SIDEBAR INFORMATION

with st.sidebar:

    st.header("About")

    st.write(
        """
        This AI agent helps HR and Business Leaders
        understand relationships between Gallup Q12
        engagement factors and business outcomes.
        """
    )

    st.subheader("Business outcomes")

    st.markdown(
        """
        - Voluntary turnover
        - Absenteeism
        - Performance
        """
    )

    st.subheader("How results are generated")

    st.markdown(
        """
        1. Statistical models calculate the evidence.
        2. Data-health and robustness checks validate it.
        3. The AI explains verified results in business language.
        """
    )

    st.info(
        "The agent reports associations, not causal relationships."
    )


# PROCESS 5 – DATA WORKSPACE

with upload_tab:

    st.subheader(
        "Upload Employee Engagement Data"
    )

    st.write(
        "Upload a CSV or Excel file containing "
        "employee survey responses."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=[
            "csv",
            "xlsx",
        ],
    )

    if uploaded_file is not None:

        try:

            # -----------------------------------------
            # LOAD AND NORMALIZE DATA
            # -----------------------------------------

            uploaded_df = read_uploaded_file(
                uploaded_file
            )

            uploaded_df = normalize_columns(
                uploaded_df
            )

            validation = validate_uploaded_data(
                uploaded_df
            )

            # -----------------------------------------
            # RESET OLD ANALYSIS WHEN FILE CHANGES
            # -----------------------------------------

            upload_signature = (
                f"{uploaded_file.name}:"
                f"{getattr(uploaded_file, 'size', 0)}"
            )

            if (
                st.session_state.get(
                    "upload_signature"
                )
                != upload_signature
            ):

                st.session_state[
                    "upload_signature"
                ] = upload_signature

                st.session_state.pop(
                    "uploaded_analysis",
                    None,
                )

            # -----------------------------------------
            # STORE UPLOADED DATA
            # -----------------------------------------

            st.session_state[
                "uploaded_data"
            ] = uploaded_df

            st.session_state[
                "uploaded_validation"
            ] = validation

            st.success(
                "File loaded successfully."
            )

            # Filled at the end so it reflects any analysis run below.
            executive_summary_slot = st.container(
                border=True
            )

            # -----------------------------------------
            # DATASET OVERVIEW
            # -----------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rows",
                f"{validation['rows']:,}",
            )

            col2.metric(
                "Columns",
                validation["columns"],
            )

            col3.metric(
                "Analysis Level",
                validation[
                    "capabilities"
                ]["level"],
            )

            # -----------------------------------------
            # DATA PREVIEW
            # -----------------------------------------

            st.subheader(
                "Data Preview"
            )

            st.dataframe(
                uploaded_df.head(20),
                width="stretch",
            )

            # -----------------------------------------
            # SCHEMA VALIDATION
            # -----------------------------------------

            st.subheader(
                "Schema Validation"
            )

            if validation["is_valid"]:

                st.success(
                    "Core survey schema is valid."
                )

            else:

                st.error(
                    "The dataset has validation issues."
                )

            if validation[
                "missing_core_columns"
            ]:

                st.warning(
                    "Missing required columns: "
                    + ", ".join(
                        validation[
                            "missing_core_columns"
                        ]
                    )
                )

            if (
                validation[
                    "duplicate_employee_ids"
                ]
                > 0
            ):

                st.warning(
                    "Duplicate employee IDs: "
                    f"{validation['duplicate_employee_ids']:,}"
                )

            # -----------------------------------------
            # AVAILABLE ANALYSIS
            # -----------------------------------------

            capabilities = validation[
                "capabilities"
            ]

            st.subheader(
                "Available Analysis"
            )

            if (
                capabilities["level"]
                == "SURVEY_ONLY"
            ):

                st.info(
                    "Survey analysis is available. "
                    "Business outcome modeling is not available "
                    "because outcome data was not provided."
                )

            elif (
                capabilities["level"]
                == "SEGMENT_ANALYSIS"
            ):

                st.info(
                    "Survey and workforce segment analysis "
                    "are available."
                )

            elif (
                capabilities["level"]
                == "OUTCOME_ANALYSIS"
            ):

                st.success(
                    "Survey and business outcome analysis "
                    "are available."
                )

                # -----------------------------------------
                # AVAILABLE BUSINESS OUTCOMES
                # -----------------------------------------

                available_outcomes = capabilities[
                    "outcomes"
                ]

                if available_outcomes:

                    readable_outcomes = {
                        "turnover":
                            "Voluntary Turnover",

                        "absence":
                            "Absenteeism",

                        "performance":
                            "Performance",
                    }

                    st.write(
                        "**Available business outcomes:**"
                    )

                    for outcome in available_outcomes:

                        st.write(
                            "✓ "
                            + readable_outcomes.get(
                                outcome,
                                outcome,
                            )
                        )

                # -----------------------------------------
                # BUSINESS OUTCOME SUMMARY
                # -----------------------------------------

                st.subheader(
                    "Business Outcome Summary"
                )

                metric_cols = st.columns(3)

                # Voluntary Turnover
                if (
                    "left_voluntarily_12m"
                    in uploaded_df.columns
                ):

                    turnover_rate = (
                        uploaded_df[
                            "left_voluntarily_12m"
                        ]
                        .mean()
                        * 100
                    )

                    metric_cols[0].metric(
                        "Voluntary Turnover Rate",
                        f"{turnover_rate:.1f}%",
                    )

                # Absenteeism
                if (
                    "unplanned_absence_days_6m"
                    in uploaded_df.columns
                ):

                    avg_absence = (
                        uploaded_df[
                            "unplanned_absence_days_6m"
                        ]
                        .mean()
                    )

                    metric_cols[1].metric(
                        "Average Absence Days",
                        f"{avg_absence:.2f}",
                    )

                # Performance
                if (
                    "performance_rating"
                    in uploaded_df.columns
                ):

                    avg_performance = (
                        uploaded_df[
                            "performance_rating"
                        ]
                        .mean()
                    )

                    metric_cols[2].metric(
                        "Average Performance Rating",
                        f"{avg_performance:.2f}",
                    )

                # -----------------------------------------
                # RUN STATISTICAL ANALYSIS
                # -----------------------------------------

                st.subheader(
                    "Statistical Outcome Analysis"
                )

                st.write(
                    "Run the analytical engine to evaluate "
                    "associations between Q12 engagement factors "
                    "and the available business outcomes."
                )

                run_analysis = st.button(
                    "Run Statistical Analysis",
                    type="primary",
                )

                if run_analysis:

                    with st.spinner(
                        "Running statistical models "
                        "and robustness checks..."
                    ):

                        try:

                            uploaded_analysis = (
                                analyze_dataframe(
                                    uploaded_df,
                                    dataset_name=(
                                        uploaded_file.name
                                    ),
                                )
                            )

                            st.session_state[
                                "uploaded_analysis"
                            ] = uploaded_analysis

                            st.success(
                                "Statistical analysis "
                                "completed successfully."
                            )

                        except Exception as exc:

                            st.error(
                                "Statistical analysis "
                                "could not be completed."
                            )

                            st.caption(
                                f"Technical detail: {exc}"
                            )

                # -----------------------------------------
                # DISPLAY STATISTICAL RESULTS
                # -----------------------------------------

                if (
                    "uploaded_analysis"
                    in st.session_state
                ):

                    uploaded_analysis = (
                        st.session_state[
                            "uploaded_analysis"
                        ]
                    )

                    findings = uploaded_analysis[
                        "findings"
                    ]

                    st.subheader(
                        "Analysis Results"
                    )

                    result_col1, result_col2, result_col3 = (
                        st.columns(3)
                    )

                    result_col1.metric(
                        "Analyzed Employees",
                        f"{uploaded_analysis['sample_size']:,}",
                    )

                    result_col2.metric(
                        "Reportable Findings",
                        len(findings),
                    )

                    robust_count = int(
                        (
                            findings[
                                "robustness_status"
                            ]
                            == "ROBUST"
                        ).sum()
                    )

                    result_col3.metric(
                        "Robust Findings",
                        robust_count,
                    )

                    if not findings.empty:

                        display_findings = findings[
                            [
                                "item",
                                "outcome",
                                "effect_type",
                                "effect",
                                "ci_low",
                                "ci_high",
                                "robustness_status",
                                "priority_eligible",
                            ]
                        ].copy()

                        display_findings = (
                            display_findings.rename(
                                columns={
                                    "item":
                                        "Q12 Item",

                                    "outcome":
                                        "Outcome",

                                    "effect_type":
                                        "Effect Type",

                                    "effect":
                                        "Effect",

                                    "ci_low":
                                        "95% CI Low",

                                    "ci_high":
                                        "95% CI High",

                                    "robustness_status":
                                        "Robustness",

                                    "priority_eligible":
                                        "Priority Eligible",
                                }
                            )
                        )

                        st.dataframe(
                            display_findings,
                            width="stretch",
                            hide_index=True,
                        )

                    else:

                        st.info(
                            "No Q12-outcome relationships "
                            "met the current statistical "
                            "reporting criteria."
                        )

            # -----------------------------------------
            # OVERALL SURVEY SUMMARY
            # -----------------------------------------

            if validation["is_valid"]:

                summary = build_survey_summary(
                    uploaded_df
                )

                st.subheader(
                    "Overall Engagement"
                )

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Employees",
                    f"{summary['employee_count']:,}",
                )

                metric2.metric(
                    "Average Q12 Score",
                    f"{summary['overall_mean']:.2f}",
                )

                item_summary = summary[
                    "item_summary"
                ].copy()

                item_summary[
                    "agree_share"
                ] = (
                    item_summary[
                        "agree_share"
                    ]
                    * 100
                )

                item_summary = (
                    item_summary.rename(
                        columns={
                            "item":
                                "Q12 Item",

                            "mean_score":
                                "Mean Score",

                            "median_score":
                                "Median Score",

                            "agree_share":
                                "Agree / Strongly Agree (%)",

                            "response_count":
                                "Responses",
                        }
                    )
                )

                st.subheader(
                    "Q12 Summary"
                )

                st.dataframe(
                    item_summary,
                    width="stretch",
                    hide_index=True,
                )

            # -----------------------------------------
            # SEGMENT ANALYSIS
            # -----------------------------------------

            profile_columns = capabilities[
                "profile_columns"
            ]

            categorical_segments = [
                column
                for column in profile_columns
                if column in [
                    "department",
                    "role_level",
                    "pay_band",
                    "shift_type",
                ]
            ]

            if categorical_segments:

                st.subheader(
                    "Workforce Segment Analysis"
                )

                selected_segment = st.selectbox(
                    "Analyze engagement by",
                    options=categorical_segments,
                    format_func=lambda x: (
                        x
                        .replace("_", " ")
                        .title()
                    ),
                )

                segment_summary = (
                    build_segment_summary(
                        uploaded_df,
                        selected_segment,
                    )
                )

                segment_summary = (
                    segment_summary
                    .sort_values(
                        "engagement_mean",
                        ascending=False,
                    )
                )

                display_segment = (
                    segment_summary.rename(
                        columns={
                            selected_segment:
                                selected_segment
                                .replace("_", " ")
                                .title(),

                            "employee_count":
                                "Employees",

                            "engagement_mean":
                                "Average Q12 Score",
                        }
                    )
                )

                st.dataframe(
                    display_segment,
                    width="stretch",
                    hide_index=True,
                )

                st.bar_chart(
                    segment_summary,
                    x=selected_segment,
                    y="engagement_mean",
                )

            # -----------------------------------------
            # VISUAL INSIGHTS
            # -----------------------------------------

            if (
                validation["is_valid"]
                and "employee_id" in uploaded_df.columns
            ):

                st.subheader(
                    "Visual Insights"
                )

                trend_tab, sorting_tab, heatmap_tab = st.tabs(
                    [
                        "Engagement trend",
                        "Sorting effect",
                        "Team heatmap",
                    ]
                )

                try:

                    # ---------- ENGAGEMENT TREND ----------

                    with trend_tab:

                        trend_group = st.selectbox(
                            "Compare engagement by",
                            options=[None] + categorical_segments,
                            format_func=lambda x: (
                                "All employees"
                                if x is None
                                else x.replace("_", " ").title()
                            ),
                            key="trend_group",
                        )

                        trend = build_engagement_trend(
                            uploaded_df,
                            trend_group,
                        )

                        trend_story = explain_engagement_trend(
                            trend
                        )

                        with st.container(border=True):

                            st.markdown(
                                f"**{trend_story['headline']}**"
                            )

                            st.markdown(
                                "\n".join(
                                    f"- {point}"
                                    for point in trend_story["points"]
                                )
                            )

                        st.altair_chart(
                            engagement_trend_chart(trend),
                            width="content",
                            alt="Percent of employees engaged by survey period",
                        )

                        with st.expander(
                            "How to read this chart",
                            icon=":material/help:",
                        ):

                            st.write(
                                trend_story["how_to_read"]
                            )

                        st.caption(
                            "Engaged = average Q12 score of "
                            f"{VISUAL_CONFIG['engaged_threshold']:.1f} or higher. "
                            "This is an approximation; Gallup's official "
                            "engagement classification is proprietary. "
                            f"Groups under {VISUAL_CONFIG['min_group_size']} "
                            "people are hidden."
                        )

                    # ---------- SORTING EFFECT ----------

                    with sorting_tab:

                        sorting_outcomes = [
                            outcome
                            for outcome in OUTCOME_RULES
                            if OUTCOME_RULES[outcome]["column"]
                            in uploaded_df.columns
                        ]

                        if not sorting_outcomes:

                            st.info(
                                "Sorting charts need at least one business "
                                "outcome (performance, turnover or absence)."
                            )

                        else:

                            sort_col1, sort_col2 = st.columns(2)

                            sort_outcome = sort_col1.selectbox(
                                "Outcome",
                                options=sorting_outcomes,
                                format_func=lambda x: (
                                    OUTCOME_RULES[x]["label"].capitalize()
                                ),
                                key="sort_outcome",
                            )

                            sort_level = sort_col2.segmented_control(
                                "Compare",
                                options=["team", "employee"],
                                format_func=lambda x: (
                                    "Teams" if x == "team" else "Employees"
                                ),
                                default="team",
                                key="sort_level",
                            ) or "team"

                            ranking = rank_sorting_power(
                                uploaded_df,
                                sort_outcome,
                                sort_level,
                            )

                            sorting_story = explain_sorting_power(
                                ranking,
                                sort_outcome,
                                sort_level,
                            )

                            with st.container(border=True):

                                st.markdown(
                                    f"**{sorting_story['headline']}**"
                                )

                                st.markdown(
                                    "\n".join(
                                        f"- {point}"
                                        for point in sorting_story["points"]
                                    )
                                )

                            st.altair_chart(
                                sorting_power_chart(ranking),
                                width="content",
                                alt="Q12 items ranked by sorting power",
                            )

                            with st.expander(
                                "How to read this chart",
                                icon=":material/help:",
                            ):

                                st.write(
                                    sorting_story["how_to_read"]
                                )

                            sort_items = st.multiselect(
                                "Items to compare",
                                options=list(ranking["item"]),
                                default=[
                                    ranking["item"].iloc[0],
                                    ranking["item"].iloc[-1],
                                ],
                                format_func=lambda x: (
                                    f"{x} · "
                                    + ranking.set_index("item")
                                    .loc[x, "label"]
                                ),
                                key=f"sort_items_{sort_outcome}_{sort_level}",
                            )

                            if sort_items:

                                items_story = explain_sorting_items(
                                    ranking,
                                    sort_items,
                                    sort_outcome,
                                    sort_level,
                                )

                                st.markdown(
                                    "\n".join(
                                        f"- {point}"
                                        for point in items_story["points"]
                                    )
                                )

                                st.altair_chart(
                                    sorting_effect_chart(
                                        uploaded_df,
                                        sort_items,
                                        sort_outcome,
                                        sort_level,
                                    ),
                                    width="content",
                                    alt="Where high- and low-performing "
                                        "groups score on each selected item",
                                )

                                with st.expander(
                                    "How to read the dot chart",
                                    icon=":material/help:",
                                ):

                                    st.write(
                                        items_story["how_to_read"]
                                    )

                            st.caption(
                                "Descriptive only: raw scores with no "
                                "controls, from the first survey wave. "
                                "Results can differ from the Statistical "
                                "Outcome Analysis, which adjusts for "
                                "department, tenure and other factors."
                            )

                    # ---------- TEAM HEATMAP ----------

                    with heatmap_tab:

                        heatmap_teams = team_scorecard(
                            uploaded_df,
                            st.session_state.get("uploaded_analysis"),
                        )

                        if heatmap_teams.empty:

                            st.info(
                                "The team heatmap needs a team_id column and "
                                f"teams with at least {MIN_TEAM_SIZE} people."
                            )

                        else:

                            heatmap_frame = analysis_frame(uploaded_df)

                            heatmap_items = [
                                item
                                for item in heatmap_teams.columns
                                if item in heatmap_frame.columns
                                and item.startswith("Q")
                                and len(item) == 3
                            ]

                            heatmap_story = explain_team_heatmap(
                                heatmap_teams,
                                heatmap_frame[heatmap_items].mean(),
                            )

                            with st.container(border=True):

                                st.markdown(
                                    f"**{heatmap_story['headline']}**"
                                )

                                st.markdown(
                                    "\n".join(
                                        f"- {point}"
                                        for point in heatmap_story["points"]
                                    )
                                )

                            st.altair_chart(
                                team_heatmap_chart(
                                    heatmap_teams,
                                    heatmap_frame[heatmap_items].mean(),
                                ),
                                width="content",
                                alt="Team Q12 scores compared with "
                                    "the company average",
                            )

                            with st.expander(
                                "How to read this chart",
                                icon=":material/help:",
                            ):

                                st.write(
                                    heatmap_story["how_to_read"]
                                )

                except Exception as exc:

                    st.warning(
                        "Some visual insights could not be drawn."
                    )

                    st.caption(
                        f"Technical detail: {exc}"
                    )

            # -----------------------------------------
            # EXECUTIVE SUMMARY
            # -----------------------------------------

            executive_summary = build_executive_summary(
                uploaded_df,
                survey_summary=(
                    build_survey_summary(uploaded_df)
                    if validation["is_valid"]
                    else None
                ),
                analysis=st.session_state.get(
                    "uploaded_analysis"
                ),
            )

            with executive_summary_slot:

                st.subheader(
                    ":material/summarize: Executive summary"
                )

                # -----------------------------------------
                # OVERVIEW CARD (GAUGE, CHANGE, RANK, TREND)
                # -----------------------------------------

                if validation["is_valid"]:

                    scope_options = [None] + [
                        column
                        for column in SCOPE_COLUMNS
                        if column in uploaded_df.columns
                    ]

                    scope_col1, scope_col2 = st.columns([1, 2])

                    overview_scope = scope_col1.segmented_control(
                        "View",
                        options=scope_options,
                        format_func=lambda x: (
                            "Company"
                            if x is None
                            else SCOPE_COLUMNS[x]
                        ),
                        default=None,
                        key="overview_scope",
                    )

                    overview_value = None

                    if overview_scope is not None:

                        overview_value = scope_col2.selectbox(
                            SCOPE_COLUMNS[overview_scope],
                            options=sorted(
                                uploaded_df[overview_scope]
                                .dropna()
                                .unique()
                            ),
                            key=f"overview_value_{overview_scope}",
                        )

                    try:

                        overview = build_overview(
                            uploaded_df,
                            overview_scope,
                            overview_value,
                        )

                    except ValueError as exc:

                        overview = None

                        st.info(str(exc))

                    if overview is not None:

                        with st.container(border=True):

                            st.markdown(
                                "#### Overview"
                            )

                            st.caption(
                                overview["title"]
                            )

                            gauge_col, detail_col = st.columns(
                                [1, 1.2],
                                gap="large",
                            )

                            with gauge_col:

                                st.markdown(
                                    "**Employee engagement Q12 insights**"
                                )

                                st.altair_chart(
                                    score_gauge_chart(
                                        overview["current"],
                                        (
                                            overview["company_current"]
                                            if overview["scope"]
                                            else None
                                        ),
                                    ),
                                    width="content",
                                    alt=(
                                        "Gauge of the average Q12 score: "
                                        f"{overview['current']:.2f} out of 5"
                                    ),
                                )

                                gauge_note = (
                                    "Average Q12 score on a 1–5 scale, "
                                    f"{overview['period_name']} "
                                    f"{overview['current_period']} · "
                                    f"{overview['employees']:,} employees"
                                )

                                if overview["scope"]:

                                    gauge_note += (
                                        " · inner gray ring = company "
                                        f"average {overview['company_current']:.2f}"
                                    )

                                st.caption(gauge_note)

                            with detail_col:

                                if overview["delta"] is not None:

                                    st.metric(
                                        (
                                            "vs. last survey ("
                                            f"{overview['period_name']} "
                                            f"{overview['previous_period']})"
                                        ),
                                        f"{overview['current']:.2f}",
                                        delta=f"{overview['delta']:+.2f}",
                                        delta_description="points",
                                    )

                                else:

                                    st.metric(
                                        "Average Q12 score",
                                        f"{overview['current']:.2f}",
                                    )

                                    st.caption(
                                        "Only one survey period, so there "
                                        "is no change to show yet."
                                    )

                                if overview["percentile"] is not None:

                                    unit_name = SCOPE_COLUMNS[
                                        overview["scope"]
                                    ].lower()

                                    st.markdown(
                                        "**How you compare:** "
                                        f"{overview['percentile_label']} "
                                        "percentile"
                                    )

                                    st.altair_chart(
                                        percentile_meter_chart(
                                            overview["percentile"],
                                            (
                                                "Higher than "
                                                f"{overview['percentile']}% "
                                                f"of {unit_name}s"
                                            ),
                                        ),
                                        width="stretch",
                                        alt=(
                                            "Percentile rank: "
                                            f"{overview['percentile_label']}"
                                        ),
                                    )

                                    st.caption(
                                        f"Ranked against the "
                                        f"{overview['n_peers']} {unit_name}s "
                                        "in your organization, not Gallup's "
                                        "external benchmark."
                                    )

                                else:

                                    st.markdown(
                                        "**Employees engaged:** "
                                        f"{overview['pct_engaged']:.0f}%"
                                    )

                                    st.altair_chart(
                                        percentile_meter_chart(
                                            overview["pct_engaged"],
                                            (
                                                f"{overview['pct_engaged']:.1f}% "
                                                "engaged"
                                            ),
                                        ),
                                        width="stretch",
                                        alt=(
                                            "Share of employees engaged: "
                                            f"{overview['pct_engaged']:.0f}%"
                                        ),
                                    )

                                    st.caption(
                                        "Engaged = average answer of 4.0 or "
                                        "higher. Pick Department or Team "
                                        "above to see how a group ranks."
                                    )

                                if len(overview["trend"]) >= 2:

                                    st.markdown(
                                        "**Trended mean**"
                                    )

                                    st.altair_chart(
                                        trend_sparkline_chart(
                                            overview["trend"],
                                            overview["period_name"],
                                        ),
                                        width="stretch",
                                        alt=(
                                            "Average Q12 score in each "
                                            "survey period"
                                        ),
                                    )

                            st.markdown(
                                explain_overview(overview)
                            )

                st.markdown(
                    executive_summary["headline"]
                )

                st.markdown(
                    "\n".join(
                        f"- {line}"
                        for line in executive_summary[
                            "highlights"
                        ]
                    )
                )

                if executive_summary["next_step"]:

                    st.info(
                        executive_summary["next_step"],
                        icon=":material/arrow_forward:",
                    )

                st.caption(
                    "Findings describe associations, "
                    "not causal relationships."
                )

        except Exception as exc:

            st.error(
                "The uploaded file could not be processed."
            )

            st.caption(
                f"Technical detail: {exc}"
            )


# PROCESS 5B – MANAGER COACH

with coach_tab:

    st.subheader(
        "Manager Coach"
    )

    st.write(
        "A short, practical brief for one team, with an optional "
        "AI coaching note written only from the verified facts."
    )

    coach_df = (
        st.session_state.get("uploaded_data")
        if st.session_state.get(
            "uploaded_validation",
            {},
        ).get("is_valid")
        else None
    )

    coach_analysis = st.session_state.get(
        "uploaded_analysis"
    )

    # Priority teams come first, so they lead the team list.
    coach_scorecard = (
        team_scorecard(
            coach_df,
            coach_analysis,
        )
        if coach_df is not None
        else None
    )

    coach_teams = (
        coach_scorecard["team_id"].tolist()
        if coach_scorecard is not None
        and not coach_scorecard.empty
        else []
    )

    priority_teams = (
        set(
            coach_scorecard.loc[
                coach_scorecard["priority"],
                "team_id",
            ]
        )
        if coach_teams
        else set()
    )

    if coach_df is None:

        st.info(
            "Upload a dataset in Data Workspace to create "
            "manager briefs."
        )

    elif not coach_teams:

        st.warning(
            "No team has enough responses to report. "
            "The dataset needs a team_id column and teams "
            f"with at least {MIN_TEAM_SIZE} responses."
        )

    else:

        selected_team = st.selectbox(
            "Team",
            options=coach_teams,
            format_func=lambda team: (
                f"{team} · priority"
                if team in priority_teams
                else team
            ),
        )

        if coach_analysis is None:

            st.caption(
                "Run Statistical Analysis in Data Workspace to "
                "link the focus area to business outcomes."
            )

        brief = manager_brief(
            coach_df,
            selected_team,
            analysis=coach_analysis,
        )

        # -----------------------------------------
        # AI COACHING NOTE
        # -----------------------------------------

        coach_key = (
            st.session_state.get("upload_signature"),
            selected_team,
            coach_analysis is not None,
        )

        coach_notes = st.session_state.setdefault(
            "coach_notes",
            {},
        )

        if st.button(
            "Write coaching note",
            icon=":material/auto_awesome:",
        ):

            with st.spinner(
                "Writing a coaching note from verified facts..."
            ):

                coach_notes[coach_key] = coach_brief(
                    brief
                )

        coach_note = coach_notes.get(
            coach_key
        )

        with st.container(border=True):

            if (
                coach_note is not None
                and coach_note["source"] == "ai"
            ):

                st.markdown(
                    coach_note["text"]
                )

                st.caption(
                    "Written by AI from the verified brief. "
                    "All numbers were checked against the brief."
                )

            else:

                st.markdown(
                    brief_markdown(brief)
                )

                if coach_note is not None:

                    st.caption(
                        "Showing the verified brief instead. "
                        f"{coach_note['reason']}"
                    )


# PROCESS 6 – CHAT WORKSPACE

with chat_tab:

    # -----------------------------------------
    # INITIALIZE CHAT HISTORY
    # -----------------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",

                "content": (
                    "Hello. I can help you explore how Gallup Q12 "
                    "engagement factors are associated with voluntary "
                    "turnover, absenteeism, and performance.\n\n"
                    "For example, you can ask:\n"
                    "- Which engagement factors are most strongly "
                    "associated with voluntary turnover?\n"
                    "- Should we prioritize Q07 for absenteeism?\n"
                    "- How is Q12 Learn and Grow related to performance?"
                ),
            }
        ]

    # -----------------------------------------
    # TEMPORARY DATA SOURCE NOTICE
    # -----------------------------------------

    # -----------------------------------------
    # ACTIVE DATA SOURCE
    # -----------------------------------------

    if (
        "uploaded_analysis"
        in st.session_state
    ):

        active_analysis = st.session_state[
            "uploaded_analysis"
        ]

        st.success(
            "The agent is using the uploaded dataset."
        )

        st.caption(
            f"Analyzed employees: "
            f"{active_analysis['sample_size']:,}"
        )

    elif (
        "uploaded_data"
        in st.session_state
    ):

        active_analysis = None

        st.warning(
            "An uploaded dataset is available, but statistical "
            "analysis has not been run yet. Go to Data Workspace "
            "and click Run Statistical Analysis before asking "
            "outcome-related questions."
        )

    else:

        active_analysis = None

        st.info(
            "No uploaded dataset is active. "
            "The agent is currently using the development dataset."
        )

    # -----------------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------------------
    # USER INPUT
    # -----------------------------------------

    user_question = st.chat_input(
        "Ask a question about employee engagement..."
    )

    # -----------------------------------------
    # RUN AGENT
    # -----------------------------------------

    if user_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing verified evidence..."
            ):

                try:

                    if active_analysis is not None:

                        answer = ask_agent(
                            user_question,
                            analysis_override=active_analysis,
                        )

                    else:

                        answer = ask_agent(
                            user_question
                        )

                    st.markdown(
                        answer
                    )

                except Exception as exc:

                    answer = (
                        "I could not complete the analysis. "
                        "Please check the application configuration "
                        "or try again."
                    )

                    st.error(
                        answer
                    )

                    # Development-only diagnostic
                    st.caption(
                        f"Technical detail: {exc}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )