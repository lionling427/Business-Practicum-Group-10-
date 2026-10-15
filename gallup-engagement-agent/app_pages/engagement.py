# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.visualization import (
    VISUAL_CONFIG,
    build_engagement_trend,
    engagement_trend_chart,
    explain_engagement_trend,
    explain_item_results,
    explain_segment_engagement,
    item_results,
    item_results_chart,
    segment_engagement,
    segment_engagement_chart,
)
from src.tools.waves import latest_wave, wave_label
from src.ui import (
    how_to_read,
    page_header,
    require_data,
    story_box,
)


# PROCESS 2 – HEADER

page_header(
    "Engagement",
    ":material/favorite:",
    "How engaged employees are, on which questions, and in which groups.",
)

df, validation = require_data()

if not validation["is_valid"]:
    st.warning(
        "The survey data has validation issues. See **Data** for details.",
        icon=":material/warning:",
    )
    st.stop()

segments = [
    column
    for column in ["department", "role_level", "pay_band", "shift_type"]
    if column in validation["capabilities"]["profile_columns"]
]

now = wave_label(latest_wave(df))


def segment_label(column):
    return "All employees" if column is None else column.replace("_", " ").capitalize()


# PROCESS 3 – ENGAGEMENT OVER TIME

with st.container(border=True):

    title_col, control_col = st.columns([1, 2], vertical_alignment="bottom")

    title_col.subheader("Engagement over time", anchor=False)

    trend_group = control_col.segmented_control(
        "Compare by",
        options=[None] + segments,
        format_func=segment_label,
        default=None,
        key="trend_group",
        label_visibility="collapsed",
    )

    trend = build_engagement_trend(df, trend_group)
    trend_story = explain_engagement_trend(trend)

    chart_col, story_col = st.columns([2, 1], gap="large")

    with chart_col:
        st.altair_chart(
            engagement_trend_chart(trend),
            width="stretch",
            alt="Percent of employees engaged in each survey",
        )

    with story_col:
        story_box(trend_story)

    st.caption(
        f"Engaged = average answer of {VISUAL_CONFIG['engaged_threshold']:.1f} or higher "
        "across the 12 questions (an approximation of Gallup's proprietary classification). "
        f"Groups under {VISUAL_CONFIG['min_group_size']} people are hidden."
    )

    how_to_read(trend_story["how_to_read"])


# PROCESS 4 – QUESTIONS AND GROUPS, SIDE BY SIDE

question_col, group_col = st.columns(2, gap="medium")

with question_col, st.container(border=True, height="stretch"):

    st.subheader(f"Results by question · {now}", anchor=False)

    results = item_results(df)

    story_box(explain_item_results(results))

    st.altair_chart(
        item_results_chart(results),
        width="stretch",
        alt="Percent agreeing with each Q12 question in the latest survey",
    )

    st.caption(
        "Bar = % who agree or strongly agree (4–5); label = % agree · average (1–5). "
        f"{results['responses']:,} people."
    )

    with st.expander("Show table", icon=":material/table_chart:"):

        columns = ["name", "agree", "average"] + (
            ["change"] if "change" in results["table"] else []
        )

        st.dataframe(
            results["table"][columns],
            hide_index=True,
            width="stretch",
            height="content",
            column_config={
                "name": st.column_config.TextColumn("Question", width="large"),
                "agree": st.column_config.NumberColumn("% agree", format="%.0f%%"),
                "average": st.column_config.NumberColumn("Average (1–5)", format="%.2f"),
                "change": st.column_config.NumberColumn(
                    "Change in % agree", format="%+.1f pts"),
            },
            alt="Agreement and average score for each Q12 question",
        )

if segments:

    with group_col, st.container(border=True, height="stretch"):

        st.subheader(f"Engagement by group · {now}", anchor=False)

        segment = st.segmented_control(
            "Group by",
            options=segments,
            format_func=segment_label,
            default=segments[0],
            key="segment_column",
            label_visibility="collapsed",
        ) or segments[0]

        groups = segment_engagement(df, segment)

        story_box(explain_segment_engagement(groups))

        st.altair_chart(
            segment_engagement_chart(df, segment),
            width="stretch",
            alt=f"Percent engaged by {segment_label(segment).lower()} in the latest survey",
        )

        st.caption(
            "Green = at or above the company's share engaged; gray = below. "
            "Dashed line = company."
        )

        with st.expander("Show table", icon=":material/table_chart:"):

            st.dataframe(
                groups["table"][[segment, "employees", "pct_engaged", "avg_q12"]],
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    segment: segment_label(segment),
                    "employees": st.column_config.NumberColumn("People", format="%d"),
                    "pct_engaged": st.column_config.NumberColumn("% engaged", format="%.0f%%"),
                    "avg_q12": st.column_config.NumberColumn("Average Q12", format="%.2f"),
                },
                alt=f"Engagement by {segment_label(segment).lower()}",
            )
