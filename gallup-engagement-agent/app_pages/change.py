# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.longitudinal import (
    INDEX,
    MODEL_LABELS,
    OUTCOME_LABELS,
    analyze_change,
    explain_item_change,
    explain_team_change,
    item_change,
    item_change_chart,
    outcome_summary,
    question_matrix,
    survey_waves,
    team_change,
    team_movers_chart,
)
from src.tools.waves import wave_label
from src.ui import (
    how_to_read,
    page_header,
    require_data,
    story_box,
)


# PROCESS 2 – HEADER

page_header(
    "Change over time",
    ":material/timeline:",
    "How engagement moved between surveys, and whether those changes go with "
    "changes in turnover, absence and performance.",
)

df, validation = require_data()

waves = survey_waves(df)

if len(waves) < 2:
    st.info(
        "This page needs at least two survey waves: a `wave` column, or "
        "`survey_date` values from two or more years.",
        icon=":material/info:",
    )
    st.stop()


# PROCESS 3 – WHAT TO COMPARE

with st.container(horizontal=True, vertical_alignment="bottom"):

    before = st.selectbox(
        "From",
        options=waves[:-1],
        index=len(waves) - 2,
        format_func=lambda w: wave_label(w).capitalize(),
        key="change_before",
        width=160,
    )

    after = st.selectbox(
        "To",
        options=[w for w in waves if w > before],
        index=len([w for w in waves if w > before]) - 1,
        format_func=lambda w: wave_label(w).capitalize(),
        key=f"change_after_{before}",
        width=160,
    )

    same_employees = st.toggle(
        "Same people only",
        value="employee_id" in df.columns,
        disabled="employee_id" not in df.columns,
        help="On: only people who answered both surveys, so joiners and leavers "
        "do not distort the change. Off: everyone who answered each survey.",
        key="change_same_employees",
    )

change = item_change(df, same_employees=same_employees, before=before, after=after)
table = change["table"]
index_row = table[table["measure"] == INDEX].iloc[0]
items = table[table["measure"] != INDEX]
who = "same people" if same_employees else "all respondents"

tiles = [
    ("Surveys compared", f"{wave_label(before).capitalize()} → {wave_label(after)[5:]}",
     None, None),
    ("People compared", f"{change['n_employees']:,}",
     None, "Answered both surveys" if same_employees else "Smaller of the two survey counts"),
    (f"Average Q12 ({who})", f"{index_row['after']:.2f}",
     float(index_row["change"]), "Average across the 12 questions for the people compared."),
    ("Clear changes",
     f"{int((items['direction'] == 'Rose').sum())} ↑  {int((items['direction'] == 'Fell').sum())} ↓",
     None, "Questions that rose (↑) or fell (↓) by more than chance would explain, "
           "after correcting for testing 12 questions."),
]

for (label, value, delta, help_text), column in zip(tiles, st.columns(len(tiles))):
    column.metric(
        label,
        value,
        delta=delta,
        delta_description=f"vs. {wave_label(before)}" if delta is not None else None,
        help=help_text,
        border=True,
    )


# PROCESS 4 – WHAT CHANGED

with st.container(border=True):

    st.subheader("What changed", anchor=False)

    change_story = explain_item_change(change)

    chart_col, story_col = st.columns([2, 1], gap="large")

    with chart_col:
        st.altair_chart(
            item_change_chart(change),
            width="stretch",
            alt="Change in average score for each Q12 question",
        )

    with story_col:
        story_box(change_story)

    how_to_read(change_story["how_to_read"])


# PROCESS 5 – TEAMS THAT MOVED MOST

teams = team_change(df, before=before, after=after)

if not teams.empty:

    with st.container(border=True):

        st.subheader("Teams that moved most", anchor=False)

        team_story = explain_team_change(teams)

        chart_col, story_col = st.columns([2, 1], gap="large")

        with chart_col:
            st.altair_chart(
                team_movers_chart(teams),
                width="stretch",
                alt="Teams with the biggest rises and falls in average engagement",
            )

        with story_col:
            story_box(team_story)

        how_to_read(team_story["how_to_read"])

        with st.expander("All teams", icon=":material/table_chart:"):

            st.dataframe(
                teams,
                hide_index=True,
                width="stretch",
                column_config={
                    "team_id": st.column_config.TextColumn("Team"),
                    "department": st.column_config.TextColumn("Department"),
                    "before": st.column_config.NumberColumn(
                        wave_label(before).capitalize(), format="%.2f"),
                    "after": st.column_config.NumberColumn(
                        wave_label(after).capitalize(), format="%.2f"),
                    "change": st.column_config.NumberColumn("Change", format="%+.2f"),
                },
                alt="Average engagement for every team in the two surveys",
            )


# PROCESS 6 – IS CHANGE LINKED TO RESULTS?

with st.container(border=True):

    st.subheader("Is change in engagement linked to results?", anchor=False)

    st.caption(
        f"Uses all {len(waves)} surveys. Each person and each team is compared with "
        "themselves over time, so stable differences between them cannot explain the "
        "result. Still an association, not proof of cause and effect."
    )

    change_analysis = st.session_state.get("change_analysis")

    if change_analysis is None:

        if st.button(
            "Run change-over-time analysis",
            type="primary",
            icon=":material/play_arrow:",
            key="run_change_analysis",
        ):

            with st.spinner("Fitting panel models..."):

                try:
                    st.session_state["change_analysis"] = analyze_change(df)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            st.toast("Change-over-time analysis complete.", icon=":material/check_circle:")
            st.rerun()

        st.stop()

    results = change_analysis["results"]

    for note in change_analysis["notes"]:
        st.caption(f":material/info: {note}")

    if results.empty:
        st.info("None of the change-over-time models could be fitted on this data.")
        st.stop()

    cards = outcome_summary(results)

    for card, column in zip(cards, st.columns(len(cards), border=True)):

        with column:

            st.caption(card["outcome"].upper())
            st.markdown(f"**{card['verdict']}**")

            for line in card["lines"]:
                badge = (":green-badge[:material/check: clear]" if line["clear"]
                         else ":gray-badge[not clear]")
                st.markdown(f"{badge} **{line['model']}:** {line['text']}")
                st.caption(line["n"])

    with st.expander("Results for each question", icon=":material/list_alt:"):

        outcomes = list(dict.fromkeys(results["outcome"]))

        outcome = st.segmented_control(
            "Outcome",
            options=outcomes,
            format_func=lambda o: OUTCOME_LABELS[o].capitalize(),
            default=outcomes[0],
            key="change_matrix_outcome",
        ) or outcomes[0]

        st.dataframe(
            question_matrix(results, outcome),
            hide_index=True,
            width="stretch",
            height="content",
            alt="Change-over-time results for each question",
        )

        st.caption(
            "✓ = clear link after correcting for testing 12 questions; · = not clear. "
            "Effects are per +1 point on that question."
        )

    with st.expander("Technical details", icon=":material/functions:"):

        st.dataframe(
            results.assign(model=results["model"].map(MODEL_LABELS))[[
                "model", "outcome", "name", "effect_type", "effect",
                "ci_low", "ci_high", "p_value", "clear_link", "n_units",
            ]],
            hide_index=True,
            width="stretch",
            column_config={
                "effect": st.column_config.NumberColumn(format="%.3f"),
                "ci_low": st.column_config.NumberColumn("95% CI low", format="%.3f"),
                "ci_high": st.column_config.NumberColumn("95% CI high", format="%.3f"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
            },
            alt="All change-over-time model results",
        )

    how_to_read(
        "**Same people** compares each person with themselves: when their engagement "
        "rises between surveys, does their own performance or absence change too? "
        "**Same teams** does the same for team averages, including turnover rates. "
        "**Score drop** looks at people with the same current score and asks whether "
        "those whose score fell since the last survey were more likely to leave. "
        "A clear link means the result is unlikely to be chance; it does not prove "
        "that raising engagement causes the change.",
        "How these models work",
    )
