# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.dashboard import (
    kpi_tiles,
    latest_responses,
    what_drives_results,
    where_we_stand,
)
from src.tools.visualization import (
    SCOPE_COLUMNS,
    build_overview,
    explain_overview,
    percentile_meter_chart,
    score_gauge_chart,
    trend_sparkline_chart,
)
from src.tools.waves import latest_wave, model_wave, wave_label
from src.ui import (
    analysis_runner,
    page_header,
    require_data,
)


# PROCESS 2 – HEADER

page_header(
    "Engagement overview",
    ":material/dashboard:",
    "Where engagement stands now, how it is changing, and what it is linked to.",
)

df, validation = require_data()

if not validation["is_valid"]:
    st.warning(
        "The survey data has validation issues. See **Data** for details.",
        icon=":material/warning:",
    )
    st.stop()

now = wave_label(latest_wave(df))


# PROCESS 3 – HEADLINE TILES (LATEST SURVEY)

st.subheader(
    f"Latest survey · {now} · {latest_responses(df):,} people",
    anchor=False,
)

tiles = kpi_tiles(df)

# Fixed grid so every tile has the same width.
for tile, column in zip(tiles, st.columns(len(tiles))):

    with column:

        st.metric(
            tile["label"],
            tile["value"],
            delta=tile["delta"],
            delta_color="inverse" if tile["inverse"] else "normal",
            delta_description="vs. last survey" if tile["delta"] is not None else None,
            chart_data=tile["trend"],
            chart_type="line",
            help=tile["help"],
            border=True,
        )


# PROCESS 4 – SCORECARD (COMPANY, DEPARTMENT OR TEAM)

with st.container(border=True):

    title_col, scope_col, value_col = st.columns(
        [1.2, 1.3, 1.5],
        vertical_alignment="bottom",
    )

    title_col.subheader("Scorecard", anchor=False)

    scope = scope_col.segmented_control(
        "View",
        options=[None] + [c for c in SCOPE_COLUMNS if c in df.columns],
        format_func=lambda x: "Company" if x is None else SCOPE_COLUMNS[x],
        default=None,
        key="overview_scope",
        label_visibility="collapsed",
    )

    value = None
    if scope is not None:
        value = value_col.selectbox(
            SCOPE_COLUMNS[scope],
            options=sorted(df[scope].dropna().unique()),
            key=f"overview_value_{scope}",
            label_visibility="collapsed",
        )

    try:
        card = build_overview(df, scope, value)
    except ValueError as exc:
        st.info(str(exc), icon=":material/visibility_off:")
        card = None

    if card is not None:

        ranked = card["percentile"] is not None
        columns = st.columns(3 if ranked else 2, gap="large")

        with columns[0]:

            st.markdown(f"**{card['title']}**")

            st.altair_chart(
                score_gauge_chart(
                    card["current"],
                    card["company_current"] if card["scope"] else None,
                ),
                width="content",
                alt=f"Average Q12 score {card['current']:.2f} out of 5",
            )

            st.caption(
                f"Average Q12 score (1–5) · {card['employees']:,} people"
                + (f" · gray ring = company {card['company_current']:.2f}"
                   if card["scope"] else "")
            )

        with columns[1]:

            if card["delta"] is not None:
                st.metric(
                    f"Change since {card['period_name']} {card['previous_period']}",
                    f"{card['delta']:+.2f} points",
                )
            else:
                st.metric("Average Q12 score", f"{card['current']:.2f}")

            if len(card["trend"]) >= 2:
                st.altair_chart(
                    trend_sparkline_chart(card["trend"], card["period_name"]),
                    width="stretch",
                    alt="Average Q12 score in each survey",
                )
            else:
                st.caption("Only one survey so far, so there is no trend yet.")

        if ranked:

            with columns[2]:

                unit = SCOPE_COLUMNS[card["scope"]].lower()

                st.metric(
                    "Rank in your organization",
                    f"{card['percentile_label']} percentile",
                    help=f"Higher than {card['percentile']}% of the other {unit}s. "
                    "Compared inside your organization, not with Gallup's benchmark.",
                )

                st.altair_chart(
                    percentile_meter_chart(
                        card["percentile"],
                        f"Higher than {card['percentile']}% of {unit}s",
                    ),
                    width="stretch",
                    alt=f"Percentile rank {card['percentile_label']}",
                )

                st.caption(f"Out of {card['n_peers']} {unit}s.")

        st.markdown(explain_overview(card))


# PROCESS 5 – KEY FINDINGS

analysis = st.session_state.get("uploaded_analysis")
drivers = what_drives_results(analysis, df)

with st.container(border=True):

    st.subheader("Key findings", anchor=False)

    stand_col, drive_col = st.columns(2, gap="large")

    with stand_col:

        st.markdown(f"**:material/place: Where we stand · {now}**")
        st.markdown("\n".join(f"- {point}" for point in where_we_stand(df)))

    with drive_col:

        st.markdown("**:material/query_stats: What drives results**")

        if drivers:

            st.markdown(
                "\n".join(
                    f"- **{driver['outcome']}:** {driver['headline']}"
                    for driver in drivers
                )
            )

            st.caption(
                f"Strongest verified link per outcome, from {wave_label(model_wave(df))}. "
                "Associations, not proof of cause and effect."
            )

        elif analysis is not None:

            st.caption("No engagement question passed all checks for any outcome.")

        else:

            st.caption("Run the statistical analysis below to see what drives results.")

    if drivers:
        st.info(
            "Next: open **Recommendations** for a plan for each team, or **Teams** "
            "to see where to listen first.",
            icon=":material/arrow_forward:",
        )

if analysis is None:
    analysis_runner(df, key="run_analysis_overview")
