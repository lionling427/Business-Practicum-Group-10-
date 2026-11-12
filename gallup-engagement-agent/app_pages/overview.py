# PROCESS 1 – IMPORTS

import streamlit as st
import html
import math

from src.tools.ceo_brief import SMALL_POINTS, SMALL_SCORE

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
from src.layout import hidden
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

tiles = kpi_tiles(df)

def executive_readout(tiles, wave, people, previous_wave, available_views):
    """Describe the two displayed tiles without recalculating their values."""
    by_label = {tile["label"]: tile for tile in tiles}
    score = by_label.get("Average Q12 score")
    engaged = by_label.get("Employees engaged")
    population = (f"Across {people:,} {'person' if people == 1 else 'people'} in {wave}"
                  if people is not None and wave else "In the latest survey")
    current = []
    if score and score.get("value"):
        current.append(f"the average Q12 score is {score['value']}")
    if engaged and engaged.get("value"):
        current.append(f"{engaged['value']} are classified as engaged")
    text = population + (", " + " and ".join(current) if current else
                         ", the headline measures are unavailable") + "."

    if previous_wave and score and engaged:
        score_delta, engaged_delta = score.get("delta"), engaged.get("delta")
        if (score_delta is not None and engaged_delta is not None
                and math.isfinite(score_delta) and math.isfinite(engaged_delta)):
            score_state = ("about the same" if abs(score_delta) < SMALL_SCORE else
                           "improved" if score_delta > 0 else "declined")
            engaged_state = ("unchanged" if engaged_delta == 0 else
                             "about the same" if abs(engaged_delta) < SMALL_POINTS else
                             "improved" if engaged_delta > 0 else "declined")
            score_phrase = ("remained about the same" if score_state == "about the same" else
                            f"{score_state} by {abs(score_delta):g} points")
            engaged_phrase = ("was unchanged" if engaged_state == "unchanged" else
                              "remained about the same" if engaged_state == "about the same" else
                              f"{engaged_state} by {abs(engaged_delta):g} "
                              f"percentage {'point' if abs(engaged_delta) == 1 else 'points'}")
            text += f" Compared with {previous_wave}, the average {score_phrase} and the engaged share {engaged_phrase}."
            if (score_state == "improved" and engaged_state == "declined" or
                    score_state == "declined" and engaged_state == "improved" or
                    score_state == "about the same" and engaged_state in ("improved", "declined")):
                text += " The measures present a mixed picture."
            elif score_state == engaged_state == "improved":
                text += " Both measures improved."
            elif score_state == engaged_state == "declined":
                text += " Both measures weakened."
    if available_views:
        names = " and ".join(available_views)
        ending = ("where that change is concentrated" if previous_wave else
                  "how the current results differ across groups")
        text += f" Review the {names} {'scorecards' if len(available_views) > 1 else 'scorecard'} below to determine {ending}."
    return text


previous_wave = None
if any(tile.get("delta") is not None for tile in tiles):
    from src.tools.dashboard import wave_summary
    summary_waves = wave_summary(df)["wave"]
    if len(summary_waves) >= 2:
        previous_wave = wave_label(summary_waves.iloc[-2])
views = [name for name, key in (("department", "department"), ("team", "team_id"))
         if key in df.columns]
readout = executive_readout(tiles, now, latest_responses(df), previous_wave, views)
st.markdown(html.escape(readout))

st.html("""<style>
.st-key-executive_tiles [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-executive_tiles [data-testid="stColumn"] { display: flex; }
.st-key-executive_tiles [data-testid="stMetric"] {
    box-sizing: border-box; width: 100%; height: 220px;
    display: flex; flex-direction: column;
}
.st-key-executive_tiles [data-testid="stMetric"] > :last-child { margin-top: auto; }
@media (max-width: 700px) {
    .st-key-executive_tiles [data-testid="stMetric"] { height: auto; }
}
</style>""")

# Fixed grid so every tile has the same width.
with st.container(key="executive_tiles"):
 for tile, column in zip(tiles, st.columns(len(tiles))):

    # Changes too small to matter read "About the same", not a red or green arrow.
     small = tile["delta"] is not None and abs(tile["delta"]) < (
         SMALL_POINTS if "%" in tile["value"] else SMALL_SCORE)

     with column:

         st.metric(
            tile["label"],
            tile["value"],
            delta=None if small else tile["delta"],   # no badge for tiny changes
            delta_color="off" if small else "inverse" if tile["inverse"] else "normal",
            delta_arrow="off" if small else "auto",
            delta_description="vs. last survey" if tile["delta"] is not None and not small else None,
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
                    "About the same" if abs(card["delta"]) < SMALL_SCORE
                    else f"{card['delta']:+.2f} points",
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

if hidden("overview.key_findings"):
    st.stop()  # last section: switched off in src/layout.py

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

            st.caption("Add turnover, absence or performance data to see what drives results.")

    if drivers:
        st.info(
            "Next: open **Recommendations** for a plan for each team, or **Teams** "
            "to see where to listen first.",
            icon=":material/arrow_forward:",
        )

if analysis is None:
    analysis_runner(df, key="run_analysis_overview")
