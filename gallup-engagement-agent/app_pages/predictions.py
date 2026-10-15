# PROCESS 1 – IMPORTS

import streamlit as st

from src.tools.prediction import (
    MIN_FORECAST_WAVES,
    MIN_GROUP,
    employee_risk_by_group,
    explain_forecast,
    explain_group_risk,
    forecast_chart,
    forecast_engagement,
    group_risk_chart,
)
from src.tools.visualization import SCOPE_COLUMNS
from src.ui import (
    how_to_read,
    page_header,
    require_data,
    story_box,
)


# PROCESS 2 – HEADER

page_header(
    "Predictions",
    ":material/online_prediction:",
    "Where turnover risk is concentrated in the next 12 months, and where engagement "
    "is heading. Every prediction is tested on data the model never saw.",
)

df, validation = require_data()

groups = [c for c in ["department", "role_level", "pay_band", "shift_type"] if c in df.columns]


def group_label(column):
    return column.replace("_", " ").capitalize()


# PROCESS 3 – TURNOVER RISK BY GROUP

with st.container(border=True):

    title_col, control_col = st.columns([1.3, 2], vertical_alignment="bottom")

    title_col.subheader("Turnover risk by group", anchor=False)

    if not groups:
        st.info("Group views need a column such as department or role level.")
        risk = None
    else:
        group = control_col.segmented_control(
            "Group by",
            options=groups,
            format_func=group_label,
            default=groups[0],
            key="risk_group",
            label_visibility="collapsed",
        ) or groups[0]

        try:
            risk = employee_risk_by_group(df, group)
        except ValueError as exc:
            risk = None
            st.info(str(exc), icon=":material/info:")

    if risk is not None and risk["status"] != "useful":

        st.warning(f"{risk['status_text']} Group risk is hidden.",
                   icon=":material/visibility_off:")

    elif risk is not None:

        table = risk["table"]
        fairness = ("Pending" if table["actual_rate"].isna().all()
                    else "Passed" if not table["fairness_flag"].any()
                    else f"{int(table['fairness_flag'].sum())} flagged")

        tiles = [
            ("Expected leavers, next 12 months", f"~{table['expected_leavers'].sum():,.0f}",
             "Sum of each group's predicted leaving rate × its people."),
            ("Model accuracy", f"{risk['auc'] * 100:.0f}%",
             "How often the model picks the leaver out of two people (50% = coin flip). "
             + ("Tested on a survey it never saw." if risk["method"] == "out_of_time"
                else "Checked by cross-validation within one survey.")),
            ("Fairness check", fairness,
             "Is the model about as accurate and well-calibrated for every group?"),
        ]

        for (label, value, help_text), column in zip(tiles, st.columns(len(tiles))):
            column.metric(label, value, help=help_text, border=True)

        chart_col, story_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.altair_chart(
                group_risk_chart(risk),
                width="stretch",
                alt=f"Share of people at high turnover risk by {group_label(group).lower()}",
            )

        with story_col:
            story_box(explain_group_risk(risk))

        st.caption(
            ":material/lock: Privacy by design: individual scores are never shown or stored, "
            f"only groups of {MIN_GROUP}+ people. The model uses only survey answers and "
            "tenure (no demographic or pay data)."
        )

        with st.expander("Details and fairness check", icon=":material/table_chart:"):

            st.dataframe(
                table.assign(
                    predicted_rate=lambda d: d["predicted_rate"] * 100,
                    actual_rate=lambda d: d["actual_rate"] * 100,
                    high_risk_share=lambda d: d["high_risk_share"] * 100,
                    calibration_gap=lambda d: d["calibration_gap"] * 100,
                )[[group, "employees", "high_risk_share", "predicted_rate",
                   "expected_leavers", "actual_rate", "calibration_gap", "auc",
                   "fairness_flag"]],
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    group: group_label(group),
                    "employees": st.column_config.NumberColumn("People", format="%d"),
                    "high_risk_share": st.column_config.NumberColumn(
                        "% at high risk", format="%.0f%%"),
                    "predicted_rate": st.column_config.NumberColumn(
                        "Predicted leaving rate", format="%.1f%%"),
                    "expected_leavers": st.column_config.NumberColumn(
                        "Expected leavers", format="%d"),
                    "actual_rate": st.column_config.NumberColumn(
                        "Actual rate (test survey)", format="%.1f%%"),
                    "calibration_gap": st.column_config.NumberColumn(
                        "Predicted − actual", format="%+.1f pts"),
                    "auc": st.column_config.NumberColumn("Accuracy (AUC)", format="%.2f"),
                    "fairness_flag": st.column_config.CheckboxColumn("Fairness flag"),
                },
                alt="Turnover risk and fairness check by group",
            )

        how_to_read(
            "The model estimates each person's chance of leaving in the next 12 months "
            "from their survey answers and tenure, then adds them up by group. Bars show "
            "the share of each group that falls in the company's riskiest fifth: the dashed "
            "line at 20% is what an average group would have, red bars are above it. "
            "Use this to decide where to listen first, never to judge individuals."
        )

st.caption(":material/groups: Team-by-team risk is on the **Teams** page.")


# PROCESS 4 – ENGAGEMENT FORECAST

with st.container(border=True):

    title_col, scope_col, value_col = st.columns([1.3, 1.2, 1.5], vertical_alignment="bottom")

    title_col.subheader("Engagement forecast", anchor=False)

    scope = scope_col.segmented_control(
        "Forecast for",
        options=[None] + [c for c in SCOPE_COLUMNS if c in df.columns],
        format_func=lambda x: "Company" if x is None else SCOPE_COLUMNS[x],
        default=None,
        key="forecast_scope",
        label_visibility="collapsed",
    )

    value = None
    if scope is not None:
        value = value_col.selectbox(
            SCOPE_COLUMNS[scope],
            options=sorted(df[scope].dropna().unique()),
            key=f"forecast_value_{scope}",
            label_visibility="collapsed",
        )

    forecast = forecast_engagement(df, scope, value)

    if not forecast["available"]:

        st.info(explain_forecast(forecast), icon=":material/hourglass_empty:")
        st.caption(f"Add more survey waves ({MIN_FORECAST_WAVES}+) and this section will "
                   "forecast the next one automatically.")

    else:

        error = forecast["backtest_mae"][forecast["method"]]
        tiles = [
            (f"Next survey (wave {forecast['next_wave']:g})", f"{forecast['forecast']:.2f}",
             "Most likely average Q12 score."),
            ("Likely range", f"{forecast['low']:.2f} – {forecast['high']:.2f}",
             "The score should land in this range about 8 times out of 10."),
            ("Typical forecast error", f"±{error:.2f}",
             "How far off this method was, on average, when tested on your past surveys."),
        ]

        for (label, val, help_text), column in zip(tiles, st.columns(len(tiles))):
            column.metric(label, val, help=help_text, border=True)

        chart_col, story_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.altair_chart(
                forecast_chart(forecast),
                width="stretch",
                alt="Average Q12 score in each survey and the forecast for the next one",
            )

        with story_col:
            with st.container(border=True):
                st.markdown(explain_forecast(forecast))

        how_to_read(
            "The dark line is the actual average score in each survey. The green dot is the "
            "forecast for the next survey, and the green bar its likely range. Two methods "
            "were tested on your past surveys (extending the trend, or assuming the latest "
            "score holds) and the more accurate one is used. It assumes surveys are evenly "
            "spaced and nothing major changes."
        )
