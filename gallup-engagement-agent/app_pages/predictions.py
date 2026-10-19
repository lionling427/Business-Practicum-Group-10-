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
from src.tools.survival import (
    HORIZON,
    demo_person_month,
    explain_monthly_risk,
    monthly_chart,
    monthly_risk,
)
from src.tools.ts_readiness import readiness
from src.tools.visualization import SCOPE_COLUMNS
from src.ui import (
    how_to_read,
    readiness_panel,
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
            risk = employee_risk_by_group(
                df, group, health_report=st.session_state.get("health_report"))
        except ValueError as exc:
            risk = None
            st.info(str(exc), icon=":material/info:")

    if risk is not None and risk["status"] != "useful":

        st.warning(f"{risk['status_text']} Group risk is hidden.",
                   icon=":material/visibility_off:")

    elif risk is not None:

        table = risk["table"]
        show_pct = risk.get("show_percentages", True)
        fairness = ("Pending" if table["actual_rate"].isna().all()
                    else "Passed" if not table["fairness_flag"].any()
                    else f"{int(table['fairness_flag'].sum())} flagged")

        tiles = [
            ("Expected leavers, next 12 months",
             f"~{table['expected_leavers'].sum():,.0f}" if show_pct else "Hidden",
             "Sum of each group's predicted leaving rate × its people." if show_pct else
             risk["calibration"]["text"]),
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

        if risk.get("survey_response_warning"):
            st.caption(":material/info: " + risk["survey_response_warning"]["text"])
        st.caption(":material/fact_check: " + risk["calibration"]["text"])

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


# PROCESS 5 – MONTHLY LEAVING RISK (TIME SERIES)

raw = st.session_state.get("uploaded_raw", df)
upload_ready = readiness(raw)["ready_for_survival"]

with st.container(border=True):

    title_col, source_col = st.columns([1.3, 2], vertical_alignment="bottom")
    title_col.subheader("Monthly leaving risk", anchor=False)

    sources = (["upload"] if upload_ready else []) + ["file", "demo"]
    source = source_col.segmented_control(
        "Data", options=sources, default=sources[0], key="survival_source",
        format_func={"upload": "My upload", "file": "Person-month file",
                     "demo": "Demo data"}.get,
        label_visibility="collapsed") or sources[0]

    st.caption(f"Who is likely to leave in each of the next {HORIZON} months, from monthly "
               "history (one row per employee per month). Tested by rolling backtests on "
               "months the model never saw.")

    monthly = None
    if source == "upload":
        monthly, key = raw, ("upload", st.session_state.get("upload_signature"))
    elif source == "demo":
        monthly, key = demo_person_month(), ("demo",)
        st.caption(":material/science: Demo data: 1,500 synthetic employees over 24 months.")
    else:
        file = st.file_uploader("Person-month file (CSV)", type=["csv"], key="person_month_file",
                                help="One row per employee per month: employee_id, period "
                                "(e.g. 2026-03), exited (1 in the month they leave) or "
                                "exit_date, plus team_id, department, tenure_years, "
                                "absence_days and pulse answers Q01–Q12 if you have them. "
                                "An example file is on the Data page.")
        if file is not None:
            import pandas as pd
            monthly, key = pd.read_csv(file), ("file", file.name, file.size)
        else:
            st.info("Upload a person-month file, or choose **Demo data** to see how it works.",
                    icon=":material/upload_file:")

    if monthly is not None:

        cache = st.session_state.setdefault("survival_results", {})
        if key not in cache:
            with st.spinner("Testing the model on past months and looking ahead..."):
                try:
                    cache[key] = monthly_risk(
                        monthly,
                        health_report=(st.session_state.get("health_report")
                                       if source == "upload" else None))
                except (ValueError, KeyError) as exc:
                    cache[key] = {"status": "error", "status_text": str(exc)}
        result = cache[key]

        if result["status"] == "not_ready":
            readiness_panel(result["readiness"])
        elif result["status"] != "useful":
            st.warning(result["status_text"], icon=":material/warning:")
            if "tests" in result:
                st.dataframe(result["tests"], hide_index=True, width="stretch")
        else:
            tiles = [
                (f"Likely leavers, next {HORIZON} months", f"~{result['expected_leavers']:,.0f}",
                 f"Of {result['people_at_risk']:,} people employed in the latest month."),
                ("Model accuracy", f"{result['auc']:.0%}",
                 "How often it ranks a leaver above a stayer, in backtests (chance = 50%)."),
                ("Simple baseline", f"{result['baseline_auc']:.0%}",
                 "The same test using each team's recent exit rate."),
                ("Predicted vs actual leavers", f"{result['calibration']:.0%}",
                 "In backtests, predicted leavers as a share of those who actually left "
                 "(100% = exactly right)."),
            ]
            for (label, value, help_text), column in zip(tiles, st.columns(len(tiles))):
                column.metric(label, value, help=help_text, border=True)

            chart_col, story_col = st.columns([3, 2], gap="large")
            with chart_col:
                st.altair_chart(monthly_chart(result), width="stretch",
                                alt="Expected leavers in each of the next months")
                st.caption("Bars = expected leavers each month; dashed line = running total.")
            with story_col:
                story_box(explain_monthly_risk(result))

            with st.expander("By department and team", icon=":material/table_chart:"):
                count_columns = {
                    "people": st.column_config.NumberColumn("People", format="%d"),
                    "average_chance": st.column_config.NumberColumn(
                        f"Chance of leaving in {HORIZON} months", format="percent"),
                    "expected_leavers": st.column_config.NumberColumn(
                        "Expected leavers", format="%.1f")}
                hide = [] if result.get("show_percentages", True) else ["average_chance",
                                                                         "expected_leavers"]
                if hide:
                    st.caption(result["calibration_check"]["text"])
                if not result["by_department"].empty:
                    st.dataframe(result["by_department"].drop(columns=hide), hide_index=True,
                                 width="stretch",
                                 column_config={"department": "Department", **count_columns})
                if not result["by_team"].empty:
                    st.dataframe(result["by_team"].head(20).drop(columns=hide), hide_index=True,
                                 width="stretch",
                                 column_config={"team_id": "Team", **count_columns})
                st.caption("Groups of 5 or more people only; no individual scores are shown.")

            with st.expander("How the model was tested", icon=":material/verified:"):
                st.dataframe(result["tests"].rename(columns={
                    "cutoff_month": "Cut-off month", "people": "People", "left": "Left",
                    "expected": "Predicted", "model_auc": "Model accuracy",
                    "baseline_auc": "Baseline accuracy"}), hide_index=True, width="stretch")
                st.dataframe(result["drivers"].rename(columns={
                    "feature": "Signal", "odds_change_per_sd": "Change in monthly odds of leaving"}),
                    hide_index=True, width="stretch",
                    column_config={"Change in monthly odds of leaving":
                                   st.column_config.NumberColumn(format="percent")})

        how_to_read(
            "Each month, the model estimates each employee's chance of leaving that month from "
            "their engagement, its recent change, tenure, absence, recent exits in their team "
            "and manager changes. It only uses what was known before that month. Chances are "
            f"added up into expected leavers per month and per group for the next {HORIZON} "
            "months, assuming conditions stay as they are now. Likelihoods, not certainties.",
            "How monthly leaving risk works",
        )
