"""Data for the Overview page: headline tiles and key findings.

Everything describes the latest survey, with the change since the survey
before it. Drivers come from the statistical models (model wave, see waves.py).
Labels come from insights.ITEM_LABELS so wording matches every other page.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import pandas as pd

from .insights import ITEM_LABELS, OUTCOME_INFO, analysis_frame, explain_finding, top_drivers
from .visualization import add_engagement_status
from .waves import latest_wave, wave_label


# PROCESS 2 – CONFIGURATION

# label, column, format, good direction ("up"/"down"), help text
TILES = [
    ("Average Q12 score", "q12_average", "{:.2f}", "up",
     "Average answer across the 12 questions (1–5)."),
    ("Employees engaged", "pct_engaged", "{:.0f}%", "up",
     "Share whose average answer is 4.0 or higher."),
    ("Voluntary turnover", "turnover", "{:.1f}%", "down",
     "Share who left voluntarily within 12 months of the survey."),
    ("Unplanned absence", "absence", "{:.1f} days", "down",
     "Average unplanned absence days per employee over 6 months."),
    ("Performance rating", "performance", "{:.2f}", "up",
     "Average performance rating (1–5)."),
]


# PROCESS 3 – PER-WAVE SUMMARY

def wave_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per survey wave: responses, Q12 average, % engaged, outcomes."""
    data = add_engagement_status(df)
    if "wave" not in data.columns:
        data = data.assign(wave=1)
    agg = {"responses": ("q12_mean", "size"), "q12_average": ("q12_mean", "mean"),
           "pct_engaged": ("engaged", "mean")}
    for key, info in OUTCOME_INFO.items():
        if info["column"] in data.columns:
            agg[key] = (info["column"], "mean")
    table = data.groupby("wave").agg(**agg).reset_index().sort_values("wave")
    table["pct_engaged"] *= 100
    if "turnover" in table:
        table["turnover"] *= 100
    return table.reset_index(drop=True)


def latest_responses(df: pd.DataFrame) -> int:
    return int(wave_summary(df)["responses"].iloc[-1])


def kpi_tiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Headline tiles for the latest survey, each with change and trend."""
    table = wave_summary(df)
    latest = table.iloc[-1]
    previous = table.iloc[-2] if len(table) >= 2 else None
    tiles = []
    for label, key, fmt, good, help_text in TILES:
        if key not in table or pd.isna(latest[key]):
            continue
        delta = None
        if previous is not None and not pd.isna(previous[key]):
            # Percent tiles change in points (1 decimal); scores in 2 decimals.
            digits = 1 if "%" in fmt else 2
            delta = round(float(latest[key] - previous[key]), digits)
        tiles.append({
            "label": label,
            "value": fmt.format(latest[key]),
            "delta": delta,
            "help": help_text,
            "trend": table[key].round(3).tolist() if len(table) >= 2 else None,
            "inverse": good == "down",
        })
    return tiles


# PROCESS 4 – KEY FINDINGS

def where_we_stand(df: pd.DataFrame) -> list[str]:
    """Three plain bullets about the latest survey."""
    data = add_engagement_status(df)
    wave = latest_wave(data)
    if wave is not None:
        data = data[data["wave"] == wave]
    items = [i for i in ITEM_LABELS if i in data.columns]
    means = data[items].mean().sort_values(ascending=False)
    agree = (data[items] >= 4).mean() * 100

    top, bottom = means.index[0], means.index[-1]
    points = [
        f"**Strongest:** {ITEM_LABELS[top]} ({top}) at {means[top]:.2f}, with "
        f"{agree[top]:.0f}% agreeing.",
        f"**Biggest opportunity:** {ITEM_LABELS[bottom]} ({bottom}) at {means[bottom]:.2f}; "
        f"only {agree[bottom]:.0f}% agree.",
    ]

    table = wave_summary(df)
    if len(table) >= 2:
        change = table["pct_engaged"].iloc[-1] - table["pct_engaged"].iloc[-2]
        first = table.iloc[0]
        trend = ("about the same as" if abs(change) < 1
                 else f"{'up' if change > 0 else 'down'} {abs(change):.1f} points from")
        points.append(
            f"**Engagement:** {table['pct_engaged'].iloc[-1]:.0f}% engaged, {trend} the "
            f"last survey. It was {first['pct_engaged']:.0f}% in {wave_label(first['wave'])}.")
    return points


def what_drives_results(analysis: dict | None, df: pd.DataFrame) -> list[dict[str, str]]:
    """Strongest verified driver per outcome, in plain language."""
    if analysis is None:
        return []
    drivers = []
    for outcome, row in top_drivers(analysis).items():
        explained = explain_finding(row, df)
        drivers.append({"outcome": OUTCOME_INFO[outcome]["label"],
                        "question": f"{explained['label']} ({explained['item']})",
                        "headline": explained["headline"]})
    return drivers


# PROCESS 5 – OUTCOMES PAGE: WHICH QUESTIONS MATTER FOR EACH OUTCOME

CAUTION_PLAIN = {
    "PRIOR_OUTCOME_SENSITIVE": "Much of this link is explained by past performance.",
    "TEAM_LEVEL_CONFOUNDING": "Mostly reflects differences between teams, not people.",
    "DIRECTION_FLIP": "The direction changes once departments are compared fairly.",
}


def driver_rows(analysis: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Every reportable finding as one plain-language row with a bar size.

    size = % lower odds of leaving (turnover), % fewer absence days (absence),
    or rating points per 1-point higher score (performance). Positive = good.
    """
    findings = analysis["findings"]
    frame = analysis_frame(df)
    rows = []
    for _, f in findings.iterrows():
        explained = explain_finding(f, df)
        effect = float(f["effect"])
        if f["outcome"] == "performance":
            # Standardized beta -> rating points per 1-point higher score.
            sd_item = float(frame[f["item"]].std())
            size = effect * float(frame["performance_rating"].std()) / sd_item if sd_item else 0.0
            size_text = f"{size:+.2f}"
        else:
            size = (1 - effect) * 100
            better, worse = ("lower", "higher") if f["outcome"] == "turnover" else ("fewer", "more")
            size_text = f"{abs(size):.0f}% {better if size >= 0 else worse}"
        codes = f.get("diagnostic_codes", [])
        codes = codes if isinstance(codes, (list, tuple)) else []
        rows.append({
            "outcome": f["outcome"],
            "item": f["item"],
            "question": f"{f['item']} · {explained['label']}",
            "size": size,
            "size_text": size_text,
            "reliability": "Passed all checks" if explained["is_robust"] else "Use with caution",
            "caution": " ".join(CAUTION_PLAIN.get(c, "") for c in codes).strip(),
            "headline": explained["headline"],
            "example": explained["example"],
        })
    return pd.DataFrame(rows, columns=["outcome", "item", "question", "size", "size_text",
                                       "reliability", "caution", "headline", "example"])


DRIVER_AXIS = {
    "turnover": "% lower odds of leaving",
    "absence": "% fewer absence days",
    "performance": "Rating points",
}


def driver_chart(rows: pd.DataFrame, outcome: str):
    """Ranked bars for one outcome, colored by reliability."""
    import altair as alt

    data = rows[rows["outcome"] == outcome].sort_values("size", ascending=False)
    y = alt.Y("question:N", sort=list(data["question"]), title=None,
              axis=alt.Axis(labelLimit=170, labelOverlap=False))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=y,
        x=alt.X("size:Q", title=DRIVER_AXIS[outcome],
                # Headroom for the value labels; room on the left for links that
                # go the other way (negative bars) so they never cover the names.
                scale=alt.Scale(domain=[min(0.0, float(data["size"].min()) * 1.3),
                                        max(0.01, float(data["size"].max())
                                            * (2.4 if (data["size"] < 0).any() else 1.8))]),
                axis=alt.Axis(tickCount=4)),
        color=alt.Color("reliability:N", title=None,
                        scale=alt.Scale(domain=["Passed all checks", "Use with caution"],
                                        range=["#2E9E3E", "#B9B1A8"]),
                        legend=None),
        tooltip=[alt.Tooltip("question:N", title="Question"),
                 alt.Tooltip("headline:N", title="What it means"),
                 alt.Tooltip("reliability:N", title="Reliability"),
                 alt.Tooltip("caution:N", title="Why caution")])
    # Labels sit just right of the bar end (or of zero for bars going left).
    labels = alt.Chart(data).transform_calculate(label_x="max(datum.size, 0)").mark_text(
        align="left", dx=5, fontSize=12).encode(y=y, x="label_x:Q", text="size_text:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#999").encode(x="x:Q")
    return (bars + zero + labels).properties(height=alt.Step(30), width="container")



# PROCESS 7 – TEAMS PAGE: ONE WATCHLIST

BAND_RANK = {"High": 2, "Elevated": 1, "Typical": 0}


def team_watchlist(scorecard: pd.DataFrame, risk_results: dict | None) -> pd.DataFrame:
    """Every reportable team in one list, most urgent first.

    Urgency = priority flag (3) + 2 per High risk band + 1 per Elevated band.
    Risk bands come only from models that passed their test.
    """
    teams = scorecard.copy()
    for outcome in ["turnover", "absence"]:
        result = (risk_results or {}).get(outcome)
        column = f"{outcome}_risk"
        if result is not None and result.get("status") == "useful":
            bands = result["teams"].set_index("team_id")
            teams[column] = teams["team_id"].map(bands["risk_band"]).fillna("—")
            teams[f"{outcome}_risk_pct"] = teams["team_id"].map(bands["risk"]) * 100
        else:
            teams[column] = "—"
    teams["urgency"] = (teams["priority"].astype(int) * 3
                        + teams["turnover_risk"].map(BAND_RANK).fillna(0) * 1
                        + teams["absence_risk"].map(BAND_RANK).fillna(0) * 1)
    teams["needs_attention"] = teams["urgency"] > 0
    keep = ["team_id", "department", "employees", "q12_average", "turnover_risk",
            "absence_risk", "focus_label", "priority", "urgency", "needs_attention"]
    keep += [c for c in ITEM_LABELS if c in teams.columns]
    return (teams.sort_values(["urgency", "q12_average"], ascending=[False, True])
            [[c for c in keep if c in teams.columns]].reset_index(drop=True))
