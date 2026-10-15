# PROCESS 1 – IMPORTS
"""Gallup-style engagement visuals.

1. Engagement trend:  % of employees "engaged" per survey wave, by group
   (like Gallup's "Annual Employee Engagement" chart).
2. Sorting effect:    for one Q12 item, where high- and low-performing teams
   (or employees) sit on the disagree-to-agree scale, plus a "sorting power"
   score that ranks all 12 items (like Gallup's "Great item / Poor item" slide).

Data functions return plain DataFrames (easy to test and to hand to the agent);
chart functions turn them into Altair charts, which Streamlit already ships with.
"""

from __future__ import annotations

import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from .data_ingestion import OUTCOME_COLUMNS, Q12_ITEMS
from .waves import model_wave


# PROCESS 2 – CONFIGURATION

ITEM_LABELS = {
    "Q01": "Know what is expected", "Q02": "Materials and equipment",
    "Q03": "Opportunity to do best", "Q04": "Recognition",
    "Q05": "Someone cares", "Q06": "Development encouraged",
    "Q07": "Opinions count", "Q08": "Mission and purpose",
    "Q09": "Co-workers committed to quality", "Q10": "Best friend at work",
    "Q11": "Progress conversations", "Q12": "Learn and grow",
}

VISUAL_CONFIG = {
    # Approximation only: Gallup's official engaged/not engaged/actively
    # disengaged classification is proprietary.
    "engaged_threshold": 4.0,       # mean Q12 score at or above this = engaged
    "min_items_answered": 9,        # need at least 9 of 12 items to classify
    "min_group_size": 5,            # suppress smaller groups (privacy rule)
    "min_team_size": 5,             # teams smaller than this are left out
    "extreme_share": 0.25,          # top vs bottom 25% of teams
    "max_points": 60,               # dots per group in the sorting chart
}

HIGH_COLOR, LOW_COLOR = "#2E9E3E", "#3D3335"     # Gallup-like green / dark brown
OVERALL_COLOR = "#1B2A4A"                         # "All employees" line (navy, like Gallup U.S.)
LINE_COLORS = ["#2E8B3E", "#2F7FA8", "#B5473F", "#8C6BB1",
               "#D08C2E", "#5B8C85", "#C2549A", "#7A7A7A"]

# How to split each outcome into "high performers" and "low performers".
# higher_is_better=False means a LOW value is the good side.
OUTCOME_RULES = {
    "performance": {"column": "performance_rating", "higher_is_better": True,
                    "label": "performance rating"},
    "turnover": {"column": "left_voluntarily_12m", "higher_is_better": False,
                 "label": "voluntary turnover"},
    "absence": {"column": "unplanned_absence_days_6m", "higher_is_better": False,
                "label": "unplanned absence"},
}


# PROCESS 3 – HELPERS

def _numeric_items(df: pd.DataFrame) -> pd.DataFrame:
    items = [i for i in Q12_ITEMS if i in df.columns]
    return df[items].apply(pd.to_numeric, errors="coerce")


def _respondents(df: pd.DataFrame) -> pd.DataFrame:
    """Keep completed surveys only (drops nonresponders if 'responded' exists)."""
    out = df.copy()
    if "responded" in out.columns:
        out = out[pd.to_numeric(out["responded"], errors="coerce") == 1]
    answered = _numeric_items(out).notna().sum(axis=1)
    return out[answered >= VISUAL_CONFIG["min_items_answered"]].copy()


def add_engagement_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add q12_mean and an 'engaged' flag (1/0) to every completed survey."""
    out = _respondents(df)
    out["q12_mean"] = _numeric_items(out).mean(axis=1)
    out["engaged"] = (out["q12_mean"] >= VISUAL_CONFIG["engaged_threshold"]).astype(int)
    return out


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Use 'wave' if there are 2+ waves; otherwise survey year if 2+ years."""
    if "wave" in df.columns and df["wave"].nunique(dropna=True) >= 2:
        return "wave"
    if "survey_date" in df.columns:
        years = pd.to_datetime(df["survey_date"], errors="coerce").dt.year
        if years.nunique(dropna=True) >= 2:
            return "survey_year"
    return None


def _auc(high: pd.Series, low: pd.Series) -> float:
    """Chance a random high performer scores higher on the item than a random
    low performer (ties count half). 0.5 = no sorting, 1.0 = perfect sorting."""
    high, low = high.dropna(), low.dropna()
    if len(high) == 0 or len(low) == 0:
        return float("nan")
    ranks = pd.concat([high, low]).rank(method="average")
    r_high = ranks.iloc[: len(high)].sum()
    return float((r_high - len(high) * (len(high) + 1) / 2) / (len(high) * len(low)))


# PROCESS 4 – ENGAGEMENT TREND DATA

def build_engagement_trend(
    df: pd.DataFrame,
    group_column: str | None = None,
) -> dict[str, Any]:
    """% engaged per period (wave / year) and group.

    Returns {"table", "time_column", "has_trend", "group_column"}.
    With only one period the table has one row per group (no trend).
    """
    data = add_engagement_status(df)
    time_col = detect_time_column(data)
    if time_col == "survey_year":
        data["survey_year"] = pd.to_datetime(data["survey_date"], errors="coerce").dt.year

    keys = [k for k in (time_col, group_column) if k]
    if not keys:
        data["_all"] = "All employees"
        keys = ["_all"]

    table = (data.groupby(keys, dropna=False)
             .agg(employees=("employee_id", "nunique"),
                  pct_engaged=("engaged", "mean"),
                  avg_q12=("q12_mean", "mean"))
             .reset_index())
    table["pct_engaged"] = table["pct_engaged"] * 100
    table = table[table["employees"] >= VISUAL_CONFIG["min_group_size"]]

    # Always add an "All employees" line when a group is chosen.
    if group_column and time_col:
        overall = (data.groupby(time_col)
                   .agg(employees=("employee_id", "nunique"),
                        pct_engaged=("engaged", "mean"),
                        avg_q12=("q12_mean", "mean"))
                   .reset_index())
        overall["pct_engaged"] *= 100
        overall[group_column] = "All employees"
        table = pd.concat([overall, table], ignore_index=True)

    return {"table": table.round({"pct_engaged": 1, "avg_q12": 2}),
            "time_column": time_col, "has_trend": time_col is not None,
            "group_column": group_column}


# PROCESS 5 – SORTING EFFECT DATA

def build_sorting_groups(
    df: pd.DataFrame,
    outcome: str = "performance",
    level: str = "team",
) -> pd.DataFrame:
    """Label each unit (team or employee) as 'High performers' / 'Low performers'.

    level="team": team averages; top vs bottom 25% of teams on the outcome
                  (closest to Gallup's slide, which compares teams).
    level="employee": individual employees; performance 4-5 vs 1-2,
                  stayed vs left, or lowest vs highest 25% absence.

    Uses the model wave only (newest wave with outcomes), matching the outcome models.
    """
    rule = OUTCOME_RULES[outcome]
    col = rule["column"]
    if col not in df.columns:
        raise ValueError(f"{col} is not in the data, so {outcome} cannot be used.")

    data = _respondents(df)
    # One wave only: the same survey the outcome models use, so each
    # employee and team is counted once.
    wave = model_wave(data)
    if wave is not None:
        data = data[data["wave"] == wave].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])
    items = [i for i in Q12_ITEMS if i in data.columns]
    data[items] = data[items].apply(pd.to_numeric, errors="coerce")

    if level == "team":
        if "team_id" not in data.columns:
            raise ValueError("team_id is needed for team-level sorting.")
        team_keys = ["team_id"] + (["wave"] if "wave" in data.columns else [])
        units = (data.groupby(team_keys)
                 .agg(size=("employee_id", "nunique"), outcome=(col, "mean"),
                      **{i: (i, "mean") for i in items})
                 .reset_index())
        units = units[units["size"] >= VISUAL_CONFIG["min_team_size"]]
        share = VISUAL_CONFIG["extreme_share"]
        lo_cut, hi_cut = units["outcome"].quantile([share, 1 - share])
        good_high = units["outcome"] >= hi_cut
        good_low = units["outcome"] <= lo_cut
        if not rule["higher_is_better"]:
            good_high, good_low = good_low, good_high
    else:
        units = data[["employee_id"] + items + [col]].rename(columns={col: "outcome"})
        if outcome == "performance":
            good_high, good_low = units["outcome"] >= 4, units["outcome"] <= 2
        elif outcome == "turnover":
            good_high, good_low = units["outcome"] == 0, units["outcome"] == 1
        else:
            lo_cut, hi_cut = units["outcome"].quantile([0.25, 0.75])
            good_high, good_low = units["outcome"] <= lo_cut, units["outcome"] >= hi_cut

    units["group"] = np.select([good_high, good_low],
                               ["High performers", "Low performers"], default=None)
    return units.dropna(subset=["group"]).reset_index(drop=True)


def rank_sorting_power(
    df: pd.DataFrame,
    outcome: str = "performance",
    level: str = "team",
) -> pd.DataFrame:
    """Rank all Q12 items by how well they separate high from low performers."""
    units = build_sorting_groups(df, outcome, level)
    high = units[units["group"] == "High performers"]
    low = units[units["group"] == "Low performers"]
    rows = []
    for item in [i for i in Q12_ITEMS if i in units.columns]:
        auc = _auc(high[item], low[item])
        rows.append({"item": item, "label": ITEM_LABELS.get(item, item),
                     "high_mean": high[item].mean(), "low_mean": low[item].mean(),
                     "gap": high[item].mean() - low[item].mean(),
                     "sorting_power": auc,
                     "n_high": int(high[item].notna().sum()),
                     "n_low": int(low[item].notna().sum())})
    out = pd.DataFrame(rows).sort_values("sorting_power", ascending=False)
    out["verdict"] = np.select(
        [out["sorting_power"] >= 0.70, out["sorting_power"] >= 0.60,
         out["sorting_power"] > 0.45],
        ["Strong sorter", "Moderate sorter", "Weak sorter"], default="Reversed")
    return out.round({"high_mean": 2, "low_mean": 2, "gap": 2, "sorting_power": 2}).reset_index(drop=True)


# PROCESS 6 – CHARTS

def engagement_trend_chart(
    trend: dict[str, Any],
    show_groups: list[str] | None = None,
) -> alt.Chart:
    """Line chart of % engaged over time (or bar chart when there is one period).

    show_groups limits which group lines are drawn ("All employees" is always kept).
    """
    table = trend["table"].copy()
    group = trend["group_column"] or "_all"
    if group not in table.columns:
        table[group] = "All employees"
    if show_groups is not None:
        table = table[table[group].isin(list(show_groups) + ["All employees"])]
    time_col = trend["time_column"]
    tooltip = [alt.Tooltip(group, title="Group"),
               alt.Tooltip("pct_engaged:Q", title="% engaged", format=".1f"),
               alt.Tooltip("employees:Q", title="Employees", format=",")]
    # Zoom to the data (with headroom) so real movement is visible; axis is labeled.
    low = max(0.0, float(table["pct_engaged"].min()) - 8)
    high = min(100.0, float(table["pct_engaged"].max()) + 8)
    y = alt.Y("pct_engaged:Q", title="% engaged", scale=alt.Scale(domain=[low, high]))

    if not trend["has_trend"]:
        return (alt.Chart(table).mark_bar(color=LINE_COLORS[0], cornerRadiusEnd=3)
                .encode(x=alt.X(f"{group}:N", sort="-y", title=None), y=y, tooltip=tooltip)
                .properties(height=320, width="container"))

    groups = list(dict.fromkeys(table[group].astype(str)))
    others = [g for g in groups if g != "All employees"]
    domain = (["All employees"] if "All employees" in groups else []) + others
    palette = ([OVERALL_COLOR] if "All employees" in groups else []) + \
        [LINE_COLORS[i % len(LINE_COLORS)] for i in range(len(others))]
    color = alt.Color(f"{group}:N", title=None, sort=domain,
                      scale=alt.Scale(domain=domain, range=palette),
                      # One line needs no legend; the section title names it.
                      legend=alt.Legend(orient="top", columns=5) if len(domain) > 1 else None)
    table["_overall"] = table[group] == "All employees"
    base = alt.Chart(table).encode(
        x=alt.X(f"{time_col}:O", title="Survey wave" if time_col == "wave" else "Year",
                axis=alt.Axis(labelAngle=0)),
        y=y, color=color, tooltip=tooltip + [alt.Tooltip(f"{time_col}:O", title="Period")])
    lines = (base.mark_line().encode(
                 strokeWidth=alt.condition("datum._overall", alt.value(4), alt.value(2.2)))
             + base.mark_circle(size=60, opacity=1))
    last = table[table[time_col] == table[time_col].max()]
    if len(others) > 5:                    # too many end labels overlap: keep the overall one
        last = last[last["_overall"]] if last["_overall"].any() else last.iloc[0:0]
    labels = (alt.Chart(last).mark_text(align="left", dx=8, fontSize=13, fontWeight="bold")
              .encode(x=f"{time_col}:O", y="pct_engaged:Q", color=color,
                      text=alt.Text("pct_engaged:Q", format=".0f")))
    return (lines + labels).properties(height=320, width="container")


def sorting_effect_chart(
    df: pd.DataFrame,
    items: list[str],
    outcome: str = "performance",
    level: str = "team",
    seed: int = 7,
) -> alt.Chart:
    """Gallup 'sorting effect' dot chart: one panel per item, + / - markers."""
    units = build_sorting_groups(df, outcome, level)
    ranking = rank_sorting_power(df, outcome, level).set_index("item")
    rng = np.random.default_rng(seed)
    n = VISUAL_CONFIG["max_points"]

    panels = []
    for item in items:
        pts = []
        for grp in ["High performers", "Low performers"]:
            g = units.loc[units["group"] == grp, [item]].dropna()
            g = g.sample(min(n, len(g)), random_state=seed)
            x = g[item].to_numpy(float)
            if level == "employee":            # 1-5 integers: spread the dots
                x = x + rng.uniform(-0.35, 0.35, len(x))
            pts.append(pd.DataFrame({"score": x, "y": rng.uniform(0, 1, len(x)),
                                     "group": grp,
                                     "sign": "+" if grp == "High performers" else "−"}))
        pts = pd.concat(pts)
        score = ranking.loc[item, "sorting_power"]
        title = f"{item} · {ITEM_LABELS.get(item, item)} — {ranking.loc[item, 'verdict']} ({score:.2f})"
        color = alt.Color("group:N", title=None,
                          scale=alt.Scale(domain=["High performers", "Low performers"],
                                          range=[HIGH_COLOR, LOW_COLOR]),
                          legend=alt.Legend(orient="top"))
        x = alt.X("score:Q", scale=alt.Scale(domain=[0.5, 5.5]),
                  axis=alt.Axis(values=[1, 2, 3, 4, 5], title="1 = Strongly disagree   ·   5 = Strongly agree"))
        y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.1, 1.1]))
        dots = alt.Chart(pts).mark_circle(size=230, opacity=0.95).encode(x=x, y=y, color=color)
        signs = alt.Chart(pts).mark_text(color="white", fontSize=15, fontWeight="bold").encode(
            x="score:Q", y="y:Q", text="sign:N")
        panels.append((dots + signs).properties(height=150, width=560, title=title))

    unit = "teams" if level == "team" else "employees"
    return (alt.vconcat(*panels)
            .properties(title=f"Sorting effect: does each item separate high- from low-performing {unit}?")
            .configure_view(stroke="#9BB8A5"))


def sorting_power_chart(ranking: pd.DataFrame) -> alt.Chart:
    """Bar chart ranking all 12 items by sorting power (0.5 = no sorting)."""
    data = ranking.assign(name=ranking["item"] + " · " + ranking["label"])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("sorting_power:Q", title="Sorting power (0.5 = no sorting, 1.0 = perfect)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], format=".1f")),
        y=alt.Y("name:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=320, labelOverlap=False, labelFontSize=12)),
        color=alt.Color("verdict:N", title=None,
                        scale=alt.Scale(domain=["Strong sorter", "Moderate sorter",
                                                "Weak sorter", "Reversed"],
                                        range=[HIGH_COLOR, "#9CCB9F", "#B9B1A8", "#B5473F"]),
                        legend=alt.Legend(orient="top", columns=2, labelLimit=0)),
        tooltip=["item", "label", "high_mean", "low_mean", "gap", "sorting_power", "n_high", "n_low"])
    no_sorting = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        strokeDash=[4, 4], color="#555").encode(x="x:Q")
    return (bars + no_sorting).properties(height=alt.Step(26), width="container")


# PROCESS 7 – SEGMENT AND TEAM CHARTS

def segment_engagement(df: pd.DataFrame, group_column: str) -> dict[str, Any]:
    """% engaged per group in the latest survey (groups of 5+), ranked."""
    data = add_engagement_status(df)
    if "wave" in data.columns and data["wave"].notna().any():
        data = data[data["wave"] == data["wave"].max()]
    table = (data.groupby(group_column)
             .agg(employees=("q12_mean", "size"), pct_engaged=("engaged", "mean"),
                  avg_q12=("q12_mean", "mean"))
             .reset_index())
    table = table[table["employees"] >= VISUAL_CONFIG["min_group_size"]].copy()
    table["pct_engaged"] *= 100
    company = float(data["engaged"].mean() * 100)
    table["vs_company"] = np.where(table["pct_engaged"] >= company, "At or above company", "Below company")
    table = table.sort_values("pct_engaged", ascending=False).reset_index(drop=True)
    return {"table": table, "company": company, "group": group_column}


def explain_segment_engagement(result: dict[str, Any]) -> dict[str, Any]:
    t, company = result["table"], result["company"]
    name = result["group"].replace("_", " ")
    if len(t) < 2:
        return {"headline": f"Only one {name} has enough people to show.", "points": []}
    top, bottom = t.iloc[0], t.iloc[-1]
    below = int((t["pct_engaged"] < company).sum())
    return {
        "headline": (f"{top[result['group']]} is the most engaged {name} "
                     f"({top['pct_engaged']:.0f}%); {bottom[result['group']]} the least "
                     f"({bottom['pct_engaged']:.0f}%)."),
        "points": [f"{below} of {len(t)} {name}s are below the company's {company:.0f}% engaged.",
                   f"The gap between highest and lowest is "
                   f"{top['pct_engaged'] - bottom['pct_engaged']:.0f} points."],
    }


def segment_engagement_chart(df: pd.DataFrame, group_column: str) -> alt.Chart:
    """% engaged by group in the latest survey, ranked, with the company
    average as a dashed line."""
    result = segment_engagement(df, group_column)
    table, company = result["table"], result["company"]
    # Explicit order: "-x" sorting is dropped when layers share the axis.
    order = list(table.sort_values("pct_engaged", ascending=False)[group_column])
    bars = alt.Chart(table).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y(f"{group_column}:N", sort=order, title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("pct_engaged:Q", title="% engaged"),
        color=alt.Color("vs_company:N", title=None,
                        scale=alt.Scale(domain=["At or above company", "Below company"],
                                        range=[HIGH_COLOR, "#B9B1A8"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip(group_column, title="Group"),
                 alt.Tooltip("pct_engaged:Q", title="% engaged", format=".1f"),
                 alt.Tooltip("avg_q12:Q", title="Average Q12", format=".2f"),
                 alt.Tooltip("employees:Q", title="Employees", format=",")])
    labels = bars.mark_text(align="left", dx=4, color="#333").encode(
        text=alt.Text("pct_engaged:Q", format=".0f"), color=alt.value("#333"))
    rule = alt.Chart(pd.DataFrame({"x": [company]})).mark_rule(
        strokeDash=[4, 4], color=OVERALL_COLOR).encode(x="x:Q")
    note = alt.Chart(pd.DataFrame({"x": [company], "t": [f"Company {company:.0f}%"]})).mark_text(
        align="left", dx=4, dy=-6, color=OVERALL_COLOR, fontWeight="bold").encode(
        x="x:Q", y=alt.value(0), text="t:N")
    return (bars + labels + rule + note).properties(height=alt.Step(30), width="container")


def team_heatmap_chart(teams: pd.DataFrame, company_means: pd.Series, max_teams: int = 25) -> alt.Chart:
    """Teams x Q12 items, colored by difference from the company average."""
    items = [i for i in Q12_ITEMS if i in teams.columns]
    top = teams.head(max_teams)
    long = top.melt(id_vars=["team_id"], value_vars=items, var_name="item", value_name="score")
    long["vs_company"] = long["score"] - long["item"].map(company_means)
    long["item_name"] = long["item"] + " · " + long["item"].map(ITEM_LABELS)
    order = list(top["team_id"])
    tooltip = [alt.Tooltip("team_id:N", title="Team"), alt.Tooltip("item_name:N", title="Question"),
               alt.Tooltip("score:Q", title="Team score", format=".2f"),
               alt.Tooltip("vs_company:Q", title="vs company", format="+.2f")]
    base = alt.Chart(long).encode(
        x=alt.X("item:N", title=None, sort=items, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("team_id:N", title=None, sort=order, axis=alt.Axis(labelOverlap=False)),
        tooltip=tooltip)
    cells = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color("vs_company:Q", title="vs company",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[-1.2, 1.2], clamp=True),
                        legend=alt.Legend(orient="right", format="+.1f")))
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("score:Q", format=".1f"),
        color=alt.condition("abs(datum.vs_company) > 0.8", alt.value("white"), alt.value("#222")))
    return (cells + text).properties(height=alt.Step(26), width="container")


# PROCESS 8 – PLAIN-LANGUAGE EXPLANATIONS
#
# Each explain_* function turns the numbers behind a chart into short sentences
# for a nontechnical leader: a headline, a few key points, and how to read it.

# Who counts as "high" vs "low" performers, per outcome and level.
GROUP_DEFINITIONS = {
    ("performance", "team"): ("teams in the top 25% on average performance rating",
                              "teams in the bottom 25%"),
    ("turnover", "team"): ("the 25% of teams with the lowest voluntary turnover",
                           "the 25% of teams with the highest"),
    ("absence", "team"): ("the 25% of teams with the fewest unplanned absence days",
                          "the 25% of teams with the most"),
    ("performance", "employee"): ("employees rated 4 or 5", "employees rated 1 or 2"),
    ("turnover", "employee"): ("employees who stayed",
                               "employees who left voluntarily within 12 months"),
    ("absence", "employee"): ("the 25% of employees with the fewest unplanned absence days",
                              "the 25% with the most"),
}


# Short names for the better / worse group, used inside sentences.
GROUP_SHORT_NAMES = {
    ("performance", "team"): ("higher-performing teams", "lower-performing teams"),
    ("turnover", "team"): ("low-turnover teams", "high-turnover teams"),
    ("absence", "team"): ("low-absence teams", "high-absence teams"),
    ("performance", "employee"): ("high performers", "low performers"),
    ("turnover", "employee"): ("employees who stayed", "employees who left"),
    ("absence", "employee"): ("low-absence employees", "high-absence employees"),
}


def _period(value: Any) -> str:
    """1.0 -> '1', 2024 -> '2024'."""
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(value)
    except (TypeError, ValueError):
        return str(value)


def _name(item: str) -> str:
    return f"{item} {ITEM_LABELS.get(item, item)}"


def _in_ten(pct: float) -> str:
    """39.7 -> 'about 4 in 10'."""
    return f"about {max(1, round(pct / 10))} in 10"


def explain_engagement_trend(trend: dict[str, Any]) -> dict[str, Any]:
    """Headline, key points and reading guide for the engagement trend chart."""
    table = trend["table"]
    time_col = trend["time_column"]
    group = trend["group_column"]
    points: list[str] = []

    overall = table[table[group] == "All employees"] if group and group in table else table
    if group and not trend["has_trend"]:
        overall = table.iloc[0:0]

    period_name = "wave" if time_col == "wave" else "year"
    headline = "Engagement by group."

    if not overall.empty:
        overall = overall.sort_values(time_col) if time_col else overall
        latest = overall.iloc[-1]
        pct = float(latest["pct_engaged"])
        headline = f"{pct:.0f}% of employees are engaged ({_in_ten(pct)})."

        if trend["has_trend"] and len(overall) >= 2:
            first = overall.iloc[0]
            change = pct - float(first["pct_engaged"])
            if abs(change) < 1:
                direction = "essentially unchanged"
            else:
                direction = f"{'up' if change > 0 else 'down'} {abs(change):.1f} points"
            points.append(
                f"Engagement went from {first['pct_engaged']:.1f}% in {period_name} "
                f"{_period(first[time_col])} to {pct:.1f}% in {period_name} "
                f"{_period(latest[time_col])}: "
                f"{direction}.")

        if trend["has_trend"] and len(overall) >= 3:
            previous = overall.iloc[-2]
            recent = pct - float(previous["pct_engaged"])
            points.append(
                f"Since the last survey ({period_name} {_period(previous[time_col])}): "
                + ("about the same." if abs(recent) < 1 else
                   f"{'up' if recent > 0 else 'down'} {abs(recent):.1f} points."))
            low = overall.loc[overall["pct_engaged"].idxmin()]
            high = overall.loc[overall["pct_engaged"].idxmax()]
            points.append(
                f"Lowest point: {low['pct_engaged']:.1f}% in {period_name} "
                f"{_period(low[time_col])}. Highest: {high['pct_engaged']:.1f}% in "
                f"{period_name} {_period(high[time_col])}.")

    if group and group in table:
        groups = table[table[group] != "All employees"]
        if time_col:
            groups = groups[groups[time_col] == groups[time_col].max()]
        if len(groups) >= 2:
            ranked = groups.sort_values("pct_engaged", ascending=False)
            top, bottom = ranked.iloc[0], ranked.iloc[-1]
            group_name = group.replace("_", " ")
            points.append(
                f"Highest {group_name}: **{top[group]}** at {top['pct_engaged']:.0f}% engaged. "
                f"Lowest: **{bottom[group]}** at {bottom['pct_engaged']:.0f}% "
                f"(a {top['pct_engaged'] - bottom['pct_engaged']:.0f}-point gap).")

        if trend["has_trend"] and time_col:
            all_groups = table[table[group] != "All employees"]
            wide = all_groups.pivot_table(index=group, columns=time_col, values="pct_engaged")
            if wide.shape[1] >= 2:
                moves = (wide.iloc[:, -1] - wide.iloc[:, 0]).dropna()
                if not moves.empty and moves.abs().max() >= 1:
                    mover = moves.abs().idxmax()
                    points.append(
                        f"Biggest change: **{mover}**, "
                        f"{'up' if moves[mover] > 0 else 'down'} {abs(moves[mover]):.1f} points.")

    how_to_read = (
        "Each line shows the share of employees who are engaged in each survey "
        f"{period_name}. An employee counts as engaged when their average answer "
        f"across the 12 questions is {VISUAL_CONFIG['engaged_threshold']:.1f} or "
        "higher on the 1–5 scale (mostly 'agree' or 'strongly agree'). "
        "The thick navy line is the whole company; a line rising means more "
        "people are engaged. Hover over a point to see how many employees it covers."
        if trend["has_trend"] else
        "Each bar shows the share of employees who are engaged. An employee counts "
        f"as engaged when their average answer is {VISUAL_CONFIG['engaged_threshold']:.1f} "
        "or higher on the 1–5 scale. Only one survey period is available, so no "
        "trend is shown yet.")

    return {"headline": headline, "points": points, "how_to_read": how_to_read}


def explain_sorting_power(
    ranking: pd.DataFrame,
    outcome: str = "performance",
    level: str = "team",
) -> dict[str, Any]:
    """Headline, key points and reading guide for the sorting-power bar chart."""
    high_def, low_def = GROUP_DEFINITIONS[(outcome, level)]
    better, worse = GROUP_SHORT_NAMES[(outcome, level)]
    unit = "team" if level == "team" else "employee"
    top, bottom = ranking.iloc[0], ranking.iloc[-1]
    pct = round(float(top["sorting_power"]) * 100)

    headline = (
        f"{_name(top['item'])} is the question that best tells {better} "
        f"apart from {worse}.")

    counts = ranking["verdict"].value_counts()
    verdict_names = {"Strong sorter": "strong", "Moderate sorter": "moderate",
                     "Weak sorter": "weak", "Reversed": "reversed"}
    summary = ", ".join(
        f"{counts[v]} {verdict_names[v]}"
        for v in verdict_names
        if v in counts)
    points = [
        f"Pick one {unit} from each group at random: {pct} times out of 100, the "
        f"one from the {better} scores higher on this question (50 would be a "
        "coin flip).",
        f"Of the {len(ranking)} questions: {summary}.",
    ]

    spread = float(top["sorting_power"] - bottom["sorting_power"])
    if spread < 0.10 and bottom["verdict"] != "Reversed":
        points.append(
            f"All questions sort about equally well ({bottom['sorting_power']:.2f} to "
            f"{top['sorting_power']:.2f}). Engagement as a whole goes with results here, "
            "so this chart alone does not single out one question; use the Statistical "
            "Outcome Analysis to choose priorities.")
    elif bottom["verdict"] != "Reversed":
        points.append(
            f"Weakest: {_name(bottom['item'])} ({bottom['sorting_power']:.2f}), which "
            "tells the two groups apart much less clearly.")

    reversed_items = ranking[ranking["verdict"] == "Reversed"]
    if not reversed_items.empty:
        names = ", ".join(_name(i) for i in reversed_items["item"])
        points.append(
            f"Reversed: {names}. Here the {worse} score higher. This raw pattern may "
            "reflect other differences between groups (such as department), so check "
            "it against the Statistical Outcome Analysis before acting on it.")

    how_to_read = (
        f"Each bar asks one question: does this survey item tell apart {high_def} "
        f"from {low_def}? The dashed line at 0.5 means no better than a coin flip. "
        "The longer the bar, the more clearly high performers agree with the item "
        "more than low performers do. Green = strong (0.70 or more), light green = "
        "moderate (0.60–0.70), grey = weak, red = reversed.")

    return {"headline": headline, "points": points, "how_to_read": how_to_read}


def explain_sorting_items(
    ranking: pd.DataFrame,
    items: list[str],
    outcome: str = "performance",
    level: str = "team",
) -> dict[str, Any]:
    """One sentence per selected item, plus a reading guide for the dot chart."""
    high_def, low_def = GROUP_DEFINITIONS[(outcome, level)]
    better, worse = GROUP_SHORT_NAMES[(outcome, level)]
    unit = "team" if level == "team" else "employee"
    rows = ranking.set_index("item")
    points = []
    for item in items:
        r = rows.loc[item]
        points.append(
            f"**{_name(item)}**: {better} average {r['high_mean']:.2f} vs "
            f"{r['low_mean']:.2f} for {worse}, a {abs(r['gap']):.2f}-point "
            f"{'gap' if r['gap'] >= 0 else 'gap in the wrong direction'} on the 1–5 scale "
            f"({r['verdict'].lower()}).")

    how_to_read = (
        f"Each dot is one {unit}. Green dots with + ('High performers' in the legend) "
        f"are {high_def}; dark dots with − ('Low performers') are {low_def}. Further right means stronger agreement (5 = strongly agree). "
        "When the green dots sit to the right of the dark dots, the question separates "
        "the two groups well; when they are mixed together, it does not. Up to "
        f"{VISUAL_CONFIG['max_points']} dots per group are shown to keep the chart readable.")

    return {"points": points, "how_to_read": how_to_read}


def explain_team_heatmap(
    teams: pd.DataFrame,
    company_means: pd.Series,
    max_teams: int = 25,
) -> dict[str, Any]:
    """Headline, key points and reading guide for the team heatmap."""
    items = [i for i in Q12_ITEMS if i in teams.columns]
    shown = teams.head(max_teams).set_index("team_id")
    diff = shown[items] - company_means[items]
    below = (diff < 0).sum(axis=1)

    worst_team = below.idxmax()
    common_gap = diff.mean().idxmin()
    points = [
        f"Team **{worst_team}** is below the company average on "
        f"{below[worst_team]} of {len(items)} questions.",
        f"Across these teams, the most common weak spot is **{_name(common_gap)}** "
        f"({diff[common_gap].mean():+.2f} vs company on average).",
    ]

    headline = f"The {len(shown)} most urgent of {len(teams)} teams, most urgent first."
    if "priority" in shown.columns and shown["priority"].any():
        n_priority = int(shown["priority"].sum())
        headline = (f"The {len(shown)} most urgent of {len(teams)} teams, most urgent first "
                    f"({n_priority} of them are priority teams).")
        points.append(
            "Priority teams are in the bottom quarter of teams on their focus question "
            "and have higher turnover or absence than the company average.")

    how_to_read = (
        "Each row is a team and each column is one of the 12 survey questions. "
        "The number in a cell is the team's average answer (1–5). Green cells are "
        "above the company average for that question, red cells are below, and "
        "yellow is about average. A row with lots of red is a team that may need "
        "support; a column with lots of red is a company-wide issue.")

    return {"headline": headline, "points": points, "how_to_read": how_to_read}


# PROCESS 9 – OVERVIEW CARD (GAUGE, CHANGE, PERCENTILE, TREND)
#
# A Gallup-style overview for the executive summary: the average Q12 score on a
# 1-5 gauge, the change since the last survey, how the unit ranks among its
# peers inside this organization, and the trended mean.

SCOPE_COLUMNS = {"department": "Department", "team_id": "Team"}

TRACK_OPACITY = 0.18          # unfilled track: same green, lighter (works in light and dark)
MUTED_COLOR = "#8A8F98"       # de-emphasis hue for the sparkline


def _period_means(data: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    if time_col is None:
        data = data.assign(_period="Current")
        time_col = "_period"
    table = (data.groupby(time_col)
             .agg(mean=("q12_mean", "mean"), employees=("q12_mean", "size"),
                  pct_engaged=("engaged", "mean"))
             .reset_index()
             .rename(columns={time_col: "period"}))
    table["pct_engaged"] *= 100
    table = table[table["employees"] >= VISUAL_CONFIG["min_group_size"]]
    return table.sort_values("period").reset_index(drop=True)


def build_overview(
    df: pd.DataFrame,
    scope: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    """Overview numbers for the whole company (scope=None) or one department/team.

    Current = latest survey period; change = vs the period before it.
    Percentile = share of peer units (other departments or teams in this
    organization, latest period) with a lower average score.
    """
    data = add_engagement_status(df)
    time_col = detect_time_column(data)
    if time_col == "survey_year":
        data["survey_year"] = pd.to_datetime(data["survey_date"], errors="coerce").dt.year

    company = _period_means(data, time_col)
    unit = data if scope is None else data[data[scope] == value]
    trend = _period_means(unit, time_col)

    if trend.empty:
        raise ValueError(
            f"Fewer than {VISUAL_CONFIG['min_group_size']} responses for this "
            "group, so it cannot be shown.")

    current = trend.iloc[-1]
    previous = trend.iloc[-2] if len(trend) >= 2 else None

    percentile, n_peers = None, 0
    if scope is not None:
        latest = data if time_col is None else data[data[time_col] == data[time_col].max()]
        peers = latest.groupby(scope)["q12_mean"].agg(["mean", "size"])
        peers = peers[peers["size"] >= VISUAL_CONFIG["min_group_size"]]
        if value in peers.index and len(peers) >= 2:
            others = peers.drop(index=value)["mean"]
            percentile = float((others < peers.loc[value, "mean"]).mean() * 100)
            n_peers = len(peers)

    period_name = "wave" if time_col == "wave" else "year" if time_col else "survey"
    return {
        "scope": scope,
        "value": value,
        "title": "Whole company" if scope is None else f"{SCOPE_COLUMNS[scope]} {value}",
        "period_name": period_name,
        "current_period": _period(current["period"]),
        "previous_period": None if previous is None else _period(previous["period"]),
        "current": round(float(current["mean"]), 2),
        "previous": None if previous is None else round(float(previous["mean"]), 2),
        "delta": None if previous is None else round(float(current["mean"] - previous["mean"]), 2) + 0.0,
        "employees": int(current["employees"]),
        "pct_engaged": round(float(current["pct_engaged"]), 1),
        "company_current": round(float(company.iloc[-1]["mean"]), 2),
        "percentile": None if percentile is None else round(percentile),
        "percentile_label": None if percentile is None else _ordinal(round(percentile)),
        "n_peers": n_peers,
        "trend": trend.assign(period=trend["period"].map(_period)),
    }


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def explain_overview(overview: dict[str, Any]) -> str:
    """One or two plain sentences that read the overview card aloud."""
    o = overview
    who = "The company" if o["scope"] is None else o["title"]
    text = f"{who} scores **{o['current']:.2f} out of 5** in {o['period_name']} {o['current_period']}"

    if o["delta"] is not None:
        if abs(o["delta"]) < 0.05:
            text += f", about the same as {o['period_name']} {o['previous_period']}"
        else:
            text += (f", {'up' if o['delta'] > 0 else 'down'} {abs(o['delta']):.2f} "
                     f"since {o['period_name']} {o['previous_period']}")
    text += f". {_in_ten(o['pct_engaged']).capitalize()} employees are engaged."

    if o["percentile"] is not None:
        unit = SCOPE_COLUMNS[o["scope"]].lower()
        text += (f" That is higher than {o['percentile']}% of the {o['n_peers'] - 1} other "
                 f"{unit}s in your organization ({_ordinal(o['percentile'])} percentile).")
    return text


def score_gauge_chart(score: float, company: float | None = None) -> alt.LayerChart:
    """Half-circle gauge for a 1-5 score; optional thin gray inner ring = company score."""
    width, height = 300, 175
    cx, cy, outer, thick = width / 2, height - 22, 120, 22
    left, span = -math.pi / 2, math.pi

    def angle(v: float) -> float:
        return left + span * (min(max(v, 1), 5) - 1) / 4

    def arc(start, end, inner, outer_r, color, opacity=1.0, tooltip=None):
        data = pd.DataFrame({"start": [start], "end": [end], "label": [tooltip or ""]})
        return alt.Chart(data).mark_arc(
            innerRadius=inner, outerRadius=outer_r, x=cx, y=cy, color=color,
            opacity=opacity, cornerRadius=4 if opacity == 1.0 else 0,
        ).encode(theta=alt.Theta("start:Q", scale=None), theta2="end:Q",
                 tooltip=alt.Tooltip("label:N", title=None) if tooltip else alt.value(None))

    layers = [
        arc(left, left + span, outer - thick, outer, HIGH_COLOR, TRACK_OPACITY),
        arc(left, angle(score), outer - thick, outer, HIGH_COLOR,
            tooltip=f"Score {score:.2f} out of 5"),
    ]
    if company is not None:
        ring_out, ring_in = outer - thick - 6, outer - thick - 14
        layers += [
            arc(left, left + span, ring_in, ring_out, MUTED_COLOR, TRACK_OPACITY),
            arc(left, angle(company), ring_in, ring_out, MUTED_COLOR,
                tooltip=f"Company {company:.2f} out of 5"),
        ]

    def text(content, x, y, size, weight="normal", align="center"):
        return alt.Chart(pd.DataFrame({"t": [content]})).mark_text(
            x=x, y=y, fontSize=size, fontWeight=weight, align=align, baseline="bottom",
        ).encode(text="t:N")

    layers += [
        text(f"{score:.2f}", cx, cy, 52, "bold"),
        text("1.0", cx - outer + thick / 2, cy + 18, 11),
        text("5.0", cx + outer - thick / 2, cy + 18, 11),
    ]
    return alt.layer(*layers).properties(width=width, height=height).configure_view(stroke=None)


def percentile_meter_chart(value: float, label: str) -> alt.LayerChart:
    """Horizontal 0-100 meter: same-green track, filled to value, marker at value."""
    x = alt.X("start:Q", scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(values=[0, 25, 50, 75, 100], title=None, grid=False,
                            domain=False, ticks=False, labelFontSize=10))
    track = alt.Chart(pd.DataFrame({"start": [0], "end": [100]})).mark_bar(
        color=HIGH_COLOR, opacity=TRACK_OPACITY, height=14, cornerRadius=4,
    ).encode(x=x, x2="end:Q")
    fill = alt.Chart(pd.DataFrame({"start": [0], "end": [value], "label": [label]})).mark_bar(
        color=HIGH_COLOR, height=14, cornerRadius=4,
    ).encode(x=x, x2="end:Q", tooltip=alt.Tooltip("label:N", title=None))
    marker = alt.Chart(pd.DataFrame({"start": [value]})).mark_tick(
        color=OVERALL_COLOR, thickness=3, size=24,
    ).encode(x=x)
    return (alt.layer(track, fill, marker)
            .properties(height=44, width="container")
            .configure_view(stroke=None))


def trend_sparkline_chart(trend: pd.DataFrame, period_name: str = "wave") -> alt.LayerChart:
    """Trended mean: muted line, latest point in the accent color, hover tooltip."""
    data = trend.assign(latest=trend["period"] == trend["period"].iloc[-1])
    lo, hi = float(data["mean"].min()), float(data["mean"].max())
    pad = max(0.1, (hi - lo) * 0.5)
    base = alt.Chart(data).encode(
        x=alt.X("period:O", title=None, sort=list(data["period"]),
                axis=alt.Axis(labelAngle=0, labelExpr=f"'{period_name.capitalize()} ' + datum.label",
                              domain=False, ticks=False, labelFontSize=10)),
        y=alt.Y("mean:Q", title=None, scale=alt.Scale(domain=[lo - pad, hi + pad]), axis=None),
        tooltip=[alt.Tooltip("period:O", title=period_name.capitalize()),
                 alt.Tooltip("mean:Q", title="Average score", format=".2f"),
                 alt.Tooltip("pct_engaged:Q", title="% engaged", format=".1f"),
                 alt.Tooltip("employees:Q", title="Employees", format=",")])
    area = base.mark_area(color=MUTED_COLOR, opacity=0.12, clip=True).encode(
        y2=alt.datum(lo - pad))
    line = base.mark_line(color=MUTED_COLOR, strokeWidth=2)
    dots = base.mark_circle(size=70, opacity=1).encode(
        color=alt.condition("datum.latest", alt.value(HIGH_COLOR), alt.value(MUTED_COLOR)))
    return (alt.layer(area, line, dots)
            .properties(height=110, width="container")
            .configure_view(stroke=None))



# PROCESS 10 – RESULTS BY QUESTION (LATEST SURVEY)

def item_results(df: pd.DataFrame) -> dict[str, Any]:
    """Per question in the latest survey: % agree (4-5), average, and the change
    in % agree since the previous survey."""
    data = _respondents(df)
    items = [i for i in Q12_ITEMS if i in data.columns]
    data[items] = data[items].apply(pd.to_numeric, errors="coerce")
    has_waves = "wave" in data.columns and data["wave"].nunique() > 1
    waves = sorted(data["wave"].dropna().unique()) if has_waves else []
    latest = data[data["wave"] == waves[-1]] if has_waves else data
    previous = data[data["wave"] == waves[-2]] if has_waves else None

    table = pd.DataFrame({
        "item": items,
        "label": [ITEM_LABELS.get(i, i) for i in items],
        "agree": [(latest[i] >= 4).mean() * 100 for i in items],
        "average": [latest[i].mean() for i in items],
    })
    if previous is not None:
        table["change"] = [(latest[i] >= 4).mean() * 100 - (previous[i] >= 4).mean() * 100
                           for i in items]
    table = table.sort_values("agree", ascending=False).reset_index(drop=True)
    table["name"] = table["item"] + " · " + table["label"]
    # Highlight the three strongest and three weakest questions.
    table["group"] = "Middle"
    table.loc[table.index[:3], "group"] = "Top 3"
    table.loc[table.index[-3:], "group"] = "Bottom 3"
    return {"table": table, "responses": len(latest),
            "wave": waves[-1] if has_waves else None,
            "previous_wave": waves[-2] if has_waves else None}


def explain_item_results(results: dict[str, Any]) -> dict[str, Any]:
    t = results["table"]
    top, bottom = t.iloc[0], t.iloc[-1]
    headline = (f"{top['label']} is the strongest question ({top['agree']:.0f}% agree); "
                f"{bottom['label']} is the biggest opportunity ({bottom['agree']:.0f}%).")
    points = []
    if "change" in t:
        rise, fall = t.loc[t["change"].idxmax()], t.loc[t["change"].idxmin()]
        since = f"since {_w_label(results['previous_wave'])}"
        if rise["change"] >= 1:
            points.append(f"Biggest rise {since}: **{rise['label']}** "
                          f"({rise['change']:+.1f} points).")
        if fall["change"] <= -1:
            points.append(f"Biggest drop {since}: **{fall['label']}** "
                          f"({fall['change']:+.1f} points).")
        if not points:
            points.append(f"No question moved by a point or more {since}.")
    return {"headline": headline, "points": points}


def _w_label(wave: Any) -> str:
    try:
        number = float(wave)
        return f"wave {int(number)}" if number.is_integer() else f"wave {wave}"
    except (TypeError, ValueError):
        return "the last survey"


def item_results_chart(results: dict[str, Any]) -> alt.LayerChart:
    """Ranked bars of % agree per question, with the average as a label."""
    t = results["table"].copy()
    t["text"] = t["agree"].map("{:.0f}%".format) + "  ·  " + t["average"].map("{:.2f}".format)
    tooltip = [alt.Tooltip("name:N", title="Question"),
               alt.Tooltip("agree:Q", title="% agree (4–5)", format=".1f"),
               alt.Tooltip("average:Q", title="Average (1–5)", format=".2f")]
    if "change" in t:
        tooltip.append(alt.Tooltip("change:Q", title="Change in % agree", format="+.1f"))
    y = alt.Y("name:N", sort=list(t["name"]), title=None,
              axis=alt.Axis(labelLimit=260, labelOverlap=False))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=y,
        x=alt.X("agree:Q", title="% agree or strongly agree",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("group:N", title=None,
                        scale=alt.Scale(domain=["Top 3", "Middle", "Bottom 3"],
                                        range=[HIGH_COLOR, "#9CCB9F", "#C27C1E"]),
                        legend=alt.Legend(orient="top")),
        tooltip=tooltip)
    labels = alt.Chart(t).mark_text(align="left", dx=6, fontSize=12).encode(
        y=y, x="agree:Q", text="text:N")
    return (bars + labels).properties(height=alt.Step(30), width="container")



# PROCESS 11 – ONE-QUESTION SORTING DOT CHART (FILLS ITS COLUMN)

def sorting_dot_chart(
    df: pd.DataFrame,
    item: str,
    outcome: str = "performance",
    level: str = "team",
    seed: int = 7,
) -> alt.LayerChart:
    """Where high- and low-performing units sit on one question (+ / − dots)."""
    units = build_sorting_groups(df, outcome, level)
    rng = np.random.default_rng(seed)
    n = VISUAL_CONFIG["max_points"]
    pts = []
    for grp in ["High performers", "Low performers"]:
        g = units.loc[units["group"] == grp, [item]].dropna()
        g = g.sample(min(n, len(g)), random_state=seed)
        x = g[item].to_numpy(float)
        if level == "employee":                       # 1-5 integers: spread the dots
            x = x + rng.uniform(-0.35, 0.35, len(x))
        pts.append(pd.DataFrame({"score": x, "y": rng.uniform(0, 1, len(x)), "group": grp,
                                 "sign": "+" if grp == "High performers" else "−"}))
    pts = pd.concat(pts)
    color = alt.Color("group:N", title=None,
                      scale=alt.Scale(domain=["High performers", "Low performers"],
                                      range=[HIGH_COLOR, LOW_COLOR]),
                      legend=alt.Legend(orient="top"))
    x = alt.X("score:Q", scale=alt.Scale(domain=[0.5, 5.5]),
              axis=alt.Axis(values=[1, 2, 3, 4, 5], title="1 = strongly disagree · 5 = strongly agree"))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.1, 1.1]))
    dots = alt.Chart(pts).mark_circle(size=170, opacity=0.9).encode(x=x, y=y, color=color)
    signs = alt.Chart(pts).mark_text(color="white", fontSize=13, fontWeight="bold").encode(
        x="score:Q", y="y:Q", text="sign:N")
    return (dots + signs).properties(height=200, width="container")


# PROCESS 12 – AGREE / NEUTRAL / DISAGREE SPLIT BY PERFORMANCE GROUP

ANSWER_BANDS = ["Disagree", "Neutral", "Agree"]
BAND_COLORS = ["#B5473F", "#B9B1A8", HIGH_COLOR]     # diverging: red · gray midpoint · green


def sorting_distribution(df: pd.DataFrame, item: str, outcome: str = "performance",
                         level: str = "team") -> dict[str, Any]:
    """Share of high- and low-performing groups that disagree / are neutral /
    agree on one question, using every unit (no sampling).

    Employees: 1–2 = disagree, 3 = neutral, 4–5 = agree.
    Teams (average scores): below 3 = disagree, 3 to under 4 = neutral, 4+ = agree.
    """
    units = build_sorting_groups(df, outcome, level)
    better, worse = GROUP_SHORT_NAMES[(outcome, level)]
    names = {"High performers": better.capitalize(), "Low performers": worse.capitalize()}
    score = units[item]
    band = np.select([score < (2.5 if level == "employee" else 3.0),
                      score < (3.5 if level == "employee" else 4.0)],
                     ["Disagree", "Neutral"], "Agree")
    data = units.assign(band=band, who=units["group"].map(names)).dropna(subset=[item])
    table = (data.groupby(["who", "band"]).size().rename("n").reset_index())
    table["share"] = table["n"] / table.groupby("who")["n"].transform("sum") * 100
    agree = table[table["band"] == "Agree"].set_index("who")["share"]
    unit = "teams" if level == "team" else "people"
    return {"table": table, "item": item, "better": names["High performers"],
            "worse": names["Low performers"], "unit": unit,
            "agree_better": float(agree.get(names["High performers"], 0.0)),
            "agree_worse": float(agree.get(names["Low performers"], 0.0)),
            "n_better": int((data["group"] == "High performers").sum()),
            "n_worse": int((data["group"] == "Low performers").sum())}


def explain_sorting_distribution(dist: dict[str, Any]) -> str:
    gap = dist["agree_better"] - dist["agree_worse"]
    verb = "agree" if dist["unit"] == "people" else "average 4 or higher"
    if abs(gap) < 5:
        tail = "about the same, so this question does not separate them."
    else:
        points = round(abs(gap))
        article = "an" if str(points).startswith("8") or points in (11, 18) else "a"
        tail = f"{article} {points}-point {'gap' if gap > 0 else 'gap the other way'}."
    return (f"**{dist['agree_better']:.0f}%** of {dist['better'].lower()} {verb}, vs "
            f"**{dist['agree_worse']:.0f}%** of {dist['worse'].lower()}: {tail}")


def sorting_distribution_chart(dist: dict[str, Any], show_legend: bool = True) -> alt.LayerChart:
    t = dist["table"].copy()
    t["order"] = t["band"].map({b: i for i, b in enumerate(ANSWER_BANDS)})
    t["label"] = np.where(t["share"] >= 8, t["share"].round().astype(int).astype(str) + "%", "")
    y = alt.Y("who:N", title=None, sort=[dist["better"], dist["worse"]],
              axis=alt.Axis(labelLimit=200))
    x = alt.X("share:Q", stack="zero", title=None, scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(values=[0, 25, 50, 75, 100], format="d", labelExpr="datum.label + '%'"))
    order = alt.Order("order:Q")
    band_label = ("1–2 disagree", "3 neutral", "4–5 agree") if dist["unit"] == "people" else (
        "Average below 3", "Average 3–4", "Average 4+")
    color = alt.Color("band:N", title=None,
                      scale=alt.Scale(domain=ANSWER_BANDS, range=BAND_COLORS),
                      legend=alt.Legend(orient="top", labelExpr=(
                          f"datum.label == 'Disagree' ? '{band_label[0]}' : "
                          f"datum.label == 'Neutral' ? '{band_label[1]}' : '{band_label[2]}'"))
                      if show_legend else None)
    tooltip = [alt.Tooltip("who:N", title="Group"), alt.Tooltip("band:N", title="Answer"),
               alt.Tooltip("share:Q", title="Share", format=".1f"),
               alt.Tooltip("n:Q", title=dist["unit"].capitalize(), format=",")]
    bars = alt.Chart(t).mark_bar(height=26).encode(y=y, x=x, color=color, order=order,
                                                   tooltip=tooltip)
    text = alt.Chart(t).mark_text(color="white", fontWeight="bold", fontSize=12).encode(
        y=y, x=alt.X("share:Q", stack="zero", bandPosition=0.5), order=order,
        text="label:N",
        detail="band:N")
    return (bars + text).properties(height=alt.Step(36), width="container")
