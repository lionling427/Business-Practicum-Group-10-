"""Change over time: how engagement moves between survey waves, and whether
changes in engagement go with changes in business results.

Descriptive
- derive_wave(df):       add a 'wave' from survey years when the file has none
- item_change(df):       change per Q12 item between the last two waves
- team_change(df):       teams that rose or fell most between the last two waves

Statistical (panel models; all describe associations, not cause and effect)
- Same employee over time: two-way fixed effects (employee + wave). Compares each
  person with themselves, so stable differences between people (personality,
  role, manager style that does not change) cannot explain the result.
- Same team over time: two-way fixed effects (team + wave) on team averages,
  covering turnover, absence and performance rates.
- Engagement trajectory: logistic regression. Among employees with the same
  current score, does a drop since the last survey go with leaving?

analyze_change(df) runs all three for the overall Q12 index and each item.
Item results are corrected for testing 12 items (Benjamini-Hochberg, q = 0.05).
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from . import stats_tools as st
from .data_ingestion import Q12_ITEMS
from .visualization import HIGH_COLOR, ITEM_LABELS, MUTED_COLOR, VISUAL_CONFIG


# PROCESS 2 – CONFIGURATION

INDEX = "q12_index"                 # mean of the 12 answers (at least 9 answered)

OUTCOME_COLUMNS = {
    "turnover": "left_voluntarily_12m",
    "absence": "unplanned_absence_days_6m",
    "performance": "performance_rating",
}

OUTCOME_LABELS = {
    "turnover": "voluntary turnover",
    "absence": "unplanned absence",
    "performance": "performance rating",
}

MODEL_LABELS = {
    "employee_fe": "Same employee over time",
    "team_fe": "Same team over time",
    "trajectory": "Engagement trajectory",
}

MIN_PANEL_EMPLOYEES = 100           # employees seen in 2+ waves
MIN_PANEL_TEAMS = 20                # teams seen in 2+ waves
MIN_TEAM_SIZE = VISUAL_CONFIG["min_team_size"]
FDR_Q = 0.05

DECLINE_COLOR = "#B5473F"           # same red as "Reversed" elsewhere in the app


# PROCESS 3 – PREPARATION

def derive_wave(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'wave' (1, 2, ...) from the survey year when the file has no wave
    column but survey dates span two or more years. Otherwise unchanged."""
    if "wave" in df.columns or "survey_date" not in df.columns:
        return df
    years = pd.to_datetime(df["survey_date"], errors="coerce").dt.year
    if years.nunique(dropna=True) < 2:
        return df
    order = {year: i + 1 for i, year in enumerate(sorted(years.dropna().unique()))}
    return df.assign(wave=years.map(order))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Respondents with numeric items and the Q12 index, one row per employee-wave."""
    data = derive_wave(df).copy()
    if "responded" in data.columns:
        data = data[pd.to_numeric(data["responded"], errors="coerce") == 1]
    items = [i for i in Q12_ITEMS if i in data.columns]
    data[items] = data[items].apply(pd.to_numeric, errors="coerce")
    answered = data[items].notna().sum(axis=1)
    data = data[answered >= VISUAL_CONFIG["min_items_answered"]].copy()
    data[INDEX] = data[items].mean(axis=1)
    if "employee_id" in data.columns and "wave" in data.columns:
        data = data.drop_duplicates(["employee_id", "wave"])
    return data


def survey_waves(df: pd.DataFrame) -> list:
    data = derive_wave(df)
    if "wave" not in data.columns:
        return []
    return sorted(data["wave"].dropna().unique().tolist())


def _measures(data: pd.DataFrame) -> list[str]:
    return [INDEX] + [i for i in Q12_ITEMS if i in data.columns]


def measure_name(measure: str) -> str:
    if measure == INDEX:
        return "overall engagement (Q12 average)"
    return f"{measure} {ITEM_LABELS.get(measure, measure)}"


def _pick_waves(data: pd.DataFrame, before: Any = None, after: Any = None) -> tuple[Any, Any]:
    """The two waves to compare: the given ones, or the last two by default."""
    waves = sorted(data["wave"].dropna().unique())
    if len(waves) < 2:
        raise ValueError("Change over time needs at least two survey waves.")
    before = waves[-2] if before is None else before
    after = waves[-1] if after is None else after
    if before == after:
        raise ValueError("Pick two different surveys to compare.")
    return (before, after) if before < after else (after, before)


# PROCESS 4 – DESCRIPTIVE CHANGE

def item_change(df: pd.DataFrame, same_employees: bool = True,
                before: Any = None, after: Any = None) -> dict[str, Any]:
    """Average score per item in the last two waves, the change, and whether
    the change is statistically clear (paired t-test for the same employees,
    Welch t-test otherwise; Benjamini-Hochberg across the 12 items)."""
    data = prepare(df)
    before, after = _pick_waves(data, before, after)
    a = data[data["wave"] == before]
    b = data[data["wave"] == after]

    if same_employees:
        if "employee_id" not in data.columns:
            raise ValueError("Comparing the same employees needs an employee_id column.")
        both = set(a["employee_id"]) & set(b["employee_id"])
        a = a[a["employee_id"].isin(both)].set_index("employee_id").sort_index()
        b = b[b["employee_id"].isin(both)].set_index("employee_id").sort_index()

    rows = []
    for m in _measures(data):
        if same_employees:
            pair = pd.concat([a[m], b[m]], axis=1, keys=["a", "b"]).dropna()
            p = stats.ttest_rel(pair["b"], pair["a"]).pvalue if len(pair) > 1 else np.nan
            mean_a, mean_b, n = pair["a"].mean(), pair["b"].mean(), len(pair)
        else:
            xa, xb = a[m].dropna(), b[m].dropna()
            p = stats.ttest_ind(xb, xa, equal_var=False).pvalue if min(len(xa), len(xb)) > 1 else np.nan
            mean_a, mean_b, n = xa.mean(), xb.mean(), min(len(xa), len(xb))
        rows.append({"measure": m, "name": measure_name(m), "before": mean_a,
                     "after": mean_b, "change": mean_b - mean_a, "p_value": p, "n": n})

    table = pd.DataFrame(rows)
    items = table["measure"] != INDEX
    table["clear_change"] = False
    table.loc[items, "clear_change"] = st.bh_fdr(table.loc[items, "p_value"].fillna(1), FDR_Q)
    table.loc[~items, "clear_change"] = table.loc[~items, "p_value"] < 0.05
    table["direction"] = np.where(
        ~table["clear_change"], "No clear change",
        np.where(table["change"] > 0, "Rose", "Fell"))

    return {"table": table.round({"before": 2, "after": 2, "change": 2}),
            "before": before, "after": after, "same_employees": same_employees,
            "n_employees": int(table["n"].max()) if len(table) else 0}


def team_change(df: pd.DataFrame, before: Any = None, after: Any = None) -> pd.DataFrame:
    """Team average Q12 index in the last two waves (teams with 5+ respondents
    in both), sorted from biggest fall to biggest rise."""
    data = prepare(df)
    if "team_id" not in data.columns:
        return pd.DataFrame()
    before, after = _pick_waves(data, before, after)
    agg = (data[data["wave"].isin([before, after])]
           .groupby(["team_id", "wave"])[INDEX].agg(["mean", "size"]).reset_index())
    agg = agg[agg["size"] >= MIN_TEAM_SIZE]
    wide = agg.pivot(index="team_id", columns="wave", values="mean").dropna()
    if wide.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"team_id": wide.index, "before": wide[before].values,
                        "after": wide[after].values})
    out["change"] = out["after"] - out["before"]
    if "department" in data.columns:
        dept = data.groupby("team_id")["department"].agg(lambda s: s.mode().iat[0])
        out["department"] = out["team_id"].map(dept)
    return out.sort_values("change").reset_index(drop=True).round(
        {"before": 2, "after": 2, "change": 2})


# PROCESS 5 – PANEL MODELS

def _fixed_effects(panel: pd.DataFrame, y: str, x: str, unit: str, cluster: str) -> dict | None:
    """Two-way fixed effects (unit + wave) by demeaning within unit and
    including wave dummies. Returns the slope on x with clustered 95% CI."""
    cols = list(dict.fromkeys([unit, cluster, "wave", y, x]))
    d = panel[cols].dropna()
    d = d[d.groupby(unit)["wave"].transform("nunique") >= 2]
    if d[unit].nunique() < 2:
        return None
    dummies = pd.get_dummies(d["wave"].astype(str), prefix="w", drop_first=True, dtype=float)
    work = pd.concat([d[[unit, y, x]], dummies], axis=1)
    dm = st.demean(work, [y, x] + list(dummies.columns), unit)
    if dm[x].abs().sum() == 0:
        return None
    try:
        fit = st.ols(dm[y], dm[[x] + list(dummies.columns)], cluster=d[cluster])
    except np.linalg.LinAlgError:
        return None
    lo, hi = fit.ci(x)
    return {"effect": float(fit.params[x]), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(fit.pvalues[x]), "n_obs": len(d), "n_units": int(d[unit].nunique()),
            "typical_change": float(d.sort_values("wave").groupby(unit)[x].diff().std())}


def _employee_fe(data: pd.DataFrame, notes: list[str]) -> list[dict]:
    if "employee_id" not in data.columns:
        notes.append("Same-employee models need an employee_id column.")
        return []
    repeat = data.groupby("employee_id")["wave"].transform("nunique") >= 2
    panel = data[repeat]
    n = panel["employee_id"].nunique()
    if n < MIN_PANEL_EMPLOYEES:
        notes.append(f"Only {n} employees answered 2+ surveys (need {MIN_PANEL_EMPLOYEES}); "
                     "same-employee models were skipped.")
        return []
    rows = []
    for outcome in ["performance", "absence"]:
        column = OUTCOME_COLUMNS[outcome]
        if column not in panel.columns:
            continue
        for m in _measures(panel):
            fit = _fixed_effects(panel, column, m, "employee_id", "employee_id")
            if fit:
                rows.append({"model": "employee_fe", "outcome": outcome, "measure": m, **fit,
                             "effect_type": "Rating points" if outcome == "performance" else "Days"})
    return rows


def _team_fe(data: pd.DataFrame, notes: list[str]) -> list[dict]:
    if "team_id" not in data.columns:
        notes.append("Same-team models need a team_id column.")
        return []
    outcomes = {k: v for k, v in OUTCOME_COLUMNS.items() if v in data.columns}
    measures = _measures(data)
    team = (data.groupby(["team_id", "wave"])
            .agg(size=(INDEX, "size"), **{m: (m, "mean") for m in measures},
                 **{k: (v, "mean") for k, v in outcomes.items()})
            .reset_index())
    team = team[team["size"] >= MIN_TEAM_SIZE]
    n = team.groupby("team_id")["wave"].nunique().ge(2).sum()
    if n < MIN_PANEL_TEAMS:
        notes.append(f"Only {n} teams have 5+ respondents in 2+ surveys (need "
                     f"{MIN_PANEL_TEAMS}); same-team models were skipped.")
        return []
    effect_types = {"turnover": "Percentage points", "absence": "Days", "performance": "Rating points"}
    rows = []
    for outcome in outcomes:
        y = team[outcome] * (100 if outcome == "turnover" else 1)
        panel = team.assign(_y=y)
        for m in measures:
            fit = _fixed_effects(panel, "_y", m, "team_id", "team_id")
            if fit:
                rows.append({"model": "team_fe", "outcome": outcome, "measure": m, **fit,
                             "effect_type": effect_types[outcome]})
    return rows


def _trajectory(data: pd.DataFrame, notes: list[str]) -> list[dict]:
    column = OUTCOME_COLUMNS["turnover"]
    if column not in data.columns or "employee_id" not in data.columns:
        return []
    data = data.sort_values(["employee_id", "wave"])
    rows = []
    for m in _measures(data):
        d = data.assign(_prev=data.groupby("employee_id")[m].shift(),
                        _prev_wave=data.groupby("employee_id")["wave"].shift())
        d = d.dropna(subset=["_prev", m, column])
        d = d.assign(_change=d[m] - d["_prev"])
        if len(d) < MIN_PANEL_EMPLOYEES or d[column].nunique() < 2:
            if m == INDEX:
                notes.append("Too few employees with a previous survey to model "
                             "engagement trajectory and turnover.")
            return rows
        numeric = [m, "_change"] + (["tenure_years"] if "tenure_years" in d.columns else [])
        categorical = [c for c in ["department", "role_level", "pay_band"] if c in d.columns]
        if d["wave"].nunique() > 1:
            categorical.append("wave")
        try:
            fit = st.logit(d[column], st.design(d, numeric, categorical),
                           cluster=d["team_id"] if "team_id" in d.columns else d["employee_id"])
        except np.linalg.LinAlgError:
            continue
        ratio, lo, hi = fit.ratio("_change")
        rows.append({"model": "trajectory", "outcome": "turnover", "measure": m,
                     "effect_type": "Odds ratio", "effect": ratio, "ci_low": lo, "ci_high": hi,
                     "p_value": float(fit.pvalues["_change"]), "n_obs": len(d),
                     "n_units": int(d["employee_id"].nunique()),
                     "typical_change": float(d["_change"].std())})
    return rows


def analyze_change(df: pd.DataFrame) -> dict[str, Any]:
    """Run all change-over-time models. Returns {"results", "notes", "waves", ...}."""
    data = prepare(df)
    if "wave" not in data.columns or data["wave"].nunique() < 2:
        raise ValueError("Change-over-time models need at least two survey waves.")

    notes: list[str] = []
    rows = _employee_fe(data, notes) + _team_fe(data, notes) + _trajectory(data, notes)
    results = pd.DataFrame(rows)
    if results.empty:
        return {"results": results, "notes": notes, "waves": sorted(data["wave"].unique())}

    # Index: plain p < 0.05. Items: Benjamini-Hochberg within each model x outcome.
    results["clear_link"] = False
    is_index = results["measure"] == INDEX
    results.loc[is_index, "clear_link"] = results.loc[is_index, "p_value"] < 0.05
    for _, idx in results[~is_index].groupby(["model", "outcome"]).groups.items():
        results.loc[idx, "clear_link"] = st.bh_fdr(results.loc[idx, "p_value"], FDR_Q)

    results["name"] = results["measure"].map(measure_name)
    results["explanation"] = results.apply(explain_change_finding, axis=1)
    return {"results": results, "notes": notes, "waves": sorted(data["wave"].unique()),
            "n_panel_employees": int(data.groupby("employee_id")["wave"].nunique().ge(2).sum())
            if "employee_id" in data.columns else 0}


# PROCESS 6 – PLAIN-LANGUAGE EXPLANATIONS

def explain_change_finding(row: pd.Series) -> str:
    name, effect = row["name"], float(row["effect"])
    subject = name if row["measure"] == INDEX else f"{name} score"
    outcome = OUTCOME_LABELS[row["outcome"]]
    unclear = "" if row.get("clear_link", True) else " This link is not statistically clear."
    typical = float(row.get("typical_change") or 0)

    if row["model"] == "trajectory":
        drop_ratio = 1 / effect if effect > 0 else float("nan")
        change = (drop_ratio - 1) * 100
        return (f"Among employees with the same current {subject}, those whose score "
                f"fell 1 point since the last survey had {abs(change):.0f}% "
                f"{'higher' if change > 0 else 'lower'} odds of leaving.{unclear}")

    who = "an employee's" if row["model"] == "employee_fe" else "a team's average"
    unit = {"Percentage points": "percentage points", "Days": "days",
            "Rating points": "rating points"}[row["effect_type"]]
    text = (f"When {who} {subject} rises 1 point between surveys, {outcome} "
            f"changes by {effect:+.2f} {unit}")
    if row["model"] == "team_fe" and typical > 0:
        text += (f". Teams typically move about {typical:.2f} points, which goes with "
                 f"{effect * typical:+.2f} {unit}")
    return text + "." + unclear


def explain_item_change(change: dict[str, Any]) -> dict[str, Any]:
    table = change["table"]
    index = table[table["measure"] == INDEX].iloc[0]
    items = table[table["measure"] != INDEX]
    who = "the same employees" if change["same_employees"] else "all respondents"

    if index["clear_change"]:
        headline = (f"Overall engagement {'rose' if index['change'] > 0 else 'fell'} "
                    f"{abs(index['change']):.2f} points from wave {_w(change['before'])} to "
                    f"wave {_w(change['after'])} ({index['before']:.2f} → {index['after']:.2f}).")
    else:
        headline = (f"Overall engagement held steady from wave {_w(change['before'])} to "
                    f"wave {_w(change['after'])} ({index['before']:.2f} → {index['after']:.2f}).")

    points = []
    rose = items[items["direction"] == "Rose"].sort_values("change", ascending=False)
    fell = items[items["direction"] == "Fell"].sort_values("change")
    if not rose.empty:
        points.append("Clearly improved: " + ", ".join(
            f"**{r['name']}** ({r['change']:+.2f})" for _, r in rose.head(3).iterrows()) + ".")
    if not fell.empty:
        points.append("Clearly declined: " + ", ".join(
            f"**{r['name']}** ({r['change']:+.2f})" for _, r in fell.head(3).iterrows()) + ".")
    if rose.empty and fell.empty:
        points.append("No single question changed by more than chance would explain.")
    points.append(f"Compared using {who} ({change['n_employees']:,} people).")

    how_to_read = (
        "Each bar is the change in a question's average score (1–5 scale) between the "
        "last two surveys. Bars to the right went up, to the left went down. Green and red "
        "bars are changes too large to be chance (after correcting for testing 12 "
        "questions); gray bars could be normal noise. Comparing the same employees "
        "removes the effect of people joining or leaving.")
    return {"headline": headline, "points": points, "how_to_read": how_to_read}


def explain_team_change(teams: pd.DataFrame) -> dict[str, Any]:
    rose, fell = teams.iloc[-1], teams.iloc[0]
    up = int((teams["change"] >= 0.25).sum())
    down = int((teams["change"] <= -0.25).sum())
    return {
        "headline": f"{up} teams rose and {down} fell by a quarter point or more.",
        "points": [
            f"Biggest rise: team **{rose['team_id']}** ({rose['before']:.2f} → {rose['after']:.2f}).",
            f"Biggest fall: team **{fell['team_id']}** ({fell['before']:.2f} → {fell['after']:.2f}).",
            "Small teams move more by chance, so treat single-team swings as a prompt for a "
            "conversation, not a verdict.",
        ],
        "how_to_read": (
            "Each row is a team. The gray dot is its average engagement in the previous "
            "survey and the colored dot is the latest one; the line shows the move. Green = "
            "improved, red = declined. Only teams with 5+ respondents in both surveys are shown."),
    }


def _w(value: Any) -> str:
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(value)
    except (TypeError, ValueError):
        return str(value)


# PROCESS 7 – CHARTS

def item_change_chart(change: dict[str, Any]) -> alt.LayerChart:
    """Diverging bars: change per question, colored by clear rise / fall / noise."""
    table = change["table"][change["table"]["measure"] != INDEX].copy()
    table["label"] = table["measure"] + " · " + table["measure"].map(ITEM_LABELS)
    limit = max(0.05, float(table["change"].abs().max()) * 1.3)
    bars = alt.Chart(table).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("change", order="descending"),
                title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("change:Q", title="Change in average score (1–5 scale)",
                scale=alt.Scale(domain=[-limit, limit]), axis=alt.Axis(format="+.2f")),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=["Rose", "Fell", "No clear change"],
                                        range=[HIGH_COLOR, DECLINE_COLOR, "#B9B1A8"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Question"),
                 alt.Tooltip("before:Q", title=f"Wave {_w(change['before'])}", format=".2f"),
                 alt.Tooltip("after:Q", title=f"Wave {_w(change['after'])}", format=".2f"),
                 alt.Tooltip("change:Q", title="Change", format="+.2f"),
                 alt.Tooltip("direction:N", title="Verdict")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#555").encode(x="x:Q")
    return (bars + zero).properties(height=alt.Step(28), width="container")


def team_movers_chart(teams: pd.DataFrame, n: int = 8) -> alt.LayerChart:
    """Dumbbell chart of the n biggest falls and n biggest rises."""
    movers = pd.concat([teams.head(n), teams.tail(n)]).drop_duplicates("team_id")
    movers = movers.assign(direction=np.where(movers["change"] >= 0, "Improved", "Declined"))
    order = list(movers.sort_values("change", ascending=False)["team_id"])
    y = alt.Y("team_id:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False))
    low = max(1.0, float(movers[["before", "after"]].min().min()) - 0.2)
    high = min(5.0, float(movers[["before", "after"]].max().max()) + 0.2)
    x_scale = alt.Scale(domain=[low, high])
    tooltip = [alt.Tooltip("team_id:N", title="Team"),
               alt.Tooltip("before:Q", title="Previous", format=".2f"),
               alt.Tooltip("after:Q", title="Latest", format=".2f"),
               alt.Tooltip("change:Q", title="Change", format="+.2f")]
    color = alt.Color("direction:N", title=None,
                      scale=alt.Scale(domain=["Improved", "Declined"],
                                      range=[HIGH_COLOR, DECLINE_COLOR]),
                      legend=alt.Legend(orient="top"))
    line = alt.Chart(movers).mark_rule(strokeWidth=2).encode(
        y=y, x=alt.X("before:Q", scale=x_scale, title="Average Q12 score (1–5)"),
        x2="after:Q", color=color, tooltip=tooltip)
    before = alt.Chart(movers).mark_circle(size=70, color=MUTED_COLOR, opacity=1).encode(
        y=y, x=alt.X("before:Q", scale=x_scale), tooltip=tooltip)
    after = alt.Chart(movers).mark_circle(size=90, opacity=1).encode(
        y=y, x=alt.X("after:Q", scale=x_scale), color=color, tooltip=tooltip)
    return (line + before + after).properties(height=alt.Step(26), width="container")



# PROCESS 8 – RESULTS BY OUTCOME (FOR LEADERS)

SHORT_UNITS = {"Rating points": "rating pts", "Days": "days", "Percentage points": "pct pts"}

BETTER = {
    "turnover": "lower voluntary turnover",
    "absence": "fewer unplanned absence days",
    "performance": "higher performance ratings",
}

MODEL_SHORT = {
    "employee_fe": "Same people",
    "team_fe": "Same teams",
    "trajectory": "Score drop",
}


def _favorable(row: pd.Series) -> bool:
    """Rising engagement goes with the good direction for this outcome."""
    if row["model"] == "trajectory":
        return row["effect"] < 1                     # odds ratio per +1 change
    if row["outcome"] == "performance":
        return row["effect"] > 0
    return row["effect"] < 0


def short_effect(row: pd.Series, typical_note: bool = True) -> str:
    """'−0.73 days per +1 point' style text for one model result."""
    if row["model"] == "trajectory":
        change = (1 / row["effect"] - 1) * 100 if row["effect"] > 0 else float("nan")
        return (f"1-point drop → {abs(change):.0f}% "
                f"{'higher' if change > 0 else 'lower'} odds of leaving")
    unit = SHORT_UNITS[row["effect_type"]]
    text = f"{row['effect']:+.2f} {unit} per +1 point"
    typical = float(row.get("typical_change") or 0)
    if typical_note and row["model"] == "team_fe" and typical > 0:
        text += f" (≈ {row['effect'] * typical:+.2f} for a typical {typical:.2f} move)"
    return text


def outcome_summary(results: pd.DataFrame) -> list[dict[str, Any]]:
    """One card per outcome: a verdict plus one short line per model
    (overall Q12 index only)."""
    index = results[results["measure"] == INDEX]
    cards = []
    for outcome in ["turnover", "absence", "performance"]:
        rows = index[index["outcome"] == outcome]
        if rows.empty:
            continue
        clear = rows[rows["clear_link"]]
        if clear.empty:
            verdict = "No clear link yet."
        elif clear.apply(_favorable, axis=1).all():
            verdict = f"Rising engagement goes with {BETTER[outcome]}."
        elif not clear.apply(_favorable, axis=1).any():
            verdict = "Rising engagement goes with worse results here; check the data."
        else:
            verdict = "Mixed evidence across models."
        lines = [{"model": MODEL_SHORT[r["model"]], "text": short_effect(r),
                  "clear": bool(r["clear_link"]),
                  "n": f"{int(r['n_units']):,} {'teams' if r['model'] == 'team_fe' else 'people'}"}
                 for _, r in rows.iterrows()]
        cards.append({"outcome": OUTCOME_LABELS[outcome].capitalize(), "key": outcome,
                      "verdict": verdict, "lines": lines})
    return cards


def question_matrix(results: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Per question for one outcome: one short column per model, ✓ when clear."""
    rows = results[(results["outcome"] == outcome) & (results["measure"] != INDEX)]
    if rows.empty:
        return pd.DataFrame()
    table = rows.assign(
        cell=rows.apply(lambda r: ("✓ " if r["clear_link"] else "· ")
                        + short_effect(r, typical_note=False), axis=1),
        column=rows["model"].map(MODEL_SHORT))
    matrix = table.pivot(index="name", columns="column", values="cell")
    order = [MODEL_SHORT[m] for m in MODEL_SHORT if MODEL_SHORT[m] in matrix.columns]
    matrix = matrix[order]
    # Questions with the most clear links first.
    clear_count = rows.groupby("name")["clear_link"].sum()
    return (matrix.assign(_clear=clear_count).sort_values("_clear", ascending=False)
            .drop(columns="_clear").reset_index().rename(columns={"name": "Question"}))
