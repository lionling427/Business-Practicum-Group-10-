"""Analytical traps: three ways engagement results commonly mislead, tested on
every outcome with the same fixed models (model_spec).

1. Department trap (Simpson's paradox)
   Fit each outcome's model with and without department. A question whose link
   flips direction, or shrinks by half or more, once departments are compared
   fairly would mislead anyone reading department-blind numbers. The app's
   reported results already control for department; this shows what a naive
   comparison would get wrong.

2. Manager trap (hidden team effects)
   How much of engagement and of each outcome sits between teams (ICC), and
   whether each question's link survives a within-team comparison (team fixed
   effects, a linear model on the same controls). A link that mostly disappears
   within teams reflects differences between teams and their managers, not
   between people.

3. Nonresponse trap
   Needs non-respondents in the file (responded = 0). Compares respondents with
   non-respondents, checks whether leavers answered less, and reweights
   respondents (inverse response probability by department, role and tenure) to
   see how far the engagement and outcome picture moves.

run_trap_checks(df) returns, per trap: status (clear | caution | trap |
cannot_test), a plain headline, points and a detail table.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import stats_tools as st
from .insights import ITEM_LABELS, analysis_frame
from .model_spec import SPECS, design_matrix
from .waves import model_wave, wave_label


# PROCESS 2 – CONFIGURATION

SHRINK = 0.50          # a link that loses half its size or more = caution
ALPHA = 0.05           # only links a naive report would show (significant) can mislead
ICC_NOTABLE = 0.10     # 10%+ of differences between teams = a notable team effect
WEIGHT_SHIFT = 0.05    # engagement moving 0.05+ points after reweighting = caution
RATE_GAP = 0.05        # leavers vs stayers response gap
OUTCOME_LABEL = {o: SPECS[o].label.split(" (")[0].lower() for o in SPECS}


def _name(item: str) -> str:
    return f"{ITEM_LABELS.get(item, item)} ({item})"


def _links(n: int) -> str:
    return f"{n} question–outcome link" + ("" if n == 1 else "s")


def _lost(share_remaining: float) -> str:
    lost = 1 - share_remaining
    return "all" if lost >= 1 else f"about {lost:.0%}"


def _affected(table: pd.DataFrame) -> list[tuple[str, str]]:
    if table.empty:
        return []
    hit = table[table["verdict"] != "ok"]
    return list(zip(hit["outcome"], hit["item"]))


def _status(flips: int, shrinks: int) -> str:
    return "trap" if flips else "caution" if shrinks else "clear"


def _compare(naive, fair, name: str) -> tuple[str, float]:
    """Compare one coefficient in a naive fit with the fair (adjusted) fit.

    'flip'   the naive link is significant and the fair one is significant the
             other way (a true reversal);
    'shrink' the naive link is significant but half or more of it disappears
             (including a non-significant change of sign);
    'ok'     otherwise, including naive links too weak to be reported anyway.
    Returns the verdict and the share of the naive link that remains."""
    a, b = float(naive.params[name]), float(fair.params[name])
    remains = b / a if a else 1.0
    if naive.pvalues[name] >= ALPHA:
        return "ok", remains
    if np.sign(a) != np.sign(b) and fair.pvalues[name] < ALPHA:
        return "flip", remains
    return ("shrink" if remains <= 1 - SHRINK else "ok"), remains


# PROCESS 3 – DEPARTMENT TRAP

def department_trap(sample: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for outcome, spec in SPECS.items():
        if spec.outcome_column not in sample.columns or "department" not in sample.columns:
            continue
        data = sample.dropna(subset=[spec.outcome_column])
        adjusted = _fit(data, outcome, drop_department=False)
        pooled = _fit(data, outcome, drop_department=True)
        ratio = spec.method in ("logit", "negbin")
        for item in st.ITEMS:
            p = float(np.exp(pooled.params[item]) if ratio else pooled.params[item])
            a = float(np.exp(adjusted.params[item]) if ratio else adjusted.params[item])
            verdict, remains = _compare(pooled, adjusted, item)
            rows.append({"outcome": outcome, "item": item, "question": _name(item),
                         "ignoring_department": p, "comparing_within_department": a,
                         "share_remaining": remains, "verdict": verdict})
    table = pd.DataFrame(rows)
    if table.empty:
        return _cannot("Needs a department column and at least one business outcome.")

    flips = table[table["verdict"] == "flip"]
    shrinks = table[table["verdict"] == "shrink"]
    status = _status(len(flips), len(shrinks))
    if status == "clear":
        headline = ("No department trap: every question's link points the same way whether or "
                    "not departments are taken into account.")
    else:
        headline = (f"Ignoring departments would mislead on "
                    f"{_links(len(flips) + len(shrinks))}. The app's results already compare people within "
                    "departments, so they are protected; department-blind reports are not.")
    points = []
    for _, r in flips.head(3).iterrows():
        points.append(f"**{r['question']} and {OUTCOME_LABEL[r['outcome']]}:** the link reverses "
                      "once departments are compared fairly (Simpson's paradox).")
    for _, r in shrinks.head(3).iterrows():
        points.append(f"**{r['question']} and {OUTCOME_LABEL[r['outcome']]}:** "
                      f"{_lost(r['share_remaining'])} of the link is really a difference "
                      "between departments.")
    return {"status": status, "headline": headline, "points": points, "table": table,
            "affected": _affected(table)}


def _fit(data: pd.DataFrame, outcome: str, drop_department: bool):
    """The outcome's fixed model, optionally without the department control."""
    spec = SPECS[outcome]
    X = design_matrix(data, outcome)
    if drop_department:
        X = X[[c for c in X.columns if not c.startswith("department_")]]
    y = data[spec.outcome_column]
    if spec.outcome_column in spec.standardize:
        y = pd.Series(st.zscore(y), index=data.index)
    fit = {"logit": st.logit, "negbin": st.negbin, "ols": st.ols}[spec.method]
    return fit(y, X, data[spec.cluster] if spec.cluster in data.columns else None)


# PROCESS 4 – MANAGER (TEAM) TRAP

def _linear_y(data: pd.DataFrame, outcome: str) -> pd.Series:
    """The outcome on a linear scale for the within-team comparison."""
    column = SPECS[outcome].outcome_column
    if outcome == "absence":
        return np.log1p(data[column].astype(float))
    if outcome == "performance":
        return pd.Series(st.zscore(data[column]), index=data.index)
    return data[column].astype(float)                    # turnover: probability scale


def manager_trap(sample: pd.DataFrame) -> dict[str, Any]:
    if "team_id" not in sample.columns:
        return _cannot("Needs a team_id column to separate team (manager) effects.")

    index = sample[list(st.ITEMS)].mean(axis=1)
    icc_engagement, _ = st.icc_oneway(index, sample["team_id"])
    iccs, rows = {"engagement": float(icc_engagement)}, []

    for outcome, spec in SPECS.items():
        if spec.outcome_column not in sample.columns:
            continue
        data = sample.dropna(subset=[spec.outcome_column])
        y = _linear_y(data, outcome)
        iccs[outcome] = float(st.icc_oneway(y, data["team_id"])[0])

        X = design_matrix(data, outcome)
        between = st.ols(y, X, data["team_id"])
        X = X.drop(columns="Intercept")
        work = pd.concat([X, y.rename("_y"), data["team_id"]], axis=1)
        demeaned = st.demean(work, list(X.columns) + ["_y"], "team_id")
        keep = [c for c in X.columns if demeaned[c].abs().sum() > 1e-9]
        within = st.ols(demeaned["_y"], demeaned[keep], data["team_id"])
        for item in st.ITEMS:
            verdict, remains = _compare(between, within, item)
            rows.append({"outcome": outcome, "item": item, "question": _name(item),
                         "across_teams": float(between.params[item]),
                         "within_same_team": float(within.params[item]),
                         "share_remaining": remains, "verdict": verdict})

    table = pd.DataFrame(rows, columns=["outcome", "item", "question", "across_teams",
                                        "within_same_team", "share_remaining", "verdict"])
    flips = table[table["verdict"] == "flip"]
    shrinks = table[table["verdict"] == "shrink"]
    notable = {k: v for k, v in iccs.items() if v >= ICC_NOTABLE}
    status = _status(len(flips), len(shrinks))
    if status == "clear" and notable:
        status = "caution"

    team_share = ", ".join(f"{OUTCOME_LABEL.get(k, k)} {v:.0%}" for k, v in iccs.items())
    headline = ("Links hold up within teams: they are about people, not just about which "
                "manager they have." if status == "clear" else
                f"{_links(len(flips) + len(shrinks))} largely "
                + ("reflects" if len(flips) + len(shrinks) == 1 else "reflect")
                + " differences between teams "
                "(and their managers) rather than between people."
                if len(flips) + len(shrinks) else
                "Some measures differ a lot between teams ("
                + ", ".join(f"{OUTCOME_LABEL.get(k, k)} {v:.0%}" for k, v in notable.items())
                + "), so compare and act team by team; each question's link still holds "
                "within teams.")
    points = [f"Share of differences that sit between teams: {team_share}."]
    for _, r in pd.concat([flips, shrinks]).head(4).iterrows():
        points.append(f"**{r['question']} and {OUTCOME_LABEL[r['outcome']]}:** "
                      + ("reverses within the same team." if r["verdict"] == "flip" else
                         f"{_lost(r['share_remaining'])} of the link disappears within the "
                         "same team."))
    if iccs.get("performance", 0) >= ICC_NOTABLE:
        points.append(f"Performance ratings differ strongly between teams "
                      f"({iccs['performance']:.0%}); part of that may be how strictly each "
                      "manager rates, so compare ratings within teams.")
    if len(flips) + len(shrinks):
        points.append("Act at team level for these: coach the manager, not only individuals.")
    return {"status": status, "headline": headline, "points": points, "table": table,
            "icc": iccs, "affected": _affected(table)}


# PROCESS 5 – NONRESPONSE TRAP

def nonresponse_trap(wave_rows: pd.DataFrame) -> dict[str, Any]:
    if "responded" not in wave_rows.columns or wave_rows["responded"].nunique() < 2:
        return _cannot("The file holds only people who answered, so nonresponse bias cannot be "
                       "tested. Include everyone invited, with responded = 1 or 0, to test it.")

    data = wave_rows.copy()
    responded = data["responded"] == 1
    rate = float(responded.mean())
    points = [f"Response rate: {rate:.0%} of {len(data):,} people invited."]
    caution = rate < 0.7
    rows = []

    # Respondents vs non-respondents on what both groups have (outcomes, tenure).
    for column, label, kind in [("left_voluntarily_12m", "voluntary turnover", "rate"),
                                ("unplanned_absence_days_6m", "absence days", "mean"),
                                ("performance_rating", "performance rating", "mean"),
                                ("tenure_years", "tenure (years)", "mean")]:
        if column not in data.columns:
            continue
        r, n = data.loc[responded, column].mean(), data.loc[~responded, column].mean()
        rows.append({"measure": label, "respondents": r, "non_respondents": n, "kind": kind})

    if "left_voluntarily_12m" in data.columns:
        by_left = data.groupby("left_voluntarily_12m")["responded"].mean()
        leavers, stayers = float(by_left.get(1, np.nan)), float(by_left.get(0, np.nan))
        gap = stayers - leavers
        if abs(gap) > RATE_GAP:
            caution = True
            points.append(f"Leavers answered less ({leavers:.0%} vs {stayers:.0%} of stayers), so "
                          "turnover links are probably understated." if gap > 0 else
                          f"Leavers answered more ({leavers:.0%} vs {stayers:.0%}), so turnover "
                          "links may be overstated.")
        else:
            points.append(f"Leavers and stayers answered at similar rates ({leavers:.0%} vs "
                          f"{stayers:.0%}).")

    # Reweighting: respondents weighted up where response was low.
    cells = [c for c in ["department", "role_level"] if c in data.columns]
    if "tenure_years" in data.columns:
        data["tenure_band"] = pd.cut(data["tenure_years"], [-np.inf, 2, 5, 10, np.inf]).astype(str)
        cells.append("tenure_band")
    if cells:
        p = data.groupby(cells)["responded"].transform("mean").clip(lower=0.05)
        resp = data[responded].assign(w=1 / p[responded])
        items = [i for i in st.ITEMS if i in resp.columns]
        index = resp[items].mean(axis=1)
        plain, weighted = float(index.mean()), float(np.average(index, weights=resp["w"]))
        shift = weighted - plain
        if abs(shift) >= WEIGHT_SHIFT:
            caution = True
        points.append(f"Reweighting respondents to match everyone invited moves average "
                      f"engagement from {plain:.2f} to {weighted:.2f} ({shift:+.2f}).")
        lowest = data.groupby(cells[0])["responded"].mean().sort_values()
        points.append(f"Lowest response: {lowest.index[0]} ({lowest.iloc[0]:.0%}); "
                      f"highest: {lowest.index[-1]} ({lowest.iloc[-1]:.0%}).")

    status = "caution" if caution else "clear"
    headline = ("Non-respondents look different from respondents, so survey results may not "
                "speak for everyone." if caution else
                "Respondents look like the whole workforce, so nonresponse is unlikely to "
                "mislead.")
    return {"status": status, "headline": headline, "points": points,
            "table": pd.DataFrame(rows), "affected": []}


def _cannot(reason: str) -> dict[str, Any]:
    return {"status": "cannot_test", "headline": reason, "points": [], "table": pd.DataFrame(),
            "affected": []}


# PROCESS 6 – RUN ALL THREE

def run_trap_checks(df: pd.DataFrame, findings: pd.DataFrame | None = None) -> dict[str, Any]:
    """All three traps on the survey the statistical models use.

    `findings` (analysis["findings"]) marks which trapped links the app itself
    reports, so a leader knows which headline results to read with care."""
    wave = model_wave(df)
    wave_rows = df if wave is None else df[df["wave"] == wave]
    sample = analysis_frame(df)                        # respondents in the model wave
    traps = {
        "wave": wave_label(wave),
        "department": department_trap(sample),
        "manager": manager_trap(sample),
        "nonresponse": nonresponse_trap(wave_rows),
    }
    reported = set() if findings is None or findings.empty else \
        set(zip(findings["outcome"], findings["item"]))
    for name in ("department", "manager"):
        hit = [pair for pair in traps[name]["affected"] if pair in reported]
        traps[name]["reported_affected"] = hit
        if name == "manager" and hit:          # department is already controlled in the models
            traps[name]["points"].insert(1, "Reported findings this affects: " + "; ".join(
                f"{_name(item)} and {OUTCOME_LABEL[outcome]}" for outcome, item in hit) + ".")
    return traps


TRAP_NAMES = {
    "department": "Misleading department effects",
    "manager": "Hidden manager (team) effects",
    "nonresponse": "Nonresponse bias",
}

STATUS_BADGES = {
    "clear": ":green-badge[:material/check: Clear]",
    "caution": ":orange-badge[:material/warning: Caution]",
    "trap": ":red-badge[:material/error: Trap found]",
    "cannot_test": ":gray-badge[:material/help: Can't test yet]",
}


def trap_summary(traps: dict[str, Any]) -> list[str]:
    """One plain line per trap, for the AI agent."""
    return [f"{TRAP_NAMES[k]}: {traps[k]['status'].replace('_', ' ')}. {traps[k]['headline']} "
            + " ".join(p.replace("**", "") for p in traps[k]["points"])
            for k in TRAP_NAMES]
