"""
Data health check (v2) for the Gallup Decision Intelligence Agent.

Part of the product: it runs automatically every time the agent loads a dataset,
before any analysis, on synthetic or real data.

Run:    python health_check.py <data folder>
Reads:  <folder>/agent_data/survey_responses.csv, individual_outcomes.csv, team_outcomes.csv
Writes: <folder>/reports/health_report.csv   for people
        <folder>/reports/health_report.json  for the agent and its statistical tools

In the app: check_upload(df) runs the same checks on one uploaded file
(survey answers and outcomes in the same table).

Status
  PASS  no issue
  WARN  the analysis continues, with the adjustment listed in "actions"
  FAIL  a data error; blocks every analysis (integrity) or only the affected outcome (dates)

Action vocabulary (read by the statistical tools)
  add_control:<column>          include this control in the model
  within_group:<column>         report the result within groups, not pooled
  suppress_groups:<n>           never show results for groups smaller than n
  same_employees_only           compare waves only on employees present in both
  add_caveat:<CODE>             attach the caveat text of check CODE to the answer
  block:all                     refuse every analysis until the data is fixed
  block:<outcome>               refuse analyses of that outcome only
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG: rename here for a real dataset
# ---------------------------------------------------------------------------
ITEMS = [f"Q{i:02d}" for i in range(1, 13)]
OUTCOMES = {  # outcome -> (value column, column holding the end of its observation window)
    "turnover": ("left_voluntarily_12m", "turnover_window_end"),
    "absence": ("unplanned_absence_days_6m", "absence_window_end"),
    "performance": ("performance_rating", "performance_review_date"),
}
SURVEY_DATE = "survey_date"
GROUPS = ["department", "role_level", "pay_band", "shift_type", "tenure_band"]
MIN_GROUP_N = 5
SMD_WARN = 0.10
RATE_GAP_WARN = 0.05
CHECKER_VERSION = "2.1"

OUTCOME_SOURCE_COLUMNS = (
    [c for pair in OUTCOMES.values() for c in pair] + ["prior_performance_rating"]
)


def strength_of(x, weak, moderate, strong):
    x = abs(x)
    return "strong" if x >= strong else "moderate" if x >= moderate else "weak" if x >= weak else "none"


def smd(a, b):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = np.sqrt((a.var() + b.var()) / 2)
    return 0.0 if sd == 0 or np.isnan(sd) else float((a.mean() - b.mean()) / sd)


def cronbach(x):
    x = x.dropna()
    k = x.shape[1]
    if len(x) < 3:
        return float("nan")
    return k / (k - 1) * (1 - x.var().sum() / x.sum(axis=1).var())


# ---------------------------------------------------------------------------
# THE CHECKS
# ---------------------------------------------------------------------------

def build_report(survey, indiv, team=None, dataset_name="dataset", base_wave=None,
                 survey_raw_columns=None):
    """Run every check. Returns {"summary": …, "checks": […]}.

    survey  one row per employee per wave: wave, employee_id, team_id, Q01-Q12, groups
    indiv   one row per employee per wave: wave, employee_id + outcome columns
    team    wave, team_id, team_size (None = derived from the survey)
    base_wave  the wave most checks describe (default: the first, as for the dev data)
    survey_raw_columns  columns of the survey file as delivered (leakage screen)
    """
    R = []

    def add(step, code, check, status, detail, metrics=None, strength="none", affects=("all",),
            actions=()):
        R.append({"step": step, "code": code, "check": check, "status": status, "detail": detail,
                  "metrics": metrics or {}, "evidence_strength": strength, "affects": list(affects),
                  "actions": list(actions)})

    survey = survey.copy()
    indiv = indiv.copy()
    for q in [i for i in ITEMS if i in survey.columns]:
        survey[q] = pd.to_numeric(survey[q], errors="coerce")
    if "responded" not in survey.columns:
        survey["responded"] = 1                       # every row is a completed survey
    if "wave" not in survey.columns:
        survey["wave"], indiv["wave"] = 1, 1
    raw_columns = survey_raw_columns if survey_raw_columns is not None else list(survey.columns)
    leak_cols = [c for c in raw_columns if any(k in c for k in ["left_", "absence", "rating"])]
    survey = survey.drop(columns=[c for c in survey.columns
                                  if c in indiv.columns and c not in ("wave", "employee_id")])
    if team is None and "team_id" in survey.columns:
        team = survey.groupby(["wave", "team_id"]).size().rename("team_size").reset_index()
    items = [i for i in ITEMS if i in survey.columns]
    outcomes = {o: cols for o, cols in OUTCOMES.items() if cols[0] in indiv.columns}
    waves = sorted(survey["wave"].dropna().unique())
    base = base_wave if base_wave is not None else waves[0]

    # -----------------------------------------------------------------------
    # STEP 1. Integrity (any FAIL blocks every analysis)
    # -----------------------------------------------------------------------
    s = "1 Integrity"
    dup = int(survey.duplicated(["wave", "employee_id"]).sum())
    add(s, "INT_DUPLICATE", "Duplicate employee per wave", "FAIL" if dup else "PASS", f"{dup:,} duplicate rows",
        {"violations": dup}, "strong" if dup else "none", actions=["block:all"] if dup else [])

    resp_mask = survey["responded"] == 1
    vals = survey.loc[resp_mask, items]
    bad = int(((vals < 1) | (vals > 5)).sum().sum())
    add(s, "INT_SCORE_RANGE", "Q12 scores within 1-5", "FAIL" if bad else "PASS", f"{bad:,} out-of-range values",
        {"violations": bad}, "strong" if bad else "none", actions=["block:all"] if bad else [])

    if "team_id" in survey.columns and team is not None:
        orphan = int((~survey["team_id"].isin(team["team_id"]) | survey["team_id"].isna()).sum())
        add(s, "INT_ORPHAN_TEAM", "Every employee maps to a team", "FAIL" if orphan else "PASS",
            f"{orphan:,} orphan rows", {"violations": orphan}, "strong" if orphan else "none",
            actions=["block:all"] if orphan else [])

    df = survey.merge(indiv, on=["wave", "employee_id"], how="left", indicator=True)
    unmatched = int((df["_merge"] != "both").sum()) if outcomes else 0
    add(s, "INT_UNMATCHED_OUTCOME", "Every survey row has outcomes", "FAIL" if unmatched else "PASS",
        f"{unmatched:,} unmatched rows", {"violations": unmatched}, "strong" if unmatched else "none",
        actions=["block:all"] if unmatched else [])

    neg = 0
    for column in ["tenure_years", "unplanned_absence_days_6m"]:
        if column in df.columns:
            neg += int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
    add(s, "INT_NEGATIVE_VALUES", "No negative tenure or absence", "FAIL" if neg else "PASS",
        f"{neg:,} negative values", {"violations": neg}, "strong" if neg else "none",
        actions=["block:all"] if neg else [])

    partial = int((vals.isna().sum(axis=1) > 0).sum())
    add(s, "INT_PARTIAL_RESPONSE", "Respondents answered all 12 items", "WARN" if partial else "PASS",
        f"{partial:,} partial responses", {"violations": partial}, "weak" if partial else "none",
        actions=["add_caveat:INT_PARTIAL_RESPONSE"] if partial else [])

    # Analysis frame: clean rows only, so later checks still describe the data
    df = df[df["_merge"] == "both"] if outcomes else df
    df = df.drop(columns="_merge").drop_duplicates(["wave", "employee_id"])
    if "tenure_years" in df.columns:
        df["tenure_band"] = pd.cut(df["tenure_years"], [-np.inf, 1.99, 5, 10, np.inf],
                                   labels=["<2y", "2-5y", "5-10y", "10y+"]).astype(str)
    groups = [g for g in GROUPS if g in df.columns]
    resp = df[df["responded"] == 1]
    w1 = df[df["wave"] == base]

    # -----------------------------------------------------------------------
    # STEP 2. Coverage
    # -----------------------------------------------------------------------
    s = "2 Coverage"
    for g in [g for g in ["department", "role_level", "pay_band", "shift_type"] if g in w1.columns]:
        share = w1[g].value_counts(normalize=True)
        if share.empty:
            continue
        status = "WARN" if share.min() < 0.02 else "PASS"
        add(s, f"COV_GROUP_SIZE_{g}", f"Group sizes: {g}", status,
            f"smallest {share.idxmin()} {share.min():.1%}, largest {share.idxmax()} {share.max():.1%}",
            {"smallest_share": round(float(share.min()), 4)}, "weak" if status == "WARN" else "none",
            actions=[f"add_caveat:COV_GROUP_SIZE_{g}"] if status == "WARN" else [])
    if team is not None:
        tw = team[team["wave"] == base]
        if len(tw):
            n_small = int((tw["team_size"] < MIN_GROUP_N).sum())
            add(s, "COV_SMALL_TEAMS", f"Teams below n={MIN_GROUP_N} (must be suppressed)",
                "WARN" if n_small else "PASS", f"{n_small / len(tw):.1%} of teams ({n_small:,})",
                {"teams": n_small, "share": round(n_small / len(tw), 4)},
                "moderate" if n_small else "none", actions=[f"suppress_groups:{MIN_GROUP_N}"] if n_small else [])

    # -----------------------------------------------------------------------
    # STEP 3. Nonresponse (the message depends on the evidence)
    # -----------------------------------------------------------------------
    s = "3 Nonresponse"
    if w1["responded"].nunique() < 2:
        add(s, "NR_DATA", "Non-respondents in the file", "WARN",
            "The file holds only completed surveys, so nonresponse bias cannot be checked. "
            "Include non-respondents (responded = 0) to test it.",
            {}, "weak", actions=["add_caveat:NR_DATA"])
    else:
        rate = w1["responded"].mean()
        add(s, "NR_OVERALL", "Overall response rate", "PASS" if rate >= 0.7 else "WARN", f"{rate:.1%}",
            {"rate": round(float(rate), 4)}, "none" if rate >= 0.7 else "moderate",
            actions=[] if rate >= 0.7 else ["add_caveat:NR_OVERALL"])
        for g in groups:
            r = w1.groupby(g)["responded"].mean()
            gap = r.max() - r.min()
            st_ = "WARN" if gap > RATE_GAP_WARN else "PASS"
            add(s, f"NR_GAP_{g}", f"Response rate gap by {g}", st_,
                f"{r.idxmin()} {r.min():.1%} vs {r.idxmax()} {r.max():.1%} (gap {gap:.1%})",
                {"gap": round(float(gap), 4)}, "moderate" if st_ == "WARN" else "none",
                actions=[f"add_caveat:NR_GAP_{g}"] if st_ == "WARN" else [])

        smds = {}
        for v in [c for c in ["tenure_years", "unplanned_absence_days_6m", "performance_rating",
                              "prior_performance_rating"] if c in w1.columns]:
            d_ = smd(w1.loc[w1.responded == 1, v], w1.loc[w1.responded == 0, v])
            smds[v] = d_
            st_ = "WARN" if abs(d_) > SMD_WARN else "PASS"
            add(s, f"NR_SMD_{v}", f"Respondents vs non-respondents: {v}", st_, f"SMD {d_:+.3f}",
                {"smd": round(d_, 3)}, strength_of(d_, 0.1, 0.2, 0.5),
                actions=[f"add_caveat:NR_SMD_{v}"] if st_ == "WARN" else [])

        if "turnover" in outcomes:
            rr = w1.groupby("left_voluntarily_12m")["responded"].mean()
            lv, sy = float(rr.get(1, np.nan)), float(rr.get(0, np.nan))
            gap = sy - lv
            worse = (smds.get("unplanned_absence_days_6m", 0) < -SMD_WARN
                     or smds.get("performance_rating", 0) > SMD_WARN
                     or smds.get("prior_performance_rating", 0) > SMD_WARN)
            if abs(gap) <= RATE_GAP_WARN:
                st_, strength, msg = "PASS", "none", (
                    f"Leavers {lv:.1%} vs stayers {sy:.1%}: response rates are similar, so "
                    "nonresponse has little effect on turnover analysis.")
            elif gap > 0 and worse:
                st_, strength, msg = "WARN", "strong", (
                    f"Leavers {lv:.1%} vs stayers {sy:.1%}, and non-respondents have more absence or "
                    "lower ratings. Turnover effects are likely understated; raise response rates in "
                    "high-risk groups.")
            elif gap > 0:
                st_, strength, msg = "WARN", "moderate", (
                    f"Leavers {lv:.1%} vs stayers {sy:.1%}. Respondents may look more positive than the "
                    "workforce; turnover effects may be understated.")
            else:
                st_, strength, msg = "WARN", "moderate", (
                    f"Leavers {lv:.1%} vs stayers {sy:.1%}: leavers answer MORE often, the reverse of "
                    "the usual pattern. Turnover effects may be overstated; check how the survey was "
                    "distributed.")
            add(s, "NR_LEAVERS", "Leavers vs stayers response rate", st_, msg,
                {"leaver_rate": round(lv, 4), "stayer_rate": round(sy, 4), "gap": round(gap, 4)}, strength,
                affects=["turnover"], actions=["add_caveat:NR_LEAVERS"] if st_ == "WARN" else [])

    # -----------------------------------------------------------------------
    # STEP 4. Response style
    # -----------------------------------------------------------------------
    s = "4 Response style"
    r1 = resp[resp["wave"] == base][items].astype(float).dropna()
    if len(r1):
        straight = float((r1.nunique(axis=1) == 1).mean())
        add(s, "RS_STRAIGHTLINE", "Straight-lining (same answer to all 12)", "WARN" if straight > 0.05 else "PASS",
            f"{straight:.1%} of respondents", {"share": round(straight, 4)}, "weak" if straight > 0.05 else "none",
            actions=["add_caveat:RS_STRAIGHTLINE"] if straight > 0.05 else [])
        extreme = float((r1.isin([1, 5]).mean(axis=1) > 0.8).mean())
        add(s, "RS_EXTREME", "Extreme responding (>80% of answers are 1 or 5)", "WARN" if extreme > 0.05 else "PASS",
            f"{extreme:.1%} of respondents", {"share": round(extreme, 4)}, "weak" if extreme > 0.05 else "none",
            actions=["add_caveat:RS_EXTREME"] if extreme > 0.05 else [])
        top = (r1 == 5).mean()
        add(s, "RS_CEILING", "Ceiling effect (item with most 5s)", "WARN" if top.max() > 0.5 else "PASS",
            f"{top.idxmax()} {top.max():.1%} answered 5", {"max_share": round(float(top.max()), 4)},
            "weak" if top.max() > 0.5 else "none", actions=["add_caveat:RS_CEILING"] if top.max() > 0.5 else [])
        agree = float((r1 >= 4).mean().mean())
        add(s, "RS_ACQUIESCENCE", "Acquiescence (share agree/strongly agree)", "WARN" if agree > 0.75 else "PASS",
            f"{agree:.1%}", {"share": round(agree, 4)}, "weak" if agree > 0.75 else "none",
            actions=["add_caveat:RS_ACQUIESCENCE"] if agree > 0.75 else [])

    # -----------------------------------------------------------------------
    # STEP 5. Measurement
    # -----------------------------------------------------------------------
    s = "5 Measurement"
    r1g = resp[resp["wave"] == base].dropna(subset=items)
    if len(r1g) >= 3:
        a_all = cronbach(r1g[items].astype(float))
        add(s, "MS_ALPHA", "Reliability (Cronbach's alpha), all", "PASS" if a_all >= 0.7 else "WARN",
            f"alpha {a_all:.2f}", {"alpha": round(float(a_all), 3)}, "none" if a_all >= 0.7 else "moderate",
            actions=[] if a_all >= 0.7 else ["add_caveat:MS_ALPHA"])
        for g in [g for g in ["department", "role_level", "tenure_band"] if g in r1g.columns]:
            a = r1g.groupby(g)[items].apply(lambda x: cronbach(x.astype(float))).dropna()
            if len(a) < 2:
                continue
            st_ = "WARN" if a.max() - a.min() > 0.05 else "PASS"
            add(s, f"MS_ALPHA_{g}", f"Reliability consistent across {g}", st_,
                f"alpha {a.min():.2f} ({a.idxmin()}) to {a.max():.2f} ({a.idxmax()})",
                {"range": round(float(a.max() - a.min()), 3)}, "weak" if st_ == "WARN" else "none",
                actions=[f"add_caveat:MS_ALPHA_{g}"] if st_ == "WARN" else [])
        if "department" in r1g.columns:
            # DIF screen: an item's department mean (in SD units) compared with the typical item in
            # that department (median across items, so one biased item cannot distort the baseline)
            z = (r1g[items] - r1g[items].mean()) / r1g[items].std()
            dept_means = z.groupby(r1g["department"]).mean()
            dif = dept_means.sub(dept_means.median(axis=1), axis=0)
            flags = [q for q in items if dif[q].max() - dif[q].min() > 0.3]
            add(s, "MS_DIF", "Item bias by department (DIF screen)", "WARN" if flags else "PASS",
                ("Items behave differently by department: " + ", ".join(flags)) if flags
                else "no item behaves differently by department",
                {"items": flags}, "moderate" if flags else "none",
                actions=["add_control:department", "add_caveat:MS_DIF"] if flags else [])

    # -----------------------------------------------------------------------
    # STEP 6. Outcome labels
    # -----------------------------------------------------------------------
    s = "6 Outcome labels"
    for g in [g for g in ["role_level", "pay_band", "department"] if g in w1.columns]:
        if "performance" in outcomes:
            p = w1.groupby(g)["performance_rating"].mean()
            d_ = (p.max() - p.min()) / w1["performance_rating"].std()
            st_ = "WARN" if d_ > 0.2 else "PASS"
            add(s, f"OL_PERF_GAP_{g}", f"Performance rating gap by {g}", st_,
                f"{p.idxmin()} {p.min():.2f} vs {p.idxmax()} {p.max():.2f} ({d_:.2f} SD)",
                {"gap_sd": round(float(d_), 3)}, strength_of(d_, 0.2, 0.5, 1.0), affects=["performance"],
                actions=[f"add_control:{g}"] if st_ == "WARN" else [])
        if "turnover" in outcomes:
            t = w1.groupby(g)["left_voluntarily_12m"].mean()
            st_ = "WARN" if t.max() - t.min() > RATE_GAP_WARN else "PASS"
            add(s, f"OL_TURNOVER_GAP_{g}", f"Turnover rate gap by {g}", st_,
                f"{t.idxmin()} {t.min():.1%} vs {t.idxmax()} {t.max():.1%}",
                {"gap": round(float(t.max() - t.min()), 4)}, "moderate" if st_ == "WARN" else "none",
                affects=["turnover"], actions=[f"add_control:{g}"] if st_ == "WARN" else [])
    if "performance" in outcomes and "prior_performance_rating" in w1.columns:
        c = float(w1[["performance_rating", "prior_performance_rating"]].corr().iloc[0, 1])
        add(s, "OL_PRIOR_ANCHOR", "Rating tracks prior rating", "WARN" if c > 0.3 else "PASS",
            f"r = {c:.2f}", {"r": round(c, 3)}, strength_of(c, 0.3, 0.5, 0.7), affects=["performance"],
            actions=["add_control:prior_performance_rating"])

    # -----------------------------------------------------------------------
    # STEP 7. Survivorship across waves (the last two waves)
    # -----------------------------------------------------------------------
    s = "7 Survivorship"
    if len(waves) >= 2:
        prev, last = waves[-2], waves[-1]
        prev_ids = set(df.loc[df.wave == prev, "employee_id"])
        w2 = df[df.wave == last]
        share = float(w2["employee_id"].isin(prev_ids).mean())
        st_ = "WARN" if share < 0.9 else "PASS"
        add(s, "SV_STAYER_SHARE", f"Wave {last} made of wave {prev} stayers", st_,
            f"{share:.1%} stayers, {1 - share:.1%} new hires", {"stayer_share": round(share, 4)},
            "moderate" if st_ == "WARN" else "none", affects=["trend"],
            actions=["same_employees_only"] if st_ == "WARN" else [])
        if "turnover" in outcomes:
            eng = resp.assign(eng=resp[items].astype(float).mean(axis=1))
            wp = df[df.wave == prev]
            left_ids = set(wp.loc[wp.left_voluntarily_12m == 1, "employee_id"])
            e1 = eng[eng.wave == prev]
            d_ = smd(e1.loc[~e1.employee_id.isin(left_ids), "eng"], e1.loc[e1.employee_id.isin(left_ids), "eng"])
            st_ = "WARN" if abs(d_) > SMD_WARN else "PASS"
            add(s, "SV_LEAVER_ENGAGEMENT", "Leavers were less engaged (later waves over-represent engaged staff)",
                st_, f"stayers vs leavers mean score SMD {d_:+.2f}", {"smd": round(d_, 3)},
                strength_of(d_, 0.1, 0.3, 0.6), affects=["trend"],
                actions=["same_employees_only", "add_caveat:SV_LEAVER_ENGAGEMENT"] if st_ == "WARN" else [])

    # -----------------------------------------------------------------------
    # STEP 8. Aggregation (Simpson's paradox screen, every item x outcome)
    # -----------------------------------------------------------------------
    s = "8 Aggregation"
    if "department" in resp.columns and outcomes:
        e1 = resp[resp.wave == base].dropna(subset=items)
        rev, pairs = [], []
        for q in items:
            x = e1[q].astype(float)
            for name, (col, _) in outcomes.items():
                y = pd.to_numeric(e1[col], errors="coerce")
                if x.std() == 0 or y.std() == 0:
                    continue
                pooled = np.corrcoef(x, y)[0, 1]
                xd = x - x.groupby(e1["department"]).transform("mean")
                od = y - y.groupby(e1["department"]).transform("mean")
                within = np.corrcoef(xd, od)[0, 1]
                if np.sign(pooled) != np.sign(within) and max(abs(pooled), abs(within)) > 0.03:
                    rev.append(f"{q} x {col} (pooled {pooled:+.2f}, within-department {within:+.2f})")
                    pairs.append({"item": q, "outcome": name, "pooled_r": round(float(pooled), 3),
                                  "within_r": round(float(within), 3)})
        affected = sorted({p["outcome"] for p in pairs}) or ["all"]
        add(s, "AG_SIMPSON", "Pooled vs within-department direction flips", "WARN" if rev else "PASS",
            "; ".join(rev) if rev else "no reversals", {"pairs": pairs}, "strong" if rev else "none",
            affects=affected,
            actions=["within_group:department", "add_control:department", "add_caveat:AG_SIMPSON"] if rev else [])

    # -----------------------------------------------------------------------
    # STEP 9. Time order and leakage (dates are checked, never assumed)
    # -----------------------------------------------------------------------
    s = "9 Leakage"
    merged = (survey[["wave", "employee_id", SURVEY_DATE]].merge(indiv, on=["wave", "employee_id"], how="inner")
              if SURVEY_DATE in survey.columns else None)
    for name, (col, date_col) in outcomes.items():
        code = f"LK_DATE_{name}"
        if merged is None or date_col not in indiv.columns:
            missing = SURVEY_DATE if merged is None else date_col
            add(s, code, f"{name.capitalize()} measured after the survey", "WARN",
                f"Column {missing} is missing, so the time order cannot be verified. "
                f"Results for {name} carry a caveat.",
                {"violations": None, "missing_column": missing}, "moderate", affects=[name],
                actions=[f"add_caveat:{code}"])
            continue
        sd = pd.to_datetime(merged[SURVEY_DATE], errors="coerce")
        od = pd.to_datetime(merged[date_col], errors="coerce")
        bad = int((od <= sd).sum())
        unreadable = int((sd.isna() | od.isna()).sum())
        if bad:
            add(s, code, f"{name.capitalize()} measured after the survey", "FAIL",
                f"{bad:,} rows have {date_col} on or before the survey date. {name.capitalize()} analysis is "
                "blocked; other outcomes can continue.", {"violations": bad, "unreadable_dates": unreadable},
                "strong", affects=[name], actions=[f"block:{name}"])
        elif unreadable:
            add(s, code, f"{name.capitalize()} measured after the survey", "WARN",
                f"{unreadable:,} rows have unreadable dates.", {"violations": 0, "unreadable_dates": unreadable},
                "weak", affects=[name], actions=[f"add_caveat:{code}"])
        else:
            add(s, code, f"{name.capitalize()} measured after the survey", "PASS",
                f"all {len(merged):,} rows: {date_col} is after the survey date", {"violations": 0},
                affects=[name])
    add(s, "LK_SURVEY_COLUMNS", "No outcome columns inside the survey file", "FAIL" if leak_cols else "PASS",
        ", ".join(leak_cols) if leak_cols else "none found", {"violations": len(leak_cols), "columns": leak_cols},
        "strong" if leak_cols else "none", actions=["block:all"] if leak_cols else [])

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    rep = pd.DataFrame(R)
    counts = rep["status"].value_counts().to_dict()
    blocked = sorted({a.split(":", 1)[1] for acts in rep["actions"] for a in acts if a.startswith("block:")})
    if "all" in blocked:
        blocked = ["all"]
    global_actions = sorted({a for acts in rep["actions"] for a in acts if not a.startswith("block:")})
    summary = {
        "checker_version": CHECKER_VERSION,
        "dataset": dataset_name,
        "base_wave": int(base) if float(base).is_integer() else base,
        "checks_run": int(len(rep)),
        "pass": int(counts.get("PASS", 0)), "warn": int(counts.get("WARN", 0)), "fail": int(counts.get("FAIL", 0)),
        "blocked_outcomes": blocked,
        "analysis_allowed": {o: ("all" not in blocked and o not in blocked) for o in OUTCOMES},
        "actions": global_actions,
    }
    return {"summary": summary, "checks": R}


# ---------------------------------------------------------------------------
# UPLOADS: one table holding survey answers and outcomes
# ---------------------------------------------------------------------------

def _model_wave(df):
    """Newest wave with outcome data (the wave the statistical models use)."""
    if "wave" not in df.columns:
        return None
    outcome_cols = [c for c, _ in OUTCOMES.values() if c in df.columns]
    labeled = df.loc[df[outcome_cols].notna().any(axis=1), "wave"] if outcome_cols else df["wave"]
    return labeled.max() if labeled.notna().any() else df["wave"].max()


def check_upload(df, dataset_name="uploaded_dataset"):
    """Run every check on one uploaded table. Outcome columns in the same table are
    expected here, so the 'outcomes inside the survey file' leakage screen is not run
    on them; the other checks describe the wave the models use."""
    data = df.copy()
    if "wave" not in data.columns:
        data["wave"] = 1
    outcome_cols = [c for c in OUTCOME_SOURCE_COLUMNS if c in data.columns]
    indiv = data[["wave", "employee_id"] + outcome_cols].drop_duplicates(["wave", "employee_id"])
    survey = data.drop(columns=outcome_cols)
    return build_report(survey, indiv, None, dataset_name, base_wave=_model_wave(data),
                        survey_raw_columns=list(survey.columns))


# ---------------------------------------------------------------------------
# PLAIN-LANGUAGE VIEW OF THE ACTIONS (for the app)
# ---------------------------------------------------------------------------

def blocked_outcomes(report):
    """['all'] = every analysis stopped; otherwise the outcomes that are blocked."""
    return list(report["summary"].get("blocked_outcomes", []))


def required_adjustments(report):
    """One plain sentence per adjustment the WARN checks require, with its reason."""
    lines = []
    seen = set()
    for check in report["checks"]:
        if check["status"] == "PASS":
            continue
        scope = ", ".join(a for a in check["affects"] if a != "all") or "every analysis"
        for action in check["actions"]:
            kind, _, value = action.partition(":")
            if kind == "block":
                text = ("Every analysis is stopped until this is fixed" if value == "all"
                        else f"{value.capitalize()} analysis is stopped until this is fixed")
            elif kind == "add_control":
                text = f"Models for {scope} also account for {value.replace('_', ' ')}"
            elif kind == "within_group":
                text = f"Results compare people within the same {value.replace('_', ' ')}"
            elif kind == "suppress_groups":
                text = f"Groups with fewer than {value} people are never shown"
            elif kind == "same_employees_only":
                text = "Changes over time compare the same people in both surveys"
            elif kind == "add_caveat":
                text = f"Results for {scope} carry a caveat"
            else:
                text = action
            key = (text, check["code"])
            if key in seen:
                continue
            seen.add(key)
            lines.append({"adjustment": text, "because": check["check"], "detail": check["detail"],
                          "code": check["code"], "status": check["status"]})
    return lines


def model_controls(report, outcome):
    """Extra categorical controls the report requires for one outcome's model."""
    controls = []
    for check in report["checks"]:
        if check["status"] == "PASS":
            continue
        if "all" not in check["affects"] and outcome not in check["affects"]:
            continue
        for action in check["actions"]:
            if action.startswith("add_control:"):
                controls.append(action.split(":", 1)[1])
    return list(dict.fromkeys(controls))


# ---------------------------------------------------------------------------
# COMMAND LINE: the development dataset folder
# ---------------------------------------------------------------------------

def main(folder):
    d = f"{folder}/agent_data"
    survey = pd.read_csv(f"{d}/survey_responses.csv", dtype={"employee_id": str, "team_id": str})
    indiv = pd.read_csv(f"{d}/individual_outcomes.csv", dtype={"employee_id": str})
    team = pd.read_csv(f"{d}/team_outcomes.csv", dtype={"team_id": str})
    report = build_report(survey, indiv, team, os.path.basename(os.path.normpath(folder)),
                          survey_raw_columns=list(survey.columns))

    os.makedirs(f"{folder}/reports", exist_ok=True)
    rep = pd.DataFrame(report["checks"])
    rep[["step", "code", "check", "status", "detail"]].to_csv(f"{folder}/reports/health_report.csv", index=False)
    with open(f"{folder}/reports/health_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    summary = report["summary"]
    pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 200)
    print(rep[["step", "code", "status", "detail"]].to_string(index=False))
    print(f"\n{summary['checks_run']} checks: {summary['pass']} PASS, {summary['warn']} WARN, "
          f"{summary['fail']} FAIL; blocked: {summary['blocked_outcomes'] or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
