"""
Synthetic ground-truth generator for the Gallup Decision Intelligence Agent (v2).

Builds a fake but realistic employee dataset whose true answers are known, so the
agent can be graded. Three business outcomes: voluntary turnover, absenteeism and
performance. Planted: 5 true effects (T1-T5), 3 true nulls (N1-N3), 8 traps (X1-X8).

Output folder layout
  <out>/agent_data/   survey_responses.csv, individual_outcomes.csv, team_outcomes.csv
                      -> the ONLY files the agent may read
  <out>/answer_key/   ground_truth.json, teams_with_manager_quality.csv,
                      full_data_with_latents.csv (small datasets only),
                      error_log.csv (dirty variants only) -> graders only
  <out>/README.md     written automatically with the real counts

Examples
  python generate_synthetic_data.py --employees 5000 --out gallup_v2_5K
  python generate_synthetic_data.py --employees 1000000 --out gallup_v2_2M
  python generate_synthetic_data.py --employees 50000 --variant null --out testsets/null
  python generate_synthetic_data.py --employees 5000 --variant dirty_A --out testsets/dirty_A

Variants
  main     the standard test data
  null     every engagement item has zero effect on every outcome
  weak     T1-T3 half as strong on the log scale, T4 beta 0.10, T5 IRR 0.87 (kept outside the no-effect band)
  seed     same as main; use --seed 2 or 3 to check stability
  dirty_A  main data + one error of each type (tests each health-check rule)
  dirty_B  main data + many overlapping errors and a missing date column
  dirty_C  main data + boundary values (some valid, some just invalid)

Multi-year variants (v3.0, 5 waves by default, last year's outcomes hidden)
  long_main      main truths over N yearly waves, with manager changes and three planted events
  long_null      null truths (no item affects any outcome); X7 nonresponse still on
  long_purenull  null truths AND no nonresponse bias (X7 off): nothing in the survey predicts turnover
  long_seed      long_main with another seed (use --seed 2)
  Events: E1 better manager (wave 3), E2 worse manager (wave 4), E3 inflated scores without
  better outcomes (wave 4). answer_key/ adds hidden_outcomes.csv, manager_changes.csv,
  team_true_risk.csv and events.csv.
  python generate_synthetic_data.py --employees 50000 --variant long_main --out testsets/long_main
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find stats_tools.py next to this file
import stats_tools as st

GENERATOR_VERSION = "3.0"

# ---------------------------------------------------------------------------
# STEP 0. Configuration
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--employees", type=int, default=5000, help="employees per wave")
ap.add_argument("--out", default="output", help="output folder")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--variant", default="main",
                choices=["main", "null", "weak", "seed", "dirty_A", "dirty_B", "dirty_C",
                         "long_main", "long_null", "long_purenull", "long_seed"])
ap.add_argument("--waves", type=int, default=None,
                help="yearly survey waves (default 2; 5 for long_* variants)")
ap.add_argument("--hide-last-outcome", choices=["auto", "yes", "no"], default="auto",
                help="blank the newest wave's outcomes and move them to answer_key/ (auto: long_* only)")
ap.add_argument("--full-data", choices=["auto", "yes", "no"], default="auto",
                help="write answer_key/full_data_with_latents.csv (auto: 2-wave sets up to 50,000 employees)")
ARGS = ap.parse_args()

LONG = ARGS.variant.startswith("long_")
BASE_VARIANT = {"long_main": "main", "long_null": "null", "long_purenull": "null",
                "long_seed": "seed"}.get(ARGS.variant, ARGS.variant)
N_WAVES = ARGS.waves or (5 if LONG else 2)
if N_WAVES < 2:
    sys.exit("--waves must be 2 or more")
HIDE_LAST = ARGS.hide_last_outcome == "yes" or (ARGS.hide_last_outcome == "auto" and LONG)

N_EMPLOYEES = ARGS.employees
N_TEAMS = max(40, round(N_EMPLOYEES / 12.5))      # average team ~12.5 people
N_SMALL_TEAMS = max(2, round(N_TEAMS * 0.05))     # X8: 5% of teams have fewer than 5 members
DEPARTMENTS = ["Sales", "Operations", "Engineering", "Finance",
               "HR", "Marketing", "Customer Service", "Logistics"]
ITEMS = st.ITEMS

ITEM_LOADING = 0.8            # item = 0.8*engagement + 0.6*noise -> inter-item r ~ 0.6
ITEM_NOISE = 0.6
ENGAGEMENT_STABILITY = 0.7    # wave-to-wave correlation of an employee's engagement

# Fix #5: item-specific difficulty. Positive = easier to agree with.
ITEM_OFFSET = {"Q01": 0.2, "Q02": 0.1, "Q03": 0.0, "Q04": -0.3, "Q05": 0.0, "Q06": -0.3,
               "Q07": 0.0, "Q08": 0.3, "Q09": 0.3, "Q10": -0.5, "Q11": -0.3, "Q12": 0.0}
LIKERT_CUTS = [0.07, 0.20, 0.45, 0.75]   # one shared set of cut points for all 12 items

# Survey and outcome dates (fix #13: the health check can verify the time order)
WAVE_SURVEY_START = {1: date(2024, 3, 4), 2: date(2025, 3, 3)}
WAVE_REVIEW_DATE = {1: date(2025, 1, 31), 2: date(2026, 1, 30)}
for _w in range(3, N_WAVES + 1):                 # later yearly waves: first Monday of March
    _d = date(2023 + _w, 3, 1)
    WAVE_SURVEY_START[_w] = date(_d.year, 3, 1 + (7 - _d.weekday()) % 7)
    WAVE_REVIEW_DATE[_w] = date(2024 + _w, 1, 30)

# Multi-year settings (long_* only): managers move, and engagement follows the manager
MGR_TO_ENGAGEMENT = 0.4        # a 1-SD better manager lifts the team's engagement by 0.4 SD
MGR_NATURAL_CHANGE = 0.08      # share of teams that get a new manager each year (random quality)
EVENTS = {                     # planted events for the prediction tests (graders only)
    "E1": {"wave": 3, "share": 0.05, "name": "better manager", "quality_change": +1.5},
    "E2": {"wave": 4, "share": 0.05, "name": "worse manager", "quality_change": -1.5},
    "E3": {"wave": 4, "share": 0.03, "name": "inflated scores", "score_shift": 1.0},
}

TRUTH = {
    # --- True relationships (T) ---------------------------------------------
    "T1_Q04_turnover_OR": 0.70,          # per scale point
    "T1_X5_tenure_lt2_OR": 0.49,         # X5: twice as strong (log-odds) for tenure < 2 years
    "T2_Q05_turnover_OR": 0.78,
    "T3_Q12_turnover_OR": 0.75,
    "T4_Q01_performance_beta": 0.20,     # standardized, on latent performance
    "T5_Q05_absence_IRR": 0.85,          # new in v2: someone cares -> fewer absence days
    # --- Traps (X) -----------------------------------------------------------
    "X1_sales_Q02_shift": -0.9,          # Sales scores low on Q02 ...
    "X1_sales_turnover_logit": 0.9,      # ... and has high turnover (Q02 has no effect)
    "X2_prior_perf_to_Q11": 0.5,         # prior performance raises Q11
    "X3_manager_to_Q07": 0.5,            # hidden manager quality raises Q07 ...
    "X3_manager_to_absence": -0.35,      # ... and lowers absenteeism
    "X4_Q01_threshold": 3,               # Q01 matters for absence only below 3
    "X4_Q01_low_absence_logrr": 0.6,
    "X6_Q07_within_dept_beta": 0.15,     # positive within departments
    "X6_dept_Q07_spread": 0.8,           # departments differ in mean Q07 ...
    "X6_dept_perf_reversal": -1.8,       # ... and rate performance in the opposite direction
    "X7_leaver_response_rate": 0.60,     # leavers answer less often ...
    "X7_stayer_response_rate": 0.86,
    "X7_leaver_engagement_slope": 1.5,   # ... and among leavers, the disengaged answer least
    "X7_stayer_engagement_slope": 0.3,
    "X8_min_group_size": 5,
}

if BASE_VARIANT == "weak":
    for k in ["T1_Q04_turnover_OR", "T1_X5_tenure_lt2_OR", "T2_Q05_turnover_OR",
              "T3_Q12_turnover_OR", "T5_Q05_absence_IRR"]:
        TRUTH[k] = round(float(np.sqrt(TRUTH[k])), 3)
    TRUTH["T4_Q01_performance_beta"] = 0.10
    # v2.1 fix: half of T5 on the log scale (0.922) is realized near 0.94 among respondents,
    # inside the 0.91-1.10 "no meaningful effect" band, so a correct agent would not report it.
    # Every true effect must be realized outside the band; 0.87 keeps T5 weaker than main (0.85).
    TRUTH["T5_Q05_absence_IRR"] = 0.87
if BASE_VARIANT == "null":
    for k in ["T1_Q04_turnover_OR", "T1_X5_tenure_lt2_OR", "T2_Q05_turnover_OR",
              "T3_Q12_turnover_OR", "T5_Q05_absence_IRR"]:
        TRUTH[k] = 1.0
    TRUTH["T4_Q01_performance_beta"] = 0.0
    TRUTH["X3_manager_to_absence"] = 0.0      # no hidden path from Q07 to absence
    TRUTH["X4_Q01_low_absence_logrr"] = 0.0   # no threshold effect
    TRUTH["X6_Q07_within_dept_beta"] = 0.0    # no within-department effect

if ARGS.variant == "long_purenull":
    # X7 off: answering does not depend on leaving or on engagement
    TRUTH["X7_leaver_response_rate"] = TRUTH["X7_stayer_response_rate"] = 0.80
    TRUTH["X7_leaver_engagement_slope"] = TRUTH["X7_stayer_engagement_slope"] = 0.0

rng = np.random.default_rng(ARGS.seed)


def to_likert(continuous):
    """Cut one continuous score into 1-5 with its own quantiles (used for ratings)."""
    cuts = np.quantile(continuous, LIKERT_CUTS)
    return (np.searchsorted(cuts, continuous) + 1).astype(int)


# ---------------------------------------------------------------------------
# STEP 1. Organization: departments, teams, hidden manager quality
# ---------------------------------------------------------------------------
def build_org():
    team_dept = rng.choice(len(DEPARTMENTS), size=N_TEAMS)
    sizes = np.zeros(N_TEAMS, dtype=int)
    small = rng.choice(N_TEAMS, N_SMALL_TEAMS, replace=False)
    sizes[small] = rng.integers(3, 5, N_SMALL_TEAMS)
    big = np.setdiff1d(np.arange(N_TEAMS), small)
    remaining = N_EMPLOYEES - sizes.sum()
    sizes[big] = rng.multinomial(remaining - 6 * len(big), np.ones(len(big)) / len(big)) + 6
    return pd.DataFrame({
        "team_id": [f"T{t:06d}" for t in range(N_TEAMS)],
        "department": [DEPARTMENTS[d] for d in team_dept],
        "team_size": sizes,
        "manager_quality": rng.normal(0, 1, N_TEAMS),     # X3: hidden, never shown to the agent
    }).assign(**({"manager_id": [f"M{t:06d}" for t in range(N_TEAMS)]} if LONG else {}))


# ---------------------------------------------------------------------------
# STEP 2. Employees: stable attributes and two hidden traits
# ---------------------------------------------------------------------------
def build_employees(teams, start_id=0):
    rows = teams.loc[teams.index.repeat(teams["team_size"])].reset_index(drop=True)
    n = len(rows)
    return pd.DataFrame({
        "employee_id": [f"E{start_id + i:08d}" for i in range(n)],
        "team_id": rows["team_id"],
        "department": rows["department"],
        "tenure_years": np.round(rng.gamma(1.6, 2.2, n), 1),
        "role_level": rng.choice(["Individual contributor", "Senior", "Manager"], n, p=[0.65, 0.25, 0.10]),
        "pay_band": rng.choice(["B1", "B2", "B3", "B4"], n, p=[0.30, 0.35, 0.25, 0.10]),
        "shift_type": rng.choice(["Day", "Evening", "Night"], n, p=[0.70, 0.20, 0.10]),
        "engagement_latent": rng.normal(0, 1, n),
        "prior_perf_latent": rng.normal(0, 1, n),
    })


# ---------------------------------------------------------------------------
# STEP 3. Survey answers: 12 correlated items, item offsets, trap shifts
# ---------------------------------------------------------------------------
def simulate_survey(emp, teams, wave, inflated_teams=()):
    t = teams.set_index("team_id")
    mgr = emp["team_id"].map(t["manager_quality"]).to_numpy()
    dept_idx = emp["department"].map({d: i for i, d in enumerate(DEPARTMENTS)}).to_numpy()
    n = len(emp)
    dept_q07_shift = np.linspace(-1, 1, len(DEPARTMENTS)) * TRUTH["X6_dept_Q07_spread"]

    eng = emp["engagement_latent"].to_numpy()
    if LONG:
        eng = eng + MGR_TO_ENGAGEMENT * mgr          # engagement follows the current manager
    cont = {q: ITEM_LOADING * eng + ITEM_NOISE * rng.normal(0, 1, n) + ITEM_OFFSET[q] for q in ITEMS}
    cont["Q02"] += np.where(emp["department"] == "Sales", TRUTH["X1_sales_Q02_shift"], 0)   # X1
    cont["Q11"] += TRUTH["X2_prior_perf_to_Q11"] * emp["prior_perf_latent"].to_numpy()        # X2
    cont["Q07"] += TRUTH["X3_manager_to_Q07"] * mgr                                           # X3
    cont["Q07"] += dept_q07_shift[dept_idx]                                                   # X6

    pooled = np.concatenate([cont[q] for q in ITEMS])
    if LONG:
        cuts = LONG_CUTS.setdefault("cuts", np.quantile(pooled, LIKERT_CUTS))   # fixed from wave 1
    else:
        cuts = np.quantile(pooled, LIKERT_CUTS)
    survey = emp.copy()
    for q in ITEMS:
        survey[q] = (np.searchsorted(cuts, cont[q]) + 1).astype(int)
    if LONG:
        survey["manager_id"] = emp["team_id"].map(t["manager_id"]).to_numpy()
    reported = None
    if len(inflated_teams):
        # E3: these teams answer higher than they feel; outcomes follow the true answers
        mask = emp["team_id"].isin(inflated_teams).to_numpy()
        reported = survey[ITEMS].copy()
        for q in ITEMS:
            reported.loc[mask, q] = (np.searchsorted(cuts, cont[q][mask] + EVENTS["E3"]["score_shift"]) + 1)
    start = np.datetime64(WAVE_SURVEY_START[wave])
    survey["survey_date"] = start + rng.integers(0, 21, n).astype("timedelta64[D]")
    if LONG:
        return survey, dept_q07_shift, reported
    return survey, dept_q07_shift


LONG_CUTS = {}


# ---------------------------------------------------------------------------
# STEP 4. Outcomes: turnover (12 months), absence (6 months), performance
# ---------------------------------------------------------------------------
def simulate_individual_outcomes(s, teams, dept_q07_shift, wave):
    t = teams.set_index("team_id")
    mgr = s["team_id"].map(t["manager_quality"]).to_numpy()
    dept_idx = s["department"].map({d: i for i, d in enumerate(DEPARTMENTS)}).to_numpy()
    n = len(s)
    new_hire = s["tenure_years"].to_numpy() < 2

    # Turnover: T1, T2, T3, X1, X5
    b_q04 = np.where(new_hire, np.log(TRUTH["T1_X5_tenure_lt2_OR"]), np.log(TRUTH["T1_Q04_turnover_OR"]))
    logit = (-1.6
             + b_q04 * (s["Q04"] - 3)
             + np.log(TRUTH["T2_Q05_turnover_OR"]) * (s["Q05"] - 3)
             + np.log(TRUTH["T3_Q12_turnover_OR"]) * (s["Q12"] - 3)
             + np.where(s["department"] == "Sales", TRUTH["X1_sales_turnover_logit"], 0)
             + 0.35 * new_hire)
    p_leave = 1 / (1 + np.exp(-logit))
    turnover = rng.binomial(1, p_leave)

    # Absenteeism: T5, X3, X4 (N1: Q06 has no effect)
    log_mu = (np.log(3.0)
              + np.log(TRUTH["T5_Q05_absence_IRR"]) * (s["Q05"] - 3)
              + TRUTH["X4_Q01_low_absence_logrr"] * (s["Q01"] < TRUTH["X4_Q01_threshold"])
              + TRUTH["X3_manager_to_absence"] * mgr
              + np.where(s["shift_type"] == "Night", 0.25, 0))
    mu = np.exp(log_mu)
    k = 2.0
    absence_days = rng.poisson(rng.gamma(k, mu / k))

    # Performance: T4, X2, X6 (Q11 has no effect of its own)
    latent = (TRUTH["T4_Q01_performance_beta"] * st.zscore(s["Q01"])
              + TRUTH["X6_Q07_within_dept_beta"] * st.zscore(s["Q07"])
              + TRUTH["X6_dept_perf_reversal"] * dept_q07_shift[dept_idx]
              + 0.55 * s["prior_perf_latent"].to_numpy()
              + rng.normal(0, 0.75, n))
    prior_rating = to_likert(s["prior_perf_latent"].to_numpy() + rng.normal(0, 0.3, n))
    performance_rating = to_likert(latent)

    sd = s["survey_date"].to_numpy()
    out = s[["employee_id"]].copy()
    out["left_voluntarily_12m"] = turnover
    out["turnover_window_end"] = sd + np.timedelta64(365, "D")
    out["unplanned_absence_days_6m"] = absence_days
    out["absence_window_end"] = sd + np.timedelta64(182, "D")
    out["prior_performance_rating"] = prior_rating
    out["performance_rating"] = performance_rating
    out["performance_review_date"] = np.datetime64(WAVE_REVIEW_DATE[wave])
    if LONG:
        return out, np.asarray(p_leave, dtype=float)
    return out


# ---------------------------------------------------------------------------
# STEP 5. Team file: basic facts only (fix #3, #4, #7)
# ---------------------------------------------------------------------------
def team_table(survey, teams):
    sizes = survey.groupby("team_id").size()
    out = teams[["team_id", "department"]].copy()
    out["team_size"] = out["team_id"].map(sizes).fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# STEP 6. Nonresponse (fix #1, X7): depends on leaving AND on engagement
# ---------------------------------------------------------------------------
def apply_nonresponse(survey, outcomes):
    left = outcomes["left_voluntarily_12m"].to_numpy() == 1
    eng = survey["engagement_latent"].to_numpy()
    p = np.empty(len(eng))
    groups = ((left, TRUTH["X7_leaver_response_rate"], TRUTH["X7_leaver_engagement_slope"]),
              (~left, TRUTH["X7_stayer_response_rate"], TRUTH["X7_stayer_engagement_slope"]))
    for mask, target, slope in groups:
        if not mask.any():
            continue
        lo, hi = -10.0, 10.0          # calibrate the intercept so the group rate hits the target
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(mid + slope * eng[mask])))).mean() < target:
                lo = mid
            else:
                hi = mid
        p[mask] = 1 / (1 + np.exp(-(mid + slope * eng[mask])))
    responded = rng.binomial(1, p).astype(bool)
    visible = survey.copy()
    visible["responded"] = responded.astype(int)
    visible[ITEMS] = visible[ITEMS].astype("Int64")
    visible.loc[~responded, ITEMS] = pd.NA
    return visible, p


# ---------------------------------------------------------------------------
# STEP 7. Second wave: stayers carry over, leavers are replaced by new hires
# ---------------------------------------------------------------------------
NEXT_ID = [N_EMPLOYEES]       # first id for the next wave's new hires


def next_wave(emp_w1, outcomes_w1, teams):
    stay = outcomes_w1["left_voluntarily_12m"].to_numpy() == 0
    stayers = emp_w1.loc[stay].copy()
    stayers["tenure_years"] = stayers["tenure_years"] + 1.0
    stayers["engagement_latent"] = (ENGAGEMENT_STABILITY * stayers["engagement_latent"]
                                    + np.sqrt(1 - ENGAGEMENT_STABILITY ** 2) * rng.normal(0, 1, len(stayers)))
    stayers["prior_perf_latent"] = st.zscore(outcomes_w1.loc[stay, "performance_rating"]) * 0.8 \
        + rng.normal(0, 0.6, len(stayers))
    leavers = emp_w1.loc[~stay, ["team_id"]]
    counts = leavers["team_id"].value_counts()
    repl = teams.set_index("team_id").loc[counts.index].reset_index()
    repl["team_size"] = counts.to_numpy()
    hires = build_employees(repl, start_id=NEXT_ID[0])
    NEXT_ID[0] += len(hires)
    hires["tenure_years"] = np.round(rng.uniform(0, 1, len(hires)), 1)
    return pd.concat([stayers, hires], ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 7A. Multi-year data: manager changes and planted events (long_* only)
# ---------------------------------------------------------------------------
def change_managers(teams, wave, used):
    """New managers before this wave's survey: random changes plus E1/E2.
    Returns (teams, change rows, event rows, E3 teams)."""
    teams = teams.copy()
    changes, events = [], []
    n = len(teams)
    natural = rng.random(n) < MGR_NATURAL_CHANGE
    new_quality = rng.normal(0, 1, n)
    eligible = np.setdiff1d(np.arange(n), list(used))

    def pick(share, condition):
        pool = eligible[condition[eligible]]
        k = min(len(pool), max(1, round(share * n)))
        return rng.choice(pool, k, replace=False) if k else np.array([], dtype=int)

    q = teams["manager_quality"].to_numpy()
    e1 = pick(EVENTS["E1"]["share"], q < -0.5) if wave == EVENTS["E1"]["wave"] else np.array([], dtype=int)
    e2 = pick(EVENTS["E2"]["share"], q > 0.5) if wave == EVENTS["E2"]["wave"] else np.array([], dtype=int)
    used.update(e1.tolist()); used.update(e2.tolist())
    eligible = np.setdiff1d(eligible, np.concatenate([e1, e2]))
    e3 = (pick(EVENTS["E3"]["share"], np.ones(n, bool)) if wave == EVENTS["E3"]["wave"]
          else np.array([], dtype=int))
    used.update(e3.tolist())

    for i in range(n):
        reason = "E1" if i in e1 else "E2" if i in e2 else "natural" if natural[i] else None
        if reason is None:
            continue
        before_q, before_id = float(q[i]), teams.at[i, "manager_id"]
        after_q = (before_q + EVENTS[reason]["quality_change"]) if reason in ("E1", "E2") else float(new_quality[i])
        after_id = f"M{wave:02d}{i:06d}"
        teams.at[i, "manager_quality"] = after_q
        teams.at[i, "manager_id"] = after_id
        changes.append({"wave": wave, "team_id": teams.at[i, "team_id"], "reason": reason,
                        "manager_id_before": before_id, "manager_id_after": after_id,
                        "quality_before": round(before_q, 4), "quality_after": round(after_q, 4)})
        if reason != "natural":
            events.append({"wave": wave, "event": reason, "name": EVENTS[reason]["name"],
                           "team_id": teams.at[i, "team_id"],
                           "quality_before": round(before_q, 4), "quality_after": round(after_q, 4)})
    e3_teams = teams.loc[e3, "team_id"].tolist()
    for tid in e3_teams:
        events.append({"wave": wave, "event": "E3", "name": EVENTS["E3"]["name"], "team_id": tid,
                       "quality_before": None, "quality_after": None})
    return teams, changes, events, e3_teams


def team_risk_table(full_all):
    """True expected turnover per team per wave (all employees, true answers)."""
    company = full_all.groupby("wave")["turnover_prob_true"].mean().rename("company_true_rate")
    g = full_all.groupby(["wave", "team_id"])
    out = pd.DataFrame({
        "department": g["department"].first(),
        "manager_id": g["manager_id"].first(),
        "team_size": g.size(),
        "respondents": g["responded"].sum(),
        "true_turnover_rate": g["turnover_prob_true"].mean().round(4),
        "realized_turnover_rate": g["left_voluntarily_12m"].mean().round(4),
        "mean_true_score": g["true_index"].mean().round(4),
    }).reset_index().merge(company.reset_index(), on="wave")
    out["company_true_rate"] = out["company_true_rate"].round(4)
    out["true_above_company"] = (out["true_turnover_rate"] > out["company_true_rate"]).astype(int)
    return out


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# STEP 8. Realized values for the answer key (fix #18, #19)
# ---------------------------------------------------------------------------
def realized_values(full_w1):
    """Fit the standard models on (a) every employee with their true answers and
    (b) respondents only, which is all the agent can see."""
    res = {}
    for label, d in (("all_employees", full_w1), ("respondents_only", full_w1[full_w1["responded"] == 1])):
        d = d.copy()
        tm = st.turnover_model(d, cluster=False)
        am = st.absence_model(d, cluster=False)
        pm = st.performance_model(d, cluster=False)
        d["new_hire"] = (d["tenure_years"] < 2).astype(float)
        ti = st.turnover_model(d, cluster=False, extra_numeric=["new_hire"], interactions=[("Q04", "new_hire")])
        d["mean_score"] = d[ITEMS].mean(axis=1)
        X = st.design(d, ["mean_score", "tenure_years"], ["department", "role_level", "pay_band"])
        mo = st.logit(d["left_voluntarily_12m"], X)
        raw = st.logit(d["left_voluntarily_12m"], st.design(d, ["Q02"]))
        zz = pd.DataFrame({"y": st.zscore(d["performance_rating"]), "x": st.zscore(d["Q07"]),
                           "department": d["department"]})
        pooled = st.ols(zz["y"], st.design(zz, ["x"])).params["x"]
        within = st.ols(zz["y"], st.design(zz, ["x"], ["department"])).params["x"]
        r = lambda f, name: round(float(np.exp(f.params[name])), 3)
        b = lambda f, name: round(float(f.params[name]), 3)
        res[label] = {
            "n": int(len(d)),
            "T1_Q04_turnover_OR": r(tm, "Q04"),
            "T1_X5_Q04_x_new_hire_logOR": b(ti, "Q04:new_hire"),
            "T2_Q05_turnover_OR": r(tm, "Q05"),
            "T3_Q12_turnover_OR": r(tm, "Q12"),
            "T4_Q01_performance_beta": b(pm, "Q01"),
            "T5_Q05_absence_IRR": r(am, "Q05"),
            "N1_Q06_absence_IRR": r(am, "Q06"),
            "N2_Q10_turnover_OR": r(tm, "Q10"),
            "N2_Q10_absence_IRR": r(am, "Q10"),
            "N2_Q10_performance_beta": b(pm, "Q10"),
            "N3_Q08_turnover_OR": r(tm, "Q08"),
            "N3_Q08_absence_IRR": r(am, "Q08"),
            "N3_Q08_performance_beta": b(pm, "Q08"),
            "X1_Q02_turnover_OR_raw": r(raw, "Q02"),
            "X1_Q02_turnover_OR_controlled": r(tm, "Q02"),
            "X4_Q01_below3_absence_RR": r(am, "q01_low"),
            "X6_Q07_performance_beta_pooled": round(float(pooled), 3),
            "X6_Q07_performance_beta_within_dept": round(float(within), 3),
            "X7_mean_score_turnover_OR": r(mo, "mean_score"),
        }
    return res


# ---------------------------------------------------------------------------
# STEP 9. Dirty variants (fix #16): inject known errors and log them
# ---------------------------------------------------------------------------
def inject_errors(survey, indiv, team, variant):
    log = []
    sv, iv, tm = survey.copy(), indiv.copy(), team.copy()
    resp_idx = sv.index[(sv["responded"] == 1) & (sv["wave"] == 1)].to_numpy()
    pick = rng.permutation(resp_idx)
    used = iter(pick)

    def entry(eid, file, wave, key, col, before, after, etype, code, status, effect):
        log.append({"error_id": eid, "file": file, "wave": wave, "key": key, "column": col,
                    "original_value": before, "injected_value": after, "error_type": etype,
                    "expected_check": code, "expected_status": status, "expected_effect": effect})

    def out_of_range(n, values, eid_start):
        for j in range(n):
            i = next(used)
            q = ITEMS[j % 12]
            v = values[j % len(values)]
            entry(f"E{eid_start + j:03d}", "survey_responses.csv", 1, sv.at[i, "employee_id"], q,
                  int(sv.at[i, q]), v, "score outside 1-5", "INT_SCORE_RANGE", "FAIL", "blocks all analysis")
            sv.at[i, q] = v

    def duplicate(n, eid_start):
        rows = []
        for j in range(n):
            i = next(used)
            rows.append(sv.loc[[i]])
            entry(f"E{eid_start + j:03d}", "survey_responses.csv", 1, sv.at[i, "employee_id"], "(whole row)",
                  "", "row repeated", "duplicate employee in a wave", "INT_DUPLICATE", "FAIL", "blocks all analysis")
        return rows

    def orphan(n, eid_start):
        for j in range(n):
            i = next(used)
            entry(f"E{eid_start + j:03d}", "survey_responses.csv", 1, sv.at[i, "employee_id"], "team_id",
                  sv.at[i, "team_id"], "T999999", "team not in team file", "INT_ORPHAN_TEAM", "FAIL",
                  "blocks all analysis")
            sv.at[i, "team_id"] = "T999999"

    def unmatched(n, eid_start):
        drop = []
        for j in range(n):
            i = next(used)
            eid = sv.at[i, "employee_id"]
            drop.append(eid)
            entry(f"E{eid_start + j:03d}", "individual_outcomes.csv", 1, eid, "(whole row)", "present",
                  "row deleted", "survey row without outcomes", "INT_UNMATCHED_OUTCOME", "FAIL", "blocks all analysis")
        return drop

    def negative(n, col, value, eid_start):
        for j in range(n):
            i = next(used)
            eid = sv.at[i, "employee_id"]
            if col == "tenure_years":
                entry(f"E{eid_start + j:03d}", "survey_responses.csv", 1, eid, col, float(sv.at[i, col]), value,
                      "impossible negative value", "INT_NEGATIVE_VALUES", "FAIL", "blocks all analysis")
                sv.at[i, col] = value
            else:
                k = iv.index[(iv["wave"] == 1) & (iv["employee_id"] == eid)][0]
                entry(f"E{eid_start + j:03d}", "individual_outcomes.csv", 1, eid, col, int(iv.at[k, col]), value,
                      "impossible negative value", "INT_NEGATIVE_VALUES", "FAIL", "blocks all analysis")
                iv.at[k, col] = value

    def partial(n, eid_start):
        for j in range(n):
            i = next(used)
            qs = [ITEMS[(j + 3) % 12], ITEMS[(j + 7) % 12]]
            entry(f"E{eid_start + j:03d}", "survey_responses.csv", 1, sv.at[i, "employee_id"], ",".join(qs),
                  "answered", "blank", "respondent skipped items", "INT_PARTIAL_RESPONSE", "WARN",
                  "analysis continues with a caveat")
            for q in qs:
                sv.at[i, q] = pd.NA

    def bad_date(n, col, outcome, days_after_survey, eid_start, status="FAIL"):
        for j in range(n):
            i = next(used)
            eid = sv.at[i, "employee_id"]
            k = iv.index[(iv["wave"] == 1) & (iv["employee_id"] == eid)][0]
            new = (pd.Timestamp(sv.at[i, "survey_date"]) + pd.Timedelta(days=days_after_survey)).date()
            entry(f"E{eid_start + j:03d}", "individual_outcomes.csv", 1, eid, col, str(iv.at[k, col])[:10],
                  str(new), "outcome dated on or before the survey" if status == "FAIL" else "boundary date (valid)",
                  f"LK_DATE_{outcome}", status,
                  f"blocks {outcome} analysis only" if status == "FAIL" else "no effect")
            iv.at[k, col] = pd.Timestamp(new)

    if variant == "dirty_A":
        extra = duplicate(1, 1)
        out_of_range(1, [9], 2)
        orphan(1, 3)
        drop = unmatched(1, 4)
        negative(1, "unplanned_absence_days_6m", -3, 5)
        partial(1, 6)
        bad_date(1, "turnover_window_end", "turnover", -30, 7)
        sv["performance_rating"] = sv["employee_id"].map(
            iv[iv["wave"] == 1].set_index("employee_id")["performance_rating"])
        entry("E008", "survey_responses.csv", "", "(column)", "performance_rating", "absent", "column added",
              "outcome column inside the survey file", "LK_SURVEY_COLUMNS", "FAIL", "blocks all analysis")
    elif variant == "dirty_B":
        extra = duplicate(40, 1)
        out_of_range(30, [0, 6, 99], 41)
        orphan(20, 71)
        drop = unmatched(15, 91)
        negative(10, "tenure_years", -2.0, 106)
        negative(15, "unplanned_absence_days_6m", -1, 116)
        partial(40, 131)
        bad_date(10, "absence_window_end", "absence", -10, 171)
        iv = iv.drop(columns=["performance_review_date"])
        entry("E181", "individual_outcomes.csv", "", "(column)", "performance_review_date", "present",
              "column deleted", "date column missing: order cannot be verified", "LK_DATE_performance", "WARN",
              "performance analysis continues with a caveat")
    else:  # dirty_C: boundary values
        extra, drop = [], []
        # valid boundaries: must NOT be flagged
        for j, (q, v) in enumerate([("Q03", 1), ("Q09", 5)]):
            i = next(used)
            entry(f"E{1 + j:03d}", "survey_responses.csv", 1, sv.at[i, "employee_id"], q, int(sv.at[i, q]), v,
                  "boundary score (valid)", "INT_SCORE_RANGE", "PASS", "no effect")
            sv.at[i, q] = v
        i = next(used)
        eid = sv.at[i, "employee_id"]
        k = iv.index[(iv["wave"] == 1) & (iv["employee_id"] == eid)][0]
        entry("E003", "individual_outcomes.csv", 1, eid, "unplanned_absence_days_6m",
              int(iv.at[k, "unplanned_absence_days_6m"]), 0, "zero absence (valid)", "INT_NEGATIVE_VALUES",
              "PASS", "no effect")
        iv.at[k, "unplanned_absence_days_6m"] = 0
        i = next(used)
        entry("E004", "survey_responses.csv", 1, sv.at[i, "employee_id"], "tenure_years",
              float(sv.at[i, "tenure_years"]), 0.0, "zero tenure (valid)", "INT_NEGATIVE_VALUES", "PASS", "no effect")
        sv.at[i, "tenure_years"] = 0.0
        bad_date(1, "absence_window_end", "absence", 1, 5, status="PASS")
        # just-invalid values: MUST be flagged
        out_of_range(2, [0, 6], 6)
        negative(1, "tenure_years", -0.1, 8)
        bad_date(1, "absence_window_end", "absence", 0, 9)

    if extra:
        sv = pd.concat([sv] + extra, ignore_index=True)
    if drop:
        iv = iv[~((iv["wave"] == 1) & (iv["employee_id"].isin(drop)))]
    return sv, iv, tm, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# README written from the real numbers (fix #17)
# ---------------------------------------------------------------------------
def write_readme(out_dir, meta):
    v = meta["variant"]
    purpose = {
        "main": "Standard test data with 5 true effects, 3 true nulls and 8 traps.",
        "null": "Every engagement item has zero effect on every outcome. A correct agent reports no item as a driver.",
        "weak": "The true effects are weaker (T1-T3 half strength, T4 beta 0.10, T5 IRR 0.87) but still outside the no-effect band. Tests detection without exaggeration.",
        "seed": "Same design as the main data with a different random seed. Results should match the main data.",
        "dirty_A": "Main data plus one error of each type. Each health-check rule should fire once.",
        "dirty_B": "Main data plus many overlapping errors and a missing date column.",
        "dirty_C": "Main data plus boundary values. Valid edges must pass; just-invalid values must fail.",
        "long_main": f"Main truths over {N_WAVES} yearly waves, with manager changes and three planted events "
                     "(E1 better manager, E2 worse manager, E3 inflated scores). Tests trends and team predictions.",
        "long_null": f"Null truths over {N_WAVES} waves: no item affects any outcome, but nonresponse (X7) is on. "
                     "Team predictions from Q12 must not pass the attribution test.",
        "long_purenull": f"Null truths over {N_WAVES} waves with no nonresponse bias: nothing in the survey "
                         "predicts turnover. Every Q12-based prediction gate should fail.",
        "long_seed": f"long_main design with another seed. Results should match long_main.",
    }[v]
    lines = [
        f"# Synthetic test data: {os.path.basename(os.path.normpath(out_dir))}",
        "",
        f"Generated by generate_synthetic_data.py v{GENERATOR_VERSION} with seed {meta['seed']}, variant `{v}`.",
        "",
        purpose,
        "",
        "## Size",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Employees per wave | {meta['employees_per_wave']:,} |",
        f"| Survey waves | {N_WAVES} |",
        f"| Survey rows | {meta['survey_rows']:,} |",
        f"| Teams | {meta['teams']:,} |",
        f"| Teams with fewer than 5 people | {meta['small_teams']:,} ({meta['small_team_share']:.1%}) |",
        f"| Departments | {len(DEPARTMENTS)} |",
        f"| Wave 1 response rate | {meta['response_rate_w1']:.1%} |",
        f"| Wave 1 voluntary turnover | {meta['turnover_w1']:.1%} |",
        "",
        "## Files",
        "",
        "| File | Who may read it | Contents |",
        "|---|---|---|",
        "| `agent_data/survey_responses.csv` | Agent | One row per employee per wave: Q01-Q12 (blank for non-respondents), department, tenure, role, pay band, shift, survey date"
        + (", manager_id (changes when a team gets a new manager)" if LONG else "") + " |",
        "| `agent_data/individual_outcomes.csv` | Agent | Turnover (12 months), unplanned absence days (6 months), prior and current performance rating, and the date each outcome window ends"
        + (f". Wave {N_WAVES} outcomes are blank: that year has not finished" if HIDE_LAST else "") + " |",
        "| `agent_data/team_outcomes.csv` | Agent | Team id, department and team size per wave |",
        "| `answer_key/ground_truth.json` | Graders only | Planted settings, realized values, true nulls, traps, seed |",
        "| `answer_key/teams_with_manager_quality.csv` | Graders only | Hidden manager quality per team (trap X3) |",
    ]
    if meta.get("full_data"):
        lines.append("| `answer_key/full_data_with_latents.csv` | Graders only | Every employee's true answers and hidden engagement |")
    if v.startswith("dirty"):
        lines.append("| `answer_key/error_log.csv` | Graders only | Every injected error and the health-check result it should produce |")
    if LONG:
        lines += [
            "| `answer_key/hidden_outcomes.csv` | Graders only | The newest wave's real outcomes (blank in agent_data), for grading forecasts |" if HIDE_LAST else "",
            "| `answer_key/manager_changes.csv` | Graders only | Every manager change: wave, team, reason, hidden quality before and after |",
            "| `answer_key/team_true_risk.csv` | Graders only | True expected turnover per team per wave |",
            "| `answer_key/events.csv` | Graders only | Teams hit by the planted events E1, E2, E3 |",
        ]
        lines = [x for x in lines if x != ""]
    lines += ["", "Never give the agent anything in `answer_key/`.", "",
              "## Rebuild", "", "```bash",
              f"python generate_synthetic_data.py --employees {meta['employees_per_wave']} --seed {meta['seed']} "
              f"--variant {v} --out {os.path.basename(os.path.normpath(out_dir))}",
              "```", "", "The same command always produces identical files."]
    with open(f"{out_dir}/README.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Answer key sections for multi-year data (long_* only)
# ---------------------------------------------------------------------------
def long_answer_key(full_all, survey_all, risk, events, manager_changes):
    by_wave = full_all.groupby("wave")
    vis = survey_all[survey_all["responded"] == 1]
    trends = []
    for w, g in by_wave:
        trends.append({
            "wave": int(w),
            "employees": int(len(g)),
            "true_mean_score_all_employees": round(float(g["true_index"].mean()), 3),
            "visible_mean_score_respondents": round(float(vis.loc[vis["wave"] == w, ITEMS].astype(float).mean(axis=1).mean()), 3),
            "turnover_rate": round(float(g["left_voluntarily_12m"].mean()), 4),
            "true_turnover_rate": round(float(g["turnover_prob_true"].mean()), 4),
            "response_rate": round(float(g["responded"].mean()), 4),
            "outcomes_hidden": bool(HIDE_LAST and w == N_WAVES),
        })
    ev = pd.DataFrame(events)
    rk = risk.set_index(["wave", "team_id"])
    true_events = []
    for code, spec in EVENTS.items():
        teams_hit = ev.loc[ev["event"] == code, "team_id"].tolist() if len(ev) else []
        w = spec["wave"]
        if not teams_hit or w > N_WAVES:
            continue
        hit = rk.index.get_level_values("team_id").isin(teams_hit)
        def delta(col, mask):
            cur = rk.loc[w][col][mask[rk.index.get_level_values("wave") == w]]
            prev = rk.loc[w - 1][col].reindex(cur.index)
            return float((cur - prev).mean())
        wave_mask = rk.index.get_level_values("wave")
        hit_w = hit[wave_mask == w]
        others_w = ~hit_w
        visible_w = vis[vis["wave"] == w].assign(idx=lambda d: d[ITEMS].astype(float).mean(axis=1))
        visible_p = vis[vis["wave"] == w - 1].assign(idx=lambda d: d[ITEMS].astype(float).mean(axis=1))
        vt = visible_w.groupby("team_id")["idx"].mean() - visible_p.groupby("team_id")["idx"].mean()
        true_events.append({
            "id": code, "name": spec["name"], "wave": w, "teams": len(teams_hit),
            "visible_score_change_event_teams": round(float(vt.reindex(teams_hit).mean()), 3),
            "visible_score_change_other_teams": round(float(vt.drop(teams_hit, errors="ignore").mean()), 3),
            "true_turnover_change_event_teams": round(delta("true_turnover_rate", hit), 4),
            "true_turnover_change_other_teams": round(delta("true_turnover_rate", ~hit), 4),
            "expected": {
                "E1": "Scores rise; true turnover falls only if items affect turnover (main), not in the null variants.",
                "E2": "Scores fall; true turnover rises only if items affect turnover (main).",
                "E3": "Scores rise but true turnover does not change: a false turnaround. A good change gate does "
                      "not count these teams as improved, and the health check flags the jump.",
            }[code],
        })
    last = risk[risk["wave"] == N_WAVES]
    prev = risk[risk["wave"] == N_WAVES - 1].set_index("team_id")
    big = last[last["team_size"] >= 5]
    prev_rate = big["team_id"].map(prev["realized_turnover_rate"])
    realized_above = (big["realized_turnover_rate"] > big["realized_turnover_rate"].mean()).astype(int)
    summary = {
        "wave": N_WAVES,
        "teams_5_plus": int(len(big)),
        "company_true_rate": round(float(last["company_true_rate"].iloc[0]), 4),
        "auc_true_rate_vs_realized_above_company": round(auc(big["true_turnover_rate"], realized_above), 3),
        "auc_previous_rate_vs_realized_above_company": round(auc(prev_rate.fillna(prev_rate.mean()), realized_above), 3),
        "note": ("auc_true_rate is the best any model could do (it knows each team's true risk); the gap to 1.0 is "
                 "pure chance in who leaves. auc_previous_rate is the 'same as last year' baseline. A team model "
                 "should land between the two."),
    }
    out = {
        "true_trends": trends,
        "true_events": true_events,
        "manager_changes": {
            "per_wave": pd.DataFrame(manager_changes).groupby("wave").size().astype(int).to_dict() if manager_changes else {},
            "natural_change_rate": MGR_NATURAL_CHANGE,
            "manager_to_engagement": MGR_TO_ENGAGEMENT,
            "note": "manager_id in survey_responses.csv changes when a team gets a new manager. Hidden quality is in answer_key/manager_changes.csv.",
        },
        "team_true_risk_summary": summary,
        "x7_prediction_note": (
            "With X7 on, people who are about to leave answer less often, and disengaged leavers least. So a team "
            "that will lose many people looks better on the survey than it is, and the respondents' team score is "
            "weakly POSITIVELY related to team turnover within a department. A team model using Q12 means can then "
            "beat a department-only model by a few AUC points even when no item has any effect (long_null). "
            "long_purenull removes X7; there the gap should vanish. A correct agent runs a within-department "
            "permutation test and shows a survey-response warning instead of claiming Q12 drives team risk."
            if ARGS.variant != "long_purenull" else
            "X7 is off in this variant: the survey response does not depend on leaving. No Q12-based team model "
            "should beat the department-and-size model."),
    }
    if HIDE_LAST:
        out["hidden_outcomes"] = {
            "wave": N_WAVES,
            "file": "answer_key/hidden_outcomes.csv",
            "grading": (f"Wave {N_WAVES} outcomes are blank in agent_data, as if the year has not finished. The app "
                        f"trains on waves 1-{N_WAVES - 1} and predicts wave {N_WAVES} team risk; grade the ranking with "
                        "team_true_risk.csv (true_above_company) and hidden_outcomes.csv (realized)."),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir):
    os.makedirs(f"{out_dir}/agent_data", exist_ok=True)
    os.makedirs(f"{out_dir}/answer_key", exist_ok=True)

    teams = build_org()
    teams_wave1 = teams.copy()        # answer key: wave-1 managers (changes are in manager_changes.csv)
    emp = build_employees(teams)
    surveys, indiv, team_out, full = [], [], [], []
    manager_changes, events, used = [], [], set()
    for wave in range(1, N_WAVES + 1):
        e3_teams = []
        if LONG and wave >= 2:
            teams, ch, ev, e3_teams = change_managers(teams, wave, used)
            manager_changes += ch
            events += ev
        if LONG:
            survey, dept_shift, reported = simulate_survey(emp, teams, wave, e3_teams)
            outcomes, p_leave = simulate_individual_outcomes(survey, teams, dept_shift, wave)
        else:
            survey, dept_shift = simulate_survey(emp, teams, wave)
            outcomes = simulate_individual_outcomes(survey, teams, dept_shift, wave)
            reported = None
        visible, resp_prob = apply_nonresponse(survey, outcomes)
        if reported is not None:          # E3: the file shows the inflated answers
            answered = visible["responded"].to_numpy() == 1
            shown = reported.astype("Int64")
            shown.loc[~answered, :] = pd.NA
            visible[ITEMS] = shown
        t_out = team_table(survey, teams)
        for df in (visible, outcomes, t_out):
            df.insert(0, "wave", wave)
        extra = {}
        if LONG:
            extra = {"turnover_prob_true": np.round(p_leave, 5),
                     "true_index": survey[ITEMS].mean(axis=1).round(4).to_numpy(),
                     "manager_quality": survey["team_id"].map(teams.set_index("team_id")["manager_quality"]).to_numpy()}
        full.append(survey.assign(wave=wave, responded=visible["responded"].to_numpy(),
                                  response_prob=np.round(resp_prob, 4), **extra)
                    .merge(outcomes, on=["wave", "employee_id"]))
        surveys.append(visible)
        indiv.append(outcomes)
        team_out.append(t_out)
        if wave < N_WAVES:
            emp = next_wave(emp, outcomes.drop(columns="wave"), teams)

    hidden = ["engagement_latent", "prior_perf_latent"]
    survey_all = pd.concat(surveys, ignore_index=True).drop(columns=hidden)
    cols = ["wave", "employee_id", "team_id"] + (["manager_id"] if LONG else []) + [
            "department", "tenure_years", "role_level", "pay_band",
            "shift_type", "survey_date"] + ITEMS + ["responded"]
    survey_all = survey_all[cols]
    indiv_all = pd.concat(indiv, ignore_index=True)
    team_all = pd.concat(team_out, ignore_index=True)
    full_all = pd.concat(full, ignore_index=True)

    hidden_outcomes = None
    if HIDE_LAST:
        # The newest year's outcomes are not known yet: blank them for the agent, keep them for graders.
        last = indiv_all["wave"] == N_WAVES
        hide = ["left_voluntarily_12m", "turnover_window_end", "unplanned_absence_days_6m",
                "absence_window_end", "performance_rating", "performance_review_date"]
        hidden_outcomes = indiv_all.loc[last].copy()
        indiv_all[hide] = indiv_all[hide].astype(object)
        indiv_all.loc[last, hide] = None

    error_log = None
    if ARGS.variant.startswith("dirty"):
        survey_all, indiv_all, team_all, error_log = inject_errors(survey_all, indiv_all, team_all, ARGS.variant)

    fmt = lambda df: df.to_csv(index=False, date_format="%Y-%m-%d")
    for name, df in (("survey_responses", survey_all), ("individual_outcomes", indiv_all),
                     ("team_outcomes", team_all)):
        with open(f"{out_dir}/agent_data/{name}.csv", "w") as f:
            f.write(fmt(df))

    # auto: small 2-wave sets only (a 5-wave file would be too large to share)
    write_full = ARGS.full_data == "yes" or (ARGS.full_data == "auto" and N_EMPLOYEES <= 50000 and not LONG)
    if write_full:
        with open(f"{out_dir}/answer_key/full_data_with_latents.csv", "w") as f:
            f.write(fmt(full_all))
    teams_wave1.to_csv(f"{out_dir}/answer_key/teams_with_manager_quality.csv", index=False)
    if error_log is not None:
        error_log.to_csv(f"{out_dir}/answer_key/error_log.csv", index=False)
    risk = None
    if LONG:
        if hidden_outcomes is not None:
            with open(f"{out_dir}/answer_key/hidden_outcomes.csv", "w") as f:
                f.write(fmt(hidden_outcomes))
        pd.DataFrame(manager_changes).to_csv(f"{out_dir}/answer_key/manager_changes.csv", index=False)
        pd.DataFrame(events).to_csv(f"{out_dir}/answer_key/events.csv", index=False)
        risk = team_risk_table(full_all)
        risk.to_csv(f"{out_dir}/answer_key/team_true_risk.csv", index=False)

    w1 = full_all[full_all["wave"] == 1]
    realized = realized_values(w1)
    tw1 = team_all[team_all["wave"] == 1]
    meta = {
        "variant": ARGS.variant, "seed": ARGS.seed, "employees_per_wave": N_EMPLOYEES,
        "survey_rows": int(len(survey_all)), "teams": int(N_TEAMS),
        "small_teams": int((tw1["team_size"] < 5).sum()),
        "small_team_share": float((tw1["team_size"] < 5).mean()),
        "response_rate_w1": float(w1["responded"].mean()),
        "turnover_w1": float(w1["left_voluntarily_12m"].mean()),
        "full_data": write_full,
    }
    rr = w1.groupby("left_voluntarily_12m")["responded"].mean()
    is_null = BASE_VARIANT == "null"
    gt = {
        "generator_version": GENERATOR_VERSION,
        "variant": ARGS.variant,
        "base_variant": BASE_VARIANT,
        "waves": N_WAVES,
        "seed": ARGS.seed,
        "dataset": {k: meta[k] for k in ["employees_per_wave", "survey_rows", "teams", "small_teams",
                                         "small_team_share", "response_rate_w1", "turnover_w1"]},
        "how_to_grade": ("Grade effect sizes against realized_values.respondents_only: that is what the agent can "
                         "estimate from agent_data. realized_values.all_employees is the population truth used to "
                         "show the X7 understatement. Planted parameters are the generator settings."),
        "parameters": TRUTH,
        "true_effects": [] if is_null else [
            {"id": "T1", "item": "Q04", "outcome": "turnover", "measure": "OR per point", "direction": "negative",
             "planted": TRUTH["T1_Q04_turnover_OR"],
             "note": "Twice as strong (log-odds) for tenure under 2 years (X5); the pooled OR is therefore below 0.70."},
            {"id": "T2", "item": "Q05", "outcome": "turnover", "measure": "OR per point", "direction": "negative",
             "planted": TRUTH["T2_Q05_turnover_OR"]},
            {"id": "T3", "item": "Q12", "outcome": "turnover", "measure": "OR per point", "direction": "negative",
             "planted": TRUTH["T3_Q12_turnover_OR"]},
            {"id": "T4", "item": "Q01", "outcome": "performance", "measure": "standardized beta",
             "direction": "positive", "planted": TRUTH["T4_Q01_performance_beta"]},
            {"id": "T5", "item": "Q05", "outcome": "absence", "measure": "IRR per point", "direction": "negative",
             "planted": TRUTH["T5_Q05_absence_IRR"]},
        ],
        "true_nulls": [
            {"id": "N1", "item": "Q06", "outcomes": ["absence"],
             "truth": "Q06 is associated with absence, but once the other items are included it has no independent effect."},
            {"id": "N2", "item": "Q10", "outcomes": ["turnover", "absence", "performance"],
             "truth": "Q10 is associated with the outcomes, but has no independent effect on any of them."},
            {"id": "N3", "item": "Q08", "outcomes": ["turnover", "absence", "performance"],
             "truth": "Q08 is associated with the outcomes, but has no independent effect on any of them."},
        ],
        "items_without_planted_effects": "Q02, Q03, Q06, Q07, Q08, Q09, Q10, Q11 have no independent effect on any "
                                         "outcome (Q07 has a small within-department effect on performance, X6).",
        "traps": [
            {"id": "X1", "name": "Confounding", "built": "Sales scores low on Q02 and has high turnover; Q02 has no effect.",
             "correct": "Control for department; the Q02 link disappears."},
            {"id": "X2", "name": "Reverse causality", "built": "Prior performance raises Q11.",
             "correct": "Control for prior rating; do not claim Q11 drives performance."},
            {"id": "X3", "name": "Hidden confounder", "built": "Hidden manager quality raises Q07 and lowers absence.",
             "correct": "Flag a likely manager effect; recommend team-level review."},
            {"id": "X4", "name": "Threshold", "built": "Absence rises only when Q01 is below 3.",
             "correct": "Test the threshold, not only a straight line."},
            {"id": "X5", "name": "Subgroup", "built": "Q04 matters twice as much for tenure under 2 years.",
             "correct": "Report the subgroup difference."},
            {"id": "X6", "name": "Simpson's paradox",
             "built": "Departments rate performance with different leniency, opposite to their mean Q07.",
             "correct": "Report the within-department (positive) result."},
            {"id": "X7", "name": "Nonresponse",
             "built": "Leavers answer less often, and among leavers the disengaged answer least.",
             "correct": "Warn that turnover effects are likely understated."},
            {"id": "X8", "name": "Small groups", "built": "About 5% of teams have fewer than 5 people.",
             "correct": "Suppress results for those teams."},
        ],
        "x7_design": {
            "description": ("Response depends on both later leaving and hidden engagement. Disengaged leavers answer "
                            "least. Group response rates are calibrated to the targets, so respondents look more "
                            "engaged than the workforce and engagement-turnover links look weaker than they are. "
                            "A side effect: among respondents, items with no real effect can show small positive "
                            "turnover odds ratios (about 1.02-1.07), because answering depends on both engagement "
                            "and leaving. These stay inside the no-effect band (0.91-1.10)."),
            "response_rate_leavers": round(float(rr.get(1, np.nan)), 3),
            "response_rate_stayers": round(float(rr.get(0, np.nan)), 3),
            "mean_score_turnover_OR_all_employees": realized["all_employees"]["X7_mean_score_turnover_OR"],
            "mean_score_turnover_OR_respondents": realized["respondents_only"]["X7_mean_score_turnover_OR"],
            "grading": "The agent cannot compute the all-employee value. It passes if it warns that the effect it "
                       "reports is likely understated.",
        },
        "realized_values": realized,
    }
    if is_null:
        gt["null_design"] = ("No engagement item affects any outcome. X1, X2 and X6 still create misleading raw "
                             "associations through department and prior performance; a correct agent controls for "
                             "them and reports no item as a driver.")
    if ARGS.variant == "long_purenull":
        gt["null_design"] += (" X7 is also off: leavers and stayers answer equally often and answering does not "
                              "depend on engagement, so nothing in the survey predicts turnover.")
    if LONG:
        gt.update(long_answer_key(full_all, survey_all, risk, events, manager_changes))
    with open(f"{out_dir}/answer_key/ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2, default=float)
    write_readme(out_dir, meta)
    print(f"[{ARGS.variant}] {len(survey_all):,} survey rows ({N_EMPLOYEES:,} employees x {N_WAVES} waves, "
          f"{N_TEAMS:,} teams) -> {out_dir}/")


if __name__ == "__main__":
    main(ARGS.out)
