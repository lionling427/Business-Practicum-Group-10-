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

GENERATOR_VERSION = "2.1"

# ---------------------------------------------------------------------------
# STEP 0. Configuration
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--employees", type=int, default=5000, help="employees per wave (2 waves are generated)")
ap.add_argument("--out", default="output", help="output folder")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--variant", default="main",
                choices=["main", "null", "weak", "seed", "dirty_A", "dirty_B", "dirty_C"])
ap.add_argument("--full-data", choices=["auto", "yes", "no"], default="auto",
                help="write answer_key/full_data_with_latents.csv (auto: only up to 50,000 employees)")
ARGS = ap.parse_args()

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

if ARGS.variant == "weak":
    for k in ["T1_Q04_turnover_OR", "T1_X5_tenure_lt2_OR", "T2_Q05_turnover_OR",
              "T3_Q12_turnover_OR", "T5_Q05_absence_IRR"]:
        TRUTH[k] = round(float(np.sqrt(TRUTH[k])), 3)
    TRUTH["T4_Q01_performance_beta"] = 0.10
    # v2.1 fix: half of T5 on the log scale (0.922) is realized near 0.94 among respondents,
    # inside the 0.91-1.10 "no meaningful effect" band, so a correct agent would not report it.
    # Every true effect must be realized outside the band; 0.87 keeps T5 weaker than main (0.85).
    TRUTH["T5_Q05_absence_IRR"] = 0.87
if ARGS.variant == "null":
    for k in ["T1_Q04_turnover_OR", "T1_X5_tenure_lt2_OR", "T2_Q05_turnover_OR",
              "T3_Q12_turnover_OR", "T5_Q05_absence_IRR"]:
        TRUTH[k] = 1.0
    TRUTH["T4_Q01_performance_beta"] = 0.0
    TRUTH["X3_manager_to_absence"] = 0.0      # no hidden path from Q07 to absence
    TRUTH["X4_Q01_low_absence_logrr"] = 0.0   # no threshold effect
    TRUTH["X6_Q07_within_dept_beta"] = 0.0    # no within-department effect

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
    })


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
def simulate_survey(emp, teams, wave):
    t = teams.set_index("team_id")
    mgr = emp["team_id"].map(t["manager_quality"]).to_numpy()
    dept_idx = emp["department"].map({d: i for i, d in enumerate(DEPARTMENTS)}).to_numpy()
    n = len(emp)
    dept_q07_shift = np.linspace(-1, 1, len(DEPARTMENTS)) * TRUTH["X6_dept_Q07_spread"]

    eng = emp["engagement_latent"].to_numpy()
    cont = {q: ITEM_LOADING * eng + ITEM_NOISE * rng.normal(0, 1, n) + ITEM_OFFSET[q] for q in ITEMS}
    cont["Q02"] += np.where(emp["department"] == "Sales", TRUTH["X1_sales_Q02_shift"], 0)   # X1
    cont["Q11"] += TRUTH["X2_prior_perf_to_Q11"] * emp["prior_perf_latent"].to_numpy()        # X2
    cont["Q07"] += TRUTH["X3_manager_to_Q07"] * mgr                                           # X3
    cont["Q07"] += dept_q07_shift[dept_idx]                                                   # X6

    pooled = np.concatenate([cont[q] for q in ITEMS])
    cuts = np.quantile(pooled, LIKERT_CUTS)
    survey = emp.copy()
    for q in ITEMS:
        survey[q] = (np.searchsorted(cuts, cont[q]) + 1).astype(int)
    start = np.datetime64(WAVE_SURVEY_START[wave])
    survey["survey_date"] = start + rng.integers(0, 21, n).astype("timedelta64[D]")
    return survey, dept_q07_shift


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
    turnover = rng.binomial(1, 1 / (1 + np.exp(-logit)))

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
    hires = build_employees(repl, start_id=N_EMPLOYEES)
    hires["tenure_years"] = np.round(rng.uniform(0, 1, len(hires)), 1)
    return pd.concat([stayers, hires], ignore_index=True)


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
        f"| Survey waves | 2 |",
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
        "| `agent_data/survey_responses.csv` | Agent | One row per employee per wave: Q01-Q12 (blank for non-respondents), department, tenure, role, pay band, shift, survey date |",
        "| `agent_data/individual_outcomes.csv` | Agent | Turnover (12 months), unplanned absence days (6 months), prior and current performance rating, and the date each outcome window ends |",
        "| `agent_data/team_outcomes.csv` | Agent | Team id, department and team size per wave |",
        "| `answer_key/ground_truth.json` | Graders only | Planted settings, realized values, true nulls, traps, seed |",
        "| `answer_key/teams_with_manager_quality.csv` | Graders only | Hidden manager quality per team (trap X3) |",
    ]
    if meta.get("full_data"):
        lines.append("| `answer_key/full_data_with_latents.csv` | Graders only | Every employee's true answers and hidden engagement |")
    if v.startswith("dirty"):
        lines.append("| `answer_key/error_log.csv` | Graders only | Every injected error and the health-check result it should produce |")
    lines += ["", "Never give the agent anything in `answer_key/`.", "",
              "## Rebuild", "", "```bash",
              f"python generate_synthetic_data.py --employees {meta['employees_per_wave']} --seed {meta['seed']} "
              f"--variant {v} --out {os.path.basename(os.path.normpath(out_dir))}",
              "```", "", "The same command always produces identical files."]
    with open(f"{out_dir}/README.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir):
    os.makedirs(f"{out_dir}/agent_data", exist_ok=True)
    os.makedirs(f"{out_dir}/answer_key", exist_ok=True)

    teams = build_org()
    emp = build_employees(teams)
    surveys, indiv, team_out, full = [], [], [], []
    for wave in (1, 2):
        survey, dept_shift = simulate_survey(emp, teams, wave)
        outcomes = simulate_individual_outcomes(survey, teams, dept_shift, wave)
        visible, resp_prob = apply_nonresponse(survey, outcomes)
        t_out = team_table(survey, teams)
        for df in (visible, outcomes, t_out):
            df.insert(0, "wave", wave)
        full.append(survey.assign(wave=wave, responded=visible["responded"].to_numpy(),
                                  response_prob=np.round(resp_prob, 4))
                    .merge(outcomes, on=["wave", "employee_id"]))
        surveys.append(visible)
        indiv.append(outcomes)
        team_out.append(t_out)
        if wave == 1:
            emp = next_wave(emp, outcomes.drop(columns="wave"), teams)

    hidden = ["engagement_latent", "prior_perf_latent"]
    survey_all = pd.concat(surveys, ignore_index=True).drop(columns=hidden)
    cols = ["wave", "employee_id", "team_id", "department", "tenure_years", "role_level", "pay_band",
            "shift_type", "survey_date"] + ITEMS + ["responded"]
    survey_all = survey_all[cols]
    indiv_all = pd.concat(indiv, ignore_index=True)
    team_all = pd.concat(team_out, ignore_index=True)
    full_all = pd.concat(full, ignore_index=True)

    error_log = None
    if ARGS.variant.startswith("dirty"):
        survey_all, indiv_all, team_all, error_log = inject_errors(survey_all, indiv_all, team_all, ARGS.variant)

    fmt = lambda df: df.to_csv(index=False, date_format="%Y-%m-%d")
    for name, df in (("survey_responses", survey_all), ("individual_outcomes", indiv_all),
                     ("team_outcomes", team_all)):
        with open(f"{out_dir}/agent_data/{name}.csv", "w") as f:
            f.write(fmt(df))

    write_full = ARGS.full_data == "yes" or (ARGS.full_data == "auto" and N_EMPLOYEES <= 50000)
    if write_full:
        with open(f"{out_dir}/answer_key/full_data_with_latents.csv", "w") as f:
            f.write(fmt(full_all))
    teams.to_csv(f"{out_dir}/answer_key/teams_with_manager_quality.csv", index=False)
    if error_log is not None:
        error_log.to_csv(f"{out_dir}/answer_key/error_log.csv", index=False)

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
    is_null = ARGS.variant == "null"
    gt = {
        "generator_version": GENERATOR_VERSION,
        "variant": ARGS.variant,
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
    with open(f"{out_dir}/answer_key/ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)
    write_readme(out_dir, meta)
    print(f"[{ARGS.variant}] {len(survey_all):,} survey rows ({N_EMPLOYEES:,} employees x 2 waves, "
          f"{N_TEAMS:,} teams) -> {out_dir}/")


if __name__ == "__main__":
    main(ARGS.out)
