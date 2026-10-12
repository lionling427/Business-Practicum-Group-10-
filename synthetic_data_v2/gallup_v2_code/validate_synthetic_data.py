"""
Validator (v2): do the planted truths really show up in the generated data?

Run after generate_synthetic_data.py, on any dataset folder:
  python validate_synthetic_data.py gallup_v2_5K
  python validate_synthetic_data.py gallup_v2_2M
  python validate_synthetic_data.py testsets/null

What it checks depends on the variant recorded in answer_key/ground_truth.json:
  main / seed  every true effect, true null and trap behaves as designed
  weak         the weaker true effects are still detected; nulls stay null
  null         no engagement item is a significant driver after FDR correction
  dirty_*      every error in answer_key/error_log.csv is caught by health_check.py

Expected values are read from ground_truth.json, never typed into this file.
Writes <folder>/reports/validation_report.csv. Result is PASS or CHECK (needs a look).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find stats_tools.py next to this file
import stats_tools as st

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "output"
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = st.ITEMS
MEANINGFUL_RATIO = 0.10      # OR or IRR within 0.91-1.10 counts as "no meaningful effect"
MEANINGFUL_BETA = 0.05       # standardized beta within +/-0.05 counts as "no meaningful effect"

gt = json.load(open(f"{FOLDER}/answer_key/ground_truth.json"))
VARIANT = gt["variant"]
REAL = gt["realized_values"]["respondents_only"]
results = []


def check(area, name, ok, expected, observed):
    results.append({"area": area, "check": name, "result": "PASS" if ok else "CHECK",
                    "expected": expected, "observed": observed})


def run_health_check():
    path = f"{FOLDER}/reports/health_report.json"
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.join(HERE, "health_check.py"), FOLDER],
                       check=True, stdout=subprocess.DEVNULL)
    return json.load(open(path))


def save_and_print():
    os.makedirs(f"{FOLDER}/reports", exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(f"{FOLDER}/reports/validation_report.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200)
    print(out.to_string(index=False))
    print(f"\n[{VARIANT}] {int((out.result == 'PASS').sum())} of {len(out)} checks PASS")


# ---------------------------------------------------------------------------
# Dirty variants: compare the error log with the health report (fix #16)
# ---------------------------------------------------------------------------
if VARIANT.startswith("dirty"):
    report = run_health_check()
    by_code = {c["code"]: c for c in report["checks"]}
    log = pd.read_csv(f"{FOLDER}/answer_key/error_log.csv")
    for code, grp in log.groupby("expected_check", sort=False):
        c = by_code.get(code)
        must = grp[grp["expected_status"] != "PASS"]
        exp_status = "FAIL" if (must["expected_status"] == "FAIL").any() else \
            "WARN" if len(must) else "PASS"
        if must.empty:
            exp_count = 0
        elif "(column)" in set(must["key"]):
            exp_count = None if code.startswith("LK_DATE") else len(must)
        else:
            exp_count = len(must)
        got_count = c["metrics"].get("violations") if c else None
        ok = c is not None and c["status"] == exp_status and (exp_count is None or got_count == exp_count)
        if ok and code.startswith("LK_DATE") and exp_status == "FAIL":
            # a date error must block only its own outcome, not the whole analysis
            ok = c["actions"] == [f"block:{code.split('_')[-1]}"]
        check("Health check", f"{code} catches the planted errors", ok,
              f"{exp_status}, {exp_count if exp_count is not None else 'column-level'} violation(s) "
              f"[{', '.join(grp['error_id'][:3])}{' + %d more' % (len(grp) - 3) if len(grp) > 3 else ''}]",
              f"{c['status'] if c else 'missing'}, {got_count} violation(s)")
    planted = set(log["expected_check"])
    others = [c for c in report["checks"] if c["code"] not in planted and c["status"] == "FAIL"]
    check("Health check", "No FAIL on checks without planted errors", not others, "none",
          ", ".join(c["code"] for c in others) or "none")
    expect_block = set()
    for _, r in log[log["expected_status"] == "FAIL"].iterrows():
        expect_block.add("all" if "all" in r["expected_effect"] else r["expected_effect"].split()[1])
    expect_block = ["all"] if "all" in expect_block else sorted(expect_block)
    check("Health check", "Blocks exactly the affected outcomes", report["summary"]["blocked_outcomes"] == expect_block,
          str(expect_block), str(report["summary"]["blocked_outcomes"]))
    save_and_print()
    sys.exit(0)

# ---------------------------------------------------------------------------
# Data used by the agent (respondents, wave 1) plus the hidden team file for X3
# ---------------------------------------------------------------------------
D = f"{FOLDER}/agent_data"
survey = pd.read_csv(f"{D}/survey_responses.csv")
indiv = pd.read_csv(f"{D}/individual_outcomes.csv")
team = pd.read_csv(f"{D}/team_outcomes.csv")
df = survey.merge(indiv, on=["wave", "employee_id"])
w1all = df[df["wave"] == 1]
w1 = w1all[w1all["responded"] == 1].copy()
mgr = pd.read_csv(f"{FOLDER}/answer_key/teams_with_manager_quality.csv").set_index("team_id")["manager_quality"]


def fmt_ratio(f, name):
    o, lo, hi = f.ratio(name)
    return f"{o:.2f} [{lo:.2f}, {hi:.2f}]"


def no_effect_ratio(f, name):
    o, lo, hi = f.ratio(name)
    return f.pvalues[name] > 0.05 or abs(np.log(o)) < np.log(1 + MEANINGFUL_RATIO)


def no_effect_beta(f, name):
    return f.pvalues[name] > 0.05 or abs(f.params[name]) < MEANINGFUL_BETA


# --- Structure ---------------------------------------------------------------
corr = w1[ITEMS].corr().to_numpy()
off = corr[np.triu_indices(12, 1)].mean()
check("Structure", "Items correlate like real Q12 data", 0.35 < off < 0.75, "mean inter-item r 0.35-0.75", f"{off:.2f}")
top2 = (w1[ITEMS] >= 4).mean()
spread = top2.max() - top2.min()
check("Structure", "Items have different score distributions (fix #5)", spread >= 0.10,
      "agree share differs by 10+ points across items",
      f"{top2.idxmin()} {top2.min():.0%} to {top2.idxmax()} {top2.max():.0%}")
tw = team[team["wave"] == 1]
share = (tw["team_size"] < 5).mean()
check("Structure", "X8 about 5% of teams under 5 people", abs(share - 0.05) < 0.01, "4%-6%",
      f"{share:.1%} ({(tw['team_size'] < 5).sum():,} of {len(tw):,})")
dates_ok = all((pd.to_datetime(df[c]) > pd.to_datetime(df["survey_date"])).all()
               for c in ["turnover_window_end", "absence_window_end", "performance_review_date"])
check("Structure", "Every outcome is dated after the survey", dates_ok, "all rows", "all rows" if dates_ok else "violations")
hc = run_health_check()
check("Structure", "Health check raises no FAIL on clean data", hc["summary"]["fail"] == 0, "0 FAIL",
      f"{hc['summary']['fail']} FAIL, {hc['summary']['warn']} WARN")

# --- Fit the three standard models --------------------------------------------
tm = st.turnover_model(w1)
am = st.absence_model(w1)
pm = st.performance_model(w1)

if VARIANT == "null":
    # Q11 -> performance is excluded here: it is the X2 trap (prior rating is measured with
    # noise, so a small residual remains) and is checked under Traps.
    tests = [(tm, q, "turnover") for q in ITEMS] + [(am, q, "absence") for q in ITEMS] + \
            [(pm, q, "performance") for q in ITEMS if q != "Q11"]
    disc = st.bh_fdr([f.pvalues[q] for f, q, _ in tests])
    meaningful = [not (no_effect_beta(f, q) if o == "performance" else no_effect_ratio(f, q))
                  for f, q, o in tests]
    found = [f"{q}->{o}" for (f, q, o), d, m in zip(tests, disc, meaningful) if d and m]
    check("Null design", "No item is a meaningful driver after FDR correction (35 tests)", not found,
          "0 items both FDR-significant and outside the no-effect band", ", ".join(found) or "0")
    small = [f"{q}->{o}" for (f, q, o), d, m in zip(tests, disc, meaningful) if d and not m]
    check("Null design", "Remaining significant results are trivially small", True,
          "any FDR-significant result stays inside the no-effect band",
          (", ".join(small) + " (within band; nonresponse selection, X7)") if small else "none")
else:
    for tid, item, model, kind in [("T1", "Q04", tm, "turnover"), ("T2", "Q05", tm, "turnover"),
                                   ("T3", "Q12", tm, "turnover"), ("T5", "Q05", am, "absence")]:
        o, lo, hi = model.ratio(item)
        key = next(k for k in REAL if k.startswith(tid + "_"))
        check("True effects", f"{tid} {item} -> {kind}", hi < 1, f"ratio < 1 (realized {REAL[key]:.2f})",
              fmt_ratio(model, item))
        # v2.1: the answer key must not list a "true effect" that a correct agent would call trivial
        check("True effects", f"{tid} realized value outside the no-effect band",
              abs(np.log(REAL[key])) >= np.log(1 + MEANINGFUL_RATIO), "realized ratio < 0.91 or > 1.10",
              f"{REAL[key]:.3f}")
    b = pm.params["Q01"]
    check("True effects", "T4 Q01 -> performance", b > 0 and pm.pvalues["Q01"] < 0.05,
          f"beta > 0 (realized {REAL['T4_Q01_performance_beta']:.2f})", f"beta {b:.2f} (p={pm.pvalues['Q01']:.3f})")
    check("True effects", "T4 realized value outside the no-effect band",
          abs(REAL["T4_Q01_performance_beta"]) >= MEANINGFUL_BETA, "|realized beta| >= 0.05",
          f"{REAL['T4_Q01_performance_beta']:.3f}")

# --- True nulls: associated alone, no independent effect in the full model -----
raw = lambda f, x: f.pvalues[x] < 0.05
r08 = st.logit(w1["left_voluntarily_12m"], st.design(w1, ["Q08"]))
r06 = st.negbin(w1["unplanned_absence_days_6m"], st.design(w1, ["Q06"]))
if VARIANT != "null":
    check("True nulls", "N1 Q06 looks related to absence on its own", raw(r06, "Q06"), "p < 0.05 alone",
          fmt_ratio(r06, "Q06"))
check("True nulls", "N1 Q06 -> absence: no independent effect", no_effect_ratio(am, "Q06"),
      "p > 0.05 or IRR 0.91-1.10", f"{fmt_ratio(am, 'Q06')} p={am.pvalues['Q06']:.2f}")
if VARIANT != "null":
    check("True nulls", "N3 Q08 looks related to turnover on its own", raw(r08, "Q08"), "p < 0.05 alone",
          fmt_ratio(r08, "Q08"))
for nid, item in [("N2", "Q10"), ("N3", "Q08")]:
    check("True nulls", f"{nid} {item} -> turnover: no independent effect", no_effect_ratio(tm, item),
          "p > 0.05 or OR 0.91-1.10", f"{fmt_ratio(tm, item)} p={tm.pvalues[item]:.2f}")
    check("True nulls", f"{nid} {item} -> absence: no independent effect", no_effect_ratio(am, item),
          "p > 0.05 or IRR 0.91-1.10", f"{fmt_ratio(am, item)} p={am.pvalues[item]:.2f}")
    check("True nulls", f"{nid} {item} -> performance: no independent effect", no_effect_beta(pm, item),
          "p > 0.05 or |beta| < 0.05", f"beta {pm.params[item]:.3f} p={pm.pvalues[item]:.2f}")

# --- Traps ---------------------------------------------------------------------
r02 = st.logit(w1["left_voluntarily_12m"], st.design(w1, ["Q02"]))
check("Traps", "X1 Q02 looks related to turnover, until department is controlled",
      raw(r02, "Q02") and no_effect_ratio(tm, "Q02"), "raw p < 0.05; controlled no effect",
      f"raw {fmt_ratio(r02, 'Q02')} -> controlled {fmt_ratio(tm, 'Q02')}")

z = pd.DataFrame({c: st.zscore(w1[c]) for c in ["performance_rating", "Q11", "Q07"]}, index=w1.index)
z["department"] = w1["department"]
q11_raw = st.ols(z["performance_rating"], st.design(z, ["Q11"])).params["Q11"]
check("Traps", "X2 Q11 looks predictive until prior rating is controlled",
      q11_raw > 2 * abs(pm.params["Q11"]), "raw beta > 2x controlled",
      f"raw {q11_raw:.2f} -> controlled {pm.params['Q11']:.2f}")

if VARIANT != "null":
    o, lo, hi = am.ratio("Q07")
    check("Traps", "X3 step 1: Q07 looks related to absence (department controlled)", hi < 1, "IRR < 1",
          fmt_ratio(am, "Q07"))
    a = w1.assign(y=np.log1p(w1["unplanned_absence_days_6m"]), q01_low=(w1["Q01"] < 3).astype(float),
                  night=(w1["shift_type"] == "Night").astype(float))
    xs = ITEMS + ["q01_low", "tenure_years", "night"]
    dept_fit = st.ols(a["y"], st.design(a, xs, ["department"]), a["team_id"])
    wt = st.demean(a, ["y"] + xs, "team_id")
    team_fit = st.ols(wt["y"], st.design(wt, xs), a["team_id"])
    b_d, b_t = dept_fit.params["Q07"], team_fit.params["Q07"]
    check("Traps", "X3 step 2: Q07 effect shrinks within teams (same manager)", abs(b_t) < 0.5 * abs(b_d),
          "shrinks by more than half", f"{b_d:.3f} -> {b_t:.3f}")
    icc, p = st.icc_oneway(a["y"], a["team_id"])
    check("Traps", "X3 step 3: absence clusters by team", icc > 0.01 and p < 0.001, "ICC > 0.01, p < 0.001",
          f"ICC {icc:.3f} (p={p:.1e})")
    a["manager_quality"] = a["team_id"].map(mgr)
    m_fit = st.ols(a["y"], st.design(a, xs + ["manager_quality"], ["department"]), a["team_id"])
    check("Traps", "X3 step 4: controlling hidden manager quality removes Q07",
          m_fit.pvalues["Q07"] > 0.05 or abs(m_fit.params["Q07"]) < 0.25 * abs(b_d),
          "p > 0.05 or under a quarter of step 1", f"{m_fit.params['Q07']:.3f} (p={m_fit.pvalues['Q07']:.2f})")
    o, lo, hi = am.ratio("q01_low")
    check("Traps", "X4 absence jumps when Q01 is below 3", lo > 1,
          f"RR > 1 (realized {REAL['X4_Q01_below3_absence_RR']:.2f})", fmt_ratio(am, "q01_low"))
    w1["new_hire"] = (w1["tenure_years"] < 2).astype(float)
    ti = st.turnover_model(w1, extra_numeric=["new_hire"], interactions=[("Q04", "new_hire")])
    check("Traps", "X5 Q04 matters more for tenure under 2 years", ti.params["Q04:new_hire"] < 0
          and ti.pvalues["Q04:new_hire"] < 0.05, "interaction < 0, p < 0.05",
          f"log-OR {ti.params['Q04:new_hire']:.2f} (p={ti.pvalues['Q04:new_hire']:.3f})")

pooled = st.ols(z["performance_rating"], st.design(z, ["Q07"])).params["Q07"]
within = st.ols(z["performance_rating"], st.design(z, ["Q07"], ["department"])).params["Q07"]
if VARIANT == "null":
    check("Traps", "X6 Q07 negative pooled, no effect within departments", pooled < 0 and abs(within) < 0.05,
          "pooled < 0, within about 0", f"pooled {pooled:.2f}, within {within:.2f}")
else:
    check("Traps", "X6 Simpson's paradox: Q07 vs performance flips", pooled < 0 < within,
          "pooled < 0 < within", f"pooled {pooled:.2f}, within {within:.2f}")

rr = w1all.groupby("left_voluntarily_12m")["responded"].mean()
check("Traps", "X7 leavers answer less often", rr[1] < rr[0] - 0.1, "gap > 10 points",
      f"leavers {rr[1]:.0%} vs stayers {rr[0]:.0%}")
if VARIANT != "null":
    w1["mean_score"] = w1[ITEMS].mean(axis=1)
    mo = st.logit(w1["left_voluntarily_12m"],
                  st.design(w1, ["mean_score", "tenure_years"], ["department", "role_level", "pay_band"]))
    seen, true = np.exp(mo.params["mean_score"]), gt["x7_design"]["mean_score_turnover_OR_all_employees"]
    check("Traps", "X7 respondents understate the engagement-turnover link", seen > true + 0.05,
          f"respondent OR above the all-employee OR {true:.2f}", f"respondent OR {seen:.2f}")
smallflag = next(c for c in hc["checks"] if c["code"] == "COV_SMALL_TEAMS")
check("Traps", "X8 health check asks to suppress small teams", "suppress_groups:5" in smallflag["actions"],
      "suppress_groups:5", ", ".join(smallflag["actions"]) or "none")

save_and_print()
