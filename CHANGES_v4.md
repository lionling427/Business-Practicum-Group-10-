# CHANGES v4: Oct 17 agent + v3 merge + multi-year data + prediction fixes

Base: Selvi's Oct 17 version (`gallup-engagement-agent/`). **All of her content is kept**
(model_spec, traps, financials ranking, Gallup resources, prompt rules 42–44, check_upload,
HealthCheckBlocked, every page). v4 adds three things on top:

1. The v3 fixes her version did not have yet (1, 2, 4, 5, 6; fix 3 and 7 were already in).
2. Multi-year synthetic data (5 waves) and the matching health check and validator.
3. The five fixes from *Prediction Review: Issues and Fixes*, in both the team model
   (`prediction.py`) and the monthly model (`survival.py`).

## 1. Files to copy into the GitHub repo

### Agent app (`gallup-engagement-agent/`)

| File | Status | What changed |
|---|---|---|
| `src/agent.py` | modified | v3 fix 1: `health_gate()` (all blocked → refuse; asked outcome blocked → refuse and list what is allowed; no outcome named → answer from allowed outcomes with an exclusion note). `outcome_is_allowed(analysis, None)` is no longer always True. `HealthCheckBlocked` from the pipeline becomes a polite refusal with no language-model call. v3 fix 6: `call_with_retry()` (429, 408/409/5xx, timeouts, empty answers; 3 retries, 2/4/8 s or Retry-After). `add_patterns_and_exclusions()`. The dev dashboard's health check now runs on the whole dev file. Prompt rules 45 (exclusions), 46 (threshold/subgroup patterns), 47 (predictions are warning signs; survey-response warning; hidden percentages; blocked predictions). |
| `src/tools/analytics.py` | modified | v3 fix 2: blocked outcomes are never fitted (`run_standard_models(..., outcomes)`), `collect_model_results` loops over fitted models, robustness diagnostics only for fitted outcomes, an analysis with every outcome blocked raises `HealthCheckBlocked`. New keys `status` (OK/PARTIAL), `allowed_outcomes`, `blocked_outcomes`, `special_findings`. v3 fixes 4–5: `threshold_diagnostics()` reads the pre-registered `q01_low` term from her absence spec (X4); `subgroup_diagnostics()` (12 item × new-hire terms, FDR) (X5); patterns attached to payloads as `related_patterns`. `analyze_dataframe` runs the health check before dropping non-respondents. |
| `src/tools/health_check.py` | modified | One file for the app and the data tools (checker 3.1). Her `check_upload` / `model_controls` API kept; v3 names added (`run_checks`, `check_dataframe`, `check_folder`, `write_report`). Multi-year checks (only with 3+ waves or `manager_id`): `OUT_PENDING`, `SV_STAYER_SHARE_wA_wB`, `NR_LEAVERS_BY_WAVE`, `RS_TEAM_JUMP`, `MG_MANAGER_CHANGE`. On every 2-wave dataset the 45 checks give the same results as before. |
| `src/tools/prediction_checks.py` | **new** | Shared prediction checks: within-department permutation test, change gate, binned calibration, health gate (`PredictionBlocked`), smoothing, survey-response (X7) warning. |
| `src/tools/prediction.py` | modified | Team risk: target = smoothed rate (k = 20) from **all** employees, 3+ points above the company rate; change features (item and index deltas, new manager); model B (department + size) and a 200-shuffle permutation test; Q12 "warning signs" listed only if the test passes; change gate; calibration (percentages hidden when off or pending); health gate; X7 warning; result cache. Employee model: health gate, calibration, X7 warning. Wording "Warning signs", not "Pushing risk up". All old keys kept. |
| `src/tools/survival.py` | modified | Monthly model: health gate, smoothed team-rate baseline, engagement attribution (100 shuffles), change gate on people whose engagement moved 0.5+ or got a new manager, binned calibration (old `calibration` ratio kept). |
| `src/tools/agent_context.py` | modified | Predictions use the health report; `team_risk_model_checks` (test result, Q12 signal, change gate, calibration, what "above normal" means, X7 warning); expected leavers hidden when not calibrated; blocked data → no predictions. |
| `src/tools/dashboard.py` | modified | Risk % only when calibrated. |
| `src/tools/insights.py` | modified | Risk outlook gives bands only when not calibrated; "warning sign" wording. |
| `src/tools/action_story.py` | modified | Passes the health report; "warning signs, not proven causes"; X7 warning. |
| `app_pages/teams.py` | modified | Passes the health report; shows the survey-response warning. |
| `app_pages/predictions.py` | modified | Passes the health report (group risk, and monthly risk for uploads); hides expected leavers / chances when not calibrated; shows calibration and X7 notes. |
| `app_pages/recommendations.py` | modified | Passes the health report to team risk. |
| `tests/test_v3_fixes.py` | **new** | 17 tests for v3 fixes 1–6, adapted to her API (`HealthCheckBlocked`). |
| `tests/test_prediction_checks.py` | **new** | 15 tests for the five prediction fixes. |

Nothing to delete or rename in the app.

### Data tools and data (`synthetic_data_v2/`)

| File | Status | What changed |
|---|---|---|
| `gallup_v2_code/generate_synthetic_data.py` | modified | v3.0: `--waves`, `--hide-last-outcome`, variants `long_main`, `long_null`, `long_purenull`, `long_seed`. 2-wave output byte-identical. |
| `gallup_v2_code/validate_synthetic_data.py` | modified | Multi-year checks. |
| `gallup_v2_code/health_check.py` | modified | Same file as the app's. |
| `gallup_v2_code/stats_tools.py` | modified | Same file as the app's (adds `extra_categorical`). |
| `gallup_v2_code/README.md` | modified | v4 commands and change table. |
| `gallup_v2_testsets/README.md` | modified | Four new rows and expected app results. |
| `gallup_v2_testsets/long_main/`, `long_null/`, `long_purenull/`, `long_seed2/` | **new** | 50,000 employees × 5 waves each (~46 MB each). |

The old data files were **not** regenerated (they would be identical). Their health reports
still say checker 3.0; the results are the same as 3.1.

## 2. Results

| Check | Result |
|---|---|
| App test suite (stand-ins for streamlit/openai) | 290 run (3 skipped): 284 pass; the 3 failures also fail on Selvi's original (`test_browser_e2e` needs a real browser + streamlit; `test_time_series::test_vanishing_people…` pandas version; `test_recommendations::test_team_plan_is_plain_language[-1]` known) |
| Dirty A/B/C, 5 questions each | 15 of 15 refused, 0 language-model calls |
| 2M analysis | T1–T5 ROBUST (Q04 0.62, Q05 0.81, Q12 0.78, Q05 absence 0.86, Q01 perf 0.13); X4 RR 1.83; X5 new hires OR 0.50 vs 0.71 (same as v3) |
| Validator, 2-wave sets | 5K 36/36, null 19/19, weak 36/36, seed2 36/36, dirty 10/10, 10/10, 5/5 |
| Validator, multi-year sets | long_main 52/52, long_seed2 52/52, long_null 35/35, long_purenull 35/35 |
| Team turnover model, long_main | passed; Q12 permutation p = 0.005; change gate passed; calibrated; graded on hidden wave 5: AUC 0.93 against true team risk |
| long_purenull (nothing in the survey predicts turnover) | model did not pass; Q12 permutation p = 0.33 → no warning signs listed |
| long_null (X7 on) | Q12 passes the permutation test because of nonresponse; the survey-response warning is shown, as planned |

## 3. Things to know

- **Speed.** The permutation test refits the team model up to 200 times (40 on very large
  files). 5K: about 2 s per outcome. 50K × 5 waves: about 13 s. 2M: about 70 s per outcome the
  first time; results are cached afterwards.
- **FDR with a blocked outcome** now runs over the 24 tests of the allowed outcomes (blocked
  outcomes are not fitted), as in v3.
- Her app already removed blocked outcome columns from every page (`app.py`); the new gates in
  `team_risk`, `employee_risk_by_group` and `monthly_risk` are a second safety layer.
- **Before merging, run on a laptop:** `python -m pytest -v` and a browser smoke test
  (no upload, clean upload, dirty upload, and one long_* set merged into one CSV).
- No `.env` was opened, copied or changed.
