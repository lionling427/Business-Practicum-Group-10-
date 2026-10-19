# Test sets (gallup_v2_testsets)

Special datasets for Project 4 testing. Each folder has the same layout as the main data:
`agent_data/` (what the agent may read), `answer_key/` (graders only) and `reports/` (check results).

| Folder | Size | Design | What it tests | Who uses it |
|---|---|---|---|---|
| `null/` | 50,000 per wave | No engagement item affects any outcome | The agent reports no item as a driver | Test runner asks the question bank; graders score against `null/answer_key/ground_truth.json` |
| `weak/` | 50,000 per wave | T1–T5 half as strong | The agent finds small effects without exaggerating them | Same as above |
| `seed2/`, `seed3/` | 50,000 per wave | Same design as the main data, new random draws | The agent's conclusions stay stable | Same as above |
| `dirty_A/` | 5,000 per wave | One error of each type | Each health-check rule fires | Data owner: run `health_check.py`, then `validate_synthetic_data.py` |
| `dirty_B/` | 5,000 per wave | Many overlapping errors and a missing date column | Errors are counted exactly | Data owner |
| `dirty_C/` | 5,000 per wave | Boundary values: valid edges and just-invalid values | No false alarms at the edges | Data owner |
| `long_main/` | 50,000 × 5 waves | Main truths over 5 years; managers change; events E1 (better manager), E2 (worse manager), E3 (inflated scores); wave 5 outcomes hidden | Trends, team risk forecasts, the change gate | Test runner; graders use `hidden_outcomes.csv` and `team_true_risk.csv` |
| `long_null/` | 50,000 × 5 waves | No item affects any outcome, nonresponse (X7) still on | Predictions show the survey-response warning; no item is called a cause | Same as above |
| `long_purenull/` | 50,000 × 5 waves | No item effects and no nonresponse bias | Q12 must fail the permutation test; the team model must not pass | Same as above |
| `long_seed2/` | 50,000 × 5 waves | long_main design, seed 2 | Stability of trend and prediction results | Same as above |

## Dirty sets: two files to compare

- `answer_key/error_log.csv` lists every error we planted: where it is, which check should catch it, and the expected status.
- `reports/health_report.csv` (and `.json`) lists what the health check actually found.

Every planted error must appear in the report with the expected status and count.
`reports/validation_report.csv` already shows this comparison: all dirty sets pass.

## Multi-year sets: what the app should show (v4 app, team turnover model)

| Set | Model test | Q12 adds warning signs (permutation) | Change gate | Graded on hidden wave 5 |
|---|---|---|---|---|
| long_main | passed (AUC 0.77 vs 0.65 last year) | yes (p = 0.005) | passed | AUC 0.93 vs true risk |
| long_null | passed (0.68 vs 0.66) | yes, because of X7, so the survey-response warning is shown | failed | ranking driven by department |
| long_purenull | did not pass (0.66 vs 0.65) | no (p = 0.33), no warning signs listed | failed | no survey signal |
| long_seed2 | passed (0.75 vs 0.66) | yes (p = 0.005) | failed (24% false turnarounds, limit 20%) | AUC 0.93 vs true risk |

## Rebuild any set

See the commands in `gallup_v2_code/README.md`. The same seed always produces identical files.
