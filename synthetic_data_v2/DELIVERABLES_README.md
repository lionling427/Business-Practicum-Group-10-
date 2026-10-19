# Gallup BI Agent: Deliverables Guide (package v4)

Use this guide to match every file you receive to its purpose: what it is, when it is used, who uses it, and what they do with it.

## 0. What is new in v4

1. **Multi-year test sets.** Four new folders in `gallup_v2_testsets/`: `long_main`, `long_null`, `long_purenull`, `long_seed2`. Each has 50,000 employees over 5 yearly waves. Managers change, and three events are planted: E1 better manager, E2 worse manager, E3 inflated scores. Wave 5 outcomes are blank, as if that year has not finished, so the app's forecasts can be graded.
2. **The tools.** Generator 3.0 (`--waves`, long variants) and a validator with multi-year checks. `health_check.py` 3.1 is now **one file shared with the app**, with 5 multi-year checks that run only on 3+ waves.
3. **The app** (`gallup-engagement-agent/`). This is Selvi's Oct 17 version with all her content kept, plus:
   - the v3 fixes it was missing (health gate, blocked outcomes never fitted, X4/X5 reported, API retry);
   - the five fixes from *Prediction Review: Issues and Fixes*.

   Details are in `CHANGES_v4.md` at the top of the folder.
4. The 2-wave data (5K, 2M, null, weak, seed, dirty) is unchanged and byte-identical.

## 1. The idea in one paragraph

We are building an AI agent that answers HR questions such as "Which engagement items are linked to our turnover?" To prove its answers are right, we run it like an exam. We generate a fake but realistic company dataset in which the correct answers are planted in advance. Then we let the agent analyze it and compare its answers with an answer key. Almost every file below belongs to one step of that exam. The one exception is the health check, which is part of the agent product itself.

```
[Write the exam]   generate_synthetic_data.py  ->  agent_data/ (exam)  +  answer_key/ (answers)
[Check the exam]   validate_synthetic_data.py  ->  validation_report.csv (planted answers really exist)
[Take the exam]    the agent loads agent_data/
                     first: health_check.py    ->  health_report.json (what to watch out for)
                     then:  answers the HR question
[Grade the exam]   graders compare the agent's answers with answer_key/ground_truth.json
```

## 2. The packages

| Package | Contents | Give it to |
|---|---|---|
| `gallup_v2_code.zip` | The four Python scripts (generator 3.0, validator, health check 3.1, stats tools) | Data owner, agent developers |
| `gallup_v2_5K_agent_data.zip` | 5,000-employee exam data (practice) | Agent developers |
| `gallup_v2_5K_answer_key.zip` | Answers for the 5K data | Data owner, graders |
| `gallup_v2_2M_agent_data_part1.zip` + `_part2.zip` | 1,000,000-employee exam data (formal test, demo), split in two to stay under the 30 MB upload limit; unzip both into the same place to get one `gallup_v2_2M/` folder | Test runner |
| `gallup_v2_2M_answer_key.zip` | Answers for the 2M data | Graders only |
| `gallup_v2_testsets.zip` | Special exams: null, weak, seed2, seed3, dirty A/B/C, and the multi-year sets long_main, long_null, long_purenull, long_seed2 (about 46 MB each) | Test runner, graders, data owner |
| `gallup_v2_docs.zip` | Spec, health check guide, change logs, this guide | Everyone |

**Rule:** the agent may read only `agent_data/` and `reports/health_report.json`. Never put an `answer_key/` folder where the agent's tools can reach it.

## 3. Every file

### 3.1 Code (`gallup_v2_code.zip`)

| File | What it is | When | Who | Does what |
|---|---|---|---|---|
| `generate_synthetic_data.py` | Exam writer | Now (v2 data); again whenever an effect size changes | Data owner | Builds a dataset and its answer key from a size, seed and variant |
| `validate_synthetic_data.py` | Exam checker | After every generation, before handing data on | Data owner | Confirms each planted effect, null and trap exists; for dirty sets, confirms every planted error was caught |
| `health_check.py` | Product component, the same file as the app's `src/tools/health_check.py` | Automatically, every time the agent loads data (synthetic or real) | The agent system | Runs 45 checks (52 on multi-year data) and lists actions: add a control, add a caveat, suppress small groups, or block an analysis |
| `stats_tools.py` | Shared statistics | Imported by the scripts above | Agent developers | Logistic, negative binomial and linear models with team-clustered errors; FDR correction; the three standard outcome models from Matrix A |
| `README.md` | How to run the code | When running or rebuilding anything | Anyone | Commands for every dataset |

### 3.2 Exam data (`*_agent_data.zip`, and `agent_data/` inside each test set)

| File | What it is | When | Who | Does what |
|---|---|---|---|---|
| `agent_data/survey_responses.csv` | One row per employee per wave: Q01–Q12 (blank for non-respondents), department, tenure, role, pay band, shift, survey date; long sets also have `manager_id` | 5K during development; 2M in formal testing | Agent developers (5K), the agent (2M) | The survey the agent analyzes |
| `agent_data/individual_outcomes.csv` | Turnover within 12 months, absence days in 6 months, prior and current performance rating, and each outcome's window end date; in long sets wave 5 outcomes are blank | Same | Same | The business outcomes linked to the survey |
| `agent_data/team_outcomes.csv` | Team id, department and team size per wave | Same | Same | Lets the agent suppress teams under 5 and cluster errors by team |
| `reports/health_report.json` | Health check result for machines | Before every analysis | The agent's statistical tools | Read the actions and apply them automatically |
| `reports/health_report.csv` | Same result for people | When reviewing data quality | Anyone; source of deck Table 10 | Read PASS, WARN and FAIL per check |
| `reports/validation_report.csv` | Exam-checker result | After generation | Data owner | Confirms the dataset is fit to grade against |
| `README.md` | Description of this dataset, written by the generator | When you open a package | Anyone | Real counts: employees, teams, small teams, response rate, and the rebuild command |

### 3.3 Answer keys (`*_answer_key.zip`, and `answer_key/` inside each test set)

| File | What it is | When | Who | Does what |
|---|---|---|---|---|
| `ground_truth.json` | The answer key | Project 4 grading | Graders | Lists true effects (planted and realized values), true nulls, traps and their correct handling, the X7 design, seed and version |
| `teams_with_manager_quality.csv` | Hidden manager quality per team | When validating trap X3 | The validator | Proves that controlling for the manager removes the Q07–absence link |
| `full_data_with_latents.csv` (5K only) | Every employee's true answers and hidden engagement, including non-respondents | When explaining how the traps were built | Team, instructor | A "god's-eye view" of the data; not produced for 2M (rebuild with the same seed if ever needed) |
| `error_log.csv` (dirty sets only) | The list of planted errors | After running the health check on a dirty set | Data owner | Compared with the health report: every error must be caught |
| `hidden_outcomes.csv` (long sets only) | The real wave 5 outcomes that are blank in `agent_data/` | Grading forecasts | Graders | Compare the app's wave 5 team risk with what happened |
| `team_true_risk.csv` (long sets only) | True expected turnover per team per wave | Grading forecasts | Graders | `true_above_company` is the target the team ranking is graded on |
| `manager_changes.csv`, `events.csv` (long sets only) | Every manager change with hidden quality, and the teams hit by E1–E3 | Grading trends and the change gate | Graders | Check that improving or slipping teams are spotted, and that E3 teams are not called improved |

**How to grade with `ground_truth.json`:**

| The agent says | Compare with | Correct when |
|---|---|---|
| Which items are linked to an outcome | `true_effects` (T1–T5) | It finds them with the right direction |
| How large an effect is | `realized_values.respondents_only` | The realized value falls inside the agent's 95% CI |
| Which items are not drivers | `true_nulls` (N1–N3) | It does not report them as meaningful drivers |
| Which teams are at risk next year (long sets) | `team_true_risk.csv`, `team_true_risk_summary` | Its ranking beats `auc_previous_rate…` and stays below `auc_true_rate…` (the best possible) |
| Whether Q12 explains team risk (long sets) | `x7_prediction_note` | In long_null it shows the survey-response warning; in long_purenull it lists no Q12 warning signs |
| How it handled a trap | `traps` (X1–X8) | It takes the listed correct action, e.g. controls for department or adds the understatement caveat |

### 3.4 Test sets (`gallup_v2_testsets.zip`)

| Folder | Design | Tests | When | Who |
|---|---|---|---|---|
| `null/` | No item affects any outcome | The agent reports no driver | Project 4 | Test runner asks; graders score |
| `weak/` | True effects half as strong | Detection without exaggeration | Project 4 | Test runner asks; graders score |
| `seed2/`, `seed3/` | Same design, new random draws | Stable conclusions | Project 4 | Test runner asks; graders score |
| `dirty_A/` | One error of each type | Every health-check rule fires | Late Project 3 | Data owner |
| `dirty_B/` | Many overlapping errors, a missing date column | Exact error counts | Late Project 3 | Data owner |
| `dirty_C/` | Boundary values | No false alarms at the edges | Late Project 3 | Data owner |
| `long_main/` | Main truths over 5 waves, manager changes, events E1–E3, wave 5 outcomes hidden | Trends, team forecasts, the change gate | Project 4 | Test runner asks; graders score |
| `long_null/` | No item effects; nonresponse (X7) on | Predictions are called warning signs, with the survey-response warning | Project 4 | Same |
| `long_purenull/` | No item effects and no nonresponse bias | No Q12 warning signs; the team model does not pass | Project 4 | Same |
| `long_seed2/` | long_main with seed 2 | Stable trend and forecast results | Project 4 | Same |

Two people handle the scored sets: a test runner asks the agent the question bank and saves every answer; two graders who did not build the agent score the answers blind against that set's own `ground_truth.json`.

### 3.5 Documents (`gallup_v2_docs.zip`)

| File | What it is | When | Who |
|---|---|---|---|
| `Gallup_Agent_Matrix_A_B_v2.pdf` | Spec: the method and controls for each outcome (Matrix A); planted effects, nulls, traps, test sets and metrics (Matrix B) | Building the agent; grading; writing the deck | Agent developers, graders |
| `Data_Validation_Bias_Checks_v2.docx` | Health check guide: what each of the 45 checks does and why, with 2M results | Writing deck Section 5; answering judges' questions | Everyone |
| `CHANGE_LOG_v1_to_v2.pdf` | All 26 issues: problem, impact, fix and evidence | When someone asks why something changed | Everyone, instructor |
| `CHANGE_LOG_v2_to_v3.pdf`, `Gallup_Agent_Matrix_A_B_v3.pdf`, `Meeting_Discussion_Points_v3.md` | v3 fixes, updated spec, meeting topics | Same | Everyone |
| `CHANGES_v4.md` (top of the folder) | Every file changed in v4 and why, with test results | Before merging into GitHub | Agent developers, data owner |
| `DELIVERABLES_README.md` | This guide | When receiving files | Everyone |

## 4. Who gets what

| Role | Packages |
|---|---|
| Data owner | All |
| Agent developer | code, 5K_agent_data, docs (not answer keys, not 2M, while developing) |
| Test runner | 2M_agent_data, and the `agent_data/` folders in testsets |
| Grader | 2M_answer_key, the `answer_key/` folders in testsets, the spec PDF |

## 5. Key numbers (2M data, seed 1)

| Item | Value |
|---|---|
| Employees per wave / survey rows / teams | 1,000,000 / 2,000,000 / 80,000 |
| Teams under 5 people | 4,000 (5.0%) |
| Response rate (wave 1); leavers vs stayers | 81.2%; 60% vs 86% |
| Health check | 45 checks: 33 PASS, 12 WARN, 0 FAIL |
| Validator | 36 of 36 checks pass |
| X7 understatement | Mean-score turnover OR 0.36 for everyone vs 0.52 among respondents |

Multi-year sets (50,000 × 5 waves): validator long_main and long_seed2 52/52, long_null and long_purenull 35/35. The app's team turnover model on long_main scores AUC 0.93 against the true wave 5 team risk.

## 6. Three rules to remember

1. The agent reads only `agent_data/` and `health_report.json`.
2. Develop on 5K; test on 2M and the test sets, so the agent is never tuned on the exam.
3. Answer keys go to graders only, ideally people who did not build the agent.
