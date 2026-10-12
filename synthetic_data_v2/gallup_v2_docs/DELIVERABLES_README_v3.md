# Gallup BI Agent: Deliverables Guide (package v3)

Use this guide to see what changed from v2, and to match each file you receive to its purpose and to the person who needs it.

## 1. What is new in v3

1. **The app.** Selvi's second version (10 pages) with seven fixes. Blocked data is now refused for every kind of question, the health check runs before the models and on uploaded files, the X4 threshold and X5 new-hire patterns are reported, and API calls retry. Details: `CHANGE_LOG_v2_to_v3.pdf`.
2. **The weak test set.** Regenerated so every true effect stays outside the "no meaningful effect" band. T5 is now realized at IRR 0.89 (it was 0.94).
3. **The tools.**
   - `health_check.py` now works as functions, with the same 45 checks and the same results.
   - The validator checks that every true effect is meaningful (36 checks on the main data).
4. **The data files.**
   - The 5K, 2M, null, seed and dirty data are byte-identical to v2, so their folder names still say `v2`.
   - Their health and validation reports were regenerated with the v3 tools.

## 2. The packages

| Package | Contents | Give it to |
|---|---|---|
| `gallup_v3_agent_app.zip` | The Streamlit app with the seven fixes, its tests (including `tests/test_v3_fixes.py`) and the 5K development data | Agent developers |
| `gallup_v3_code.zip` | The four data tools: generator 2.1, validator, health check 3.0, stats tools | Data owner, agent developers |
| `gallup_v3_5K_agent_data.zip` | 5,000-employee practice data and reports | Agent developers |
| `gallup_v3_5K_answer_key.zip` | Answers for the 5K data | Data owner, graders |
| `gallup_v3_2M_agent_data_part1.zip` + `_part2.zip` | 1,000,000-employee test data, split in two to stay under the upload limit. Unzip both into the same place to get one `gallup_v2_2M/` folder | Test runner |
| `gallup_v3_2M_answer_key.zip` | Answers for the 2M data | Graders only |
| `gallup_v3_testsets.zip` | null, weak (new), seed2, seed3, dirty A/B/C | Test runner, graders, data owner |
| `gallup_v3_docs.zip` | Spec v3, health check guide, both change logs, this guide, meeting discussion points | Everyone |

**Rule:** the agent may read only `agent_data/` and `reports/health_report.json`. Never put an `answer_key/` folder where the agent's tools can reach it.

## 3. How to update your copy

1. **App.** Unzip `gallup_v3_agent_app.zip`, then copy your own `.env` into it if you use one (it is not included).
2. **Check it works.** Run:
   ```bash
   python -m pytest -v
   python -m streamlit run app.py
   ```
   One older test, `test_team_plan_is_plain_language[position=-1]`, already failed before v3.
3. **Smoke test in the browser.**
   - Start with no upload.
   - Upload a clean file. You should see a green badge.
   - Upload a dirty test set, with its three tables merged into one CSV. You should see a red badge, and the agent refuses.
4. **Data.** Only replace `testsets/weak/`. Everything else can stay.

## 4. Key numbers

| Item | Value |
|---|---|
| Health check | 45 checks. 2M: 33 PASS, 12 WARN, 0 FAIL. 5K: 32 PASS, 13 WARN, 0 FAIL |
| Validator | 5K, 2M, weak, seed2, seed3: 36/36. null 19/19. dirty_A 10/10. dirty_B 10/10. dirty_C 5/5 |
| App on 2M | T1–T5 found as ROBUST: Q04 0.62, Q05 0.81, Q12 0.78 (turnover OR); Q05 0.86 (absence IRR); Q01 0.13 (performance β). Q07 and Q11 shown as CAUTION. X4 rate ratio 1.83 (key 1.83). X5 new hires OR 0.50 vs others 0.71 |
| App on dirty A/B/C | All questions refused, no language-model call |
| App tests | 180 of 181 pass (the one failure existed before v3) |

## 5. Three rules to remember

1. The agent reads only `agent_data/` and `health_report.json`.
2. Develop on 5K. Test on 2M and the test sets, so the agent is never tuned on the exam.
3. Answer keys go to graders only, ideally people who did not build the agent.
