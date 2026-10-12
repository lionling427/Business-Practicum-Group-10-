# Team meeting: discussion points after v3

## 1. Cost estimate: how should we calculate and present money?

**What the app does now** (`src/tools/insights.py`, `dollar_impact()`; assumptions in `src/tools/financial.py`):

1. Takes the top turnover item and its odds ratio from the model.
2. Applies it to the company's own turnover rate for employees who answer 1–3 on that item, as if each moved up one point.
3. Multiplies the "fewer leavers" by a cost per leaver: average salary $60,000 × 50% = $30,000 (editable on the Financials page).
4. Selvi's 10K example: about 238 fewer leavers ≈ $7.1M a year.

**Questions to decide**

- Is converting an odds ratio into dollars acceptable? Our own maintenance guide (ch. 21) says it is not; the app labels it a scenario. We need one position for the deck.
- Which source do we cite for replacement cost (share of salary), and do we show a range instead of one number?
- Is "everyone who answers 1–3 moves up one point" a realistic scenario, or should the user choose the share of people and the size of the lift?
- Respondents understate the turnover link (X7). Should the estimate say it is conservative?
- Absence days and performance: include them, or turnover only?

**Homework:** each person brings one or two sources for replacement-cost assumptions and one example of how a consulting firm presents engagement ROI.

## 2. Language model: switch to GPT-5.4?

- No code change is needed. Set the deployment before starting the app:
  `export GW_AZURE_OPENAI_DEPLOYMENT=gpt-5.4` (default stays `gpt-5-nano`).
- gpt-5.4 is listed in the course sandbox and works with the Chat Completions call the app already uses. Only `gpt-5-pro` needs the Responses API.
- **Proposal:** decide with data, not by name. Run the same question bank on gpt-5-nano, gpt-5.4-mini and gpt-5.4, then compare:
  - numbers copied exactly
  - required caveats included
  - no causal wording
  - refusals when data is blocked
  - answer time and token cost (and 429 rate limits)
- **Questions:** who runs the comparison, and which question bank version do we use?

## 3. Five-year data: do we agree to build it into our generator?

**Why:** the Trends page and Gallup's own reports focus on change over time. Our current exam data has only two waves.

**What would change**

| Part | Change | Size of work |
|---|---|---|
| `generate_synthetic_data.py` | `--waves N` option; yearly leavers and new hires; planted trend events (e.g. a dip and recovery, one department declining, a manager programme) | Medium: two-wave logic already exists |
| `ground_truth.json` | New "true trends" section, so trend answers can be graded | Small |
| `health_check.py` | Survivorship checks use only the first two waves today; extend to every pair of waves | Small |
| `validate_synthetic_data.py` | Checks that each planted trend is in the data | Small |
| App | Decide which wave the models use (see 5) | Small |

**Risk:** low, if the 5-year set is an extra test set (e.g. 50,000 employees × 5 waves) and the graded 2-wave sets stay unchanged. Five waves at 1,000,000 employees would be 5M rows, which is too large to share.

**Questions**

- Do we agree?
- Do we replace Selvi's 5-year file with one from our generator, so trends get an answer key?
- Which trend events should we plant?

## 4. Demo: show accuracy or compatibility?

We need both, but each answers a different judge question. Use them in this order:

| Step | File | What it proves |
|---|---|---|
| 1. Accuracy | 2M synthetic data with answer key | The agent's answers match the planted truth: 5/5 effects, 0 false drivers, traps handled |
| 2. Safety | A dirty test set | The agent refuses to interpret bad data |
| 3. Compatibility | A file not made by our generator (e.g. Selvi's 10K, or a public HR file reshaped to Q12 columns) | It runs end to end on data we did not design: columns map, the health check runs, caveats appear |

- Accuracy can only be proven on data with an answer key, so step 1 is the core claim.
- Step 3 shows it is not tuned to our own files. Say clearly that its numbers cannot be graded.
- Real company data is not available (Gallup items are proprietary; employee data is confidential).
- **Question:** which file do we use for step 3?

## 5. Other decisions

- **Analysis wave:** the app models the newest wave with outcomes (wave 2); the answer key and `analyze_dataset` use wave 1. Which one do we grade?
- **Blocked outcomes on dashboard pages:** the agent no longer discusses blocked outcomes, but the Overview, Financials and Trends pages still show them. Should we add a page-level filter?
- **API key:** the course guide says to keep the key in the `GW_AZURE_OPENAI_KEY` environment variable. The app also reads `.env` (git-ignored). Do we keep `.env`?
- **Merge:** who reviews v3 and merges it into GitHub? Run `python -m pytest -v` and a browser smoke test first.
- **Known failing test:** `test_team_plan_is_plain_language[position=-1]` already failed before v3. Who fixes it?
