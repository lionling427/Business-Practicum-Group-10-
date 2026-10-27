# CHANGES (branch Will): financial section update

Base: Jenny's `main` at commit 2b708c0 (v4). `main` is not modified; everything below is on the
`Will` branch only. Source of the changes: the memo *Financial Impact Review: Issues and Fixes*.

## Renamed file

`gallup-engagement-agent/src/tools/financial.py` is now
`gallup-engagement-agent/src/tools/financial_updated.py`. Every import was updated
(`src/tools/*`, `app_pages/*`, `tests/*`). Nothing else was renamed.

## What changed, by fix (applied in the order 1, 2, 4, 3, 5, 6)

| Fix | Change | Files |
|---|---|---|
| 1 Costs count everyone | `outcome_frame()` (all employees in the model wave), `default_assumptions()` (headcount = all employees); `current_costs` and `team_excess_cost` use them. The action story scales expected leavers against all employees, so by default it matches the Predictions page | `financial_updated.py`, `action_story.py`, every page that built default assumptions |
| 2 One engine | `engagement_value` and `priorities.factor_value` call `scenario.simulate()` (each person's own risk). `simulate` accepts a shared model fit (`fits`), and `factor_value` caches its cost-independent results on the analysis. `dollar_impact` and the shortcut `avoided_leavers` are no longer used by the Financials page (they stay in the code for the tests and for comparison) | `financial_updated.py`, `priorities.py`, `scenario.py`, `audit.py` |
| 4 Ranges | Every scenario row and every factor has `low` / `high` (`value_low` / `value_high`) from the 95% interval; `break_even_per_person()`; the audit range check reads the same numbers | `financial_updated.py`, `priorities.py`, `audit.py`, `financials.py` |
| 3 One named program | The headline is one program (the "Fix first" factors, or the top factor), simulated jointly; people helped by two questions count once. `engagement_value(items=...)` lets a caller name the program | `financial_updated.py`, `financials.py` |
| 5 Survey-response note | `money_warning()` reuses the prediction pages' X7 rates; shown on Financials, CEO brief, Report pack, Action story, Scenarios, and in the facts Ask AI reads | `financial_updated.py`, five pages, `agent_context.py` |
| 6 Where the money is | Teams smoothed toward the company rate (k = 20), department view (`department_excess_cost`), team costs also compared with the team's own department, year-over-year check (`team_cost_persistence`: passes when the costliest 1% of teams keep at least 75% of their extra cost next year; pending with one year). Failed check hides team rankings on the Financials page, in Ask AI facts and in the action story's named teams. Story text uses thousands separators and "<1%" | `financial_updated.py`, `financials.py`, `agent_context.py`, `action_story.py` |

## Tests

`tests/test_financial.py` was rewritten around the new behavior (35 tests): costs equal the real
leaver count, Financials equals Scenarios for the same scenario, the headline is one program,
every figure has a range, small teams are smoothed, the department view, the year-over-year check
(including a pure-chance dataset that must fail), and the survey-response note (shown, absent,
and read from the health report). Imports in the other test files were updated for the rename.

## Things to know

- **The year-over-year check is borderline on the 2M data.** With k = 20 the costliest 760 teams
  keep 74.7% of their extra cost against the 75% bar, so the page hides team rankings there.
  Across other flagged-team counts the ratio is 69% to 85%; raw rates keep about 30%; k = 40 overshoots.
  k and the bar are proposals and should be tuned on the `long_*` sets, not on the 2M data.
- **Numbers on the page move.** Costs today, every dollar figure and the department and team
  tables change with Fix 1 and Fix 2. See the memo update for before and after figures.
- **Predictions page unchanged.** Its expected-leavers count still uses respondents only; that
  belongs to the Prediction Review, not this change.
- **Speed.** The factor ranking fits each outcome model once per analysis (about 30 s on 2M the
  first time, then cached on the analysis).

## Second update: Sunny's EV brief and the Gallup methodology notes

Analysis of what was taken and what was not is in the hand-off message; the code is below. All
on branch `Will`, committed locally, not pushed.

| Idea from the brief | Change | Where |
|---|---|---|
| Exposure vs opportunity | `exposure_vs_opportunity()`; page caption and Ask AI fact | `financial_updated.py`, `financials.py`, `agent_context.py` |
| Assumption registry with source labels | `Assumptions.sources`, `assumption_register()`; source picker in the form; table under "Assumptions and limits"; placeholders are labelled as placeholders | same |
| ROI, net value, payback | `program_economics()` from a user-entered yearly program cost and a "share of modeled effect delivered" slider; downside/base/upside from the 95% range | same |
| Break-even / threshold | break-even cost, required share of the modeled effect, and `break_even_lift()` (the smallest average rise for 1-3 scorers that covers the cost, from the Scenarios engine; no lift is assumed) | same |
| Uncertainty vs sensitivity | `assumption_sensitivity()` table, kept apart from the 95% statistical range | same |
| Calculation receipts | `calculation_receipts()` / `receipt_markdown()`; expander on the page; short version in Ask AI facts | same |
| Double-counting and guardrail warnings | `engagement_value()["warnings"]` shown under the program | `financial_updated.py` |
| Utility-analysis productivity value (Hunter-Schmidt, as Gallup describes) | optional `sd_productivity_pct`; value per point = salary × SD% ÷ SD(rating); off by default; explicit value wins | `financial_updated.py`, `scenario.py`, `priorities.py` |
| Gallup 2024 meta-analysis as a check | `gallup_benchmark_check()`: team-level correlations vs Gallup's observed mean ± 1.28 SD; plausibility only, never feeds dollars | `financial_updated.py`, `financials.py` |

Not implemented on purpose: Gallup's top-down cost-of-disengagement figure (proprietary
classification and GDP method, excludes turnover); using Gallup's correlations or its reported
engagement growth as an effect size in dollars; a default for the SD of productivity; the brief's
later phases (intervention library, optimizer, learning loop), which need evidence this data does not hold.

Tests: `tests/test_financial.py` now has 44 tests; `tests/test_app_pages.py` has a page test with a program
cost entered. Ask AI facts stay under the 20,000-character limit.
