# Code: Gallup Decision Intelligence Agent, synthetic data (package v3)

Four Python files. They need only `numpy`, `pandas` and `scipy` (no statsmodels).

| File | Role | Who runs it |
|---|---|---|
| `generate_synthetic_data.py` | Builds a test dataset and its answer key | Data owner |
| `validate_synthetic_data.py` | Confirms every planted effect, null and trap is really in the data | Data owner, after each generation |
| `health_check.py` | Product component: 45 data checks run before any analysis, on synthetic or real data. v3: also callable as functions (`run_checks`, `check_dataframe`, `check_folder`) so the app checks uploaded files | The agent system, automatically |
| `stats_tools.py` | Shared statistics: logistic, negative binomial and linear models with team-clustered errors, FDR correction, and the three standard outcome models | Imported by the other files; basis for the agent's statistical tools |

## Quick start

```bash
pip install numpy pandas scipy

# 1. Build data (the same command always produces identical files)
python generate_synthetic_data.py --employees 5000    --seed 2 --out gallup_v2_5K
python generate_synthetic_data.py --employees 1000000 --seed 1 --out gallup_v2_2M

# 2. Health check (writes <folder>/reports/health_report.csv and .json)
python health_check.py gallup_v2_5K

# 3. Validate (writes <folder>/reports/validation_report.csv)
python validate_synthetic_data.py gallup_v2_5K
```

## Test sets

```bash
python generate_synthetic_data.py --employees 50000 --seed 1 --variant null  --out testsets/null
python generate_synthetic_data.py --employees 50000 --seed 1 --variant weak  --out testsets/weak
python generate_synthetic_data.py --employees 50000 --seed 2 --variant seed  --out testsets/seed2
python generate_synthetic_data.py --employees 50000 --seed 3 --variant seed  --out testsets/seed3
python generate_synthetic_data.py --employees 5000  --seed 2 --variant dirty_A --out testsets/dirty_A
python generate_synthetic_data.py --employees 5000  --seed 2 --variant dirty_B --out testsets/dirty_B
python generate_synthetic_data.py --employees 5000  --seed 2 --variant dirty_C --out testsets/dirty_C
```

For a dirty set, run `health_check.py` first, then `validate_synthetic_data.py`: the validator compares
`answer_key/error_log.csv` (errors we planted) with `reports/health_report.json` (errors the check found).

## Where the settings live

- Every planted effect is in the `TRUTH` dictionary at the top of `generate_synthetic_data.py`. Change numbers there only.
- The validator reads expected values from `answer_key/ground_truth.json`, so it never needs editing when settings change.
- The three outcome models (controls per outcome, from Matrix A) are defined once in `stats_tools.py`.
- Health-check thresholds are in the CONFIG block at the top of `health_check.py`; rename columns there for real data.

## What changed in package v3

| File | Version | Change |
|---|---|---|
| `generate_synthetic_data.py` | 2.1 | `weak` variant: T5 (Q05 -> absence) planted at IRR 0.87 instead of 0.922. At 0.922 the realized value among respondents was 0.94, inside the 0.91-1.10 "no meaningful effect" band, so a correct agent would not report it. Now realized 0.886. All other variants produce byte-identical files to v2. |
| `validate_synthetic_data.py` | - | Five new checks: every true effect's realized value must lie outside the no-effect band (T1-T5). Main data now 36 checks. |
| `health_check.py` | 3.0 | Same 45 checks and identical results; the logic now sits in functions so the app can run it on uploads. A step that cannot run (missing column in an uploaded file) becomes a WARN; a broken integrity step blocks everything. |
| `stats_tools.py` | - | Unchanged. |
