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

## Dirty sets: two files to compare

- `answer_key/error_log.csv` lists every error we planted: where it is, which check should catch it, and the expected status.
- `reports/health_report.csv` (and `.json`) lists what the health check actually found.

Every planted error must appear in the report with the expected status and count.
`reports/validation_report.csv` already shows this comparison: all dirty sets pass.

## Rebuild any set

See the commands in `gallup_v2_code/README.md`. The same seed always produces identical files.
