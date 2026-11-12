"""Build six cumulative, date-aware uploads from one longitudinal simulation.

Run: python synthetic_data_v2/six_wave_demo/build.py
Only this package directory is written. All checks fail closed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT.parent / "gallup_v2_code" / "generate_synthetic_data.py"
APP = ROOT.parents[1] / "gallup-engagement-agent"
WORK = ROOT / "generated_work"
UPLOADS = ROOT / "uploads"
PRIVATE = ROOT / "private_answer_key"
WAVES = 6
EMPLOYEES = 5000
SEED = 1
SCENARIO = "long_main"
OUTCOMES = {
    "turnover": ("left_voluntarily_12m", "turnover_window_end"),
    "absence": ("unplanned_absence_days_6m", "absence_window_end"),
    "performance": ("performance_rating", "performance_review_date"),
}
PRIVATE_COLUMNS = {
    "engagement_latent", "prior_perf_latent", "manager_quality",
    "turnover_prob_true", "true_index", "response_prob", "event", "reason",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_tables():
    survey = pd.read_csv(WORK / "agent_data/survey_responses.csv", dtype={"employee_id": str, "team_id": str, "manager_id": str})
    indiv = pd.read_csv(WORK / "agent_data/individual_outcomes.csv", dtype={"employee_id": str})
    team = pd.read_csv(WORK / "agent_data/team_outcomes.csv", dtype={"team_id": str})
    truth = pd.read_csv(PRIVATE / "full_data_with_latents.csv", dtype={"employee_id": str, "team_id": str, "manager_id": str})
    for label, table, keys in (
        ("survey", survey, ["wave", "employee_id"]),
        ("outcomes", indiv, ["wave", "employee_id"]),
        ("team", team, ["wave", "team_id"]),
        ("private truth", truth, ["wave", "employee_id"]),
    ):
        require(not table.duplicated(keys).any(), f"Duplicate {label} keys")
    require(len(survey) == WAVES * EMPLOYEES, "Unexpected survey population")
    require(set(survey.wave) == set(range(1, WAVES + 1)), "Unexpected survey waves")
    require(len(survey) == len(indiv) == len(truth), "Source row counts differ")
    require(set(zip(survey.wave, survey.employee_id)) == set(zip(indiv.wave, indiv.employee_id)),
            "Survey and outcome keys differ")
    require(set(zip(survey.wave, survey.team_id)) <= set(zip(team.wave, team.team_id)),
            "Survey has unknown team")
    require((survey.merge(team[["wave", "team_id", "department"]],
                          on=["wave", "team_id"], validate="many_to_one",
                          suffixes=("", "_team"))["department"] ==
             survey.merge(team[["wave", "team_id", "department"]],
                          on=["wave", "team_id"], validate="many_to_one",
                          suffixes=("", "_team"))["department_team"]).all(),
            "Department differs between survey and team")
    require(set(zip(survey.wave, survey.employee_id)) ==
            set(zip(truth.wave, truth.employee_id)), "Private truth keys differ")
    return survey, team, truth


def check_continuity(survey, truth):
    departments = survey.groupby("team_id")["department"].nunique()
    require((departments == 1).all(), "Team changed department")
    team_by_person = survey.groupby("employee_id")["team_id"].nunique()
    require((team_by_person == 1).all(), "Stayer changed team")
    first = survey.groupby("employee_id")["wave"].min()
    last = survey.groupby("employee_id")["wave"].max()
    counts = survey.groupby("employee_id").size()
    require((counts == last - first + 1).all(), "Employee disappeared then reappeared")
    prior = truth[["wave", "employee_id", "left_voluntarily_12m"]].copy()
    prior["wave"] += 1
    following = survey[["wave", "employee_id"]].merge(
        prior, on=["wave", "employee_id"], how="left", validate="one_to_one")
    require(not (following["left_voluntarily_12m"] == 1).any(), "Leaver reappeared")
    require((survey.groupby(["wave", "team_id"])["manager_id"].nunique() == 1).all(),
            "Multiple managers in a team-wave")
    require(survey.groupby("wave").size().eq(EMPLOYEES).all(), "Wave headcount changed")


def build_combined(survey, truth):
    # The generator's private full table retains the last wave's true outcomes.
    # Use only these approved outcome fields, never a hidden trait or truth field.
    cols = ["wave", "employee_id"]
    for value, date in OUTCOMES.values():
        cols += [value, date]
    cols += ["prior_performance_rating"]
    combined = survey.merge(truth[cols], on=["wave", "employee_id"],
                            how="left", validate="one_to_one", sort=False)
    survey_columns = list(survey.columns)
    combined = combined[survey_columns + [
        "left_voluntarily_12m", "turnover_window_end",
        "unplanned_absence_days_6m", "absence_window_end",
        "prior_performance_rating", "performance_rating", "performance_review_date",
    ]]
    require(len(combined) == len(survey), "Combined row count changed")
    require(not (set(combined.columns) & PRIVATE_COLUMNS), "Hidden field entered combined data")
    return combined


def visible_snapshot(combined, latest):
    frame = combined.loc[combined.wave <= latest].copy()
    dates = pd.to_datetime(frame.survey_date, errors="coerce")
    require(dates.notna().all(), "Invalid survey date")
    cutoff = dates.loc[frame.wave == latest].max().normalize()
    require(pd.notna(cutoff), "Missing snapshot cutoff")
    counts = {}
    for name, (value, date) in OUTCOMES.items():
        available = pd.to_datetime(frame[date], errors="coerce")
        require(available.notna().all(), f"Missing true {name} availability date")
        known = available.le(cutoff)
        frame.loc[~known, [value, date]] = pd.NA
        require(frame.loc[known, value].notna().all(), f"Known {name} value missing")
        require(frame.loc[~known, [value, date]].isna().all().all(),
                f"Pending {name} leaked")
        counts[name] = {"known": int(known.sum()), "pending": int((~known).sum())}
    # The generator samples prior_performance_rating from a latent trait at the
    # current wave. No date establishes when that rating became available.
    frame["prior_performance_rating"] = pd.NA
    require(frame["prior_performance_rating"].isna().all(), "Undated prior rating leaked")
    require(not frame.duplicated(["wave", "employee_id"]).any(), "Duplicate upload rows")
    require(not (set(frame.columns) & PRIVATE_COLUMNS), "Private field in upload")
    require(set(frame.wave) == set(range(1, latest + 1)), "Snapshot wave gap")
    require(len(frame) == latest * EMPLOYEES, "Snapshot row count wrong")
    return frame, cutoff, counts


def app_check(path, latest, expected_model_wave):
    sys.path.insert(0, str(APP))
    from src.tools.data_ingestion import detect_capabilities, normalize_columns, read_uploaded_file, validate_uploaded_data
    from src.tools.longitudinal import derive_wave
    from src.tools.waves import latest_wave, model_wave

    start = time.perf_counter()
    with path.open("rb") as stream:
        data = derive_wave(normalize_columns(read_uploaded_file(stream)))
    seconds = round(time.perf_counter() - start, 3)
    validation = validate_uploaded_data(data)
    require(validation["capabilities"]["has_core"], "App rejected core schema")
    require(latest_wave(data) == latest, "App latest wave mismatch")
    require(model_wave(data) == expected_model_wave, "App model wave mismatch")
    return {"capability": detect_capabilities(data)["level"],
            "model_wave": expected_model_wave, "ingestion_seconds": seconds}


def main():
    require(not WORK.exists() and not UPLOADS.exists() and not PRIVATE.exists(),
            "Package outputs already exist; refusing to overwrite")
    command = [sys.executable, str(GENERATOR), "--variant", SCENARIO,
               "--waves", str(WAVES), "--employees", str(EMPLOYEES),
               "--seed", str(SEED), "--full-data", "yes",
               "--out", str(WORK)]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    generation_seconds = round(time.perf_counter() - started, 3)
    PRIVATE.mkdir()
    for source in (WORK / "answer_key").iterdir():
        require(source.is_file(), "Unexpected answer-key directory")
        shutil.move(str(source), PRIVATE / source.name)
    (WORK / "answer_key").rmdir()
    survey, team, truth = source_tables()
    check_continuity(survey, truth)
    combined = build_combined(survey, truth)
    UPLOADS.mkdir()
    started = time.perf_counter()
    manifest = {"scenario": SCENARIO, "seed": SEED, "employees_per_wave": EMPLOYEES,
                "waves": WAVES, "generation_seconds": generation_seconds,
                "generator_command": "python synthetic_data_v2/gallup_v2_code/generate_synthetic_data.py --variant long_main --waves 6 --employees 5000 --seed 1 --full-data yes --out synthetic_data_v2/six_wave_demo/generated_work",
                "prior_performance": "Masked: generator has no dated proof it predates the row's survey.",
                "snapshots": [], "private_answer_key_hashes": {}}
    schema = None
    for wave in range(1, WAVES + 1):
        frame, cutoff, counts = visible_snapshot(combined, wave)
        require(schema is None or list(frame.columns) == schema, "Upload schema changed")
        schema = list(frame.columns)
        filename = f"long_main_seed01_wave_{wave:02d}_asof.csv"
        path = UPLOADS / filename
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        entry = {"filename": filename, "latest_wave": wave,
                 "included_waves": list(range(1, wave + 1)),
                 "cutoff_date": cutoff.date().isoformat(),
                 "rows": len(frame),
                 "rows_by_wave": {str(k): int(v) for k, v in frame.wave.value_counts().sort_index().items()},
                 "unique_employees": int(frame.employee_id.nunique()),
                 "unique_teams": int(frame.team_id.nunique()),
                 "outcomes": counts, "bytes": path.stat().st_size, "sha256": digest(path)}
        expected_model = int(frame.loc[frame[[v for v, _ in OUTCOMES.values()]].notna().any(axis=1), "wave"].max()) if any(x["known"] for x in counts.values()) else wave
        entry["application"] = app_check(path, wave, expected_model)
        manifest["snapshots"].append(entry)
    manifest["export_seconds"] = round(time.perf_counter() - started, 3)
    manifest["private_answer_key_hashes"] = {
        path.name: digest(path) for path in sorted(PRIVATE.iterdir()) if path.is_file()}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generation_seconds": generation_seconds,
                      "export_seconds": manifest["export_seconds"],
                      "snapshots": manifest["snapshots"]}, indent=2))


if __name__ == "__main__":
    main()
