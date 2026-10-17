"""Is this data ready for time-series prediction?

Scores any upload (survey waves or a person-month file) against what time-series
models need, in plain words, with how to fix each gap:

  time column       real dates or months, not only survey wave numbers
  time points       4+ for a trend forecast, 12+ for monthly leaving risk
  exit timing       exit dates (or a per-period 'exited' flag), not only "left in 12 months"
  stable IDs        the same employee_id across periods, once per period
  leavers kept      people who disappear are marked as exits (no survivorship bias)
  no rows after exit  nothing recorded after someone left (a sign of leakage)
  history kept      team/department as they were then, not today's copied back
  non-respondents   people who did not answer are kept (responded = 0)
  enough exits      enough leavers for a model to learn from
  same questions    all 12 questions present in every period with survey data

readiness(df) returns the checks, what each kind of prediction needs, and a verdict.
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_ingestion import Q12_ITEMS


# PROCESS 2 – CONFIGURATION

TIME_COLUMNS = ["period", "month", "date", "snapshot_date", "survey_date", "wave"]
MIN_TREND_POINTS = 4
MIN_SURVIVAL_POINTS = 12
MIN_EXITS = 30
GOOD_EXITS = 100
STATUS_ORDER = ["fail", "warn", "pass"]


def _check(key, check, status, result, fix="") -> dict[str, str]:
    return {"key": key, "check": check, "status": status, "result": result,
            "fix": fix if status != "pass" else ""}


def time_column(df: pd.DataFrame) -> str | None:
    return next((c for c in TIME_COLUMNS if c in df.columns), None)


def period_index(values: pd.Series) -> pd.Series:
    """Sortable period key: monthly periods for dates, the value itself otherwise."""
    if pd.api.types.is_numeric_dtype(values):
        return values
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().mean() > 0.9:
        return parsed.dt.to_period("M")
    return values


def exit_flags(df: pd.DataFrame, period: pd.Series) -> pd.Series | None:
    """1 in the period an employee exits, from 'exited' or 'exit_date'."""
    if "exited" in df.columns:
        return pd.to_numeric(df["exited"], errors="coerce").fillna(0).astype(int)
    if "exit_date" in df.columns and isinstance(period.dtype, pd.PeriodDtype):
        exit_month = pd.to_datetime(df["exit_date"], errors="coerce").dt.to_period("M")
        return (exit_month == period).astype(int)
    return None


# PROCESS 3 – THE CHECKS

def readiness(df: pd.DataFrame) -> dict[str, Any]:
    checks = []
    tcol = time_column(df)
    has_ids = "employee_id" in df.columns

    # Time column.
    if tcol is None:
        checks.append(_check("time", "Time column", "fail", "No date, month or wave column.",
                             "Add a 'period' column (e.g. 2026-03) to every row."))
        return _finish(checks, df, None, None)
    period = period_index(df[tcol])
    dated = isinstance(period.dtype, pd.PeriodDtype)
    checks.append(_check(
        "time", "Time column", "pass" if dated else "warn",
        f"Uses '{tcol}'" + (" with real dates (monthly)." if dated else
                             ": survey numbers only, without dates."),
        "Add a 'period' column with the month (e.g. 2026-03), so models know how far apart "
        "the points are."))

    # Time points.
    n_points = int(period.nunique())
    status = ("pass" if n_points >= MIN_SURVIVAL_POINTS else
              "warn" if n_points >= MIN_TREND_POINTS else "fail")
    checks.append(_check(
        "points", "Number of time points", status,
        f"{n_points} time points. A trend forecast needs {MIN_TREND_POINTS}+; monthly leaving "
        f"risk needs {MIN_SURVIVAL_POINTS}+.",
        "Add monthly HR snapshots, or run short quarterly pulse surveys, to build up history."))

    # Exit timing.
    flags = exit_flags(df, period)
    if flags is not None:
        checks.append(_check("exits", "When people left", "pass",
                             "Exits are dated (per period), so the model can learn when "
                             "people leave."))
    elif "left_voluntarily_12m" in df.columns:
        checks.append(_check("exits", "When people left", "warn",
                             "Only 'left within 12 months' (yes/no): no exit dates.",
                             "Add 'exit_date' for every leaver (or an 'exited' flag per period)."))
    else:
        checks.append(_check("exits", "When people left", "fail", "No exit information.",
                             "Add 'exit_date' for every leaver."))

    # Stable IDs.
    if not has_ids:
        checks.append(_check("ids", "Same employee across time", "fail", "No employee_id column.",
                             "Add a stable employee_id that never changes."))
    else:
        dupes = int(pd.DataFrame({"id": df["employee_id"], "t": period}).duplicated().sum())
        seen = pd.DataFrame({"id": df["employee_id"], "t": period}).groupby("id")["t"].nunique()
        repeat = float((seen >= 2).mean()) if len(seen) else 0.0
        status = "fail" if dupes else "pass" if repeat >= 0.5 else "warn"
        checks.append(_check(
            "ids", "Same employee across time", status,
            (f"{dupes:,} employees appear twice in the same period." if dupes else
             f"{repeat:.0%} of employees appear in 2 or more periods."),
            "Keep one row per employee per period, with the same employee_id every time."))

    # Leavers kept / rows after exit.
    if has_ids and n_points >= 2:
        frame = pd.DataFrame({"id": df["employee_id"], "t": period})
        if flags is not None:
            frame["exit"] = flags.to_numpy()
        elif "left_voluntarily_12m" in df.columns:          # wave data: the 12-month flag
            frame["exit"] = pd.to_numeric(df["left_voluntarily_12m"], errors="coerce").fillna(0).to_numpy()
        last_t = frame["t"].max()
        last_seen = frame.groupby("id")["t"].max()
        vanished = last_seen[last_seen < last_t].index
        if len(vanished):
            explained = (frame[frame["id"].isin(vanished)].groupby("id")["exit"].max().mean()
                         if "exit" in frame.columns else 0.0)
            checks.append(_check(
                "leavers", "Leavers kept with an exit record",
                "pass" if explained >= 0.8 else "warn",
                f"{len(vanished):,} people stop appearing before the last period; "
                f"{explained:.0%} of them have an exit recorded.",
                "Keep leavers' rows up to their exit and mark the exit; people who just vanish "
                "make later periods look healthier than they are."))
        if flags is not None:
            first_exit = frame[frame["exit"] == 1].groupby("id")["t"].min()
            after = frame["id"].map(first_exit)
            rows_after = int((after.notna() & (frame["t"] > after)).sum())
            checks.append(_check(
                "after_exit", "Nothing recorded after an exit",
                "fail" if rows_after else "pass",
                (f"{rows_after:,} rows are dated after the person left." if rows_after else
                 "No rows after anyone's exit."),
                "Remove rows after an exit; they leak the future into training."))

    # History kept (point-in-time attributes).
    if has_ids and n_points >= 3 and "team_id" in df.columns:
        moves = pd.DataFrame({"id": df["employee_id"], "team": df["team_id"]}) \
            .groupby("id")["team"].nunique()
        movers = float((moves > 1).mean())
        checks.append(_check(
            "history", "Team as it was at the time",
            "pass" if movers > 0 else "warn",
            (f"{movers:.1%} of employees change team over time, so history looks real."
             if movers > 0 else "Nobody ever changes team, which suggests today's team was "
             "copied onto past rows."),
            "Take team, department and manager from the HR system as of each period."))

    # Non-respondents.
    if "responded" in df.columns:
        zero = int((pd.to_numeric(df["responded"], errors="coerce") == 0).sum())
        checks.append(_check("nonresponse", "People who did not answer are kept",
                             "pass" if zero else "warn",
                             f"{zero:,} rows of people who did not answer." if zero else
                             "Only people who answered are in the file.",
                             "Include everyone invited, with responded = 0 for non-answers."))

    # Enough exits.
    n_exits = (int(flags.sum()) if flags is not None else
               int(pd.to_numeric(df["left_voluntarily_12m"], errors="coerce").sum())
               if "left_voluntarily_12m" in df.columns else 0)
    status = "pass" if n_exits >= GOOD_EXITS else "warn" if n_exits >= MIN_EXITS else "fail"
    checks.append(_check("events", "Enough leavers to learn from", status,
                         f"{n_exits:,} exits in the data (models learn well from {GOOD_EXITS}+).",
                         "Add more history (more months) or more of the workforce."))

    # Same questions in every period.
    items = [i for i in Q12_ITEMS if i in df.columns]
    if items:
        answered = pd.DataFrame({"t": period}).join(df[items].notna())
        by_period = answered.groupby("t")[items].any()
        with_survey = by_period[by_period.any(axis=1)]
        partial = with_survey[~with_survey.all(axis=1)]
        checks.append(_check(
            "questions", "Same questions in every survey", "pass" if partial.empty else "warn",
            (f"All {len(items)} questions in every one of {len(with_survey)} survey periods."
             if partial.empty else
             f"{len(partial)} survey periods miss some questions."),
            "Keep the same questions and 1–5 scale; note any wording change."))

    return _finish(checks, df, period, flags)


# PROCESS 4 – WHAT CAN RUN, AND THE VERDICT

def _finish(checks, df, period, flags) -> dict[str, Any]:
    status = {c["key"]: c["status"] for c in checks}
    n_points = int(period.nunique()) if period is not None else 0
    has_items = any(i in df.columns for i in Q12_ITEMS)
    can = {
        "Engagement trend forecast": (
            n_points >= MIN_TREND_POINTS and has_items,
            f"needs {MIN_TREND_POINTS}+ time points with survey answers"),
        "Team risk from survey waves": (
            n_points >= 2 and ("left_voluntarily_12m" in df.columns or flags is not None),
            "needs 2+ waves with turnover"),
        "Monthly leaving risk (time series)": (
            status.get("time") == "pass" and status.get("exits") == "pass"
            and n_points >= MIN_SURVIVAL_POINTS and status.get("ids") in ("pass", "warn")
            and status.get("events") in ("pass", "warn") and status.get("after_exit") != "fail",
            f"needs dated months, exit dates, {MIN_SURVIVAL_POINTS}+ months and "
            f"{MIN_EXITS}+ exits"),
    }
    counts = {s: sum(c["status"] == s for c in checks) for s in STATUS_ORDER}
    ready = can["Monthly leaving risk (time series)"][0]
    verdict = ("Ready for monthly time-series prediction." if ready and not counts["fail"] else
               f"Not ready for monthly time-series prediction yet: {counts['fail']} "
               f"item{'s are' if counts['fail'] > 1 else ' is'} missing (see below)." if counts["fail"] else
               "Partly ready: see the warnings for what to add.")
    return {"checks": checks, "can_run": can, "counts": counts, "verdict": verdict,
            "ready_for_survival": ready}
