"""Which survey wave each part of the app uses: one rule, one place.

- latest_wave(df): the newest survey (what leaders see as "now").
- model_wave(df):  the newest survey that has business outcomes. The statistical
  models, analysis_frame(), sorting charts, recommendations and financials all
  use this wave, so their numbers agree with each other and stay current.
  With outcomes in every wave, model_wave == latest_wave.
- model_sample(df): the rows of model_wave, ready for analyze_dataframe().
"""

# PROCESS 1 – IMPORTS

from __future__ import annotations

from typing import Any

import pandas as pd


# PROCESS 2 – CONFIGURATION

OUTCOME_COLUMNS = [
    "left_voluntarily_12m",
    "unplanned_absence_days_6m",
    "performance_rating",
]


# PROCESS 3 – WAVE RULES

def latest_wave(df: pd.DataFrame) -> Any:
    if "wave" not in df.columns or df["wave"].isna().all():
        return None
    return df["wave"].max()


def model_wave(df: pd.DataFrame) -> Any:
    """Newest wave with any outcome data; the newest wave if there are no outcomes."""
    if "wave" not in df.columns or df["wave"].isna().all():
        return None
    outcomes = [c for c in OUTCOME_COLUMNS if c in df.columns]
    if outcomes:
        labeled = df.loc[df[outcomes].notna().any(axis=1), "wave"]
        if labeled.notna().any():
            return labeled.max()
    return df["wave"].max()


def model_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the model wave (all rows when there is no wave column)."""
    wave = model_wave(df)
    return df if wave is None else df[df["wave"] == wave]


def wave_label(wave: Any) -> str:
    try:
        number = float(wave)
        return f"wave {int(number)}" if number.is_integer() else f"wave {wave}"
    except (TypeError, ValueError):
        return "the survey"
