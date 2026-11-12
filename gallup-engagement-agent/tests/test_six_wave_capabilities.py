"""Focused capability and as-of snapshot checks for the six-wave demonstration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.data_ingestion import detect_capabilities, normalize_columns, read_uploaded_file
from src.tools.waves import latest_wave, model_wave


UPLOADS = Path(__file__).resolve().parents[2] / "synthetic_data_v2/six_wave_demo/uploads"


def frame(**outcomes):
    return pd.DataFrame({"employee_id": ["E1"], **{f"Q{i:02d}": [3] for i in range(1, 13)}, **outcomes})


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        ({}, []),
        ({"left_voluntarily_12m": [None]}, []),
        ({"left_voluntarily_12m": [" "]}, []),
        ({"left_voluntarily_12m": ["bad"]}, []),
        ({"left_voluntarily_12m": [np.nan]}, []),
        ({"left_voluntarily_12m": [np.inf]}, []),
        ({"left_voluntarily_12m": [2]}, []),
        ({"left_voluntarily_12m": [0]}, ["turnover"]),
        ({"left_voluntarily_12m": [1]}, ["turnover"]),
        ({"unplanned_absence_days_6m": [0]}, ["absence"]),
        ({"unplanned_absence_days_6m": [-1]}, []),
        ({"performance_rating": [0]}, ["performance"]),
        ({"performance_rating": [4]}, ["performance"]),
        ({"performance_rating": ["Infinity"]}, []),
        ({"performance_rating": [""], "performance_review_date": ["2025-01-01"]}, []),
    ],
)
def test_observed_outcomes(outcomes, expected):
    capability = detect_capabilities(frame(**outcomes))
    assert capability["outcomes"] == expected
    assert capability["level"] == ("OUTCOME_ANALYSIS" if expected else "SURVEY_ONLY")


@pytest.mark.parametrize("wave,model", [(1, 1), (2, 1), (6, 5)])
def test_real_snapshot_ingestion(wave, model):
    path = UPLOADS / f"long_main_seed01_wave_{wave:02d}_asof.csv"
    with path.open("rb") as uploaded:
        data = normalize_columns(read_uploaded_file(uploaded))
    capabilities = detect_capabilities(data)
    assert latest_wave(data) == wave
    assert model_wave(data) == model
    assert capabilities["level"] == ("SURVEY_ONLY" if wave == 1 else "OUTCOME_ANALYSIS")
    if wave == 1:
        assert not data[["left_voluntarily_12m", "unplanned_absence_days_6m",
                         "performance_rating"]].notna().any().any()
    if wave == 6:
        newest = data[data.wave == 6]
        assert newest[["left_voluntarily_12m", "unplanned_absence_days_6m",
                       "performance_rating"]].isna().all().all()
