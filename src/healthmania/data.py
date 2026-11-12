"""Loading, validating and cleaning the sleep/stress table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BMI_ALIASES, ID_COLUMNS, NO_DISORDER, RAW_REQUIRED, TARGET_DISORDER


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV and check that the expected columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def duplicate_report(df: pd.DataFrame) -> dict:
    """Count exact duplicate records (ignoring ID/index columns).

    Duplicates that fall on both sides of a train/test split let a model
    memorise rows, which inflates reported accuracy.
    """
    content = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    n_dup = int(content.duplicated().sum())
    return {
        "n_rows": len(df),
        "n_duplicate_rows": n_dup,
        "duplicate_fraction": round(n_dup / len(df), 4) if len(df) else 0.0,
    }


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Return an analysis-ready frame.

    Steps: drop ID columns, label missing sleep disorder as 'No Disorder'
    (the source encodes absence as blank), harmonise BMI spellings, split
    blood pressure into numeric systolic/diastolic, optionally drop exact
    duplicates.
    """
    out = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns]).copy()
    out[TARGET_DISORDER] = out[TARGET_DISORDER].fillna(NO_DISORDER)
    out["BMI Category"] = out["BMI Category"].replace(BMI_ALIASES)

    bp = out["Blood Pressure"].astype(str).str.split("/", expand=True)
    out["SystolicBP"] = pd.to_numeric(bp[0], errors="raise").astype(int)
    out["DiastolicBP"] = pd.to_numeric(bp[1], errors="raise").astype(int)
    out = out.drop(columns=["Blood Pressure"])

    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    return out
