"""Inference helpers. Takes raw, human-readable values and returns predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import BMI_ALIASES, TARGETS


def load_bundle(path: str | Path = "models/healthmania_model.joblib") -> dict:
    return joblib.load(path)


def predict_records(bundle: dict, records: list[dict]) -> pd.DataFrame:
    """Predict stress level and sleep disorder for a list of raw records.

    Each record needs: Gender, Age, Occupation, Sleep Duration, Quality of Sleep,
    Physical Activity Level, BMI Category, Heart Rate, Daily Steps, SystolicBP, DiastolicBP.
    Preprocessing (encoding, scaling) is part of the fitted pipeline, so nothing is re-fitted here.
    """
    X = pd.DataFrame(records)
    if "BMI Category" in X:
        X["BMI Category"] = X["BMI Category"].replace(BMI_ALIASES)
    X = X[bundle["features"]]
    out = pd.DataFrame(index=X.index)
    for target in TARGETS:
        pipe = bundle["models"][target]
        out[f"Predicted {target}"] = pipe.predict(X)
        if hasattr(pipe, "predict_proba"):
            out[f"{target} confidence"] = pipe.predict_proba(X).max(axis=1).round(3)
    return out
