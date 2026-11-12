"""Model zoo and the preprocessing + classifier pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from .config import (
    BINARY_FEATURES,
    BMI_ORDER,
    GENDER_ORDER,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """Encode and scale features. Fitted on training data only, inside the pipeline."""
    return ColumnTransformer(
        [
            ("gender", OrdinalEncoder(categories=[GENDER_ORDER], handle_unknown="error"),
             BINARY_FEATURES),
            ("bmi", OrdinalEncoder(categories=[BMI_ORDER], handle_unknown="error"),
             ORDINAL_FEATURES),
            ("occupation", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_classifier(name: str):
    if name == "linear_svm":  # the 2024 baseline
        return SVC(kernel="linear", class_weight=None, random_state=RANDOM_STATE)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["linear_svm", "logreg", "random_forest", "hist_gb"]


def build_pipeline(name: str) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", make_classifier(name))])
