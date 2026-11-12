"""Cross-validated model comparison, hold-out reports and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import RANDOM_STATE
from .models import MODEL_NAMES, build_pipeline

SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"}


def majority_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the most common class."""
    return float(y.value_counts(normalize=True).iloc[0])


def compare_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV for every candidate model (training data only)."""
    min_class = int(y.value_counts().min())
    k = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in MODEL_NAMES:
        res = cross_validate(build_pipeline(name), X, y, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(scores))
            row[f"{metric}_std"] = float(np.std(scores))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def holdout_metrics(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = pipe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "n_test": len(y_test),
    }


def plot_confusion(pipe, X_test, y_test, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(pipe, X_test, y_test, title: str, path: Path) -> pd.DataFrame:
    """Permutation importance on raw input columns (macro-F1 drop when shuffled)."""
    res = permutation_importance(
        pipe, X_test, y_test, scoring="f1_macro", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    imp = (
        pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean,
                      "std": res.importances_std})
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color="#3b7ea1")
    ax.set_xlabel("Drop in macro-F1 when shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.sort_values("importance", ascending=False)
