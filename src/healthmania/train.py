"""Train, compare and evaluate models for both targets.

    python -m healthmania.train --data data/Sleep_Stress_data.csv
    python -m healthmania.train --data data/Sleep_Stress_data.csv --keep-duplicates   # 2024 conditions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from . import __version__
from .config import (
    BINARY_FEATURES,
    FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_STATE,
    TARGETS,
)
from .data import clean, duplicate_report, load_raw
from .evaluate import (
    compare_models,
    holdout_metrics,
    majority_baseline,
    plot_confusion,
    plot_permutation_importance,
)
from .models import build_pipeline


def stratified_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split once, stratifying on the joint (stress, disorder) label when possible."""
    key = df[TARGETS].astype(str).agg("|".join, axis=1)
    if key.value_counts().min() < 2:  # rare combinations: fall back to the sleep-disorder label
        key = df[TARGETS[1]]
    if key.value_counts().min() < 2:
        key = None
    return train_test_split(df, test_size=test_size, random_state=RANDOM_STATE, stratify=key)


def feature_summary(X: pd.DataFrame) -> dict:
    """Training-set ranges and category levels used by the dashboard to build inputs."""
    categorical = BINARY_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES
    return {
        "numeric": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
            for c in NUMERIC_FEATURES
        },
        "categorical": {c: sorted(X[c].unique().tolist()) for c in categorical},
    }


def run(data: Path, out_models: Path, out_reports: Path, keep_duplicates: bool) -> dict:
    out_models.mkdir(parents=True, exist_ok=True)
    fig_dir = out_reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data)
    dup = duplicate_report(raw)
    df = clean(raw, drop_duplicates=not keep_duplicates)
    train_df, test_df = stratified_split(df)
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]

    report: dict = {
        "version": __version__,
        "data_file": data.name,
        "duplicates_in_raw_file": dup,
        "duplicates_dropped": not keep_duplicates,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "targets": {},
    }
    bundle: dict = {"features": FEATURES, "models": {}, "summary": feature_summary(X_train)}

    for target in TARGETS:
        y_train, y_test = train_df[target], test_df[target]
        cv = compare_models(X_train, y_train)
        best = cv.loc[0, "model"]
        pipe = build_pipeline(best).fit(X_train, y_train)
        metrics = holdout_metrics(pipe, X_test, y_test)
        metrics["majority_baseline_accuracy"] = majority_baseline(y_test)

        slug = target.lower().replace(" ", "_")
        plot_confusion(pipe, X_test, y_test, f"{target}: {best} (hold-out)",
                       fig_dir / f"confusion_{slug}.png")
        imp = plot_permutation_importance(pipe, X_test, y_test, f"{target}: permutation importance",
                                          fig_dir / f"importance_{slug}.png")

        report["targets"][target] = {
            "selected_model": best,
            "cv_comparison": cv.round(4).to_dict(orient="records"),
            "holdout": metrics,
            "top_features": imp.head(5)[["feature", "importance"]].round(4).to_dict("records"),
        }
        bundle["models"][target] = pipe

    joblib.dump(bundle, out_models / "healthmania_model.joblib")
    (out_reports / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_reports / "metrics.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    d = report["duplicates_in_raw_file"]
    data_line = (
        f"- Data file: `{report['data_file']}` ({d['n_rows']} rows, "
        f"{d['n_duplicate_rows']} exact duplicates = {d['duplicate_fraction']:.1%})"
    )
    lines = [
        "# Latest training run",
        "",
        data_line,
        f"- Duplicates dropped before splitting: **{report['duplicates_dropped']}**",
        f"- Train / test rows: {report['n_train']} / {report['n_test']}",
        "",
    ]
    for target, t in report["targets"].items():
        h = t["holdout"]
        lines += [
            f"## {target}", "",
            f"Selected by cross-validated macro-F1: **{t['selected_model']}**", "",
            "| Model | Accuracy | Balanced acc. | Macro-F1 |", "|---|---|---|---|",
        ]
        for r in t["cv_comparison"]:
            lines.append(
                f"| {r['model']} | {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | "
                f"{r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_std']:.3f} | "
                f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} |"
            )
        holdout = (
            f"Hold-out (n={h['n_test']}): accuracy {h['accuracy']:.3f}, balanced accuracy "
            f"{h['balanced_accuracy']:.3f}, macro-F1 {h['macro_f1']:.3f} "
            f"(majority-class baseline accuracy {h['majority_baseline_accuracy']:.3f})."
        )
        lines += ["", holdout, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Keep exact duplicate rows (reproduces the 2024 notebook's conditions).")
    args = p.parse_args(argv)
    run(args.data, args.models_dir, args.reports_dir, args.keep_duplicates)
    print((args.reports_dir / "metrics.md").read_text(encoding="utf-8"))
    print(f"Saved model to {args.models_dir / 'healthmania_model.joblib'}")


if __name__ == "__main__":
    main()
