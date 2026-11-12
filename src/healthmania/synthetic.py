"""Tiny synthetic generator that mimics the raw schema.

FOR SMOKE TESTS AND DEMOS ONLY. Relationships are invented, so model scores
on this data say nothing about real-world performance.

    python -m healthmania.synthetic data/sample_synthetic.csv --rows 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OCCUPATIONS = ["Doctor", "Nurse", "Engineer", "Lawyer", "Teacher", "Accountant", "Salesperson"]


def make_synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quality = rng.integers(4, 10, n)
    duration = np.clip(rng.normal(5.5 + 0.35 * quality, 0.5), 5.5, 8.6).round(1)
    stress = np.clip(np.round(10 - quality + rng.normal(0, 0.8, n)), 3, 8).astype(int)
    bmi = rng.choice(["Normal", "Overweight", "Obese"], n, p=[0.55, 0.35, 0.10])
    sys_bp = (115 + 8 * (bmi != "Normal") + rng.integers(0, 15, n)).astype(int)
    dia_bp = (75 + 5 * (bmi != "Normal") + rng.integers(0, 10, n)).astype(int)

    risk = (bmi != "Normal").astype(int) + (duration < 6.5).astype(int) + (stress >= 7).astype(int)
    probs = np.where(risk[:, None] >= 2, [[0.25, 0.35, 0.40]], [[0.85, 0.08, 0.07]])
    disorder = np.array(
        [rng.choice(["None", "Insomnia", "Sleep Apnea"], p=p) for p in probs], dtype=object
    )
    disorder[disorder == "None"] = np.nan  # the source leaves absence blank

    return pd.DataFrame({
        "Person ID": np.arange(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(27, 60, n),
        "Occupation": rng.choice(OCCUPATIONS, n),
        "Sleep Duration": duration,
        "Quality of Sleep": quality,
        "Physical Activity Level": rng.integers(30, 91, n),
        "Stress Level": stress,
        "BMI Category": bmi,
        "Blood Pressure": [f"{s}/{d}" for s, d in zip(sys_bp, dia_bp)],
        "Heart Rate": rng.integers(65, 87, n),
        "Daily Steps": rng.integers(3000, 10001, n),
        "Sleep Disorder": disorder,
    })


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(args.rows, args.seed).to_csv(args.out, index=False)
    print(f"Wrote {args.rows} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
