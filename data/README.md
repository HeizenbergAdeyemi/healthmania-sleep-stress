# Data

Real data is **not committed** to this repository.

## Getting the dataset

1. Download *Sleep Health and Lifestyle Dataset* from Kaggle:
   <https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset>
2. Save the CSV in this folder, e.g. `data/Sleep_Stress_data.csv`.

## Expected columns

`Gender`, `Age`, `Occupation`, `Sleep Duration`, `Quality of Sleep`,
`Physical Activity Level`, `Stress Level`, `BMI Category`, `Blood Pressure` (as `"120/80"`),
`Heart Rate`, `Daily Steps`, `Sleep Disorder` (blank = no disorder).
`Person ID` and `Unnamed: 0` are dropped automatically if present.

## About the 2024 file

The 2024 notebook trained on a 1,374-row file produced by resampling the 374 public rows
(`notebooks/00_augmentation_2024.ipynb`) and rank-encoding several numeric columns. It adds
no new information and leaks across splits, so do not use it for evaluation. The public file
itself has 242 exact duplicate rows (ignoring `Person ID`).

## Synthetic sample

`sample_synthetic.csv` is randomly generated (`python -m healthmania.synthetic`) so the
pipeline and dashboard can be demonstrated without real data. Its relationships are invented.
