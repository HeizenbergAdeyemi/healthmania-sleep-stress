# Healthmania: Predicting Stress and Sleep Disorder from Lifestyle and Vitals

A reproducible machine-learning workflow on tabular health data. Given lifestyle
measures (sleep duration and quality, physical activity, daily steps) and
physiological signals (heart rate, systolic/diastolic blood pressure, BMI category),
the models predict two outcomes at once: **stress level** and **sleep disorder**
(none, insomnia, sleep apnea).

The project began in 2024 as the modelling core of **Healthmania**, a health-monitoring
dashboard. This repository is the cleaned-up, reproducible version: the original notebook
is preserved, and the pipeline has been rebuilt with leakage-safe preprocessing,
cross-validated model comparison, class-imbalance-aware metrics, a tested package and a
small Streamlit dashboard.

> **Not a medical device.** This is a research and education project. Predictions must not
> be used for diagnosis, treatment or any clinical decision.

## Why this project

Wearables and self-reported health data are small, noisy, imbalanced and easy to
over-interpret. This repo is a worked example of doing that carefully:

- treating a health prediction task as a measurement problem (what does each variable
  actually capture, and is the label learnable beyond duplicates?)
- evaluating with metrics that respect class imbalance, against a majority-class baseline
- auditing the earlier analysis and documenting what changed and why
  ([docs/audit_of_2024_notebook.md](docs/audit_of_2024_notebook.md))

## Repository layout

```
.
├── src/healthmania/        # installable package
│   ├── config.py           # schema, feature groups, encodings
│   ├── data.py             # load, validate, clean, duplicate report
│   ├── models.py           # preprocessing ColumnTransformer + model zoo
│   ├── evaluate.py         # CV comparison, hold-out metrics, plots
│   ├── train.py            # CLI: train, compare, evaluate, save
│   ├── predict.py          # inference on raw human-readable records
│   └── synthetic.py        # tiny synthetic generator for demos/tests
├── app/streamlit_app.py    # dashboard: score a profile, view model performance
├── notebooks/              # original 2024 notebook, unmodified
├── docs/                   # audit of the 2024 notebook, model card
├── tests/                  # pytest suite (cleaning, splits, models, end to end)
├── data/                   # dataset instructions + synthetic sample (no real data committed)
└── .github/workflows/      # CI: lint + tests
```

## Quickstart

```bash
git clone <your-repo-url> && cd healthmania-sleep-stress
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Try the full pipeline on the bundled synthetic sample (scores are meaningless)
make demo

# 2. Or train on the real data (see data/README.md for where to get it)
python -m healthmania.train --data data/Sleep_Stress_data.csv

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

Training writes `models/healthmania_model.joblib`, `reports/metrics.json`,
`reports/metrics.md` and figures in `reports/figures/`.

The 2024 file is not included here. To see how near-copies inflate scores, train on your own
copy of it, writing to scratch folders so nothing is overwritten:

    python -m healthmania.train --data "data/Sleep_Stress data.csv" --models-dir scratch_models --reports-dir scratch_reports

## Method

| Step | 2024 notebook | This repo |
|---|---|---|
| Missing sleep disorder | filled with "No Disorder" | same |
| Blood pressure | split into systolic/diastolic | same |
| Duplicates and augmentation | 1,000 resampled near-copies added, then random split | public 374-row file; exact duplicates reported and dropped before splitting |
| Occupation encoding | arbitrary integers (1-11) | one-hot (no false ordering) |
| BMI encoding | ordinal | ordinal (Underweight < Normal < Overweight < Obese) |
| Scaling | `StandardScaler` fit before inference, re-fit on new inputs at prediction time | fitted on training data only, inside a `Pipeline` |
| Split | random 70/15/15, test set never scored | stratified 80/20 hold-out; model selection by 5-fold CV on the training part only |
| Models | linear SVM | linear SVM (baseline), logistic regression, random forest, gradient boosting |
| Metrics | accuracy | accuracy, balanced accuracy, macro-F1, majority-class baseline |
| Interpretation | none | permutation importance, confusion matrices |

## Results

**Original 2024 notebook** (saved outputs, linear SVM, random split, duplicates kept,
validation n = 206):

| Target | Train accuracy | Validation accuracy |
|---|---|---|
| Stress level (6 classes) | 99.7% | 99.0% |
| Sleep disorder (3 classes) | 91.8% | 94.7% |

For reference, always predicting "No Disorder" gives about 59% accuracy on the full
1,374-row file (816 of 1,374 rows), and the largest stress class is about 20%.

That 1,374-row file was produced by resampling the 374 public rows (see
`notebooks/00_augmentation_2024.ipynb`), so those scores measure memorisation of near-copies.

**This repository's pipeline** (public 374-row file; 242 exact duplicates removed,
leaving 132 unique records; stratified 80/20 split, 5-fold CV on the training part):

| Target | Best model (CV) | CV accuracy | CV balanced acc. | CV macro-F1 | Hold-out accuracy (n=27) | Majority baseline |
|---|---|---|---|---|---|---|
| Stress level | Linear SVM | 0.886 ± 0.071 | 0.892 ± 0.070 | 0.887 ± 0.076 | 0.778 | 0.296 |
| Sleep disorder | Linear SVM | 0.752 ± 0.056 | 0.694 ± 0.094 | 0.699 ± 0.084 | 0.704 | 0.556 |

With so few unique records the hold-out is noisy (one error moves accuracy by about
3.7 points), so the cross-validated figures are the main evidence. The differences between
the top models are within one standard deviation, except random forest on sleep disorder,
which is clearly worse. The 2024 scores above came from an augmented file of resampled
near-copies, so they are not comparable to these.

## Limitations

- **Small, weakly documented data.** The public Kaggle dataset has a few hundred rows and
  does not document how it was collected. Results may not transfer to real populations.
- **Cross-sectional, self-reported measures.** Associations between sleep, stress and
  vitals here are not causal.
- **Possible label leakage.** Near-perfect stress accuracy suggests the label may be almost
  a function of the other columns (or of duplicated rows). The importance plots and the
  duplicate comparison are there to probe this.
- **No external validation** and no calibration analysis yet.
- Full details: [docs/model_card.md](docs/model_card.md).

## Roadmap

- [ ] Calibration curves and decision-threshold analysis
- [ ] SHAP or partial-dependence explanations
- [ ] Ordinal models for stress level
- [ ] Validate on a second sleep/wearable dataset
- [ ] Wearable time-series features (HRV, sleep stages) instead of daily summaries
- [ ] Experiment tracking (MLflow) and a Dockerfile

## Data and credits

Data: *Sleep Health and Lifestyle Dataset* on Kaggle
(<https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset>).
The dataset is not redistributed here; see its page for licence and terms.

Originally developed in 2024 as part of the Healthmania project. Add collaborators here.

Licensed under the MIT License.
