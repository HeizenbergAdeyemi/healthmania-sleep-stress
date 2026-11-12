# Audit of the 2024 notebook

The original notebook (`notebooks/01_original_sleep_stress_svm_2024.ipynb`) is kept
unmodified. This note records what a careful re-read found and how the package addresses
each point. Findings come from the notebook's code and its saved outputs.

## What worked

- Clear structure: sourcing, cleaning, encoding, split, model, evaluation, prediction.
- A sensible multi-output setup (`MultiOutputClassifier` over a scaled linear SVM).
- Sleep disorder blanks handled correctly as "no disorder"; blood pressure parsed into
  numeric systolic/diastolic values.

## Issues and fixes

| # | Finding (evidence) | Why it matters | Fix in this repo |
|---|---|---|---|
| 1 | The 1,374-row file was made by `augment_data`, which resamples original rows 1,000 times. Only `Sleep Duration` (a float) gets noise, because `isinstance(x, (int, float))` is false for `int64`. | Every added row is a near-copy of an original, and the random split puts copies on both sides, so validation accuracy measures memorisation. | Train and evaluate on the unaugmented 374-row file; deduplicate before splitting. |
| 2 | `LabelEncoder` was applied to numeric columns (`Age`, `Heart Rate`, `Daily Steps` and others). | Real values became ranks (`Age = 0, 1, ...`), losing units and spacing. | Raw units kept; scaling happens inside the `Pipeline`. |
| 3 | The final `predict_stress_and_sleep_disorder` calls `scaler.fit_transform` on the input batch. | The scaler is re-fitted on whatever is passed in, so predictions depend on the batch and not on the training distribution. | Scaling lives in a `Pipeline` fitted once on training data. |
| 4 | The example call returns a list of two tuples, unpacked into `predicted_stress_level, predicted_sleep_disorder`. The printed output (`('Low', 'No Disorder')` twice) is therefore two rows labelled as two fields. | Misleading output. | `predict_records()` returns a labelled DataFrame. |
| 5 | The "Low/High" stress label uses an arbitrary cut at class 5. | Not grounded in the data's scale. | The model's own class is reported, on the dataset's scale. |
| 6 | The test set is created but never scored (only one row is inspected). | Reported numbers are validation scores used during development. | Single hold-out, evaluated once after CV-based model selection. |
| 7 | Occupation is mapped to integers 1-11. | A linear model reads that as an ordering and spacing that does not exist. | One-hot encoding with unseen-category handling. |
| 8 | Markdown claims mapping preserves order for "Sleep Disorder"; it is a nominal target. | Conceptual error (harmless for labels, misleading in text). | Targets kept as class labels; only BMI is ordinal. |
| 9 | Only accuracy is reported. Sleep disorder is imbalanced (816 / 287 / 271; the majority class is about 59%). | Accuracy hides weak minority-class performance. | Balanced accuracy, macro-F1, majority baseline, confusion matrices. |
| 10 | Stress accuracy of 99.0% (2 errors in 206). | Suspiciously high for a self-reported construct; may be duplicates, a near-deterministic label, or both. | Compare runs with and without duplicates; inspect permutation importance. |
| 11 | `df['col'].fillna(..., inplace=True)` is chained assignment. | Breaks or warns under newer pandas (copy-on-write). | Direct assignment. |
| 12 | Actual-vs-predicted scatter plots of discrete labels overplot. | Hard to read. | Confusion-matrix plots. |

## Not changed

The scientific question, the target definitions and the choice to predict both outcomes
from the same features. Whether stress level should be treated as ordinal rather than as
six nominal classes is left as a roadmap item.
