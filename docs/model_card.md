# Model card: Healthmania stress and sleep-disorder classifiers

## Intended use
Educational and portfolio demonstration of a tabular health-data workflow. Not intended
for clinical use, screening, insurance, employment or any decision about a person.

## Models
Two separate classifiers share one preprocessing pipeline, one per target. For each target
the best of four candidates (linear SVM, logistic regression, random forest, histogram
gradient boosting) is chosen by 5-fold stratified cross-validated macro-F1 on the training
split, refitted, and scored once on a hold-out set.

## Data
Sleep Health and Lifestyle Dataset (Kaggle), a small tabular file of adults with sleep,
activity, BMI category, heart rate, daily steps and blood pressure. Collection method,
consent and population are not documented. Any file derived from it should record how it
was altered (see `data/README.md`).

## Inputs and outputs
- Inputs: gender, age, occupation, sleep duration, sleep quality, physical activity level,
  BMI category, heart rate, daily steps, systolic and diastolic blood pressure.
- Outputs: stress level (on the dataset's scale) and sleep disorder (none, insomnia, sleep apnea).

## Evaluation
Accuracy, balanced accuracy and macro-F1, each compared with a majority-class baseline.
Generated per run in `reports/metrics.md`. Permutation importance shows which raw inputs
drive each prediction on the hold-out set.

## Known limitations
- Small sample; wide uncertainty on hold-out metrics (see the CV standard deviations).
- Possible duplicate and label leakage; re-run with and without `--keep-duplicates`.
- Occupation categories are sparse (some have a handful of records).
- Self-reported stress and sleep quality are subjective and cross-sectional.
- No external validation, calibration or temporal validation.

## Fairness and ethics
Gender and occupation are model inputs and no subgroup performance analysis has been done.
Sleep disorders are diagnosed clinically; a model of this kind cannot replace assessment.
Predicted probabilities, where shown, are uncalibrated.
