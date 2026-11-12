import json

import pandas as pd
import pytest

from healthmania.config import FEATURES, TARGETS
from healthmania.data import clean, duplicate_report, load_raw
from healthmania.models import MODEL_NAMES, build_pipeline
from healthmania.predict import load_bundle, predict_records
from healthmania.train import run, stratified_split


def test_clean_splits_bp_and_fills_disorder(raw_df):
    df = clean(raw_df)
    assert {"SystolicBP", "DiastolicBP"} <= set(df.columns)
    assert "Blood Pressure" not in df.columns
    assert "Person ID" not in df.columns
    assert df["Sleep Disorder"].notna().all()
    assert "No Disorder" in set(df["Sleep Disorder"])


def test_clean_harmonises_bmi_spelling(raw_df):
    raw = raw_df.copy()
    raw.loc[raw.index[:5], "BMI Category"] = "Normal Weight"
    assert "Normal Weight" not in set(clean(raw)["BMI Category"])


def test_duplicates_are_detected_and_dropped(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:30]], ignore_index=True)
    doubled["Person ID"] = range(len(doubled))  # IDs differ, content does not
    assert duplicate_report(doubled)["n_duplicate_rows"] == 30
    assert len(clean(doubled, drop_duplicates=True)) == len(clean(raw_df))
    assert len(clean(doubled, drop_duplicates=False)) == len(doubled)


def test_load_raw_rejects_missing_columns(tmp_path, raw_df):
    path = tmp_path / "bad.csv"
    raw_df.drop(columns=["Heart Rate"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Heart Rate"):
        load_raw(path)


def test_split_has_no_row_overlap_after_dedup(raw_df):
    df = clean(raw_df)
    train, test = stratified_split(df)
    overlap = pd.merge(train[FEATURES + TARGETS], test[FEATURES + TARGETS], how="inner")
    assert len(overlap) == 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_fits_and_predicts(raw_df, name):
    df = clean(raw_df)
    pipe = build_pipeline(name).fit(df[FEATURES], df["Sleep Disorder"])
    assert len(pipe.predict(df[FEATURES])) == len(df)


def test_end_to_end_train_and_predict(raw_csv, tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    report = run(raw_csv, models, reports, keep_duplicates=False)
    assert set(report["targets"]) == set(TARGETS)
    assert json.loads((reports / "metrics.json").read_text())["n_train"] > 0
    assert (reports / "figures" / "confusion_sleep_disorder.png").exists()

    bundle = load_bundle(models / "healthmania_model.joblib")
    record = {
        "Gender": "Male", "Age": 40, "Occupation": "Engineer", "Sleep Duration": 6.5,
        "Quality of Sleep": 6, "Physical Activity Level": 55, "BMI Category": "Normal Weight",
        "Heart Rate": 72, "Daily Steps": 6000, "SystolicBP": 125, "DiastolicBP": 82,
    }
    unseen = dict(record, Occupation="Astronaut")  # unseen category must not crash
    out = predict_records(bundle, [record, unseen])
    assert len(out) == 2
    assert {"Predicted Stress Level", "Predicted Sleep Disorder"} <= set(out.columns)
