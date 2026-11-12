"""Healthmania dashboard: explore a trained model and score a single profile.

    streamlit run app/streamlit_app.py

Research/education demo only. Not a medical device and not for clinical decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from healthmania.predict import load_bundle, predict_records

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "healthmania_model.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Healthmania", page_icon="😴", layout="wide")
st.title("Healthmania: sleep, stress and vitals")
st.caption(
    "Educational demo of a tabular health-data model. Not a medical device; "
    "predictions must not be used for diagnosis or treatment."
)

if not MODEL_PATH.exists():
    st.warning(
        "No trained model found. Run `python -m healthmania.train --data <csv>` "
        "(or `make demo` for the synthetic sample) first."
    )
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


bundle = get_bundle()
summary = bundle["summary"]
tab_predict, tab_perf = st.tabs(["Score a profile", "Model performance"])

with tab_predict:
    st.write("Inputs adapt to the ranges seen in the training data.")
    cols = st.columns(3)
    record: dict = {}
    cat = summary["categorical"]
    with cols[0]:
        record["Gender"] = st.selectbox("Gender", cat["Gender"])
        record["Occupation"] = st.selectbox("Occupation", cat["Occupation"])
        record["BMI Category"] = st.selectbox("BMI category", cat["BMI Category"])
    numeric_order = list(summary["numeric"])
    for i, name in enumerate(numeric_order):
        rng = summary["numeric"][name]
        lo, hi, med = rng["min"], rng["max"], rng["median"]
        is_int = float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer()
        with cols[(i + 1) % 3]:
            if hi <= lo:
                record[name] = lo
                st.number_input(name, value=float(lo), disabled=True)
            elif is_int:
                record[name] = st.slider(name, int(lo), int(hi), int(med))
            else:
                record[name] = st.slider(name, float(lo), float(hi), float(med), step=0.1)

    if st.button("Predict", type="primary"):
        result = predict_records(bundle, [record]).iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Predicted stress level", str(result["Predicted Stress Level"]))
        c2.metric("Predicted sleep disorder", str(result["Predicted Sleep Disorder"]))
        st.caption(
            "Stress level is on the dataset's own scale. Confidence values, where shown below, "
            "are raw model probabilities and are not calibrated."
        )
        st.dataframe(result.astype(str).to_frame("value"))

with tab_perf:
    if not REPORT_PATH.exists():
        st.info("No metrics report found. Run the training script to generate one.")
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        dup = report["duplicates_in_raw_file"]
        st.write(
            f"Data: `{report['data_file']}`, {dup['n_rows']} rows, "
            f"{dup['n_duplicate_rows']} exact duplicates "
            f"({dup['duplicate_fraction']:.1%}); duplicates dropped: "
            f"**{report['duplicates_dropped']}**."
        )
        for target, t in report["targets"].items():
            st.subheader(target)
            h = t["holdout"]
            m = st.columns(4)
            m[0].metric("Model", t["selected_model"])
            m[1].metric("Accuracy", f"{h['accuracy']:.3f}")
            m[2].metric("Balanced acc.", f"{h['balanced_accuracy']:.3f}")
            m[3].metric("Macro-F1", f"{h['macro_f1']:.3f}")
            st.caption(f"Majority-class baseline accuracy: {h['majority_baseline_accuracy']:.3f}")
            st.dataframe(pd.DataFrame(t["cv_comparison"]).round(3), hide_index=True)
            slug = target.lower().replace(" ", "_")
            f1, f2 = st.columns(2)
            for col, kind in ((f1, "confusion"), (f2, "importance")):
                img = FIG_DIR / f"{kind}_{slug}.png"
                if img.exists():
                    col.image(str(img))
