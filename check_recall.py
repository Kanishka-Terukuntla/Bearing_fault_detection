"""
Checks what config.PREDICTION_THRESHOLD costs you on the detection side.

Loads the labeled dataset, restricts to bearings we KNOW are faulty
(status in Marginal/Unacceptable), runs the trained models on them, and
compares:
    {fault}_label  -> ground truth (from the 0.70 LABELING threshold)
    {fault}_pred   -> model's prediction (using config.PREDICTION_THRESHOLD)

Prints recall per fault: of the rows truly labeled faulty, how many did the
model still catch at the current prediction threshold.

Usage (from the project root):
    python check_recall.py                     # acceleration models (default)
    python check_recall.py --domain velocity   # velocity models
"""
import argparse

import pandas as pd

import config
from src.predict_utils import load_artifacts, predict_faults


def main(domain: str = "acceleration"):
    config.set_domain(domain)
    df = pd.read_csv(config.LABELED_CSV)
    faulty = df[df["status"].isin(config.TRAIN_STATUSES)].reset_index(drop=True)
    print(f"Known-faulty rows (status in {config.TRAIN_STATUSES}): {len(faulty)}")

    meta, encoders, medians, models = load_artifacts()
    result = predict_faults(faulty, meta, encoders, medians, models)  # uses config.PREDICTION_THRESHOLD

    print(f"\nPrediction threshold in use: {config.PREDICTION_THRESHOLD}\n")
    for fault in models.keys():
        label_col = f"{fault}_label"
        pred_col = f"{fault}_pred"

        true_pos = ((result[label_col] == 1) & (result[pred_col] == 1)).sum()
        false_neg = ((result[label_col] == 1) & (result[pred_col] == 0)).sum()
        actual_pos = (result[label_col] == 1).sum()
        recall = true_pos / actual_pos if actual_pos else 0.0

        print(f"{fault}: recall = {recall:.2%}  "
              f"(caught {true_pos}/{actual_pos} known-faulty rows, missed {false_neg})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    args = parser.parse_args()
    main(domain=args.domain)
