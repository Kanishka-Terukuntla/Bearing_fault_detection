"""
Sweeps a range of prediction thresholds and reports, per fault, BOTH:
    - recall on known-faulty rows (Marginal/Unacceptable) — detection power
    - flag rate on Normal rows — false-alarm rate

Run this once instead of testing thresholds one at a time. Pick the row per
fault where the trade-off looks acceptable for your use case, then set
PREDICTION_THRESHOLD in config.py accordingly (thresholds can also be set
per-fault if you want — see the note at the bottom of the output).

Usage (from the project root):
    python threshold_sweep.py
    python threshold_sweep.py --domain velocity
"""
import argparse

import pandas as pd

import config
from src.predict_utils import load_artifacts, prepare_features

THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def main(domain: str = "acceleration"):
    config.set_domain(domain)
    df = pd.read_csv(config.LABELED_CSV)
    faulty = df[df["status"].isin(config.TRAIN_STATUSES)].reset_index(drop=True)
    normal = df[df["status"] == config.HOLDOUT_NORMAL_STATUS].reset_index(drop=True)

    meta, encoders, medians, models = load_artifacts()

    X_faulty = prepare_features(faulty, meta, encoders, medians)
    X_normal = prepare_features(normal, meta, encoders, medians)

    # compute probabilities ONCE per fault, then just re-threshold in the loop (fast)
    faulty_probs = {f: m.predict_proba(X_faulty)[:, 1] for f, m in models.items()}
    normal_probs = {f: m.predict_proba(X_normal)[:, 1] for f, m in models.items()}

    for fault in models.keys():
        print(f"\n{'='*70}\n{fault}\n{'='*70}")
        print(f"{'threshold':>10} | {'recall (known-faulty)':>22} | {'flag rate (normal)':>20}")
        print("-" * 60)

        y_true_faulty = faulty[f"{fault}_label"].values
        fprobs = faulty_probs[fault]
        nprobs = normal_probs[fault]

        for t in THRESHOLDS:
            preds_faulty = (fprobs >= t).astype(int)
            true_pos = ((y_true_faulty == 1) & (preds_faulty == 1)).sum()
            actual_pos = (y_true_faulty == 1).sum()
            recall = true_pos / actual_pos if actual_pos else 0.0

            preds_normal = (nprobs >= t).astype(int)
            flag_rate = preds_normal.mean() if len(preds_normal) else 0.0

            print(f"{t:>10.2f} | {recall:>21.2%} | {flag_rate:>19.2%}")

    print(
        "\nNote: thresholds can differ per fault. Once you pick a value per fault, "
        "you can either set one global config.PREDICTION_THRESHOLD (simplest), or "
        "extend predict_utils.predict_faults() to accept a dict of {fault: threshold} "
        "if the faults clearly need different cutoffs (ask me and I'll wire it up)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    args = parser.parse_args()
    main(domain=args.domain)
