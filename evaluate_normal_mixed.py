"""
Evaluates the MIXED-domain production system: each fault routed to whichever
domain (acceleration/velocity) wins it per config.FAULT_DOMAIN, each using its
own tuned config.PREDICTION_THRESHOLDS value.

Requires both domains to already be fully built:
    python -m src.build_dataset            && python -m src.labeling            && python -m src.train
    python -m src.build_dataset --domain velocity && python -m src.labeling --domain velocity && python -m src.train --domain velocity

Usage (from the project root):
    python evaluate_normal_mixed.py
"""
import pandas as pd

import config
from src.predict_utils import load_mixed_artifacts, predict_faults_mixed


def main():
    config.set_domain("acceleration")
    accel_labeled = pd.read_csv(config.LABELED_CSV)
    config.set_domain("velocity")
    vel_labeled = pd.read_csv(config.LABELED_CSV)
    config.set_domain("acceleration")  # restore

    accel_normal = accel_labeled[accel_labeled["status"] == config.HOLDOUT_NORMAL_STATUS].reset_index(drop=True)
    vel_normal = vel_labeled[vel_labeled["status"] == config.HOLDOUT_NORMAL_STATUS].reset_index(drop=True)
    print(f"Normal-status rows: acceleration={len(accel_normal)}, velocity={len(vel_normal)}")

    print(f"\nFault -> domain routing: {config.FAULT_DOMAIN}")
    print(f"Thresholds: {config.PREDICTION_THRESHOLDS}\n")

    per_fault_artifacts = load_mixed_artifacts()
    result = predict_faults_mixed(
        {"acceleration": accel_normal, "velocity": vel_normal},
        per_fault_artifacts,
    )

    n = len(result)
    flagged = result[result["any_fault_pred"] == 1]
    print(f"Normal-status rows flagged by the MIXED system: {len(flagged)} / {n} ({len(flagged)/n:.2%})")

    for fault in per_fault_artifacts.keys():
        col = f"{fault}_pred"
        if col in result.columns:
            rate = result[col].mean()
            domain = per_fault_artifacts[fault]["domain"]
            print(f"  {fault} (via {domain}): {int(result[col].sum())}/{n} flagged ({rate:.2%})")

    print(f"\nDistinct Normal bearings with >=1 flagged packet: {flagged['bearingLocationId'].nunique()}")

    out_path = config.OUTPUTS_DIR / "normal_flag_report_mixed.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote full report -> {out_path}")


if __name__ == "__main__":
    main()
