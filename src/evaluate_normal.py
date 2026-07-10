"""
The model is trained only on Marginal/Unacceptable rows. This script checks
how many Normal-status bearings get flagged (predicted faulty) by the trained
models — i.e. an approximate false-positive rate on healthy equipment.

Usage:
    python src/evaluate_normal.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.predict_utils import load_artifacts, predict_faults


def main(domain: str = "acceleration"):
    config.set_domain(domain)
    df = pd.read_csv(config.LABELED_CSV)
    normal_df = df[df["status"] == config.HOLDOUT_NORMAL_STATUS].reset_index(drop=True)
    print(f"Normal-status rows available: {len(normal_df)} / {len(df)} total")

    if normal_df.empty:
        print("No Normal-status rows found — nothing to evaluate.")
        return

    meta, encoders, medians, models = load_artifacts()
    result = predict_faults(normal_df, meta, encoders, medians, models)

    n = len(result)
    flagged = result[result["any_fault_pred"] == 1]
    print(f"\nNormal-status rows flagged as faulty by the model: {len(flagged)} / {n} "
          f"({len(flagged) / n:.2%})")

    for fault in models.keys():
        flagged_f = result[f"{fault}_pred"].sum()
        print(f"  {fault}: {flagged_f}/{n} flagged ({flagged_f / n:.2%})")

    per_bearing = (
        flagged.groupby("bearingLocationId")
        .size()
        .reset_index(name="flagged_packet_count")
        .sort_values("flagged_packet_count", ascending=False)
    )
    print(f"\nDistinct Normal bearings with >=1 flagged packet: {per_bearing['bearingLocationId'].nunique()}")

    result.to_csv(config.NORMAL_FLAG_REPORT_CSV, index=False)
    print(f"\nWrote full report -> {config.NORMAL_FLAG_REPORT_CSV}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    args = parser.parse_args()
    main(domain=args.domain)
