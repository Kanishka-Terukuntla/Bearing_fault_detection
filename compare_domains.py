"""
Puts acceleration-domain and velocity-domain results side by side:
  - CV summary (F1/precision/recall/AUC per fault, best model)
  - Normal-bearing flag rate per fault

Requires both domains to have already been built/labeled/trained:
    python -m src.build_dataset                     && \\
    python -m src.labeling                           && \\
    python -m src.train                              && \\
    python -m src.evaluate_normal

    python -m src.build_dataset --domain velocity    && \\
    python -m src.labeling --domain velocity         && \\
    python -m src.train --domain velocity            && \\
    python -m src.evaluate_normal --domain velocity

Usage (from the project root):
    python compare_domains.py
"""
import pandas as pd

import config


def load_domain_results(domain: str):
    config.set_domain(domain)
    cv_path = config.CV_SUMMARY_CSV
    flag_path = config.NORMAL_FLAG_REPORT_CSV

    if not cv_path.exists():
        print(f"[{domain}] No CV summary found at {cv_path} — run the pipeline for this domain first.")
        return None, None

    try:
        cv = pd.read_csv(cv_path)
    except pd.errors.EmptyDataError:
        print(f"[{domain}] {cv_path} is empty — no faults had enough positives to train. Skipping.")
        return None, None

    if cv.empty:
        print(f"[{domain}] {cv_path} has no rows — no faults had enough positives to train. Skipping.")
        return None, None

    best_per_fault = cv.loc[cv.groupby("fault")["f1_mean"].idxmax()].copy()
    best_per_fault["domain"] = domain

    flag_summary = None
    if flag_path.exists():
        flagged = pd.read_csv(flag_path)
        rows = []
        for fault in config.FAULT_COLS:
            col = f"{fault}_pred"
            if col in flagged.columns:
                rows.append({"domain": domain, "fault": fault, "normal_flag_rate": flagged[col].mean()})
        flag_summary = pd.DataFrame(rows)
    else:
        print(f"[{domain}] No normal-flag report found at {flag_path} — run evaluate_normal for this domain.")

    return best_per_fault, flag_summary


def main():
    accel_cv, accel_flag = load_domain_results("acceleration")
    vel_cv, vel_flag = load_domain_results("velocity")

    print("\n" + "=" * 70)
    print("BEST MODEL PER FAULT — ACCELERATION vs VELOCITY")
    print("=" * 70)
    cv_parts = [d for d in (accel_cv, vel_cv) if d is not None]
    if cv_parts:
        combined_cv = pd.concat(cv_parts, ignore_index=True)
        combined_cv = combined_cv[["fault", "domain", "model", "f1_mean", "f1_std",
                                    "precision_mean", "recall_mean", "auc_mean"]]
        combined_cv = combined_cv.sort_values(["fault", "domain"])
        print(combined_cv.to_string(index=False))
    else:
        print("No CV results available for either domain yet.")

    print("\n" + "=" * 70)
    print("NORMAL-BEARING FLAG RATE — ACCELERATION vs VELOCITY")
    print("=" * 70)
    flag_parts = [d for d in (accel_flag, vel_flag) if d is not None]
    if flag_parts:
        combined_flag = pd.concat(flag_parts, ignore_index=True)
        pivot = combined_flag.pivot(index="fault", columns="domain", values="normal_flag_rate")
        print(pivot.to_string())
    else:
        print("No normal-flag reports available for either domain yet.")

    print(
        "\nHow to read this: for each fault, compare f1_mean (higher = better overall) "
        "and normal_flag_rate (lower = fewer false alarms on healthy bearings) between "
        "the two domains. Velocity is the ISO-10816-standard measurement for overall "
        "vibration severity, so it's plausible it does better/worse than acceleration "
        "depending on the fault type and frequency range involved — there's no "
        "assumption baked in here about which one should win."
    )


if __name__ == "__main__":
    main()
