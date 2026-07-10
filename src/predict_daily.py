"""
Daily live prediction pipeline. Reuses the EXACT SAME code as your historical
training pipeline (build_dataset.py, labeling.py) instead of a separate
in-memory feature-extraction path. Steps:

  1. Login to the AAMS API (pooled connection, see api_client.py).
  2. Fetch today's (or a chosen date's) raw waveforms + bearing catalogue in
     ONE paginated walk (all measuring types), in parallel across (bearing,
     axis) pairs, and SAVE them to disk at data/live/{date}/ in the exact
     same schema as your historical data/raw/ folder.
  3. Merge the live catalogue with your local catalogue.json to fill in
     bearing geometry (innerRacePass/outerRacePass/rollElementPass/
     cageRotation) that the live API often doesn't include.
  4. Run src.build_dataset.build() on that saved data — same function used
     for historical data.
  5. Run src.labeling.compute_prediction_features() — same function used at
     training time — to add shaft_frequency, fault frequencies, envelope_amp.
  6. Score with the trained models and flag bearings predicted faulty.
  7. Append results to outputs/daily_predictions*.csv (history) and write
     outputs/daily_predictions_latest*.csv (today's run only).

Default mode is VELOCITY-ONLY — only the velocity domain is built and scored
(the acceleration domain isn't built at all, saving real time since feature
extraction runs on every packet). Mixed and single-domain-acceleration modes
are still available if you need them later.

Usage:
    python -m src.predict_daily                          # velocity only (default), today (UTC)
    python -m src.predict_daily --date 2026-07-06
    python -m src.predict_daily --workers 40 --deadline 3600
    python -m src.predict_daily --mode mixed              # BPFI/BPFO/BSF via accel, FTF via velocity
    python -m src.predict_daily --mode single --domain acceleration
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.build_dataset import build as build_dataset
from src.fetch_live_data import build_merged_catalogue, save_live_waveforms
from src.labeling import compute_prediction_features
from src.predict_utils import load_artifacts, load_mixed_artifacts, predict_faults, predict_faults_mixed


def fetch_and_build_today(target_date: str, domains=("velocity",),
                           max_workers: int = 40, max_total_seconds: int = 1800):
    """
    Fetches + saves today's raw data ONCE, merges the catalogue, then runs
    build_dataset.build() only for the requested domain(s). Returns a dict
    {domain: df}, each with derived features (shaft_frequency, fault
    frequencies, envelope_amp) already computed.

    Only building the domain(s) you actually need (default: just velocity)
    roughly halves feature-extraction time vs. always building both.
    """
    live_bearings = save_live_waveforms(target_date, config.LIVE_DATA_ROOT,
                                         max_workers=max_workers, max_total_seconds=max_total_seconds)

    day_dir = config.LIVE_DATA_ROOT / target_date
    merged_catalogue_path = day_dir / "catalogue.json"
    build_merged_catalogue(live_bearings, config.CATALOGUE_PATH, merged_catalogue_path)

    dfs = {}
    for domain in domains:
        print(f"\nBuilding {domain}-domain features (reusing build_dataset.build)...")
        out_csv = day_dir / ("waveform.csv" if domain == "acceleration" else "waveform_velocity.csv")
        df = build_dataset(raw_root=day_dir, catalogue_path=merged_catalogue_path,
                            out_csv=out_csv, domain=domain)
        dfs[domain] = compute_prediction_features(df)

    return dfs


def run_velocity(target_date: str, max_workers: int = 40, max_total_seconds: int = 1800):
    """The fast default path: fetch once, build+score velocity only."""
    dfs = fetch_and_build_today(target_date, domains=("velocity",),
                                 max_workers=max_workers, max_total_seconds=max_total_seconds)
    vel_df = dfs["velocity"]
    config.set_domain("velocity")
    print(f"\nExtracted features for {len(vel_df)} packets across {vel_df['bearingLocationId'].nunique()} bearings")

    meta, encoders, medians, models = load_artifacts()
    result = predict_faults(vel_df, meta, encoders, medians, models)
    result["run_date"] = target_date
    result["domain"] = "velocity"

    latest_path = config.OUTPUTS_DIR / "daily_predictions_latest_velocity.csv"
    result.to_csv(latest_path, index=False)
    print(f"Wrote latest run -> {latest_path}")

    history_path = config.OUTPUTS_DIR / "daily_predictions_velocity.csv"
    if history_path.exists():
        history = pd.concat([pd.read_csv(history_path), result], ignore_index=True)
    else:
        history = result
    history.to_csv(history_path, index=False)
    print(f"Appended to history -> {history_path}")

    _print_summary(result, target_date)
    return result


def run_mixed(target_date: str, max_workers: int = 40, max_total_seconds: int = 1800):
    dfs = fetch_and_build_today(target_date, domains=("acceleration", "velocity"),
                                 max_workers=max_workers, max_total_seconds=max_total_seconds)
    accel_df, vel_df = dfs["acceleration"], dfs["velocity"]
    print(f"\nExtracted features for {len(accel_df)} packets across "
          f"{accel_df['bearingLocationId'].nunique()} bearings (both domains)")

    per_fault_artifacts = load_mixed_artifacts()
    result = predict_faults_mixed({"acceleration": accel_df, "velocity": vel_df}, per_fault_artifacts)

    meta_cols = ["bearingLocationId", "date", "axis", "packet_number", "status",
                 "machineName", "customerName", "geometry_known"]
    meta_cols = [c for c in meta_cols if c in accel_df.columns]
    result = result.merge(
        accel_df[meta_cols].drop_duplicates(subset=["bearingLocationId", "date", "axis", "packet_number"]),
        on=["bearingLocationId", "date", "axis", "packet_number"], how="left",
    )
    result["run_date"] = target_date
    result["domain"] = "mixed"

    latest_path = config.OUTPUTS_DIR / "daily_predictions_latest_mixed.csv"
    result.to_csv(latest_path, index=False)
    print(f"Wrote latest run -> {latest_path}")

    history_path = config.OUTPUTS_DIR / "daily_predictions_mixed.csv"
    if history_path.exists():
        history = pd.concat([pd.read_csv(history_path), result], ignore_index=True)
    else:
        history = result
    history.to_csv(history_path, index=False)
    print(f"Appended to history -> {history_path}")

    _print_summary(result, target_date)
    return result


def run_single_domain(target_date: str, domain: str, max_workers: int = 40, max_total_seconds: int = 1800):
    dfs = fetch_and_build_today(target_date, domains=(domain,),
                                 max_workers=max_workers, max_total_seconds=max_total_seconds)
    df = dfs[domain]
    config.set_domain(domain)
    print(f"\nExtracted features for {len(df)} packets across {df['bearingLocationId'].nunique()} bearings")

    meta, encoders, medians, models = load_artifacts()
    result = predict_faults(df, meta, encoders, medians, models)
    result["run_date"] = target_date
    result["domain"] = domain

    latest_path = config.OUTPUTS_DIR / "daily_predictions_latest.csv"
    result.to_csv(latest_path, index=False)
    print(f"Wrote latest run -> {latest_path}")

    if config.DAILY_PREDICTIONS_CSV.exists():
        history = pd.concat([pd.read_csv(config.DAILY_PREDICTIONS_CSV), result], ignore_index=True)
    else:
        history = result
    history.to_csv(config.DAILY_PREDICTIONS_CSV, index=False)
    print(f"Appended to history -> {config.DAILY_PREDICTIONS_CSV}")

    _print_summary(result, target_date)
    return result


def _print_summary(result: pd.DataFrame, target_date: str):
    flagged = result[result["any_fault_pred"] == 1]
    print(f"\n{target_date}: {flagged['bearingLocationId'].nunique()} bearings flagged "
          f"out of {result['bearingLocationId'].nunique()} total")

    if "status" in result.columns:
        normal_flagged = result[(result["status"] == config.HOLDOUT_NORMAL_STATUS) & (result["any_fault_pred"] == 1)]
        normal_all = result[result["status"] == config.HOLDOUT_NORMAL_STATUS]
        if len(normal_all):
            n_normal = normal_all["bearingLocationId"].nunique()
            n_normal_flagged = normal_flagged["bearingLocationId"].nunique()
            print(f"Of the Normal-status bearings: {n_normal_flagged}/{n_normal} flagged today")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (UTC). Defaults to today.")
    parser.add_argument("--mode", choices=["velocity", "mixed", "single"], default="velocity",
                         help="'velocity' (default): fast path, velocity model only. "
                              "'mixed': BPFI/BPFO/BSF via acceleration, FTF via velocity. "
                              "'single': one domain's models for everything (use --domain).")
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration",
                         help="Only used when --mode single.")
    parser.add_argument("--workers", type=int, default=40,
                         help="Parallel API requests for fetching raw waveforms (default: 40).")
    parser.add_argument("--deadline", type=int, default=1800,
                         help="Max seconds to wait for all fetches before moving on with whatever "
                              "completed (default: 1800 = 30 min). Use 0 for no deadline.")
    args = parser.parse_args()
    target_date = args.date or time.strftime("%Y-%m-%d", time.gmtime())
    deadline = args.deadline if args.deadline > 0 else None

    if args.mode == "velocity":
        run_velocity(target_date, args.workers, deadline)
    elif args.mode == "mixed":
        run_mixed(target_date, args.workers, deadline)
    else:
        run_single_domain(target_date, args.domain, args.workers, deadline)


if __name__ == "__main__":
    main()
