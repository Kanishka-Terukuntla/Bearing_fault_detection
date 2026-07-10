"""
Walk RAW_DATA_ROOT (date-folders full of waveform JSON files), extract features
for every packet, join with catalogue.json, and write outputs/waveform.csv (or
outputs/waveform_velocity.csv) with exactly your target column schema.

Usage:
    python -m src.build_dataset                      # acceleration (raw signal, default)
    python -m src.build_dataset --domain velocity     # integrate accel -> velocity (mm/s) first
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.feature_extraction import extract_all_features, integrate_accel_to_velocity, normalize_packet

TARGET_COLUMNS = [
    "bearingLocationId", "date", "analytics_type", "packet_number", "axis",
    "status", "rpm", "sampling_rate", "num_samples", "mean", "std",
    "variance", "rms", "peak", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
    "energy", "entropy", "dominant_frequency", "max_fft", "spectral_energy",
    "spectral_centroid", "spectral_spread", "spectral_entropy",
    "spectral_flatness", "rolloff_85", "rolloff_95", "band_0_500",
    "band_500_1000", "band_1000_2000", "band_2000_4000", "band_4000_end",
    "num_fft_peaks", "mean_peak_height", "std_peak_height",
    "mean_peak_prominence", "std_peak_prominence", "peak_spacing_mean",
    "peak_spacing_std", "all_peak_frequencies", "all_peak_amplitudes",
    "all_peak_prominences", "machineRpm", "bearingNumber",
    "manufacturerName", "type", "innerRacePass", "outerRacePass",
    "rollElementPass", "cageRotation",
]


def load_catalogue(catalogue_path: Path) -> pd.DataFrame:
    with open(catalogue_path, "r") as f:
        records = json.load(f)
    cat = pd.DataFrame(records)
    keep = [
        "bearingLocationId", "machineRpm", "bearingNumber",
        "manufacturerName", "type", "innerRacePass", "outerRacePass",
        "rollElementPass", "cageRotation",
    ]
    missing = [c for c in keep if c not in cat.columns]
    if missing:
        raise ValueError(f"catalogue.json is missing expected columns: {missing}")

    # geometry_known is only present on merged live catalogues (see
    # fetch_live_data.build_merged_catalogue). Historical catalogue.json files
    # are assumed complete -> default True.
    if "geometry_known" in cat.columns:
        keep = keep + ["geometry_known"]
    else:
        cat["geometry_known"] = True
        keep = keep + ["geometry_known"]

    cat = cat[keep].drop_duplicates(subset="bearingLocationId", keep="last")
    return cat


def find_waveform_files(raw_root: Path, catalogue_path: Path = None):
    """Recursively find every JSON file under raw_root that isn't the catalogue."""
    catalogue_path = catalogue_path or config.CATALOGUE_PATH
    for p in sorted(raw_root.rglob("*.json")):
        if p.resolve() == catalogue_path.resolve():
            continue
        yield p


def rows_from_file(path: Path, domain: str = "acceleration"):
    """Yield one feature-row dict per packet inside a single waveform JSON file."""
    try:
        with open(path, "r") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return

    bearing_id = doc.get("bearingLocationId")
    file_date = doc.get("date")
    top_status = doc.get("status")
    top_rpm = doc.get("rpm")
    packets = doc.get("packets", [])

    for i, raw_pkt in enumerate(packets, start=1):
        pkt = normalize_packet(raw_pkt)
        samples = pkt.get("samples", [])
        sr = pkt.get("sampling_rate") or pkt.get("sr")

        signal = np.asarray([s for s in samples if isinstance(s, (int, float))], dtype=np.float64)
        if domain == "velocity":
            signal = integrate_accel_to_velocity(signal, sr)
        feats = extract_all_features(signal, sr)

        row = {
            "bearingLocationId": bearing_id,
            "date": pkt.get("timestamp", file_date) or file_date,
            "analytics_type": pkt.get("analytics_type"),
            "packet_number": i,
            "axis": pkt.get("axis"),
            "status": top_status,
            "rpm": pkt.get("rpm", top_rpm),
            "sampling_rate": sr,
            "num_samples": pkt.get("num_samples", len(samples)),
        }
        row.update(feats)
        yield row


def build(raw_root: Path = None, catalogue_path: Path = None, out_csv: Path = None,
          domain: str = "acceleration") -> pd.DataFrame:
    # Authoritatively set the domain here (not just in __main__) so that
    # domain-aware parameters (config.get_param, used inside feature
    # extraction for e.g. peak-detection prominence) always resolve
    # correctly, regardless of what the caller's config.CURRENT_DOMAIN was
    # set to before calling this function.
    config.set_domain(domain)

    raw_root = raw_root or config.RAW_DATA_ROOT
    catalogue_path = catalogue_path or config.CATALOGUE_PATH
    out_csv = out_csv or config.WAVEFORM_CSV

    print(f"Domain: {domain}")
    print(f"Scanning waveform files under: {raw_root}")
    files = list(find_waveform_files(raw_root, catalogue_path))
    print(f"Found {len(files)} waveform JSON files")

    all_rows = []
    for path in tqdm(files, desc="Extracting features"):
        all_rows.extend(rows_from_file(path, domain=domain))

    if not all_rows:
        raise RuntimeError(
            f"No packets found under {raw_root}. Check RAW_DATA_ROOT in config.py "
            "and that your JSON files contain a 'packets' list."
        )

    df = pd.DataFrame(all_rows)

    print(f"Loading catalogue: {catalogue_path}")
    cat = load_catalogue(catalogue_path)
    df = df.merge(cat, on="bearingLocationId", how="left")

    missing_meta = df["machineRpm"].isna().sum()
    if missing_meta:
        print(f"  [WARN] {missing_meta}/{len(df)} rows have no catalogue match "
              f"(bearingLocationId not found in catalogue.json)")

    # list-type columns need to be stored as JSON strings in the CSV
    for col in ["all_peak_frequencies", "all_peak_amplitudes", "all_peak_prominences",
                "envelope_all_peak_frequencies", "envelope_all_peak_amplitudes"]:
        df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (list, np.ndarray)) else "[]")

    # enforce exact target column order (extra cols, if any, appended at the end)
    ordered = [c for c in TARGET_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in TARGET_COLUMNS]
    df = df[ordered + extra]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    args = parser.parse_args()

    config.set_domain(args.domain)
    build(domain=args.domain)
