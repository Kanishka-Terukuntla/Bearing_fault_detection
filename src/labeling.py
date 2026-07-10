"""
Same labeling strategy as your original notebook:
  1. Compute shaft_frequency from machineRpm.
  2. Compute BPFI/BPFO/BSF/FTF (Hz) from catalogue multipliers * shaft_frequency.
  3. For each fault, score confidence via harmonic matching + sideband detection
     + local SNR, then threshold into a binary label.

Usage:
    python src/labeling.py
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FAULT_COLS = config.FAULT_COLS


def parse_list_col(val):
    """Handles: real list, stringified list '[1.0, 2.0]', comma-separated string, or NaN."""
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if pd.isna(val):
        return []
    if isinstance(val, str):
        s = val.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
        try:
            return [float(x) for x in s.strip("[]").split(",") if x.strip()]
        except ValueError:
            return []
    return []


def local_noise_floor(amplitudes, prominences):
    """Median amplitude excluding the single dominant peak — robust to shaft-freq dominance."""
    if len(amplitudes) < 2:
        return np.median(amplitudes) if len(amplitudes) else 0.0
    amps = np.array(amplitudes)
    sorted_amps = np.sort(amps)[:-1]
    return np.median(sorted_amps) if len(sorted_amps) > 0 else np.median(amps)


def match_peaks_near(freqs, amps, target_freq, tol_pct):
    if target_freq is None or pd.isna(target_freq) or target_freq <= 0 or len(freqs) == 0:
        return False, 0.0
    tol = target_freq * tol_pct
    matched_amps = [a for f, a in zip(freqs, amps) if abs(f - target_freq) <= tol]
    if matched_amps:
        return True, max(matched_amps)
    return False, 0.0


def compute_fault_confidence(row, fault_freq_col, shaft_freq_col="shaft_frequency"):
    fault_freq = row[fault_freq_col]
    shaft_freq = row.get(shaft_freq_col, np.nan)
    freqs = row["all_peak_frequencies_parsed"]
    amps = row["all_peak_amplitudes_parsed"]

    if pd.isna(fault_freq) or fault_freq <= 0 or len(freqs) == 0:
        return pd.Series({"confidence": 0.0, "harmonics_matched": 0,
                           "sidebands_found": 0, "snr": 0.0})

    noise_floor = local_noise_floor(amps, row["all_peak_prominences_parsed"])
    noise_floor = max(noise_floor, 1e-8)

    harmonics_matched = 0
    best_harmonic_amp = 0.0
    for k in range(1, config.MAX_HARMONICS + 1):
        matched, amp = match_peaks_near(freqs, amps, fault_freq * k, config.get_param("HARMONIC_TOLERANCE_PCT"))
        if matched:
            harmonics_matched += 1
            best_harmonic_amp = max(best_harmonic_amp, amp)

    sidebands_found = 0
    if not pd.isna(shaft_freq) and shaft_freq > 0:
        for k in config.SIDEBAND_K:
            for target in [fault_freq + k * shaft_freq, fault_freq - k * shaft_freq]:
                matched, _ = match_peaks_near(freqs, amps, target, config.get_param("SIDEBAND_TOLERANCE_PCT"))
                if matched:
                    sidebands_found += 1

    snr = best_harmonic_amp / noise_floor if best_harmonic_amp > 0 else 0.0
    snr_pass = snr >= config.get_param("LOCAL_SNR_MULT")
    snr_score = min(snr / (config.get_param("LOCAL_SNR_MULT") * 2), 1.0)

    harmonic_score = harmonics_matched / config.MAX_HARMONICS
    sideband_score = min(sidebands_found / (2 * len(config.SIDEBAND_K)), 1.0)

    confidence = (config.W_HARMONIC * harmonic_score +
                  config.W_SIDEBAND * sideband_score +
                  config.W_SNR * snr_score)

    if not snr_pass:
        confidence *= 0.3

    if harmonics_matched < 2:
        confidence *= 0.3

    return pd.Series({
        "confidence": confidence,
        "harmonics_matched": harmonics_matched,
        "sidebands_found": sidebands_found,
        "snr": snr,
    })


def add_fault_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """shaft_frequency + BPFI/BPFO/BSF/FTF (Hz) from catalogue multipliers."""
    df = df.copy()
    df["shaft_frequency"] = df["machineRpm"].astype(float) / 60.0
    for fault, mult_col in config.FAULT_MULT_COLS.items():
        df[fault] = df[mult_col].astype(float) * df["shaft_frequency"]
    return df


def compute_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes every derived feature the trained models actually need beyond
    the raw per-packet feature_extraction.py output: shaft_frequency,
    BPFI/BPFO/BSF/FTF (Hz), and {fault}_envelope_amp.

    Used by BOTH label_dataset() (training-time, on historical data) and
    predict_daily.py (live data) — critical that both paths compute these
    identically, or the model will see a different feature distribution at
    inference time than it was trained on.
    """
    df = df.copy()

    for col in ["envelope_all_peak_frequencies", "envelope_all_peak_amplitudes"]:
        if col in df.columns:
            df[col + "_parsed"] = df[col].apply(parse_list_col)
        else:
            df[col + "_parsed"] = [[] for _ in range(len(df))]

    df = add_fault_frequencies(df)

    for fault in config.FAULT_COLS:
        df[f"{fault}_envelope_amp"] = df.apply(
            lambda row: match_peaks_near(
                row["envelope_all_peak_frequencies_parsed"],
                row["envelope_all_peak_amplitudes_parsed"],
                row[fault],
                config.get_param("HARMONIC_TOLERANCE_PCT"),
            )[1],
            axis=1,
        )
    return df


def label_dataset(df: pd.DataFrame, mode: str = None) -> tuple[pd.DataFrame, list, list]:
    mode = mode or config.get_param("ACTIVE_MODE")
    df = df.copy()

    for col in ["all_peak_frequencies", "all_peak_amplitudes", "all_peak_prominences"]:
        if col in df.columns:
            df[col + "_parsed"] = df[col].apply(parse_list_col)
        else:
            df[col + "_parsed"] = [[] for _ in range(len(df))]

    df = compute_prediction_features(df)

    leakage_cols = []
    for fault in config.FAULT_COLS:
        print(f"Labeling {fault}...")
        result = df.apply(lambda row: compute_fault_confidence(row, fault_freq_col=fault), axis=1)
        df[f"{fault}_confidence"] = result["confidence"]
        df[f"{fault}_harmonics_matched"] = result["harmonics_matched"]
        df[f"{fault}_sidebands_found"] = result["sidebands_found"]
        df[f"{fault}_snr"] = result["snr"]
        df[f"{fault}_label"] = (result["confidence"] >= config.LABEL_THRESHOLD[mode]).astype(int)

        leakage_cols += [f"{fault}_confidence", f"{fault}_harmonics_matched",
                          f"{fault}_sidebands_found", f"{fault}_snr"]
        print(f"  -> {fault}_label positive rate: {df[f'{fault}_label'].mean():.3%}")

    label_cols = [f"{f}_label" for f in config.FAULT_COLS]
    print("\nLabel distribution:\n", df[label_cols].sum())
    print("Multi-label overlap (rows with >1 fault):", (df[label_cols].sum(axis=1) > 1).sum())

    return df, label_cols, leakage_cols


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    parser.add_argument("--label-mode", choices=["strict", "balanced", "loose"], default=None,
                         help="Override config.ACTIVE_MODE for this run only (doesn't change config.py).")
    args = parser.parse_args()
    config.set_domain(args.domain)

    src = config.WAVEFORM_CSV
    df = pd.read_csv(src)
    labeled, label_cols, leakage_cols = label_dataset(df, mode=args.label_mode)
    labeled.to_csv(config.LABELED_CSV, index=False)
    print(f"\nWrote labeled dataset -> {config.LABELED_CSV}")
