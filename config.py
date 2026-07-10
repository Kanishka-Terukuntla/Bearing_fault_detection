"""
Central configuration for the bearing fault detection project.
Edit the values in this file (or override via environment variables, or a
.env file in the project root) before running.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Load a .env file from the project root, if present, BEFORE reading any
# os.environ values below — so AAMS_EMAIL/AAMS_PASSWORD/etc. in a .env file
# work the same as real environment variables, without you having to export
# them manually every session.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("[config] python-dotenv not installed — .env file (if any) will be ignored. "
          "Run: pip install python-dotenv --break-system-packages")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root folder that contains your date sub-folders (2026-05-22/, 2026-05-25/, ...)
# and catalogue.json, exactly as shown in your screenshot.
RAW_DATA_ROOT = Path(os.environ.get("BFD_RAW_DATA_ROOT", PROJECT_ROOT / "data" / "raw"))

# Path to the catalogue / bearing-details JSON (list of bearing metadata dicts)
CATALOGUE_PATH = Path(os.environ.get("BFD_CATALOGUE_PATH", RAW_DATA_ROOT / "catalogue.json"))

DATA_DIR = PROJECT_ROOT / "data"
LIVE_DATA_ROOT = DATA_DIR / "live"  # daily API pulls saved here, same schema as data/raw
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

WAVEFORM_CSV = OUTPUTS_DIR / "waveform.csv"                 # raw extracted features (all statuses)
LABELED_CSV = OUTPUTS_DIR / "waveform_labeled.csv"           # + fault confidence/labels
TRAIN_CSV = OUTPUTS_DIR / "train_dataset.csv"                # Marginal/Unacceptable only, used for training
CV_SUMMARY_CSV = OUTPUTS_DIR / "cv_summary.csv"
NORMAL_FLAG_REPORT_CSV = OUTPUTS_DIR / "normal_flag_report.csv"
DAILY_PREDICTIONS_CSV = OUTPUTS_DIR / "daily_predictions.csv"   # append-only history of daily predictions

for d in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain switch: acceleration (default, backward-compatible paths above) vs.
# velocity (integrated from acceleration — see feature_extraction.integrate_
# accel_to_velocity). Call config.set_domain("velocity") at the top of a
# script's main() to repoint every path below at the parallel velocity
# outputs/models, without touching any other code.
# ---------------------------------------------------------------------------
CURRENT_DOMAIN = "acceleration"


def set_domain(domain: str):
    """Switch WAVEFORM_CSV/LABELED_CSV/TRAIN_CSV/CV_SUMMARY_CSV/
    NORMAL_FLAG_REPORT_CSV/DAILY_PREDICTIONS_CSV/MODELS_DIR to the
    acceleration (default) or velocity variant, in-place on this module."""
    global CURRENT_DOMAIN, WAVEFORM_CSV, LABELED_CSV, TRAIN_CSV, CV_SUMMARY_CSV
    global NORMAL_FLAG_REPORT_CSV, DAILY_PREDICTIONS_CSV, MODELS_DIR

    domain = domain.lower()
    if domain not in ("acceleration", "velocity"):
        raise ValueError(f"Unknown domain '{domain}' — must be 'acceleration' or 'velocity'")

    CURRENT_DOMAIN = domain
    suffix = "" if domain == "acceleration" else "_velocity"

    WAVEFORM_CSV = OUTPUTS_DIR / f"waveform{suffix}.csv"
    LABELED_CSV = OUTPUTS_DIR / f"waveform_labeled{suffix}.csv"
    TRAIN_CSV = OUTPUTS_DIR / f"train_dataset{suffix}.csv"
    CV_SUMMARY_CSV = OUTPUTS_DIR / f"cv_summary{suffix}.csv"
    NORMAL_FLAG_REPORT_CSV = OUTPUTS_DIR / f"normal_flag_report{suffix}.csv"
    DAILY_PREDICTIONS_CSV = OUTPUTS_DIR / f"daily_predictions{suffix}.csv"
    MODELS_DIR = PROJECT_ROOT / ("models" if domain == "acceleration" else "models_velocity")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain-specific overrides for peak-extraction sensitivity and labeling
# strictness. Falls back to the base module-level value when the current
# domain has no override for a given parameter. Use this to loosen thresholds
# for one domain (e.g. velocity, whose spectrum shape differs after
# integration) without touching a domain that's already tuned (acceleration).
#
# Set via config.set_domain() + config.get_param("NAME") instead of reading
# config.NAME directly, anywhere the value should vary by domain.
# ---------------------------------------------------------------------------
DOMAIN_OVERRIDES = {
    # Tested and reverted: loosening velocity's peak-extraction/labeling
    # thresholds raised offline F1 (0.34->0.65 for FTF) but made the REAL
    # metric worse — flag rate on held-out Normal bearings went from
    # ~0.5% to 1.43% for FTF via velocity, and recall at the deployed
    # threshold actually dropped (98.89% -> 82.93%). The F1 gain was an
    # artifact of loosening pushing the labeled-positive rate from ~2-9%
    # up to 40%, making the classification task easier, not the detector
    # better. Left empty so velocity uses the same tuned strict settings
    # as acceleration. See conversation history if you want to reference
    # the loose values that were tried (FFT_PEAK_MIN_PROMINENCE_FRAC=0.01,
    # HARMONIC/SIDEBAND_TOLERANCE_PCT=0.05, LOCAL_SNR_MULT=2.0, "loose" mode).
}


def get_param(name: str):
    """Resolve a labeling/feature-extraction parameter, applying the current
    domain's override (DOMAIN_OVERRIDES[CURRENT_DOMAIN]) if one exists for
    `name`, otherwise falling back to this module's top-level value."""
    override = DOMAIN_OVERRIDES.get(CURRENT_DOMAIN, {})
    if name in override:
        return override[name]
    return globals()[name]

# ---------------------------------------------------------------------------
# AAMS API (for daily live predictions)
# ---------------------------------------------------------------------------
AAMS_BASE_URL = os.environ.get("AAMS_BASE_URL")
AAMS_EMAIL = os.environ.get("AAMS_EMAIL", "")
AAMS_PASSWORD = os.environ.get("AAMS_PASSWORD", "")

# ---------------------------------------------------------------------------
# Statuses used for training vs. held-out false-positive checking
# ---------------------------------------------------------------------------
TRAIN_STATUSES = ["Marginal", "Unacceptable"]
HOLDOUT_NORMAL_STATUS = "Normal"

# ---------------------------------------------------------------------------
# Fault frequency multipliers -> Hz conversion
# BPFI/BPFO/BSF/FTF (Hz) = multiplier_from_catalogue * shaft_frequency (Hz)
# shaft_frequency (Hz) = machineRpm / 60
# ---------------------------------------------------------------------------
FAULT_MULT_COLS = {
    "BPFI": "innerRacePass",
    "BPFO": "outerRacePass",
    "BSF": "rollElementPass",
    "FTF": "cageRotation",
}
FAULT_COLS = list(FAULT_MULT_COLS.keys())

# ---------------------------------------------------------------------------
# Labeling config (same strategy as your original notebook)
# ---------------------------------------------------------------------------
HARMONIC_TOLERANCE_PCT = 0.03
MAX_HARMONICS = 4
SIDEBAND_K = [1, 2]
SIDEBAND_TOLERANCE_PCT = 0.03
LOCAL_SNR_MULT = 4.0

W_HARMONIC = 0.5
W_SIDEBAND = 0.3
W_SNR = 0.2

LABEL_THRESHOLD = {
    "strict": 0.70,
    "balanced": 0.50,
    "loose": 0.35,
}
ACTIVE_MODE = os.environ.get("BFD_LABEL_MODE", "strict")

# ---------------------------------------------------------------------------
# Prediction-time threshold(s) — separate from the LABELING threshold above.
# This controls what probability a trained model's output must clear to
# flag a row as faulty at inference time (evaluate_normal.py, predict_daily.py,
# the dashboard). Chosen from a recall-vs-flag-rate sweep (see threshold_sweep.py).
# ---------------------------------------------------------------------------
PREDICTION_THRESHOLD = 0.75   # fallback, used only for faults not listed below

PREDICTION_THRESHOLDS = {
    "BPFI": 0.60,
    "BPFO": 0.60,
    "BSF": 0.60,
    "FTF": 0.60,
}

# ---------------------------------------------------------------------------
# Per-fault domain routing: which domain's model actually wins each fault,
# based on the acceleration-vs-velocity comparison (compare_domains.py).
# Used by predict_utils.load_mixed_artifacts() / predict_faults_mixed() to
# build one production system that uses the best domain per fault, rather
# than forcing a single domain for everything.
# ---------------------------------------------------------------------------
FAULT_DOMAIN = {
    "BPFI": "acceleration",
    "BPFO": "acceleration",
    "BSF": "acceleration",
    "FTF": "velocity",
}

# ---------------------------------------------------------------------------
# Frequency-band edges (Hz) for band-energy features
# ---------------------------------------------------------------------------
FREQ_BANDS = [
    ("band_0_500", 0, 500),
    ("band_500_1000", 500, 1000),
    ("band_1000_2000", 1000, 2000),
    ("band_2000_4000", 2000, 4000),
    ("band_4000_end", 4000, None),  # None -> up to Nyquist
]

# find_peaks params for FFT-peak features
FFT_PEAK_MIN_PROMINENCE_FRAC = 0.02  # fraction of max spectrum amplitude

N_SPLITS = 5
RANDOM_STATE = 42
