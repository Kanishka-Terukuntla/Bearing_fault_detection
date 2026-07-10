# Bearing Fault Detection

End-to-end pipeline: raw vibration waveforms → engineered features → physics-informed
labels (harmonics + sidebands + SNR) → trained fault classifiers (BPFI/BPFO/BSF/FTF) →
daily live scoring against the AAMS API → Streamlit dashboard.

## Project structure

```
bearing_fault_detection/
├── config.py                  # all paths, API creds, labeling/feature constants
├── requirements.txt
├── data/
│   └── raw/                   # <-- put your data here: YYYY-MM-DD/ folders + catalogue.json
├── src/
│   ├── feature_extraction.py  # per-packet time/frequency/FFT-peak feature computation
│   ├── build_dataset.py       # raw JSON -> outputs/waveform.csv (matches your exact schema)
│   ├── labeling.py            # shaft_frequency, BPFI/BPFO/BSF/FTF, confidence scoring, labels
│   ├── train.py                # GroupKFold CV (XGB/LightGBM/RF), SMOTE, saves best model per fault
│   ├── evaluate_normal.py     # applies trained models to held-out Normal rows (false-positive check)
│   ├── api_client.py           # AAMS API wrapper (login / list bearings / get raw)
│   ├── predict_daily.py       # pulls today's data from the live API and scores it
│   └── predict_utils.py       # shared model-loading / prediction helpers
├── dashboard/
│   └── app.py                  # Streamlit dashboard (local only)
├── models/                     # saved models + encoders + meta.json (created by train.py)
├── outputs/                    # all generated CSVs (created by the scripts)
└── scripts_dev/
    └── gen_synthetic.py         # generates fake data to smoke-test the pipeline — not for production
```

## Setup

```bash
cd bearing_fault_detection
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

Point the project at your real data by editing `config.py` (or environment variables):

```bash
export BFD_RAW_DATA_ROOT="/path/to/your/date-folders"
export BFD_CATALOGUE_PATH="/path/to/your/date-folders/catalogue.json"
export AAMS_EMAIL="ml-service@yourcompany.com"
export AAMS_PASSWORD="********"
```

Your raw data folder should look like your screenshot:

```
raw_data/
├── 2026-05-22/
│   ├── <bearing_capture_1>.json
│   └── ...
├── 2026-05-25/
├── 2026-06-01/
├── ...
└── catalogue.json
```

Each waveform JSON file is expected to look like the example you shared: a top-level
`bearingLocationId`, `date`, `status`, and a `packets` list, where each packet has
`samples`, `sampling_rate`/`sr`, `axis`, etc. (Both the live-API camelCase field names
and your local snake_case field names are handled automatically — see
`normalize_packet()` in `feature_extraction.py`.)

`catalogue.json` should be the list of bearing metadata dicts (your example with
`machineRpm`, `innerRacePass`, `outerRacePass`, `rollElementPass`, `cageRotation`, etc.),
keyed by `bearingLocationId`.

## Run order (from the project root)

```bash
# 1. Extract features from every raw waveform file -> outputs/waveform.csv
python -m src.build_dataset

# 2. Compute shaft_frequency, BPFI/BPFO/BSF/FTF (Hz), confidence scores, and binary labels
python -m src.labeling

# 3. Train models (Marginal + Unacceptable rows only) and save to models/
python -m src.train

# 4. Check how many Normal-status bearings the trained model still flags (false-positive check)
python -m src.evaluate_normal

# 5. (Optional, CLI) pull today's data from the live API and score it
python -m src.predict_daily                    # defaults to today (UTC)
python -m src.predict_daily --date 2026-07-06  # or a specific date

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard also has a **"Fetch & Predict"** button on the Daily Live Predictions tab,
so you don't have to run step 5 from the command line — it calls the same code directly.

## How labeling works (unchanged from your original strategy)

For every packet:
1. `shaft_frequency = machineRpm / 60`
2. `BPFI/BPFO/BSF/FTF (Hz) = catalogue_multiplier * shaft_frequency`
3. For each fault frequency, check:
   - **Harmonics**: peaks near `1x, 2x, 3x, 4x` the fault frequency (±3% tolerance)
   - **Sidebands**: peaks near `fault_freq ± k*shaft_freq` for k=1,2 (only if shaft_freq known)
   - **Local SNR**: best-matched harmonic amplitude vs. median amplitude of all other peaks
4. Confidence = `0.5*harmonic_score + 0.3*sideband_score + 0.2*snr_score`, hard-capped to
   30% of its value if the SNR gate fails.
5. Binary label = `confidence >= threshold` (default `balanced` = 0.50; see `config.LABEL_THRESHOLD`).

All of this lives in `src/labeling.py`, unchanged logically from what you had, just
wired up to compute `shaft_frequency`/`BPFI`/`BPFO`/`BSF`/`FTF` from the catalogue first.

## Training strategy

- **Training set**: only rows with `status in ['Marginal', 'Unacceptable']`
  (per your instruction — the labeling heuristic is most trustworthy on equipment
  already known to be degraded).
- **GroupKFold by `bearingLocationId`**: no single bearing's packets leak across
  train/test folds.
- **SMOTE**: applied only inside each training fold (and again on the full training
  set for the final refit) — never on test data.
- **Model selection**: XGBoost / LightGBM / RandomForest compared per fault; the
  best (highest mean F1) is refit on the full training set and saved.
- Faults with fewer than 5 positive examples are skipped (not enough for 5-fold CV) —
  you'll see this reported explicitly when you run `train.py`.

## Normal-bearing false-positive check

`evaluate_normal.py` (and the dashboard's "✅ Normal-Bearing Flag Check" tab) applies
the trained models to every `Normal`-status row that was **held out of training** and
reports what fraction get flagged as faulty. This is your over-triggering / false-alarm
rate on equipment that's supposedly healthy — worth watching per fault type, since a
high rate on one fault (vs. the others) usually means that fault's labeling threshold
or feature set needs tightening.

## What changed in the F1-improvement pass

Four changes were made together to raise model quality (not just re-balance the
existing precision/recall trade-off):

1. **Envelope-spectrum features** (`feature_extraction.py`, `labeling.py`) — the
   standard bearing-diagnostics technique: rectify the signal via its Hilbert-transform
   envelope, then FFT that envelope. Bearing impact frequencies show up far more cleanly
   here than in the raw spectrum. New columns: `envelope_dominant_frequency`,
   `envelope_max_amplitude`, `envelope_spectral_entropy`, `envelope_energy`, and
   `{fault}_envelope_amp` (envelope amplitude right at each fault's base frequency) —
   the latter is a genuine new model feature, not a leakage column.
2. **StratifiedGroupKFold** instead of plain `GroupKFold` (`train.py`) — stratifies
   folds by the fault label (not just by bearing), which is why your F1_std was so
   large before (±0.2-0.3): some folds had very different fault composition than others.
3. **Per-fault hyperparameter search** (`RandomizedSearchCV`, small grid, `train.py`) —
   each fault now gets its own tuned XGBoost/LightGBM/RandomForest instead of one
   fixed hyperparameter set for all four faults.
4. **SMOTETomek instead of plain SMOTE**, and a **soft-voting ensemble** of the three
   tuned models compared against each individually — whichever wins (by mean F1) gets
   saved and used.

None of this touches labeling — your tuned `LOCAL_SNR_MULT=4.0` and
`harmonics_matched < 2` rule are unchanged. Rerun the full chain (`build_dataset` →
`labeling` → `train` → `evaluate_normal`) since the feature schema changed (envelope
columns are new), even though the labels themselves won't change.

Two things to know before you run it:
- **Training now takes noticeably longer** — hyperparameter search runs a small
  `RandomizedSearchCV` per model per fault before the final CV comparison.
- **`SoftVotingEnsemble`** (in `predict_utils.py`) is a small custom model class used
  when the ensemble wins for a given fault. It's saved/loaded via `joblib` like any
  other model — nothing else needs to change on your end, `evaluate_normal.py`,
  `predict_daily.py`, and the dashboard all already import from `predict_utils.py`
  where this class lives.

## Acceleration vs. Velocity comparison

Every script now supports `--domain {acceleration,velocity}` (default: `acceleration`).
`velocity` mode integrates the raw acceleration waveform (in g) into velocity (in mm/s)
via frequency-domain integration — the standard technique, and the ISO-10816 convention
for overall vibration severity — before running the exact same feature extraction,
labeling, and training code. Nothing about the architecture differs between the two;
only the signal being analyzed does.

Outputs are kept fully separate so you can run and compare both:

| | acceleration (default) | velocity |
|---|---|---|
| raw features | `outputs/waveform.csv` | `outputs/waveform_velocity.csv` |
| labeled | `outputs/waveform_labeled.csv` | `outputs/waveform_labeled_velocity.csv` |
| CV summary | `outputs/cv_summary.csv` | `outputs/cv_summary_velocity.csv` |
| normal-flag report | `outputs/normal_flag_report.csv` | `outputs/normal_flag_report_velocity.csv` |
| saved models | `models/` | `models_velocity/` |

Run both full chains, then compare:

```bash
# acceleration (as before)
python -m src.build_dataset
python -m src.labeling
python -m src.train
python -m src.evaluate_normal

# velocity
python -m src.build_dataset --domain velocity
python -m src.labeling --domain velocity
python -m src.train --domain velocity
python -m src.evaluate_normal --domain velocity

# side-by-side comparison (F1/precision/recall/AUC + normal-flag rate, both domains)
python compare_domains.py
```

`check_recall.py` and `threshold_sweep.py` also accept `--domain velocity` if you want
to compare recall-vs-flag-rate trade-off curves between the two.

## F1 reporting: two numbers, don't confuse them

`train.py`'s CV output now shows two F1 values per model:

- **`F1`** — computed at scikit-learn's default 0.5 probability cutoff. This is what all earlier
  numbers in this project referred to.
- **`F1@optThresh`** — the *best possible* F1 on that fold, found by scanning all thresholds via
  `precision_recall_curve`. This is an **upper bound, not a deployable number** — it picks the
  best threshold using that fold's own test labels, which you don't have at real prediction time.
  Use it to see how much headroom threshold tuning could theoretically buy you; use
  `threshold_sweep.py` (which reports recall/flag-rate honestly out-of-sample) to actually pick
  a deployable threshold.

Changing `config.PREDICTION_THRESHOLD(S)` does **not** change either of these numbers — that
setting only affects inference-time scripts (`evaluate_normal.py`, `predict_daily.py`, the
dashboard, `evaluate_normal_mixed.py`), not the CV metrics reported during training.

## Mixed-domain production system

Since acceleration and velocity don't win the same faults (see the comparison above — velocity
wins FTF by a wide margin, acceleration wins the other three), `config.FAULT_DOMAIN` routes each
fault to whichever domain's model actually performs best for it:

```python
FAULT_DOMAIN = {
    "BPFI": "acceleration",
    "BPFO": "acceleration",
    "BSF": "acceleration",
    "FTF": "velocity",
}
```

`src/predict_utils.py` has two new functions to support this:
- `load_mixed_artifacts()` — loads each fault's model from its routed domain's `models/` folder.
- `predict_faults_mixed()` — runs each fault against the right domain's features and merges
  results into one table (join key: `bearingLocationId`, `date`, `axis`, `packet_number`).

`evaluate_normal_mixed.py` validates the actual system you'd deploy — both domains' models
together, each fault using its own domain and threshold — against held-out Normal bearings.
This is the number that matters most; the individual per-domain `evaluate_normal.py` runs are
useful for comparison, but `evaluate_normal_mixed.py` is what a real deployment looks like.

## Daily prediction: how it actually works

`predict_daily.py` does NOT run a separate in-memory feature pipeline anymore — it
saves live API data to disk and reuses your exact `build_dataset.py`/`labeling.py`
code, so training-time and prediction-time feature computation can't silently drift
apart. The flow:

1. Login, fetch the live bearing catalogue + today's raw waveforms from the API.
2. **Save raw waveforms to `data/live/{date}/*.json`** — same schema as your historical
   `data/raw/{date}/` files.
3. **Merge the live catalogue with your local `catalogue.json`** (`src/fetch_live_data.py`
   `build_merged_catalogue()`) — the live `GET /ml/bearings` response often doesn't include
   bearing geometry (`innerRacePass`/`outerRacePass`/`rollElementPass`/`cageRotation`), even
   though your bulk `catalogue.json` does. Live values win when present (status, rpm, etc.);
   geometry falls back to your local catalogue, keyed by `bearingLocationId`. The merged
   result is saved to `data/live/{date}/catalogue.json`.
4. **Any bearing missing geometry in BOTH sources gets a warning printed** (bearing IDs
   listed), not a silent guess or a crash. Its `shaft_frequency` still computes correctly
   (only needs live `machineRpm`), but its raw geometry feature columns are `NaN` — those
   get filled with **training-set median values** at prediction time (`prepare_features()`
   in `predict_utils.py`, same median-imputation logic used everywhere else in this
   project), and its `{fault}_envelope_amp` features fall back to `0.0` (meaning "no signal
   found at an unknown frequency") rather than crashing or producing garbage. If you see
   these warnings repeatedly for the same bearings, add their geometry to your local
   `catalogue.json` — the fallback is safe, not ideal.
5. **`build_dataset.build()`** — the same function used for historical data — runs against
   the saved `data/live/{date}/` folder + merged catalogue, once per domain.
6. **`labeling.compute_prediction_features()`** — the same function used at training time —
   adds `shaft_frequency`, `BPFI`/`BPFO`/`BSF`/`FTF` (Hz), and `{fault}_envelope_amp`.
7. Scored with the trained models (mixed system by default, per `config.FAULT_DOMAIN`).

## Fetching speed (large fleets)

`fetch_live_data.py` fetches all (bearing, axis) pairs in parallel via a thread pool
(`--workers`, default 20) with a hard wall-clock deadline (`--deadline`, default 1800s/30min)
so a handful of slow/stuck requests can't block the whole run — whatever completes by the
deadline is kept, stragglers are logged and skipped. No customer filtering — fetches
everything `list_bearings()` returns for your account's scope.

```bash
python -m src.predict_daily --workers 30 --deadline 3600   # more parallelism, longer deadline
python -m src.predict_daily --deadline 0                    # no deadline, wait for everything
```

## Domain-specific thresholds (config.get_param)

Peak-detection sensitivity and labeling strictness can now differ per domain via
`config.DOMAIN_OVERRIDES` + `config.get_param("NAME")`, instead of one fixed value used
everywhere. Currently velocity uses deliberately looser settings than acceleration
(lower FFT peak prominence, wider harmonic/sideband tolerance, lower SNR gate, "loose"
label mode) — acceleration's tuned strict settings are untouched. `build_dataset.build()`
calls `config.set_domain()` itself now (not just the CLI), so this resolves correctly no
matter what called it from — including `predict_daily.py`, which builds both domains in
one run.

If you retune either domain further, edit `config.DOMAIN_OVERRIDES["velocity"]` (or add
an `"acceleration"` key) rather than the base `HARMONIC_TOLERANCE_PCT`/etc. values directly
— those are still the acceleration/default values other code falls back to.

## Notes / things to sanity-check on your real data

- `predict_daily.py` calls the live AAMS API directly (needs `AAMS_EMAIL`/`AAMS_PASSWORD`
  for a `SUPER_ADMIN` or `ANALYST` (customer scope `["*"]`) account, per the API doc).
  It was not run against the live API in this environment (no network access to
  `apiv3.aams.io` from here) — test it from your own machine first with a single bearing
  before running it over your whole fleet.
- The full pipeline (`build_dataset` → `labeling` → `train` → `evaluate_normal`) **was**
  run end-to-end here against synthetic data with injected BPFI/BPFO fault signatures,
  and produced sensible results (correct positive rates, correct feature/label separation,
  no leakage). See `scripts_dev/gen_synthetic.py` if you want to regenerate that test data.
- `innerRacePass`/`outerRacePass`/`rollElementPass`/`cageRotation` are assumed to be
  **multipliers of shaft frequency** (the standard bearing-fault convention — e.g. BPFI
  ≈ 4.9x shaft speed for a typical bearing). Double check one bearing's numbers against
  a known-good reference if these labels look off.
