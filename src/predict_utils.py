"""
Shared helpers to load saved models/encoders and run predictions on any
feature DataFrame that has the same raw columns as waveform.csv.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class SoftVotingEnsemble:
    """
    Averages predict_proba across a set of already-configured (unfitted) model
    factory functions. Defined here (not in train.py) so it's importable
    wherever a saved model gets unpickled via joblib.load().
    """

    def __init__(self, model_fns: dict):
        self.model_fns = model_fns
        self.models_ = {}

    def fit(self, X, y):
        self.models_ = {}
        for name, fn in self.model_fns.items():
            m = fn()
            m.fit(X, y)
            self.models_[name] = m
        return self

    def predict_proba(self, X):
        import numpy as np
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models_.values()], axis=0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self):
        # model_fns holds lambdas (unpicklable) — only needed during .fit(),
        # not for prediction. Drop it from the saved state.
        state = self.__dict__.copy()
        state["model_fns"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def load_artifacts():
    meta_path = config.MODELS_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained models found at {config.MODELS_DIR}. Run src/train.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    encoders = joblib.load(config.MODELS_DIR / "label_encoders.joblib")
    medians = joblib.load(config.MODELS_DIR / "impute_medians.joblib")

    models = {}
    for fault in meta["faults_trained"]:
        models[fault] = joblib.load(config.MODELS_DIR / f"{fault}_model.joblib")

    return meta, encoders, medians, models


def load_mixed_artifacts(fault_domain: dict = None):
    """
    Loads each fault's model from whichever domain config.FAULT_DOMAIN (or the
    passed-in fault_domain dict) says wins that fault. Returns a dict keyed by
    fault name: {fault: {"meta":..., "encoders":..., "medians":..., "model":...,
    "domain":...}} — each fault carries its own domain-specific artifacts,
    since acceleration and velocity models were trained on different feature
    values (even though the column names match).

    Temporarily calls config.set_domain() per fault to resolve MODELS_DIR, then
    restores whatever domain was active before this call.
    """
    fault_domain = fault_domain or config.FAULT_DOMAIN
    original_domain = config.CURRENT_DOMAIN

    per_fault = {}
    try:
        for fault, domain in fault_domain.items():
            config.set_domain(domain)
            meta_path = config.MODELS_DIR / "meta.json"
            if not meta_path.exists():
                print(f"  [WARN] no trained models for domain '{domain}' at {config.MODELS_DIR} — "
                      f"skipping {fault}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if fault not in meta.get("faults_trained", []):
                print(f"  [WARN] {fault} was not trained in domain '{domain}' — skipping")
                continue

            encoders = joblib.load(config.MODELS_DIR / "label_encoders.joblib")
            medians = joblib.load(config.MODELS_DIR / "impute_medians.joblib")
            model = joblib.load(config.MODELS_DIR / f"{fault}_model.joblib")

            per_fault[fault] = {
                "meta": meta, "encoders": encoders, "medians": medians,
                "model": model, "domain": domain,
            }
    finally:
        config.set_domain(original_domain)

    return per_fault


def predict_faults_mixed(domain_dfs: dict, per_fault_artifacts: dict, threshold=None, thresholds: dict = None):
    """
    Mixed-domain version of predict_faults(). `domain_dfs` is
    {"acceleration": accel_df, "velocity": velocity_df}.

    IMPORTANT: rows are aligned by POSITION, not by a business key. This is
    safe (and much less error-prone than a key-based merge) because both
    domains are built by build_dataset.py walking the exact same files in the
    exact same order — row i in accel_df and row i in velocity_df are always
    the same physical packet. A key-based merge risks silent row duplication
    if (bearingLocationId, date, axis, packet_number) isn't perfectly unique,
    which inflates both the denominator and the flagged count.

    All dataframes in domain_dfs MUST have been produced by identical
    filtering (same status filter, same .reset_index(drop=True)) so they're
    still row-aligned when passed in here. This function checks that all
    domains have equal length and raises if not, rather than silently
    misaligning data.
    """
    thresholds = thresholds or {}

    lengths = {domain: len(df) for domain, df in domain_dfs.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Domain row counts don't match: {lengths}. Rows must be positionally "
            f"aligned (same filtering, same reset_index) across all domains passed "
            f"to predict_faults_mixed(), or predictions will be silently wrong."
        )

    base_domain = "acceleration" if "acceleration" in domain_dfs else next(iter(domain_dfs))
    key_cols = ["bearingLocationId", "date", "axis", "packet_number"]
    out = domain_dfs[base_domain][key_cols].reset_index(drop=True).copy()

    for fault, artifacts in per_fault_artifacts.items():
        domain = artifacts["domain"]
        if domain not in domain_dfs:
            print(f"  [WARN] {fault} needs domain '{domain}' data, which wasn't provided — skipping")
            continue

        df = domain_dfs[domain].reset_index(drop=True)
        X = prepare_features(df, artifacts["meta"], artifacts["encoders"], artifacts["medians"])
        probs = artifacts["model"].predict_proba(X)[:, 1]

        t = (
            thresholds.get(fault)
            or getattr(config, "PREDICTION_THRESHOLDS", {}).get(fault)
            or threshold
            or 0.5
        )

        # positional assignment — guaranteed 1:1, no merge, no duplication risk
        out[f"{fault}_prob"] = probs
        out[f"{fault}_pred"] = (probs >= t).astype(int)

    pred_cols = [c for c in out.columns if c.endswith("_pred")]
    fault_names = [c[:-5] for c in pred_cols]
    out["any_fault_pred"] = (out[pred_cols].fillna(0).sum(axis=1) > 0).astype(int)
    out["predicted_faults"] = out[pred_cols].fillna(0).apply(
        lambda r: ", ".join([fault_names[i] for i, v in enumerate(r) if v]) or "None", axis=1
    )
    return out


def prepare_features(df: pd.DataFrame, meta, encoders, medians) -> pd.DataFrame:
    """Build the model-ready X matrix from a raw feature dataframe (waveform.csv schema)."""
    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    X = df[feature_cols].copy()

    for col in meta["categorical_cols"]:
        le = encoders[col]
        X[col] = X[col].astype(str)
        known = set(le.classes_)
        # unseen categories -> map to the most frequent known class to avoid crashing
        fallback = le.classes_[0] if len(le.classes_) else ""
        X[col] = X[col].apply(lambda v: v if v in known else fallback)
        X[col] = le.transform(X[col])

    X = X.fillna(medians)
    # any column still fully missing from medians (e.g. new col) -> 0
    X = X.fillna(0)
    return X


def predict_faults(df: pd.DataFrame, meta, encoders, medians, models, threshold=None, thresholds: dict = None):
    """
    Returns a copy of df with added columns:
        {fault}_prob, {fault}_pred  for every trained fault.

    Threshold resolution order per fault:
        1. `thresholds[fault]` if given
        2. `config.PREDICTION_THRESHOLDS[fault]` if defined there
        3. `threshold` arg if given
        4. `config.PREDICTION_THRESHOLD` (global fallback)
    """
    thresholds = thresholds or {}
    X = prepare_features(df, meta, encoders, medians)

    out = df.copy()
    for fault, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        t = (
            thresholds.get(fault)
            or getattr(config, "PREDICTION_THRESHOLDS", {}).get(fault)
            or threshold
            or getattr(config, "PREDICTION_THRESHOLD", 0.5)
        )
        out[f"{fault}_prob"] = probs
        out[f"{fault}_pred"] = (probs >= t).astype(int)

    fault_names = list(models.keys())
    pred_cols = [f"{f}_pred" for f in fault_names]
    out["any_fault_pred"] = (out[pred_cols].sum(axis=1) > 0).astype(int)
    out["predicted_faults"] = out[pred_cols].apply(
        lambda r: ", ".join([fault_names[i] for i, v in enumerate(r) if v]) or "None", axis=1
    )
    return out
