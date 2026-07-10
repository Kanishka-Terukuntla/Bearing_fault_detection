"""
Train fault-detection models.

- Trains ONLY on rows with status in Marginal/Unacceptable (per your instruction).
- StratifiedGroupKFold by bearingLocationId, stratified on the fault label so
  folds have comparable positive rates (reduces the huge F1 variance you saw
  with plain GroupKFold).
- Per-model hyperparameter search (RandomizedSearchCV, small grid) before the
  final CV comparison — each fault gets its own tuned XGBoost/LightGBM/RandomForest.
- SMOTETomek (oversample + clean overlapping/noisy samples) inside training
  folds only, instead of plain SMOTE.
- Also evaluates a soft-voting ensemble of the three tuned models per fault,
  and picks whichever (best single model OR ensemble) has the highest mean F1.
- Saves: model(s), feature list, label encoders, median-imputation values,
  and the CV summary table.

Usage:
    python src/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.labeling import label_dataset
from src.predict_utils import SoftVotingEnsemble

EXCLUDE_BASE = {
    "bearingLocationId", "date", "analytics_type", "packet_number", "axis",
    "status", "bearingNumber", "manufacturerName", "type", "geometry_known",
    "all_peak_frequencies", "all_peak_amplitudes", "all_peak_prominences",
    "all_peak_frequencies_parsed", "all_peak_amplitudes_parsed", "all_peak_prominences_parsed",
    "envelope_all_peak_frequencies", "envelope_all_peak_amplitudes",
    "envelope_all_peak_frequencies_parsed", "envelope_all_peak_amplitudes_parsed",
    "BPFI", "BPFO", "BSF", "FTF",
}
# Note: {fault}_envelope_amp columns (e.g. BPFI_envelope_amp) are NOT excluded —
# they're a legitimate new model feature (envelope amplitude at that fault's base
# frequency), not a leakage column. They're separate from the *_confidence/_snr/
# _harmonics_matched columns in LEAKAGE_COLS, which ARE derived from the labels.

BASE_MODELS = {
    "XGBoost": lambda: XGBClassifier(
        eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1,
    ),
    "LightGBM": lambda: LGBMClassifier(
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
    ),
    "RandomForest": lambda: RandomForestClassifier(
        class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1,
    ),
}

PARAM_DISTRIBUTIONS = {
    "XGBoost": {
        "n_estimators": [150, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [150, 300, 500],
        "max_depth": [3, 4, 5, 6, -1],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 10, 15, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    },
}

SEARCH_N_ITER = 8
SEARCH_CV_SPLITS = 3


def build_feature_frame(df: pd.DataFrame, label_cols, leakage_cols):
    exclude = EXCLUDE_BASE | set(label_cols) | set(leakage_cols)
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    encoders = {}
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    return X, feature_cols, categorical_cols, encoders, medians


def tune_hyperparams(X, y, groups, model_name):
    """Small RandomizedSearchCV per model, respecting bearing grouping."""
    n_splits = min(SEARCH_CV_SPLITS, int(y.sum()), int(len(y) - y.sum()))
    n_splits = max(n_splits, 2)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=BASE_MODELS[model_name](),
        param_distributions=PARAM_DISTRIBUTIONS[model_name],
        n_iter=SEARCH_N_ITER,
        scoring="f1",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        error_score=0.0,
    )
    search.fit(X, y, groups=groups)
    return search.best_params_


def make_tuned_model_fn(model_name, params):
    def _fn():
        base = BASE_MODELS[model_name]()
        base.set_params(**params)
        return base
    return _fn


def resample(X_train, y_train):
    if y_train.sum() >= 2 and (len(y_train) - y_train.sum()) >= 2:
        try:
            smt = SMOTETomek(random_state=config.RANDOM_STATE)
            return smt.fit_resample(X_train, y_train)
        except ValueError:
            return X_train, y_train
    return X_train, y_train


def best_f1_at_optimal_threshold(y_true, probs):
    """
    Finds the threshold (on this fold's probabilities) that maximizes F1,
    and returns (best_f1, best_threshold). This answers 'what's the ceiling
    F1 achievable by threshold choice alone, for this already-trained model'
    — separate from actually improving the model itself.
    """
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1s = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(precisions[:-1]),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    if f1s.size == 0:
        return 0.0, 0.5
    best_idx = int(np.argmax(f1s))
    return float(f1s[best_idx]), float(thresholds[best_idx])


def run_cv(X, y, groups, tuned_model_fns, fault_name):
    n_splits = min(config.N_SPLITS, int(y.sum()), int(len(y) - y.sum()))
    n_splits = max(n_splits, 2)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []

    candidates = dict(tuned_model_fns)
    candidates["Ensemble(soft-vote)"] = lambda: SoftVotingEnsemble(tuned_model_fns)

    for model_name, model_fn in candidates.items():
        fold_f1, fold_prec, fold_rec, fold_auc = [], [], [], []
        fold_f1_opt, fold_thresh_opt = [], []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_res, y_train_res = resample(X_train, y_train)

            model = model_fn()
            model.fit(X_train_res, y_train_res)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

            fold_f1.append(f1_score(y_test, preds, zero_division=0))
            fold_prec.append(precision_score(y_test, preds, zero_division=0))
            fold_rec.append(recall_score(y_test, preds, zero_division=0))
            try:
                fold_auc.append(roc_auc_score(y_test, probs))
            except ValueError:
                fold_auc.append(np.nan)

            f1_opt, thresh_opt = best_f1_at_optimal_threshold(y_test, probs)
            fold_f1_opt.append(f1_opt)
            fold_thresh_opt.append(thresh_opt)

        fold_results.append({
            "fault": fault_name, "model": model_name,
            "f1_mean": np.mean(fold_f1), "f1_std": np.std(fold_f1),
            "precision_mean": np.mean(fold_prec),
            "recall_mean": np.mean(fold_rec),
            "auc_mean": np.nanmean(fold_auc),
            "f1_at_optimal_threshold_mean": np.mean(fold_f1_opt),
            "optimal_threshold_mean": np.mean(fold_thresh_opt),
        })
        print(f"  {model_name:22s} F1={np.mean(fold_f1):.3f}±{np.std(fold_f1):.3f}  "
              f"Prec={np.mean(fold_prec):.3f}  Rec={np.mean(fold_rec):.3f}  AUC={np.nanmean(fold_auc):.3f}  "
              f"F1@optThresh={np.mean(fold_f1_opt):.3f} (thresh≈{np.mean(fold_thresh_opt):.2f})")

    return fold_results


def main(domain: str = "acceleration"):
    config.set_domain(domain)
    print(f"Loading labeled dataset: {config.LABELED_CSV}")
    df = pd.read_csv(config.LABELED_CSV)

    if "all_peak_frequencies_parsed" not in df.columns:
        df, label_cols, leakage_cols = label_dataset(df)
    else:
        label_cols = [f"{f}_label" for f in config.FAULT_COLS]
        leakage_cols = []
        for f in config.FAULT_COLS:
            leakage_cols += [f"{f}_confidence", f"{f}_harmonics_matched",
                              f"{f}_sidebands_found", f"{f}_snr"]

    print(f"\nFiltering to training statuses: {config.TRAIN_STATUSES}")
    train_df = df[df["status"].isin(config.TRAIN_STATUSES)].reset_index(drop=True)
    print(f"Training rows: {len(train_df)} / {len(df)} total")
    train_df.to_csv(config.TRAIN_CSV, index=False)

    X, feature_cols, categorical_cols, encoders, medians = build_feature_frame(
        train_df, label_cols, leakage_cols
    )
    groups = train_df["bearingLocationId"]

    assert not (set(feature_cols) & set(leakage_cols)), "LEAKAGE DETECTED IN FEATURE_COLS"
    assert not (set(feature_cols) & set(label_cols)), "LABEL COLUMN IN FEATURE_COLS"
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    all_results = []
    final_models = {}

    for fault in config.FAULT_COLS:
        y = train_df[f"{fault}_label"].values
        print(f"\n{'='*60}\n{fault} — positive rate: {y.mean():.3%}\n{'='*60}")

        if y.sum() < config.N_SPLITS:
            print(f"  SKIPPING {fault}: too few positives ({int(y.sum())}) for {config.N_SPLITS}-fold CV")
            continue

        print("  Tuning hyperparameters (RandomizedSearchCV per model)...")
        tuned_model_fns = {}
        for model_name in BASE_MODELS:
            best_params = tune_hyperparams(X, y, groups, model_name)
            tuned_model_fns[model_name] = make_tuned_model_fn(model_name, best_params)
            print(f"    {model_name}: {best_params}")

        fold_results = run_cv(X, y, groups, tuned_model_fns, fault)
        all_results.extend(fold_results)

        best = max(fold_results, key=lambda r: r["f1_mean"])
        best_name = best["model"]
        final_model_fn = (
            (lambda: SoftVotingEnsemble(tuned_model_fns)) if best_name.startswith("Ensemble")
            else tuned_model_fns[best_name]
        )

        X_res, y_res = resample(X, y)
        model = final_model_fn()
        model.fit(X_res, y_res)
        final_models[fault] = {"model": model, "model_name": best_name}
        print(f"{fault}: refit {best_name} (F1={best['f1_mean']:.3f}) on {len(X)} rows -> {len(X_res)} after SMOTETomek")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(config.CV_SUMMARY_CSV, index=False)
    print(f"\nWrote CV summary -> {config.CV_SUMMARY_CSV}")

    for fault, payload in final_models.items():
        joblib.dump(payload["model"], config.MODELS_DIR / f"{fault}_model.joblib")

    joblib.dump(encoders, config.MODELS_DIR / "label_encoders.joblib")
    joblib.dump(medians, config.MODELS_DIR / "impute_medians.joblib")

    meta = {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "faults_trained": list(final_models.keys()),
        "model_names": {f: p["model_name"] for f, p in final_models.items()},
        "label_threshold_mode": config.ACTIVE_MODE,
        "label_threshold_value": config.LABEL_THRESHOLD[config.ACTIVE_MODE],
    }
    with open(config.MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(final_models)} models -> {config.MODELS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["acceleration", "velocity"], default="acceleration")
    args = parser.parse_args()
    main(domain=args.domain)
