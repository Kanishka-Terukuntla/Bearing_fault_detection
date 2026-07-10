"""
Local Streamlit dashboard for the bearing fault detection project.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="Bearing Fault Detection", layout="wide")
st.title("Bearing Fault Detection — Monitoring Dashboard")
st.caption("Local-only Streamlit app. Nothing here is sent anywhere except direct calls to the AAMS API.")

# Global domain selector — controls Dataset Overview, Model Performance, and
# Normal-Bearing Flag Check tabs (they read whatever config.CURRENT_DOMAIN is
# set to). Defaults to velocity. The Daily Live Predictions tab has its own
# separate mode selector (velocity/mixed/single) since a daily run can score
# with a specific domain regardless of what you're browsing here.
selected_domain = st.sidebar.radio(
    "Domain (Overview / Model Performance / Normal-Bearing Flag Check tabs)",
    options=["velocity", "acceleration"],
    index=0,
    help="Which domain's outputs/models these three tabs read. Does not affect the "
         "Daily Live Predictions tab, which has its own mode selector.",
)
config.set_domain(selected_domain)
st.sidebar.caption(f"Currently viewing: **{selected_domain}** "
                    f"(`{config.LABELED_CSV.name}`, `{config.MODELS_DIR.name}/`)")


def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


tab_overview, tab_models, tab_normal_check, tab_daily = st.tabs(
    ["📊 Dataset Overview", "🤖 Model Performance", "✅ Normal-Bearing Flag Check", "📅 Daily Live Predictions"]
)

# =============================================================================
# TAB 1 — Dataset overview
# =============================================================================
with tab_overview:
    labeled = safe_read_csv(config.LABELED_CSV)
    if labeled is None:
        st.warning(
            f"No labeled dataset found at `{config.LABELED_CSV}`.\n\n"
            "Run, in order:\n"
            "```\npython src/build_dataset.py\npython src/labeling.py\n```"
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total packets", f"{len(labeled):,}")
        c2.metric("Distinct bearings", f"{labeled['bearingLocationId'].nunique():,}")
        c3.metric("Date range", f"{labeled['date'].min()} → {labeled['date'].max()}")

        st.subheader("Status distribution")
        status_counts = labeled["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.plotly_chart(px.bar(status_counts, x="status", y="count", color="status"),
                         use_container_width=True)

        st.subheader("Fault label positive rate (confidence-based labeling)")
        label_cols = [f"{f}_label" for f in config.FAULT_COLS if f"{f}_label" in labeled.columns]
        if label_cols:
            rates = labeled[label_cols].mean().reset_index()
            rates.columns = ["fault", "positive_rate"]
            rates["fault"] = rates["fault"].str.replace("_label", "")
            st.plotly_chart(px.bar(rates, x="fault", y="positive_rate", text_auto=".1%"),
                             use_container_width=True)

            st.subheader("Multi-fault overlap")
            overlap = (labeled[label_cols].sum(axis=1) > 1).sum()
            st.write(f"Rows flagged with more than one fault by the labeling heuristic: **{overlap}**")

        with st.expander("Preview labeled data"):
            st.dataframe(labeled.head(200))

# =============================================================================
# TAB 2 — Model performance (CV summary)
# =============================================================================
with tab_models:
    summary = safe_read_csv(config.CV_SUMMARY_CSV)
    meta_path = config.MODELS_DIR / "meta.json"

    if summary is None or not meta_path.exists():
        st.warning(
            "No trained models found yet. Run:\n```\npython src/train.py\n```"
        )
    else:
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        st.write(f"Trained on status: `{config.TRAIN_STATUSES}` · "
                 f"Label threshold mode: `{meta['label_threshold_mode']}` "
                 f"(confidence ≥ {meta['label_threshold_value']})")

        st.subheader("Cross-validated performance by fault & model")
        st.dataframe(summary.style.format({
            "f1_mean": "{:.3f}", "f1_std": "{:.3f}", "precision_mean": "{:.3f}",
            "recall_mean": "{:.3f}", "auc_mean": "{:.3f}",
        }))

        best_per_fault = summary.loc[summary.groupby("fault")["f1_mean"].idxmax()]
        st.subheader("Selected (best) model per fault")
        st.dataframe(best_per_fault[["fault", "model", "f1_mean", "precision_mean", "recall_mean", "auc_mean"]])

        fig = px.bar(summary, x="fault", y="f1_mean", color="model", barmode="group",
                     error_y="f1_std", title="F1 score by fault and model")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3 — Normal-bearing false positive check
# =============================================================================
with tab_normal_check:
    st.write(
        "The model is trained **only** on Marginal/Unacceptable rows. This tab checks how many "
        "Normal-status bearings the trained model still flags as faulty — an approximate "
        "false-positive / over-triggering rate."
    )
    normal_report = safe_read_csv(config.NORMAL_FLAG_REPORT_CSV)

    if normal_report is None:
        st.warning(
            "No report yet. Run:\n```\npython src/evaluate_normal.py\n```"
        )
    else:
        n = len(normal_report)
        flagged = normal_report[normal_report["any_fault_pred"] == 1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal-status packets evaluated", f"{n:,}")
        c2.metric("Flagged as faulty", f"{len(flagged):,}")
        c3.metric("Flag rate", f"{len(flagged) / n:.2%}" if n else "—")

        st.subheader("Flag rate by fault type")
        fault_names = [f for f in config.FAULT_COLS if f"{f}_pred" in normal_report.columns]
        rates = pd.DataFrame({
            "fault": fault_names,
            "flag_rate": [normal_report[f"{f}_pred"].mean() for f in fault_names],
        })
        st.plotly_chart(px.bar(rates, x="fault", y="flag_rate", text_auto=".1%"), use_container_width=True)

        st.subheader("Distinct Normal bearings with ≥1 flagged packet")
        per_bearing = (
            flagged.groupby(["bearingLocationId", "predicted_faults"])
            .size()
            .reset_index(name="flagged_packets")
            .sort_values("flagged_packets", ascending=False)
        )
        st.dataframe(per_bearing)
        st.download_button("Download full normal-flag report (CSV)",
                            normal_report.to_csv(index=False), file_name="normal_flag_report.csv")

# =============================================================================
# TAB 4 — Daily live predictions
# =============================================================================
with tab_daily:
    st.write("Pull today's (or any date's) waveform data straight from the AAMS API, score it, "
             "and see which bearings are flagged.")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        run_date = st.date_input("Date to fetch")
    with col_b:
        run_mode = st.radio(
            "Prediction mode",
            options=["Velocity only (recommended)", "Mixed", "Single domain"],
            help="Velocity only: fast path, scores everything with the velocity models — "
                 "this is the default per current guidance to use velocity. Mixed: BPFI/BPFO/BSF "
                 "via acceleration, FTF via velocity (config.FAULT_DOMAIN). Single domain: one "
                 "domain's models for everything, for diagnostics/comparison.",
        )
        single_domain_choice = None
        if run_mode == "Single domain":
            single_domain_choice = st.selectbox("Domain", options=["acceleration", "velocity"])

    with col_c:
        workers = st.number_input("Parallel workers", min_value=5, max_value=100, value=40, step=5)
        deadline_min = st.number_input("Deadline (minutes)", min_value=5, max_value=180, value=30, step=5)

    run_button = st.button("Fetch & Predict", type="primary")

    if run_button:
        with st.spinner(f"Pulling waveforms + predicting for {run_date}... this can take a while for many bearings"):
            try:
                date_str = run_date.strftime("%Y-%m-%d")
                deadline_sec = int(deadline_min * 60)
                if run_mode == "Velocity only (recommended)":
                    from src.predict_daily import run_velocity
                    result = run_velocity(date_str, max_workers=workers, max_total_seconds=deadline_sec)
                    latest_filename = "daily_predictions_latest_velocity.csv"
                elif run_mode == "Mixed":
                    from src.predict_daily import run_mixed
                    result = run_mixed(date_str, max_workers=workers, max_total_seconds=deadline_sec)
                    latest_filename = "daily_predictions_latest_mixed.csv"
                else:
                    from src.predict_daily import run_single_domain
                    result = run_single_domain(date_str, single_domain_choice,
                                                max_workers=workers, max_total_seconds=deadline_sec)
                    latest_filename = "daily_predictions_latest.csv"

                st.success(f"Done — {result['bearingLocationId'].nunique()} bearings scored.")
                st.session_state["daily_latest_filename"] = latest_filename
            except Exception as e:
                st.error(f"Prediction run failed: {e}")

    latest_filename = st.session_state.get("daily_latest_filename", "daily_predictions_latest_velocity.csv")
    latest = safe_read_csv(config.OUTPUTS_DIR / latest_filename)
    # fall back to whichever one exists if the preferred one doesn't
    if latest is None:
        for fallback in ["daily_predictions_latest_velocity.csv", "daily_predictions_latest_mixed.csv",
                          "daily_predictions_latest.csv"]:
            latest = safe_read_csv(config.OUTPUTS_DIR / fallback)
            if latest is not None:
                latest_filename = fallback
                break

    if latest is None:
        st.info("No prediction run yet. Click **Fetch & Predict** above, "
                 "or run `python -m src.predict_daily` from the command line.")
    else:
        mode_label = "mixed" if "mixed" in latest_filename else latest.get("domain", pd.Series(["?"])).iloc[0]
        st.subheader(f"Latest run — {latest['run_date'].iloc[0] if 'run_date' in latest.columns else ''} "
                     f"({mode_label})")

        flagged = latest[latest["any_fault_pred"] == 1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Bearings scored", f"{latest['bearingLocationId'].nunique():,}")
        c2.metric("Bearings flagged", f"{flagged['bearingLocationId'].nunique():,}")
        c3.metric("Flag rate", f"{flagged['bearingLocationId'].nunique() / max(latest['bearingLocationId'].nunique(),1):.2%}")

        if "status" in latest.columns:
            status_options = sorted(latest["status"].dropna().unique())
            status_filter = st.multiselect("Filter by status", options=status_options, default=status_options)
            view = latest[latest["status"].isin(status_filter)]
        else:
            view = latest

        display_cols = [c for c in ["bearingLocationId", "machineName", "customerName", "status", "axis",
                                     "geometry_known", "predicted_faults", "any_fault_pred"] if c in view.columns]
        display_cols += [f"{f}_prob" for f in config.FAULT_COLS if f"{f}_prob" in view.columns]

        st.dataframe(view[display_cols].sort_values("any_fault_pred", ascending=False))

        if "geometry_known" in view.columns and (~view["geometry_known"]).any():
            n_unknown = (~view["geometry_known"]).sum()
            st.caption(f"⚠️ {n_unknown} row(s) above have `geometry_known = False` — bearing geometry "
                       f"wasn't found in the live API or your local catalogue.json, so those predictions "
                       f"fall back to training-set median values and are less reliable. Consider adding "
                       f"these bearings' geometry to catalogue.json.")

        st.download_button("Download latest predictions (CSV)", latest.to_csv(index=False),
                            file_name=latest_filename)

    history = safe_read_csv(config.DAILY_PREDICTIONS_CSV)
    if history is not None and "run_date" in history.columns:
        st.subheader("Flag-rate trend over time")
        trend = (
            history.groupby("run_date")
            .apply(lambda g: g[g["any_fault_pred"] == 1]["bearingLocationId"].nunique() / max(g["bearingLocationId"].nunique(), 1))
            .reset_index(name="flag_rate")
        )
        st.plotly_chart(px.line(trend, x="run_date", y="flag_rate", markers=True), use_container_width=True)
