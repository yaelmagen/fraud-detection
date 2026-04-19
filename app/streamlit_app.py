"""
Streamlit-based Fraud Decision System Simulation.

Simulates a real-time transaction stream where each incoming event is
processed, scored, explained, and stored in a dynamic history.

Key lifecycle:
  - Models are loaded from pretrained artifacts (models/ directory)
  - Retraining is manual, triggered from the sidebar
  - 80/20 split: first 80% for warm-up / training, last 20% for simulation

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from src.feature_store import FeatureStore
from src.feature_engine import (
    compute_features,
    get_region,
    get_geo_location,
    FINAL_FEATURES,
    COLD_START_TIME_DELTA,
)
from src.models import FraudEnsemble, CAT_FEATURES
from src.scoring import ScoreCalibrator
from src.explainability import Explainer, human_readable_reasons
from src import model_registry
from src.drift import read_trigger, check_drift, clear_trigger


# ======================================================================
# Paths & constants
# ======================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
XLSX_PATH = os.path.join(DATA_DIR, "DS_Test_Dataset.xlsx")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")

WARMUP_RATIO = 0.80  # first 80% for warm-up / training


# ======================================================================
# Data loading & preprocessing (run once, cached)
# ======================================================================
@st.cache_data(show_spinner="Loading dataset …")
def load_raw_data():
    df = pd.read_excel(XLSX_PATH)
    df["payment_timestamp"] = pd.to_datetime(df["payment_timestamp"])
    df["first_approved_payment_timestamp"] = pd.to_datetime(
        df["first_approved_payment_timestamp"]
    )
    df = df.sort_values("payment_timestamp").reset_index(drop=True)

    # --- Feature engineering (same as notebook) ---
    df["region"] = df["country"].apply(get_region)
    df["geo_location"] = df.apply(
        lambda r: get_geo_location(str(r["country"]), r.get("state")), axis=1
    )
    df["is_us_transaction"] = (df["country"] == "US").astype(int)
    df["device_type"] = df["device_type"].fillna("Unknown")
    df["is_broken_record"] = (df["device_type"] == "Unknown").astype(int)
    df["payment_instrument"] = df["payment_instrument"].fillna("Unknown")
    df["seniority"] = (
        df["payment_timestamp"] - df["first_approved_payment_timestamp"]
    ).dt.days
    df["hour"] = df["payment_timestamp"].dt.hour
    df["is_new_user"] = (df["num_approved_payments_per_user"] == 1).astype(int)

    df = df.sort_values(["user_id", "payment_timestamp"])
    df["time_since_last_payment"] = (
        df.groupby("user_id")["payment_timestamp"]
        .diff()
        .dt.total_seconds()
        / 3600
    )
    df["time_since_last_payment"] = df["time_since_last_payment"].fillna(
        COLD_START_TIME_DELTA
    )
    df["prev_avg_amount"] = df.groupby("user_id")[
        "total_amount_usd"
    ].transform(lambda x: x.shift(1).expanding().mean())
    df["amount_to_avg_ratio"] = (
        df["total_amount_usd"] / df["prev_avg_amount"]
    ).fillna(1.0)
    df.drop(columns=["prev_avg_amount"], inplace=True)
    df["is_same_as_prev_amount"] = (
        df["total_amount_usd"]
        == df.groupby("user_id")["total_amount_usd"].shift(1)
    ).astype(int)

    temp = df.set_index("payment_timestamp")
    rolling_count = (
        temp.groupby(["user_id", "total_amount_usd"])["total_amount_usd"]
        .rolling("24h", closed="left")
        .count()
    )
    df["same_amount_count_24h"] = rolling_count.values
    df["same_amount_count_24h"] = df["same_amount_count_24h"].fillna(0)

    df["prev_region"] = df.groupby("user_id")["region"].shift(1)
    df["prev_country"] = df.groupby("user_id")["country"].shift(1)
    df["prev_state"] = df.groupby("user_id")["state"].shift(1)
    df["time_diff_hours"] = (
        df.groupby("user_id")["payment_timestamp"]
        .diff()
        .dt.total_seconds()
        / 3600
    )
    df["is_impossible_travel"] = (
        (df["region"] != df["prev_region"])
        & df["prev_region"].notna()
        & (df["time_diff_hours"] < 10)
    ).astype(int)
    df["is_suspicious_velocity"] = (
        (df["region"] == df["prev_region"])
        & df["prev_region"].notna()
        & (
            ((df["country"] != df["prev_country"]) & df["prev_country"].notna())
            | (
                (df["country"] == "US")
                & (df["state"] != df["prev_state"])
                & df["prev_state"].notna()
            )
        )
        & (df["time_diff_hours"] < 3)
    ).astype(int)
    df.drop(
        columns=["prev_region", "prev_country", "prev_state", "time_diff_hours"],
        inplace=True,
        errors="ignore",
    )
    df = df.sort_values("payment_timestamp").reset_index(drop=True)
    return df


# ======================================================================
# Load pretrained models (default path — no retraining)
# ======================================================================
@st.cache_resource(show_spinner="Loading pretrained models …")
def load_pretrained_system(_version_key: int):
    """Load pretrained models + bootstrap feature store.

    ``_version_key`` is used as a cache key so the resource is
    reloaded when the version changes (e.g. after retraining).
    """
    raw = load_raw_data()
    split_idx = int(len(raw) * WARMUP_RATIO)
    warmup_df = raw.iloc[:split_idx].copy()
    live_df = raw.iloc[split_idx:].copy().reset_index(drop=True)

    # Feature Store (warm-up data only)
    store = FeatureStore(HISTORY_PATH)
    store.bootstrap(warmup_df)

    # Load pretrained ensemble
    ensemble = FraudEnsemble.from_pretrained()
    metadata = ensemble._metadata

    # Restore calibrator from saved bounds
    calibrator = ScoreCalibrator()
    cb = metadata["calibrator_bounds"]
    calibrator.if_min = cb["if_min"]
    calibrator.if_max = cb["if_max"]
    calibrator.lof_min = cb["lof_min"]
    calibrator.lof_max = cb["lof_max"]

    # SHAP explainer
    bg_sample = warmup_df.sample(n=min(200, len(warmup_df)), random_state=42)
    explainer = Explainer(ensemble)
    explainer.build(bg_sample)

    return store, ensemble, calibrator, explainer, live_df, warmup_df


# ======================================================================
# Retrain models (manual, from UI)
# ======================================================================
def retrain_models(warmup_df: pd.DataFrame):
    """Train new models on warm-up data and save as next version."""
    ensemble = FraudEnsemble()
    ensemble.fit(warmup_df)

    # Compute calibration
    X = warmup_df[FINAL_FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    X_proc = ensemble.preprocessor.transform(X)
    X_scaled = ensemble.scaler.transform(X_proc)
    if_scores = ensemble.iso_forest.decision_function(X_proc)
    lof_scores = ensemble.lof.decision_function(X_scaled)
    calibrator_bounds = {
        "if_min": float(np.min(if_scores)),
        "if_max": float(np.max(if_scores)),
        "lof_min": float(np.min(lof_scores)),
        "lof_max": float(np.max(lof_scores)),
    }

    ver = ensemble.save(
        calibrator_bounds=calibrator_bounds,
        training_rows=len(warmup_df),
    )
    clear_trigger()
    return ver


# ======================================================================
# Session-state helpers
# ======================================================================
def _init_session_state():
    if "txn_idx" not in st.session_state:
        st.session_state.txn_idx = 0
    if "scored_history" not in st.session_state:
        st.session_state.scored_history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "model_version_key" not in st.session_state:
        st.session_state.model_version_key = model_registry.latest_version() or 0


# ======================================================================
# Feedback persistence
# ======================================================================
def _save_feedback(payment_id, label):
    row = {
        "payment_id": payment_id,
        "feedback": label,
        "timestamp": pd.Timestamp.now(),
    }
    if os.path.exists(FEEDBACK_PATH):
        fb = pd.read_csv(FEEDBACK_PATH)
    else:
        fb = pd.DataFrame()
    fb = pd.concat([fb, pd.DataFrame([row])], ignore_index=True)
    fb.to_csv(FEEDBACK_PATH, index=False)


# ======================================================================
# UI
# ======================================================================
def main():
    st.set_page_config(
        page_title="Fraud Decision System",
        page_icon="🛡️",
        layout="wide",
    )

    _init_session_state()

    # ------------------------------------------------------------------
    # Sidebar — Model Management
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Model Management")

        # Show available versions
        versions = model_registry.list_versions()
        if versions:
            st.info(f"**Available versions:** {', '.join(f'v{v}' for v in versions)}")
            st.caption(f"Currently loaded: **v{st.session_state.model_version_key}**")
        else:
            st.error("No pretrained models found. Click **Retrain Models** below.")

        st.markdown("---")

        # Load Pretrained Models button
        if st.button("📦 Load Pretrained Models", use_container_width=True):
            lv = model_registry.latest_version()
            if lv:
                st.session_state.model_version_key = lv
                st.session_state.txn_idx = 0
                st.session_state.scored_history = []
                st.session_state.current_result = None
                load_pretrained_system.clear()
                st.success(f"Loaded **v{lv}**. Simulation reset.")
                st.rerun()
            else:
                st.error("No models found. Train first.")

        # Retrain Models button
        if st.button("🔄 Retrain Models", use_container_width=True):
            with st.spinner("Retraining … this may take a minute."):
                raw = load_raw_data()
                split_idx = int(len(raw) * WARMUP_RATIO)
                warmup_df = raw.iloc[:split_idx].copy()
                ver = retrain_models(warmup_df)
            st.session_state.model_version_key = ver
            st.session_state.txn_idx = 0
            st.session_state.scored_history = []
            st.session_state.current_result = None
            load_pretrained_system.clear()
            st.success(f"Models retrained and saved as **v{ver}**")
            st.rerun()

        st.markdown("---")

        # Drift detection
        st.subheader("Drift Detection")
        trigger = read_trigger()
        if trigger.get("retrain_required"):
            st.warning(
                f"⚠️ Data drift detected — retraining recommended.\n\n"
                f"**Reason:** {trigger.get('reason', 'N/A')}"
            )
        else:
            st.success("✅ No drift detected.")
        if trigger.get("last_checked"):
            st.caption(f"Last checked: {trigger['last_checked']}")

    # ------------------------------------------------------------------
    # Load system (pretrained)
    # ------------------------------------------------------------------
    if not versions:
        st.error(
            "No pretrained models found in `models/`. "
            "Use the **Retrain Models** button in the sidebar or run "
            "`python scripts/train_models.py`."
        )
        st.stop()

    store, ensemble, calibrator, explainer, live_df, warmup_df = (
        load_pretrained_system(st.session_state.model_version_key)
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.markdown(
        """
        <h1 style='text-align:center;'>🛡️ Real-Time Fraud Decision System</h1>
        <p style='text-align:center; color:gray;'>
            Hybrid Ensemble: Isolation Forest + LOF + Deterministic Rules
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Progress
    total = len(live_df)
    idx = st.session_state.txn_idx
    st.progress(min(idx / max(total, 1), 1.0), text=f"Processed {idx} / {total} live transactions")

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        process_next = st.button("⏩ Process Next", type="primary", use_container_width=True)
    with col_btn2:
        process_batch = st.button("⏭️ Process 10", use_container_width=True)
    with col_btn3:
        st.write("")  # spacer

    # ------------------------------------------------------------------
    # Process transaction(s)
    # ------------------------------------------------------------------
    n_to_process = 0
    if process_next:
        n_to_process = 1
    elif process_batch:
        n_to_process = 10

    for _ in range(n_to_process):
        if st.session_state.txn_idx >= total:
            st.warning("All live transactions have been processed.")
            break

        txn = live_df.iloc[st.session_state.txn_idx]

        # 1. Retrieve user state from feature store
        user_state = store.get_user_state(txn["user_id"])

        # 2. Compute point-in-time features
        features = compute_features(txn, user_state)

        # 3. Score
        if_raw, lof_raw = ensemble.raw_scores(features)
        risk_score, decision, breakdown = calibrator.score(
            if_raw, lof_raw, features
        )

        # 4. Update feature store (with region info for geo flags)
        txn_with_region = txn.copy()
        txn_with_region["region"] = get_region(str(txn["country"]))
        store.append_transaction(txn_with_region)

        result = {
            "payment_id": txn["payment_id"],
            "user_id": txn["user_id"],
            "amount": txn["total_amount_usd"],
            "country": txn["country"],
            "timestamp": str(txn["payment_timestamp"]),
            "risk_score": risk_score,
            "decision": decision,
            "breakdown": breakdown,
            "features": features,
        }

        st.session_state.scored_history.append(result)
        st.session_state.current_result = result
        st.session_state.txn_idx += 1

    # --- Run drift check after every batch ---
    if st.session_state.txn_idx > 0 and st.session_state.txn_idx % 10 == 0:
        processed = live_df.iloc[: st.session_state.txn_idx]
        if len(processed) >= 10:
            check_drift(warmup_df, processed)

    # ------------------------------------------------------------------
    # Current transaction details
    # ------------------------------------------------------------------
    result = st.session_state.current_result
    if result is not None:
        st.markdown("---")
        ts_str = pd.to_datetime(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        lcol, rcol = st.columns([2, 3])
        with lcol:
            st.subheader("Latest Transaction")
        with rcol:
            st.markdown(
                f"<p style='margin-top:12px; color:gray; font-size:1.1em;'>"
                f"🕐 {ts_str}</p>",
                unsafe_allow_html=True,
            )

        # Color-coded decision badge
        color_map = {
            "Approve": "#27ae60",
            "Review": "#f39c12",
            "Block": "#e74c3c",
        }
        badge_color = color_map.get(result["decision"], "#95a5a6")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Payment ID", result["payment_id"])
        with col2:
            st.metric("Amount (USD)", f"${result['amount']:,.2f}")
        with col3:
            st.metric("Risk Score", f"{result['risk_score']:.4f}")
        with col4:
            st.markdown(
                f"<div style='background-color:{badge_color}; color:white; "
                f"padding:18px; border-radius:10px; text-align:center; "
                f"font-size:1.3em; font-weight:bold;'>"
                f"{result['decision']}</div>",
                unsafe_allow_html=True,
            )

        # Score breakdown
        st.markdown("#### Score Breakdown")
        bd = result["breakdown"]
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("IF Risk", f"{bd['if_risk']:.4f}")
        bcol2.metric("LOF Risk", f"{bd['lof_risk']:.4f}")
        bcol3.metric("ML Combined", f"{bd['ml_risk']:.4f}")
        bcol4.metric("Hard Flag", "🚨 YES" if bd["hard_flag"] else "✅ No")

        # Explainability
        st.markdown("#### Explanation")
        reasons = human_readable_reasons(result["features"], result["breakdown"], result["decision"])
        for r in reasons:
            st.markdown(f"- {r}")

        # SHAP waterfall
        with st.expander("📊 SHAP Waterfall Plot", expanded=False):
            try:
                fig = explainer.waterfall_figure(result["features"])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP plot error: {e}")

        # Key features table
        with st.expander("🔎 Feature Details", expanded=False):
            display_feats = {
                k: v
                for k, v in result["features"].items()
                if k in FINAL_FEATURES
            }
            st.dataframe(
                pd.DataFrame([display_feats]),
                use_container_width=True,
            )

        # ------------------------------------------------------------------
        # Feedback buttons
        # ------------------------------------------------------------------
        st.markdown("#### Analyst Feedback")
        fb1, fb2, fb3 = st.columns([1, 1, 2])
        with fb1:
            if st.button("🚫 Confirm Fraud", key=f"fraud_{result['payment_id']}"):
                _save_feedback(result["payment_id"], "fraud")
                st.success("Feedback saved: **Confirmed Fraud**")
        with fb2:
            if st.button("✅ False Positive", key=f"fp_{result['payment_id']}"):
                _save_feedback(result["payment_id"], "false_positive")
                st.success("Feedback saved: **False Positive**")

    # ------------------------------------------------------------------
    # History table
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Transaction History")
    if st.session_state.scored_history:
        hist_df = pd.DataFrame(st.session_state.scored_history)[
            [
                "payment_id",
                "user_id",
                "amount",
                "country",
                "timestamp",
                "risk_score",
                "decision",
            ]
        ]

        def _color_decision(val):
            colors = {
                "Approve": "background-color: #d4edda; color: #155724;",
                "Review": "background-color: #fff3cd; color: #856404;",
                "Block": "background-color: #f8d7da; color: #721c24;",
            }
            return colors.get(val, "")

        styled = hist_df[::-1].style.map(
            _color_decision, subset=["decision"]
        )
        st.dataframe(styled, use_container_width=True, height=400)

        # Summary stats
        scol1, scol2, scol3 = st.columns(3)
        decisions = hist_df["decision"].value_counts()
        scol1.metric(
            "Approved",
            decisions.get("Approve", 0),
        )
        scol2.metric(
            "Under Review",
            decisions.get("Review", 0),
        )
        scol3.metric(
            "Blocked",
            decisions.get("Block", 0),
        )
    else:
        st.info("Click **Process Next** to start the simulation.")


if __name__ == "__main__":
    main()
