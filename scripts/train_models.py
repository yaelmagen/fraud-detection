"""
Offline training script — trains the ensemble using the notebook's
data split strategy and saves versioned model artifacts to ``models/``.

Split strategy (matches notebook):
  - Last 100 rows → holdout (used as live stream in Streamlit)
  - Remaining rows → 65% train / 35% test (chronological)
  - SHAP explainer is built on test data

Usage:
    python scripts/train_models.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.feature_engine import (
    get_region,
    get_geo_location,
    FINAL_FEATURES,
    COLD_START_TIME_DELTA,
)
from src.models import FraudEnsemble, CAT_FEATURES

XLSX_PATH = os.path.join(PROJECT_ROOT, "data", "DS_Test_Dataset.xlsx")
TRAIN_RATIO = 0.65
HOLDOUT_SIZE = 100


def load_and_engineer(path: str) -> pd.DataFrame:
    """Load raw data and apply full feature engineering."""
    df = pd.read_excel(path)
    df["payment_timestamp"] = pd.to_datetime(df["payment_timestamp"]).dt.floor('s')
    df["first_approved_payment_timestamp"] = pd.to_datetime(
        df["first_approved_payment_timestamp"]
    ).dt.floor('s')
    df = df.sort_values("payment_timestamp").reset_index(drop=True)

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


def main():
    print("Loading and engineering features …")
    df = load_and_engineer(XLSX_PATH)

    # Split: holdout last 100, then 65/35 train/test on remainder
    holdout_df = df.iloc[-HOLDOUT_SIZE:].copy()
    remaining = df.iloc[:-HOLDOUT_SIZE]
    train_size = int(len(remaining) * TRAIN_RATIO)
    train_df = remaining.iloc[:train_size].copy()
    test_df = remaining.iloc[train_size:].copy()

    print(f"  Total rows : {len(df)}")
    print(f"  Train      : {len(train_df)} ({TRAIN_RATIO:.0%})")
    print(f"  Test       : {len(test_df)}")
    print(f"  Holdout    : {len(holdout_df)}")

    print("Training ensemble …")
    ensemble = FraudEnsemble()
    ensemble.fit(train_df)

    print("Computing calibration scores on training data …")
    X = train_df[FINAL_FEATURES].copy()
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
    print(f"  IF  range: [{calibrator_bounds['if_min']:.4f}, {calibrator_bounds['if_max']:.4f}]")
    print(f"  LOF range: [{calibrator_bounds['lof_min']:.4f}, {calibrator_bounds['lof_max']:.4f}]")

    print("Saving model artifacts …")
    ver = ensemble.save(
        calibrator_bounds=calibrator_bounds,
        training_rows=len(train_df),
    )
    print(f"  Saved as v{ver}")
    print("Done.")


if __name__ == "__main__":
    main()
