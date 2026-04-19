"""
Point-in-Time Feature Engine.

Computes features for a single incoming transaction using ONLY
data available up to (but not including) that transaction.
All user-level aggregates come from the FeatureStore cache —
no leakage, no full-history recomputation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ======================================================================
# Region mapping (mirrors notebook logic)
# ======================================================================
_REGION_MAP = {
    "Central_West_Europe": [
        "GB", "DE", "FR", "IT", "ES", "NL", "NO", "PT", "CH", "AT", "SE",
        "DK", "FI", "BE", "IE", "GR", "LU", "IS", "MT", "AD", "MC", "GL",
        "AX", "FO", "GG", "JE", "LI", "PL", "CZ", "SK", "HU", "SI",
    ],
    "East_Europe_Balkans": [
        "RO", "UA", "MD", "EE", "LT", "LV", "RS", "MK", "BA", "AL", "ME",
    ],
    "ME_Caucasus": ["IL", "CY", "AM", "GE", "KZ", "AZ", "TR"],
    "North_America": [
        "US", "CA", "MX", "PR", "AG", "AW", "BB", "BM", "BQ", "BS", "CW",
        "DM", "DO", "GD", "GP", "GU", "LC", "MF", "MQ", "TC", "VG", "VI",
    ],
    "South_Latam_America": [
        "BR", "CL", "CO", "PE", "AR", "UY", "EC", "GT", "HN", "CR", "PY",
        "SV", "NI", "GF", "GY",
    ],
    "Asia_Oceania": [
        "SG", "JP", "AE", "IN", "TW", "MY", "HK", "BH", "KW", "QA", "KH",
        "ID", "LA", "KR", "UZ", "TH", "NP", "MN", "TJ", "LK", "PK", "OM",
        "NC", "PF", "TO", "TV",
    ],
    "Africa": [
        "CI", "MA", "MU", "BJ", "GH", "ET", "BW", "AO", "RW", "CG", "LR",
        "DZ", "EG", "MG", "TN", "TG", "GQ", "RE", "SC",
    ],
    "Oceania": ["AU", "NZ", "CK"],
}

_COUNTRY_TO_REGION: Dict[str, str] = {}
for _region, _countries in _REGION_MAP.items():
    for _c in _countries:
        _COUNTRY_TO_REGION[_c] = _region


def get_region(country: str) -> str:
    return _COUNTRY_TO_REGION.get(country, "Other")


# ======================================================================
# Geo-location helper
# ======================================================================
def get_geo_location(country: str, state) -> str:
    if country == "US" and pd.notna(state):
        return f"US_{state}"
    return str(country)


# ======================================================================
# Feature computation (point-in-time)
# ======================================================================
COLD_START_TIME_DELTA = 999999.0  # hours, same as notebook


def compute_features(
    txn: pd.Series,
    user_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the full feature vector for *txn* using only prior history
    available via *user_state* (from FeatureStore).

    Returns a dict of feature_name -> value, matching `final_features`
    used during model training.
    """
    ts = pd.Timestamp(txn["payment_timestamp"])
    amount = float(txn["total_amount_usd"])
    country = str(txn["country"])
    state = txn.get("state", None)
    device = txn.get("device_type", None)
    instrument = txn.get("payment_instrument", None)
    currency = str(txn["currency"])
    merchant = txn["merchant_id"]
    first_approved = pd.Timestamp(txn["first_approved_payment_timestamp"])

    region = get_region(country)
    geo_location = get_geo_location(country, state)

    # --- Cold-start defaults ---
    is_new_user = 1
    amount_to_avg_ratio = 1.0
    same_amount_count_24h = 0.0
    time_since_last_payment = COLD_START_TIME_DELTA
    is_impossible_travel = 0
    is_suspicious_velocity = 0
    is_same_as_prev_amount = 0

    if user_state is not None and user_state["txn_count"] > 0:
        is_new_user = 0

        # amount_to_avg_ratio
        prev_avg = user_state["amount_avg"]
        if prev_avg > 0:
            amount_to_avg_ratio = amount / prev_avg
        else:
            amount_to_avg_ratio = 1.0

        # time_since_last_payment (hours)
        last_ts = pd.Timestamp(user_state["last_timestamp"])
        delta_seconds = (ts - last_ts).total_seconds()
        time_since_last_payment = delta_seconds / 3600.0

        # same_amount_count_24h (from 24h rolling cache)
        cutoff_24h = ts - pd.Timedelta(hours=24)
        same_count = sum(
            1
            for t, a in user_state["amounts_24h"]
            if pd.Timestamp(t) >= cutoff_24h and pd.Timestamp(t) < ts and a == amount
        )
        same_amount_count_24h = float(same_count)

        # is_same_as_prev_amount
        if user_state["amounts_24h"]:
            last_amount = user_state["amounts_24h"][-1][1]
            is_same_as_prev_amount = int(amount == last_amount)

        # Geographic flags
        prev_region = user_state.get("last_region")
        prev_country = user_state.get("last_country")
        prev_state = user_state.get("last_state")
        time_diff_hours = time_since_last_payment

        if prev_region and prev_region != region and time_diff_hours < 10:
            is_impossible_travel = 1

        if prev_region and prev_region == region:
            country_changed = prev_country is not None and prev_country != country
            state_changed = (
                country == "US"
                and prev_state is not None
                and state != prev_state
            )
            if (country_changed or state_changed) and time_diff_hours < 3:
                is_suspicious_velocity = 1

    # Seniority (days)
    seniority = (ts - first_approved).days

    # Broken record flag (notebook: only device_type == 'Unknown')
    is_broken_record = int(
        pd.isna(device) or str(device).strip() == ""
        or device == "Unknown"
    )

    # Handle missing categorical values
    device_clean = "Unknown" if (pd.isna(device) or str(device).strip() == "") else str(device)
    instrument_clean = "Unknown" if (pd.isna(instrument) or str(instrument).strip() == "") else str(instrument)

    # US transaction flag
    is_us_transaction = int(country == "US")

    # Hour (for cyclical encoding)
    hour = ts.hour

    # Time to complete payment
    time_to_complete = float(txn.get("time_to_complete_payment", 0))

    features = {
        "total_amount_usd": amount,
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "seniority": seniority,
        "time_since_last_payment": time_since_last_payment,
        "time_to_complete_payment": time_to_complete,
        "same_amount_count_24h": same_amount_count_24h,
        "is_same_as_prev_amount": is_same_as_prev_amount,
        "is_impossible_travel": is_impossible_travel,
        "is_suspicious_velocity": is_suspicious_velocity,
        "is_broken_record": is_broken_record,
        "is_us_transaction": is_us_transaction,
        "is_new_user": is_new_user,
        "geo_location": geo_location,
        "payment_instrument": instrument_clean,
        "device_type": device_clean,
        "currency": currency,
        "merchant_id": merchant,
        "hour": hour,
    }
    return features


# Ordered list that matches the training pipeline
FINAL_FEATURES = [
    "total_amount_usd",
    "amount_to_avg_ratio",
    "seniority",
    "time_since_last_payment",
    "time_to_complete_payment",
    "same_amount_count_24h",
    "is_same_as_prev_amount",
    "is_impossible_travel",
    "is_suspicious_velocity",
    "is_broken_record",
    "is_us_transaction",
    "is_new_user",
    "geo_location",
    "payment_instrument",
    "device_type",
    "currency",
    "merchant_id",
    "hour",
]
