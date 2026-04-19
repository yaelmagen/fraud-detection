"""
Lightweight Drift Detection.

Provides a simple heuristic check comparing the live stream's
distribution against the training baseline.  When drift is detected
the retrain trigger file is updated so the UI can surface a warning.

Signals checked:
  - Shift in average transaction amount (>20% change)
  - Shift in top-3 country distribution (Jaccard distance)
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIGGER_PATH = os.path.join(_PROJECT_ROOT, "data", "retrain_trigger.json")

_DEFAULT_TRIGGER: Dict[str, Any] = {
    "retrain_required": False,
    "reason": "",
    "last_checked": "",
}


# ------------------------------------------------------------------
# Trigger file I/O
# ------------------------------------------------------------------
def read_trigger() -> Dict[str, Any]:
    """Read the retrain trigger file, creating a default if missing."""
    if not os.path.exists(TRIGGER_PATH):
        write_trigger(_DEFAULT_TRIGGER)
    with open(TRIGGER_PATH, "r") as f:
        return json.load(f)


def write_trigger(data: Dict[str, Any]) -> None:
    with open(TRIGGER_PATH, "w") as f:
        json.dump(data, f, indent=2)


def clear_trigger() -> None:
    """Reset the trigger to default (no drift)."""
    write_trigger(_DEFAULT_TRIGGER)


# ------------------------------------------------------------------
# Drift heuristics
# ------------------------------------------------------------------
def check_drift(
    training_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    amount_col: str = "total_amount_usd",
    country_col: str = "country",
    amount_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Compare training baseline vs recent live transactions.

    Returns a trigger dict ready to be written to the JSON file.
    """
    reasons = []

    # --- Average amount shift ---
    train_mean = training_df[amount_col].mean()
    recent_mean = recent_df[amount_col].mean()
    if train_mean > 0:
        pct_change = abs(recent_mean - train_mean) / train_mean
        if pct_change > amount_threshold:
            reasons.append(
                f"Average amount shifted {pct_change:.0%} "
                f"(train={train_mean:.2f}, recent={recent_mean:.2f})."
            )

    # --- Country distribution shift ---
    train_top = set(
        training_df[country_col].value_counts().head(5).index
    )
    recent_top = set(
        recent_df[country_col].value_counts().head(5).index
    )
    if train_top and recent_top:
        jaccard = 1 - len(train_top & recent_top) / len(train_top | recent_top)
        if jaccard > 0.4:
            reasons.append(
                f"Top-5 country distribution diverged (Jaccard={jaccard:.2f})."
            )

    trigger = {
        "retrain_required": len(reasons) > 0,
        "reason": " | ".join(reasons) if reasons else "",
        "last_checked": datetime.utcnow().isoformat(),
    }
    write_trigger(trigger)
    return trigger
