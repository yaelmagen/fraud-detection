"""
Lightweight Feature Store for real-time fraud detection simulation.

Maintains two layers:
  1. Raw transaction history (DataFrame / CSV)
  2. In-memory user-level aggregate cache (dict) for O(1) lookups

The cache stores per-user:
  - txn_count: total transactions so far
  - amount_sum: cumulative sum of amounts
  - amount_avg: running average amount
  - last_timestamp: last payment timestamp (pd.Timestamp)
  - last_country: last country code
  - last_state: last state code
  - last_region: last geographic region
  - amounts_24h: list of (timestamp, amount) tuples for rolling 24h window
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class FeatureStore:
    """In-memory feature store backed by CSV persistence."""

    def __init__(self, history_path: str):
        self.history_path = history_path
        self.history: pd.DataFrame = pd.DataFrame()
        self._user_cache: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def bootstrap(self, warmup_df: pd.DataFrame) -> None:
        """Initialize the store with the warm-up (training) data."""
        warmup_df = warmup_df.sort_values("payment_timestamp").copy()
        self.history = warmup_df.copy()

        # Build per-user cache from warm-up data
        for uid, group in warmup_df.groupby("user_id"):
            group = group.sort_values("payment_timestamp")
            last_row = group.iloc[-1]
            amounts = group["total_amount_usd"].values
            timestamps = group["payment_timestamp"].values

            # Build 24h rolling window list (keep last 48h for safety)
            cutoff = last_row["payment_timestamp"] - pd.Timedelta(hours=48)
            recent = group[group["payment_timestamp"] >= cutoff]
            amounts_24h = list(
                zip(
                    recent["payment_timestamp"].tolist(),
                    recent["total_amount_usd"].tolist(),
                )
            )

            self._user_cache[uid] = {
                "txn_count": len(group),
                "amount_sum": float(np.sum(amounts)),
                "amount_avg": float(np.mean(amounts)),
                "last_timestamp": pd.Timestamp(last_row["payment_timestamp"]),
                "last_country": last_row.get("country", None),
                "last_state": last_row.get("state", None),
                "last_region": last_row.get("region", None),
                "amounts_24h": amounts_24h,
            }

        self._persist()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Return cached aggregates for a user, or None if new user."""
        return self._user_cache.get(user_id, None)

    # ------------------------------------------------------------------
    # Updates (incremental)
    # ------------------------------------------------------------------
    def append_transaction(self, txn: pd.Series) -> None:
        """Append a scored transaction and update user cache incrementally."""
        # Append to history DataFrame
        self.history = pd.concat(
            [self.history, txn.to_frame().T], ignore_index=True
        )

        uid = txn["user_id"]
        ts = pd.Timestamp(txn["payment_timestamp"])
        amount = float(txn["total_amount_usd"])
        country = txn.get("country", None)
        state = txn.get("state", None)
        region = txn.get("region", None)

        if uid in self._user_cache:
            cache = self._user_cache[uid]
            cache["txn_count"] += 1
            cache["amount_sum"] += amount
            cache["amount_avg"] = cache["amount_sum"] / cache["txn_count"]
            cache["last_timestamp"] = ts
            cache["last_country"] = country
            cache["last_state"] = state
            cache["last_region"] = region

            # Update 24h rolling window – prune entries older than 48h
            cutoff = ts - pd.Timedelta(hours=48)
            cache["amounts_24h"] = [
                (t, a) for t, a in cache["amounts_24h"] if pd.Timestamp(t) >= cutoff
            ]
            cache["amounts_24h"].append((ts, amount))
        else:
            self._user_cache[uid] = {
                "txn_count": 1,
                "amount_sum": amount,
                "amount_avg": amount,
                "last_timestamp": ts,
                "last_country": country,
                "last_state": state,
                "last_region": region,
                "amounts_24h": [(ts, amount)],
            }

    def persist(self) -> None:
        """Public method to save history to disk."""
        self._persist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Write history to CSV."""
        self.history.to_csv(self.history_path, index=False)
