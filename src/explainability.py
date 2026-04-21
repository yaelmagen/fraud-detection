"""
Explainability Module — SHAP waterfall + human-readable rule reasons.

Provides two complementary explanation layers:
  1. SHAP-based feature importance (waterfall plot)
  2. Rule-based human-readable text reasons
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from src.models import FraudEnsemble, FINAL_FEATURES


class Explainer:
    """Wraps SHAP around the Isolation Forest inside the ensemble."""

    def __init__(self, ensemble: FraudEnsemble):
        self.ensemble = ensemble
        self._shap_explainer: Optional[shap.Explainer] = None

    # ------------------------------------------------------------------
    def load_pretrained(self, explainer_path: str = "shap_explainer.pkl") -> "Explainer":
        """
        Load a pre-fitted SHAP explainer from disk.
        
        Args:
            explainer_path: Path to the serialized SHAP explainer
        """
        import joblib
        self._shap_explainer = joblib.load(explainer_path)
        return self

    # ------------------------------------------------------------------
    def build(self, background_sample: pd.DataFrame, already_processed: bool = False) -> "Explainer":
        """
        Initialize the SHAP explainer with a background dataset.
        
        Args:
            background_sample: Background data for SHAP
            already_processed: If True, background_sample is already preprocessed
        """
        if already_processed:
            X_proc = background_sample
        else:
            X_bg = background_sample[FINAL_FEATURES].copy()
            for col in ["merchant_id", "geo_location", "payment_instrument",
                         "device_type", "currency"]:
                X_bg[col] = X_bg[col].astype("category")
            X_proc = self.ensemble.preprocessor.transform(X_bg)

        self._shap_explainer = shap.Explainer(
            self.ensemble.iso_forest,
            X_proc,
        )
        return self

    # ------------------------------------------------------------------
    def shap_values_for(self, feature_dict: dict):
        """Compute SHAP values for a single observation."""
        X_proc = self.ensemble.transform_single(feature_dict)
        sv = self._shap_explainer(X_proc)
        feature_names = self.ensemble.get_feature_names()
        sv.feature_names = feature_names
        return sv

    # ------------------------------------------------------------------
    def waterfall_figure(self, feature_dict: dict):
        """Return a matplotlib Figure with the SHAP waterfall plot."""
        sv = self.shap_values_for(feature_dict)
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.sca(ax)
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        plt.tight_layout()
        return fig


# ======================================================================
# Rule-based human-readable reasons
# ======================================================================
def human_readable_reasons(
    features: Dict[str, Any],
    breakdown: Dict[str, float],
    decision: str = "Approve",
) -> List[str]:
    """
    Generate decision-aligned explanations.
    - Approve  → highlight safety signals (why it passed)
    - Review   → mixed: note mild concerns + mitigating factors
    - Block    → highlight risk signals (why it was blocked)
    """
    if decision == "Block":
        return _risk_reasons(features, breakdown)
    elif decision == "Review":
        return _review_reasons(features, breakdown)
    else:
        return _approve_reasons(features, breakdown)


# ----------------------------------------------------------------------
# Block — emphasize what triggered the high score
# ----------------------------------------------------------------------
def _risk_reasons(
    features: Dict[str, Any], breakdown: Dict[str, float]
) -> List[str]:
    reasons: List[str] = []

    if features.get("is_impossible_travel", 0) == 1:
        reasons.append(
            "🚨 **Impossible Travel** — Transaction originated from a "
            "different geographic region within <10 hours of the previous one."
        )
    if features.get("is_suspicious_velocity", 0) == 1:
        reasons.append(
            "⚠️ **Suspicious Velocity** — Country or state changed within "
            "<3 hours of the last transaction (same region)."
        )
    if features.get("is_broken_record", 0) == 1:
        reasons.append(
            "⚠️ **Broken Record** — Missing device type metadata, "
            "indicating potential data manipulation."
        )
    same_24 = features.get("same_amount_count_24h", 0)
    if same_24 >= 1:
        reasons.append(
            f"⚠️ **Same-Amount Burst** — {int(same_24)} identical-amount "
            f"transaction(s) by this user in the last 24 h."
        )
    ratio = features.get("amount_to_avg_ratio", 1.0)
    if ratio >= 5.0:
        reasons.append(
            f"💰 **Amount Spike** — Transaction amount is "
            f"**{ratio:.1f}x** the user's historical average."
        )
    if features.get("is_new_user", 0) == 1:
        reasons.append(
            "🆕 **New User** — No prior transaction history; cold-start "
            "defaults applied."
        )
    tslp = features.get("time_since_last_payment", 999999)
    if tslp < 0.0167:
        reasons.append(
            f"⏱️ **Rapid Repeat** — Only {tslp * 3600:.0f} seconds since "
            f"this user's last transaction."
        )
    if breakdown.get("if_risk", 0) > 0.8:
        reasons.append(
            f"🤖 **Isolation Forest** flagged this transaction "
            f"(risk={breakdown['if_risk']:.2f})."
        )
    if breakdown.get("lof_risk", 0) > 0.8:
        reasons.append(
            f"🔍 **LOF** detected a local density anomaly "
            f"(risk={breakdown['lof_risk']:.2f})."
        )
    if not reasons:
        reasons.append(
            "🔴 ML ensemble score exceeded the Block threshold "
            f"(score={breakdown.get('final', 0):.2f})."
        )
    return reasons


# ----------------------------------------------------------------------
# Approve — emphasize why the transaction is safe
# ----------------------------------------------------------------------
def _approve_reasons(
    features: Dict[str, Any], breakdown: Dict[str, float]
) -> List[str]:
    reasons: List[str] = []

    ratio = features.get("amount_to_avg_ratio", 1.0)
    if ratio <= 2.0:
        reasons.append(
            f"✅ **Normal Amount** — Transaction is **{ratio:.1f}x** the "
            f"user's average, within expected range."
        )

    if features.get("is_new_user", 0) == 0:
        seniority = features.get("seniority", 0)
        # Note: The actual risk impact of seniority (positive/negative) 
        # should be interpreted from the SHAP plot for this specific transaction
        reasons.append(
            f"**Established User** Account has **{seniority} days** of "
            f"transaction history for behavioral context."
        )

    if (features.get("is_impossible_travel", 0) == 0
            and features.get("is_suspicious_velocity", 0) == 0):
        reasons.append(
            "✅ **Consistent Geography** — No geographic anomalies detected."
        )

    if features.get("same_amount_count_24h", 0) == 0:
        reasons.append(
            "✅ **Unique Amount** — No duplicate-amount pattern in the last 24 h."
        )

    if features.get("is_broken_record", 0) == 0:
        reasons.append(
            "✅ **Complete Metadata** — Device type information is present."
        )

    if breakdown.get("if_risk", 0) < 0.3 and breakdown.get("lof_risk", 0) < 0.3:
        reasons.append(
            f"✅ **Low ML Risk** — Both Isolation Forest "
            f"({breakdown.get('if_risk', 0):.2f}) and LOF "
            f"({breakdown.get('lof_risk', 0):.2f}) report low anomaly scores."
        )

    if not reasons:
        reasons.append("✅ No significant risk indicators detected.")
    return reasons


# ----------------------------------------------------------------------
# Review — balanced: note mild concerns + what looks normal
# ----------------------------------------------------------------------
def _review_reasons(
    features: Dict[str, Any], breakdown: Dict[str, float]
) -> List[str]:
    reasons: List[str] = []
    reasons.append(
        f"🟡 **Elevated ML Score** — Combined risk "
        f"({breakdown.get('final', 0):.2f}) falls in the Review band."
    )

    # Note any mild concerns
    ratio = features.get("amount_to_avg_ratio", 1.0)
    if ratio >= 3.0:
        reasons.append(
            f"⚠️ **Above-Average Amount** — Transaction is **{ratio:.1f}x** "
            f"the user's historical average."
        )

    if features.get("is_new_user", 0) == 1:
        reasons.append(
            "🆕 **New User** — Limited history; cold-start defaults applied."
        )

    same_24 = features.get("same_amount_count_24h", 0)
    if same_24 >= 1:
        reasons.append(
            f"⚠️ **Same-Amount Activity** — {int(same_24)} identical-amount "
            f"transaction(s) in the last 24 h."
        )

    # Note mitigating factors
    if (features.get("is_impossible_travel", 0) == 0
            and features.get("is_suspicious_velocity", 0) == 0):
        reasons.append(
            "✅ **No Geographic Anomalies** — Location pattern is consistent."
        )

    if features.get("is_broken_record", 0) == 0:
        reasons.append(
            "✅ **Complete Metadata** — Device information is present."
        )

    if breakdown.get("if_risk", 0) < 0.5:
        reasons.append(
            f"✅ **Isolation Forest** considers this within normal range "
            f"(risk={breakdown['if_risk']:.2f})."
        )
    elif breakdown.get("if_risk", 0) > 0.7:
        reasons.append(
            f"⚠️ **Isolation Forest** reports elevated risk "
            f"({breakdown['if_risk']:.2f})."
        )

    if breakdown.get("lof_risk", 0) > 0.7:
        reasons.append(
            f"⚠️ **LOF** detected a mild density deviation "
            f"({breakdown['lof_risk']:.2f})."
        )

    return reasons
