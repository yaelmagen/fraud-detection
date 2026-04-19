"""
Scoring Engine — Normalization + Weighted Ensemble + Hard Rules.

Three perspectives are combined into a Unified Risk Score [0-1]:

1. **Isolation Forest** (70% weight) — global outlier detection
2. **LOF**              (30% weight) — local density anomalies
3. **Statistical / Rules Layer** — deterministic hard flags that
   can override or boost the ML score.

Decision thresholds:
  - Approve  : score < 0.5
  - Review   : 0.5 ≤ score < 0.8
  - Block    : score ≥ 0.8
"""

import numpy as np
from typing import Dict, Any, Tuple

IF_WEIGHT = 0.7
LOF_WEIGHT = 0.3

# Thresholds
APPROVE_THRESHOLD = 0.5
REVIEW_THRESHOLD = 0.8

# Hard-rule boost: when a deterministic flag fires we set a floor
HARD_RULE_FLOOR = 0.85  # guarantees "Block" for critical rules


class ScoreCalibrator:
    """
    Maintains running min/max of raw IF and LOF scores observed during
    the warm-up phase so that individual transaction scores can be
    normalized to [0, 1] without data leakage.
    """

    def __init__(self):
        self.if_min = np.inf
        self.if_max = -np.inf
        self.lof_min = np.inf
        self.lof_max = -np.inf

    # ------------------------------------------------------------------
    def calibrate_from_training(
        self, if_scores: np.ndarray, lof_scores: np.ndarray
    ) -> None:
        """Compute normalization bounds from warm-up scores."""
        self.if_min = float(np.min(if_scores))
        self.if_max = float(np.max(if_scores))
        self.lof_min = float(np.min(lof_scores))
        self.lof_max = float(np.max(lof_scores))

    # ------------------------------------------------------------------
    def _normalize(self, val: float, vmin: float, vmax: float) -> float:
        """Min-max normalize then invert (lower raw → higher risk)."""
        if vmax == vmin:
            return 0.5
        normed = (val - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)
        return float(1.0 - normed)  # invert: anomalous scores are low

    # ------------------------------------------------------------------
    def score(
        self,
        if_raw: float,
        lof_raw: float,
        features: Dict[str, Any],
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Compute final risk score, decision label, and component breakdown.

        Returns
        -------
        risk_score : float in [0, 1]
        decision   : one of "Approve", "Review", "Block"
        breakdown  : dict with if_risk, lof_risk, ml_risk, hard_flag, final
        """
        if_risk = self._normalize(if_raw, self.if_min, self.if_max)
        lof_risk = self._normalize(lof_raw, self.lof_min, self.lof_max)

        ml_risk = IF_WEIGHT * if_risk + LOF_WEIGHT * lof_risk

        # --- Deterministic hard-flag layer ---
        hard_flag = _check_hard_flags(features)

        if hard_flag:
            risk_score = max(ml_risk, HARD_RULE_FLOOR)
        else:
            risk_score = ml_risk

        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # Decision
        if risk_score >= REVIEW_THRESHOLD:
            decision = "Block"
        elif risk_score >= APPROVE_THRESHOLD:
            decision = "Review"
        else:
            decision = "Approve"

        breakdown = {
            "if_risk": round(if_risk, 4),
            "lof_risk": round(lof_risk, 4),
            "ml_risk": round(ml_risk, 4),
            "hard_flag": int(hard_flag),
            "final": round(risk_score, 4),
        }
        return risk_score, decision, breakdown


# ======================================================================
# Hard-flag rules (deterministic)
# ======================================================================
def _check_hard_flags(features: Dict[str, Any]) -> bool:
    """Return True if any deterministic rule fires."""
    if features.get("is_impossible_travel", 0) == 1:
        return True
    if features.get("is_suspicious_velocity", 0) == 1:
        return True
    if features.get("same_amount_count_24h", 0) >= 1:
        return True
    if features.get("is_broken_record", 0) == 1:
        return True
    return False
