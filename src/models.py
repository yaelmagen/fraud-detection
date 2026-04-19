"""
Ensemble of Unsupervised Models: Isolation Forest + Local Outlier Factor.

Mirrors the notebook's model architecture:
  - IsolationForest(n_estimators=200, contamination=0.01)
  - LOF(n_neighbors=20, contamination=0.01, novelty=True)

Both models are trained through a preprocessing pipeline that handles:
  - CountEncoder for categorical features
  - Cyclical (sin/cos) encoding for hour
  - Passthrough for numeric features

Supports save/load via the model_registry for versioned persistence.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from category_encoders import CountEncoder
from typing import Optional, Dict, Any, List

from src.feature_engine import FINAL_FEATURES
from src import model_registry


# ------------------------------------------------------------------
# Hour cyclical encoding (same as notebook)
# ------------------------------------------------------------------
def _sin_cos_encode(df):
    hours = df.values.flatten() if hasattr(df, "values") else df.flatten()
    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    return np.column_stack([sin_hour, cos_hour])


def _get_hour_names(transformer, input_features):
    return ["hour_sin", "hour_cos"]


# ------------------------------------------------------------------
# Feature groupings
# ------------------------------------------------------------------
CAT_FEATURES = [
    "merchant_id",
    "geo_location",
    "payment_instrument",
    "device_type",
    "currency",
]
CYCLICAL_FEATURE = ["hour"]
NUMERIC_FEATURES = [
    f for f in FINAL_FEATURES if f not in CAT_FEATURES + CYCLICAL_FEATURE
]


# ------------------------------------------------------------------
# Build preprocessing pipeline
# ------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                CountEncoder(normalize=True, handle_unknown=0),
                CAT_FEATURES,
            ),
            (
                "hour_cyc",
                FunctionTransformer(
                    _sin_cos_encode, feature_names_out=_get_hour_names
                ),
                CYCLICAL_FEATURE,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


class FraudEnsemble:
    """Wraps IF + LOF behind a single .fit() / .score() interface."""

    def __init__(self):
        self.preprocessor = _build_preprocessor()
        self.iso_forest = IsolationForest(
            n_estimators=200, contamination=0.01, random_state=42
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20, contamination=0.01, novelty=True
        )
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "FraudEnsemble":
        """
        Fit both models on the warm-up (training) data.
        *train_df* must contain all columns listed in FINAL_FEATURES.
        """
        X = train_df[FINAL_FEATURES].copy()

        # Ensure categorical columns are category dtype
        for col in CAT_FEATURES:
            X[col] = X[col].astype("category")

        X_proc = self.preprocessor.fit_transform(X)

        # Isolation Forest (works on raw preprocessed)
        self.iso_forest.fit(X_proc)

        # LOF needs scaled data
        X_scaled = self.scaler.fit_transform(X_proc)
        self.lof.fit(X_scaled)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform_single(self, feature_dict: dict) -> np.ndarray:
        """Preprocess a single feature dict → 1-row numpy array."""
        row_df = pd.DataFrame([feature_dict])
        for col in CAT_FEATURES:
            row_df[col] = row_df[col].astype("category")
        X_proc = self.preprocessor.transform(row_df[FINAL_FEATURES])
        return X_proc

    # ------------------------------------------------------------------
    def raw_scores(self, feature_dict: dict):
        """
        Return raw decision_function scores for both models.
        Lower (more negative) = more anomalous for both IF and LOF.
        """
        X_proc = self.transform_single(feature_dict)
        X_scaled = self.scaler.transform(X_proc)

        if_score = self.iso_forest.decision_function(X_proc)[0]
        lof_score = self.lof.decision_function(X_scaled)[0]
        return if_score, lof_score

    # ------------------------------------------------------------------
    def get_feature_names(self):
        """Return feature names after preprocessing."""
        return list(self.preprocessor.get_feature_names_out())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        calibrator_bounds: Dict[str, float],
        training_rows: int,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save all artifacts as a new versioned snapshot. Returns version."""
        return model_registry.save_model(
            iso_forest=self.iso_forest,
            lof=self.lof,
            preprocessor=self.preprocessor,
            scaler=self.scaler,
            calibrator_bounds=calibrator_bounds,
            feature_list=FINAL_FEATURES,
            training_rows=training_rows,
            extra_meta=extra_meta,
        )

    @classmethod
    def from_pretrained(cls, version: Optional[int] = None) -> "FraudEnsemble":
        """Load a saved version (default: latest) into a FraudEnsemble."""
        iso_forest, lof, preprocessor, scaler, metadata = (
            model_registry.load_model(version)
        )
        obj = cls.__new__(cls)
        obj.iso_forest = iso_forest
        obj.lof = lof
        obj.preprocessor = preprocessor
        obj.scaler = scaler
        obj._fitted = True
        obj._metadata = metadata
        return obj
