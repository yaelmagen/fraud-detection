"""
Model Version Registry.

Manages versioned model artifacts stored under the ``models/`` directory.

Naming convention:
    models/
        isolation_forest_v1.pkl
        lof_v1.pkl
        preprocessor_v1.pkl
        scaler_v1.pkl
        metadata_v1.json      # feature list, calibrator bounds, training info

Version numbers are auto-incremented (v1, v2, …); existing versions are
never overwritten.
"""

import os
import re
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


_MODELS_DIR: Optional[str] = None


def _models_dir() -> str:
    global _MODELS_DIR
    if _MODELS_DIR is None:
        _MODELS_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
        )
        os.makedirs(_MODELS_DIR, exist_ok=True)
    return _MODELS_DIR


# ------------------------------------------------------------------
# Version discovery
# ------------------------------------------------------------------
def list_versions() -> List[int]:
    """Return sorted list of available version numbers."""
    versions = set()
    for fname in os.listdir(_models_dir()):
        m = re.search(r"_v(\d+)\.", fname)
        if m:
            versions.add(int(m.group(1)))
    return sorted(versions)


def latest_version() -> Optional[int]:
    """Return the highest version number, or None if no models saved."""
    vs = list_versions()
    return vs[-1] if vs else None


def next_version() -> int:
    """Return the next version number to use."""
    lv = latest_version()
    return (lv or 0) + 1


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
def _paths(version: int) -> Dict[str, str]:
    d = _models_dir()
    return {
        "iso_forest": os.path.join(d, f"isolation_forest_v{version}.pkl"),
        "lof": os.path.join(d, f"lof_v{version}.pkl"),
        "preprocessor": os.path.join(d, f"preprocessor_v{version}.pkl"),
        "scaler": os.path.join(d, f"scaler_v{version}.pkl"),
        "metadata": os.path.join(d, f"metadata_v{version}.json"),
    }


def get_paths(version: Optional[int] = None) -> Dict[str, str]:
    """Return file paths for a given version (default: latest)."""
    if version is None:
        version = latest_version()
    if version is None:
        raise FileNotFoundError("No model versions found in models/")
    return _paths(version)


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
def save_model(
    iso_forest,
    lof,
    preprocessor,
    scaler,
    calibrator_bounds: Dict[str, float],
    feature_list: List[str],
    training_rows: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Persist all model artifacts as a new version.

    Returns the version number used.
    """
    ver = next_version()
    paths = _paths(ver)

    joblib.dump(iso_forest, paths["iso_forest"])
    joblib.dump(lof, paths["lof"])
    joblib.dump(preprocessor, paths["preprocessor"])
    joblib.dump(scaler, paths["scaler"])

    meta = {
        "version": ver,
        "created_at": datetime.utcnow().isoformat(),
        "feature_list": feature_list,
        "calibrator_bounds": calibrator_bounds,
        "training_rows": training_rows,
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)

    return ver


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load_model(version: Optional[int] = None) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """
    Load model artifacts for a given version (default: latest).

    Returns
    -------
    iso_forest, lof, preprocessor, scaler, metadata_dict
    """
    paths = get_paths(version)

    iso_forest = joblib.load(paths["iso_forest"])
    lof = joblib.load(paths["lof"])
    preprocessor = joblib.load(paths["preprocessor"])
    scaler = joblib.load(paths["scaler"])

    with open(paths["metadata"], "r") as f:
        metadata = json.load(f)

    return iso_forest, lof, preprocessor, scaler, metadata
