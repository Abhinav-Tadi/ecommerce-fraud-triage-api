"""
inference.py — model loading and prediction.

Model and config are loaded once at module import (Lambda cold start), not on every request. 
Loading a 41 MB XGBoost model per-request would make the service unusable.
"""

import json
import joblib
from pathlib import Path
from scripts.preprocess import preprocess_input

# Load once at startup
_ROOT = Path(__file__).resolve().parent.parent

with open(_ROOT / "model" / "model_config.json") as f:
    _CONFIG = json.load(f)

_MODEL          = joblib.load(_ROOT / "model" / "model.joblib")
_THRESHOLD      = _CONFIG["threshold"]
_BEST_ITERATION = _CONFIG["best_iteration"]


def predict(raw: dict) -> dict:
    """
    Parameters
    ----------
    raw : dict  — raw transaction fields from API request (output of model.model_dump())

    Returns
    -------
    dict with keys: prediction (int), probability (float), threshold (float)
    """
    features = preprocess_input(raw)

    # iteration_range is explicit to avoid relying on best_ntree_limit
    # surviving joblib serialisation across XGBoost versions.
    proba = float(
        _MODEL.predict_proba(
            features,
            iteration_range=(0, _BEST_ITERATION + 1),
        )[0, 1]
    )

    return {
        "prediction":  int(proba >= _THRESHOLD),
        "probability": proba,
        "threshold":   _THRESHOLD,
    }