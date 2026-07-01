"""
inference.py — model loading and prediction.

Model and config are loaded once at module import (Lambda cold start),
not on every request. Loading a ~41 MB XGBoost model per-request would
make the service unusable.
"""

import json
from pathlib import Path
from xgboost import XGBClassifier
from scripts.preprocess import preprocess_input

_ROOT = Path(__file__).resolve().parent.parent

with open(_ROOT / "model" / "model_config.json") as f:
    _CONFIG = json.load(f)

_MODEL = XGBClassifier()
_MODEL.load_model(str(_ROOT / "model" / "model.ubj"))
_THRESHOLD      = _CONFIG["threshold"]
_BEST_ITERATION = _CONFIG["best_iteration"]


def predict(raw: dict) -> dict:
    features = preprocess_input(raw)

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