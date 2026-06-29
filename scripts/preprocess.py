"""
preprocess.py — single source of truth for feature transformation.

Called identically at training time (notebook) and inference time (app/inference.py).
Any divergence between those two paths is reverse leakage.

Key design decisions:
- Category maps loaded once at import, not per-request (saves ~5ms per call)
- Missing features -> NaN; XGBoost handles NaN natively via learned default directions
- TransactionDT is NOT an API input field; hour_of_day and day_of_week_proxy are
  computed here from it if provided
- Unseen categorical values (e.g. new email domains post-training) -> NaN
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load artifacts once at module import, not on every prediction
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "model" / "model_config.json"
_MAPS_PATH   = _ROOT / "model" / "category_maps.json"

with open(_CONFIG_PATH) as f:
    _CONFIG = json.load(f)

with open(_MAPS_PATH) as f:
    _CATEGORY_MAPS = json.load(f)

FEATURE_COLS = _CONFIG["feature_cols"]   # 422 features, in model-expected order

# Public API
def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    Transform a raw transaction dict into a model-ready single-row DataFrame.

    Parameters
    ----------
    raw : dict
        Keys match IEEE-CIS column names (TransactionAmt, card4, V258, ...).
        Missing keys become NaN. All 422 features are optional except TransactionAmt.

    Returns
    -------
    pd.DataFrame — 1 row x 422 columns, in model-expected order.
    """
    row: dict = {}

    # 1. TransactionAmt: log1p transform
    #    Training applied this; inference must apply it identically.
    amt = raw.get("TransactionAmt")
    row["TransactionAmt"] = np.log1p(float(amt)) if amt is not None else np.nan

    # 2. Time features: derived from TransactionDT, not passed directly.
    #    TransactionDT is a seconds-offset from an undisclosed Vesta reference point.
    #    hour_of_day and day_of_week_proxy are what the model actually sees.
    dt = raw.get("TransactionDT")
    if dt is not None:
        dt = float(dt)
        row["hour_of_day"]       = (dt % 86_400) / 3_600
        row["day_of_week_proxy"] = (dt // 86_400) % 7
    else:
        row["hour_of_day"]       = np.nan
        row["day_of_week_proxy"] = np.nan

    # 3. Categorical features: apply saved ordinal mappings.
    #    Unseen values (new email domains, new device types) -> NaN.
    #    XGBoost uses its learned default branch direction for NaN.
    for col, mapping in _CATEGORY_MAPS.items():
        val = raw.get(col)
        if val is None:
            row[col] = np.nan
        else:
            val_str = str(val).strip()
            row[col] = float(mapping[val_str]) if val_str in mapping else np.nan

    # 4. All remaining numeric features: pass through.
    _handled = (
        {"TransactionAmt", "TransactionDT", "hour_of_day", "day_of_week_proxy"}
        | set(_CATEGORY_MAPS)
    )
    for col in FEATURE_COLS:
        if col in _handled:
            continue
        val = raw.get(col)
        if val is None:
            row[col] = np.nan
        else:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = np.nan  # Defensive: bad type -> NaN, not crash

    # 5. Build DataFrame with exactly the features the model expects, in order.
    #    Columns not yet in `row` are filled as NaN.
    df = pd.DataFrame([{col: row.get(col, np.nan) for col in FEATURE_COLS}])
    return df[FEATURE_COLS]