"""
tests/test_preprocess.py — regression suite for scripts/preprocess.py's
categorical encoding. Consolidates verify_categorical_fix.py,
verify_schema_bool_passthrough.py, and find_collisions.py.
Run: pytest tests/test_preprocess.py -v
"""
import json
import math
from pathlib import Path
from collections import defaultdict

from scripts.preprocess import preprocess_input, _CASE_SENSITIVE_COLUMNS
from app.schema import TransactionInput


# --- Case-insensitive matching -------------------------------------------

def test_exact_case_still_matches():
    row = preprocess_input({"TransactionAmt": 100, "card4": "visa"})
    assert row["card4"].iloc[0] == 3.0

def test_mismatched_case_now_matches():
    row = preprocess_input({"TransactionAmt": 100, "card4": "Visa"})
    assert row["card4"].iloc[0] == 3.0

def test_json_bool_true_bridges_to_T():
    row = preprocess_input({"TransactionAmt": 100, "id_35": True})
    assert row["id_35"].iloc[0] == 1.0

def test_json_bool_false_bridges_to_F():
    row = preprocess_input({"TransactionAmt": 100, "id_35": False})
    assert row["id_35"].iloc[0] == 0.0

def test_unseen_category_stays_nan():
    row = preprocess_input({"TransactionAmt": 100, "card4": "not_a_real_network"})
    assert math.isnan(row["card4"].iloc[0])

def test_missing_field_stays_nan():
    row = preprocess_input({"TransactionAmt": 100})
    assert math.isnan(row["card4"].iloc[0])


# --- Case-sensitive exception (DeviceInfo) --------------------------------

def test_deviceinfo_uppercase_keeps_own_code():
    row = preprocess_input({"TransactionAmt": 100, "DeviceInfo": "ALCATEL"})
    assert row["DeviceInfo"].iloc[0] == 75.0

def test_deviceinfo_titlecase_keeps_different_code():
    row = preprocess_input({"TransactionAmt": 100, "DeviceInfo": "Alcatel"})
    assert row["DeviceInfo"].iloc[0] == 119.0

def test_deviceinfo_does_not_case_fold():
    row = preprocess_input({"TransactionAmt": 100, "DeviceInfo": "alcatel"})
    assert math.isnan(row["DeviceInfo"].iloc[0])

def test_deviceinfo_missing_stays_nan():
    row = preprocess_input({"TransactionAmt": 100})
    assert math.isnan(row["DeviceInfo"].iloc[0])


# --- Schema-level: JSON bool actually survives to model_dump() -----------

def test_schema_preserves_json_bool_type():
    raw_json = '{"TransactionAmt": 100, "id_35": true, "id_36": false}'
    dumped = TransactionInput.model_validate_json(raw_json).model_dump()
    assert isinstance(dumped["id_35"], bool)
    assert isinstance(dumped["id_36"], bool)


# --- Guard against a FUTURE retrain silently invalidating the exclusion ---

def test_only_known_column_has_case_collisions():
    """
    find_collisions.py told you, ONCE, that DeviceInfo was the only column
    with a case collision. That fact is now hardcoded as
    _CASE_SENSITIVE_COLUMNS in preprocess.py. Nothing currently re-checks
    it against reality. If the model gets retrained and category_maps.json
    changes — DeviceInfo's collisions get cleaned up, or a DIFFERENT column
    develops one — this hardcoded set goes stale silently. This test makes
    that impossible: it re-derives the actual collision set from the live
    file, every run, and fails loudly the moment it disagrees with the
    hardcoded exclusion.
    """
    with open(Path("model") / "category_maps.json") as f:
        maps = json.load(f)

    collided_columns = set()
    for col, mapping in maps.items():
        buckets = defaultdict(list)
        for raw_val in mapping:
            buckets[raw_val.lower()].append(raw_val)
        if any(len(v) > 1 for v in buckets.values()):
            collided_columns.add(col)

    assert collided_columns == _CASE_SENSITIVE_COLUMNS, (
        f"category_maps.json's actual collisions {collided_columns} no "
        f"longer match _CASE_SENSITIVE_COLUMNS {_CASE_SENSITIVE_COLUMNS} "
        "in preprocess.py. Update the hardcoded set to match."
    )

def test_case_sensitive_column_bool_bridge_tries_both_cases(monkeypatch):
    """
    No real column today is both case-colliding and boolean-valued -- the
    gap flagged when the case-sensitive exclusion shipped. Synthesizes
    that scenario via monkeypatch, using V1 (normally a plain numeric
    passthrough) as a stand-in, rather than waiting for some future
    retrain to produce one for real.
    """
    import scripts.preprocess as pp

    fake_mapping = {"T": 1, "F": 0, "SomeOtherValue": 2}
    monkeypatch.setitem(pp._CATEGORY_MAPS, "V1", fake_mapping)
    monkeypatch.setattr(pp, "_CASE_SENSITIVE_COLUMNS",
                         pp._CASE_SENSITIVE_COLUMNS | {"V1"})

    row_true = pp.preprocess_input({"TransactionAmt": 100, "V1": True})
    row_false = pp.preprocess_input({"TransactionAmt": 100, "V1": False})

    assert row_true["V1"].iloc[0] == 1.0
    assert row_false["V1"].iloc[0] == 0.0