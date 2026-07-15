"""
demo/fraud_demo_logic.py — pure logic for the Streamlit demo.

Deliberately contains zero Streamlit rendering calls (no st.title, st.form, st.selectbox, etc.) and zero top-level side effects beyond simple constant
definitions. This is what makes it safe to import directly in tests.

Why this file exists separately from streamlit_app.py: a Streamlit script's entire top-level body executes as a side effect of being imported as a
Python module (Streamlit apps are conventionally written as top-level scripts, not wrapped in functions). Importing pure helpers directly from a
file that also contains top-level st.* calls forces that whole UI script to run once, bare, outside any managed script-run context — and this silently
corrupts Streamlit's internal form-tracking state for any AppTest run of the same file later in the same process. Confirmed: with nothing importing a
Streamlit script as a plain module first, AppTest runs clean; the moment something does, every subsequent AppTest run of that file in the same process
throws StreamlitInvalidFormCallbackError, even on widgets that are not textually inside any st.form().

Splitting logic out of the rendering file sidesteps the problem entirely.
"""

import requests

API_BASE = "https://8456ksu3u8.execute-api.us-east-1.amazonaws.com"
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

# Mirrors model/model_config.json at the time this file was written — this value is NOT fetched live from the API.
OPERATING_THRESHOLD = 0.0957
RECALL_AT_THRESHOLD = 0.85
PRECISION_AT_THRESHOLD = 0.67

NOT_PROVIDED = "— not provided —"

PRODUCT_CODES = ["W", "C", "R", "H", "S"]
CARD_NETWORKS = ["visa", "mastercard", "american express", "discover"]
CARD_TYPES = ["debit", "credit", "debit or credit", "charge card"]
DEVICE_TYPES = [NOT_PROVIDED, "desktop", "mobile"]

# Curated subset of category_maps.json's P_emaildomain / R_emaildomain keys — confirmed present in both maps. Full vocabulary (59-60 domains) is in
# model/category_maps.json; anything not in that file resolves to NaN, which the model handles, so this list doesn't need to be exhaustive.
EMAIL_DOMAINS = [
    NOT_PROVIDED, "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "comcast.net", "live.com", "protonmail.com",
    "anonymous.com",
]

# Preset scenarios — illustrative starting points, not labeled outcomes. The point is to show different corners of the input space; whether any
# given preset gets flagged is for the live model to decide.
PRESETS = {
    "Everyday small purchase": {
        "amt": 42.50, "hour": 14, "day_proxy": 3, "product": "W",
        "card_network": "visa", "card_type": "debit",
        "email": "gmail.com", "device": "desktop",
    },
    "Late-night higher-value order": {
        "amt": 480.00, "hour": 3, "day_proxy": 5, "product": "R",
        "card_network": "discover", "card_type": "credit",
        "email": "anonymous.com", "device": "mobile",
    },
    "Early-morning order, no device on file": {
        "amt": 15.00, "hour": 7, "day_proxy": 0, "product": "C",
        "card_network": "mastercard", "card_type": "credit",
        "email": "anonymous.com", "device": NOT_PROVIDED,
    },
}

DEFAULTS = {
    "amt": 89.99, "hour": 14, "day_proxy": 3, "product": "W",
    "card_network": "visa", "card_type": "debit",
    "email": "gmail.com", "device": "desktop",
    "card1": "", "card2": "", "card3": "", "card5": "",
    "addr1": "", "addr2": "", "dist1": "", "dist2": "",
    "r_email": NOT_PROVIDED, "device_info": "",
}


def parse_optional_float(raw):
    """Parse a text-input string into (value, warning_message).
    warning_message is None when parsing succeeds or the input was blank (blank means 'omit this field', not an error)."""
    if raw is None or str(raw).strip() == "":
        return None, None
    try:
        return float(raw), None
    except ValueError:
        return None, f"needs to be a number — ignoring '{raw}'."


def build_payload(amt, hour, day_proxy, product, card_network, card_type, email, device, advanced):
    """Pure function: assembles the exact JSON body sent to /predict. Fields the caller didn't provide are omitted entirely, not sent as null
    — matches how TransactionInput/preprocess_input treat missing keys."""
    payload = {"TransactionAmt": float(amt)}

    # TransactionDT has no real calendar meaning (offset from an undisclosed Vesta reference point) — this value exists purely to reproduce the two
    # engineered features the model actually trained on.
    payload["TransactionDT"] = float(day_proxy) * 86400 + float(hour) * 3600

    payload["ProductCD"] = product
    payload["card4"] = card_network
    payload["card6"] = card_type

    if email != NOT_PROVIDED:
        payload["P_emaildomain"] = email
    if device != NOT_PROVIDED:
        payload["DeviceType"] = device

    for k, v in advanced.items():
        if v is not None and v != NOT_PROVIDED and v != "":
            payload[k] = v

    return payload


def call_predict(payload, timeout=20):
    """Real network call — not pure, but has zero Streamlit dependency, so importing this module never touches Streamlit's runtime at all."""
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=timeout)
        return resp, None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        return None, str(e)


def warm_up():
    """Best-effort ping to reduce the odds of a cold start hitting the first real prediction request. Failure here should be swallowed by the caller
    — it's an optimization, not a correctness requirement."""
    requests.get(HEALTH_URL, timeout=15)