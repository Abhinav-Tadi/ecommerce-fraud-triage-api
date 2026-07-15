"""
demo/streamlit_app.py — Live demo frontend for the e-commerce fraud triage API.

Calls the deployed AWS Lambda + API Gateway endpoint directly. No model runs in this app; this is a thin client over the real, deployed inference service.

All non-rendering logic (payload construction, presets, the API call) lives in fraud_demo_logic.py, deliberately, so it can be unit-tested without ever
importing this file as a plain Python module.
"""

import json
import streamlit as st

from fraud_demo_logic import (
    PREDICT_URL, HEALTH_URL, OPERATING_THRESHOLD, RECALL_AT_THRESHOLD,
    PRECISION_AT_THRESHOLD, NOT_PROVIDED, PRODUCT_CODES, CARD_NETWORKS,
    CARD_TYPES, DEVICE_TYPES, EMAIL_DOMAINS, PRESETS, DEFAULTS,
    parse_optional_float, build_payload, call_predict, warm_up,
)

st.set_page_config(
    page_title="Fraud Triage — Live Demo",
    page_icon="🛡️",
    layout="centered",
)

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

if "warmed_up" not in st.session_state:
    try:
        warm_up()
    except Exception:
        pass  # best-effort only; a failed warm-up ping is not a real error
    st.session_state["warmed_up"] = True


def optional_float_input(label, key, help_text=None):
    """UI wrapper around the pure parse_optional_float — this is the only place Streamlit and the parsing logic meet."""
    raw = st.text_input(label, key=key, help=help_text, placeholder="leave blank to omit")
    value, warning = parse_optional_float(raw)
    if warning:
        st.warning(f"'{label}' {warning}")
    return value


def render_result(resp, err, payload):
    st.divider()
    if err == "timeout":
        st.error(
            "Request timed out. The Lambda function may be cold-starting "
            "(observed cold start ≈7s in testing) — try again."
        )
        return
    if err is not None:
        st.error(f"Could not reach the API: {err}")
        return

    if resp.status_code == 200:
        data = resp.json()
        prediction = data["prediction"]
        probability = data["probability"]
        threshold = data["threshold"]

        if prediction == 1:
            st.error(
                f"🚩 **FLAGGED FOR MANUAL REVIEW** — "
                f"probability {probability:.2%} ≥ threshold {threshold:.2%}"
            )
        else:
            st.success(
                f"✅ **PASSED** — "
                f"probability {probability:.2%} < threshold {threshold:.2%}"
            )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"Threshold is set for ~{RECALL_AT_THRESHOLD:.0%} recall / "
            f"~{PRECISION_AT_THRESHOLD:.0%} precision on held-out test data "
            f"— not the default 0.5. See DECISIONS.md for why."
        )
        with st.expander("Raw request / response"):
            st.markdown("**Sent to `/predict`:**")
            st.json(payload)
            st.markdown("**Received:**")
            st.json(data)

    elif resp.status_code == 422:
        st.warning(
            "HTTP 422 — the API rejected this request as malformed. "
            "That's Pydantic validation working correctly, not a bug."
        )
        st.json(resp.json())
    else:
        st.error(f"Unexpected response: HTTP {resp.status_code}")
        st.code(resp.text or "(empty body)")

# Page
st.title("🛡️ E-commerce Fraud Triage — Live Demo")
st.caption(
    "This form calls a real XGBoost model deployed on AWS Lambda behind "
    "API Gateway — the request leaves this app and hits the actual live "
    "endpoint used in the project's README."
)

with st.expander("Read this before you draw conclusions from the numbers below", expanded=False):
    st.markdown(f"""
The model scores **422 features** per transaction. This form exposes the
handful that are human-interpretable: amount, product code, card network/type,
purchaser email domain, device type, and time-of-day. The other ~400 features
are Vesta's anonymized behavioral/device signals (`V1`-`V339`, `C1`-`C14`,
`D1`-`D15`, `M1`-`M9`, `id_01`-`id_38`) — a demo visitor has no way to
meaningfully set these, so they're sent as missing, which the model handles
natively (this is also the realistic case: 75.6% of real transactions in the
training data have no identity record at all).

**The catch:** the single most important feature (`V258`) alone accounts for
24.7% of this model's decision weight, and the second (`V201`) another 8.6%
— together a third of the model's total signal, from features this demo
can't touch. Changing the fields below will move the probability, but won't
always flip the flag/pass verdict, because the features doing most of the
work are constant (missing) across every submission here. That's a real
limitation of exposing a tabular fraud model through a browser form, not
a bug in this demo.

Operating threshold: **{OPERATING_THRESHOLD}** (not 0.5) — chosen for
~{RECALL_AT_THRESHOLD:.0%} recall / ~{PRECISION_AT_THRESHOLD:.0%} precision.
Full reasoning in `DECISIONS.md`.
""")

tab_guided, tab_raw = st.tabs(["Guided form", "Raw JSON (power users)"])

with tab_guided:
    preset_choice = st.selectbox(
        "Try an example (fills in the fields below — nothing is sent until you click Score)",
        ["— none, start blank —"] + list(PRESETS.keys()),
        key="preset_select",
    )
    # Manual apply-and-rerun instead of on_change — sidesteps Streamlit's form-callback policy checking entirely rather than relying on getting
    # "outside the form" nesting right by convention (see fraud_demo_logic.py docstring for why that convention alone once broke on this exact file).
    if preset_choice in PRESETS and st.session_state.get("_last_preset") != preset_choice:
        for k, v in PRESETS[preset_choice].items():
            st.session_state[k] = v
        st.session_state["_last_preset"] = preset_choice
        st.rerun()

    with st.form("guided_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input(
                "Transaction amount (USD)", min_value=0.01, step=1.0, key="amt"
            )
            product = st.selectbox("Product code", PRODUCT_CODES, key="product")
            card_network = st.selectbox("Card network", CARD_NETWORKS, key="card_network")
            card_type = st.selectbox("Card type", CARD_TYPES, key="card_type")
        with col2:
            hour = st.slider("Hour of day (0 = midnight)", 0, 23, key="hour")
            day_proxy = st.slider(
                "Position in Vesta's internal 7-day cycle (0-6)", 0, 6, key="day_proxy",
                help=(
                    "Weaker, murkier signal than hour-of-day (fraud rate only "
                    "ranges 3.15%-3.72% across this vs. a 4.6x swing by hour) "
                    "and the dataset doesn't disclose which value maps to which "
                    "real weekday. Included for completeness, not because it's "
                    "the demo's strongest lever."
                ),
            )
            email = st.selectbox("Purchaser email domain", EMAIL_DOMAINS, key="email")
            device = st.selectbox("Device type", DEVICE_TYPES, key="device")

        with st.expander("Advanced / masked features (optional)"):
            st.caption(
                "Masked numeric attributes Vesta didn't disclose the exact "
                "meaning of — included for completeness. Leave blank to omit "
                "(the model handles missing values natively)."
            )
            acol1, acol2 = st.columns(2)
            with acol1:
                card1 = optional_float_input("card1", "card1", "Masked card attribute")
                card2 = optional_float_input("card2", "card2", "Masked card attribute")
                card3 = optional_float_input("card3", "card3", "Masked card attribute")
                card5 = optional_float_input("card5", "card5", "Masked card attribute")
            with acol2:
                addr1 = optional_float_input("addr1", "addr1", "Masked billing region code")
                addr2 = optional_float_input("addr2", "addr2", "Masked billing region code")
                dist1 = optional_float_input("dist1", "dist1", "Masked distance metric")
                dist2 = optional_float_input("dist2", "dist2", "Masked distance metric")
            r_email = st.selectbox("Recipient email domain", EMAIL_DOMAINS, key="r_email")
            device_info = st.text_input(
                "Device info (exact string)", key="device_info",
                help=(
                    "Free text, but the model only recognizes ~1,786 exact "
                    "device strings from the original dataset (e.g. "
                    "'SAMSUNG SM-G935F Build/NRD90M'). Anything else resolves "
                    "to missing — included for completeness, not because "
                    "typing a guess here will do anything."
                ),
            )

        submitted = st.form_submit_button("Score this transaction", type="primary")

    if submitted:
        advanced = {
            "card1": card1, "card2": card2, "card3": card3, "card5": card5,
            "addr1": addr1, "addr2": addr2, "dist1": dist1, "dist2": dist2,
            "R_emaildomain": r_email if r_email != NOT_PROVIDED else None,
            "DeviceInfo": device_info if device_info else None,
        }
        payload = build_payload(
            amt, hour, day_proxy, product, card_network, card_type,
            email, device, advanced,
        )
        with st.spinner("Calling the live Lambda endpoint (cold start can take ~7s)..."):
            resp, err = call_predict(payload)
        render_result(resp, err, payload)

with tab_raw:
    st.caption(
        "Send an arbitrary JSON body straight to the deployed endpoint. "
        "You can add any of the model's other ~400 features (V1-V339, "
        "C1-C14, D1-D15, M1-M9, id_01-id_38) as extra keys — see this "
        "project's model/category_maps.json on GitHub for the exact "
        "categorical vocabularies. Vesta never published the value ranges "
        "for the V/C/D columns, so this demo doesn't guess at realistic "
        "numbers for them."
    )
    default_body = json.dumps({"TransactionAmt": 150.0}, indent=2)
    raw_text = st.text_area("Request body", value=default_body, height=180)
    if st.button("Send raw request"):
        try:
            raw_payload = json.loads(raw_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            with st.spinner("Calling the live Lambda endpoint..."):
                resp, err = call_predict(raw_payload)
            render_result(resp, err, raw_payload)

st.divider()
st.caption(
    "Portfolio project — not a real fraud-detection service, and no real "
    "personal or financial information is required or should be entered. "
    "Source: github.com/Abhinav-Tadi/ecommerce-fraud-triage-api"
)