"""
demo/test_streamlit_app.py — smoke tests for the Streamlit demo frontend.

Uses Streamlit's official AppTest framework for the UI-facing tests. These verify the app's code paths (widget wiring, presets, rendering for each
response type) without a browser. They deliberately do not depend on network access — an earlier version of this file asserted "the form
submission produces an error," which only held in an environment with no route to the live AWS endpoint. In Manual Testing, the same submission hits
the actual Lambda function, gets a real 200 back, and correctly renders a success message instead — which isn't a bug, but it does
mean that test's assumption was wrong. Fixed by mocking fraud_demo_logic.call_predict so these tests are deterministic regardless of
network conditions. A separate, clearly-marked test at the bottom actually hits the live endpoint and is skipped by default.

Deliberately does not import streamlit_app anywhere — only AppTest ever touches that file. Pure-logic assertions and mocking both go through 
fraud_demo_logic instead. Importing a Streamlit script as a plain module runs its entire top-level body once, bare, which corrupts Streamlit's
internal form-tracking state for any AppTest run of the same file later in the same process — confirmed directly while building this demo.

Run: cd demo && pytest test_streamlit_app.py -v
Run including the live endpoint check: RUN_LIVE_ENDPOINT_TEST=1 pytest test_streamlit_app.py -v
"""
import os
import pytest
from streamlit.testing.v1 import AppTest
import fraud_demo_logic
from fraud_demo_logic import build_payload, NOT_PROVIDED, parse_optional_float

APP_PATH = "streamlit_app.py"


class _FakeResponse:
    """Minimal stand-in for requests.Response, just enough for render_result."""
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


def _fresh_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def _submit_with_mocked_predict(monkeypatch, resp, err):
    """Patch fraud_demo_logic.call_predict for one AppTest run, submit the guided form with default values, and return the resulting AppTest."""
    monkeypatch.setattr(fraud_demo_logic, "call_predict", lambda *a, **k: (resp, err))
    at = _fresh_app()
    for b in at.button:
        if b.label == "Score this transaction":
            b.click().run()
            break
    return at


def test_initial_load_has_no_exception():
    at = _fresh_app()
    assert not at.exception


def test_default_widget_values():
    at = _fresh_app()
    assert at.selectbox(key="product").value == "W"
    assert at.selectbox(key="card_network").value == "visa"
    assert at.selectbox(key="card_type").value == "debit"
    assert at.selectbox(key="email").value == "gmail.com"
    assert at.selectbox(key="device").value == "desktop"
    assert at.slider(key="hour").value == 14
    assert at.slider(key="day_proxy").value == 3
    assert at.number_input(key="amt").value == 89.99


def test_preset_overwrites_form_fields():
    at = _fresh_app()
    at.selectbox(key="preset_select").select("Late-night higher-value order").run()
    assert not at.exception
    assert at.number_input(key="amt").value == 480.0
    assert at.slider(key="hour").value == 3
    assert at.selectbox(key="product").value == "R"
    assert at.selectbox(key="card_network").value == "discover"
    assert at.selectbox(key="card_type").value == "credit"
    assert at.selectbox(key="email").value == "anonymous.com"
    assert at.selectbox(key="device").value == "mobile"


def test_switching_between_presets_updates_again():
    at = _fresh_app()
    at.selectbox(key="preset_select").select("Everyday small purchase").run()
    assert at.number_input(key="amt").value == 42.50
    at.selectbox(key="preset_select").select("Early-morning order, no device on file").run()
    assert not at.exception
    assert at.number_input(key="amt").value == 15.00
    assert at.selectbox(key="device").value == NOT_PROVIDED


def test_submit_renders_flagged_message_on_prediction_1(monkeypatch):
    resp = _FakeResponse(200, {"prediction": 1, "probability": 0.42, "threshold": 0.0957})
    at = _submit_with_mocked_predict(monkeypatch, resp, None)
    assert not at.exception
    assert len(at.error) == 1
    assert "FLAGGED FOR MANUAL REVIEW" in at.error[0].value
    assert "42.00%" in at.error[0].value


def test_submit_renders_passed_message_on_prediction_0(monkeypatch):
    resp = _FakeResponse(200, {"prediction": 0, "probability": 0.001, "threshold": 0.0957})
    at = _submit_with_mocked_predict(monkeypatch, resp, None)
    assert not at.exception
    assert len(at.success) == 1
    assert "PASSED" in at.success[0].value
    assert len(at.error) == 0


def test_submit_handles_timeout_without_crashing(monkeypatch):
    at = _submit_with_mocked_predict(monkeypatch, None, "timeout")
    assert not at.exception
    assert len(at.error) == 1
    assert "cold-starting" in at.error[0].value


def test_submit_handles_unexpected_status_without_crashing(monkeypatch):
    resp = _FakeResponse(403, text="Forbidden")
    at = _submit_with_mocked_predict(monkeypatch, resp, None)
    assert not at.exception
    assert len(at.error) == 1
    assert "403" in at.error[0].value


def test_raw_tab_rejects_malformed_json_cleanly():
    at = _fresh_app()
    for t in at.text_area:
        if t.label == "Request body":
            t.set_value("{not valid json").run()
            break
    for b in at.button:
        if b.label == "Send raw request":
            b.click().run()
            break
    assert not at.exception
    assert any("Invalid JSON" in e.value for e in at.error)


def test_advanced_field_rejects_non_numeric_input():
    at = _fresh_app()
    for t in at.text_input:
        if t.key == "card1":
            t.set_value("not-a-number").run()
            break
    for b in at.button:
        if b.label == "Score this transaction":
            b.click().run()
            break
    assert not at.exception
    assert any("needs to be a number" in w.value for w in at.warning)


def test_build_payload_omits_unprovided_fields():
    payload = build_payload(
        89.99, 14, 3, "W", "visa", "debit", "gmail.com", "desktop",
        {"card1": None, "card2": None, "card3": None, "card5": None,
         "addr1": None, "addr2": None, "dist1": None, "dist2": None,
         "R_emaildomain": None, "DeviceInfo": None},
    )
    assert payload["TransactionAmt"] == 89.99
    assert payload["TransactionDT"] == 3 * 86400 + 14 * 3600
    assert payload["ProductCD"] == "W"
    assert payload["card4"] == "visa"
    assert payload["card6"] == "debit"
    assert payload["P_emaildomain"] == "gmail.com"
    assert payload["DeviceType"] == "desktop"
    for k in ("card1", "card2", "addr1", "R_emaildomain", "DeviceInfo"):
        assert k not in payload


def test_build_payload_includes_provided_advanced_fields():
    payload = build_payload(
        15.0, 7, 0, "C", "mastercard", "credit", NOT_PROVIDED, NOT_PROVIDED,
        {"card1": 150.0, "card2": None, "card3": None, "card5": None,
         "addr1": 300.0, "addr2": None, "dist1": None, "dist2": None,
         "R_emaildomain": "yahoo.com", "DeviceInfo": "SAMSUNG SM-G935F Build/NRD90M"},
    )
    assert "P_emaildomain" not in payload
    assert "DeviceType" not in payload
    assert payload["card1"] == 150.0
    assert payload["addr1"] == 300.0
    assert "card2" not in payload
    assert payload["R_emaildomain"] == "yahoo.com"
    assert payload["DeviceInfo"] == "SAMSUNG SM-G935F Build/NRD90M"


def test_parse_optional_float_blank_means_omit():
    value, warning = parse_optional_float("")
    assert value is None
    assert warning is None


def test_parse_optional_float_valid_number():
    value, warning = parse_optional_float("123.45")
    assert value == 123.45
    assert warning is None


def test_parse_optional_float_garbage_input():
    value, warning = parse_optional_float("not-a-number")
    assert value is None
    assert warning is not None
    assert "not-a-number" in warning


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_ENDPOINT_TEST") != "1",
    reason="Hits the real AWS endpoint. Run with "
           "RUN_LIVE_ENDPOINT_TEST=1 pytest test_streamlit_app.py -v to include it.",
)
def test_live_endpoint_returns_a_valid_prediction():
    resp, err = fraud_demo_logic.call_predict({"TransactionAmt": 10.0})
    assert err is None, f"Could not reach the live endpoint: {err}"
    assert resp.status_code == 200, f"Live endpoint returned HTTP {resp.status_code}"
    data = resp.json()
    assert set(data.keys()) >= {"prediction", "probability", "threshold"}
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0