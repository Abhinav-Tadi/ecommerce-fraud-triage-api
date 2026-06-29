"""
main.py — FastAPI application.
"""

from fastapi import FastAPI
from app.schema import TransactionInput, PredictionOutput
from app.inference import predict

app = FastAPI(
    title="E-commerce Fraud Triage API",
    description=(
        "Real-time card-not-present fraud scoring on the IEEE-CIS dataset. "
        "Returns a binary flag (0=pass, 1=review) and calibrated probability. "
        "Operating threshold set at 0.0957 for ~85% recall / 67% precision."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: TransactionInput) -> PredictionOutput:
    result = predict(payload.model_dump())
    return PredictionOutput(**result)