"""
schema.py — Pydantic request/response models.

Design: TransactionAmt is the only required field. All others are optional; missing values become NaN in preprocessing, 
which XGBoost handles natively.

The 300+ V-features (V1-V339), C-features, D-features, and id_ features are accepted via extra="allow". 
Enumerating them individually would make this file unreadable and is not needed for type-safety on fields a caller would realistically
know about.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Required 
    TransactionAmt: float                   # Raw dollar amount; log1p applied in preprocess

    # Time
    TransactionDT: Optional[float] = None   # Seconds from Vesta's internal reference point.
                                            # Absent -> hour_of_day and day_of_week_proxy are NaN.

    # Product / Card
    ProductCD: Optional[str]   = None
    card1:     Optional[float] = None
    card2:     Optional[float] = None
    card3:     Optional[float] = None
    card4:     Optional[str]   = None       # "visa" | "mastercard" | "discover" | "american express"
    card5:     Optional[float] = None
    card6:     Optional[str]   = None       # "credit" | "debit" | "charge card" | "debit or credit"

    # Address / Distance
    addr1:  Optional[float] = None
    addr2:  Optional[float] = None
    dist1:  Optional[float] = None
    dist2:  Optional[float] = None

    # Email
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    # Device
    DeviceType: Optional[str] = None        # "desktop" | "mobile"
    DeviceInfo: Optional[str] = None

    # V-features, C-features, D-features, M-features, id_ fields:
    # accepted via extra="allow". Pass any of the 422 model features
    # as additional JSON keys; unrecognised keys are silently ignored.


class PredictionOutput(BaseModel):
    prediction:  int    # 0 = pass, 1 = flag for manual review
    probability: float  # Raw model score (0-1)
    threshold:   float  # Operating threshold used for this decision