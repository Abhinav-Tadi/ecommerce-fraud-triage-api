"""
schema.py - Pydantic request/response models.

Design: TransactionAmt is the only required field. All others are optional; missing values become NaN in preprocessing, which XGBoost handles natively.

The 300+ V-features (V1-V339), C-features, D-features, and id_ features are accepted via extra="allow". 
Enumerating them individually would make this file unreadable and is not needed for type-safety on fields a caller would realistically know about.

TransactionAmt must be strictly positive (Field(gt=0)). Two independent reasons:
  1. Numerical: preprocess.py applies np.log1p(TransactionAmt). log1p(-1) is exactly -inf, and log1p(x) for x < -1 is nan. A value at or below -1
     would otherwise reach the model as a corrupted number instead of being rejected at the API boundary.
  2. Business logic: a card-not-present transaction amount of zero or negative has no real-world meaning here. The training data's observed minimum
     was $0.251 - the model has never seen zero or negative amounts, so nothing meaningful could come out of scoring one anyway.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Required 
    TransactionAmt: float = Field(gt=0)     # Raw dollar amount; log1p applied in preprocess. Must be > 0 - see module docstring.

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
    # accepted via extra="allow". Pass any of the 422 model features as additional JSON keys; unrecognised keys are silently ignored.

class PredictionOutput(BaseModel):
    prediction:  int    # 0 = pass, 1 = flag for manual review
    probability: float  # Raw model score (0-1)
    threshold:   float  # Operating threshold used for this decision