import os
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Torque(Nm)",
    "Hydraulic_Pressure(bar)",
    "Cutting(kN)",
    "Coolant_Pressure(bar)",
    "Spindle_Speed(RPM)",
    "Coolant_Temperature",
]

FAILURE_CLASS = 1
MANUAL_REVIEW_THRESHOLD = 0.60

_model = None
_load_error = None

try:
  # قراءة الداتا ست الحقيقية مباشرة من مجلد المشروع
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  csv_path = os.path.join(BASE_DIR, "dataset.csv")

  df = pd.read_csv(csv_path)

  # تأكد إن اسم عمود الـ Target عندك في ملف الـ csv هو "Failure" أو غيره حسب ملفك
  # لو اسم العمود مختلف عندك قولي عليه
  target_column = "Failure" if "Failure" in df.columns else df.columns[-1]

  X = df[FEATURE_COLUMNS]
  y = df[target_column]

  _model = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", RandomForestClassifier(random_state=42)),
  ])
  _model.fit(X, y)
except Exception as exc:
  _load_error = str(exc)
  print(f"Error loading dataset or training: {exc}")


class SensorReading(BaseModel):
  model_config = ConfigDict(populate_by_name=True)

  torque_nm: float = Field(..., alias="Torque(Nm)")
  hydraulic_pressure_bar: float = Field(..., alias="Hydraulic_Pressure(bar)")
  cutting_kn: float = Field(..., alias="Cutting(kN)")
  coolant_pressure_bar: float = Field(..., alias="Coolant_Pressure(bar)")
  spindle_speed_rpm: float = Field(..., alias="Spindle_Speed(RPM)")
  coolant_temperature: float = Field(..., alias="Coolant_Temperature")


class PredictionResponse(BaseModel):
  prediction: Literal["Machine_Failure", "No_Machine_Failure"]
  needs_maintenance: bool
  probability_failure: float
  confidence: float
  needs_manual_review: bool


app = FastAPI(
    title="Machine Downtime Predictor API", version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
  return {
      "status": "ok",
      "service": "machine-downtime-predictor",
      "model_loaded": _model is not None,
  }


@app.get("/health")
def health():
  if _model is None:
    raise HTTPException(
        status_code=503, detail=f"Model failed to load: {_load_error}"
    )
  return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
  if _model is None:
    raise HTTPException(
        status_code=503, detail=f"Model is not available: {_load_error}"
    )

  data = reading.model_dump(by_alias=True)
  row = pd.DataFrame(
      [[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
  )

  probabilities = _model.predict_proba(row)[0]
  classes = list(_model.classes_)
  prob_failure = float(probabilities[classes.index(FAILURE_CLASS)])

  is_failure = prob_failure >= 0.5
  confidence = prob_failure if is_failure else (1 - prob_failure)

  return PredictionResponse(
      prediction="Machine_Failure" if is_failure else "No_Machine_Failure",
      needs_maintenance=is_failure,
      probability_failure=round(prob_failure, 4),
      confidence=round(confidence, 4),
      needs_manual_review=confidence < MANUAL_REVIEW_THRESHOLD,
  )
