"""
Machine Downtime Predictor - FastAPI backend
Loads the trained scikit-learn pipeline (model.pkl) once at cold start and
exposes a /predict endpoint used by the Next.js frontend.

Entry point used by Vercel's Python runtime (must expose a module-level
`app` ASGI object). Also runnable locally with:
    uvicorn api.index:app --reload
"""

import os
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Model loading (runs once per cold start, not per-request)
# --------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
# Try a few candidate locations: Vercel's Python runtime docs warn that the
# working directory at runtime is the project root, not this file's folder,
# so we check both a __file__-relative path and a cwd-relative one.
_CANDIDATE_PATHS = [
    _HERE.parent / "model" / "machine_downtime_pipeline.pkl",
    Path.cwd() / "model" / "machine_downtime_pipeline.pkl",
    Path.cwd() / "backend" / "model" / "machine_downtime_pipeline.pkl",
]
MODEL_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])

# The exact order/names the pipeline was fitted on.
FEATURE_COLUMNS = [
    "Torque(Nm)",
    "Hydraulic_Pressure(bar)",
    "Cutting(kN)",
    "Coolant_Pressure(bar)",
    "Spindle_Speed(RPM)",
    "Coolant_Temperature",
]

# Class 1 = machine failure / needs maintenance, class 0 = normal operation.
FAILURE_CLASS = 1

# Below this confidence, flag the reading for manual review instead of
# trusting the model blindly.
MANUAL_REVIEW_THRESHOLD = 0.60

_model = None
_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
except Exception as exc:  # noqa: BLE001 - we want to surface any load issue
    _load_error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class SensorReading(BaseModel):
    """Matches the six sensor features the model was trained on.

    Field names are plain snake_case for a clean JSON body from the
    frontend; `alias` maps each one back to the exact column name the
    pipeline expects.
    """

    model_config = ConfigDict(populate_by_name=True)

    torque_nm: float = Field(..., alias="Torque(Nm)", description="Torque (Nm)")
    hydraulic_pressure_bar: float = Field(..., alias="Hydraulic_Pressure(bar)", description="Hydraulic pressure (bar)")
    cutting_kn: float = Field(..., alias="Cutting(kN)", description="Cutting force (kN)")
    coolant_pressure_bar: float = Field(..., alias="Coolant_Pressure(bar)", description="Coolant pressure (bar)")
    spindle_speed_rpm: float = Field(..., alias="Spindle_Speed(RPM)", description="Spindle speed (RPM)")
    coolant_temperature: float = Field(..., alias="Coolant_Temperature", description="Coolant temperature (°C)")


class PredictionResponse(BaseModel):
    prediction: Literal["Machine_Failure", "No_Machine_Failure"]
    needs_maintenance: bool
    probability_failure: float
    confidence: float
    needs_manual_review: bool


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="Machine Downtime Predictor API",
    description="Predictive maintenance inference API for sensor readings.",
    version="1.0.0",
)

# CORS: allow the deployed frontend + local dev. Set FRONTEND_ORIGIN on
# Vercel (backend project) to your frontend's URL, e.g.
# https://your-frontend.vercel.app
_allowed_origins = {"http://localhost:3000"}
_extra_origin = os.environ.get("FRONTEND_ORIGIN")
if _extra_origin:
    _allowed_origins.add(_extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "machine-downtime-predictor", "model_loaded": _model is not None}


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {_load_error}")
    return {"status": "healthy"}


@app.get("/debug")
def debug():
    """Diagnostic endpoint - safe to remove once the deployment is working.
    Shows exactly where this function looked for the model file and why
    loading it did or didn't work."""
    import sklearn

    return {
        "cwd": str(Path.cwd()),
        "file_dir": str(_HERE),
        "resolved_model_path": str(MODEL_PATH),
        "resolved_path_exists": MODEL_PATH.exists(),
        "candidate_paths": [{"path": str(p), "exists": p.exists()} for p in _CANDIDATE_PATHS],
        "sklearn_version": sklearn.__version__,
        "model_loaded": _model is not None,
        "load_error": _load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model is not available: {_load_error}")

    data = reading.model_dump(by_alias=True)
    row = pd.DataFrame([[data[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Basic sanity check: reject rows that are entirely non-finite.
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise HTTPException(status_code=400, detail="Sensor values must be finite numbers.")

    try:
        probabilities = _model.predict_proba(row)[0]
        classes = list(_model.classes_)
        prob_failure = float(probabilities[classes.index(FAILURE_CLASS)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    is_failure = prob_failure >= 0.5
    confidence = prob_failure if is_failure else (1 - prob_failure)

    return PredictionResponse(
        prediction="Machine_Failure" if is_failure else "No_Machine_Failure",
        needs_maintenance=is_failure,
        probability_failure=round(prob_failure, 4),
        confidence=round(confidence, 4),
        needs_manual_review=confidence < MANUAL_REVIEW_THRESHOLD,
    )
