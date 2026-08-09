"""
api/main.py — FastAPI Model Serving Application (Session 5)
=============================================================
RUN LOCALLY:
    uvicorn api.main:app --reload --port 8000

RUN INSIDE DOCKER (Session 6):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

ENDPOINTS:
    GET  /health    → service status, model name/version, loaded flag
    POST /predict    → accepts CustomerFeatures JSON, returns churn prediction

INTERACTIVE DOCS (Swagger UI):
    http://localhost:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# Allow `from src.predict import ...` when run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import load_model, predict_churn


# ── Global model state ──────────────────────────────────────────────────────
# Loaded once at startup and reused for every request — loading a model
# on every request would be extremely slow and wasteful.
model = None
model_version = "unknown"
model_source = "none"


# ── Lifespan: load the model once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version, model_source
    try:
        model, model_version = load_model()
        model_source = "registry" if model_version != "local" else "local_pkl"
        print(f"[api] Model ready — version: {model_version}, source: {model_source}")
    except Exception as e:
        print(f"[api] ERROR loading model: {e}")
        print("[api] /predict will return 503 until a model becomes available.")
    yield
    # (no shutdown cleanup needed)


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predicts customer churn probability using an XGBoost model "
        "served from the MLflow Model Registry. Built during the "
        "Krysha Academy MLOps Workshop."
    ),
    version="1.0.0",
    lifespan=lifespan,
    root_path="/vdvc init ishwanath-150"
)


# ── Request schema ───────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Feature schema for a single churn prediction request.
    Field names and order match the Telco Customer Churn dataset
    (minus customerID and Churn, which are not inputs).
    """
    gender:           int   # 0 = Female, 1 = Male
    SeniorCitizen:    int   # 0 = No, 1 = Yes
    Partner:          int   # 0 = No, 1 = Yes
    Dependents:       int   # 0 = No, 1 = Yes
    tenure:           int   # months as a customer (0–72)
    PhoneService:     int
    MultipleLines:    int
    InternetService:  int   # 0 = DSL, 1 = Fibre optic, 2 = No
    OnlineSecurity:   int
    OnlineBackup:     int
    DeviceProtection: int
    TechSupport:      int
    StreamingTV:      int
    StreamingMovies:  int
    Contract:         int   # 0 = Month-to-month, 1 = One year, 2 = Two year
    PaperlessBilling: int
    PaymentMethod:    int   # 0–3
    MonthlyCharges:   float
    TotalCharges:     float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
                "tenure": 24, "PhoneService": 1, "MultipleLines": 0,
                "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1,
                "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
                "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
                "PaymentMethod": 2, "MonthlyCharges": 65.5, "TotalCharges": 1572.0,
            }
        }
    )


# ── Response schema ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability: float
    prediction:        str
    confidence:        str
    model_version:     str


class HealthResponse(BaseModel):
    status:        str
    model_name:    str
    model_version: str
    model_source:  str
    model_loaded:  bool


# ── Health check endpoint ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Service health check. Used by load balancers, monitoring tools,
    and Session 7's deployment verification.
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_name="churn-prediction-xgboost",
        model_version=model_version,
        model_source=model_source,
        model_loaded=model is not None,
    )


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns 503 if the model failed to load at startup (check /health first).
    Returns 422 automatically if the request body doesn't match the schema
    (handled by FastAPI + Pydantic — no custom code needed).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Check MLflow Registry connection "
                "or confirm models/best_model.pkl exists."
            ),
        )

    result = predict_churn(model, features.model_dump())

    return PredictionResponse(
        churn_probability=result.churn_probability,
        prediction=result.prediction,
        confidence=result.confidence,
        model_version=model_version,
    )


# ── Root endpoint (friendly landing) ─────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API — see /docs for interactive documentation",
        "health": "/health",
        "predict": "/predict (POST)",
    }
