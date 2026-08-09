"""
src/predict.py — Shared Prediction Logic (Session 5) 23
=====================================================
This module is imported by api/main.py and any script that needs
to make predictions. Keeping prediction logic here (not inside main.py)
means it can be unit-tested independently of the web framework.

USAGE:
    from src.predict import load_model, predict_churn

    model = load_model()                     # loads from MLflow Registry
    result = predict_churn(model, features)  # returns PredictionResult
"""

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import mlflow.sklearn
import pandas as pd


# ── Model source configuration ────────────────────────────────────────────────
MODEL_NAME  = "churn-prediction-xgboost"
MODEL_STAGE = "Production"
FALLBACK_PATH = "models/best_model.pkl"  # used if MLflow is unavailable


# ── Result data class ─────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Structured output from a single prediction call."""
    churn_probability: float
    prediction:        str     # "Will churn" or "Will not churn"
    confidence:        str     # "High", "Medium", or "Low"
    model_source:      str     # "registry" or "local_pkl"
    model_version:     str     # registry version or "local"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(tracking_uri: Optional[str] = None) -> tuple:
    """
    Load the Production model from the MLflow Model Registry.
    Falls back to the local .pkl file if MLflow is unavailable.

    Returns:
        (model, version_str)  — the sklearn-compatible model and its version tag
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://98.130.129.135:5001")
    mlflow.set_tracking_uri(uri)

    # ── Try MLflow Registry first ────────────────────────────────────────────
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)

        # Fetch the version number for logging and API responses
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        version = versions[0].version if versions else "unknown"

        print(f"[predict] Loaded from MLflow Registry: {MODEL_NAME} v{version} ({MODEL_STAGE})")
        return model, version

    except Exception as e:
        print(f"[predict] WARNING: MLflow Registry unavailable — {e}")
        print(f"[predict] Falling back to local model: {FALLBACK_PATH}")

    # ── Fallback: local pkl ──────────────────────────────────────────────────
    if os.path.exists(FALLBACK_PATH):
        with open(FALLBACK_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[predict] Loaded from local pkl: {FALLBACK_PATH}")
        return model, "local"

    raise RuntimeError(
        f"No model found. MLflow Registry unavailable and {FALLBACK_PATH} does not exist.\n"
        f"Run `python src/train.py` to generate a local model, or start MLflow with "
        f"`mlflow ui --port 5001` and register a model."
    )


# ── Prediction function ───────────────────────────────────────────────────────
def predict_churn(model, features: dict) -> PredictionResult:
    """
    Run a single churn prediction.

    Args:
        model:    sklearn-compatible model (from load_model)
        features: dict of feature name → value (matching CustomerFeatures schema)

    Returns:
        PredictionResult with probability, label, and confidence band
    """
    # Convert to DataFrame (required by sklearn pipeline)
    df = pd.DataFrame([features])

    # Predict
    churn_prob = float(model.predict_proba(df)[0][1])
    prediction = "Will churn" if churn_prob >= 0.5 else "Will not churn"

    # Confidence band
    # High:   probability is clearly on one side (>0.75 or <0.25)
    # Medium: probability is moderately decisive (>0.60 or <0.40)
    # Low:    probability is close to the 0.5 decision boundary
    if churn_prob >= 0.75 or churn_prob <= 0.25:
        confidence = "High"
    elif churn_prob >= 0.60 or churn_prob <= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResult(
        churn_probability=round(churn_prob, 4),
        prediction=prediction,
        confidence=confidence,
        model_source="registry",
        model_version="unknown",   # overridden by the API on startup
    )


# ── Expected feature columns (order must match training data) ─────────────────
# This list is used by api/main.py's Pydantic schema for field ordering.
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
