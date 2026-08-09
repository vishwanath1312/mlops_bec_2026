"""
src/train.py — Training Script with MLflow Experiment Tracking (Session 2)
===========================================================================
USAGE (command-line):
    python src/train.py                                      # default params
    python src/train.py --n_estimators 200 --max_depth 4    # custom params

USAGE (via DVC pipeline):
    dvc repro                                                # reads params.yaml

WHAT THIS SCRIPT DOES:
    1. Loads and preprocesses the Telco Customer Churn dataset
    2. Trains an XGBoost classifier with the given hyperparameters
    3. Logs ALL parameters, metrics, and the model artifact to MLflow
    4. Saves the trained model to models/best_model.pkl (for DVC to track)
    5. Writes metrics/scores.json (for dvc metrics show / GitHub Actions)
"""

import argparse
import json
import os
import sys
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ── Data loading and preprocessing ────────────────────────────────────────────
def load_and_preprocess(data_path: str):
    """
    Load the Telco Customer Churn CSV and return feature matrix X and target y.

    Key preprocessing decisions:
    - TotalCharges: convert to numeric with errors='coerce' to handle any
      embedded strings (this is the fix for Notebook Break 3).
    - Missing values: median imputation — safe for tree-based models.
    - Categorical encoding: LabelEncoder on all object columns.
    - customerID: dropped (high-cardinality identifier, not predictive).
    """
    df = pd.read_csv(data_path)

    # FIX FOR NOTEBOOK BREAK 3 ──────────────────────────────────────────────
    # In the broken notebook, TotalCharges was not converted to numeric.
    # If upstream data contains strings like '$85.70' or ' ', pd.to_numeric
    # with errors='coerce' handles all of them gracefully, filling NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # ────────────────────────────────────────────────────────────────────────

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode all remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "customerID" in cat_cols:
        cat_cols.remove("customerID")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"]

    return X, y


# ── Training function ─────────────────────────────────────────────────────────
def train(
    data_path: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: int,
    gamma: float,
    test_size: float,
    random_state: int,
    experiment_name: str,
) -> dict:
    """
    Train the XGBoost model and log everything to MLflow.

    Returns a dict of evaluation metrics.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    X, y = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class balance in both splits
    )
    print(f"[train] Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"[train] Churn rate: {y.mean():.2%}")

    # ── MLflow experiment context ────────────────────────────────────────────
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ── Log all hyperparameters ──────────────────────────────────────────
        params = {
            "n_estimators":     n_estimators,
            "max_depth":        max_depth,
            "learning_rate":    learning_rate,
            "subsample":        subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma":            gamma,
            "test_size":        test_size,
            "random_state":     random_state,
            "data_path":        data_path,
        }
        mlflow.log_params(params)

        # ── Train ────────────────────────────────────────────────────────────
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "auc_roc":   round(float(roc_auc_score(y_test, y_prob)), 4),
            "f1_score":  round(float(f1_score(y_test, y_pred)), 4),
        }

        # ── Log metrics ──────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Log the model as an MLflow artifact ──────────────────────────────
        # This is what Session 4 uses to register from the Run ID.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,  # registration happens in Session 4
        )

        # ── Also save model to disk (for DVC to track) ───────────────────────
        os.makedirs("models", exist_ok=True)
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # ── Write metrics/scores.json (for dvc metrics show) ─────────────────
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/scores.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Console summary ───────────────────────────────────────────────────
        print("\n" + "═" * 50)
        print("  TRAINING COMPLETE")
        print("═" * 50)
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
        print("─" * 50)
        print(f"  MLflow Run ID : {run_id}")
        print(f"  Experiment    : {experiment_name}")
        print(f"  Model saved   : models/best_model.pkl")
        print(f"  Metrics saved : metrics/scores.json")
        print("═" * 50 + "\n")
        print(f"  Copy this Run ID for Session 4:")
        print(f"  >>> {run_id}")

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost churn model with MLflow tracking"
    )
    parser.add_argument("--data_path",        default="data/raw/telco_churn.csv")
    parser.add_argument("--n_estimators",     type=int,   default=300)
    parser.add_argument("--max_depth",        type=int,   default=8)
    parser.add_argument("--learning_rate",    type=float, default=0.05)
    parser.add_argument("--subsample",        type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=int,   default=1)
    parser.add_argument("--gamma",            type=float, default=0.0)
    parser.add_argument("--test_size",        type=float, default=0.2)
    parser.add_argument("--random_state",     type=int,   default=42)
    parser.add_argument("--experiment_name",  default="churn_prediction")
    parser.add_argument("--tracking_uri",     default="http://localhost:5001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    metrics = train(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        test_size=args.test_size,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
    )

    # Exit with non-zero code if metrics are clearly terrible
    # (a basic sanity check — the real gate lives in src/evaluate.py)
    if metrics["auc_roc"] < 0.60:
        print(f"[train] ERROR: AUC-ROC {metrics['auc_roc']:.4f} is below sanity threshold (0.60)")
        sys.exit(1)
