"""
src/evaluate.py — Automated Evaluation Gates (Session 4)
=========================================================
USAGE:
    # Load from MLflow run (primary workflow — after Session 2):
    python src/evaluate.py --run_id YOUR_RUN_ID_HERE

    # Load from local pkl (DVC pipeline fallback):
    python src/evaluate.py --model_path models/best_model.pkl

    # Custom thresholds:
    python src/evaluate.py --run_id abc123 --min_auc_roc 0.85

EXIT CODES:
    0 → all gates passed (safe to register and deploy)
    1 → one or more gates failed (deployment BLOCKED)

    The non-zero exit code is what GitHub Actions detects in Session 8.
    If this script exits with 1, the CI pipeline stops — the Docker
    image is never built and the bad model never reaches production.

WHAT THIS SOLVES:
    Notebook Break 5: "No automated tests — accuracy dropped from 91%
    to 64% with nobody noticing."
    This script makes that impossible.
"""

import argparse
import json
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Reuse the preprocessing function from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import load_and_preprocess


# ── Validation Gates ─────────────────────────────────────────────────────────
# These are the minimum thresholds a model must pass to be deployed.
# They are overrideable via command-line args so the CI pipeline can use
# the same values stored in params.yaml.
DEFAULT_GATES = {
    "accuracy": 0.78,
    "auc_roc":  0.80,
    "f1_score": 0.55,
    
}


def evaluate(
    data_path: str,
    model=None,
    run_id: str = None,
    model_path: str = None,
    gates: dict = None,
) -> dict:
    """
    Load a model, evaluate it on a holdout set, and check it against
    all quality gates. Returns a dict of metrics.
    Raises SystemExit(1) if any gate fails.
    """
    gates = gates or DEFAULT_GATES

    # ── Load model ───────────────────────────────────────────────────────────
    if model is not None:
        # Called directly with a model object (e.g., from test suite)
        pass
    elif run_id:
        print(f"[evaluate] Loading model from MLflow run: {run_id}")
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
    elif model_path:
        print(f"[evaluate] Loading model from path: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        print("[evaluate] ERROR: provide --run_id or --model_path")
        sys.exit(1)

    # ── Load and split data ──────────────────────────────────────────────────
    X, y = load_and_preprocess(data_path)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Compute metrics ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc_roc":  round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    }

    # ── Print evaluation results ─────────────────────────────────────────────
    print("\n── Evaluation Results ─────────────────────────────")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("──────────────────────────────────────────────────\n")

    # ── Apply gates ──────────────────────────────────────────────────────────
    failed = []
    print("── Quality Gate Results ───────────────────────────")
    for gate_name, threshold in gates.items():
        value = metrics[gate_name]
        passed = value >= threshold
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  [{symbol}] {gate_name:<12}: {value:.4f}  (min: {threshold})  {status}")
        if not passed:
            failed.append(f"{gate_name} = {value:.4f} < {threshold}")
    print("──────────────────────────────────────────────────\n")

    # ── Write eval metrics for DVC and CI ───────────────────────────────────
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_scores.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[evaluate] Metrics written to metrics/eval_scores.json")

    # ── Gate decision ────────────────────────────────────────────────────────
    if failed:
        print("GATE FAILED — Model did not meet quality thresholds.")
        print("Deployment BLOCKED. Investigate and retrain.")
        for f_msg in failed:
            print(f"  ✗  {f_msg}")
        print()
        sys.exit(1)   # ← GitHub Actions detects this non-zero exit code

    print("All gates PASSED. Model approved for the MLflow Registry.")
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model against quality gates"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run_id",     help="MLflow Run ID")
    source.add_argument("--model_path", help="Local path to .pkl file",
                        default="models/best_model.pkl")

    parser.add_argument("--data_path",     default="data/raw/telco_churn.csv")
    parser.add_argument("--min_accuracy",  type=float, default=DEFAULT_GATES["accuracy"])
    parser.add_argument("--min_auc_roc",   type=float, default=DEFAULT_GATES["auc_roc"])
    parser.add_argument("--min_f1_score",  type=float, default=DEFAULT_GATES["f1_score"])
    parser.add_argument("--tracking_uri",  default="http://98.130.129.135:5001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    custom_gates = {
        "accuracy": args.min_accuracy,
        "auc_roc":  args.min_auc_roc,
        "f1_score": args.min_f1_score,
    }

    evaluate(
        data_path=args.data_path,
        run_id=args.run_id,
        model_path=args.model_path if not args.run_id else None,
        gates=custom_gates,
    )
