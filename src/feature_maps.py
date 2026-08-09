"""
src/feature_maps.py — Shared Category Encoding Maps (used by the Bonus UI)
=============================================================================
WHY THIS FILE EXISTS:
    src/train.py fits a fresh sklearn LabelEncoder on each categorical column
    at training time. LabelEncoder assigns integer codes in ALPHABETICAL
    order of the unique string values it sees — e.g. for a column with
    values ["Yes", "No"], it assigns No=0, Yes=1 (alphabetical).

    The FastAPI schema (api/main.py) and the trained model both expect
    these exact integer codes. This file hardcodes that same alphabetical
    mapping so the bonus Streamlit UI (ui/streamlit_app.py) can show
    human-readable dropdowns ("Yes" / "No") while sending the correct
    encoded integers to the API — without duplicating this logic.

CAVEAT (worth discussing with students):
    This only works because the standard Telco Customer Churn dataset
    always contains the full, fixed vocabulary for each column. Hardcoding
    the encoder's output like this is a workshop simplification — in a
    real production system, you would persist the *fitted* LabelEncoder
    (or better, use a OneHotEncoder / ColumnTransformer saved alongside
    the model) rather than re-deriving the mapping by hand. This is a
    great "what would you do differently in production?" discussion point
    for Session 10 (Interview Prep).
"""

# ── Two-category columns: alphabetical -> No=0, Yes=1 ────────────────────────
YES_NO_MAP = {"No": 0, "Yes": 1}

# ── Gender: alphabetical -> Female=0, Male=1 ──────────────────────────────────
GENDER_MAP = {"Female": 0, "Male": 1}

# ── MultipleLines: alphabetical -> No=0, No phone service=1, Yes=2 ───────────
MULTIPLE_LINES_MAP = {"No": 0, "No phone service": 1, "Yes": 2}

# ── InternetService: alphabetical -> DSL=0, Fiber optic=1, No=2 ──────────────
INTERNET_SERVICE_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}

# ── Shared by OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, ───
# ── StreamingTV, StreamingMovies: alphabetical ->                          ───
# ── No=0, No internet service=1, Yes=2                                     ───
INTERNET_DEPENDENT_MAP = {"No": 0, "No internet service": 1, "Yes": 2}

# ── Contract: alphabetical -> Month-to-month=0, One year=1, Two year=2 ──────
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# ── PaymentMethod: alphabetical ->                                          ──
# ── Bank transfer (automatic)=0, Credit card (automatic)=1,                ──
# ── Electronic check=2, Mailed check=3                                     ──
PAYMENT_METHOD_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}
