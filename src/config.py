import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_model.pkl"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

METRICS_PATH = os.path.join(
    REPORTS_DIR,
    "metrics.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    REPORTS_DIR,
    "confusion_matrix.png"
)

ROC_CURVE_PATH = os.path.join(
    REPORTS_DIR,
    "roc_curve.png"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    REPORTS_DIR,
    "feature_importance.png"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)