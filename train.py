import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_IMPORTANCE_PATH
)

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import apply_smote
from src.train_model import train_model
from src.evaluation_model import evaluate

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading Dataset...\n")

df = load_data(DATA_PATH)

print(
    f"Dataset Shape: {df.shape}"
)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

print(
    "\nPreprocessing Data...\n"
)

X, y, scaler = preprocess_data(
    df
)

# --------------------------------------------------
# SMOTE
# --------------------------------------------------

print(
    "\nApplying SMOTE...\n"
)

X_resampled, y_resampled = (
    apply_smote(
        X,
        y
    )
)

print(
    "Class Balancing Complete"
)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

print(
    "\nTraining XGBoost Model...\n"
)

model = train_model(
    X_train,
    y_train
)

print(
    "Model Training Complete"
)

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

print(
    "\nEvaluating Model...\n"
)

metrics = evaluate(
    model,
    X_test,
    y_test
)

print(
    "\nModel Metrics:"
)

for key, value in metrics.items():

    print(
        f"{key}: {value:.4f}"
    )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print(
    "\nGenerating Feature Importance...\n"
)

importance = (
    model.feature_importances_
)

feature_names = list(
    df.drop(
        "Class",
        axis=1
    ).columns
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    importance_df[
        "Feature"
    ].head(15),
    importance_df[
        "Importance"
    ].head(15)
)

plt.xticks(
    rotation=45
)

plt.title(
    "Top 15 Important Features"
)

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_PATH
)

plt.close()

print(
    "Feature Importance Saved"
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel Saved:\n{MODEL_PATH}"
)

# --------------------------------------------------
# SAVE SCALER
