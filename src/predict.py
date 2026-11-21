import joblib
import numpy as np


def load_model(
    model_path
):

    return joblib.load(
        model_path
    )


def predict_transaction(
    model,
    features
):

    features = np.array(
        features
    ).reshape(
        1,
        -1
    )

    prediction = (
        model.predict(
            features
        )[0]
    )

    probability = (
        model.predict_proba(
            features
        )[0][1]
    )

    return {
        "prediction": (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        ),
        "probability": float(
            probability
        )
    }