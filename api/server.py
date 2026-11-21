from fastapi import FastAPI

from database.database import (
    engine,
    SessionLocal
)

from database.models import Base

from database.crud import (
    create_transaction
)

from src.prediction import (
    load_model,
    predict_transaction
)

from src.config import (
    MODEL_PATH
)

from api.schemas import (
    FraudRequest
)

Base.metadata.create_all(
    bind=engine
)

app = FastAPI(
    title="Fraud Detection API"
)

model = load_model(
    MODEL_PATH
)


@app.get("/")
def home():

    return {
        "message":
        "Fraud Detection API Running"
    }


@app.post("/predict")
def predict(
    request: FraudRequest
):

    features = [
        request.Time,
        request.V1,
        request.V2,
        request.V3,
        request.V4,
        request.V5,
        request.V6,
        request.V7,
        request.V8,
        request.V9,
        request.V10,
        request.V11,
        request.V12,
        request.V13,
        request.V14,
        request.V15,
        request.V16,
        request.V17,
        request.V18,
        request.V19,
        request.V20,
        request.V21,
        request.V22,
        request.V23,
        request.V24,
        request.V25,
        request.V26,
        request.V27,
        request.V28,
        request.Amount
    ]

    result = predict_transaction(
        model,
        features
    )

    db = SessionLocal()

    create_transaction(
        db,
        amount=request.Amount,
        prediction=result[
            "prediction"
        ],
        probability=result[
            "probability"
        ]
    )

    db.close()

    return result