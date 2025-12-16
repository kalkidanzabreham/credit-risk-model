from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.data_processing import prepare_model_dataset

app = FastAPI(title="Credit Risk Prediction API")

# Load best model from MLflow Registry
MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"
PREPROCESSOR_NAME = "RandomForest_Preprocessor"

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)
preprocessor = mlflow.sklearn.load_model(
    f"models:/{PREPROCESSOR_NAME}/latest" 
)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])

    # Feature engineering
    processed_df, _ = prepare_model_dataset(input_df)
    X = preprocessor.transform(processed_df)    # Prediction
    risk_prob = model.predict_proba(X)[0, 1]
    is_high_risk = bool(risk_prob >= 0.5)

    return PredictionResponse(
        risk_probability=risk_prob,
        is_high_risk=is_high_risk,
    )
