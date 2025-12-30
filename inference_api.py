from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = joblib.load("model/churn_model.pkl")

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_calls: int = Field(..., ge=0, le=50)

@app.post("/predict")
def predict_churn(data: CustomerData):
    features = np.array([[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        data.support_calls
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 2)
    }
