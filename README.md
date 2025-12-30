# End-to-End-ML-System-Customer-Churn-Prediction-API
End-to-end machine learning system that trains a churn prediction model and serves real-time predictions via a production-ready FastAPI service.
# End-to-End ML System: Customer Churn Prediction

## Problem
Customer churn is a major business risk. The goal of this project is to predict whether a customer is likely to churn based on usage and account-level features, enabling proactive retention strategies.

## Solution
This project implements a complete machine learning pipeline:
- Data ingestion and preprocessing
- Model training and evaluation
- Model persistence
- Real-time inference via REST API

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- FastAPI
- Uvicorn

## ML Approach
- Model: Random Forest Classifier
- Features: customer tenure, usage, support interactions, billing data
- Target: binary churn label (0 = No churn, 1 = Churn)

## Architecture


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python train.py
uvicorn inference_api:app --reload
POST /predict
{
  "tenure": 12,
  "monthly_charges": 70,
  "total_charges": 840,
  "support_calls": 3
}
expected result:
Accuracy ~85% on validation data
Low latency inference (<50ms)
Tradeoffs & Improvements
Replace RandomForest with XGBoost
Add feature store
Add model monitoring & drift detection
CAN Connect to Flutter mobile frontend
Dockerize and deploy to cloud
