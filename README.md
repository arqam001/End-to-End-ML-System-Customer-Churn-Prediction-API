# End-to-End ML System: Customer Churn Prediction API

End-to-end machine learning system that trains a churn prediction model and serves real-time predictions via a production-ready FastAPI service.

The system is packaged as a Docker container and exposed via a FastAPI service for real-time inference.

---

## Problem
Customer churn is a major business risk. The goal of this project is to predict whether a customer is likely to churn based on usage and account-level features, enabling proactive retention strategies.

---

## Solution
This project implements a complete machine learning pipeline:
- Data ingestion and preprocessing
- Model training and evaluation
- Model persistence
- Real-time inference via REST API

---

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- FastAPI
- Uvicorn
- Docker

---

## ML Approach
- Model: Random Forest Classifier
- Features: customer tenure, usage, support interactions, billing data
- Target: binary churn label (0 = No churn, 1 = Churn)

---

## Architecture
Client → FastAPI → Trained ML Model → Prediction

The service is containerized using Docker for consistent local and production deployments.

---

## How to Run (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Run the API
```bash
uvicorn inference_api:app --reload
```

The API will be available at:
```
http://localhost:8000/docs
```

---

## Example Prediction Request

**POST** `/predict`

```json
{
  "tenure": 12,
  "monthly_charges": 70,
  "total_charges": 840,
  "support_calls": 3
}
```

---

## Results
- Accuracy ~85% on validation data
- Low-latency inference (<50ms)

---

## Running with Docker

### Build the image
```bash
docker build -t churn-ml-api .
```

### Run the container
```bash
docker run -p 8000:8000 churn-ml-api
```

The API will be available at:
```
http://localhost:8000/docs
```

---

## Tradeoffs & Improvements
- Replace RandomForest with XGBoost
- Add feature store
- Add model monitoring and drift detection
- Connect to a Flutter mobile frontend
- Deploy to cloud infrastructure

---

## Why This Matters
- Demonstrates an end-to-end ML system
- Shows production-ready API design
- Containerized for real-world deployment
