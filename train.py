import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
data = pd.read_csv("data/churn.csv")

X = data.drop("churn", axis=1)
y = data["churn"]

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 3),
    "precision": round(precision_score(y_test, preds), 3),
    "recall": round(recall_score(y_test, preds), 3),
    "trained_at": datetime.utcnow().isoformat(),
    "model_params": model.get_params()
}

# Save model
joblib.dump(model, "model/churn_model.pkl")

# Save metrics
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Training complete")
print(metrics)
