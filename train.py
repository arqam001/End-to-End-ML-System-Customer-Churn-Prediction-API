import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
accuracy = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "model/churn_model.pkl")
print("Model saved to model/churn_model.pkl")
