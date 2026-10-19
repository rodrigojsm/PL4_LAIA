from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train simple model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
import os
import pickle

os.makedirs("data", exist_ok=True)

with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model trained and saved as model.pkl")