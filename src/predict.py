import os
import joblib
import pandas as pd

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Example input (must match training features)
sample_data = pd.DataFrame([{
    "return_1d": 0.002,
    "sma_5": 150,
    "sma_20": 148,
    "volatility_20": 0.015,
    "sentiment_mean": 0.12,
    "headline_count": 8
}])

# Predict
prediction = model.predict(sample_data)
probability = model.predict_proba(sample_data)[0][1]

print("\nPrediction:")
print("Will price go up tomorrow?", "YES" if prediction[0] == 1 else "NO")
print("Confidence:", round(probability, 3))
