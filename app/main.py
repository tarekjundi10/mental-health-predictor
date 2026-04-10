import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(base_dir, "models", "model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))

app = FastAPI(
    title="Mental Health Risk Predictor API",
    description="Predicts whether an individual is at high or low risk for a mental health condition based on lifestyle factors.",
    version="1.0.0"
)

class PatientInput(BaseModel):
    age: float
    gender: int                      # 0 = Female, 1 = Male
    exercise_level: int              # 0 = High, 1 = Low, 2 = Moderate
    diet_type: int                   # 0 = Balanced, 1 = Vegan, 2 = Vegetarian
    sleep_hours: float
    work_hours_per_week: float
    screen_time_per_day: float
    social_interaction_score: float
    happiness_score: float

@app.get("/")
def root():
    return {
        "message": "Mental Health Risk Predictor API is running.",
        "docs": "/docs"
    }

@app.post("/predict")
def predict(data: PatientInput):
    features = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Exercise Level": data.exercise_level,
        "Diet Type": data.diet_type,
        "Sleep Hours": data.sleep_hours,
        "Work Hours per Week": data.work_hours_per_week,
        "Screen Time per Day (Hours)": data.screen_time_per_day,
        "Social Interaction Score": data.social_interaction_score,
        "Happiness Score": data.happiness_score
    }])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "risk_level": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": round(float(probability), 3),
        "recommendation": (
            "Consider consulting a mental health professional."
            if prediction == 1 else
            "No immediate concern based on current lifestyle indicators."
        )
    }