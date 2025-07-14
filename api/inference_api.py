"""
Nutrition Model Inference API
----------------------------
- POST /predict: Accepts a list of records (dicts of feature values) and returns predictions for Protein, Sodium, and Calories.
- Assumes input features are already preprocessed (encoded/scaled) as in training.

Example request:
POST /predict
{
  "instances": [
    {"Age": 30, "Weight": 70, ...},
    {"Age": 45, "Weight": 80, ...}
  ]
}

Example response:
{
  "predictions": [
    {"Protein": 65.2, "Sodium": 2.1, "Calories": 2200.5},
    {"Protein": 72.8, "Sodium": 2.4, "Calories": 2500.0}
  ]
}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "nutrition_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "nutrition_model_metadata.json")
PREPROCESS_PATH = os.path.join(BASE_DIR, "data", "preprocessing_objects.pkl")

# Load model, metadata, and preprocessing objects at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(METADATA_PATH):
    raise RuntimeError(f"Metadata file not found at {METADATA_PATH}")
if not os.path.exists(PREPROCESS_PATH):
    raise RuntimeError(f"Preprocessing objects not found at {PREPROCESS_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
with open(PREPROCESS_PATH, "rb") as f:
    preprocessing_objects = pickle.load(f)

FEATURE_NAMES = metadata["feature_names"]
TARGET_NAMES = ["Protein", "Sodium", "Calories"]
label_encoders = preprocessing_objects["label_encoders"]
scaler = preprocessing_objects["scaler"]

# Define which features are categorical/numerical (update as needed)
CATEGORICALS = ["Gender", "Activity Level", "Dietary Preference", "Disease", "Age_Group", "Weight_Category", "Disease_Severity", "activity_goal_combo"]
NUMERICALS = ["Ages", "Height", "Weight", "BMI", "Disease_Count"]

import numpy as np

def preprocess_input(raw_instances):
    # Convert to DataFrame
    df = pd.DataFrame(raw_instances)
    # --- Feature engineering: BMI ---
    if "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    # --- Disease flags ---
    if "Disease" in df.columns:
        diseases = ["Weight Gain", "Hypertension", "Heart Disease", "Kidney Disease", "Diabetes", "Acne"]
        for disease in diseases:
            df[f"Has_{disease.replace(' ', '_')}"] = df["Disease"].str.contains(disease, case=False, na=False).astype(int)
        df["Disease_Count"] = df["Disease"].str.count(",") + 1
        df["has_metabolic_disorder"] = df["Disease"].str.contains("diabetes|hypertension", case=False, na=False).astype(int)
        df["has_cardiac_risk"] = df["Disease"].str.contains("heart disease", case=False, na=False).astype(int)
    # --- Encode categoricals ---
    for col in label_encoders:
        if col in df.columns:
            le = label_encoders[col]
            df[f"{col}_encoded"] = le.transform(df[col].astype(str))
    # --- Derived categoricals (use defaults if missing) ---
    for col in ["Age_Group", "Weight_Category", "Disease_Severity", "activity_goal_combo"]:
        if col not in df.columns:
            df[col] = "None"
        if col in label_encoders:
            le = label_encoders[col]
            df[f"{col}_encoded"] = le.transform(df[col].astype(str))
    # --- Scale numericals ---
    for col in NUMERICALS:
        scaled_col = f"{col}_scaled"
        if col in df.columns and scaler is not None and hasattr(scaler, "mean_"):
            # Use scaler fitted on training data
            try:
                df[scaled_col] = scaler.transform(df[[col]]).flatten()
            except Exception:
                # If scaler expects more features, skip scaling for demo
                pass
    # --- Fill missing features with 0 or default ---
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    # --- Reorder columns ---
    X = df[FEATURE_NAMES]
    return X

app = FastAPI(title="Nutrition Model Inference API")

# Allow CORS for all origins (for testing/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

@app.get("/")
def root():
    return {"message": "Welcome to the Nutrition Model Inference API! Use POST /predict to get predictions."}

@app.post("/predict")
def predict(request: PredictRequest):
    # Preprocess raw input
    try:
        X = preprocess_input(request.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")
    # Predict
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    # Format response
    results = [
        {k: float(v) for k, v in zip(TARGET_NAMES, row)}
        for row in preds
    ]
    return {"predictions": results} 