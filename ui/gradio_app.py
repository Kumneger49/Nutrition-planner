import gradio as gr
import pickle
import json
import os
import pandas as pd

# Load model and preprocessing objects
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "nutrition_model.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "model", "nutrition_model_metadata.json")
PREPROCESS_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessing_objects.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
with open(PREPROCESS_PATH, "rb") as f:
    preprocessing_objects = pickle.load(f)

FEATURE_NAMES = metadata["feature_names"]
label_encoders = preprocessing_objects["label_encoders"]
scaler = preprocessing_objects["scaler"]
NUMERICALS = ["Ages", "Height", "Weight", "BMI", "Disease_Count"]

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
            # Ensure we pass a list of strings to the encoder, not a single string
            # Print for debug
            print(f"Encoding column {col}, values: {df[col].tolist()}")
            df[f"{col}_encoded"] = le.transform(df[col].astype(str).tolist())
    # --- Derived categoricals (use defaults if missing) ---
    for col in ["Age_Group", "Weight_Category", "Disease_Severity", "activity_goal_combo"]:
        if col not in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].classes_[0]  # Use the first valid class as default
            else:
                df[col] = ""  # fallback
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

def predict_nutrition(Age, Height, Weight, Gender, Activity_Level, Dietary_Preference, Disease):
    print("Disease type:", type(Disease), "value:", Disease)
    # Disease is a list from CheckboxGroup, but could be a string if only one selected
    if isinstance(Disease, list):
        if Disease:
            disease_str = ", ".join(sorted(Disease))
        else:
            disease_str = ""
    elif isinstance(Disease, str):
        disease_str = Disease
    else:
        disease_str = ""
    print("Disease string passed to DataFrame:", disease_str)
    raw_instance = {
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "Gender": Gender,
        "Activity Level": Activity_Level,
        "Dietary Preference": Dietary_Preference,
        "Disease": disease_str
    }
    X = preprocess_input([raw_instance])
    preds = model.predict(X)[0]
    return float(preds[0]), float(preds[1]), float(preds[2])

# Disease options for user-friendly multi-select
DISEASE_OPTIONS = [
    "Diabetes, Acne, Hypertension, Heart Disease",
    "Diabetes, Acne, Hypertension, Kidney Disease",
    "Diabetes, Acne, Weight Gain, Hypertension, Heart Disease",
    "Diabetes, Acne, Weight Gain, Hypertension, Heart Disease, Kidney Disease",
    "Diabetes, Acne, Weight Loss, Hypertension, Heart Disease, Kidney Disease",
    "Hypertension, Heart Disease",
    "Hypertension, Heart Disease, Kidney Disease",
    "Hypertension, Kidney Disease",
    "Kidney Disease",
    "Weight Gain",
    "Weight Gain, Hypertension, Heart Disease",
    "Weight Gain, Hypertension, Heart Disease, Kidney Disease",
    "Weight Gain, Kidney Disease"
]

# Define Gradio interface
demo = gr.Interface(
    fn=predict_nutrition,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Height (cm)"),
        gr.Number(label="Weight (kg)"),
        gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(
            choices=[
                "Extremely Active", "Lightly Active", "Moderately Active", "Sedentary", "Very Active"
            ],
            label="Activity Level",
            value="Sedentary"
        ),
        gr.Dropdown(choices=["Omnivore", "Vegetarian", "Vegan"], label="Dietary Preference", value="Omnivore"),
        gr.Dropdown(choices=DISEASE_OPTIONS, label="Disease", value=DISEASE_OPTIONS[0])
    ],
    outputs=[
        gr.Number(label="Protein"),
        gr.Number(label="Sodium"),
        gr.Number(label="Calories")
    ],
    title="Personalized Nutrition Predictor",
    description="Enter your details to get personalized nutrition recommendations."
)

if __name__ == "__main__":
    demo.launch(share=True)