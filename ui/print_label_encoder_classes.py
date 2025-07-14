import pickle
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESS_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessing_objects.pkl")

with open(PREPROCESS_PATH, "rb") as f:
    preprocessing_objects = pickle.load(f)
label_encoders = preprocessing_objects["label_encoders"]

derived_cols = ["Age_Group", "Weight_Category", "Disease_Severity", "activity_goal_combo"]

for col in derived_cols:
    if col in label_encoders:
        print(f"{col}: {label_encoders[col].classes_}")
    else:
        print(f"{col}: No label encoder found.") 