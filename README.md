# 🍎 Nutrition Planner: Robust Nutrition Prediction

A machine learning project for predicting personalized nutritional needs (Protein, Sodium, Calories) using a robust, leakage-free pipeline and modern deployment tools.

---

## 🎯 Project Overview

- **Goal:** Predict individual daily needs for Protein, Sodium, and Calories based on demographic and health profile.
- **Model:** MultiOutputRegressor with XGBoost (leakage-free, robust features only)
- **Deployment:** FastAPI inference API and user-friendly Gradio UI
- **Status:** Ready for local and cloud deployment (e.g., Hugging Face Spaces)

---

## 🏗️ Pipeline Highlights

- **Data Leakage Prevention:** All features derived from targets have been removed. Only safe, non-leaky features (demographics, disease flags, encoded categoricals) are used.
- **Feature Engineering:** Includes BMI, disease flags, and robust categorical encoding.
- **Targets:** Only well-predicted targets are modeled and served: **Protein, Sodium, Calories**.
- **Model:**
  - XGBoost (via MultiOutputRegressor)
  - Hyperparameter tuning with GridSearchCV
  - Realistic, trustworthy performance (test R² ≈ 0.44–0.47)
- **Preprocessing:** All encoders and scalers are saved and reused for inference.

---

## 🚀 Usage

### 1. **FastAPI Inference API**

- **Location:** `api/inference_api.py`
- **How to run:**
  ```bash
  uvicorn api.inference_api:app --reload --port 8001
  ```
- **Endpoints:**
  - `POST /predict` — Accepts raw, human-friendly input (JSON) and returns predictions for Protein, Sodium, and Calories.
- **Preprocessing:** The API automatically encodes, scales, and engineers features to match the training pipeline.
- **Swagger UI:** Visit `http://127.0.0.1:8001/docs` for interactive API docs.

### 2. **Gradio User Interface**

- **Location:** `ui/gradio_app.py`
- **How to run:**
  ```bash
  python ui/gradio_app.py
  ```
- **Features:**
  - User-friendly form for all inputs (dropdowns, number fields, multi-select for diseases)
  - Robust handling of all preprocessing and encoding
  - Returns predictions for Protein, Sodium, and Calories
  - Ready for deployment to Hugging Face Spaces

---

## 🧑‍💻 Development & Data Pipeline

- **Data Preprocessing:**
  - All leaky features removed
  - Feature list saved to `data/feature_columns.json`
  - Preprocessing objects (encoders, scaler) saved to `data/preprocessing_objects.pkl`
- **Model Training:**
  - XGBoost with hyperparameter tuning
  - Only safe features and well-predicted targets
  - Model and metadata saved to `model/`
- **Evaluation:**
  - Only Protein, Sodium, and Calories are evaluated and reported
  - All scripts and logs updated for reproducibility

---

## 🏁 Deployment

- **Local:** Run FastAPI or Gradio as above
- **Cloud:**
  - Push to Hugging Face Spaces for public demo (Gradio recommended)
  - All dependencies listed in `requirements.txt`
- **Artifacts:**
  - Model: `model/nutrition_model.pkl`
  - Metadata: `model/nutrition_model_metadata.json`
  - Preprocessing: `data/preprocessing_objects.pkl`

---

## 🛠️ Troubleshooting

- **Unseen label errors:** Only use allowed values for categorical fields (see UI dropdowns)
- **File not found:** Ensure all model and preprocessing files are in place
- **API/UI errors:** Check logs and ensure dependencies are installed

---

## 📈 Next Steps

- Further feature engineering and data quality improvements
- Explore ensemble or alternative algorithms (e.g., LightGBM)
- Add more user guidance and documentation
- Continue logging all major changes for reproducibility

---

## 🕰️ Legacy Note

This project previously used a RandomForest-based pipeline and predicted 7 targets. The current version is focused, robust, and production-ready. See `project.log` for a full history of changes and lessons learned.

--- 

## 🖼️ Gradio UI Example

Below is a screenshot of the user-friendly Gradio interface for personalized nutrition prediction:

![Gradio UI Screenshot](Screenshot%202025-07-14%20at%202.49.18%E2%80%AFPM.png) 