# 🍎 Nutrition Prediction Model

A machine learning project that predicts personalized nutritional needs for individuals based on their profile using multi-output regression.

## 🎯 Project Goal

Train a multi-output regression model that predicts the following nutrition targets:
- Daily Calorie Target
- Protein
- Sugar
- Sodium
- Calories
- Carbohydrates
- Fiber

## 🏗️ Model Architecture

- **Base Model**: `RandomForestRegressor`
- **Wrapper**: `MultiOutputRegressor` from scikit-learn
- **Features**: Age, Gender, Height, Weight, Activity Level, Dietary Preference
- **Targets**: 7 nutritional metrics

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## 🚀 Setup & Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure your dataset is in the correct location**:
   ```
   data/detailed_meals_macros_CLEANED.csv
   ```

## 🎮 Usage

### Run the complete pipeline:
```bash
python nutrition_prediction_model.py
```

### Expected Output:
```
🍎 Nutrition Prediction Model
==================================================
📊 Loading dataset...
Dataset shape: (1700, 35)
Columns: ['Ages', 'Gender', 'Height', 'Weight', ...]

🔧 Preprocessing data...
Dropping rows with missing target values...
Dropped 0 rows with missing target values
Handling missing values in feature columns...
Encoding categorical features...
Encoded Gender: {'Female': 0, 'Male': 1}
Encoded Activity Level: {'Lightly Active': 0, 'Moderately Active': 1, 'Sedentary': 2, 'Very Active': 3}
Encoded Dietary Preference: {'Omnivore': 0, 'Vegan': 1, 'Vegetarian': 2}

📋 Preparing train/test split...
Feature matrix shape: (1700, 6)
Target matrix shape: (1700, 7)
Training set: 1360 samples
Test set: 340 samples

🤖 Training MultiOutputRegressor with RandomForestRegressor...
✅ Model training completed!

📈 Evaluating model performance...
Daily Calorie Target: R² = 0.XXXX
Protein: R² = 0.XXXX
Sugar: R² = 0.XXXX
Sodium: R² = 0.XXXX
Calories: R² = 0.XXXX
Carbohydrates: R² = 0.XXXX
Fiber: R² = 0.XXXX

Overall R² Score: 0.XXXX

💾 Saving processed data to processed_nutrition_data.csv...
✅ Processed data saved successfully!

🎉 Pipeline completed successfully!
Model trained on 1360 samples
Model tested on 340 samples
Overall model performance: R² = 0.XXXX

✅ Model ready for use!
```

## 📊 Data Processing Pipeline

The script performs the following steps:

1. **Data Loading**: Loads the CSV dataset
2. **Data Exploration**: Shows dataset shape, columns, data types, and missing values
3. **Data Cleaning**: 
   - Drops rows with missing target values
   - Handles missing values in features (median for numerical, mode for categorical)
4. **Feature Engineering**:
   - Encodes categorical variables using LabelEncoder
   - Creates final feature set
5. **Train/Test Split**: 80% training, 20% testing
6. **Model Training**: MultiOutputRegressor with RandomForestRegressor
7. **Model Evaluation**: R² scores for each target variable
8. **Data Export**: Saves processed dataset

## 🔧 Model Configuration

The RandomForestRegressor is configured with:
- `n_estimators=100`: Number of trees
- `max_depth=10`: Maximum tree depth
- `min_samples_split=5`: Minimum samples to split
- `min_samples_leaf=2`: Minimum samples per leaf
- `random_state=42`: For reproducibility
- `n_jobs=-1`: Use all CPU cores

## 📁 Output Files

- `processed_nutrition_data.csv`: Cleaned and preprocessed dataset
- Console output with detailed performance metrics

## 🎯 Model Performance

The model provides:
- Individual R² scores for each nutritional target
- Overall R² score for the complete model
- Detailed training and testing statistics

## 🔍 Troubleshooting

**Common Issues:**
1. **File not found**: Ensure `detailed_meals_macros_CLEANED.csv` is in the `data/` folder
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Memory issues**: Reduce `n_estimators` in the RandomForestRegressor

## 📈 Next Steps

Potential improvements:
- Feature scaling/normalization
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Model persistence (save/load trained model)
- Real-time prediction API
- Feature importance analysis 