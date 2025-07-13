#!/usr/bin/env python3
"""
Nutrition Prediction Model
Multi-output regression model to predict personalized nutritional needs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration
    """
    print("📊 Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def identify_target_columns():
    """
    Define the target columns for prediction
    """
    target_columns = [
        'Daily Calorie Target',
        'Protein', 
        'Sugar',
        'Sodium',
        'Calories',
        'Carbohydrates',
        'Fiber'
    ]
    return target_columns

def identify_feature_columns():
    """
    Define the feature columns for prediction
    """
    feature_columns = [
        'Ages',
        'Gender',
        'Height',
        'Weight', 
        'Activity Level',
        'Dietary Preference'
    ]
    return feature_columns

def preprocess_data(df, target_columns, feature_columns):
    """
    Preprocess the data: handle missing values, encode categoricals, etc.
    """
    print("\n🔧 Preprocessing data...")
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # 1. Drop rows with missing target values
    print(f"Dropping rows with missing target values...")
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=target_columns)
    dropped_rows = initial_rows - len(df_processed)
    print(f"Dropped {dropped_rows} rows with missing target values")
    
    # 2. Handle missing values in feature columns
    print(f"Handling missing values in feature columns...")
    
    # For numerical features, use median imputation
    numerical_features = ['Ages', 'Height', 'Weight']
    for col in numerical_features:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    # For categorical features, use mode imputation
    categorical_features = ['Gender', 'Activity Level', 'Dietary Preference']
    for col in categorical_features:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")
    
    # 3. Encode categorical features
    print(f"Encoding categorical features...")
    label_encoders = {}
    
    for col in categorical_features:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 4. Create final feature set
    final_features = numerical_features + [f'{col}_encoded' for col in categorical_features]
    
    print(f"Final feature columns: {final_features}")
    print(f"Target columns: {target_columns}")
    
    return df_processed, final_features, label_encoders

def prepare_train_test_data(df_processed, final_features, target_columns):
    """
    Prepare training and testing data
    """
    print("\n📋 Preparing train/test split...")
    
    # Extract features and targets
    X = df_processed[final_features].values
    y = df_processed[target_columns].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, target_columns):
    """
    Train the multi-output regression model
    """
    print("\n🤖 Training MultiOutputRegressor with RandomForestRegressor...")
    
    # Create the base model
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Wrap in MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("✅ Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test, target_columns):
    """
    Evaluate the model performance
    """
    print("\n📈 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate R² score for each target
    r2_scores = {}
    for i, target in enumerate(target_columns):
        score = r2_score(y_test[:, i], y_pred[:, i])
        r2_scores[target] = score
        print(f"{target}: R² = {score:.4f}")
    
    # Calculate overall R² score
    overall_r2 = r2_score(y_test, y_pred)
    print(f"\nOverall R² Score: {overall_r2:.4f}")
    
    return r2_scores, overall_r2

def save_processed_data(df_processed, output_path="processed_nutrition_data.csv"):
    """
    Save the cleaned and preprocessed dataset
    """
    print(f"\n💾 Saving processed data to {output_path}...")
    df_processed.to_csv(output_path, index=False)
    print("✅ Processed data saved successfully!")

def main():
    """
    Main function to run the complete pipeline
    """
    print("🍎 Nutrition Prediction Model")
    print("=" * 50)
    
    # File path
    file_path = "data/detailed_meals_macros_CLEANED.csv"
    
    try:
        # 1. Load and explore data
        df = load_and_explore_data(file_path)
        
        # 2. Identify target and feature columns
        target_columns = identify_target_columns()
        feature_columns = identify_feature_columns()
        
        # 3. Preprocess data
        df_processed, final_features, label_encoders = preprocess_data(
            df, target_columns, feature_columns
        )
        
        # 4. Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_train_test_data(
            df_processed, final_features, target_columns
        )
        
        # 5. Train model
        model = train_model(X_train, y_train, target_columns)
        
        # 6. Evaluate model
        r2_scores, overall_r2 = evaluate_model(model, X_test, y_test, target_columns)
        
        # 7. Save processed data (optional)
        save_processed_data(df_processed)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"Model trained on {X_train.shape[0]} samples")
        print(f"Model tested on {X_test.shape[0]} samples")
        print(f"Overall model performance: R² = {overall_r2:.4f}")
        
        # Return model and encoders for potential use
        return model, label_encoders, final_features, target_columns
        
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
        print("Please make sure the CSV file is in the correct location.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your data and try again.")
        return None, None, None, None

if __name__ == "__main__":
    # Run the complete pipeline
    result = main()
    if result[0] is not None:
        model, label_encoders, final_features, target_columns = result
        print("\n✅ Model ready for use!")
    else:
        print("\n❌ Model training failed. Please check the errors above.") 