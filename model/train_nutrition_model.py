#!/usr/bin/env python3
"""
Nutrition Prediction Model Training Script
Multi-output regression model to predict personalized nutritional needs
"""

# Data manipulation and analysis
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model persistence
import joblib
import pickle

# Utilities
import warnings
import os
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class NutritionModelTrainer:
    """
    Trainer class for nutrition prediction model using Random Forest + MultiOutputRegressor
    """
    
    def __init__(self, train_data_path: str = "data/train_nutrition_data.csv", 
                 test_data_path: str = "data/test_nutrition_data.csv",
                 preprocessing_path: str = "data/preprocessing_objects.pkl"):
        """
        Initialize the trainer with data paths
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.preprocessing_path = preprocessing_path
        
        # Data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        
        # Model containers
        self.model: Optional[MultiOutputRegressor] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        
        # Preprocessing objects
        self.label_encoders: Optional[Dict[str, Any]] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Performance metrics
        self.train_scores: Dict[str, Dict[str, float]] = {}
        self.test_scores: Dict[str, Dict[str, float]] = {}
        self.cv_scores: Dict[str, float] = {}
        
    def load_data(self) -> bool:
        """
        Load the preprocessed training and test datasets
        """
        print("📊 Loading preprocessed datasets...")
        
        try:
            # Load training data
            self.train_data = pd.read_csv(self.train_data_path)
            print(f"✅ Training data loaded: {self.train_data.shape}")
            
            # Load test data
            self.test_data = pd.read_csv(self.test_data_path)
            print(f"✅ Test data loaded: {self.test_data.shape}")
            
            # Load preprocessing objects
            with open(self.preprocessing_path, 'rb') as f:
                preprocessing_objects = pickle.load(f)
            
            self.label_encoders = preprocessing_objects['label_encoders']
            self.scaler = preprocessing_objects['scaler']
            print("✅ Preprocessing objects loaded")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please run the data preprocessing script first!")
            return False
    
    def prepare_features_and_targets(self) -> bool:
        """
        Prepare features and targets for training
        """
        print("\n🔧 Preparing features and targets...")
        
        if self.train_data is None or self.test_data is None:
            print("❌ Data not loaded. Please run load_data() first.")
            return False
        
        # Define target columns
        self.target_names = [
            'Daily Calorie Target',
            'Protein', 
            'Sugar',
            'Sodium',
            'Calories',
            'Carbohydrates',
            'Fiber'
        ]
        
        # Get feature columns (exclude targets and original categorical columns)
        # Original categorical columns that should be excluded
        original_categorical_columns = [
            'Gender', 'Activity Level', 'Dietary Preference', 'Disease',
            'Age_Group', 'Weight_Category', 'Disease_Severity',
            'activity_goal_combo'  # Exclude the string version
        ]
        
        # All columns to exclude
        exclude_columns = self.target_names + original_categorical_columns
        
        # Get feature columns (all columns except targets and original categoricals)
        self.feature_names = [col for col in self.train_data.columns if col not in exclude_columns]
        
        print(f"Target columns: {self.target_names}")
        print(f"Feature columns ({len(self.feature_names)}): {self.feature_names}")
        print(f"Excluded original categorical columns: {original_categorical_columns}")
        
        # Prepare training data
        self.X_train = self.train_data[self.feature_names]
        self.y_train = self.train_data[self.target_names]
        
        # Prepare test data
        self.X_test = self.test_data[self.feature_names]
        self.y_test = self.test_data[self.target_names]
        
        print(f"✅ Training features shape: {self.X_train.shape}")
        print(f"✅ Training targets shape: {self.y_train.shape}")
        print(f"✅ Test features shape: {self.X_test.shape}")
        print(f"✅ Test targets shape: {self.y_test.shape}")
        
        return True
    
    def create_model(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42) -> MultiOutputRegressor:
        """
        Create the Random Forest + MultiOutputRegressor model
        """
        print(f"\n🌲 Creating Random Forest + MultiOutputRegressor model...")
        
        # Create base Random Forest regressor
        base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        # Wrap in MultiOutputRegressor
        self.model = MultiOutputRegressor(base_rf)
        
        print(f"✅ Model created with {n_estimators} estimators")
        return self.model
    
    def train_model(self) -> Optional[MultiOutputRegressor]:
        """
        Train the model on the training data
        """
        print("\n🚀 Training the model...")
        
        if self.X_train is None or self.y_train is None:
            print("❌ Features and targets not prepared. Please run prepare_features_and_targets() first.")
            return None
        
        if self.model is None:
            self.create_model()
        
        # Train the model
        start_time = datetime.now()
        self.model.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Model training completed in {training_time:.2f} seconds")
        return self.model
    
    def evaluate_model(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Evaluate model performance on training and test sets
        """
        print("\n📊 Evaluating model performance...")
        
        if self.model is None or self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            print("❌ Model or data not ready. Please run train_model() first.")
            return {}, {}
        
        if self.target_names is None:
            print("❌ Target names not set. Please run prepare_features_and_targets() first.")
            return {}, {}
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics for each target
        self.train_scores = {}
        self.test_scores = {}
        
        print("\n🎯 Performance Metrics by Target:")
        print("=" * 80)
        print(f"{'Target':<20} {'Train R²':<12} {'Test R²':<12} {'Train RMSE':<12} {'Test RMSE':<12}")
        print("=" * 80)
        
        for i, target in enumerate(self.target_names):
            # Training metrics
            train_r2 = r2_score(self.y_train.iloc[:, i], y_train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(self.y_train.iloc[:, i], y_train_pred[:, i]))
            
            # Test metrics
            test_r2 = r2_score(self.y_test.iloc[:, i], y_test_pred[:, i])
            test_rmse = np.sqrt(mean_squared_error(self.y_test.iloc[:, i], y_test_pred[:, i]))
            
            # Store scores
            self.train_scores[target] = {'r2': train_r2, 'rmse': train_rmse}
            self.test_scores[target] = {'r2': test_r2, 'rmse': test_rmse}
            
            print(f"{target:<20} {train_r2:<12.3f} {test_r2:<12.3f} {train_rmse:<12.3f} {test_rmse:<12.3f}")
        
        # Overall performance
        overall_train_r2 = r2_score(self.y_train, y_train_pred)
        overall_test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("=" * 80)
        print(f"{'OVERALL':<20} {overall_train_r2:<12.3f} {overall_test_r2:<12.3f}")
        print("=" * 80)
        
        return self.train_scores, self.test_scores
    
    def get_feature_importance(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Get feature importance from the Random Forest model
        """
        print("\n🔍 Analyzing feature importance...")
        
        if self.model is None or self.target_names is None or self.feature_names is None:
            print("❌ Model or feature names not ready. Please run train_model() first.")
            return {}, {}
        
        # Get feature importance from each estimator in MultiOutputRegressor
        feature_importance: Dict[str, Dict[str, float]] = {}
        
        for i, target in enumerate(self.target_names):
            estimator = self.model.estimators_[i]
            importance = estimator.feature_importances_
            feature_importance[target] = dict(zip(self.feature_names, importance))
        
        # Calculate average importance across all targets
        avg_importance: Dict[str, float] = {}
        for feature in self.feature_names:
            avg_importance[feature] = np.mean([feature_importance[target][feature] for target in self.target_names])
        
        # Sort by average importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25} {importance:.4f}")
        
        return feature_importance, avg_importance
    
    def save_model(self, model_path: str = "model/nutrition_model.pkl") -> str:
        """
        Save the trained model and metadata
        """
        print(f"\n💾 Saving model to {model_path}...")
        
        if self.model is None or self.feature_names is None or self.target_names is None:
            print("❌ Model not ready. Please run train_model() first.")
            return ""
        
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            print("❌ Data not ready. Please run prepare_features_and_targets() first.")
            return ""
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'train_scores': self.train_scores,
            'test_scores': self.test_scores,
            'model_type': 'RandomForestRegressor + MultiOutputRegressor',
            'training_date': datetime.now().isoformat(),
            'data_shape': {
                'train_features': self.X_train.shape,
                'train_targets': self.y_train.shape,
                'test_features': self.X_test.shape,
                'test_targets': self.y_test.shape
            }
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {metadata_path}")
        
        return model_path
    
    def run_complete_training(self) -> Optional[MultiOutputRegressor]:
        """
        Run the complete training pipeline
        """
        print("🍎 Nutrition Model Training Pipeline")
        print("=" * 50)
        
        try:
            # 1. Load data
            if not self.load_data():
                return None
            
            # 2. Prepare features and targets
            if not self.prepare_features_and_targets():
                return None
            
            # 3. Create and train model
            model = self.train_model()
            if model is None:
                return None
            
            # 4. Evaluate model
            self.evaluate_model()
            
            # 5. Analyze feature importance
            self.get_feature_importance()
            
            # 6. Save model
            model_path = self.save_model()
            if not model_path:
                return None
            
            print("\n✅ Training pipeline completed successfully!")
            print(f"🎯 Model ready for predictions: {model_path}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return None

def main() -> Optional[MultiOutputRegressor]:
    """
    Main function to run the training pipeline
    """
    # Initialize trainer
    trainer = NutritionModelTrainer()
    
    # Run complete training
    model = trainer.run_complete_training()
    
    if model is not None:
        print("\n🚀 Model training completed successfully!")
        print("📁 Model files saved in the model/ directory")
        return model
    else:
        print("\n❌ Model training failed. Please check the errors above.")
        return None

if __name__ == "__main__":
    # Run the training pipeline
    model = main() 