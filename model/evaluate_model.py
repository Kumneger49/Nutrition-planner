#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Nutrition Prediction
Analyzes model performance and suitability for personalized nutrition recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NutritionModelEvaluator:
    """
    Comprehensive evaluator for nutrition prediction model
    """
    
    def __init__(self, model_path: str = "model/nutrition_model.pkl", 
                 metadata_path: str = "model/nutrition_model_metadata.json",
                 test_data_path: str = "data/test_nutrition_data.csv"):
        """
        Initialize the evaluator
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.test_data_path = test_data_path
        
        # Load model and metadata
        self.model = None
        self.metadata = None
        self.test_data = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """
        Load the trained model and test data
        """
        print("📊 Loading model and test data...")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load test data
        self.test_data = pd.read_csv(self.test_data_path)
        
        # Prepare features and targets
        feature_names = self.metadata['feature_names']
        target_names = self.metadata['target_names']
        
        self.X_test = self.test_data[feature_names]
        self.y_test = self.test_data[target_names]
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        print(f"✅ Model loaded: {self.metadata['model_type']}")
        print(f"✅ Test data shape: {self.X_test.shape}")
        print(f"✅ Predictions shape: {self.y_pred.shape}")
    
    def analyze_overall_performance(self) -> Dict[str, float]:
        """
        Analyze overall model performance
        """
        print("\n🎯 Overall Model Performance Analysis")
        print("=" * 60)
        
        # Calculate overall metrics
        overall_r2 = r2_score(self.y_test, self.y_pred)
        overall_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        overall_mae = mean_absolute_error(self.y_test, self.y_pred)
        
        print(f"Overall R² Score: {overall_r2:.3f}")
        print(f"Overall RMSE: {overall_rmse:.3f}")
        print(f"Overall MAE: {overall_mae:.3f}")
        
        # Performance interpretation
        if overall_r2 >= 0.8:
            performance_level = "Excellent"
        elif overall_r2 >= 0.6:
            performance_level = "Good"
        elif overall_r2 >= 0.4:
            performance_level = "Moderate"
        elif overall_r2 >= 0.2:
            performance_level = "Poor"
        else:
            performance_level = "Very Poor"
        
        print(f"Performance Level: {performance_level}")
        
        return {
            'r2': overall_r2,
            'rmse': overall_rmse,
            'mae': overall_mae,
            'performance_level': performance_level
        }
    
    def analyze_target_performance(self) -> pd.DataFrame:
        """
        Analyze performance for each target variable
        """
        print("\n📊 Individual Target Performance Analysis")
        print("=" * 80)
        
        results = []
        target_names = self.metadata['target_names']
        
        for i, target in enumerate(target_names):
            y_true = self.y_test.iloc[:, i]
            y_pred = self.y_pred[:, i]
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate percentage error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Performance interpretation
            if r2 >= 0.8:
                performance = "Excellent"
            elif r2 >= 0.6:
                performance = "Good"
            elif r2 >= 0.4:
                performance = "Moderate"
            elif r2 >= 0.2:
                performance = "Poor"
            else:
                performance = "Very Poor"
            
            results.append({
                'Target': target,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE (%)': mape,
                'Performance': performance
            })
        
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False, float_format='%.3f'))
        
        return df_results
    
    def analyze_prediction_errors(self):
        """
        Analyze prediction errors and their distribution
        """
        print("\n🔍 Prediction Error Analysis")
        print("=" * 50)
        
        target_names = self.metadata['target_names']
        
        # Calculate errors for each target
        errors = {}
        for i, target in enumerate(target_names):
            y_true = self.y_test.iloc[:, i]
            y_pred = self.y_pred[:, i]
            errors[target] = y_true - y_pred
        
        # Create error analysis plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, target in enumerate(target_names):
            if i < 7:  # We have 7 targets
                ax = axes[i]
                target_errors = errors[target]
                
                # Plot error distribution
                ax.hist(target_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', alpha=0.8)
                ax.set_title(f'{target}\nError Distribution')
                ax.set_xlabel('Prediction Error')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_error = np.mean(target_errors)
                std_error = np.std(target_errors)
                ax.text(0.05, 0.95, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide the last subplot if we have 7 targets
        if len(target_names) == 7:
            axes[7].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('model/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print error statistics
        print("\nError Statistics Summary:")
        print("-" * 40)
        for target in target_names:
            target_errors = errors[target]
            print(f"{target}:")
            print(f"  Mean Error: {np.mean(target_errors):.2f}")
            print(f"  Std Error: {np.std(target_errors):.2f}")
            print(f"  Min Error: {np.min(target_errors):.2f}")
            print(f"  Max Error: {np.max(target_errors):.2f}")
            print()
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance
        """
        print("\n🏆 Feature Importance Analysis")
        print("=" * 50)
        
        # Get feature importance from each estimator
        feature_names = self.metadata['feature_names']
        target_names = self.metadata['target_names']
        
        importance_df = pd.DataFrame(index=feature_names)
        
        for i, target in enumerate(target_names):
            estimator = self.model.estimators_[i]
            importance_df[target] = estimator.feature_importances_
        
        # Calculate average importance
        importance_df['Average'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Average', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['Average'])
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Average Feature Importance')
        plt.title('Top 15 Most Important Features (Average across all targets)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('model/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, (feature, importance) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {feature:<30} {importance['Average']:.4f}")
        
        return importance_df
    
    def analyze_model_suitability(self) -> Dict[str, str]:
        """
        Analyze if the model is suitable for personalized nutrition recommendations
        """
        print("\n🎯 Model Suitability Analysis for Nutrition Recommendations")
        print("=" * 70)
        
        # Get performance metrics
        target_performance = self.analyze_target_performance()
        
        # Define acceptability thresholds for nutrition recommendations
        acceptability_thresholds = {
            'Daily Calorie Target': {'r2': 0.5, 'mape': 15.0},
            'Protein': {'r2': 0.5, 'mape': 20.0},
            'Sugar': {'r2': 0.3, 'mape': 25.0},
            'Sodium': {'r2': 0.5, 'mape': 20.0},
            'Calories': {'r2': 0.5, 'mape': 15.0},
            'Carbohydrates': {'r2': 0.3, 'mape': 25.0},
            'Fiber': {'r2': 0.3, 'mape': 30.0}
        }
        
        suitability_analysis = {}
        overall_suitable = True
        
        print("\nSuitability Assessment by Target:")
        print("-" * 50)
        
        for _, row in target_performance.iterrows():
            target = row['Target']
            r2 = row['R²']
            mape = row['MAPE (%)']
            
            threshold = acceptability_thresholds.get(target, {'r2': 0.4, 'mape': 25.0})
            
            r2_acceptable = r2 >= threshold['r2']
            mape_acceptable = mape <= threshold['mape']
            target_suitable = r2_acceptable and mape_acceptable
            
            if not target_suitable:
                overall_suitable = False
            
            status = "✅ Suitable" if target_suitable else "❌ Not Suitable"
            
            print(f"{target}:")
            print(f"  R²: {r2:.3f} (threshold: {threshold['r2']:.1f}) {'✅' if r2_acceptable else '❌'}")
            print(f"  MAPE: {mape:.1f}% (threshold: {threshold['mape']:.1f}%) {'✅' if mape_acceptable else '❌'}")
            print(f"  Status: {status}")
            print()
            
            suitability_analysis[target] = {
                'r2': r2,
                'mape': mape,
                'r2_acceptable': r2_acceptable,
                'mape_acceptable': mape_acceptable,
                'suitable': target_suitable
            }
        
        # Overall assessment
        print("Overall Assessment:")
        print("-" * 20)
        if overall_suitable:
            print("✅ Model is SUITABLE for personalized nutrition recommendations")
            print("   - All targets meet acceptability thresholds")
            print("   - Predictions are accurate enough for practical use")
        else:
            print("❌ Model needs IMPROVEMENT before practical use")
            print("   - Some targets don't meet acceptability thresholds")
            print("   - Consider hyperparameter tuning or feature engineering")
        
        return {
            'overall_suitable': overall_suitable,
            'target_analysis': suitability_analysis
        }
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations for model improvement
        """
        print("\n💡 Model Improvement Recommendations")
        print("=" * 50)
        
        recommendations = []
        
        # Analyze current performance
        target_performance = self.analyze_target_performance()
        overall_r2 = target_performance['R²'].mean()
        
        if overall_r2 < 0.5:
            recommendations.append("🔧 **High Priority**: Model shows significant overfitting (train R² >> test R²)")
            recommendations.append("   - Implement cross-validation for hyperparameter tuning")
            recommendations.append("   - Consider regularization techniques")
            recommendations.append("   - Reduce model complexity")
        
        if overall_r2 < 0.6:
            recommendations.append("📊 **Medium Priority**: Overall performance is moderate")
            recommendations.append("   - Collect more training data if possible")
            recommendations.append("   - Engineer additional relevant features")
            recommendations.append("   - Try different algorithms (XGBoost, LightGBM)")
        
        # Check for specific problematic targets
        poor_performers = target_performance[target_performance['R²'] < 0.4]
        if not poor_performers.empty:
            recommendations.append("🎯 **Target-Specific Issues**:")
            for _, row in poor_performers.iterrows():
                recommendations.append(f"   - {row['Target']}: R² = {row['R²']:.3f} (needs improvement)")
        
        # Data quality recommendations
        recommendations.append("📈 **Data Quality**:")
        recommendations.append("   - Verify data quality and consistency")
        recommendations.append("   - Check for outliers that might affect training")
        recommendations.append("   - Ensure proper feature scaling")
        
        # Model deployment recommendations
        recommendations.append("🚀 **Deployment Considerations**:")
        recommendations.append("   - Implement confidence intervals for predictions")
        recommendations.append("   - Add model monitoring for drift detection")
        recommendations.append("   - Consider ensemble methods for better robustness")
        
        # Print recommendations
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def run_complete_evaluation(self):
        """
        Run the complete evaluation pipeline
        """
        print("🔍 Comprehensive Nutrition Model Evaluation")
        print("=" * 60)
        
        # 1. Overall performance
        overall_perf = self.analyze_overall_performance()
        
        # 2. Target-specific performance
        target_perf = self.analyze_target_performance()
        
        # 3. Error analysis
        self.analyze_prediction_errors()
        
        # 4. Feature importance
        importance_df = self.analyze_feature_importance()
        
        # 5. Suitability analysis
        suitability = self.analyze_model_suitability()
        
        # 6. Recommendations
        recommendations = self.generate_recommendations()
        
        # Summary
        print("\n📋 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"Overall R² Score: {overall_perf['r2']:.3f}")
        print(f"Performance Level: {overall_perf['performance_level']}")
        print(f"Suitable for Production: {'✅ Yes' if suitability['overall_suitable'] else '❌ No'}")
        print(f"Recommendations Generated: {len(recommendations)}")
        
        return {
            'overall_performance': overall_perf,
            'target_performance': target_perf,
            'suitability': suitability,
            'recommendations': recommendations
        }

def main():
    """
    Main function to run the evaluation
    """
    evaluator = NutritionModelEvaluator()
    results = evaluator.run_complete_evaluation()
    
    return results

if __name__ == "__main__":
    results = main() 