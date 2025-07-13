#!/usr/bin/env python3
"""
Test script for data preprocessing
"""

from data_preprocessing import NutritionDataPreprocessor

def test_preprocessing():
    """
    Test the preprocessing pipeline
    """
    print("🧪 Testing Data Preprocessing...")
    
    try:
        # Initialize preprocessor
        preprocessor = NutritionDataPreprocessor("detailed_meals_macros_CLEANED.csv")
        
        # Run preprocessing
        df_processed, final_features, target_columns, label_encoders, summary = preprocessor.run_complete_preprocessing(
            scale_features=True,
            add_derived_features=True
        )
        
        print("\n✅ Test passed!")
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Features: {len(final_features)}")
        print(f"Targets: {len(target_columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_preprocessing() 