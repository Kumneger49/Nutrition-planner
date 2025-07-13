#!/usr/bin/env python3
"""
Test script to verify disease features are properly included
"""

from data_preprocessing import NutritionDataPreprocessor

def test_disease_features():
    """
    Test that disease features are properly included in preprocessing
    """
    print("🧪 Testing Disease Features in Preprocessing...")
    
    try:
        # Initialize preprocessor
        preprocessor = NutritionDataPreprocessor("detailed_meals_macros_CLEANED.csv")
        
        # Test 1: Basic preprocessing with disease features
        print("\n📋 Test 1: Basic preprocessing with disease features")
        df_processed, final_features, target_columns, label_encoders, summary = preprocessor.run_complete_preprocessing(
            scale_features=True,
            add_derived_features=True,
            use_one_hot_diseases=False
        )
        
        print("✅ Test 1 passed!")
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Total features: {len(final_features)}")
        print(f"Targets: {len(target_columns)}")
        
        # Check for disease-related features
        disease_features = [f for f in final_features if 'disease' in f.lower() or f.startswith('Has_')]
        print(f"\n🏥 Disease-related features found: {len(disease_features)}")
        for feature in disease_features:
            print(f"  - {feature}")
        
        # Check for disease column in original features
        if 'Disease_encoded' in final_features:
            print("\n✅ Disease column properly encoded!")
        else:
            print("\n❌ Disease column missing from features!")
        
        # Check for disease binary flags
        disease_flags = [f for f in final_features if f.startswith('Has_')]
        print(f"\n🏷️ Disease binary flags: {len(disease_flags)}")
        for flag in disease_flags:
            print(f"  - {flag}")
        
        # Test 2: One-hot encoding for diseases
        print("\n📋 Test 2: One-hot encoding for diseases")
        preprocessor2 = NutritionDataPreprocessor("detailed_meals_macros_CLEANED.csv")
        df_processed2, final_features2, target_columns2, label_encoders2, summary2 = preprocessor2.run_complete_preprocessing(
            scale_features=False,
            add_derived_features=True,
            use_one_hot_diseases=True
        )
        
        # Check for one-hot disease features
        one_hot_diseases = [f for f in final_features2 if f.startswith('Disease_') and not f.endswith('_encoded')]
        print(f"\n🏷️ One-hot disease features: {len(one_hot_diseases)}")
        for feature in one_hot_diseases:
            print(f"  - {feature}")
        
        print("\n✅ Test 2 passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_disease_features() 