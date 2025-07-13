#!/usr/bin/env python3
"""
Data Preprocessing for Nutrition Prediction Model
Handles data cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pickle

class NutritionDataPreprocessor:
    """
    Comprehensive data preprocessor for nutrition prediction dataset
    """
    
    def __init__(self, file_path):
        """
        Initialize the preprocessor with the dataset path
        """
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.target_columns = None
        
    def load_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.file_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def validate_columns(self):
        """
        Validate that all required columns are present in the dataset
        """
        print("\n🔍 Validating required columns...")
        
        # Check if dataframe is loaded
        if self.df is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Define required columns
        required_columns = self.target_columns + self.feature_columns
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(self.df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ All required columns present: {len(required_columns)} columns")
        print(f"Target columns: {self.target_columns}")
        print(f"Feature columns: {self.feature_columns}")
        
        return True
    
    def define_columns(self):
        """
        Define target and feature columns
        """
        # Target columns (nutritional values to predict)
        self.target_columns = [
            'Daily Calorie Target',
            'Protein', 
            'Sugar',
            'Sodium',
            'Calories',
            'Carbohydrates',
            'Fiber'
        ]
        
        # Feature columns (input variables)
        self.feature_columns = [
            'Ages',
            'Gender', 
            'Height',
            'Weight',
            'Activity Level',
            'Dietary Preference',
            'Disease'  # Added disease column as a key feature
        ]
        
        # Categorical features that need encoding
        self.categorical_features = [
            'Gender',
            'Activity Level', 
            'Dietary Preference',
            'Disease'  # Disease is categorical and needs encoding
        ]
        
        # Numerical features
        self.numerical_features = [
            'Ages',
            'Height', 
            'Weight'
        ]
        
        print(f"Target columns: {self.target_columns}")
        print(f"Feature columns: {self.feature_columns}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numerical features: {self.numerical_features}")
        
        return self.target_columns, self.feature_columns
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("\n🔧 Handling missing values...")
        
        # Create a copy to avoid modifying original data
        self.df_processed = self.df.copy()
        
        # 1. Drop rows with missing target values
        initial_rows = len(self.df_processed)
        self.df_processed = self.df_processed.dropna(subset=self.target_columns)
        dropped_rows = initial_rows - len(self.df_processed)
        print(f"Dropped {dropped_rows} rows with missing target values")
        
        # 2. Handle missing values in feature columns
        # For numerical features, use median imputation
        numerical_features = ['Ages', 'Height', 'Weight']
        for col in numerical_features:
            if col in self.df_processed.columns and self.df_processed[col].isnull().sum() > 0:
                median_val = self.df_processed[col].median()
                self.df_processed[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # For categorical features, use mode imputation
        categorical_features = ['Gender', 'Activity Level', 'Dietary Preference']
        for col in categorical_features:
            if col in self.df_processed.columns and self.df_processed[col].isnull().sum() > 0:
                mode_val = self.df_processed[col].mode()[0]
                self.df_processed[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        print(f"Final dataset shape after handling missing values: {self.df_processed.shape}")
        
        return self.df_processed
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets BEFORE any encoding/scaling
        """
        print(f"\n✂️ Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
        
        # Get features and targets
        X = self.df_processed[self.feature_columns].copy()
        y = self.df_processed[self.target_columns].copy()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create train and test dataframes
        self.train_data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
        
        print(f"✅ Data split completed:")
        print(f"   - Training set: {self.train_data.shape}")
        print(f"   - Test set: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def encode_categorical_features_train(self):
        """
        Encode categorical features using ONLY training data to prevent data leakage
        """
        print("\n🔤 Encoding categorical features (training data only)...")
        
        for col in self.categorical_features:
            if col in self.train_data.columns:
                le = LabelEncoder()
                # Fit encoder on training data only
                self.train_data[f'{col}_encoded'] = le.fit_transform(self.train_data[col].astype(str))
                # Transform test data using same encoder
                self.test_data[f'{col}_encoded'] = le.transform(self.test_data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        print("Categorical features encoded successfully!")
    
    def scale_features_train(self, scale_features=True):
        """
        Scale numerical features using ONLY training data statistics
        """
        if not scale_features:
            return
        
        print("\n⚖️ Scaling numerical features (training data only)...")
        
        # Get numerical features that need scaling
        numerical_features = ['Ages', 'Height', 'Weight', 'BMI', 'Disease_Count']
        numerical_features = [col for col in numerical_features if col in self.train_data.columns]
        
        if numerical_features:
            # Create scaler if not already created
            if self.scaler is None:
                self.scaler = StandardScaler()
            
            # Create scaled versions using training data statistics
            for col in numerical_features:
                if col in self.train_data.columns:
                    scaled_col_name = f'{col}_scaled'
                    
                    # Fit scaler on training data only
                    train_scaled = self.scaler.fit_transform(
                        self.train_data[col].values.reshape(-1, 1)
                    ).flatten()
                    self.train_data[scaled_col_name] = train_scaled
                    
                    # Transform test data using same scaler
                    test_scaled = self.scaler.transform(
                        self.test_data[col].values.reshape(-1, 1)
                    ).flatten()
                    self.test_data[scaled_col_name] = test_scaled
                    
                    print(f"Scaled {col} -> {scaled_col_name}")
            
            print("Features scaled successfully!")
        else:
            print("No numerical features to scale.")
    
    def add_derived_features_split(self):
        """
        Add derived features to both train and test sets
        """
        print("\n🔧 Adding derived features to split datasets...")
        
        for dataset in [self.train_data, self.test_data]:
            # BMI (Body Mass Index)
            if 'Height' in dataset.columns and 'Weight' in dataset.columns:
                height_m = dataset['Height'] / 100
                dataset['BMI'] = dataset['Weight'] / (height_m ** 2)

            # Age groups
            if 'Ages' in dataset.columns:
                dataset['Age_Group'] = pd.cut(
                    dataset['Ages'], 
                    bins=[0, 25, 35, 50, 65, 100], 
                    labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elderly']
                )

            # Weight categories
            if 'Weight' in dataset.columns:
                dataset['Weight_Category'] = pd.cut(
                    dataset['Weight'],
                    bins=[0, 60, 80, 100, 200],
                    labels=['Light', 'Normal', 'Heavy', 'Very_Heavy']
                )

            # Disease-specific features
            if 'Disease' in dataset.columns:
                diseases = ['Weight Gain', 'Hypertension', 'Heart Disease', 'Kidney Disease', 'Diabetes', 'Acne']
                for disease in diseases:
                    dataset[f'Has_{disease.replace(" ", "_")}'] = dataset['Disease'].str.contains(disease, case=False, na=False).astype(int)
                # Count total number of diseases per person
                dataset['Disease_Count'] = dataset['Disease'].str.count(',') + 1
                # Disease severity
                dataset['Disease_Severity'] = pd.cut(
                    dataset['Disease_Count'],
                    bins=[0, 1, 2, 3, 10],
                    labels=['None', 'Mild', 'Moderate', 'Severe']
                )
                # Disease group features
                dataset['has_metabolic_disorder'] = dataset['Disease'].str.contains('diabetes|hypertension', case=False, na=False).astype(int)
                dataset['has_cardiac_risk'] = dataset['Disease'].str.contains('heart disease', case=False, na=False).astype(int)

            # --- NEW ENGINEERED FEATURES ---
            # Nutrient ratios (handle division by zero)
            def safe_div(n, d):
                return np.where((d == 0) | (pd.isnull(d)), np.nan, n / d)

            if 'Protein' in dataset.columns and 'Calories' in dataset.columns:
                dataset['protein_to_calories'] = safe_div(dataset['Protein'], dataset['Calories'])
            if 'Carbohydrates' in dataset.columns and 'Calories' in dataset.columns:
                dataset['carbs_to_calories'] = safe_div(dataset['Carbohydrates'], dataset['Calories'])
            if 'Fat' in dataset.columns and 'Calories' in dataset.columns:
                dataset['fat_to_calories'] = safe_div(dataset['Fat'], dataset['Calories'])
            if 'Sugar' in dataset.columns and 'Fiber' in dataset.columns:
                dataset['sugar_to_fiber'] = safe_div(dataset['Sugar'], dataset['Fiber'])

            # Normalize nutrients by body size
            if 'Protein' in dataset.columns and 'Weight' in dataset.columns:
                dataset['protein_per_kg'] = safe_div(dataset['Protein'], dataset['Weight'])
            if 'Daily Calorie Target' in dataset.columns and 'Weight' in dataset.columns:
                dataset['calories_per_kg'] = safe_div(dataset['Daily Calorie Target'], dataset['Weight'])

            # Interaction feature
            if 'Activity Level' in dataset.columns and 'Disease_Severity' in dataset.columns:
                dataset['activity_goal_combo'] = dataset['Activity Level'].astype(str) + '_' + dataset['Disease_Severity'].astype(str)

        print("Derived features added to both datasets!")
    
    def encode_derived_categoricals_split(self):
        """
        Encode derived categorical features using training data only
        """
        print("\n🔤 Encoding derived categorical features (training data only)...")
        
        derived_categoricals = ['Age_Group', 'Weight_Category', 'Disease_Severity', 'activity_goal_combo']
        
        for col in derived_categoricals:
            if col in self.train_data.columns:
                le = LabelEncoder()
                # Fit encoder on training data only
                self.train_data[f'{col}_encoded'] = le.fit_transform(self.train_data[col].astype(str))
                # Transform test data using same encoder
                self.test_data[f'{col}_encoded'] = le.transform(self.test_data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        print("Derived categorical features encoded successfully!")
    
    def one_hot_encode_diseases(self, use_one_hot=False):
        """
        Optional one-hot encoding for diseases (useful for multi-label classification)
        """
        if not use_one_hot or 'Disease' not in self.df_processed.columns:
            return
        
        print("\n🏷️ Creating one-hot encoded disease features...")
        
        # Get all unique diseases from the dataset
        all_diseases = set()
        for diseases in self.df_processed['Disease'].dropna():
            disease_list = [d.strip() for d in diseases.split(',')]
            all_diseases.update(disease_list)
        
        # Create one-hot encoded columns
        for disease in sorted(all_diseases):
            col_name = f'Disease_{disease.replace(" ", "_")}'
            self.df_processed[col_name] = self.df_processed['Disease'].str.contains(
                disease, case=False, na=False
            ).astype(int)
            print(f"Created {col_name}")
        
        print(f"✅ Created {len(all_diseases)} one-hot disease features")
        
        return list(all_diseases)
    
    def get_final_feature_set(self, dataset='train'):
        """
        Get the final set of features for modeling
        """
        print(f"\n📋 Creating final feature set for {dataset} dataset...")
        
        # Choose which dataset to use
        if dataset == 'train':
            data = self.train_data
        elif dataset == 'test':
            data = self.test_data
        else:
            raise ValueError("dataset must be 'train' or 'test'")
        
        # Start with scaled numerical features (if available) or original numerical features
        final_features = []
        numerical_features = ['Ages', 'Height', 'Weight']
        for col in numerical_features:
            scaled_col = f'{col}_scaled'
            if scaled_col in data.columns:
                final_features.append(scaled_col)
            elif col in data.columns:
                final_features.append(col)
        
        # Add encoded categorical features (PRIORITY: use encoded versions)
        for col in self.categorical_features:
            encoded_col = f'{col}_encoded'
            if encoded_col in data.columns:
                final_features.append(encoded_col)
                print(f"✅ Using encoded feature: {encoded_col}")
            else:
                print(f"⚠️  Warning: No encoded version found for {col}")
        
        # Add derived numerical features
        derived_numerical = ['BMI', 'Disease_Count']
        for col in derived_numerical:
            scaled_col = f'{col}_scaled'
            if scaled_col in data.columns:
                final_features.append(scaled_col)
            elif col in data.columns:
                final_features.append(col)
        
        # Add disease binary flags
        disease_flags = ['Has_Weight_Gain', 'Has_Hypertension', 'Has_Heart_Disease', 
                        'Has_Kidney_Disease', 'Has_Diabetes', 'Has_Acne']
        for col in disease_flags:
            if col in data.columns:
                final_features.append(col)
        
        # Add encoded derived categorical features
        derived_categoricals = ['Age_Group_encoded', 'Weight_Category_encoded', 'Disease_Severity_encoded']
        for col in derived_categoricals:
            if col in data.columns:
                final_features.append(col)
        
        # --- NEW ENGINEERED FEATURES ---
        engineered_features = [
            'protein_to_calories',
            'carbs_to_calories',
            'fat_to_calories',
            'sugar_to_fiber',
            'protein_per_kg',
            'calories_per_kg',
            'has_metabolic_disorder',
            'has_cardiac_risk',
            'activity_goal_combo',
        ]
        for col in engineered_features:
            if col in data.columns:
                final_features.append(col)
                print(f"✅ Using engineered feature: {col}")
        
        print(f"\n🎯 Final feature set created with {len(final_features)} features:")
        print(f"   - Numerical features: {len([f for f in final_features if '_scaled' in f or f in ['Ages', 'Height', 'Weight', 'BMI', 'Disease_Count']])}")
        print(f"   - Categorical features: {len([f for f in final_features if '_encoded' in f])}")
        print(f"   - Binary features: {len([f for f in final_features if f.startswith('Has_') or f.startswith('has_')])}")
        print(f"   - Engineered features: {len([f for f in final_features if f in engineered_features])}")
        
        return final_features
    
    def save_processed_data(self, output_path="processed_nutrition_data.csv"):
        """
        Save the processed dataset
        """
        print(f"\n💾 Saving processed data to {output_path}...")
        self.df_processed.to_csv(output_path, index=False)
        print("✅ Processed data saved successfully!")
        
        return output_path
    
    def save_split_datasets(self):
        """
        Save train and test datasets separately
        """
        print("\n💾 Saving split datasets...")
        
        # Save training data
        train_filename = "train_nutrition_data.csv"
        self.train_data.to_csv(train_filename, index=False)
        print(f"✅ Training data saved: {train_filename}")
        
        # Save test data
        test_filename = "test_nutrition_data.csv"
        self.test_data.to_csv(test_filename, index=False)
        print(f"✅ Test data saved: {test_filename}")
        
        # Save encoders and scaler
        preprocessing_objects = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        print("✅ Preprocessing objects saved: preprocessing_objects.pkl")
        
        return train_filename, test_filename
    
    def get_data_summary(self):
        """
        Get a summary of the processed data
        """
        print("\n📊 Processed Data Summary:")
        print(f"Original shape: {self.df.shape}")
        print(f"Processed shape: {self.df_processed.shape}")
        print(f"Target columns: {len(self.target_columns)}")
        print(f"Feature columns: {len(self.get_final_feature_set())}")
        print(f"Label encoders created: {len(self.label_encoders)}")
        
        return {
            'original_shape': self.df.shape,
            'processed_shape': self.df_processed.shape,
            'target_columns': self.target_columns,
            'feature_columns': self.get_final_feature_set(),
            'label_encoders': self.label_encoders
        }
    
    def run_complete_preprocessing(self, scale_features=True, add_derived_features=True, use_one_hot_diseases=False, save_data=True):
        """
        Run the complete preprocessing pipeline
        
        Args:
            scale_features (bool): Whether to scale numerical features
            add_derived_features (bool): Whether to add derived features
            use_one_hot_diseases (bool): Whether to use one-hot encoding for diseases
            save_data (bool): Whether to save the processed data to CSV
        """
        try:
            print("🚀 Starting complete preprocessing pipeline...")
            
            # 1. Load data
            self.load_data()
            
            # 2. Define columns
            self.define_columns()
            
            # 3. Validate columns
            self.validate_columns()
            
            # 4. Analyze dataset balance
            balance_info = self.analyze_dataset_balance()
            
            # 5. Handle missing values
            self.handle_missing_values()
            
            # 6. Split data into train and test
            self.split_data()
            
            # 7. Encode categorical features (training data only)
            self.encode_categorical_features_train()
            
            # 8. Scale numerical features (training data only)
            if scale_features:
                self.scale_features_train(scale_features)
            
            # 9. Add derived features to split datasets
            if add_derived_features:
                self.add_derived_features_split()
                self.encode_derived_categoricals_split()
            
            # 10. Optional one-hot encoding for diseases
            if use_one_hot_diseases:
                self.one_hot_encode_diseases(use_one_hot=True)
            
            # 11. Save split datasets
            train_filename, test_filename = self.save_split_datasets()
            
            # 12. Get summary
            summary = self.get_data_summary()
            
            print("\n✅ Data preprocessing completed!")
            print(f"Processed data shape: {self.df_processed.shape}")
            print(f"Features ready for modeling: {len(self.get_final_feature_set())}")
            print(f"Targets ready for modeling: {len(self.target_columns)}")
            
            return self.train_data, self.test_data, self.label_encoders, summary
            
        except FileNotFoundError:
            print("❌ Error: File not found!")
            print("Please make sure 'detailed_meals_macros_CLEANED.csv' is in the current directory.")
            return None, None, None, None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None, None, None, None

    def analyze_dataset_balance(self):
        """
        Analyze the balance of the dataset across various dimensions
        """
        print("\n📊 Analyzing Dataset Balance...")
        print("=" * 50)
        
        # 1. Target variable distributions
        print("\n🎯 Target Variable Distributions:")
        for target in self.target_columns:
            if target in self.df.columns:
                print(f"\n{target}:")
                print(f"  Mean: {self.df[target].mean():.2f}")
                print(f"  Std: {self.df[target].std():.2f}")
                print(f"  Min: {self.df[target].min():.2f}")
                print(f"  Max: {self.df[target].max():.2f}")
                print(f"  Missing: {self.df[target].isnull().sum()}")
        
        # 2. Categorical feature distributions
        print("\n🔤 Categorical Feature Distributions:")
        for feature in self.categorical_features:
            if feature in self.df.columns:
                print(f"\n{feature}:")
                value_counts = self.df[feature].value_counts()
                print(f"  Total unique values: {len(value_counts)}")
                for value, count in value_counts.items():
                    percentage = (count / len(self.df)) * 100
                    print(f"    {value}: {count} ({percentage:.1f}%)")
        
        # 3. Disease distribution analysis
        if 'Disease' in self.df.columns:
            print("\n🏥 Disease Distribution Analysis:")
            
            # Count individual diseases
            all_diseases = []
            for diseases in self.df['Disease'].dropna():
                disease_list = [d.strip() for d in diseases.split(',')]
                all_diseases.extend(disease_list)
            
            disease_counts = pd.Series(all_diseases).value_counts()
            print(f"  Total disease mentions: {len(all_diseases)}")
            print(f"  Unique diseases: {len(disease_counts)}")
            print("  Disease frequency:")
            for disease, count in disease_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    {disease}: {count} ({percentage:.1f}%)")
            
            # Analyze disease combinations
            disease_combinations = self.df['Disease'].value_counts()
            print(f"\n  Disease combinations (top 10):")
            for combo, count in disease_combinations.head(10).items():
                percentage = (count / len(self.df)) * 100
                print(f"    {combo}: {count} ({percentage:.1f}%)")
        
        # 4. Age and gender distribution
        print("\n👥 Demographics Distribution:")
        if 'Ages' in self.df.columns:
            print(f"  Age distribution:")
            print(f"    Mean age: {self.df['Ages'].mean():.1f}")
            print(f"    Age range: {self.df['Ages'].min()} - {self.df['Ages'].max()}")
            
            # Age groups
            age_groups = pd.cut(self.df['Ages'], bins=[0, 25, 35, 50, 65, 100], 
                               labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elderly'])
            age_group_counts = age_groups.value_counts()
            for group, count in age_group_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    {group}: {count} ({percentage:.1f}%)")
        
        if 'Gender' in self.df.columns:
            print(f"  Gender distribution:")
            gender_counts = self.df['Gender'].value_counts()
            for gender, count in gender_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    {gender}: {count} ({percentage:.1f}%)")
        
        # 5. Activity level and dietary preference
        print("\n🏃 Activity Level Distribution:")
        if 'Activity Level' in self.df.columns:
            activity_counts = self.df['Activity Level'].value_counts()
            for activity, count in activity_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    {activity}: {count} ({percentage:.1f}%)")
        
        print("\n🥗 Dietary Preference Distribution:")
        if 'Dietary Preference' in self.df.columns:
            diet_counts = self.df['Dietary Preference'].value_counts()
            for diet, count in diet_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"    {diet}: {count} ({percentage:.1f}%)")
        
        # 6. Overall dataset balance assessment
        print("\n⚖️ Balance Assessment:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Targets: {len(self.target_columns)}")
        
        # Check for severe imbalances
        imbalances = []
        for feature in self.categorical_features:
            if feature in self.df.columns:
                value_counts = self.df[feature].value_counts()
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = max_count / min_count
                if imbalance_ratio > 10:
                    imbalances.append(f"{feature}: {imbalance_ratio:.1f}x")
        
        if imbalances:
            print(f"  ⚠️  Potential imbalances detected:")
            for imbalance in imbalances:
                print(f"    {imbalance}")
        else:
            print(f"  ✅ Dataset appears reasonably balanced")
        
        print("=" * 50)
        
        return {
            'total_samples': len(self.df),
            'imbalances': imbalances,
            'disease_count': len(disease_counts) if 'Disease' in self.df.columns else 0
        }

def main():
    """
    Main function to run the complete preprocessing pipeline
    """
    print("🍎 Nutrition Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = NutritionDataPreprocessor("./detailed_meals_macros_CLEANED.csv")
    
    # Run complete preprocessing with data saving enabled
    result = preprocessor.run_complete_preprocessing(
        scale_features=True,
        add_derived_features=True,
        use_one_hot_diseases=False,
        save_data=True  # Save the processed data for model training
    )
    
    if result[0] is not None:
        print("\n✅ Data ready for modeling!")
        print("📁 Split datasets saved:")
        print("   - train_nutrition_data.csv")
        print("   - test_nutrition_data.csv")
        print("   - preprocessing_objects.pkl")
        print("🚀 You can now use these files for model training!")
        return result
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.")
        return None

if __name__ == "__main__":
    # Run the preprocessing pipeline
    result = main()
    if result is not None and result[0] is not None:
        print("\n✅ Data ready for modeling!")
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.") 