import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

def load_data(filepath):
    """Load the customer churn raw dataset."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Handle missing values, clean TotalCharges, and drop customerID."""
    df = df.copy()
    
    # 1. Handle spaces in TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with the median
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    # 2. Drop customerID column
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    elif 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    return df

def engineer_features(df):
    """Create tenure_group and charges_per_month features."""
    df = df.copy()
    
    # 1. Bin tenure into groups
    # Bins: 0-12, 13-24, 25-48, 49-60, 61-72
    bins = [-1, 12, 24, 48, 60, 72]
    labels = ['0-12', '13-24', '25-48', '49-60', '61-72']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    
    # 2. Create charges_per_month = TotalCharges / tenure
    # Handle tenure = 0 division by zero by setting charges_per_month to MonthlyCharges
    df['charges_per_month'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    return df

def encode_features(df):
    """Encode binary variables and one-hot encode multi-class variables."""
    df = df.copy()
    
    # Binary variables (Yes/No, Male/Female)
    binary_cols = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'Churn': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    # Other categorical variables for One-Hot Encoding
    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod', 'tenure_group'
    ]
    
    # Perform One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert dummy columns (boolean) to integer (0/1)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    return df

def preprocess_pipeline(raw_filepath, output_dir='data/processed', models_dir='models'):
    """Full preprocessing pipeline from raw data to balanced train/test sets."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load and clean
    df = load_data(raw_filepath)
    df = clean_data(df)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Encoding
    df = encode_features(df)
    
    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 4. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Feature Scaling (StandardScaler)
    # We fit scaling only on numerical features: tenure, MonthlyCharges, TotalCharges, charges_per_month
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
    
    scaler = StandardScaler()
    
    # Fit scaler on training set and transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Save the fitted scaler for Stage 4 Streamlit app
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    # Save the feature columns order for matching columns during inference
    feature_columns = list(X.columns)
    with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_columns, f)
        
    # Save test set (always evaluate models on the original scaled, but imbalanced test set!)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save original scaled train set
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_original.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train_original.csv'), index=False)
    
    # 6. Handle Class Imbalance
    # Setup SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Setup SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_scaled, y_train)
    
    # 7. Save balanced datasets
    X_train_smote.to_csv(os.path.join(output_dir, 'X_train_smote.csv'), index=False)
    y_train_smote.to_csv(os.path.join(output_dir, 'y_train_smote.csv'), index=False)
    
    X_train_smoteenn.to_csv(os.path.join(output_dir, 'X_train_smoteenn.csv'), index=False)
    y_train_smoteenn.to_csv(os.path.join(output_dir, 'y_train_smoteenn.csv'), index=False)
    
    print("Preprocessed files shapes:")
    print(f"- Original Train Set: {X_train_scaled.shape}, target churn: {y_train.value_counts().to_dict()}")
    print(f"- SMOTE Train Set: {X_train_smote.shape}, target churn: {y_train_smote.value_counts().to_dict()}")
    print(f"- SMOTEENN Train Set: {X_train_smoteenn.shape}, target churn: {y_train_smoteenn.value_counts().to_dict()}")
    print(f"- Test Set: {X_test_scaled.shape}, target churn: {y_test.value_counts().to_dict()}")
    print("Preprocessing completed successfully!")

if __name__ == '__main__':
    preprocess_pipeline('data/raw/telco_churn.csv')
