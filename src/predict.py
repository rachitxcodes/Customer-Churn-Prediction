import os
import json
import joblib
import pandas as pd
import numpy as np

def predict_churn(customer_data, models_dir='models'):
    """
    Perform inference on a single customer or batch of customers.
    
    Parameters:
    - customer_data: dict or list of dicts containing customer attributes.
    - models_dir: path to directory containing model artifacts.
    
    Returns:
    - Dict containing predictions and probabilities.
    """
    # 1. Load artifacts
    model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    cols_path = os.path.join(models_dir, 'feature_columns.json')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(cols_path)):
        raise FileNotFoundError("Model artifacts not found. Please run the training pipeline first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(cols_path, 'r') as f:
        feature_columns = json.load(f)
        
    # Convert single dict to list if necessary
    if isinstance(customer_data, dict):
        customer_data = [customer_data]
        
    results = []
    
    for customer in customer_data:
        # 2. Reconstruct features vector
        features = {col: 0 for col in feature_columns}
        
        # Binary variables mapping
        binary_map = {
            'gender': {'Female': 0, 'Male': 1},
            'Partner': {'No': 0, 'Yes': 1},
            'Dependents': {'No': 0, 'Yes': 1},
            'PhoneService': {'No': 0, 'Yes': 1},
            'PaperlessBilling': {'No': 0, 'Yes': 1}
        }
        
        for k, v in binary_map.items():
            if k in customer:
                features[k] = v.get(customer[k], 0)
                
        # SeniorCitizen is already binary numeric (0/1 or Yes/No)
        if 'SeniorCitizen' in customer:
            val = customer['SeniorCitizen']
            features['SeniorCitizen'] = 1 if val == 'Yes' or val == 1 or val == '1' else 0
            
        # Numerical values
        tenure = int(customer.get('tenure', 0))
        monthly_charges = float(customer.get('MonthlyCharges', 0.0))
        total_charges = float(customer.get('TotalCharges', 0.0))
        
        features['tenure'] = tenure
        features['MonthlyCharges'] = monthly_charges
        features['TotalCharges'] = total_charges
        
        # Calculated numerical values
        features['charges_per_month'] = total_charges / tenure if tenure > 0 else monthly_charges
        
        # Categorical One-Hot mapping
        # MultipleLines
        ml = customer.get('MultipleLines', 'No')
        if ml == "No phone service":
            features['MultipleLines_No phone service'] = 1
        elif ml == "Yes":
            features['MultipleLines_Yes'] = 1
            
        # InternetService
        is_type = customer.get('InternetService', 'No')
        if is_type == "Fiber optic":
            features['InternetService_Fiber optic'] = 1
        elif is_type == "No":
            features['InternetService_No'] = 1
            
        # Add-ons
        addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for addon in addons:
            val = customer.get(addon, 'No')
            if val == "No internet service":
                features[f'{addon}_No internet service'] = 1
            elif val == "Yes":
                features[f'{addon}_Yes'] = 1
                
        # Contract
        contract = customer.get('Contract', 'Month-to-month')
        if contract == "One year":
            features['Contract_One year'] = 1
        elif contract == "Two year":
            features['Contract_Two year'] = 1
            
        # PaymentMethod
        pm = customer.get('PaymentMethod', 'Mailed check')
        if pm == "Credit card (automatic)":
            features['PaymentMethod_Credit card (automatic)'] = 1
        elif pm == "Electronic check":
            features['PaymentMethod_Electronic check'] = 1
        elif pm == "Mailed check":
            features['PaymentMethod_Mailed check'] = 1
            
        # tenure_group
        if 13 <= tenure <= 24:
            features['tenure_group_13-24'] = 1
        elif 25 <= tenure <= 48:
            features['tenure_group_25-48'] = 1
        elif 49 <= tenure <= 60:
            features['tenure_group_49-60'] = 1
        elif 61 <= tenure <= 72:
            features['tenure_group_61-72'] = 1
            
        # Convert to DataFrame
        df_inst = pd.DataFrame([features])[feature_columns]
        
        # Scale numerical values
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
        df_inst[num_cols] = scaler.transform(df_inst[num_cols])
        
        # Predict probability
        prob = float(model.predict_proba(df_inst)[0][1])
        prediction = int(model.predict(df_inst)[0])
        
        results.append({
            'churn_risk': 'HIGH' if prob >= 0.50 else 'LOW',
            'churn_probability': prob,
            'prediction': prediction
        })
        
    return results[0] if len(results) == 1 else results

if __name__ == '__main__':
    # Test inference function with dummy customer
    test_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    try:
        pred_res = predict_churn(test_customer)
        print("Inference successful! Result:")
        print(json.dumps(pred_res, indent=2))
    except Exception as e:
        print("Inference error:", e)
