import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end styling
st.markdown("""
<style>
    /* Premium fonts and typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant Title card */
    .title-card {
        background: linear-gradient(135deg, #2A9D8F 0%, #1D6F66 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Prediction output styling */
    .metric-container {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        border-left: 5px solid #2a9d8f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .high-risk {
        background-color: #FFF3CD;
        border-left: 5px solid #DC3545;
    }
    
    .low-risk {
        background-color: #D1E7DD;
        border-left: 5px solid #198754;
    }
    
    .risk-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .risk-text {
        font-size: 16px;
        color: #495057;
    }
    
    /* Section dividers and labels */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #1d6f66;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cache Model, Preprocessors, and SHAP Explainer
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns

@st.cache_resource
def load_shap_explainer(_model):
    # Load background dataset (100 samples is ideal for speed and stability)
    background_df = pd.read_csv('data/processed/X_train_smoteenn.csv').head(100)
    # Instantiate SHAP explainer
    explainer = shap.Explainer(_model, background_df)
    return explainer

try:
    model, scaler, feature_columns = load_artifacts()
    explainer = load_shap_explainer(model)
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts or SHAP background data: {e}. Please ensure modeling ran successfully first.")
    artifacts_loaded = False

# 3. Application Title Card
st.markdown("""
<div class="title-card">
    <h1 style='margin:0; font-weight:800;'>📞 Telecom Customer Churn Predictor</h1>
    <p style='margin:5px 0 0 0; opacity:0.9; font-size:16px;'>
        Leverage machine learning and Explainable AI (SHAP) to diagnose customer retention risks in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

if artifacts_loaded:
    # Sidebar: Grouped User Input fields
    st.sidebar.markdown("### 👤 Demographic Details")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen (Age >= 65)", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has a Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    
    st.sidebar.markdown("### 🧾 Subscription & Billing Details")
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=10.0, max_value=120.0, value=50.0, step=0.5)
    total_charges = st.sidebar.slider("Total Charges ($)", min_value=10.0, max_value=9000.0, value=500.0, step=10.0)
    
    st.sidebar.markdown("### 🔌 Services Subscribed")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    
    # Hide options if no internet service
    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    else:
        online_security = st.sidebar.selectbox("Online Security Add-on", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup Add-on", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection Add-on", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support Add-on", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV Add-on", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies Add-on", ["No", "Yes"])
        
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    # Create main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='section-title'>Customer Profile Summary</div>", unsafe_allow_html=True)
        
        info_df = pd.DataFrame({
            "Attribute": ["Gender", "Senior Citizen", "Partner", "Dependents", "Tenure", "Contract Type", "Internet Service", "Monthly Charges", "Total Charges"],
            "Value": [gender, senior_citizen, partner, dependents, f"{tenure} months", contract, internet_service, f"${monthly_charges}", f"${total_charges}"]
        })
        st.table(info_df)
        
        # Display Model Performance Metrics in a collapsed expander for transparency
        with st.expander("🔬 Model Performance Metadata"):
            st.markdown("""
            **Tuned Classifier**: Logistic Regression (regularized)  
            **Balancing Method**: SMOTE + ENN (Edited Nearest Neighbors)  
            
            * **Sensitivity/Recall (Class 1)**: **87.70%** (Catches ~88% of churners)
            * **ROC-AUC**: **83.26%**
            * **Overall Test Accuracy**: **70.19%**
            * **F1-Score**: **60.97%**
            """)
        
    with col2:
        st.markdown("<div class='section-title'>Diagnostic & Prediction Panel</div>", unsafe_allow_html=True)
        predict_btn = st.button("📊 Run Churn Diagnosis", use_container_width=True)
        
        if predict_btn:
            # 4. Preprocess input to match training format
            # Initialize feature dict with 0s
            input_dict = {col: 0 for col in feature_columns}
            
            # Map binary variables
            input_dict['gender'] = 1 if gender == "Male" else 0
            input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
            input_dict['Partner'] = 1 if partner == "Yes" else 0
            input_dict['Dependents'] = 1 if dependents == "Yes" else 0
            input_dict['PhoneService'] = 1 if phone_service == "Yes" else 0
            input_dict['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
            
            # Numerical columns
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = total_charges
            
            # Calculated numerical columns
            input_dict['charges_per_month'] = total_charges / tenure if tenure > 0 else monthly_charges
            
            # Categorical One-Hot mapping
            # MultipleLines
            if multiple_lines == "No phone service":
                input_dict['MultipleLines_No phone service'] = 1
            elif multiple_lines == "Yes":
                input_dict['MultipleLines_Yes'] = 1
                
            # InternetService
            if internet_service == "Fiber optic":
                input_dict['InternetService_Fiber optic'] = 1
            elif internet_service == "No":
                input_dict['InternetService_No'] = 1
                
            # OnlineSecurity
            if online_security == "No internet service":
                input_dict['OnlineSecurity_No internet service'] = 1
            elif online_security == "Yes":
                input_dict['OnlineSecurity_Yes'] = 1
                
            # OnlineBackup
            if online_backup == "No internet service":
                input_dict['OnlineBackup_No internet service'] = 1
            elif online_backup == "Yes":
                input_dict['OnlineBackup_Yes'] = 1
                
            # DeviceProtection
            if device_protection == "No internet service":
                input_dict['DeviceProtection_No internet service'] = 1
            elif device_protection == "Yes":
                input_dict['DeviceProtection_Yes'] = 1
                
            # TechSupport
            if tech_support == "No internet service":
                input_dict['TechSupport_No internet service'] = 1
            elif tech_support == "Yes":
                input_dict['TechSupport_Yes'] = 1
                
            # StreamingTV
            if streaming_tv == "No internet service":
                input_dict['StreamingTV_No internet service'] = 1
            elif streaming_tv == "Yes":
                input_dict['StreamingTV_Yes'] = 1
                
            # StreamingMovies
            if streaming_movies == "No internet service":
                input_dict['StreamingMovies_No internet service'] = 1
            elif streaming_movies == "Yes":
                input_dict['StreamingMovies_Yes'] = 1
                
            # Contract
            if contract == "One year":
                input_dict['Contract_One year'] = 1
            elif contract == "Two year":
                input_dict['Contract_Two year'] = 1
                
            # PaymentMethod
            if payment_method == "Credit card (automatic)":
                input_dict['PaymentMethod_Credit card (automatic)'] = 1
            elif payment_method == "Electronic check":
                input_dict['PaymentMethod_Electronic check'] = 1
            elif payment_method == "Mailed check":
                input_dict['PaymentMethod_Mailed check'] = 1
                
            # tenure_group
            if 13 <= tenure <= 24:
                input_dict['tenure_group_13-24'] = 1
            elif 25 <= tenure <= 48:
                input_dict['tenure_group_25-48'] = 1
            elif 49 <= tenure <= 60:
                input_dict['tenure_group_49-60'] = 1
            elif 61 <= tenure <= 72:
                input_dict['tenure_group_61-72'] = 1
                
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Ensure correct columns order
            input_df = input_df[feature_columns]
            
            # Fit scaling only on the 4 numerical columns
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            # Get predictions
            prob = model.predict_proba(input_df)[0][1]
            churn_risk = "HIGH" if prob >= 0.50 else "LOW"
            
            # Visual presentation of risk
            if churn_risk == "HIGH":
                st.markdown(f"""
                <div class="metric-container high-risk">
                    <div class="risk-title" style="color: #842029;">⚠️ HIGH CHURN RISK</div>
                    <div class="risk-text">This customer has a <b>{prob * 100:.1f}%</b> probability of canceling subscription.</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
            else:
                st.markdown(f"""
                <div class="metric-container low-risk">
                    <div class="risk-title" style="color: #0f5132;">✅ LOW CHURN RISK</div>
                    <div class="risk-text">This customer has a <b>{prob * 100:.1f}%</b> probability of canceling subscription.</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
                
            # 5. SHAP Explanations UI section
            st.markdown("<div class='section-title'>🔍 Explainable AI (XAI) Attribution</div>", unsafe_allow_html=True)
            st.write("This plot shows the major features pushing the customer towards (red) or away from (blue) churn risk:")
            
            try:
                # Calculate SHAP values for this instance
                shap_values = explainer(input_df)
                
                # Plot waterfall chart
                fig, ax = plt.subplots(figsize=(10, 5))
                # Set a tight bounding layout for neat presentation
                shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                plt.title("Attribution of Prediction Drivers (SHAP log-odds)", fontsize=12, fontweight='bold', pad=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as ex:
                st.warning(f"Could not calculate SHAP plot: {ex}")
                
            # 6. Structured loyalty recommendations based on prediction
            st.markdown("<div class='section-title'>💡 Recommended Loyalty Action Plan</div>", unsafe_allow_html=True)
            if churn_risk == "HIGH":
                st.markdown("""
                * **Offer Contract Upgrades**: The customer is likely on a Month-to-month plan. Offer a discount if they commit to a 1 or 2-year contract.
                * **Price Incentive**: The customer has high monthly charges. Offer a service bundle or loyalty discount (e.g., $10 off monthly invoice for 6 months).
                * **Shift to Auto-Pay**: If they pay via Electronic Check, incentivize moving to automatic Credit Card/Bank payments with a one-time $10 credit.
                * **Service Check-in**: If they are using Fiber Optic, schedule a customer care outreach to resolve potential tech/speed performance issues.
                """)
            else:
                st.markdown("""
                * **Customer is Stable**: No immediate financial retention action required.
                * **Upsell Opportunities**: Since churn risk is low, they might be receptive to add-on security packages or premium streaming subscriptions.
                """)
