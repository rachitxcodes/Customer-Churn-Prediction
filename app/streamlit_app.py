import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap

# 1. Page Configuration
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Theme adapting to light & dark modes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Top Banner */
    .banner-container {
        background: linear-gradient(135deg, #2A9D8F 0%, #1D6F66 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Elegant Info Cards */
    .dashboard-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 700;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
    }
    
    /* Custom Gauge styles */
    .gauge-wrapper {
        text-align: center;
        padding: 10px 0;
    }
    
    .gauge-value {
        font-size: 56px;
        font-weight: 800;
        margin: 5px 0;
    }
    
    .gauge-bar-bg {
        background-color: rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin: 12px 0;
    }
    
    .gauge-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease-in-out;
    }
    
    .risk-badge {
        font-size: 15px;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 30px;
        display: inline-block;
        margin-top: 5px;
    }
    
    .risk-badge.high {
        background-color: rgba(220, 53, 69, 0.15);
        color: #EA4335;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .risk-badge.low {
        background-color: rgba(25, 135, 84, 0.15);
        color: #34A853;
        border: 1px solid rgba(25, 135, 84, 0.3);
    }
    
    /* Expanders styling */
    .stExpander {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        background-color: var(--secondary-background-color) !important;
        margin-bottom: 12px !important;
        box-shadow: none !important;
    }
    
    /* Sidebar hide styling override */
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cache Model and Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns

@st.cache_resource
def load_shap_explainer(_model):
    # Load background dataset (100 samples is ideal for speed)
    background_df = pd.read_csv('data/processed/X_train_smoteenn.csv').head(100)
    explainer = shap.Explainer(_model, background_df)
    return explainer

try:
    model, scaler, feature_columns = load_artifacts()
    explainer = load_shap_explainer(model)
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure you ran Stage 3 modeling first.")
    artifacts_loaded = False

# 3. Top Banner Title Card
st.markdown("""
<div class="banner-container">
    <h1 style='margin:0; font-weight:800; font-size: 28px; letter-spacing: -0.5px;'>📞 Telecom Customer Retention Diagnostics</h1>
    <p style='margin:6px 0 0 0; opacity:0.85; font-size:15px; font-weight: 300;'>
        Configure a customer profile on the left and run immediate diagnostics to calculate churn risk and interpret attribution drivers.
    </p>
</div>
""", unsafe_allow_html=True)

if artifacts_loaded:
    # Set up columns for side-by-side display
    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown("### 🛠️ Customer Profile Configurator")
        st.write("Modify the sections below to construct the customer profile:")
        
        # Section 1: Demographics
        with st.expander("👤 Demographic Details", expanded=True):
            dem_c1, dem_c2 = st.columns(2)
            with dem_c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen (Age >= 65)", ["No", "Yes"])
            with dem_c2:
                partner = st.selectbox("Has a Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                
        # Section 2: Services
        with st.expander("🔌 Service Subscriptions", expanded=True):
            ser_c1, ser_c2 = st.columns(2)
            with ser_c1:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
            with ser_c2:
                if internet_service == "No":
                    online_security = "No internet service"
                    online_backup = "No internet service"
                    device_protection = "No internet service"
                    tech_support = "No internet service"
                    streaming_tv = "No internet service"
                    streaming_movies = "No internet service"
                    st.info("No active internet service selected.")
                else:
                    online_security = st.selectbox("Online Security", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
                    
        # Section 3: Billing & Contract
        with st.expander("🧾 Billing & Contract details", expanded=True):
            bil_c1, bil_c2 = st.columns(2)
            with bil_c1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            with bil_c2:
                tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                monthly_charges = st.slider("Monthly Charges ($)", min_value=10.0, max_value=120.0, value=50.0, step=0.5)
                total_charges = st.slider("Total Charges ($)", min_value=10.0, max_value=9000.0, value=500.0, step=10.0)

        st.write("")
        predict_btn = st.button("📊 Run Churn Diagnosis & Attribution", use_container_width=True, type="primary")

    with col_output:
        st.markdown("### 📊 Diagnostics Dashboard")
        
        # Run prediction directly at startup, or wait for click
        # To make it look interactive and responsive right away, let's run a baseline on first load!
        # This keeps the dashboard from looking "empty" when they open it.
        
        # Preprocess inputs
        input_dict = {col: 0 for col in feature_columns}
        
        input_dict['gender'] = 1 if gender == "Male" else 0
        input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
        input_dict['Partner'] = 1 if partner == "Yes" else 0
        input_dict['Dependents'] = 1 if dependents == "Yes" else 0
        input_dict['PhoneService'] = 1 if phone_service == "Yes" else 0
        input_dict['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
        
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges
        input_dict['charges_per_month'] = total_charges / tenure if tenure > 0 else monthly_charges
        
        # Categorical mappings
        if multiple_lines == "No phone service":
            input_dict['MultipleLines_No phone service'] = 1
        elif multiple_lines == "Yes":
            input_dict['MultipleLines_Yes'] = 1
            
        if internet_service == "Fiber optic":
            input_dict['InternetService_Fiber optic'] = 1
        elif internet_service == "No":
            input_dict['InternetService_No'] = 1
            
        if online_security == "No internet service":
            input_dict['OnlineSecurity_No internet service'] = 1
        elif online_security == "Yes":
            input_dict['OnlineSecurity_Yes'] = 1
            
        if online_backup == "No internet service":
            input_dict['OnlineBackup_No internet service'] = 1
        elif online_backup == "Yes":
            input_dict['OnlineBackup_Yes'] = 1
            
        if device_protection == "No internet service":
            input_dict['DeviceProtection_No internet service'] = 1
        elif device_protection == "Yes":
            input_dict['DeviceProtection_Yes'] = 1
            
        if tech_support == "No internet service":
            input_dict['TechSupport_No internet service'] = 1
        elif tech_support == "Yes":
            input_dict['TechSupport_Yes'] = 1
            
        if streaming_tv == "No internet service":
            input_dict['StreamingTV_No internet service'] = 1
        elif streaming_tv == "Yes":
            input_dict['StreamingTV_Yes'] = 1
            
        if streaming_movies == "No internet service":
            input_dict['StreamingMovies_No internet service'] = 1
        elif streaming_movies == "Yes":
            input_dict['StreamingMovies_Yes'] = 1
            
        if contract == "One year":
            input_dict['Contract_One year'] = 1
        elif contract == "Two year":
            input_dict['Contract_Two year'] = 1
            
        if payment_method == "Credit card (automatic)":
            input_dict['PaymentMethod_Credit card (automatic)'] = 1
        elif payment_method == "Electronic check":
            input_dict['PaymentMethod_Electronic check'] = 1
        elif payment_method == "Mailed check":
            input_dict['PaymentMethod_Mailed check'] = 1
            
        if 13 <= tenure <= 24:
            input_dict['tenure_group_13-24'] = 1
        elif 25 <= tenure <= 48:
            input_dict['tenure_group_25-48'] = 1
        elif 49 <= tenure <= 60:
            input_dict['tenure_group_49-60'] = 1
        elif 61 <= tenure <= 72:
            input_dict['tenure_group_61-72'] = 1
            
        input_df = pd.DataFrame([input_dict])[feature_columns]
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Predictions
        prob = model.predict_proba(input_df)[0][1]
        churn_risk = "HIGH" if prob >= 0.50 else "LOW"
        
        # Color definitions for theme-friendly gauges
        gauge_color = "#EA4335" if churn_risk == "HIGH" else "#34A853"
        risk_class = "high" if churn_risk == "HIGH" else "low"
        risk_label = "⚠️ HIGH CHURN RISK" if churn_risk == "HIGH" else "✅ LOW CHURN RISK"
        
        # HTML styled metric card
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">Analysis Output</div>
            <div class="gauge-wrapper">
                <div style="font-size: 15px; font-weight: 500; color: #888;">Churn Probability</div>
                <div class="gauge-value" style="color: {gauge_color};">{prob * 100:.1f}%</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill" style="width: {prob * 100}%; background-color: {gauge_color};"></div>
                </div>
                <div class="risk-badge {risk_class}">{risk_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP attribution card
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">Explainable AI (SHAP) Prediction Drivers</div>
            <p style="font-size: 13.5px; color: #888; margin-bottom: 15px;">
                Waterfall plot illustrating features shifting customer probability towards churn (red) or retention (blue):
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            shap_values = explainer(input_df)
            
            # Format figures to be transparent & modern
            fig, ax = plt.subplots(figsize=(9, 4.5))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            plt.title("Attribution Drivers (SHAP log-odds)", fontsize=11, fontweight='bold', pad=12)
            plt.gcf().patch.set_facecolor('none')  # Transparent background
            ax.patch.set_facecolor('none')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as ex:
            st.error(f"Attribution error: {ex}")
            
        # Loyalty Action plan card
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">💡 Proactive Retention Strategy</div>
        </div>
        """, unsafe_allow_html=True)
        
        if churn_risk == "HIGH":
            recs = []
            if contract == "Month-to-month":
                recs.append("**Contract Incentives**: Pitch a 1-year or 2-year contract upgrade to lock in stability. Month-to-month contracts have a **42.7%** baseline churn rate in our historical data.")
            if monthly_charges > 70.0:
                recs.append(f"**Billing Relief**: This customer has high monthly billing (**${monthly_charges}**). Provide a custom service bundle discount or a temporary **$10 monthly loyalty credit** for 6 months.")
            if payment_method == "Electronic check":
                recs.append("**Billing Automation**: Promote moving away from Electronic Check (associated with a high **45.3%** churn rate) to automatic bank transfer / credit card with a one-time $10 account credit.")
            if internet_service == "Fiber optic":
                recs.append("**Quality Technical Outreach**: Schedule a technical health check-in call to ensure satisfaction, as Fiber Optic subscribers show high cost/quality sensitivity.")
            if internet_service != "No" and online_security == "No":
                recs.append("**Security Package Trial**: Offer a 30-day trial of our Online Security add-on to increase service stickiness and peace of mind.")
            if internet_service != "No" and tech_support == "No":
                recs.append("**Premium Support Trial**: Offer a 3-month free trial of our premium Tech Support service to resolve setup or service quality issues.")
                
            # Fallback if no specific trigger fits but overall risk is high
            if not recs:
                recs.append("**Proactive Care Call**: Schedule a customer care call to discuss their overall experience and check for service satisfaction.")
                
            for rec in recs:
                st.markdown(f"- {rec}")
        else:
            st.markdown("""
            * **Account Stability**: The customer holds standard low-risk indicators. No loyalty pricing adjustment is required.
            * **Premium Upselling**: Leverage their loyalty indicators to introduce value-added services (e.g., Device Protection packages, premium cloud backups, or entertainment add-ons).
            """)
            
        with st.expander("🔬 Model Information"):
            st.markdown("""
            * **Classifier**: Regularized Logistic Regression (C=0.628)
            * **Balancing Strategy**: SMOTE + ENN (Edited Nearest Neighbors)
            * **Training Metrics**: Recall: **87.7%** (Class 1) | ROC-AUC: **83.3%** | Accuracy: **70.2%**
            """)
