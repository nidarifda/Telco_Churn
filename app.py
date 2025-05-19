import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

st.set_page_config(page_title="Telco Churn Predictor")
st.title("🔍 Telco Churn Prediction App")

# User Inputs
st.header("📋 Enter Customer Information")
monthly_charge = st.slider("Monthly Charge", 10, 150, 70)
tenure_months = st.slider("Tenure in Months", 0, 72, 24)
avg_gb = st.slider("Avg Monthly GB Download", 0.0, 100.0, 15.0)
satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)

# Prepare input for prediction
if st.button("Predict Churn"):
    input_data = np.array([[monthly_charge, tenure_months, avg_gb, satisfaction]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 🔮 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn (probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay (probability: {prob:.2f})")
