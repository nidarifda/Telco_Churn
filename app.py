import streamlit as st
import numpy as np
import joblib

# === Page Configuration ===
st.set_page_config(
    page_title="TelcoChurn AI",
    page_icon="📊",
    layout="centered"
)

# === Load Model and Scaler ===
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# === Sidebar ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8370/8370201.png", use_column_width=True)
    st.markdown("## About This App")
    st.markdown("""
This AI-powered tool predicts the likelihood of a telco customer churning based on key behavioral indicators.

🔹 Real-time ML inference  
🔹 Streamlined feature input  
🔹 Built with using XGBoost & Streamlit  
    """)
    st.markdown("---")
    st.caption("Created by [Your Name](https://github.com/yourusername)")

# === Main Title ===
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📡 TelcoChurn AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict customer churn with confidence.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Input Fields ===
st.markdown("#### Customer Input")

col1, col2 = st.columns(2)
with col1:
    monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 70.0, step=1.0)
    tenure_months = st.slider("Tenure (Months)", 0, 72, 24)

with col2:
    avg_gb = st.slider("Avg Monthly GB Download", 0.0, 100.0, 20.0)
    satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)

st.markdown("")

# === Predict Button ===
if st.button("🔍 Predict Churn", use_container_width=True):
    input_data = np.array([[monthly_charge, tenure_months, avg_gb, satisfaction]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.\n\n**Churn Probability: {prob:.2%}**")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\n**Churn Probability: {prob:.2%}**")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em;'>"
    "📡 TelcoChurn AI • Powered by XGBoost • © 2025 All rights reserved"
    "</div>",
    unsafe_allow_html=True
)
