import streamlit as st
import numpy as np
import joblib

# === Page config ===
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# === Load model and scaler ===
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# === Sidebar branding ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Logo_telco.png/320px-Logo_telco.png", use_column_width=True)
    st.markdown("### About This App")
    st.info("""
    Predict customer churn using a machine learning model trained on telco behavioral data.

    ✅ Real-time prediction  
    ✅ Lightweight and fast  
    ✅ Accurate with minimal input

    Built by: [Your Name](https://github.com/yourusername)
    """)

# === Main app ===
st.title("Telco Customer Churn Prediction")
st.markdown("Use this tool to estimate whether a customer is likely to **churn** based on their current service metrics.")

st.markdown("---")

# === Input form layout ===
col1, col2 = st.columns(2)

with col1:
    monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 70.0)
    tenure_months = st.slider("Tenure (Months)", 0, 72, 24)

with col2:
    avg_gb = st.slider("Avg Monthly GB Download", 0.0, 100.0, 20.0)
    satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)

# === Prediction button ===
st.markdown("")

if st.button("🔍 Predict Churn"):
    input_data = np.array([[monthly_charge, tenure_months, avg_gb, satisfaction]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.\n\n**Probability: {prob:.2%}**")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\n**Probability: {prob:.2%}**")

# === Footer ===
st.markdown("---")
st.markdown("<center>© 2025 TelcoChurn AI | All rights reserved.</center>", unsafe_allow_html=True)
