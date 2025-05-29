import streamlit as st
import numpy as np
import joblib

# === Page Configuration ===
st.set_page_config(
    page_title="IBM Telco Churn Prediction",
    layout="wide"
)

# === Load Model and Scaler ===
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# === Custom Styling ===
st.markdown("""
<style>
body {
    background-color: #dfeffe !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #dfeffe;
}
section[data-testid="stSidebar"] {
    background-color: #1b2e70 !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
h1, h2, h3, h4 {
    color: #1b2e70 !important;
    text-align: center;
}
.stButton>button {
    background-color: #1b2e70;
    color: white;
    font-weight: bold;
    border-radius: 6px;
}
.card {
    background-color: #f4f4f4;
    padding: 2rem;
    border-radius: 12px;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8370/8370201.png", use_column_width=True)
    st.markdown("## About This App")
    st.markdown("""
This AI-powered tool predicts the likelihood of a telco customer churning based on key behavioral indicators.

🔹 Real-time ML inference  
🔹 Streamlined feature input  
🔹 Powered by XGBoost & Streamlit  
    """)
    st.markdown("---")
    st.caption("Created by [Your Name](https://github.com/yourusername)")

# === Main Container Card ===
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("<h1>Telco Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predict customer churn with confidence.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;'>Customer Input</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 70.0, step=1.0)
        tenure_months = st.slider("Tenure (Months)", 0, 72, 24)

    with col2:
        data_plan = st.selectbox("Data Plan", [
            "10 GB", "30 GB", "50 GB", "100 GB", "200 GB", "300 GB", "Unlimited"
        ])
        gb_mapping = {
            "10 GB": 10,
            "30 GB": 30,
            "50 GB": 50,
            "100 GB": 100,
            "200 GB": 200,
            "300 GB": 300,
            "Unlimited": 500
        }
        avg_gb = gb_mapping[data_plan]
        satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)

    # === Prediction ===
    if st.button("Predict Churn"):
        input_data = np.array([[monthly_charge, tenure_months, avg_gb, satisfaction]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ This customer is **likely to churn**.\n\n**Churn Probability: {prob:.2%}**")
        else:
            st.success(f"✅ This customer is **likely to stay**.\n\n**Churn Probability: {prob:.2%}**")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "📡 TelcoChurn AI • Powered by XGBoost • © 2025 All rights reserved"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)  # Close card
