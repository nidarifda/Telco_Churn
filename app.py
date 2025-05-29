import streamlit as st
import numpy as np
import joblib

# === Page Configuration ===
st.set_page_config(
    page_title="IBM Telco Churn Prediction",
    layout="centered"
)

# === Custom Styling ===
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #dfeffe;  /* Soft Blue */
        }
        .scrollable-container {
            background-color: #f4f4f4;
            max-height: 90vh;
            overflow-y: auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1b2e70 !important;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        p {
            text-align: center;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #1b2e70;
            color: white;
            font-weight: bold;
            border-radius: 6px;
        }
        section[data-testid="stSidebar"] {
            background-color: #1b2e70 !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        .stFileUploader, .stExpander, .stSelectbox, .stSlider, .stNumberInput {
            background-color: #f4f4f4 !important;
            border-radius: 10px !important;
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Model and Scaler ===
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# === Sidebar ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8370/8370201.png", use_column_width=True)
    st.markdown("## About This App", unsafe_allow_html=True)
    st.markdown("""
This AI-powered tool predicts the likelihood of a telco customer churning based on key behavioral indicators.

🔹 Real-time ML inference  
🔹 Streamlined feature input  
🔹 Powered by XGBoost & Streamlit  
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Created by [Your Name](https://github.com/yourusername)")

# === Main Scrollable Content Container ===
st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

# === Title & Description ===
st.markdown("<h1>Telco Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p>Predict customer churn with confidence.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Input Fields ===
st.markdown("#### Customer Input")

col1, col2 = st.columns(2)
with col1:
    monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 70.0, step=1.0)
    tenure_months = st.slider("Tenure (Months)", 0, 72, 24, key="tenure")

with col2:
    data_plan = st.selectbox("Data Plan", [
        "10 GB", "30 GB", "50 GB", "100 GB", "200 GB", "300 GB", "Unlimited"
    ], key="data_plan")

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
    satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3, key="satisfaction")

# === Predict Button ===
if st.button("Predict Churn"):
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

# === Close Scroll Container ===
st.markdown("</div>", unsafe_allow_html=True)
