import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Student Risk Prediction System",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("../models/student_risk_model.pkl")

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🎓 Student Risk Prediction System")
st.write("Predict student academic risk based on attendance, performance, and fee status.")

st.header("Enter Student Details")

# ✅ Inputs based on training features
attendance_percentage = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

avg_score = st.number_input(
    "Average Score",
    min_value=0,
    max_value=100,
    value=70
)

fee_status = st.selectbox(
    "Fee Status",
    ["Paid", "Partial", "Overdue"]
)

# -------------------------------
# One-Hot Encoding (EXACT MATCH)
# -------------------------------
fee_status_Paid = 1 if fee_status == "Paid" else 0
fee_status_Partial = 1 if fee_status == "Partial" else 0
fee_status_Overdue = 1 if fee_status == "Overdue" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Risk Level"):
    input_df = pd.DataFrame(
        [[
            attendance_percentage,
            avg_score,
            fee_status_Overdue,
            fee_status_Partial,
            fee_status_Paid
        ]],
        columns=[
            'attendance_percentage',
            'avg_score',
            'fee_status_Overdue',
            'fee_status_Partial',
            'fee_status_Paid'
        ]
    )

    prediction = model.predict(input_df)[0]

    # Risk label mapping (based on LabelEncoder)
    risk_map = {
        0: "Low Risk ✅",
        1: "Medium Risk ⚠️",
        2: "High Risk 🚨"
    }

    st.subheader("Prediction Result")
    st.success(f"Predicted Risk Level: **{risk_map[prediction]}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI-Based Student Risk Prediction | Developed by Yuvraj")


