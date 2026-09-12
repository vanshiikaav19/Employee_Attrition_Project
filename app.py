import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "attrition_model.pkl"
DATA_PATH = BASE_DIR / "data" / "HR_Employee_Attrition.csv"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("📊 Employee Attrition Prediction System")
st.write(
    "A machine learning based HR analytics dashboard "
    "for identifying employee attrition risk."
)

st.divider()

# -----------------------------
# Overview metrics
# -----------------------------
total_employees = len(df)
attrition_count = (df["Attrition"] == "Yes").sum()
attrition_rate = attrition_count / total_employees * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", total_employees)

with col2:
    st.metric("Employees Left", attrition_count)

with col3:
    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")

st.divider()

# -----------------------------
# Dataset overview
# -----------------------------
st.subheader("📋 Employee Dataset")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# -----------------------------
# Department analysis
# -----------------------------
st.subheader("🏢 Attrition by Department")

department_data = (
    df.groupby("Department")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Attrition Rate (%)")
)

st.bar_chart(
    department_data.set_index("Department")
)

# -----------------------------
# Main prediction section
# -----------------------------
st.subheader("🔍 Employee Risk Prediction")

st.write(
    "Select an employee from the dataset to calculate their "
    "predicted probability of attrition."
)

employee_index = st.selectbox(
    "Select Employee",
    df.index,
    format_func=lambda x: f"Employee {df.loc[x, 'EmployeeNumber']}"
)

employee = df.loc[[employee_index]].copy()

# Remove target and irrelevant columns
drop_columns = [
    "Attrition",
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

employee_input = employee.drop(
    columns=[c for c in drop_columns if c in employee.columns]
)

# -----------------------------
# Prediction
# -----------------------------
try:
    probability = model.predict_proba(employee_input)[0][1]
    prediction = model.predict(employee_input)[0]

    risk_percentage = probability * 100

    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Attrition Probability",
            f"{risk_percentage:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:
        actual_status = employee["Attrition"].iloc[0]
        st.metric(
            "Actual Status",
            actual_status
        )

    # Progress bar
    st.write("Risk Score")
    st.progress(min(probability, 1.0))

    # -----------------------------
    # Employee details
    # -----------------------------
    st.subheader("👤 Employee Details")

    detail_columns = [
        "Age",
        "Department",
        "JobRole",
        "JobLevel",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "OverTime",
        "BusinessTravel",
        "DistanceFromHome"
    ]

    available_columns = [
        c for c in detail_columns if c in employee.columns
    ]

    st.dataframe(
        employee[available_columns].T.rename(
            columns={employee_index: "Value"}
        ),
        use_container_width=True
    )

except Exception as e:
    st.error(f"Prediction error: {e}")

# -----------------------------
# Important project note
# -----------------------------
st.divider()

st.info(
    "⚠️ This application is an HR analytics demonstration. "
    "Risk scores should support human review and should not be "
    "used as the sole basis for employment decisions."
)

st.caption(
    "Employee Attrition Prediction Project | Machine Learning & HR Analytics"
)