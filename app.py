# ==========================================================
# CUSTOMER LIFETIME VALUE PREDICTION APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/tuned_model.pkl")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("💰 Customer Lifetime Value")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Predict CLV",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Overview")

st.sidebar.write("**Model Type:** Random Forest (Tuned)")
st.sidebar.write("🎯 **R² Score:** 0.6954")
st.sidebar.write("📦 **Dataset Size:** 9,134 Customers")
st.sidebar.write("🧮 **Features Used:** 23 Features")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Developer")

st.sidebar.write("**Ashi Saini**")
st.sidebar.caption("ML & Data Science Enthusiast")

st.sidebar.markdown("---")

st.sidebar.success("🚀 Built using Streamlit + Scikit-learn")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("💰 Customer Lifetime Value Prediction")

    st.caption("👨‍💻 Developed by **Ashi Saini** | End-to-End Machine Learning Project")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Project Overview")

        st.write("""
This application predicts the **Customer Lifetime Value (CLV)** of an insurance customer using Machine Learning.

Customer Lifetime Value helps businesses estimate how much revenue a customer is expected to generate throughout their relationship with the company.

The prediction model is trained using the IBM Marketing Customer Value Analysis dataset.
""")

    with col2:

        st.subheader("📊 Model Performance")

        st.metric("R² Score", "0.6954")
        st.metric("MAE", "1468.01")
        st.metric("RMSE", "3961.93")

    st.markdown("---")

    st.subheader("📁 Dataset Information")

    info = pd.DataFrame({
        "Property": [
            "Rows",
            "Columns",
            "Target Variable",
            "ML Algorithm"
        ],
        "Value": [
            "9134",
            "24",
            "Customer Lifetime Value",
            "Random Forest Regressor"
        ]
    })

    st.table(info)

    st.success("Use the sidebar to navigate through the application.")




# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "📈 Predict CLV":

    st.title("📈 Customer Lifetime Value Prediction")

    st.markdown("---")

    st.subheader("Enter Customer Information")

    col1, col2 = st.columns(2)

    # =========================
    # COLUMN 1
    # =========================

    with col1:

        state = st.selectbox(
            "State",
            ["Arizona", "California", "Nevada", "Oregon", "Washington"]
        )

        response = st.selectbox(
            "Response",
            ["Yes", "No"]
        )

        coverage = st.selectbox(
            "Coverage",
            ["Basic", "Extended", "Premium"]
        )

        education = st.selectbox(
            "Education",
            [
                "Bachelor",
                "College",
                "High School or Below",
                "Master",
                "Doctor"
            ]
        )

        employment = st.selectbox(
            "Employment Status",
            [
                "Employed",
                "Unemployed",
                "Medical Leave",
                "Disabled",
                "Retired"
            ]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        income = st.number_input(
            "Income",
            min_value=0,
            value=50000
        )

        location = st.selectbox(
            "Location Code",
            ["Urban", "Suburban", "Rural"]
        )

        marital = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Divorced"]
        )

    # =========================
    # COLUMN 2
    # =========================

    with col2:

        monthly_premium = st.number_input(
            "Monthly Premium Auto",
            min_value=0,
            value=90
        )

        last_claim = st.number_input(
            "Months Since Last Claim",
            min_value=0,
            value=5
        )

        inception = st.number_input(
            "Months Since Policy Inception",
            min_value=0,
            value=30
        )

        complaints = st.number_input(
            "Number of Open Complaints",
            min_value=0,
            value=0
        )

        policies = st.number_input(
            "Number of Policies",
            min_value=1,
            value=2
        )

        policy_type = st.selectbox(
            "Policy Type",
            ["Corporate Auto", "Personal Auto", "Special Auto"]
        )

        policy = st.selectbox(
            "Policy",
            [
                "Corporate L1",
                "Corporate L2",
                "Corporate L3",
                "Personal L1",
                "Personal L2",
                "Personal L3",
                "Special L1"
            ]
        )

        renew = st.selectbox(
            "Renew Offer Type",
            ["Offer1", "Offer2", "Offer3", "Offer4"]
        )

        sales = st.selectbox(
            "Sales Channel",
            ["Agent", "Branch", "Call Center", "Web"]
        )

        claim = st.number_input(
            "Total Claim Amount",
            min_value=0.0,
            value=400.0
        )

        vehicle_class = st.selectbox(
            "Vehicle Class",
            [
                "Four-Door Car",
                "Two-Door Car",
                "SUV",
                "Sports Car",
                "Luxury SUV",
                "Luxury Car"
            ]
        )

        vehicle_size = st.selectbox(
            "Vehicle Size",
            ["Small", "Medsize", "Large"]
        )

        month = st.slider(
            "Month",
            1,
            12,
            1
        )

        day = st.slider(
            "Day",
            1,
            31,
            15
        )

    st.markdown("---")

    predict = st.button(
        "💰 Predict Customer Lifetime Value",
        use_container_width=True
    )

        # ==========================================================
    # PREDICTION
    # ==========================================================

    if predict:

        input_data = pd.DataFrame({

            "State": [state],
            "Response": [response],
            "Coverage": [coverage],
            "Education": [education],
            "EmploymentStatus": [employment],
            "Gender": [gender],
            "Income": [income],
            "Location Code": [location],
            "Marital Status": [marital],
            "Monthly Premium Auto": [monthly_premium],
            "Months Since Last Claim": [last_claim],
            "Months Since Policy Inception": [inception],
            "Number of Open Complaints": [complaints],
            "Number of Policies": [policies],
            "Policy Type": [policy_type],
            "Policy": [policy],
            "Renew Offer Type": [renew],
            "Sales Channel": [sales],
            "Total Claim Amount": [claim],
            "Vehicle Class": [vehicle_class],
            "Vehicle Size": [vehicle_size],
            "Month": [month],
            "Day": [day]

        })

        prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        st.metric(
            label="Predicted Customer Lifetime Value",
            value=f"${prediction:,.2f}"
        )

        # ======================================================
        # CUSTOMER CATEGORY
        # ======================================================

        if prediction < 5000:

            st.success("🟢 Low Value Customer")

            st.info(
                """
**Recommendation**

• Basic retention campaigns

• Email marketing

• Standard customer support
                """
            )

        elif prediction < 10000:

            st.warning("🟡 Medium Value Customer")

            st.info(
                """
**Recommendation**

• Personalized offers

• Loyalty rewards

• Cross-selling opportunities
                """
            )

        else:

            st.error("🔴 High Value Customer")

            st.info(
                """
**Recommendation**

• Premium customer service

• Dedicated relationship manager

• Exclusive rewards

• High-priority retention strategy
                """
            )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Income", f"${income:,}")

        with col2:
            st.metric("Policies", policies)

        with col3:
            st.metric("Monthly Premium", f"${monthly_premium}")

    # ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.markdown("---")

    st.subheader("📈 Feature Importance")

    st.image(
        "images/feature_importance.png",
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Lifetime Value Distribution")

        st.image(
            "images/clv_distribution.png",
            use_container_width=True
        )

    with col2:

        st.subheader("Income Distribution")

        st.image(
            "images/income_distribution.png",
            use_container_width=True
        )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Coverage")

        st.image(
            "images/coverage.png",
            use_container_width=True
        )

    with col4:

        st.subheader("Vehicle Class")

        st.image(
            "images/vehicle_class.png",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    st.image(
        "images/correlation_heatmap.png",
        use_container_width=True
    )

    st.success("All EDA visualizations generated during model development.")


   # ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.subheader("📌 Project Description")

    st.write("""
This project predicts the **Customer Lifetime Value (CLV)** of insurance customers using Machine Learning.

The model estimates the expected long-term value of a customer, helping businesses improve customer retention, optimize marketing campaigns, and increase profitability.
""")

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    tech = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Streamlit",
            "Joblib"
        ],
        "Purpose": [
            "Programming Language",
            "Data Analysis",
            "Numerical Computing",
            "Machine Learning",
            "Data Visualization",
            "Web Application",
            "Model Serialization"
        ]
    })

    st.table(tech)

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.write("**Random Forest Regressor (Hyperparameter Tuned)**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R² Score", "0.6954")

    with col2:
        st.metric("MAE", "1468.01")

    with col3:
        st.metric("RMSE", "3961.93")

    st.markdown("---")

    st.info(
        """
### 📌 Project Workflow

✔ Data Collection

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Model Training

✔ Hyperparameter Tuning

✔ Feature Importance Analysis

✔ Streamlit Deployment
"""
    )

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
### Ashi Saini

🎓 Machine Learning Enthusiast

💻 Python

🤖 Machine Learning

📊 Data Science

🌐 Streamlit
""")

    with col2:
        st.metric("Project", "Customer Lifetime Value Prediction")
        st.metric("Algorithm", "Random Forest")
        st.metric("R² Score", "0.6954")

    st.markdown("---")

    st.success("✅ Developed by Ashi Saini | End-to-End Machine Learning Project")