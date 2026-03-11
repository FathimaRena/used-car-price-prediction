import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Model
# ------------------------------


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = pickle.load(open(model_path, "rb"))

# ------------------------------
# Load Dataset
# ------------------------------
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "car_data.csv")
data = pd.read_csv(data_path)

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title(" AI Powered Used Car Price Prediction")
st.write("Predict the resale value of a used car using Machine Learning.")

st.markdown("---")

# ------------------------------
# Data Visualizations
# ------------------------------

st.header("📊 Data Insights")

st.subheader("Selling Price Distribution")

fig1, ax1 = plt.subplots()
sns.histplot(data["Selling_Price"], kde=True)
st.pyplot(fig1)

st.subheader("Fuel Type vs Selling Price")

fig2, ax2 = plt.subplots()
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=data)
st.pyplot(fig2)

st.markdown("---")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Car Details")

year = st.sidebar.slider("Manufacturing Year", 2000, 2024, 2015)

present_price = st.sidebar.number_input(
    "Present Price (in Lakhs)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000
)

owner = st.sidebar.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# ------------------------------
# Feature Engineering
# ------------------------------

fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0

seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

car_age = 2024 - year

features = np.array([[
    present_price,
    kms_driven,
    owner,
    car_age,
    fuel_diesel,
    fuel_petrol,
    seller_individual,
    transmission_manual
]])

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Price"):

    prediction = model.predict(features)

    price = round(prediction[0], 2)

    st.success(f"💰 Estimated Car Price: ₹ {price} Lakhs")

    st.balloons()

# ------------------------------
# Feature Importance
# ------------------------------

st.markdown("---")
st.header("Feature Importance")

try:
    importances = model.feature_importances_

    feature_names = [
        "Present Price",
        "KMs Driven",
        "Owner",
        "Car Age",
        "Diesel",
        "Petrol",
        "Individual Seller",
        "Manual Transmission"
    ]

    fig3, ax3 = plt.subplots()
    sns.barplot(x=importances, y=feature_names)
    st.pyplot(fig3)

except:
    st.info("Feature importance not available for this model.")

st.markdown("---")

