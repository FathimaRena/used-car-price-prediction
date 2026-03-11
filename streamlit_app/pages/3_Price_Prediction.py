import streamlit as st
import pickle
import numpy as np

st.title("💰 Car Price Prediction")

model = pickle.load(open("models/best_model.pkl", "rb"))

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
    [0,1,2,3]
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","CNG"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["Dealer","Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)

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

if st.button("Predict Price 🚀"):

    prediction = model.predict(features)

    price = round(prediction[0],2)

    st.success(f"Estimated Car Price: ₹ {price} Lakhs")