import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Data Analysis")

data = pd.read_csv("data/car_data.csv")

st.subheader("Selling Price Distribution")

fig, ax = plt.subplots()
sns.histplot(data["Selling_Price"], kde=True)
st.pyplot(fig)

st.subheader("Fuel Type vs Selling Price")

fig2, ax2 = plt.subplots()
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=data)
st.pyplot(fig2)