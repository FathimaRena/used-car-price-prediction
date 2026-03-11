import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Model Performance")

data = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "R2 Score": [
        0.82,
        0.91,
        0.92,
        0.94
    ]
}

df = pd.DataFrame(data)

st.write("### Model Comparison")
st.dataframe(df)

fig, ax = plt.subplots()

ax.bar(df["Model"], df["R2 Score"])

ax.set_ylabel("R2 Score")
ax.set_title("Model Performance Comparison")

st.pyplot(fig)