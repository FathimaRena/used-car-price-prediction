#  Used Car Price Prediction

A Machine Learning web application that predicts the resale value of used cars based on various features such as year, fuel type, kilometers driven, and transmission.

The project includes **data preprocessing, feature engineering, model comparison, and a Streamlit dashboard for predictions and analysis.**

---

##  Project Overview

Used car prices depend on many factors such as vehicle age, mileage, fuel type, and ownership history.
This project uses **machine learning models** to estimate the resale value of a used car.

The application allows users to:

* Analyze dataset insights
* Compare model performance
* Predict car resale prices interactively

---

## Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Streamlit**

---

## Machine Learning Models Used

The following models were trained and evaluated:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

The **best performing model is saved and used for predictions.**

---

## Features of the Application

### Data Analysis

* Selling price distribution
* Fuel type vs selling price

### Model Performance

* Comparison of ML models
* R² score visualization

### Price Prediction

Users can input:

* Manufacturing year
* Present price
* Kilometers driven
* Number of previous owners
* Fuel type
* Seller type
* Transmission type

The system predicts the **estimated resale value of the car.**

---

## Project Structure

```
used-car-price-prediction
│
├── data
│   └── car_data.csv
│
├── models
│   └── best_model.pkl
│
├── notebooks
│
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_models.py
│
├── streamlit_app
│   ├── app.py
│   └── pages
│       ├── data_analysis.py
│       ├── model_performance.py
│       └── price_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Running the Project Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/used-car-price-prediction.git
```

### 2️⃣ Navigate to project folder

```
cd used-car-price-prediction
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Train the model

```
python src/train_models.py
```

### 5️⃣ Run Streamlit app

```
streamlit run streamlit_app/app.py
```

---

##  Example Prediction Workflow

1. User enters car details
2. Features are processed
3. Trained ML model predicts resale value
4. Result is displayed in the dashboard

---

## Future Improvements

* Hyperparameter tuning
* Additional dataset features
* Deployment on cloud
* Explainable AI (SHAP)

---

## 👩‍💻 Author

Fathima Rena
B.Tech AI & ML

---
