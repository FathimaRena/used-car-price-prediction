import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from data_preprocessing import load_and_clean_data
from feature_engineering import feature_engineering


# Load dataset
df = load_and_clean_data("data/car_data.csv")

# Feature engineering
df = feature_engineering(df)

# Split data
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Models
# ------------------------------

lr = LinearRegression()
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
xgb = XGBRegressor()

# ------------------------------
# Train Models
# ------------------------------

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ------------------------------
# Predictions
# ------------------------------

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
xgb_pred = xgb.predict(X_test)

# ------------------------------
# Model Evaluation
# ------------------------------

print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Gradient Boosting R2:", r2_score(y_test, gb_pred))
print("XGBoost R2:", r2_score(y_test, xgb_pred))

# ------------------------------
# Save Best Model (XGBoost)
# ------------------------------

os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "best_model.pkl")

pickle.dump(xgb, open(model_path, "wb"))

print("Model saved at:", model_path)

print("Model training complete")