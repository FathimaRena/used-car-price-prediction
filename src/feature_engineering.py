import pandas as pd

def feature_engineering(df):

    current_year = 2025

    # create new feature
    df["CarAge"] = current_year - df["Year"]

    # drop unnecessary columns
    df.drop(["Year", "Car_Name"], axis=1, inplace=True)

    # encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df