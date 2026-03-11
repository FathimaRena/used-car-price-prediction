import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # check missing values
    df.dropna(inplace=True)

    return df