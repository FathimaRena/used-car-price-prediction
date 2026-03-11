import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("RMSE:", rmse)
    print("R2 Score:", r2)