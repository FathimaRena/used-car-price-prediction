import pickle
import numpy as np

model = pickle.load(open("../models/best_model.pkl", "rb"))

def predict_price(features):

    prediction = model.predict(features)

    return prediction