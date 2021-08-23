import pickle
from typing import List
import pandas as pd

model_pkl = "models/random_forest_model.pkl"
rf_model = pickle.load(open(model_pkl, "rb"))
# function to train and load the model during startup
def init_models():
    initiate_model()

def initiate_model():
    global rf_model
    rf_model = pickle.load(open(model_pkl, "rb"))

# function to predict
def predict_survival(query_data):
    X = list(query_data.dict().values())
    X=pd.DataFrame([X], columns=["patient_age","operation_year","axillary_nodes"])
    prediction = rf_model.predict(X)
    if(prediction == 1):
        return "the patient survived 5 years or longer"
    else:
        return "the patient died within 5 year"

