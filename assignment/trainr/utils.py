import os
import pickle
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier


class QueryInToTrain(BaseModel):
    patient_age: int 
    operation_year: int
    axillary_nodes: int


rf = RandomForestClassifier(n_estimators=100, max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None)


model_pkl = "models/random_forest_model.pkl"


# training model here
def init_models():
    initiate_model()

def initiate_model():
    if not os.path.isfile(model_pkl):
        print("trainer if")
        pickle.dump(rf, open(model_pkl, "wb"))
        col_name=["patient_age","operation_year","axillary_nodes","survival_status"]
        df = pd.read_csv('data/haberman.csv' , names=col_name)
        x, y = df[["patient_age","operation_year","axillary_nodes"]], df[["survival_status"]]
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
        rf.fit(X_train, y_train)
        pickle.dump(rf, open(model_pkl, "wb"))

# function to train and save
def save_train_model(data):
    # load the model
    rf = pickle.load(open(model_pkl, "rb"))
    X = [list(d.dict().values())[:-1] for d in data]
    y = [d.survival_status for d in data]
    rf.fit(X, y)
    pickle.dump(rf, open(model_pkl, "wb"))
    return

