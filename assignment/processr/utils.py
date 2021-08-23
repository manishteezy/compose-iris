import os
import pickle
from sklearn.naive_bayes import GaussianNB

# function to process data and return it in correct format
def process_haberman(data):
   
    processed = [
        {
            "patient_age" : d.patient_age ,
            "operation_year": d.operation_year,
            "axillary_nodes": d.axillary_nodes
        }
        for d in data
    ]

    return processed
