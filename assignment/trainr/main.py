import uvicorn
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from utils import init_models, save_train_model
from typing import List

PREDICTR_ENDPOINT = os.getenv("PREDICTR_ENDPOINT")

# defining the main app
app = FastAPI(title="trainr", docs_url="/")

# calling the load_model during startup.
app.add_event_handler("startup", init_models)

class QueryInToTrain(BaseModel):
    patient_age: int 
    operation_year: int
    axillary_nodes: int
    survival_status: int

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/haberman/train", status_code=200)
def model_train(data: List[QueryInToTrain]):
    save_train_model(data) 
    response = requests.post(f"{PREDICTR_ENDPOINT}/reload_model")
    return {"detail": "Training successful"}

# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)

