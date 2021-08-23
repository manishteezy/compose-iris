import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import init_models, predict_survival

# defining the main app
app = FastAPI(title="predictr", docs_url="/")
app.add_event_handler("startup", init_models)
class QueryIn(BaseModel):
    patient_age: int 
    operation_year: int
    axillary_nodes: int
  
class QueryOut(BaseModel):
    survival_status: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}

@app.post("/haberman/predict", response_model=QueryOut, status_code=200)
def predict(query_data: QueryIn):
    output = {"survival_status": predict_survival(query_data)}
    return output


@app.post("/reload_model", status_code=200)
# Route to reload the model from file
def reload_model():
    init_models()
    output = {"detail": "Model successfully loaded"}
    return output

# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
