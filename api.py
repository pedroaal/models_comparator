import pandas as pd
from fastapi import FastAPI, UploadFile, File

from models.knn_model import KNNModel

app = FastAPI()

@app.post("/predict")
def predict(data: list):
  model = KNNModel()
  predictions = model.predict(data)
  return {"prediction": predictions.tolist()}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)