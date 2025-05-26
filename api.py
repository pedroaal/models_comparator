import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from models.mlp_model import MLPModel

app = FastAPI()


class Weather(BaseModel):
  datetime: str
  ambtemp: float
  cougm3: float
  no2ugm3: float
  o3ugm3: float
  pm25: float
  rainfall: float
  so2ugm3: float
  solarrad: float
  uv_index: float


@app.post("/predict")
def predict(data: Weather):
  # Convert datetime string to datetime object
  dt = pd.to_datetime(data.datetime)

  predict_data = [
    data.cougm3,
    data.no2ugm3,
    data.o3ugm3,
    data.pm25,
    data.rainfall,
    data.so2ugm3,
    data.uv_index,
    dt.year,
    dt.month,
    dt.hour,
  ]
  model = MLPModel()
  predictions = model.predict(predict_data)
  return {"prediction": predictions[0]}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)

# TODO: implementar un html para la api
