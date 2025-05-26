import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from models import MLPModel, handle_datetime, handle_rainfall, transform_scaler

app = FastAPI()

numerical_features = [
  "COUGM3",
  "NO2UGM3",
  "O3UGM3",
  "PM25",
  "SO2UGM3",
  "UV_INDEX",
]


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
  predict_df = pd.DataFrame(
    [
      {
        "DATETIME": data.datetime,
        "COUGM3": data.cougm3,
        "NO2UGM3": data.no2ugm3,
        "O3UGM3": data.o3ugm3,
        "PM25": data.pm25,
        "RAINFALL": data.rainfall,
        "SO2UGM3": data.so2ugm3,
        "UV_INDEX": data.uv_index,
      }
    ]
  )

  predict_df = handle_datetime(predict_df)
  predict_df = handle_rainfall(predict_df)
  predict_df[numerical_features] = transform_scaler(
    predict_df[numerical_features]
  )

  model = MLPModel()
  predictions = model.predict(predict_df)
  return {"prediction": predictions[0]}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)

# TODO: implementar un html para la api
