from fastapi import FastAPI

from models.gradient_boosting_model import GradientBoostingModel

app = FastAPI()

@app.post("/predict")
def predict(data: list):
  model = GradientBoostingModel()
  predictions = model.predict(data)
  return {"prediction": predictions.tolist()}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)