# models/random_forest_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .transformers import clean_pipeline


class RandomForestModel:
  def __init__(self, path="rf_model.pkl"):
    self.pipeline = Pipeline(
      [
        ("cleaning", clean_pipeline()),
        (
          "model",
          RandomForestRegressor(n_estimators=100, random_state=28),
        ),
      ]
    )
    self.model_path = path

  def train(self, df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    self.pipeline.fit(X, y)
    joblib.dump(self.pipeline, self.model_path)

  def predict(self, data: list[float]):
    data_array = np.array(data).reshape(1, -1)
    try:
      loaded_model = joblib.load(self.model_path)
      predictions = loaded_model.predict(data_array)
      return predictions
    except Exception as e:
      print(f"Error loading model: {e}")
      return [0.0]

  def evaluate(self, df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = joblib.load(self.model_path)
    y_pred = model.predict(X)

    metrics = {
      "mse": mean_squared_error(y, y_pred),
      "mae": mean_absolute_error(y, y_pred),
      "r2": r2_score(y, y_pred),
    }

    print(f"\nMSE: {metrics['mse']:.4f}")
    print(f"\nMAE: {metrics['mae']:.4f}")
    print(f"\nR2-score: {metrics['r2']:.4f}")

    return metrics
