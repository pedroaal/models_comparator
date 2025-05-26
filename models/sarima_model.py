# models/sarima_model.py
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .transformers import clean_pipeline


class SARIMAModel:
  def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), path="sarima_model.pkl"):
    self.pipeline = Pipeline(
      [
        ("preprocessing", clean_pipeline()),
        (
          "sarima",
          SARIMAX(
            order=order,
            seasonal_order=seasonal_order,
          ),
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
    try:
      loaded_model = joblib.load(self.model_path)
      prediction = loaded_model.predict(data)
      return prediction
    except Exception as e:
      print(f"Error loading model: {e}")
      return [0.0]

  def evaluate(self, df: pd.DataFrame, target_column: str):
    y_pred = self.pipeline.predict(start=0, end=len(df) - 1)
    y = df[target_column]

    metrics = {
      "mse": mean_squared_error(y, y_pred),
      "mae": mean_absolute_error(y, y_pred),
      "r2": r2_score(y, y_pred),
    }

    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R2-score: {metrics['r2']:.4f}")

    return metrics
