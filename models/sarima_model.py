# models/sarima_model.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SARIMAModel:
  def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    self.order = order
    self.seasonal_order = seasonal_order

  def train(self, df: pd.DataFrame, target_column: str):
    self.model = SARIMAX(
      df[target_column],
      order=self.order,
      seasonal_order=self.seasonal_order,
    )
    self.model_fit = self.model.fit(disp=False)

  def predict(self, steps: int):
    forecast = self.model_fit.forecast(steps=steps)
    return forecast

  def evaluate(self, df: pd.DataFrame, target_column: str):
    y_pred = self.model_fit.predict(start=0, end=len(df) - 1)
    y_true = df[target_column]

    metrics = {
      "mse": mean_squared_error(y_true, y_pred),
      "mae": mean_absolute_error(y_true, y_pred),
      "r2": r2_score(y_true, y_pred),
    }

    print(f"\nMSE: {metrics['mse']:.4f}")
    print(f"\nMAE: {metrics['mae']:.4f}")
    print(f"\nR2-score: {metrics['r2']:.4f}")

    return metrics
