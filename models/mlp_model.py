# models/mlp_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .transformers import clean_pipeline


class MLPModel:
  def __init__(self, path="mlp_model.pkl"):
    self.pipeline = Pipeline(
      [
        ("cleaning", clean_pipeline()),
        (
          "model",
          MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.5,
            random_state=28,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
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

    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R2-score: {metrics['r2']:.4f}")

    return metrics
