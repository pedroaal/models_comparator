import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .transformers import clean_pipeline


class GradientBoostingModel:
  def __init__(self, path="gb_model.pkl"):
    self.pipeline = Pipeline(
      [
        ("cleaning", clean_pipeline()),
        (
          "model",
          GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=28,
          ),
        ),
      ]
    )
    self.model_path = path

  def train(self, X, y):
    self.pipeline.fit(X, y)
    joblib.dump(self.pipeline, self.model_path)

  def predict(self, data: list[float]):
    data_array = np.array(data).reshape(1, -1)
    print(data_array)

    try:
      loaded_model = joblib.load(self.model_path)
      predictions = loaded_model.predict(data_array)
      return predictions
    except Exception as e:
      print(f"Error loading model: {e}")
      return [0.0]

  def evaluate(self, X, y):
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
