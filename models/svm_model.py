import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SVMModel:
  def __init__(self, path="svm_model.joblib"):
    self.model = OneClassSVM(nu=0.5, kernel="rbf", gamma="scale")
    self.model_path = path

  def train(self, X):
    self.model.fit(X)
    joblib.dump(self.model, self.model_path)

  def predict(self, features: list):
    model = joblib.load(self.model_path)
    return model.predict(np.array(features))

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
