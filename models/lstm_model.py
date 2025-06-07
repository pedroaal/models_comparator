# models/lstm_model.py
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense


class LSTMModel:
  def __init__(self, input_shape, output_shape, path="lstm_model.joblib"):
    self.input_shape = input_shape
    self.output_shape = output_shape
    model = self._build_model()
    self.model = Pipeline(
      [("preprocessing", clean_pipeline()), ("model", model())]
    )
    self.model_path = path

  def _build_model(self):
    model = Sequential()
    model.add(LSTM(50, input_shape=self.input_shape))
    model.add(Dense(self.output_shape))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

  def train(self, X, y, epochs=100):
    self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=2)
    joblib.dump(self.model, self.model_path)

  def predict(self, data: np.array):
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    return self.model.predict(data)

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
