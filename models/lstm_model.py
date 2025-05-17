# models/lstm_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import LSTM, Dense

from .transformers import clean_pipeline


class LSTMModel:
  def __init__(self, input_shape, output_shape, path="lstm_model.pkl"):
    self.input_shape = input_shape
    self.output_shape = output_shape
    model = self._build_model()
    self.pipeline = Pipeline([("preprocessing", clean_pipeline()), ("model", model())])
    self.model_path = path

  def _build_model(self):
    model = Sequential()
    model.add(LSTM(50, input_shape=self.input_shape))
    model.add(Dense(self.output_shape))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

  def train(self, df: pd.DataFrame, target_column: str, epochs=100):
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(df)
    # X = scaled_data[:, :-1]
    # y = scaled_data[:, -1]
    # X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    X = df.drop(columns=[target_column])
    y = df[target_column]

    self.pipeline.fit(X, y, epochs=epochs, batch_size=32, verbose=2)
    joblib.dump(self.pipeline, self.model_path)

  def predict(self, data: np.array):
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    return self.model.predict(data)

  def evaluate(self, df: pd.DataFrame, target_column: str):
    y_pred = self.predict(df.drop(columns=[target_column]))
    y_true = df[target_column]
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2-score: {r2:.4f}")
    return mse, mae, r2
