# models/lstm_model.py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import time


class LSTMModel:
  def __init__(
    self, window_size, features, output_shape=1, path="lstm_model.h5"
  ):
    self.window_size = window_size
    self.input_shape = (window_size, features)
    self.output_shape = output_shape
    self.model = self._build_model()
    self.model_path = path

  def _build_model(self):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(self.output_shape))

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model

  def train(self, X, y, epochs=100):
    early_stopping = EarlyStopping(
      monitor="val_loss", patience=20, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
      monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    )

    X, y = self.create_sequences(X, y)
    start_time = time.time()
    self.model.fit(
      X,
      y,
      epochs=epochs,
      batch_size=32,
      validation_split=0.2,
      callbacks=[early_stopping, reduce_lr],
      verbose=1,
    )
    end_time = time.time()
    print(f"LSTM training completed in {end_time - start_time:.2f} seconds")
    self.model.save(self.model_path)

  def predict(self, data):
    return self.model.predict(data)

  def evaluate(self, X, y):
    model = load_model(self.model_path)
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

  def create_sequences(self, X, y):
    X_seq, y_seq = [], []

    for i in range(len(X) - self.window_size):
      X_seq.append(X.iloc[i : (i + self.window_size)].values)
      y_seq.append(y.iloc[i + self.window_size])

    return np.array(X_seq), np.array(y_seq)
