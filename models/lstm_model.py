# models/lstm_model.py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import time


class LSTMModel:
  def __init__(
    self, features, window_size=24, output_shape=1, path="lstm_model.h5"
  ):
    self.window_size = window_size
    self.input_shape = (window_size, features)
    self.output_shape = output_shape
    self.model = self._build_model()
    self.model_path = path

  def _build_model(self):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=self.input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(self.output_shape))

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    return model

  def train(self, X, y, epochs=100):
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    start_time = time.time()
    self.model.fit(
      X,
      y,
      epochs=epochs,
      batch_size=32,
      validation_split=0.2,
      callbacks=[early_stopping],
      verbose=1,
    )
    end_time = time.time()
    print(f"LSTM training completed in {end_time - start_time:.2f} seconds")
    self.model.save(self.model_path, save_format="h5")

  def predict(self, data):
    model = load_model(self.model_path, compile=False)
    return self.model.predict(data)

  def evaluate(self, X, y):
    model = self.model
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
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

  def plot_results(self, X, y, save_path="lstm_results.png"):
    """
    Plot results for LSTM Model
    """
    model = self.model
    y_pred = model.predict(X)

    y_pred = y_pred.squeeze()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("LSTM Neural Network Regression Results", fontsize=16)

    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y, y_pred, alpha=0.6)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    axes[0, 0].set_title("Actual vs Predicted Values")

    # Plot 2: Residuals
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")

    # Plot 3: Loss Curve
    axes[1, 0].plot(model.history.history['loss'], label='Training Loss')
    axes[1, 0].plot(model.history.history['val_loss'], label='Validation Loss')
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Training and Validation Loss Curve")
    axes[1, 0].legend()

    # Plot 4: Error Distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color="skyblue")
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residuals Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
