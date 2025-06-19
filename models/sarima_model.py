# models/sarima_model.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time


class SARIMAModel:
  def __init__(
    self,
    target_column,
    path="sarima_model.joblib",
  ):
    self.model_path = path
    self.target_column = target_column
    self.anomalies = None
    self.model = None

  def train(self, X, y):
    model = SARIMAX(
      y,
      exog=X,
      order=(2, 0, 2),
      seasonal_order=(1, 1, 1, 12),
      enforce_stationarity=False,
      enforce_invertibility=False,
    )
    start_time = time.time()
    model = model.fit(disp=False)
    end_time = time.time()
    print(f"Sarima training completed in {end_time - start_time:.2f} seconds")
    self.model = model
    joblib.dump(model, self.model_path)

  def predict(self, X: list[float]):
    try:
      loaded_model = joblib.load(self.model_path)
      prediction = loaded_model.get_forecast(steps=len(X), exog=X)
      return prediction.predicted_mean
    except Exception as e:
      print(f"Error loading model: {e}")
      return [0.0]

  def evaluate(self, X, y):
    forecast_result = self.model.get_forecast(steps=len(y), exog=X)
    y_pred = forecast_result.predicted_mean

    metrics = {
      "mse": mean_squared_error(y, y_pred),
      "mae": mean_absolute_error(y, y_pred),
      "r2": r2_score(y, y_pred),
    }

    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R2-score: {metrics['r2']:.4f}")

    return metrics

  def find_sarima_parameters(self, series, seasonal_period=12):
    """Find optimal SARIMA parameters using grid search"""
    print("Searching for optimal SARIMA parameters...")

    # Parameter ranges for grid search
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)

    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)

    best_aic = np.inf
    best_params = None
    best_seasonal_params = None

    for p in p_range:
      for d in d_range:
        for q in q_range:
          for P in P_range:
            for D in D_range:
              for Q in Q_range:
                try:
                  model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                  )
                  fitted_model = model.fit(disp=False)

                  if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
                    best_seasonal_params = (P, D, Q, seasonal_period)

                except:
                  continue

    print(f"Best SARIMA parameters: {best_params}")
    print(f"Best seasonal parameters: {best_seasonal_params}")
    print(f"Best AIC: {best_aic:.4f}")

    return best_params, best_seasonal_params

  def plot_results(self, save_path="sarima_results.png"):
    # Residual diagnostics
    residuals = self.model.resid
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.title("Residuals")

    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20)
    plt.title("Residuals Distribution")

    plt.subplot(2, 2, 3)
    plot_acf(residuals, ax=plt.gca(), lags=20)

    plt.subplot(2, 2, 4)
    plot_pacf(residuals, ax=plt.gca(), lags=20)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
