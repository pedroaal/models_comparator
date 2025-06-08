# models/sarima_model.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf


class SARIMAModel:
  def __init__(
    self,
    target_column,
    path="sarima_model.joblib",
  ):
    self.model_path = path
    self.target_column = target_column
    self.anomalies = None

  def train(self, X):
    model = SARIMAX(
      X,
      order=(2, 0, 2),
      seasonal_order=(1, 1, 1, 12),
      enforce_stationarity=False,
      enforce_invertibility=False,
    )
    model.fit(disp=False)
    joblib.dump(model, self.model_path)

  def predict(self, data: list[float]):
    try:
      loaded_model = joblib.load(self.model_path)
      prediction = loaded_model.predict(data)
      return prediction
    except Exception as e:
      print(f"Error loading model: {e}")
      return [0.0]

  def evaluate(self, X):
    model = joblib.load(self.model_path)

    n_periods = len(X)
    forecast_res = model.get_forecast(steps=n_periods)
    forecast = forecast_res.predicted_mean
    pred_intervals = forecast_res.conf_int(alpha=0.05)

    # Extract actual values
    actual_values = X.values
    lower_bound = pred_intervals.iloc[:, 0].values
    upper_bound = pred_intervals.iloc[:, 1].values

    # Identify anomalies
    anomalies_idx = (actual_values < lower_bound) | (
      actual_values > upper_bound
    )

    print(f"\nAnomaly Detection Results:")
    print(f"Total observations: {len(X)}")
    print(f"Anomalies detected: {anomalies_idx.sum()}")
    print(f"Anomaly rate: {(anomalies_idx.sum() / len(X) * 100):.2f}%")

    return anomalies_idx

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

  def plot_results(self, df, figsize=(15, 12)):
    """Plot the results of anomaly detection"""
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Plot 1: Original series with anomalies
    axes[0].plot(
      df.index, df[self.target_column], "b-", label="Actual", alpha=0.7
    )
    axes[0].plot(df.index, model.fittedvalues, "r-", label="Fitted", alpha=0.7)

    # Highlight anomalies
    anomaly_points = df[df["is_anomaly"]]
    axes[0].scatter(
      anomaly_points.index,
      anomaly_points[self.target_column],
      color="red",
      s=50,
      label="Anomalies",
      zorder=5,
    )

    axes[0].fill_between(
      df.index,
      df["lower_bound"],
      df["upper_bound"],
      alpha=0.2,
      color="gray",
      label="Confidence Interval",
    )
    axes[0].set_title(f"{self.target_column} - Actual vs Fitted with Anomalies")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    axes[1].plot(df.index, self.residuals, "g-", alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.8)
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Anomaly scores
    axes[2].plot(df.index, df["anomaly_score"], "purple", alpha=0.7)
    axes[2].scatter(
      anomaly_points.index,
      anomaly_points["anomaly_score"],
      color="red",
      s=50,
      zorder=5,
    )
    axes[2].set_title("Anomaly Scores")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: ACF of residuals
    plot_acf(self.residuals.dropna(), ax=axes[3], lags=20)
    axes[3].set_title("ACF of Residuals")

    plt.tight_layout()
    plt.show()

  def get_anomaly_summary(self):
    """Get summary of detected anomalies"""
    if self.anomalies is None:
      print("No anomalies detected yet. Run detect_anomalies first.")
      return None

    anomaly_data = self.anomalies[self.anomalies["is_anomaly"]].copy()

    if len(anomaly_data) == 0:
      print("No anomalies detected in the data.")
      return None

    summary = {
      "anomaly_dates": anomaly_data.index.tolist(),
      "anomaly_values": anomaly_data[self.target_column].tolist(),
      "predicted_values": anomaly_data["predicted"].tolist(),
      "anomaly_scores": anomaly_data["anomaly_score"].tolist(),
      "anomaly_count": len(anomaly_data),
      "anomaly_rate": len(anomaly_data) / len(self.anomalies) * 100,
    }

    print("Anomaly Summary:")
    print("-" * 50)
    for i, (date, actual, predicted, score) in enumerate(
      zip(
        summary["anomaly_dates"],
        summary["anomaly_values"],
        summary["predicted_values"],
        summary["anomaly_scores"],
      )
    ):
      print(
        f"Anomaly {i + 1}: {date.strftime('%Y-%m')} | "
        f"Actual: {actual:.2f} | Predicted: {predicted:.2f} | "
        f"Score: {score:.2f}"
      )

    return summary
