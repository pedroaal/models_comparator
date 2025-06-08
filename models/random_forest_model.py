# models/random_forest_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
  confusion_matrix,
  roc_curve,
  auc,
  classification_report,
)


class RandomForestModel:
  def __init__(self, path="rf_model.joblib"):
    self.model = RandomForestRegressor(n_estimators=100, random_state=28)
    self.model_path = path

  def train(self, X, y):
    self.model.fit(X, y)
    joblib.dump(self.model, self.model_path)

  def predict(self, data: list[float]):
    data_array = np.array(data).reshape(1, -1)
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

  def plot_results(
    self,
    X,
    y_true,
    save_path="random_forest_results.png",
    task_type="regression",
  ):
    """
    Plot results for Random Forest Model
    task_type: 'regression' or 'classification'
    """
    model = joblib.load(self.model_path)
    y_pred = model.predict(X)

    if task_type == "regression":
      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("Random Forest Regression Results", fontsize=16)

      # Plot 1: Actual vs Predicted
      axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
      axes[0, 0].plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
      )
      axes[0, 0].set_xlabel("Actual Values")
      axes[0, 0].set_ylabel("Predicted Values")
      axes[0, 0].set_title("Actual vs Predicted Values")

      # Plot 2: Residuals
      residuals = y_true - y_pred
      axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
      axes[0, 1].axhline(y=0, color="r", linestyle="--")
      axes[0, 1].set_xlabel("Predicted Values")
      axes[0, 1].set_ylabel("Residuals")
      axes[0, 1].set_title("Residual Plot")

      # Plot 3: Feature Importance
      if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        axes[1, 0].barh(feature_names, feature_importance)
        axes[1, 0].set_xlabel("Importance")
        axes[1, 0].set_title("Feature Importance")

      # Plot 4: Error Distribution
      axes[1, 1].hist(residuals, bins=30, alpha=0.7, color="skyblue")
      axes[1, 1].set_xlabel("Residuals")
      axes[1, 1].set_ylabel("Frequency")
      axes[1, 1].set_title("Residuals Distribution")

    else:  # Classification
      # Convert regression predictions to classification (you may need to adjust this)
      y_pred_class = np.round(y_pred).astype(int)
      y_true_class = y_true.astype(int)

      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("Random Forest Classification Results", fontsize=16)

      # Plot 1: Confusion Matrix
      cm = confusion_matrix(y_true_class, y_pred_class)
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
      axes[0, 0].set_xlabel("Predicted")
      axes[0, 0].set_ylabel("Actual")
      axes[0, 0].set_title("Confusion Matrix")

      # Plot 2: Feature Importance
      if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
        axes[0, 1].barh(feature_names, feature_importance)
        axes[0, 1].set_xlabel("Importance")
        axes[0, 1].set_title("Feature Importance")

      # Plot 3: ROC Curve (for binary classification)
      if len(np.unique(y_true_class)) == 2:
        fpr, tpr, _ = roc_curve(y_true_class, y_pred)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(
          fpr,
          tpr,
          color="darkorange",
          lw=2,
          label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        axes[1, 0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC Curve")
        axes[1, 0].legend(loc="lower right")

      # Plot 4: Classification Report Heatmap
      report = classification_report(
        y_true_class, y_pred_class, output_dict=True
      )
      report_df = pd.DataFrame(report).iloc[:-1, :].T
      sns.heatmap(report_df, annot=True, cmap="Blues", ax=axes[1, 1])
      axes[1, 1].set_title("Classification Report")

    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
