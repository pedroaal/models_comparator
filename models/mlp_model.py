# models/mlp_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import (
  confusion_matrix,
  roc_curve,
  auc,
  classification_report,
)
import time


class MLPModel:
  def __init__(self, path="mlp_model.joblib"):
    self.model = Pipeline(
      [
        ("pca", PCA(n_components=8)),
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

  def train(self, X, y):
    start_time = time.time()
    self.model.fit(X, y)
    end_time = time.time()
    print(f"MLP training completed in {end_time - start_time:.2f} seconds")
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

  def get_best_estimator(self, X, y):
    model = MLPRegressor(
      hidden_layer_sizes=(16, 8),
      activation="relu",
      solver="adam",
      alpha=0.5,
      random_state=28,
      learning_rate="adaptive",
      learning_rate_init=0.001,
      max_iter=500,
      early_stopping=True,
    )

    param_grid = [
      {
        "hidden_layer_sizes": [(8, 4), (16, 8), (32, 16), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "alpha": [0.001, 0.01, 0.1, 0.5],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [300, 500, 1000],
      },
    ]

    grid_search = GridSearchCV(
      model,
      param_grid,
      cv=3,
      scoring="neg_mean_squared_error",
      n_jobs=1,
      verbose=2,
    )

    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)

    return grid_search.best_estimator_

  def plot_results(
    self, X, y_true, save_path="mlp_results.png", task_type="regression"
  ):
    """
    Plot results for MLP Model
    task_type: 'regression' or 'classification'
    """
    model = joblib.load(self.model_path)
    y_pred = model.predict(X)

    if task_type == "regression":
      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("MLP Neural Network Regression Results", fontsize=16)

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

      # Plot 3: Learning Curve (Loss curve if available)
      if hasattr(model.named_steps["model"], "loss_curve_"):
        axes[1, 0].plot(model.named_steps["model"].loss_curve_)
        axes[1, 0].set_xlabel("Iterations")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Training Loss Curve")
      else:
        axes[1, 0].text(
          0.5,
          0.5,
          "Loss curve not available",
          horizontalalignment="center",
          verticalalignment="center",
        )
        axes[1, 0].set_title("Training Loss Curve (N/A)")

      # Plot 4: Error Distribution
      axes[1, 1].hist(residuals, bins=30, alpha=0.7, color="skyblue")
      axes[1, 1].set_xlabel("Residuals")
      axes[1, 1].set_ylabel("Frequency")
      axes[1, 1].set_title("Residuals Distribution")

    else:  # Classification
      # Convert regression predictions to classification
      y_pred_class = np.round(y_pred).astype(int)
      y_true_class = y_true.astype(int)

      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("MLP Neural Network Classification Results", fontsize=16)

      # Plot 1: Confusion Matrix
      cm = confusion_matrix(y_true_class, y_pred_class)
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
      axes[0, 0].set_xlabel("Predicted")
      axes[0, 0].set_ylabel("Actual")
      axes[0, 0].set_title("Confusion Matrix")

      # Plot 2: Learning Curve
      if hasattr(model.named_steps["model"], "loss_curve_"):
        axes[0, 1].plot(model.named_steps["model"].loss_curve_)
        axes[0, 1].set_xlabel("Iterations")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Training Loss Curve")

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
