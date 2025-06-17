import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import (
  confusion_matrix,
  roc_curve,
  auc,
  classification_report,
)


class SVMModel:
  def __init__(self, path="svm_model.joblib"):
    self.model = SVR(kernel="rbf", C=100, epsilon=0.5, gamma="auto")
    self.model_path = path

  def train(self, X, y):
    self.model.fit(X, y)
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

  def get_best_estimator(self, X, y):
    model = SVR()

    param_grid = [
      {
        # RBF Kernel (Primary recommendation)
        "kernel": ["rbf"],
        "C": [10, 100, 1000],
        "epsilon": [0.2, 0.5, 1.0],
        "gamma": ["scale", "auto", 0.001],
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
    self, X, y_true, save_path="svm_results.png", task_type="regression"
  ):
    """
    Plot results for SVR (Support Vector Regression) or SVC (Support Vector Classification)
    """
    model = joblib.load(self.model_path)
    y_pred = model.predict(X)

    if task_type == "regression":
      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("Support Vector Regression (SVR) Results", fontsize=16)

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

      # Plot 3: Error Distribution
      axes[1, 0].hist(residuals, bins=30, alpha=0.7, color="skyblue")
      axes[1, 0].set_xlabel("Residuals")
      axes[1, 0].set_ylabel("Frequency")
      axes[1, 0].set_title("Residuals Distribution")

      # Plot 4: Support Vectors visualization (if 2D data)
      if X.shape[1] == 2:
        axes[1, 1].scatter(
          X[:, 0], X[:, 1], c=y_true, cmap="viridis", alpha=0.6
        )
        # Highlight support vectors if available
        if hasattr(model, "support_"):
          axes[1, 1].scatter(
            X[model.support_, 0],
            X[model.support_, 1],
            s=100,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
          )
        axes[1, 1].set_xlabel("Feature 1")
        axes[1, 1].set_ylabel("Feature 2")
        axes[1, 1].set_title("Support Vectors (red circles)")
      else:
        # For higher dimensional data, show prediction error vs actual values
        abs_errors = np.abs(residuals)
        axes[1, 1].scatter(y_true, abs_errors, alpha=0.6)
        axes[1, 1].set_xlabel("Actual Values")
        axes[1, 1].set_ylabel("Absolute Error")
        axes[1, 1].set_title("Absolute Error vs Actual Values")

    else:  # Classification
      # Convert to classification labels if needed
      y_pred_class = np.round(y_pred).astype(int)
      y_true_class = y_true.astype(int)

      fig, axes = plt.subplots(2, 2, figsize=(15, 12))
      fig.suptitle("Support Vector Classification (SVC) Results", fontsize=16)

      # Plot 1: Confusion Matrix
      cm = confusion_matrix(y_true_class, y_pred_class)
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
      axes[0, 0].set_xlabel("Predicted")
      axes[0, 0].set_ylabel("Actual")
      axes[0, 0].set_title("Confusion Matrix")

      # Plot 2: Decision boundary (if 2D data)
      if X.shape[1] == 2:
        h = 0.02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
          np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )

        try:
          Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
          Z = Z.reshape(xx.shape)
          axes[0, 1].contourf(xx, yy, Z, alpha=0.3)
          axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_true_class, cmap="viridis")
          axes[0, 1].set_xlabel("Feature 1")
          axes[0, 1].set_ylabel("Feature 2")
          axes[0, 1].set_title("Decision Boundary")
        except:
          axes[0, 1].text(
            0.5,
            0.5,
            "Decision boundary\nnot available",
            horizontalalignment="center",
            verticalalignment="center",
          )
          axes[0, 1].set_title("Decision Boundary (N/A)")
      else:
        axes[0, 1].text(
          0.5,
          0.5,
          "Decision boundary\nonly for 2D data",
          horizontalalignment="center",
          verticalalignment="center",
        )
        axes[0, 1].set_title("Decision Boundary (N/A)")

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
      else:
        axes[1, 0].text(
          0.5,
          0.5,
          "ROC curve only\nfor binary classification",
          horizontalalignment="center",
          verticalalignment="center",
        )
        axes[1, 0].set_title("ROC Curve (N/A)")

      # Plot 4: Classification Report Heatmap
      try:
        report = classification_report(
          y_true_class, y_pred_class, output_dict=True
        )
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        sns.heatmap(report_df, annot=True, cmap="Blues", ax=axes[1, 1])
        axes[1, 1].set_title("Classification Report")
      except:
        axes[1, 1].text(
          0.5,
          0.5,
          "Classification report\nnot available",
          horizontalalignment="center",
          verticalalignment="center",
        )
        axes[1, 1].set_title("Classification Report (N/A)")

    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # Print classification report for classification tasks
    if task_type == "classification":
      print("\nClassification Report:")
      try:
        print(classification_report(y_true_class, y_pred_class))
      except:
        print("Classification report not available")
