import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
  silhouette_score,
  calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
  confusion_matrix,
  roc_curve,
  auc,
  classification_report,
)


class DBSCANModel:
  def __init__(self, path="dbscan_model.joblib"):
    self.model = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
    self.model_path = path
    self.labels_ = None

  def train(self, X):
    self.labels_ = self.model.fit_predict(X)
    # Save the fitted model and labels
    model_data = {
      "labels": self.labels_,
      "core_sample_indices": self.model.core_sample_indices_,
      "components": self.model.components_
      if hasattr(self.model, "components_")
      else None,
      "eps": self.model.eps,
      "min_samples": self.model.min_samples,
    }
    joblib.dump(model_data, self.model_path)

  def predict(self, features: list):
    """
    For DBSCAN, we return anomalies (points labeled as -1)
    Returns: 1 for normal points, -1 for anomalies
    """
    model_data = joblib.load(self.model_path)

    # For new predictions, we can use distance to existing clusters
    # This is a simplified approach - for production, consider using
    # a separate anomaly detection method based on cluster centroids
    X_new = (
      np.array(features).reshape(1, -1)
      if len(np.array(features).shape) == 1
      else np.array(features)
    )

    # Simple heuristic: if no core samples exist, return anomaly
    if (
      model_data["core_sample_indices"] is None
      or len(model_data["core_sample_indices"]) == 0
    ):
      return np.array([-1] * len(X_new))

    # For this implementation, return -1 (anomaly) as placeholder
    # In practice, you'd implement distance-based classification
    return np.array([-1] * len(X_new))

  def evaluate(self, X):
    """
    Evaluate DBSCAN clustering performance
    For anomaly detection, y can be true anomaly labels if available
    """
    model_data = joblib.load(self.model_path)
    labels = model_data["labels"]

    # Calculate clustering metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_samples = len(labels)

    metrics = {
      "n_clusters": n_clusters,
      "n_noise_points": n_noise,
      "noise_ratio": n_noise / n_samples,
    }

    # Calculate silhouette score (only if we have more than one cluster)
    if n_clusters > 1:
      # Remove noise points for silhouette calculation
      mask = labels != -1
      if np.sum(mask) > 1:
        metrics["silhouette_score"] = silhouette_score(X[mask], labels[mask])
      else:
        metrics["silhouette_score"] = -1
    else:
      metrics["silhouette_score"] = -1

    # Calculate Calinski-Harabasz score (only for non-noise points)
    mask = labels != -1
    if n_clusters > 1 and np.sum(mask) > n_clusters:
      metrics["calinski_harabasz_score"] = calinski_harabasz_score(
        X[mask], labels[mask]
      )
    else:
      metrics["calinski_harabasz_score"] = 0

    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Number of noise points: {metrics['n_noise_points']}")
    print(f"Noise ratio: {metrics['noise_ratio']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")

    return metrics

  def get_anomalies(self, X):
    """
    Get anomaly points from training data
    Returns indices and data points of anomalies (noise points)
    """
    model_data = joblib.load(self.model_path)
    labels = model_data["labels"]

    anomaly_mask = labels == -1
    anomaly_indices = np.where(anomaly_mask)[0]
    anomaly_points = X[anomaly_mask] if len(X) == len(labels) else None

    return anomaly_indices, anomaly_points

  def get_clusters_info(self):
    """
    Get information about discovered clusters
    """
    model_data = joblib.load(self.model_path)
    labels = model_data["labels"]

    unique_labels = set(labels)
    cluster_info = {}

    for label in unique_labels:
      if label == -1:
        cluster_info["noise"] = np.sum(labels == -1)
      else:
        cluster_info[f"cluster_{label}"] = np.sum(labels == label)

    return cluster_info

  def plot_results(self, X, y_true=None, save_path="dbscan_results.png"):
    """
    Plot results for DBSCAN clustering
    """
    model_data = joblib.load(self.model_path)
    labels = model_data["labels"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("DBSCAN Clustering Results", fontsize=16)

    # Plot 1: Cluster visualization (2D PCA)
    if X.shape[1] > 2:
      pca = PCA(n_components=2)
      X_pca = pca.fit_transform(X)
    else:
      X_pca = X

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
      if k == -1:
        # Black used for noise
        col = "black"
        marker = "x"
        label = "Noise"
      else:
        marker = "o"
        label = f"Cluster {k}"

      class_member_mask = labels == k
      xy = X_pca[class_member_mask]
      axes[0, 0].scatter(
        xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.6, s=50, label=label
      )

    axes[0, 0].set_xlabel("First Principal Component")
    axes[0, 0].set_ylabel("Second Principal Component")
    axes[0, 0].set_title("DBSCAN Clustering Visualization")
    axes[0, 0].legend()

    # Plot 2: Cluster sizes
    cluster_info = self.get_clusters_info()
    cluster_names = list(cluster_info.keys())
    cluster_sizes = list(cluster_info.values())

    axes[0, 1].bar(cluster_names, cluster_sizes, color="skyblue")
    axes[0, 1].set_xlabel("Clusters")
    axes[0, 1].set_ylabel("Number of Points")
    axes[0, 1].set_title("Cluster Sizes")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: Confusion Matrix (if true labels provided)
    if y_true is not None:
      # Convert clustering labels to anomaly detection labels
      y_pred_anomaly = (labels == -1).astype(int)
      y_true_anomaly = y_true.astype(int)

      cm = confusion_matrix(y_true_anomaly, y_pred_anomaly)
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
      axes[1, 0].set_xlabel("Predicted (0=Normal, 1=Anomaly)")
      axes[1, 0].set_ylabel("Actual (0=Normal, 1=Anomaly)")
      axes[1, 0].set_title("Confusion Matrix (Anomaly Detection)")
    else:
      axes[1, 0].text(
        0.5,
        0.5,
        "No true labels provided",
        horizontalalignment="center",
        verticalalignment="center",
      )
      axes[1, 0].set_title("Confusion Matrix (N/A)")

    # Plot 4: ROC Curve (if true labels provided)
    if y_true is not None:
      y_pred_anomaly = (labels == -1).astype(int)
      y_true_anomaly = y_true.astype(int)

      fpr, tpr, _ = roc_curve(y_true_anomaly, y_pred_anomaly)
      roc_auc = auc(fpr, tpr)

      axes[1, 1].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2f})",
      )
      axes[1, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
      axes[1, 1].set_xlim([0.0, 1.0])
      axes[1, 1].set_ylim([0.0, 1.05])
      axes[1, 1].set_xlabel("False Positive Rate")
      axes[1, 1].set_ylabel("True Positive Rate")
      axes[1, 1].set_title("ROC Curve (Anomaly Detection)")
      axes[1, 1].legend(loc="lower right")
    else:
      axes[1, 1].text(
        0.5,
        0.5,
        "No true labels provided",
        horizontalalignment="center",
        verticalalignment="center",
      )
      axes[1, 1].set_title("ROC Curve (N/A)")

    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # Classification Report (if true labels provided)
    if y_true is not None:
      y_pred_anomaly = (labels == -1).astype(int)
      y_true_anomaly = y_true.astype(int)
      print("\nAnomaly Detection Classification Report:")
      print(
        classification_report(
          y_true_anomaly, y_pred_anomaly, target_names=["Normal", "Anomaly"]
        )
      )
