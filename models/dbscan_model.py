import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
  silhouette_score,
  adjusted_rand_score,
  calinski_harabasz_score,
)


class DBSCANModel:
  def __init__(self, path="dbscan_model.joblib"):
    self.model = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
    self.model_path = path
    self.labels_ = None

  def train(self, X):
    self.model.fit_predict(X)
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

  def evaluate(self, X, y=None):
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

    # If true labels are provided, calculate ARI
    if y is not None:
      metrics["adjusted_rand_score"] = adjusted_rand_score(y, labels)

    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Number of noise points: {metrics['n_noise_points']}")
    print(f"Noise ratio: {metrics['noise_ratio']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")

    if y is not None:
      print(f"Adjusted Rand Score: {metrics['adjusted_rand_score']:.4f}")

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
