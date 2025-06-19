import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
  silhouette_score,
  calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import time
from sklearn.metrics import (
  confusion_matrix,
  roc_curve,
  auc,
  classification_report,
)


class DBSCANModel:
  def __init__(self, path="dbscan_model.joblib"):
    self.model = DBSCAN(eps=0.5, min_samples=5)
    self.model_path = path
    self.labels = None
    self.anomalies = None

  def train(self, X):
    start_time = time.time()
    labels = self.model.fit_predict(X)
    end_time = time.time()
    print(f"DBSCAN training completed in {end_time - start_time:.2f} seconds")
    self.labels = labels
    # Save the fitted model and labels
    model_data = {
      "labels": labels,
      "core_sample_indices": self.model.core_sample_indices_,
      "components": self.model.components_
      if hasattr(self.model, "components_")
      else None,
      "eps": self.model.eps,
      "min_samples": self.model.min_samples,
    }
    joblib.dump(model_data, self.model_path)

  def predict(self, X):
    """
    For DBSCAN, we return anomalies (points labeled as -1)
    Returns: 1 for normal points, -1 for anomalies
    """
    model_data = joblib.load(self.model_path)

    # For new predictions, we can use distance to existing clusters
    # This is a simplified approach - for production, consider using
    # a separate anomaly detection method based on cluster centroids
    X_new = (
      np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
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
    labels = self.labels

    # Anomalías son las ventanas con etiqueta -1
    self.anomalies = np.where(labels == -1)[0]
    mask = labels != -1  # quitamos outliers

    metrics = {
      "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
      "n_noise_points": np.sum(labels == -1),
      "noise_ratio": np.sum(labels == -1) / len(labels),
      "silhouette_score": -1,
      "calinski_harabasz_score": 0,
    }

    if np.unique(labels[mask]).size > 1:
      metrics["silhouette_score"] = silhouette_score(X[mask], labels[mask])
      metrics["calinski_harabasz_score"] = calinski_harabasz_score(
        X[mask], labels[mask]
      )

    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Number of noise points: {metrics['n_noise_points']}")
    print(f"Noise ratio: {metrics['noise_ratio']:.4f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")

    return metrics

  def plot_anomalies(self, X, save_path="dbscan_anomalies.png"):
    anomaly_indices = self.anomalies + 12 - 1

    plt.figure(figsize=(12, 8))
    plt.plot(X.values, label="Serie original")
    plt.scatter(
      anomaly_indices, X.iloc[anomaly_indices], color="red", label="Anomalías"
    )
    plt.title("Anomalías detectadas por DBSCAN")
    plt.xlabel("Índice temporal")
    plt.ylabel("UV Index")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

  def plot_pca(self, X, save_path="dbscan_pca.png"):
    labels = self.labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 8))
    for cluster_label in set(labels):
      mask = labels == cluster_label
      plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster_label}",
        alpha=0.6,
      )

    plt.title("Visualización de clusters DBSCAN (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

  def plot_window(self, X, save_path="dbscan_results.png"):
    labels = self.labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters))

    for i, cluster_id in enumerate(sorted(set(labels))):
      if cluster_id == -1 or i >= n_clusters:
        continue
      idx = np.where(labels == cluster_id)[0][0]
      axes[i].plot(X[idx])
      axes[i].set_title(f"Ejemplo de ventana del Cluster {cluster_id}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

  def get_anomalies(self, X, window_size=12):
    anomalies = self.anomalies
    anomaly_indices_in_series = anomalies + window_size - 1
    df_tmp = X.iloc[anomaly_indices_in_series]
    print(df_tmp.head())
    return df_tmp
