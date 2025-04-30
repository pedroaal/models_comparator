import pandas as pd
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class SVMModel:
  def __init__(self):
    self.pipeline = Pipeline([
      ('preprocessing', clean_pipeline()),
      ('svm', SVC())
    ])
    self.model_path = "svm_model.pkl"

  def train(self, data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    self.pipeline.fit(X, y)
    joblib.dump(self.pipeline, self.model_path)

  def predict(self, features: list):
    model = joblib.load(self.model_path)
    return model.predict(np.array(features))

  def evaluate(self, df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = joblib.load(self.model_path)
    y_pred = model.predict(X)
    
    metrics = {
      'accuracy': accuracy_score(y, y_pred),
      'precision': precision_score(y, y_pred, average='weighted'),
      'recall': recall_score(y, y_pred, average='weighted'),
      'f1': f1_score(y, y_pred, average='weighted'),
      'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"\nRecall: {metrics['recall']:.4f}")
    print(f"\nF1-score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:\n")
    print(metrics['confusion_matrix'])
    
    return metrics
