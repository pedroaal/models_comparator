import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


def handle_negative_values(df):
  new_df = df.copy()
  pollutant_cols = ["COUGM3", "NO2UGM3", "O3UGM3", "PM25", "SO2UGM3"]
  new_df[pollutant_cols] = new_df[pollutant_cols].clip(lower=0)
  return new_df


def handle_datetime(df):
  new_df = df.copy()
  new_df["DATETIME"] = pd.to_datetime(
    new_df["DATETIME"], format="%d-%b-%Y %H:%M"
  )
  new_df = new_df.assign(
    YEAR=new_df["DATETIME"].dt.year,
    MONTH=new_df["DATETIME"].dt.month,
    HOUR=new_df["DATETIME"].dt.hour,
  )

  # hora como variable cíclica (0 a 23)
  new_df["HOUR_SIN"] = np.sin(2 * np.pi * new_df["HOUR"] / 24)
  new_df["HOUR_COS"] = np.cos(2 * np.pi * new_df["HOUR"] / 24)

  # mes como variable cíclica (1 a 12)
  new_df["MONTH_SIN"] = np.sin(2 * np.pi * new_df["MONTH"] / 12)
  new_df["MONTH_COS"] = np.cos(2 * np.pi * new_df["MONTH"] / 12)

  new_df = new_df.drop(columns=["DATETIME"])

  return new_df


def handle_rainfall(df):
  new_df = df.copy()

  new_df["HAS_RAINFALL"] = new_df["RAINFALL"].apply(lambda x: 1 if x > 0 else 0)
  new_df = new_df.drop(columns=["RAINFALL"])

  return new_df


# Fit and save the scaler
def fit_scaler(df, path="scaler.joblib"):
  scaler = RobustScaler()
  scaler.fit(df)
  joblib.dump(scaler, path)

  return scaler


# Load and use the scaler
def transform_scaler(df, path="scaler.joblib"):
  scaler = joblib.load(path)

  return scaler.transform(df)


class CustomTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = handle_datetime(X)
    X = handle_rainfall(X)
    return X


def clean_pipeline():
  return Pipeline(
    [
      ("pca", PCA(n_components=0.95)),
    ]
  )
