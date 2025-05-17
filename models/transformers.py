import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class NegativeValueHandler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    pollutant_cols = ["COUGM3", "NO2UGM3", "O3UGM3", "PM25", "SO2UGM3"]
    X[pollutant_cols] = X[pollutant_cols].clip(lower=0)
    return X


class DatetimeHandler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()

    X["DATETIME"] = pd.to_datetime(X["DATETIME"], format="%d-%b-%Y %H:%M")
    X = X.assign(Year=X["DATETIME"].dt.year, Month=X["DATETIME"].dt.month, Hour=X["DATETIME"].dt.hour)
    X = X.drop(columns=["DATETIME"])

    return X


class RainfallHandler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()

    X["HAS_RAINFALL"] = X["RAINFALL"].apply(lambda x: 1 if x > 0 else 0)
    X = X.drop(columns=["RAINFALL"])

    return X


def clean_pipeline():
  return Pipeline(
    [
      ("datetime_handler", DatetimeHandler()),
      ("rainfall_handler", RainfallHandler()),
      # TODO: implementar un pca
      ("scaler", StandardScaler()),
    ]
  )
