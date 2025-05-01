import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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


class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()

    # Transform datetime column
    X["DATETIME"] = pd.to_datetime(X["DATETIME"], format="%d-%b-%Y %H:%M")
    X = X.assign(Year=X["DATETIME"].dt.year, Month=X["DATETIME"].dt.month, Hour=X["DATETIME"].dt.hour)
    X = X.drop(columns=["DATETIME"])

    # quitar la variable de radiacion solar
    X = X.drop(columns=["SOLARRAD"])

    X["HAS_RAINFALL"] = X["RAINFALL"].apply(lambda x: 1 if x > 0 else 0)
    return X


def clean_pipeline():
  return Pipeline(
    [
      ("negative_fixer", NegativeValueHandler()),
      ("weather_engineer", WeatherFeatureEngineer()),
      ("imputer", SimpleImputer(strategy="mean")),
      # TODO: implementar un pca
      ("scaler", StandardScaler()),
    ]
  )
