import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def handle_negative_values(df):
  df_tmp = df.copy()
  pollutant_cols = ["COUGM3", "NO2UGM3", "O3UGM3", "PM25", "SO2UGM3"]
  df_tmp[pollutant_cols] = df_tmp[pollutant_cols].clip(lower=0)
  return df_tmp


def handle_datetime(df, format="%d-%b-%Y %H:%M", date_index=False):
  df_tmp = df.copy()
  df_tmp["DATETIME"] = pd.to_datetime(df_tmp["DATETIME"], format=format)
  df_tmp = df_tmp.assign(
    YEAR=df_tmp["DATETIME"].dt.year,
    MONTH=df_tmp["DATETIME"].dt.month,
    HOUR=df_tmp["DATETIME"].dt.hour,
  )

  # hora como variable cíclica (0 a 23)
  df_tmp["HOUR_SIN"] = np.sin(2 * np.pi * df_tmp["HOUR"] / 24)
  df_tmp["HOUR_COS"] = np.cos(2 * np.pi * df_tmp["HOUR"] / 24)

  # mes como variable cíclica (1 a 12)
  df_tmp["MONTH_SIN"] = np.sin(2 * np.pi * df_tmp["MONTH"] / 12)
  df_tmp["MONTH_COS"] = np.cos(2 * np.pi * df_tmp["MONTH"] / 12)

  if date_index:
    df_tmp.set_index("DATETIME", inplace=True)
  else:
    df_tmp = df_tmp.drop(columns=["DATETIME"])

  return df_tmp


def handle_rainfall(df):
  df_tmp = df.copy()

  df_tmp["HAS_RAINFALL"] = df_tmp["RAINFALL"].apply(lambda x: 1 if x > 0 else 0)
  df_tmp = df_tmp.drop(columns=["RAINFALL"])

  return df_tmp


def handle_uv(df):
  df_tmp = df.copy()
  df_tmp = df_tmp[df_tmp["UV_INDEX"] > 0]

  return df_tmp


def handle_window(df, window_size=12, target_col=None):
  df_tmp = df.copy()

  if target_col is not None:
    target_idx = df.columns.get_loc(target_col)
    return np.array(
      [df_tmp.iloc[i + window_size, target_idx] for i in range(len(df_tmp) - window_size)]
    )

  return np.array(
    [df_tmp.iloc[i : i + window_size].values for i in range(len(df_tmp) - window_size)]
  )


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
