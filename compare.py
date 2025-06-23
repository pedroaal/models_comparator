import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


from models import (
  DBSCANModel,
  SARIMAModel,
  RandomForestModel,
  SVMModel,
  LSTMModel,
  MLPModel,
  fit_scaler,
  transform_scaler,
  handle_datetime,
  handle_rainfall,
  handle_window,
)


def run_dbscan_model(X, X_series, X_scaled, skip=False):
  if skip:
    return

  print("\n=== DBSCAN ===")
  model = DBSCANModel()
  model.train(X_scaled)
  model.evaluate(X_scaled)
  model.plot_anomalies(X)
  model.plot_pca(X_scaled)
  model.plot_window(X_series)
  model.get_anomalies(X)


def run_sarima_model(X, y, skip=False):
  if skip:
    return

  print("\n=== SARIMA ===")
  model = SARIMAModel(period=12)
  # model.find_sarima_parameters(dataset)
  model.train(X, y)
  model.evaluate(X, y)
  model.plot_results(X, y)


def run_random_forest_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== Random Forest ===")
  model = RandomForestModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)
  model.evaluate(X_test, y_test)
  model.plot_results(X_test, y_test)


def run_svm_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== Support vector machine ===")
  model = SVMModel()
  # model.get_best_estimator(
  #   pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
  # )
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)
  model.plot_results(X_test, y_test)


def run_lstm_model(X_train, X_test, y_train, y_test, scaler, skip=False):
  if skip:
    return

  print("\n=== LSTM ===")
  features = int(X_train.shape[2])
  model = LSTMModel(features, window_size=24)
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test, scaler)
  model.plot_results(X_test, y_test, scaler)


def run_mlp_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== MLP Classifier ===")
  model = MLPModel()
  # model.get_best_estimator(
  #   pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
  # )
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)
  model.plot_results(X_test, y_test)


def main():
  df = pd.read_csv("data.csv")
  df.dropna(inplace=True)

  print("\n=== Initial dataset ===")
  print(df.head())

  target_column = "UV_INDEX"

  df = handle_datetime(df)
  # df = handle_datetime(df, date_index=True)  # for sarima
  df = handle_rainfall(df)
  # df = handle_uv(df)  # remove where uv_index is 0

  print("\n=== Data engineering dataset ===")
  print(df.head())

  X = df.drop(columns=[target_column], axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

  fit_scaler(X_train)
  X_train = transform_scaler(X_train)
  X_test = transform_scaler(X_test)

  run_random_forest_model(X_train, X_test, y_train, y_test, skip=True)
  run_svm_model(X_train, X_test, y_train, y_test, skip=True)
  run_mlp_model(X_train, X_test, y_train, y_test, skip=True)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28, shuffle=False)

  X_scaler = RobustScaler()
  X_train_scaled = X_scaler.fit_transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)

  y_scaler = RobustScaler()
  y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
  y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

  y_series = handle_window(y)  # for dbscan
  y_series_scaled = StandardScaler().fit_transform(y_series)  # for dbscan

  # Anomaly detector models
  run_dbscan_model(y, y_series, y_series_scaled, skip=True)
  run_sarima_model(X, y, skip=True)

  # only for ltsm
  X_train_series = handle_window(pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index), window_size=24)
  y_train_series = handle_window(pd.DataFrame(y_train_scaled, index=y_train.index), window_size=24, target_column=True)
  X_test_series = handle_window(pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index), window_size=24)
  y_test_series = handle_window(pd.DataFrame(y_test_scaled, index=y_test.index), window_size=24, target_column=True)

  run_lstm_model(
    X_train_series,
    X_test_series,
    y_train_series,
    y_test_series,
    scaler=y_scaler,
    skip=False,
  )


if __name__ == "__main__":
  main()
