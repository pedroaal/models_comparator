import pandas as pd
from sklearn.model_selection import train_test_split


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
  handle_uv,
)


def run_dbscan_model(data, skip=False):
  if skip:
    return

  print("\n=== DBSCAN ===")
  model = DBSCANModel()
  model.train(data)
  model.evaluate(data)
  model.get_anomalies(data)
  model.get_clusters_info()
  model.plot_results(data)


def run_sarima_model(
  X_train, y_train, X_test, y_test, target_column, skip=False
):
  if skip:
    return

  print("\n=== SARIMA ===")
  model = SARIMAModel(target_column)
  # model.find_sarima_parameters(dataset)
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)
  model.plot_results()


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


def run_lstm_model(
  X_train, X_test, y_train, y_test, window_size=24, skip=False
):
  if skip:
    return

  print("\n=== LSTM ===")
  model = LSTMModel(window_size, X_train.shape[1])
  model.train(X_train, y_train, epochs=100)
  model.evaluate(X_test, y_test)


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
  numerical_features = [
    "AMBTEMP",
    "COUGM3",
    "NO2UGM3",
    "O3UGM3",
    "PM25",
    "SO2UGM3",
  ]

  df = handle_datetime(df)
  # df = handle_datetime(df, remove_date=False)
  df = handle_rainfall(df)
  df = handle_uv(df)  # only for sarima

  print("\n=== Data engineering dataset ===")
  print(df.head())

  X = df.drop(columns=[target_column], axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28
  )

  fit_scaler(X_train[numerical_features])
  X_train[numerical_features] = transform_scaler(X_train[numerical_features])
  X_test[numerical_features] = transform_scaler(X_test[numerical_features])

  df_scaled = df.copy()
  df_scaled[numerical_features] = transform_scaler(
    df_scaled[numerical_features]
  )

  # Anomaly detector models
  run_dbscan_model(df_scaled, skip=True)
  run_sarima_model(X_train, y_train, X_test, y_test, target_column, skip=False)

  # Predictive models
  run_random_forest_model(X_train, X_test, y_train, y_test, skip=True)
  run_svm_model(X_train, X_test, y_train, y_test, skip=True)
  run_lstm_model(
    X_train,
    X_test,
    y_train,
    y_test,
    window_size=24,
    skip=True,
  )
  run_mlp_model(X_train, X_test, y_train, y_test, skip=True)


if __name__ == "__main__":
  main()
