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
)


def run_dbscan_model(X_train, X_test, skip=False):
  if skip:
    return

  print("\n=== DBSCAN ===")
  model = DBSCANModel()
  model.train(X_train)
  model.evaluate(X_test)
  model.get_anomalies(X_test)
  model.get_clusters_info()


def run_sarima_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== SARIMA ===")
  model = SARIMAModel()
  model.train(X_train, y_train)


def run_random_forest_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== Random Forest ===")
  model = RandomForestModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)
  model.evaluate(X_test, y_test)


def run_svm_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== Support vector machine ===")
  model = SVMModel()
  model.train(X_train)
  model.evaluate(X_test, y_test)


def run_lstm_model(X_train, X_test, y_train, y_test, input_shape, skip=False):
  if skip:
    return

  print("\n=== LSTM ===")
  model = LSTMModel(input_shape=input_shape, output_shape=1)
  model.train(X_train, y_train, epochs=100)
  model.evaluate(X_test, y_test)


def run_mlp_model(X_train, X_test, y_train, y_test, skip=False):
  if skip:
    return

  print("\n=== MLP Classifier ===")
  model = MLPModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)


def main():
  df = pd.read_csv("data.csv")
  df = df.drop(columns=["SOLARRAD"])
  df.dropna(inplace=True)

  print("\n=== FIRST 5 ROWS ===")
  print(df.head())

  target_column = "AMBTEMP"
  numerical_features = [
    "COUGM3",
    "NO2UGM3",
    "O3UGM3",
    "PM25",
    "SO2UGM3",
    "UV_INDEX",
  ]

  df = handle_datetime(df)
  df = handle_rainfall(df)

  X = df.drop(columns=[target_column], axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=28
  )

  print(f"\nFinal shape: {X.shape}")

  fit_scaler(X_train[numerical_features])
  X_train[numerical_features] = transform_scaler(X_train[numerical_features])
  X_test[numerical_features] = transform_scaler(X_test[numerical_features])

  train_data = pd.concat([X_train, y_train], axis=1)
  test_data = pd.concat([X_test, y_test], axis=1)

  # Anomaly detector models
  run_dbscan_model(train_data, test_data, skip=False)
  run_sarima_model(X_train, X_test, y_train, y_test, skip=True)
  run_random_forest_model(X_train, X_test, y_train, y_test, skip=True)

  # Predictive models
  run_svm_model(X_train, X_test, y_train, y_test, skip=True)
  run_lstm_model(
    X_train,
    X_test,
    y_train,
    y_test,
    input_shape=(1, df.shape[1] - 1),
    skip=True,
  )
  run_mlp_model(X_train, X_test, y_train, y_test, skip=True)


if __name__ == "__main__":
  main()
