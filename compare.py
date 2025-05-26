import pandas as pd
from sklearn.model_selection import train_test_split


from models import (
  GradientBoostingModel,
  RandomForestModel,
  MLPModel,
  fit_scaler,
  transform_scaler,
  handle_datetime,
  handle_rainfall,
)


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

  fit_scaler(X_train[numerical_features])
  X_train[numerical_features] = transform_scaler(X_train[numerical_features])
  X_test[numerical_features] = transform_scaler(X_test[numerical_features])

  # # Run svm model
  # print("\n=== Support vector machine ===")
  # model = SVMModel()
  # model.train(X_train, y_train)
  # model.evaluate(X_test, y_test)

  # # Run gradient boost model
  print("\n=== Gradient boost ===")
  model = GradientBoostingModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)

  # # Run random forest model
  print("\n=== Random Forest ===")
  model = RandomForestModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)

  # # Run SARIMA model
  # print("\n=== SARIMA ===")
  # model = SARIMAModel()
  # model.train(X_train, y_train)
  # model.evaluate(X_test, y_test)

  # # Run LSTM model
  # print("\n=== LSTM ===")
  # model = LSTMModel(input_shape=(1, df.shape[1] - 1), output_shape=1)
  # model.train(X_train, y_train, epochs=100)
  # model.evaluate(X_test, y_test)

  # Run MLP Classifier model
  print("\n=== MLP Classifier ===")
  model = MLPModel()
  model.train(X_train, y_train)
  model.evaluate(X_test, y_test)


if __name__ == "__main__":
  main()
