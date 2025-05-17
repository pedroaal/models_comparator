import pandas as pd
from sklearn.model_selection import train_test_split

from models import SARIMAModel, LSTMModel, MLPModel


def main():
  df = pd.read_csv("data.csv")
  df = df.drop(columns=["SOLARRAD"])
  df.dropna(inplace=True)

  print("\n=== FIRST 5 ROWS ===")
  print(df.head())

  target_column = "AMBTEMP"

  X = df.drop(columns=[target_column], axis=1)
  y = df[target_column]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)
  df_train = pd.concat([X_train, y_train], axis=1)
  df_test = pd.concat([X_test, y_test], axis=1)

  # # Run svm model
  # print("\n=== Support vector machine ===")
  # model = SVMModel()
  # model.train(df_train, target_column)
  # model.evaluate(df_test, target_column)

  # # Run gradient boost model
  # print("\n=== Gradient boost ===")
  # model = GradientBoostingModel()
  # model.train(df_train, target_column)
  # model.evaluate(df_test, target_column)

  # # Run random forest model
  # print("\n=== Random Forest ===")
  # model = RandomForestModel()
  # model.train(df_train, target_column)
  # model.evaluate(df_test, target_column)

  # # Run SARIMA model
  # print("\n=== SARIMA ===")
  # model = SARIMAModel()
  # model.train(df_train, target_column)
  # model.evaluate(df_test, target_column)

  # # Run LSTM model
  # print("\n=== LSTM ===")
  # model = LSTMModel(input_shape=(1, df.shape[1] - 1), output_shape=1)
  # model.train(df_train, target_column, epochs=100)
  # model.evaluate(df_test, target_column)

  # Run MLP Classifier model
  print("\n=== MLP Classifier ===")
  model = MLPModel()
  model.train(df_train, target_column)
  model.evaluate(df_test, target_column)


if __name__ == "__main__":
  main()
