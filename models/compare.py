import pandas as pd
from sklearn.model_selection import train_test_split

from gradient_boosting_model import GradientBoostingModel

def main():
  df = pd.read_csv('data.csv')
  df.fillna(df.mean(), inplace=True)
  target_column = 'AMBTEMP'

  X = df.drop(columns=[target_column], axis=1)
  y = df[target_column]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)
  df_train = pd.concat([X_train, y_train], axis=1)
  df_test = pd.concat([X_test, y_test], axis=1)
  
  # Run gradient boost model
  print("\n=== Gradient boost, kbest ===")
  model = GradientBoostingModel()
  model.train(df_train, target_column)
  model.evaluate(df_test, target_column)
  
if __name__ == "__main__":
  main()