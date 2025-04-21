import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from transformers import get_pipeline

class GradientBoostingModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('cleaning', get_pipeline()),
            ('model', GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=3,
                random_state=28
            ))
        ])
        self.model_path = f"gb_model.pkl"

    def train(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, self.model_path)

    def predict(self, features: list):
        model = joblib.load(self.model_path)
        return model.predict(np.array(features))

    def evaluate(self, df: pd.DataFrame, target_column: str):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = joblib.load(self.model_path)
        y_pred = model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }

        print(f"Mean Squared Error: {metrics['mse']:.2f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.2f}")
        print(f"Mean Absolute Error: {metrics['mae']:.2f}")
        print(f"R-squared: {metrics['r2']:.2f}")
        
        return metrics
