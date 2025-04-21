import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # Calculate feature importance using correlation with target
        if y is not None:
            self.feature_importance = abs(X.corrwith(y))
            self.selected_features = self.feature_importance[self.feature_importance >= self.threshold].index
        return self
        
    def transform(self, X):
        return X[self.selected_features]

def get_preprocessing_pipeline():
    return Pipeline([
        # ('feature_selector', FeatureSelector()),
        ('scaler', StandardScaler())
    ])