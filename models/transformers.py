from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class NegativeValueHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        pollutant_cols = ['COUGM3', 'NO2UGM3', 'O3UGM3', 'PM25', 'SO2UGM3']
        X[pollutant_cols] = X[pollutant_cols].clip(lower=0)
        return X

class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['HAS_RAINFALL'] = X['RAINFALL'].apply(lambda x: 1 if x > 0 else 0)
        return X

def clean_pipeline():
    return Pipeline([
        ('negative_fixer', NegativeValueHandler()),
        ('weather_engineer', WeatherFeatureEngineer()),
        ('imputer', SimpleImputer(strategy='mean')),
        # todo: quitar la varialble de radiacion solar
        # todo: agregar la columna date-time separando el mes, ano, hora
        # todo: implementar un pca
        ('scaler', StandardScaler())
    ])