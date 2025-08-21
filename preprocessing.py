import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np

class CLVDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'clv_6month'
        
    def fit_transform(self, df):
        # Feature engineering
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['customer_duration'] = (df['last_purchase_date'] - df['first_purchase_date']).dt.days
        df['purchase_intensity'] = df['total_purchases'] / df['customer_duration']
        
        # Define features
        numeric_features = ['age', 'total_purchases', 'total_spend', 
                          'avg_purchase_value', 'purchase_frequency_days',
                          'customer_duration', 'purchase_intensity']
        categorical_features = ['gender', 'income_tier', 'segment']
        self.feature_columns = numeric_features + categorical_features
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
            
        X = self.preprocessor.fit_transform(df[self.feature_columns])
        y = df[self.target_column].values
        
        return X, y, df['segment'].values
        
    def transform(self, df):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet")
        return self.preprocessor.transform(df[self.feature_columns])
