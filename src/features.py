from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Ensure time format
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # temporal feature
            df['year'] = df['DATE'].dt.year
            df['month'] = df['DATE'].dt.month
            df['day'] = df['DATE'].dt.day  
        
        #  Feature renaming and filtering

        rename_map = {
            'u10': 'U10M', 'u10m': 'U10M',
            'v10': 'V10M', 'v10m': 'V10M',
            't2m': 't2m',  '2t': 't2m',
            'sp': 'SP',
            'r': 'r',      'hum': 'r',
            'uvb': 'uvb',  'rad': 'uvb'
        }
        df = df.rename(columns=rename_map)
        
        # Final features

        target_features = [
            'Lon', 'Lat', 
            'NO2_obs', 'SO2_obs', 'PM2.5_obs', 
            't2m', 'U10M', 'V10M', 'SP', 'r', 'uvb', 
            'O3_gc', 
            'year', 'month', 'day'
        ]
        
        # Check and retain existing columns
        final_cols = [c for c in target_features if c in df.columns]
        
    
        if 'O3_obs' in df.columns:
            final_cols.append('O3_obs')
        elif 'O3' in df.columns:
            final_cols.append('O3')
            
        return df[final_cols]