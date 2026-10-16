from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

def train_model(X_train, y_train):
    print("[-] [Model] Start training XGBoost...")
    
    pipeline = make_pipeline(
        StandardScaler(),
        XGBRegressor(objective='reg:squarederror', n_jobs=1, random_state=42)
    )
    
    params = {
        'xgbregressor__n_estimators': [200, 400, 600],
        'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
        'xgbregressor__max_depth': [3, 6, 9],
        'xgbregressor__subsample': [0.7, 0.8, 0.9],
        'xgbregressor__colsample_bytree': [0.7, 0.8]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        pipeline, params, n_iter=10, cv=tscv, 
        scoring='r2', verbose=1, n_jobs=1
    )
    
    search.fit(X_train, y_train)
    print(f"    best params: {search.best_params_}")
    return search.best_estimator_