import sys
import os
import joblib
import numpy as np
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import CONFIG
from src.preprocessing import process_geoschem, process_era5
from src.data_matcher import match_grid_to_stations
from src.features import FeatureEngineer
from src.model import train_model
from src.analysis import evaluate_model

def main():
    print("=== GBA O3 Deviation correction (Strict Paper Replication) ===")
    
    # 1. preprocessing 
    process_geoschem(CONFIG['raw_gc'], CONFIG['gc_processed'])
    process_era5(CONFIG['raw_era5'], CONFIG['era5_processed'])
    if os.path.exists(CONFIG['raw_hum']): process_era5(CONFIG['raw_hum'], CONFIG['hum_processed'])
    if os.path.exists(CONFIG['raw_rad']): process_era5(CONFIG['raw_rad'], CONFIG['rad_processed'])
    
    # 2. match
    df = match_grid_to_stations(CONFIG)
    if df.empty: return

    # 3. wash
    target_col = 'O3_obs'
    df = df[(np.abs(stats.zscore(df[target_col])) < 3)]
    
    # 4.Strict Mode
    print("[-] Application feature selection (Page 11 Table)...")
    engineer = FeatureEngineer()
    df_processed = engineer.fit_transform(df)
    
    print(f"[-] Final feature list ({len(df_processed.columns)-1} 个):")
    print([c for c in df_processed.columns if c != target_col])
    
    # 5. Random shuffling and partitioning
    df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 6. Training and Preservation
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, CONFIG['model_save'])

if __name__ == "__main__":
    main()