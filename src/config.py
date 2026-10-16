import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get('OZONE_DATA_DIR', os.path.join(BASE_DIR, 'data'))
))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG = {
    #Raw input (using large files)
    'raw_gc': os.path.join(DATA_DIR, 'daily_mean.nc'),
    'raw_era5': os.path.join(DATA_DIR, 'regrided.nc'),
    'raw_hum': os.path.join(DATA_DIR, 'humidity_regrid.nc'),
    'raw_rad': os.path.join(DATA_DIR, 'radiation_regrid.nc'),
    'obs_csv': os.path.join(DATA_DIR, 'gba_pollutants_jan2020_complete.csv'),

    #Intermediate processing results
    'gc_processed': os.path.join(CACHE_DIR, 'gc_daily_ugm3.nc'),
    'era5_processed': os.path.join(CACHE_DIR, 'era5_daily_mean.nc'),
    'hum_processed': os.path.join(CACHE_DIR, 'hum_daily_mean.nc'),
    'rad_processed': os.path.join(CACHE_DIR, 'rad_daily_mean.nc'),
    
    'matched_data': os.path.join(CACHE_DIR, 'final_dataset_full.csv'),
    'model_save': os.path.join(MODEL_DIR, 'bias_correction_model_full.pkl')
}
