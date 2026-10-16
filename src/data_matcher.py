import pandas as pd
import xarray as xr
import numpy as np
import os

def match_grid_to_stations(config):

    print("[-] [Match] Start spatiotemporal alignment (Strict Feature Mode)...")
    
    # 1. Read NC files (sort lat/lon to support interp)
    ds_gc = xr.open_dataset(config['gc_processed']).sortby(['lat', 'lon'])
    ds_era5 = xr.open_dataset(config['era5_processed']).sortby(['lat', 'lon'])
    
    ds_hum, ds_rad = None, None
    if 'hum_processed' in config and os.path.exists(config['hum_processed']):
        ds_hum = xr.open_dataset(config['hum_processed']).sortby(['lat', 'lon'])
    if 'rad_processed' in config and os.path.exists(config['rad_processed']):
        ds_rad = xr.open_dataset(config['rad_processed']).sortby(['lat', 'lon'])

    # 2. Read CSV
    print(f"    -> Read CSV: {config['obs_csv']}")
    obs_df = pd.read_csv(config['obs_csv'])
    obs_df.columns = [c.strip().lower() for c in obs_df.columns]
    
    # conversion date
    if 'date' in obs_df.columns:
        try:
            obs_df['DATE'] = pd.to_datetime(obs_df['date'].astype(str), format='%Y%m%d')
        except:
            obs_df['DATE'] = pd.to_datetime(obs_df['date'])
    
    matched_rows = []
    
    # Define variable list (ERA5 typically includes u10, v10, t2m, sp)
    era5_vars = list(ds_era5.data_vars)
    hum_vars = list(ds_hum.data_vars) if ds_hum else []
    rad_vars = list(ds_rad.data_vars) if ds_rad else [] # 应该是 uvb

    print("\n[-] Start line by line matching...")
    
    for idx, row in obs_df.iterrows():
        time = row['DATE']
        lon = row.get('lon', row.get('longitude'))
        lat = row.get('lat', row.get('latitude'))
        
        # === Extract CSV pollutants (Page 11 Table) ===
        row_dict = {
            'DATE': time, 'Lon': lon, 'Lat': lat,
            'O3_obs': row.get('o3'),
            'NO2_obs': row.get('no2'),
            'SO2_obs': row.get('so2'),
            'PM2.5_obs': row.get('pm2.5')
        }
        
        if pd.isna(row_dict['NO2_obs']) or pd.isna(row_dict['O3_obs']):
            continue

        # Longitude correction
        target_lon = lon + 360 if (ds_gc['lon'].max() > 180 and lon < 0) else lon

        # === Interpolation function (bilinear) ===
        def get_val(ds, vars):
            res = {}
            try:
                point = ds.interp(lon=target_lon, lat=lat, method='linear') \
                          .sel(date=time, method='nearest')
                for v in vars:
                    res[v] = point[v].item()
            except:
                for v in vars: res[v] = np.nan
            return res

        # (A) GEOS-Chem
        try:
            row_dict['O3_gc'] = ds_gc['O3_gc'].interp(lon=target_lon, lat=lat, method='linear') \
                                             .sel(date=time, method='nearest').item()
        except:
            row_dict['O3_gc'] = np.nan
            
        # (B) Meteorological variables (t2m, u10, v10, sp, r, uvb)
        row_dict.update(get_val(ds_era5, era5_vars))
        if ds_hum: row_dict.update(get_val(ds_hum, hum_vars))
        if ds_rad: row_dict.update(get_val(ds_rad, rad_vars))

        matched_rows.append(row_dict)

    df_clean = pd.DataFrame(matched_rows).dropna()
    print(f"\n[-] Matching completed, valid samples: {len(df_clean)}")
    df_clean.to_csv(config['matched_data'], index=False)
    return df_clean