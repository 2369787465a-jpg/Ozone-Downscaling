import xarray as xr
import os

def process_geoschem(input_path, output_path):
    """
    1.1: GEOS-Chem unit conversion (kg/kg -> ug/m3)
    """
    if os.path.exists(output_path):
        print(f"[-] [Cache] GEOS Chem processed: {os.path.basename(output_path)}")
        return

    print(f"[-] [Preprocessing] Processing GEOS Chem: {os.path.basename(input_path)}")
    ds = xr.open_dataset(input_path)
    
    # 1. Variable name adaptation
    var_name = 'SpeciesConcVV_O3'
    if var_name not in ds:
        for v in ['O3', 'o3', 'SpeciesConc_O3']:
            if v in ds:
                var_name = v
                break
    
    # 2. unit conversion
    rho_air = 1.18 
    ds_new = ds[[var_name]].copy()
    ds_new = ds_new.rename({var_name: 'O3_gc'})
    ds_new['O3_gc'] = ds_new['O3_gc'] * rho_air * 1e9
    
    # 3. Coordinate standardization
    rename_dict = {}
    if 'time' in ds_new.coords: rename_dict['time'] = 'date'
    if 'longitude' in ds_new.coords: rename_dict['longitude'] = 'lon'
    if 'latitude' in ds_new.coords: rename_dict['latitude'] = 'lat'
    
    if rename_dict:
        ds_new = ds_new.rename(rename_dict)
    
    ds_new.to_netcdf(output_path)
    print(f"    -> Save completed ")

def process_era5(input_path, output_path):
    """
    1.2:Downsampling of meteorological data (Hourly -> Daily Mean)
    """
    if os.path.exists(output_path):
        print(f"[-] [Cache] Meteorological data processed: {os.path.basename(output_path)}")
        return

    print(f"[-] [Preprocessing] Calculate daily average: {os.path.basename(input_path)}")
    # Block reading
    ds = xr.open_dataset(input_path, chunks={'valid_time': 24})
    
    # 1. Coordinate standardization
    rename_dict = {}
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'valid_time' in ds.coords: rename_dict['valid_time'] = 'time'
    
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    # 2. Resample daily and calculate the average
    ds_daily = ds.resample(time='1D').mean()
    
    # 3. Change back to date to match
    ds_daily = ds_daily.rename({'time': 'date'})
    
    # Calculate and save immediately
    ds_daily.to_netcdf(output_path)
    print(f"    -> Save completed ")