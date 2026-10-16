# Input data

The original data archive and derived NetCDF/CSV datasets are not distributed in this repository. Their redistribution terms and exact download provenance were not recorded in the supplied project folder. Obtain the inputs through their original providers and check their terms before use. The project report identifies **ERA5/ECMWF**, **GEOS-Chem**, and regional ground-monitoring observations as the underlying sources; this repository does not provide provider credentials or claim to recreate the original downloads.

Place compatible prepared inputs here, or set `OZONE_DATA_DIR` to an existing data directory. Run `python scripts/check_data.py` to check file presence.

| Filename | Required | Expected contents |
| --- | --- | --- |
| `daily_mean.nc` | Yes | Daily, near-surface GEOS-Chem ozone; variable `SpeciesConcVV_O3` (fallbacks `O3`, `o3`, `SpeciesConc_O3`); one value per time/latitude/longitude grid cell. |
| `regrided.nc` | Yes | Prepared hourly ERA5 fields on the analysis grid; time coordinate `valid_time` or `time`; latitude/longitude coordinates `lat`/`lon` or `latitude`/`longitude`. |
| `gba_pollutants_jan2020_complete.csv` | Yes | Station-day observations with `date`, `lat`, `lon`, `o3`, `no2`, `so2`, `pm2.5`; column names are normalized to lowercase. `latitude`/`longitude` are accepted alternatives. Dates may use `YYYYMMDD` or a pandas-readable date format. |
| `humidity_regrid.nc` | Optional | Hourly near-surface humidity, normally `r` (or feature alias `hum`), with compatible coordinates. |
| `radiation_regrid.nc` | Optional | Hourly UV radiation, normally `uvb` (or feature alias `rad`), with compatible coordinates. |

The supplied research files cover January 2020. GEOS-Chem's input may already use the daily `date` coordinate. ERA5 preprocessing expects time-indexed fields and takes daily means. For the published workflow, input NetCDF fields have already been prepared on the study grid; **the code does not download data or implement the original coarse-grid-to-0.01-degree preparation step**.

## Variables and units

- The meteorology feature aliases are `t2m`/`2t`, `u10`/`u10m`, `v10`/`v10m`, `sp`, `r`/`hum`, and `uvb`/`rad`.
- `src/preprocessing.py` retains the original ozone conversion `value * 1.18 * 1e9`, which assumes a **mass mixing ratio in kg/kg**, a fixed air density of 1.18 kg/m3, and a target concentration in micrograms/m3. The variable name alone does not establish the units. The supplied `daily_mean.nc` does not contain a units attribute for `SpeciesConcVV_O3`; confirm the preparation history before using this conversion. Volume/molar mixing ratios require a different conversion.
- Pollutant observation units must agree with the converted ozone target. Meteorological units must remain consistent across training and evaluation.
- Time matching uses nearest dates; spatial matching uses bilinear interpolation. Unmatched or incomplete rows are dropped. The CSV must contain real latitude/longitude values, not projected coordinates.
- Humidity and radiation are optional in `main.py`, but the complete `generate.py` figure suite includes UVB dependence plots. Provide both auxiliary datasets to reproduce that complete suite.

## Data privacy and cleanup

Raw files, generated caches and trained models are ignored by Git. When changing input datasets, remove the project's processed files in `cache/` before rerunning: preprocessing reuses existing cached files by filename. The raw-data folder remains untouched.

