# Real Data Interfaces

This module provides interfaces for loading real-world data into the CSEE-DML pipeline.

## Data Sources

### 1. City Information (`city_info.py`)
- **What**: City coordinates, administrative divisions, region classification
- **Source**: User-provided CSV at `data/city_info.csv`
- **Fallback**: 63 hardcoded major Chinese cities (for testing)
- **Template**: Run `python -c "from real_data.city_info import generate_template_csv; generate_template_csv()"` to generate a template CSV

### 2. Policy Pilot Lists (`policy_pilots.py`)
- **What**: Verified lists of climate-adaptive city pilots and sponge city pilots
- **Source**: Official government documents (hardcoded, verified)
  - 气候适应型城市建设试点 (28 cities, 2017) — 发改气候〔2017〕343号
  - 海绵城市建设试点 (30 cities, 2015-2016) — 财政部/住建部/水利部

### 3. MODIS NDVI (`gee_ndvi.py`)
- **What**: 16-day NDVI time series from MODIS MOD13A2 (1km, 2000-present)
- **Source**: Google Earth Engine
- **Auth**: Service account key JSON file
  ```bash
  export EE_SERVICE_ACCOUNT_KEY=/path/to/key.json
  ```
- **Usage**:
  ```python
  from real_data.gee_ndvi import initialize_gee, extract_ndvi_panel
  initialize_gee(service_account=True)
  ndvi_ts = extract_ndvi_panel(city_info_df, start_year=2005, end_year=2023)
  ```

### 4. ERA5 Weather Events (`era5_weather.py`)
- **What**: Extreme weather events (heat waves, heavy rain, drought)
- **Source**: ERA5 reanalysis via Copernicus CDS API
- **Auth**: `~/.cdsapirc` file with CDS API key
  ```
  url: https://cds.climate.copernicus.eu/api
  key: {your-uid}:{your-api-key}
  ```
- **Usage**:
  ```python
  from real_data.era5_weather import extract_weather_panel
  events = extract_weather_panel(city_info_df, start_year=2005, end_year=2023)
  ```

### 5. City Statistical Yearbook (`city_stats.py`)
- **What**: Socio-economic control variables (GDP, population, green rate, etc.)
- **Source**: Chinese City Statistical Yearbook (Excel/CSV)
- **Format**: Place files in `data/yearbook/yearbook_{year}.xlsx`
- **Auto-mapping**: Chinese column names are auto-mapped to English

## Usage

### Option A: Full real data pipeline
```bash
# 1. Set up credentials (GEE + CDS)
# 2. Place city_info.csv and yearbook files
python main.py --real-data
```

### Option B: Partial real data (with fallbacks)
```python
from real_data.data_loader import load_real_panel
# Missing sources use synthetic fallbacks
panel, ndvi_ts, events = load_real_panel()
```

### Option C: Extract raw data separately
```python
# Extract NDVI from GEE
from real_data.gee_ndvi import initialize_gee, extract_ndvi_panel
initialize_gee(service_account=True)
ndvi_ts = extract_ndvi_panel(city_info, save_path="data/ndvi_panel.pkl")

# Extract weather from ERA5
from real_data.era5_weather import extract_weather_panel
events = extract_weather_panel(city_info, save_path="data/weather_events.pkl")

# Then run pipeline
python main.py --real-data
```

## Data Pipeline Flow

```
city_info.csv ──→ city_info.py ──→ city metadata
                                      │
policy_pilots.py ──→ treatment assignment
                                      │
GEE API ──→ gee_ndvi.py ──→ NDVI time series ──┐
                                                ├──→ data_loader.py ──→ panel
CDS API ──→ era5_weather.py ──→ weather events ─┘         │
                                                          ↓
yearbook/*.xlsx ──→ city_stats.py ──→ control vars ──→ csee_computation
                                                          │
                                                          ↓
                                                     DML estimation
```

## File Structure

```
real_data/
  __init__.py
  city_info.py          # City metadata (coordinates, admin divisions)
  policy_pilots.py      # Verified pilot city lists
  gee_ndvi.py            # Google Earth Engine NDVI extraction
  era5_weather.py        # ERA5 extreme weather event extraction
  city_stats.py          # City statistical yearbook loader
  data_loader.py         # Main loader combining all sources
  README.md              # This file
```
