import os
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directories

MODEL_DIR = os.path.join(ROOT_DIR, 'models/')

INTRADAY_MODEL_DIR = os.path.join(MODEL_DIR, 'intraday/')
DAY_AHEAD_MODEL_DIR = os.path.join(MODEL_DIR, 'day_ahead/')

os.makedirs(INTRADAY_MODEL_DIR, exist_ok=True)
os.makedirs(DAY_AHEAD_MODEL_DIR, exist_ok=True)

# Data directories

DATA_DIR = os.path.join(ROOT_DIR, "data/")

# Raw data directories

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/")

## OTE data

OTE_DATA_DIR = os.path.join(RAW_DATA_DIR, "ote/")

DAY_AHEAD_DIR = os.path.join(OTE_DATA_DIR, "day_ahead/")
INTRA_DAY_DIR = os.path.join(OTE_DATA_DIR, "intra_day/")

os.makedirs(DAY_AHEAD_DIR, exist_ok=True)
os.makedirs(INTRA_DAY_DIR, exist_ok=True)

## entso-e data

ENTSOE_DATA_DIR = os.path.join(RAW_DATA_DIR, "entsoe/")

GENERATION_FORECAST_DIR = os.path.join(ENTSOE_DATA_DIR, "generation_forecast/")
WIND_AND_SOLAR_FORECAST_DIR = os.path.join(ENTSOE_DATA_DIR, "wind_and_solar_forecast/")
LOAD_FORECAST_DIR = os.path.join(ENTSOE_DATA_DIR, "load_forecast/")

os.makedirs(GENERATION_FORECAST_DIR, exist_ok=True)
os.makedirs(WIND_AND_SOLAR_FORECAST_DIR, exist_ok=True)
os.makedirs(LOAD_FORECAST_DIR, exist_ok=True)

## open-meteo data

OPEN_METEO_DATA_DIR = os.path.join(RAW_DATA_DIR, "open_meteo/")

WEATHER_DATA_DIR = os.path.join(OPEN_METEO_DATA_DIR, "weather/")

os.makedirs(WEATHER_DATA_DIR, exist_ok=True)

# Processed data directories

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed/")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Environment variables

load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
