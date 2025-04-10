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

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim/")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed/")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Environment variables

load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
