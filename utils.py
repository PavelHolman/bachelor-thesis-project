import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed/")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
