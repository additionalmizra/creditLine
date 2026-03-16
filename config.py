import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
FEATURE_STORE = os.getenv("FEATURE_STORE", "feature_store/offline_features.parquet")
SEED = int(os.getenv("SEED", 42))
