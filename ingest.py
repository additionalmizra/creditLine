from pathlib import Path
import pandas as pd
from config import DATA_DIR

RAW = Path(DATA_DIR) / "raw" / "creditcard.csv"
INTERIM = Path(DATA_DIR) / "interim" / "transactions.csv"

def load_raw() -> pd.DataFrame:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing dataset: {RAW}. Add Kaggle 'creditcard.csv'.")
    df = pd.read_csv(RAW)
    return df

def synth_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    n = len(df)
    df = df.copy()
    df["channel"] = np.where(np.random.rand(n) > 0.7, "POS", "ECOM")
    df["device_id"] = (np.random.randint(1000, 9999, n)).astype(str)
    df["ip_country"] = np.random.choice(["US","CA","GB","DE","FR","IN"], n)
    df["merchant_category"] = np.random.choice(["grocery","electronics","fashion","fuel","gaming"], n)
    df["distance_home"] = (abs(np.random.randn(n) * 10)).astype(float)
    df["previous_declines_24h"] = np.random.poisson(0.1, n)
    return df

def write_interim(df: pd.DataFrame):
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM, index=False)

if __name__ == "__main__":
    df = load_raw()
    df = synth_enrichment(df)
    write_interim(df)
    print(f"Wrote interim dataset: {INTERIM}")
