from pathlib import Path
import pandas as pd
import numpy as np
from config import DATA_DIR, FEATURE_STORE

INTERIM = Path(DATA_DIR) / "interim" / "transactions.csv"

# The Kaggle dataset has 'Time', 'Amount', V1..V28, 'Class' (0/1)

def build_features() -> pd.DataFrame:
    df = pd.read_csv(INTERIM)
    if 'Class' in df.columns:
        df.rename(columns={"Class": "is_fraud"}, inplace=True)

    # Time features
    if 'Time' in df.columns:
        df["tx_seconds"] = df["Time"].astype(float)
    else:
        df["tx_seconds"] = range(len(df))
    df["tx_hour"] = (df["tx_seconds"] // 3600 % 24).astype(int)
    df["is_night"] = ((df["tx_hour"] <= 6) | (df["tx_hour"] >= 22)).astype(int)

    # Amount features
    if 'Amount' in df.columns:
        df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-6)
    else:
        df["Amount"] = 0.0
        df["amount_zscore"] = 0.0

    # Velocity (synthetic user_id for demo)
    if 'device_id' not in df.columns:
        df['device_id'] = 1000
    df["user_id"] = (df["device_id"].astype(int) % 5000)
    df.sort_values(["user_id", "tx_seconds"], inplace=True)
    df["user_txn_count_1h"] = (
        df.groupby("user_id")["tx_seconds"].rolling(50).count().reset_index(0,drop=True)
    )

    # Encode categoricals
    cat_cols = [c for c in ["channel","ip_country","merchant_category"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    # Target
    if 'is_fraud' not in df.columns:
        df['is_fraud'] = 0

    Path(FEATURE_STORE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_STORE, index=False)
    return df

if __name__ == "__main__":
    out = build_features()
    print(f"Wrote features to {FEATURE_STORE} with shape={out.shape}")
