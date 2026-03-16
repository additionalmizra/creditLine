import joblib
from pathlib import Path
import pandas as pd
from config import FEATURE_STORE, MODEL_DIR

OUT = Path("data/processed/scores.parquet")

if __name__ == "__main__":
    df = pd.read_parquet(FEATURE_STORE)
    model = joblib.load(Path(MODEL_DIR) / "rf_model.joblib")
    proba = model.predict_proba(df.drop(columns=["is_fraud"]))[:,1]
    cols = [c for c in ["Amount","tx_hour","channel","ip_country","merchant_category","is_night","is_fraud"] if c in df.columns]
    df_out = df[cols].copy()
    df_out["fraud_score"] = proba
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT, index=False)
    print(f"Wrote scores to {OUT}")
