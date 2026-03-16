"""
Generate a synthetic credit card dataset mimicking the Kaggle creditcard.csv format.
Columns: Time, V1..V28, Amount, Class (0=legit, 1=fraud)
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 10000          # total transactions
FRAUD_RATE = 0.017 # ~1.7 % fraud (realistic)

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

# Time: seconds elapsed (simulate 2 days)
time_col = np.sort(np.random.uniform(0, 172800, N))

# V1..V28: PCA-like features (random normal, fraud shifted)
v_legit = np.random.randn(n_legit, 28)
v_fraud = np.random.randn(n_fraud, 28) + np.random.uniform(-2, 2, (1, 28))

V = np.vstack([v_legit, v_fraud])
labels = np.array([0]*n_legit + [1]*n_fraud)

# Shuffle
idx = np.random.permutation(N)
V = V[idx]
labels = labels[idx]

# Amount
amount_legit = np.abs(np.random.lognormal(3, 1.5, N))
amount_legit = np.round(amount_legit, 2)

df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", time_col)
df["Amount"] = amount_legit
df["Class"] = labels

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "creditcard.csv", index=False)
print(f"Synthetic dataset saved to {out / 'creditcard.csv'}  ({len(df)} rows, fraud rate={labels.mean():.3f})")
