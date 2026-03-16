# Credit Card Fraud Detection — End‑to‑End Project

**Objective:** Detect fraudulent card transactions with a realistic, end‑to‑end pipeline (ingestion → features → training → scoring → monitoring → dashboard). Includes model explainability (SHAP-ready) and class‑imbalance strategies.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Prepare features & train model
make features
make train

# Score and explore dashboard
make score
streamlit run dashboard/app.py
```
Data: place Kaggle `creditcard.csv` in `data/raw/`.
