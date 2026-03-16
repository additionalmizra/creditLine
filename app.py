import pandas as pd
import plotly.express as px
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Fraud Ops", layout="wide")

data_path = Path("data/processed/scores.parquet")
if not data_path.exists():
    st.error("Run `make score` first to generate scores.")
    st.stop()

df = pd.read_parquet(data_path)

st.title("Credit Card Fraud Detection — Ops Dashboard")

thr = st.slider("Alert threshold", 0.0, 1.0, 0.9, 0.01)
df["alert"] = (df["fraud_score"] >= thr).astype(int)

col1, col2, col3 = st.columns(3)
with col1: st.metric("Total Txns", len(df))
with col2: st.metric("Alerts", int(df["alert"].sum()))
with col3: st.metric("Alert Rate", f"{100*df['alert'].mean():.2f}%")

fig = px.histogram(df, x="fraud_score", nbins=50, title="Fraud Score Distribution")
st.plotly_chart(fig, use_container_width=True)

if 'merchant_category' in df.columns:
    by_cat = df.groupby("merchant_category")["alert"].mean().reset_index().sort_values("alert", ascending=False)
    fig2 = px.bar(by_cat, x="merchant_category", y="alert", title="Alert Rate by Merchant Category")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Alerts")
st.dataframe(df.sort_values("fraud_score", ascending=False).head(200))
