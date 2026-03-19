

An end-to-end machine learning system to predict **credit default risk** for loan applicants — covering data ingestion, exploratory analysis, feature engineering, model training, and a deployed web application.

---

 Problem Statement

Financial institutions lose billions annually to loan defaults. This project builds a predictive pipeline to assess the **probability of credit default** for a given applicant, enabling data-driven lending decisions.



 Project Architecture

creditLine/
│
├── data/ # Raw and processed datasets
├── feature_store/ # Feature engineering pipeline & versioned features
├── models/ # Trained and serialized ML models
├── 01_eda.ipynb # Exploratory Data Analysis notebook
├── app.py # Streamlit/Flask web application
├── init.py # Package initialization
├── Dockerfile # Container configuration
├── Makefile # Automated build & run commands
├── .env.example # Environment variable template
└── README.md



## 📊 Key Features

- **Exploratory Data Analysis** — Distribution analysis, correlation heatmaps, missing value treatment, and outlier detection across all features
- **Feature Store** — Centralized feature engineering pipeline ensuring reproducibility and reusability across model versions
- **ML Pipeline** — End-to-end training pipeline with preprocessing, model selection, hyperparameter tuning, and evaluation
- **Dockerized Deployment** — Fully containerized application for consistent, one-command deployment across environments
- **Interactive Web App** — Real-time credit risk scoring interface via `app.py`
