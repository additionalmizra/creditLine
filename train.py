import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from config import FEATURE_STORE, MODEL_DIR, SEED

def train():
    df = pd.read_parquet(FEATURE_STORE)
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"])  # tree model handles scaling

    rus = RandomUnderSampler(random_state=SEED)
    X_bal, y_bal = rus.fit_resample(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, pr_aucs = [], []
    best_model, best_auc = None, -1

    for tr_idx, va_idx in skf.split(X_bal, y_bal):
        Xtr, Xva = X_bal.iloc[tr_idx], X_bal.iloc[va_idx]
        ytr, yva = y_bal.iloc[tr_idx], y_bal.iloc[va_idx]

        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xva)[:,1]
        auc = roc_auc_score(yva, proba)
        pr_auc = average_precision_score(yva, proba)
        aucs.append(auc); pr_aucs.append(pr_auc)
        if auc > best_auc:
            best_auc, best_model = auc, model

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, Path(MODEL_DIR) / "rf_model.joblib")

    print({
        "cv_roc_auc": float(np.mean(aucs)),
        "cv_pr_auc": float(np.mean(pr_aucs)),
        "best_fold_auc": float(best_auc),
    })

if __name__ == "__main__":
    train()
