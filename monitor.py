import pandas as pd
import numpy as np

def psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    qs = np.linspace(0, 1, bins+1)
    e_cuts = np.quantile(expected, qs)
    a_cuts = np.quantile(actual, qs)
    e_hist, _ = np.histogram(expected, bins=e_cuts)
    a_hist, _ = np.histogram(actual, bins=a_cuts)
    e_pct = (e_hist + 1e-6) / (e_hist.sum() + 1e-6)
    a_pct = (a_hist + 1e-6) / (a_hist.sum() + 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
