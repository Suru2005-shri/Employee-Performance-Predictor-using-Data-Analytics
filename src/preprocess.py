"""
preprocess.py — Data Cleaning Only
====================================
In the scalable version, preprocessing is handled inside the Pipeline.
This module only does: load → clean → return X, y.
No manual encoding. No manual scaling. The Pipeline does it all.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import CFG


def load_and_clean(path: str = None) -> pd.DataFrame:
    """Load CSV and perform basic cleaning."""
    path = path or CFG.DATA_CSV
    df   = pd.read_csv(path)
    before = len(df)

    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # Remove salary outliers
    q1, q99 = df["salary"].quantile([0.01, 0.99])
    df = df[(df["salary"] >= q1) & (df["salary"] <= q99)].reset_index(drop=True)

    print(f"  [clean]  {before} → {len(df)} rows")
    return df


def get_X_y(path: str = None):
    """Return feature matrix X and target series y. Pipeline handles the rest."""
    df = load_and_clean(path)
    X  = df.drop(columns=[CFG.TARGET_COL])
    y  = df[CFG.TARGET_COL].astype(str)
    print(f"  [split]  X={X.shape}  classes={y.value_counts().to_dict()}")
    return X, y


if __name__ == "__main__":
    X, y = get_X_y()
    print(X.head(2).to_string())
