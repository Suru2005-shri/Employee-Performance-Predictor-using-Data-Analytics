"""
preprocess.py
-------------
Handles all data cleaning, encoding, and feature engineering steps.
Returns train/test splits ready for model training.

Run standalone:  python src/preprocess.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(rel):
    return os.path.join(ROOT_DIR, rel)


# ── 1. Load ──────────────────────────────────────────────────────────
def load_data(path=None):
    if path is None:
        path = _path('data/hr_dataset.csv')
    df = pd.read_csv(path)
    print(f"  [load]    {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ── 2. Clean ─────────────────────────────────────────────────────────
def clean_data(df):
    before = len(df)
    df = df.drop_duplicates().dropna()
    q_low  = df['salary'].quantile(0.01)
    q_high = df['salary'].quantile(0.99)
    df = df[(df['salary'] >= q_low) & (df['salary'] <= q_high)]
    df = df.reset_index(drop=True)
    print(f"  [clean]   {before} -> {len(df)} rows")
    return df


# ── 3. Feature Engineering ───────────────────────────────────────────
def feature_engineer(df):
    df = df.copy()
    df['productivity_ratio']    = df['projects_completed'] / (df['avg_monthly_hours'] / 160.0)
    df['engagement_score']      = (0.4 * df['satisfaction_score']
                                   + 0.3 * df['peer_review_score']
                                   + 0.3 * df['manager_rating'])
    df['career_pace']           = df['experience_years'] / (df['age'] - 21).clip(lower=1)
    df['training_effectiveness']= df['training_hours'] / (df['training_hours'].mean() + 1e-6)
    print(f"  [feature] 4 new features added")
    return df


# ── 4. Encode & Scale ────────────────────────────────────────────────
def encode_and_scale(df):
    df = df.copy()

    # ── Target ──────────────────────────────────────────────────────
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df['performance_label'].astype(str))
    print(f"  [encode]  Target classes: {list(le_target.classes_)}")

    # ── Build X (drop non-feature cols) ─────────────────────────────
    drop_cols = ['employee_id', 'performance_score', 'performance_label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # ── Encode categoricals — use pandas assign to avoid dtype conflict ──
    cat_cols = ['gender', 'education', 'department']
    le_dict  = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            encoded_vals = le.fit_transform(X[col].astype(str))
            # Assign as a new int column — safe for all pandas versions
            X[col] = encoded_vals.astype(int)
            le_dict[col] = le

    feature_names = list(X.columns)

    # ── Scale ────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Save artefacts ───────────────────────────────────────────────
    models_dir = _path('models')
    os.makedirs(models_dir, exist_ok=True)
    pickle.dump(le_target,     open(os.path.join(models_dir, 'le_target.pkl'),     'wb'))
    pickle.dump(le_dict,       open(os.path.join(models_dir, 'le_dict.pkl'),       'wb'))
    pickle.dump(scaler,        open(os.path.join(models_dir, 'scaler.pkl'),        'wb'))
    pickle.dump(feature_names, open(os.path.join(models_dir, 'feature_names.pkl'), 'wb'))
    print(f"  [encode]  {len(feature_names)} features | saved to models/")

    return X_scaled, y_encoded, le_target, feature_names


# ── 5. Full Pipeline ─────────────────────────────────────────────────
def get_train_test(path=None):
    print("\n[PREPROCESS] Starting...")
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineer(df)
    X, y, le_target, feature_names = encode_and_scale(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [split]   Train={X_train.shape}  Test={X_test.shape}")
    print("[PREPROCESS] Done.\n")
    return X_train, X_test, y_train, y_test, le_target, feature_names


if __name__ == '__main__':
    get_train_test()
