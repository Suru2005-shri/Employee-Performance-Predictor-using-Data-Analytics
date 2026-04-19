"""
preprocess.py
-------------
Handles all data cleaning, encoding, and feature engineering steps.
Returns train/test splits ready for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from sklearn.preprocessing import StandardScaler


def load_data(path='../data/hr_dataset.csv'):
    df = pd.read_csv(path)
    print(f"📂 Loaded dataset: {df.shape}")
    return df


def clean_data(df):
    """Drop duplicates, handle missing values."""
    df = df.drop_duplicates()
    df = df.dropna()

    # Remove obvious outliers in salary
    q_low  = df['salary'].quantile(0.01)
    q_high = df['salary'].quantile(0.99)
    df = df[(df['salary'] >= q_low) & (df['salary'] <= q_high)]

    print(f"✅ After cleaning: {df.shape}")
    return df


def feature_engineer(df):
    """Create new meaningful features from existing ones."""

    # Productivity ratio
    df['productivity_ratio'] = df['projects_completed'] / (df['avg_monthly_hours'] / 160)

    # Engagement score (weighted combination)
    df['engagement_score'] = (
        0.4 * df['satisfaction_score']
        + 0.3 * df['peer_review_score']
        + 0.3 * df['manager_rating']
    )

    # Experience-to-age ratio (career pace)
    df['career_pace'] = df['experience_years'] / (df['age'] - 21).clip(lower=1)

    # Training effectiveness (training hours relative to avg)
    df['training_effectiveness'] = df['training_hours'] / (df['training_hours'].mean() + 1)

    print("✅ Feature engineering done")
    return df

    
def encode_and_scale(df):
    """Encode categoricals, scale numerics, return arrays + encoders."""

    df = df.copy()

    # Drop non-feature columns
    drop_cols = ['employee_id', 'performance_score', 'performance_label']
    X = df.drop(columns=drop_cols)
    y = df['performance_label']

    # Encode target
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Encode categorical features
    cat_cols = ['gender', 'education', 'department']
    le_dict = {}

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Encoding done. Classes: {le_target.classes_}")

    return X_scaled, y_encoded, scaler, le_target, le_dict, list(X.columns)


def get_train_test(path='data/hr_dataset.csv'):
    """Full pipeline: load → clean → engineer → encode → split."""

    df = load_data(path)
    df = clean_data(df)
    df = feature_engineer(df)

    # Already returns scaled data + everything
    X, y, scaler, le_target, le_dict, feature_names = encode_and_scale(df)

    # ONLY split (no re-scaling, no column extraction)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, le_target, le_dict, feature_names

if __name__ == '__main__':
    get_train_test()
