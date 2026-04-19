"""
predict.py
----------
Single-employee and batch CSV prediction.

Run standalone:  python src/predict.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR  not in sys.path: sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

def _path(rel):
    return os.path.join(ROOT_DIR, rel)


# ── Load artefacts ───────────────────────────────────────────────────
def load_artifacts(model_dir=None):
    if model_dir is None:
        model_dir = _path('models')
    elif not os.path.isabs(model_dir):
        model_dir = os.path.join(ROOT_DIR, model_dir)

    def _load(name):
        return pickle.load(open(os.path.join(model_dir, name), 'rb'))

    return (
        _load('best_model.pkl'),
        _load('scaler.pkl'),
        _load('le_target.pkl'),
        _load('le_dict.pkl'),
        _load('feature_names.pkl'),
    )


# ── Feature engineering (must match preprocess.py exactly) ──────────
def _engineer(df):
    df = df.copy()
    df['productivity_ratio']     = df['projects_completed'] / (df['avg_monthly_hours'] / 160.0)
    df['engagement_score']       = (0.4 * df['satisfaction_score']
                                    + 0.3 * df['peer_review_score']
                                    + 0.3 * df['manager_rating'])
    df['career_pace']            = df['experience_years'] / (df['age'] - 21).clip(lower=1)
    df['training_effectiveness'] = df['training_hours'] / (df['training_hours'].mean() + 1e-6)
    return df


# ── Safe categorical encoding (pandas 2.x compatible) ───────────────
def _encode_cats(df, le_dict):
    """
    Encode categorical columns using saved LabelEncoders.
    Avoids pandas 2.x StringDtype conflict by using direct column replacement.
    """
    df = df.copy()
    for col, le in le_dict.items():
        if col in df.columns:
            encoded = le.transform(df[col].astype(str)).astype(int)
            # Re-assign as a brand-new int Series to bypass dtype guard
            df[col] = pd.Series(encoded, index=df.index, dtype=int)
    return df


# ── HR recommendation engine ─────────────────────────────────────────
HR_RECS = {
    'High': [
        "🌟 Fast-track promotion candidate — present to leadership",
        "🎯 Assign to high-impact strategic / cross-functional projects",
        "🏆 Nominate for quarterly recognition award",
        "📈 Offer leadership development track or mentorship role",
    ],
    'Medium': [
        "📚 Enroll in targeted skill-development training (role-specific)",
        "🤝 Pair with a high-performing mentor for 3 months",
        "🎯 Co-create clear quarterly OKRs with direct manager",
        "💬 Schedule bi-weekly coaching 1:1 to track progress",
    ],
    'Low': [
        "🚨 Initiate a formal 90-day Performance Improvement Plan (PIP)",
        "🔍 Root-cause 1:1 — workload? burnout? role mismatch?",
        "📞 HR well-being check-in — mental health support referral",
        "🛠 Immediate structured training + daily manager check-in",
    ],
}


# ── Single prediction ────────────────────────────────────────────────
def predict_single(employee_dict, model_dir=None):
    model, scaler, le_target, le_dict, feature_names = load_artifacts(model_dir)

    df = pd.DataFrame([employee_dict])
    df = _encode_cats(df, le_dict)
    df = _engineer(df)

    # Add any missing columns (safety net)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    X          = scaler.transform(df)
    pred_enc   = model.predict(X)[0]
    label      = le_target.inverse_transform([pred_enc])[0]
    proba      = model.predict_proba(X)[0]
    confidence = round(float(proba.max()) * 100, 1)
    prob_dict  = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(le_target.classes_, proba)
    }

    return {
        'label'          : label,
        'confidence'     : confidence,
        'probabilities'  : prob_dict,
        'recommendations': HR_RECS.get(label, []),
    }


# ── Batch prediction ─────────────────────────────────────────────────
def predict_batch(csv_path, output_path=None, model_dir=None):
    if output_path is None:
        output_path = _path('outputs/batch_predictions.csv')

    model, scaler, le_target, le_dict, feature_names = load_artifacts(model_dir)

    df_raw = pd.read_csv(csv_path)
    df     = _encode_cats(df_raw.copy(), le_dict)
    df     = _engineer(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    X      = scaler.transform(df)
    preds  = le_target.inverse_transform(model.predict(X))
    confs  = (model.predict_proba(X).max(axis=1) * 100).round(1)

    df_raw['predicted_label']       = preds
    df_raw['prediction_confidence'] = confs

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    print(f"  Batch predictions saved to '{output_path}'")
    print(df_raw['predicted_label'].value_counts().to_string())
    return df_raw


# ── Standalone demo ──────────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'age': 32, 'gender': 'Female', 'education': 'Master',
        'department': 'Engineering', 'experience_years': 8,
        'salary': 75000, 'training_hours': 60, 'projects_completed': 12,
        'avg_monthly_hours': 180, 'satisfaction_score': 4.2,
        'last_promotion_years': 2, 'absenteeism_days': 3,
        'peer_review_score': 4.0, 'manager_rating': 4.5,
    }
    result = predict_single(sample)
    print(f"  Label        : {result['label']}")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"  Probabilities: {result['probabilities']}")
    print("  HR Actions:")
    for r in result['recommendations']:
        print(f"    {r}")
