"""
predict.py
----------
Provides functions for:
  - single employee prediction (used by the Streamlit UI)
  - batch CSV prediction
  - confidence scoring + HR recommendation generation
"""

import pandas as pd
import numpy as np
import pickle
import json
import os


def load_artifacts(model_dir='../models'):
    """Load all saved model artifacts."""
    model    = pickle.load(open(f'{model_dir}/best_model.pkl',    'rb'))
    scaler   = pickle.load(open(f'{model_dir}/scaler.pkl',        'rb'))
    le_target = pickle.load(open(f'{model_dir}/le_target.pkl',   'rb'))
    le_dict  = pickle.load(open(f'{model_dir}/le_dict.pkl',       'rb'))
    features = pickle.load(open(f'{model_dir}/feature_names.pkl', 'rb'))
    return model, scaler, le_target, le_dict, features


def engineer_features(df):
    """Apply same feature engineering as training."""
    df = df.copy()
    df['productivity_ratio']      = df['projects_completed'] / (df['avg_monthly_hours'] / 160)
    df['engagement_score']        = (
        0.4 * df['satisfaction_score']
        + 0.3 * df['peer_review_score']
        + 0.3 * df['manager_rating']
    )
    df['career_pace']             = df['experience_years'] / (df['age'] - 21).clip(lower=1)
    df['training_effectiveness']  = df['training_hours'] / (df['training_hours'].mean() + 1e-5)
    return df


def generate_hr_recommendation(label, confidence, row):
    """Generate actionable HR recommendations based on prediction."""
    recs = {
        'High': [
            "🌟 Consider for fast-track promotion.",
            "🎯 Assign to high-impact strategic projects.",
            "🏆 Nominate for recognition / awards program.",
            "📈 Offer leadership development opportunities."
        ],
        'Medium': [
            "📚 Enroll in targeted skill-development training.",
            "🤝 Assign a mentor from high-performers team.",
            "🎯 Set clear quarterly performance goals.",
            "💬 Schedule regular 1:1 coaching sessions."
        ],
        'Low': [
            "🚨 Initiate a formal Performance Improvement Plan (PIP).",
            "🔍 Conduct root-cause analysis (workload, burnout?).",
            "📞 HR to schedule well-being check-in meeting.",
            "🛠 Provide immediate targeted training & support."
        ]
    }
    return recs.get(label, [])


def predict_single(employee_dict, model_dir='../models'):
    """
    Predict performance for a single employee.

    Parameters:
      employee_dict: dict with keys matching raw HR features
    Returns:
      dict with label, confidence, probabilities, recommendations
    """
    model, scaler, le_target, le_dict, feature_names = load_artifacts(model_dir)

    df = pd.DataFrame([employee_dict])

    # Encode categoricals
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Engineer features
    df = engineer_features(df)

    # Align columns to training order
    df = df[feature_names]

    # Scale
    X = scaler.transform(df)

    # Predict
    pred_encoded = model.predict(X)[0]
    label        = le_target.inverse_transform([pred_encoded])[0]
    proba        = model.predict_proba(X)[0]
    confidence   = round(float(proba.max()) * 100, 1)

    prob_dict = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(le_target.classes_, proba)
    }

    recommendations = generate_hr_recommendation(label, confidence, employee_dict)

    return {
        'label'          : label,
        'confidence'     : confidence,
        'probabilities'  : prob_dict,
        'recommendations': recommendations
    }


def predict_batch(csv_path, output_path='../outputs/batch_predictions.csv', model_dir='../models'):
    """Predict performance for all employees in a CSV file."""
    model, scaler, le_target, le_dict, feature_names = load_artifacts(model_dir)

    df_raw = pd.read_csv(csv_path)
    df     = df_raw.copy()

    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    df = engineer_features(df)
    df = df[feature_names]
    X  = scaler.transform(df)

    preds  = le_target.inverse_transform(model.predict(X))
    probas = model.predict_proba(X).max(axis=1)

    df_raw['predicted_label']      = preds
    df_raw['prediction_confidence'] = (probas * 100).round(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    print(f"✅ Batch predictions saved to {output_path}")
    print(df_raw['predicted_label'].value_counts())
    return df_raw


if __name__ == '__main__':
    # Example single prediction
    sample_employee = {
        'age'                 : 32,
        'gender'              : 'Female',
        'education'           : 'Master',
        'department'          : 'Engineering',
        'experience_years'    : 8,
        'salary'              : 75000,
        'training_hours'      : 60,
        'projects_completed'  : 12,
        'avg_monthly_hours'   : 180,
        'satisfaction_score'  : 4.2,
        'last_promotion_years': 2,
        'absenteeism_days'    : 3,
        'peer_review_score'   : 4.0,
        'manager_rating'      : 4.5
    }

    result = predict_single(sample_employee)
    print("\n🔮 Prediction Result:")
    print(f"  Label       : {result['label']}")
    print(f"  Confidence  : {result['confidence']}%")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  HR Actions  :")
    for r in result['recommendations']:
        print(f"    {r}")
