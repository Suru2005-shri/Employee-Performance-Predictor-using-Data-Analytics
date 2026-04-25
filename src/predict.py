"""
predict.py — Scalable Prediction Engine
=========================================
Loads ONE pipeline.pkl and predicts.
No manual encoding. No separate scaler files. No feature list files.
Just: raw dict → pipeline.predict() → label + confidence + recommendations.
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from config import CFG

HR_RECS = {
    "High"  : ["🌟 Fast-track promotion candidate — present to leadership",
               "🎯 Assign to high-impact strategic / cross-functional projects",
               "🏆 Nominate for quarterly recognition award",
               "📈 Offer leadership development track or mentorship role"],
    "Medium": ["📚 Enroll in targeted skill-development training (role-specific)",
               "🤝 Pair with a high-performing mentor for 3 months",
               "🎯 Co-create clear quarterly OKRs with direct manager",
               "💬 Schedule bi-weekly coaching 1:1 to track progress"],
    "Low"   : ["🚨 Initiate a formal 90-day Performance Improvement Plan (PIP)",
               "🔍 Root-cause 1:1 — workload? burnout? role mismatch?",
               "📞 HR well-being check-in — mental health support referral",
               "🛠 Immediate structured training + daily manager check-in"],
}

# Positive weight = this feature INCREASES risk when low
# Negative weight = this feature DECREASES risk when high
RISK_SCORE_WEIGHTS = {
    "absenteeism_days"  :  0.3,   # more absent  → higher risk
    "satisfaction_score": -0.3,   # more satisfied → lower risk
    "training_hours"    : -0.2,   # more training  → lower risk
    "manager_rating"    : -0.2,   # better rating  → lower risk
}


def load_pipeline():
    if not os.path.exists(CFG.PIPELINE_PKL):
        raise FileNotFoundError(
            f"Pipeline not found at {CFG.PIPELINE_PKL}. "
            "Run `python main.py` first to train the model."
        )
    with open(CFG.PIPELINE_PKL, "rb") as f:
        return pickle.load(f)


def compute_risk_score(emp: dict) -> float:
    """
    Returns a 0-10 attrition risk score.
    Higher = more at risk of leaving.

    Logic:
      absenteeism_days   (0-30)  : high days   → high risk  (+)
      satisfaction_score (1-5)   : high score  → low risk   (-)
      training_hours     (0-100) : high hours  → low risk   (-)
      manager_rating     (1-5)   : high rating → low risk   (-)
    """
    score = 5.0  # neutral baseline

    ranges = {
        "absenteeism_days"  : (0,  30, "positive"),
        "satisfaction_score": (1,   5, "negative"),
        "training_hours"    : (0, 100, "negative"),
        "manager_rating"    : (1,   5, "negative"),
    }
    weights = {
        "absenteeism_days"  : 2.5,
        "satisfaction_score": 2.5,
        "training_hours"    : 1.5,
        "manager_rating"    : 1.5,
    }

    for feat, (lo, hi, direction) in ranges.items():
        if feat not in emp:
            continue
        val        = float(emp[feat])
        normalised = (val - lo) / (hi - lo)   # 0..1
        if direction == "positive":
            score += weights[feat] * normalised        # more → riskier
        else:
            score -= weights[feat] * normalised        # more → safer

    return round(max(0.0, min(10.0, score)), 1)


def predict_single(emp: dict) -> dict:
    """
    Predict performance for ONE employee.

    Parameters
    ----------
    emp : dict of raw HR feature values

    Returns
    -------
    dict with keys: label, confidence, probabilities, recommendations, risk_score
    """
    pipe  = load_pipeline()
    df    = pd.DataFrame([emp])

    label  = pipe.predict(df)[0]
    proba  = pipe.predict_proba(df)[0]
    classes = pipe.classes_

    conf      = round(float(proba.max()) * 100, 1)
    prob_dict = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    risk      = compute_risk_score(emp)

    return {
        "label"          : label,
        "confidence"     : conf,
        "probabilities"  : prob_dict,
        "recommendations": HR_RECS.get(label, []),
        "risk_score"     : risk,
    }


def predict_batch(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Predict performance for every row in a CSV.
    Saves enriched CSV and returns the DataFrame.
    """
    if output_path is None:
        output_path = os.path.join(CFG.OUTPUTS_DIR, "batch_predictions.csv")

    pipe   = load_pipeline()
    df_raw = pd.read_csv(csv_path)

    preds  = pipe.predict(df_raw)
    probas = pipe.predict_proba(df_raw).max(axis=1)
    risks  = df_raw.apply(lambda r: compute_risk_score(r.to_dict()), axis=1)

    df_out = df_raw.copy()
    df_out["predicted_label"]       = preds
    df_out["prediction_confidence"] = (probas * 100).round(1)
    df_out["attrition_risk_score"]  = risks

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"  Batch saved → {output_path}")
    print(df_out["predicted_label"].value_counts().to_string())
    return df_out


if __name__ == "__main__":
    sample = {
        "age": 32, "gender": "Female", "education": "Master",
        "department": "Engineering", "experience_years": 8,
        "salary": 75_000, "training_hours": 60, "projects_completed": 12,
        "avg_monthly_hours": 180, "satisfaction_score": 4.2,
        "last_promotion_years": 2, "absenteeism_days": 3,
        "peer_review_score": 4.0, "manager_rating": 4.5,
        "performance_score": 0,   # ignored by pipeline
    }
    result = predict_single(sample)
    print(f"  Label       : {result['label']}")
    print(f"  Confidence  : {result['confidence']}%")
    print(f"  Risk Score  : {result['risk_score']}/10")
    print(f"  Probs       : {result['probabilities']}")
