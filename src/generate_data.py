"""
generate_data.py
----------------
Generates a realistic synthetic HR dataset.
Now reads all settings from config.CFG.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from config import CFG, ensure_dirs

np.random.seed(CFG.RANDOM_STATE)

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
EDUCATION   = ["High School", "Bachelor", "Master", "PhD"]
GENDER      = ["Male", "Female", "Other"]
EDU_BONUS   = {"High School": 0, "Bachelor": 10_000, "Master": 20_000, "PhD": 35_000}


def generate_hr_dataset(n: int = None) -> pd.DataFrame:
    if n is None:
        n = CFG.N_SAMPLES

    age              = np.random.randint(22, 60, n)
    experience_years = np.clip(age - 22 + np.random.randint(-2, 3, n), 0, 38)
    dept             = np.random.choice(DEPARTMENTS, n)
    edu              = np.random.choice(EDUCATION, n, p=[0.10, 0.55, 0.28, 0.07])
    gen              = np.random.choice(GENDER, n, p=[0.52, 0.45, 0.03])

    salary = (
        30_000
        + experience_years * 1_500
        + np.array([EDU_BONUS[e] for e in edu])
        + np.random.randint(-5_000, 5_000, n)
    )

    training_hours       = np.random.randint(0, 100, n)
    projects_completed   = np.random.randint(1, 25, n)
    avg_monthly_hours    = np.random.randint(140, 280, n)
    satisfaction_score   = np.round(np.random.uniform(1.0, 5.0, n), 1)
    last_promotion_years = np.random.randint(0, 10, n)
    absenteeism_days     = np.random.randint(0, 30, n)
    peer_review_score    = np.round(np.random.uniform(1.0, 5.0, n), 1)
    manager_rating       = np.round(np.random.uniform(1.0, 5.0, n), 1)

    # Weighted performance formula (domain knowledge)
    perf_score = (
        0.25 * (training_hours / 100 * 5)
        + 0.20 * (projects_completed / 25 * 5)
        + 0.20 * satisfaction_score
        + 0.15 * peer_review_score
        + 0.15 * manager_rating
        - 0.05 * (absenteeism_days / 30 * 5)
        + np.random.normal(0, 0.3, n)
    )
    perf_score = np.clip(perf_score, 1.0, 5.0)

    performance_label = pd.cut(
        perf_score,
        bins=[0, 2.5, 3.5, 5.0],
        labels=["Low", "Medium", "High"],
    )

    return pd.DataFrame({
        "employee_id"         : range(1001, 1001 + n),
        "age"                 : age,
        "gender"              : gen,
        "education"           : edu,
        "department"          : dept,
        "experience_years"    : experience_years,
        "salary"              : salary,
        "training_hours"      : training_hours,
        "projects_completed"  : projects_completed,
        "avg_monthly_hours"   : avg_monthly_hours,
        "satisfaction_score"  : satisfaction_score,
        "last_promotion_years": last_promotion_years,
        "absenteeism_days"    : absenteeism_days,
        "peer_review_score"   : peer_review_score,
        "manager_rating"      : manager_rating,
        "performance_score"   : np.round(perf_score, 2),
        "performance_label"   : performance_label,
    })


if __name__ == "__main__":
    ensure_dirs()
    df = generate_hr_dataset()
    df.to_csv(CFG.DATA_CSV, index=False)
    print(f"Saved {len(df)} rows → {CFG.DATA_CSV}")
    print(df["performance_label"].value_counts().to_string())
