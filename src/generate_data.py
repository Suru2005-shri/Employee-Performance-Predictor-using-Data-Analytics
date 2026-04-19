"""
generate_data.py
----------------
Creates a 1000-row synthetic HR dataset with realistic correlations.
Run standalone:  python src/generate_data.py
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)


def generate_hr_dataset(n=1000):
    departments = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
    education   = ['High School', 'Bachelor', 'Master', 'PhD']
    gender      = ['Male', 'Female', 'Other']

    age              = np.random.randint(22, 60, n)
    experience_years = np.clip(age - 22 + np.random.randint(-2, 3, n), 0, 38)
    dept             = np.random.choice(departments, n)
    edu              = np.random.choice(education, n, p=[0.10, 0.55, 0.28, 0.07])
    gen              = np.random.choice(gender, n, p=[0.52, 0.45, 0.03])

    edu_bonus = {'High School': 0, 'Bachelor': 10000, 'Master': 20000, 'PhD': 35000}
    salary = (
        30000
        + experience_years * 1500
        + np.array([edu_bonus[e] for e in edu])
        + np.random.randint(-5000, 5000, n)
    )

    training_hours        = np.random.randint(0, 100, n)
    projects_completed    = np.random.randint(1, 25, n)
    avg_monthly_hours     = np.random.randint(140, 280, n)
    satisfaction_score    = np.round(np.random.uniform(1.0, 5.0, n), 1)
    last_promotion_years  = np.random.randint(0, 10, n)
    absenteeism_days      = np.random.randint(0, 30, n)
    peer_review_score     = np.round(np.random.uniform(1.0, 5.0, n), 1)
    manager_rating        = np.round(np.random.uniform(1.0, 5.0, n), 1)

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
        labels=['Low', 'Medium', 'High']
    )

    df = pd.DataFrame({
        'employee_id'         : range(1001, 1001 + n),
        'age'                 : age,
        'gender'              : gen,
        'education'           : edu,
        'department'          : dept,
        'experience_years'    : experience_years,
        'salary'              : salary,
        'training_hours'      : training_hours,
        'projects_completed'  : projects_completed,
        'avg_monthly_hours'   : avg_monthly_hours,
        'satisfaction_score'  : satisfaction_score,
        'last_promotion_years': last_promotion_years,
        'absenteeism_days'    : absenteeism_days,
        'peer_review_score'   : peer_review_score,
        'manager_rating'      : manager_rating,
        'performance_score'   : np.round(perf_score, 2),
        'performance_label'   : performance_label,
    })
    return df


if __name__ == '__main__':
    data_dir = os.path.join(ROOT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    df = generate_hr_dataset(1000)
    out = os.path.join(data_dir, 'hr_dataset.csv')
    df.to_csv(out, index=False)
    print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} cols -> '{out}'")
    print(df['performance_label'].value_counts().to_string())
