"""
main.py
-------
One-command runner — executes all 6 phases end-to-end.

Usage:
    python main.py
"""

import os
import sys

# ── Add src/ to sys.path BEFORE any local imports ────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, 'src')
for p in [SRC_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Now import from src/ directly (no 'src.' prefix) ─────────────────
from generate_data import generate_hr_dataset
from preprocess    import get_train_test
from train_model   import (
    train_all_models, evaluate_models, detailed_report,
    save_best_model, get_feature_importance,
)
from eda           import run_all as run_eda
from predict       import predict_single

import pandas as pd


def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


if __name__ == '__main__':

    # ── Phase 1: Generate dataset ─────────────────────────────────
    banner("PHASE 1 — Generating Synthetic HR Dataset")
    data_dir = os.path.join(ROOT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    df = generate_hr_dataset(1000)
    csv_path = os.path.join(data_dir, 'hr_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} cols  ->  '{csv_path}'")
    print(df['performance_label'].value_counts().to_string())

    # ── Phase 2: Preprocess ───────────────────────────────────────
    banner("PHASE 2 — Preprocessing & Feature Engineering")
    X_train, X_test, y_train, y_test, le_target, feat_names = get_train_test(csv_path)

    # ── Phase 3: Train ────────────────────────────────────────────
    banner("PHASE 3 — Training ML Models")
    fitted = train_all_models(X_train, y_train)

    # ── Phase 4: Evaluate ─────────────────────────────────────────
    banner("PHASE 4 — Evaluation & Saving")
    df_results, best_name = evaluate_models(fitted, X_test, y_test, le_target)
    best_model = fitted[best_name]

    detailed_report(best_model, X_test, y_test, le_target, best_name)
    get_feature_importance(best_model, feat_names, best_name)
    save_best_model(best_model, best_name, df_results)

    # ── Phase 5: EDA ──────────────────────────────────────────────
    banner("PHASE 5 — Generating EDA Charts")
    run_eda()

    # ── Phase 6: Demo prediction ──────────────────────────────────
    banner("PHASE 6 — Demo Prediction")
    sample = {
        'age': 34, 'gender': 'Female', 'education': 'Master',
        'department': 'Engineering', 'experience_years': 10,
        'salary': 82000, 'training_hours': 70, 'projects_completed': 15,
        'avg_monthly_hours': 185, 'satisfaction_score': 4.3,
        'last_promotion_years': 1, 'absenteeism_days': 2,
        'peer_review_score': 4.4, 'manager_rating': 4.6,
    }
    result = predict_single(sample)
    print(f"  Profile     : Age={sample['age']}, {sample['gender']}, "
          f"{sample['department']}, {sample['education']}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']}%")
    print(f"  Probabilities: {result['probabilities']}")
    print("  HR Recommendations:")
    for r in result['recommendations']:
        print(f"    {r}")

    # ── Done ──────────────────────────────────────────────────────
    banner("ALL PHASES COMPLETE")
    print("""
  Output folders:
    images/   <- 7 EDA charts (PNG)
    models/   <- best_model.pkl + encoders + scaler
    outputs/  <- reports, CSVs, metadata JSONs

  Launch dashboard:
    streamlit run app.py

  GitHub upload:
    git init
    git add .
    git commit -m "feat: Employee Performance Predictor v1.0"
    git remote add origin https://github.com/YOUR_USERNAME/employee-performance-predictor.git
    git push -u origin main
    """)
