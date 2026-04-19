"""
main.py
-------
One-command project runner.
Runs: data generation → preprocessing → training → EDA → prediction demo
Then prints instructions to launch the Streamlit UI.
"""

import os
import sys

# Add src to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.generate_data import generate_hr_dataset
from src.preprocess    import get_train_test
from src.train_model   import (
    train_all_models, evaluate_models, detailed_report,
    save_best_model, get_feature_importance
)
from src.eda           import run_all as run_eda
from src.predict       import predict_single

import pandas as pd
import os


def banner(text):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


if __name__ == '__main__':

    # ── PHASE 1: Generate dataset ─────────────────────────────────
    banner("PHASE 1 — Generating Synthetic HR Dataset")
    os.makedirs('data', exist_ok=True)
    df = generate_hr_dataset(1000)
    df.to_csv('data/hr_dataset.csv', index=False)
    print(f"✅ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df['performance_label'].value_counts().to_string())

    # ── PHASE 2: Preprocessing ────────────────────────────────────
    banner("PHASE 2 — Preprocessing & Feature Engineering")
    X_train, X_test, y_train, y_test, le_target, feat_names = get_train_test('data/hr_dataset.csv')

    # ── PHASE 3: Train Models ─────────────────────────────────────
    banner("PHASE 3 — Training ML Models")
    fitted = train_all_models(X_train, y_train)

    # ── PHASE 4: Evaluate ─────────────────────────────────────────
    banner("PHASE 4 — Model Evaluation")
    df_results, best_name = evaluate_models(fitted, X_test, y_test, le_target)
    best_model = fitted[best_name]

    detailed_report(best_model, X_test, y_test, le_target, best_name)
    get_feature_importance(best_model, feat_names, best_name)
    save_best_model(best_model, best_name, df_results)

    # ── PHASE 5: EDA Charts ───────────────────────────────────────
    banner("PHASE 5 — Generating EDA Charts")
    run_eda()

    # ── PHASE 6: Demo Prediction ──────────────────────────────────
    banner("PHASE 6 — Demo Prediction")
    sample = {
        'age': 34, 'gender': 'Female', 'education': 'Master',
        'department': 'Engineering', 'experience_years': 10,
        'salary': 82000, 'training_hours': 70, 'projects_completed': 15,
        'avg_monthly_hours': 185, 'satisfaction_score': 4.3,
        'last_promotion_years': 1, 'absenteeism_days': 2,
        'peer_review_score': 4.4, 'manager_rating': 4.6
    }
    result = predict_single(sample, model_dir='models')
    print(f"\n  Employee Profile: {sample['age']}yr {sample['gender']}, "
          f"{sample['department']}, {sample['education']}")
    print(f"  Predicted Label  : {result['label']}")
    print(f"  Confidence       : {result['confidence']}%")
    print(f"  Probabilities    : {result['probabilities']}")
    print("\n  HR Recommendations:")
    for r in result['recommendations']:
        print(f"    {r}")

    # ── DONE ──────────────────────────────────────────────────────
    banner("✅ ALL PHASES COMPLETE")
    print("""
  📊 Charts saved in  : images/
  🤖 Model saved in   : models/
  📋 Reports in       : outputs/

  🚀 Launch the Dashboard:
     streamlit run app.py

  📁 Upload to GitHub:
     git init
     git add .
     git commit -m "feat: Employee Performance Predictor v1.0"
     git remote add origin https://github.com/YOUR_USERNAME/employee-performance-predictor.git
     git push -u origin main
    """)
