"""
main.py — Scalable One-Command Runner
=======================================
Runs all phases using the new Pipeline architecture.

Usage:
    python main.py           # standard run
    python main.py --tune    # with GridSearchCV tuning (slower)
    python main.py --test    # also run pytest after training
"""

import os, sys, argparse

ROOT   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CFG, ensure_dirs
from generate_data import generate_hr_dataset
from train_model   import run_training
from eda           import run_all as run_eda
from predict       import predict_single


def banner(text):
    print("\n" + "=" * 62)
    print(f"  {text}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Performance Predictor")
    parser.add_argument("--tune",    action="store_true", help="Run GridSearchCV")
    parser.add_argument("--test",    action="store_true", help="Run pytest after training")
    parser.add_argument("--samples", type=int, default=CFG.N_SAMPLES,
                        help=f"Dataset size (default {CFG.N_SAMPLES})")
    args = parser.parse_args()

    ensure_dirs()

    # ── Phase 1: Data generation ─────────────────────────────────
    banner("PHASE 1 — Generating Synthetic HR Dataset")
    df = generate_hr_dataset(args.samples)
    df.to_csv(CFG.DATA_CSV, index=False)
    print(f"  {args.samples} rows saved → {CFG.DATA_CSV}")
    print("  " + df["performance_label"].value_counts().to_string().replace("\n", "\n  "))

    # ── Phase 2: Train ───────────────────────────────────────────
    banner("PHASE 2 — Training with Cross-Validation")
    best_pipe, best_name, acc, f1 = run_training(tune=args.tune)

    # ── Phase 3: EDA charts ──────────────────────────────────────
    banner("PHASE 3 — Generating EDA Charts")
    run_eda()

    # ── Phase 4: Demo prediction ─────────────────────────────────
    banner("PHASE 4 — Demo Prediction (Pipeline)")
    sample = {
        "age":34, "gender":"Female", "education":"Master",
        "department":"Engineering", "experience_years":10,
        "salary":82000, "training_hours":70, "projects_completed":15,
        "avg_monthly_hours":185, "satisfaction_score":4.3,
        "last_promotion_years":1, "absenteeism_days":2,
        "peer_review_score":4.4, "manager_rating":4.6,
        "performance_score":0,
    }
    result = predict_single(sample)
    print(f"  Prediction   : {result['label']}  ({result['confidence']}% confidence)")
    print(f"  Risk Score   : {result['risk_score']}/10")
    print(f"  Probabilities: {result['probabilities']}")
    print("  Recommendations:")
    for r in result["recommendations"]:
        print(f"    {r}")

    # ── Phase 5: Tests ───────────────────────────────────────────
    if args.test:
        banner("PHASE 5 — Running Test Suite")
        import subprocess
        result_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=ROOT
        )
        if result_proc.returncode != 0:
            print("  Some tests FAILED — review above output")
        else:
            print("  All tests PASSED")

    banner("ALL PHASES COMPLETE")
    print(f"""
  Model       : {best_name}
  Accuracy    : {acc*100:.1f}%
  F1 Score    : {f1*100:.1f}%
  Pipeline    : {CFG.PIPELINE_PKL}

  Launch dashboard:
    streamlit run app.py

  Serve API (after: pip install fastapi uvicorn):
    uvicorn src.api:app --host 0.0.0.0 --port 8000

  Run tests:
    pytest tests/ -v

  GitHub:
    git add . && git commit -m "feat: scalable pipeline v2.0"
    git push
    """)
