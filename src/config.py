"""
config.py — Single Source of Truth
====================================
ALL paths, hyperparameters, feature lists, and constants live here.
Change ONE value here and the entire project adapts automatically.

How to use:
    from config import CFG
    path = CFG.DATA_CSV
    params = CFG.RF_PARAMS
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ── Project Root (one level above src/) ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────
    ROOT_DIR      : str = ROOT
    DATA_DIR      : str = os.path.join(ROOT, "data")
    MODELS_DIR    : str = os.path.join(ROOT, "models")
    OUTPUTS_DIR   : str = os.path.join(ROOT, "outputs")
    IMAGES_DIR    : str = os.path.join(ROOT, "images")
    SRC_DIR       : str = os.path.join(ROOT, "src")
    TESTS_DIR     : str = os.path.join(ROOT, "tests")

    # ── Key file paths ───────────────────────────────────────────────
    DATA_CSV      : str = os.path.join(ROOT, "data", "hr_dataset.csv")
    PIPELINE_PKL  : str = os.path.join(ROOT, "models", "pipeline.pkl")
    METADATA_JSON : str = os.path.join(ROOT, "outputs", "model_metadata.json")
    COMPARISON_CSV: str = os.path.join(ROOT, "outputs", "model_comparison.csv")
    FI_CSV        : str = os.path.join(ROOT, "outputs", "feature_importance.csv")
    REPORT_JSON   : str = os.path.join(ROOT, "outputs", "classification_report.json")
    CM_JSON       : str = os.path.join(ROOT, "outputs", "confusion_matrix.json")

    # ── Dataset ──────────────────────────────────────────────────────
    N_SAMPLES     : int = 1000
    RANDOM_STATE  : int = 42
    TEST_SIZE     : float = 0.20

    # ── Feature groups ───────────────────────────────────────────────
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "gender", "education", "department"
    ])
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        "age", "experience_years", "salary",
        "training_hours", "projects_completed", "avg_monthly_hours",
        "satisfaction_score", "last_promotion_years", "absenteeism_days",
        "peer_review_score", "manager_rating",
        # engineered
        "productivity_ratio", "engagement_score",
        "career_pace", "training_effectiveness",
    ])
    DROP_COLUMNS: List[str] = field(default_factory=lambda: [
        "employee_id", "performance_score", "performance_label"
    ])
    TARGET_COL: str = "performance_label"
    CLASSES: List[str] = field(default_factory=lambda: ["High", "Low", "Medium"])

    # ── Model hyperparameters ─────────────────────────────────────────
    RF_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })
    GB_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": 42,
    })
    LR_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
        "C": 1.0,
    })
    SVM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "kernel": "rbf",
        "probability": True,
        "class_weight": "balanced",
        "random_state": 42,
        "C": 1.0,
    })
    KNN_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "n_neighbors": 7,
        "weights": "distance",
    })

    # ── Cross-validation ─────────────────────────────────────────────
    CV_FOLDS: int = 5
    CV_SCORING: str = "f1_weighted"

    # ── Grid search (for best model tuning) ──────────────────────────
    RF_GRID: Dict[str, Any] = field(default_factory=lambda: {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
    })

    # ── API ───────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Performance thresholds ────────────────────────────────────────
    AT_RISK_SAT_THRESHOLD   : float = 2.5
    AT_RISK_ABSENT_THRESHOLD: int   = 18
    ATTRITION_COST_PER_EMP  : int   = 50_000
    TRAINING_MEAN           : float = 50.0   # fixed mean for inference


# Global singleton
CFG = Config()


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [CFG.DATA_DIR, CFG.MODELS_DIR, CFG.OUTPUTS_DIR, CFG.IMAGES_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    print("=== Project Config ===")
    for k, v in CFG.__dict__.items():
        print(f"  {k:<25} = {v}")
