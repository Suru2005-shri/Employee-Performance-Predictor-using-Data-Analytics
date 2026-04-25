"""
pipeline.py — Scalable sklearn Pipeline
=========================================
The CORE scalability upgrade.

Instead of 5 separate pickle files (model + scaler + 2 encoders + feature list),
everything lives in ONE sklearn Pipeline object:

  Input raw dict/DataFrame
        ↓
  FeatureEngineer  (custom transformer — adds 4 features)
        ↓
  ColumnTransformer
    ├─ OneHotEncoder  ← categorical cols
    └─ StandardScaler ← numeric cols
        ↓
  Classifier  (any sklearn estimator)
        ↓
  Prediction + probabilities

Benefits:
  - Add a new feature: edit FeatureEngineer only
  - Swap the model: change one line in build_pipeline()
  - No manual encoding step anywhere in the codebase
  - One pickle file to deploy
  - sklearn-compatible: works with GridSearchCV, cross_val_score, etc.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from config import CFG


# ══════════════════════════════════════════════════════════════════════
# Custom Transformer 1: Feature Engineer
# ══════════════════════════════════════════════════════════════════════
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds 4 domain-driven features.
    Fully sklearn-compatible — works inside Pipeline and GridSearchCV.
    To add a new feature: add one line here. Nothing else needs to change.
    """

    def fit(self, X, y=None):
        return self   # stateless — no fitting needed

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        df["productivity_ratio"]     = df["projects_completed"] / (df["avg_monthly_hours"] / 160.0).clip(lower=0.1)
        df["engagement_score"]       = (0.4 * df["satisfaction_score"]
                                        + 0.3 * df["peer_review_score"]
                                        + 0.3 * df["manager_rating"])
        df["career_pace"]            = df["experience_years"] / (df["age"] - 21).clip(lower=1)
        df["training_effectiveness"] = df["training_hours"] / CFG.TRAINING_MEAN

        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + ["productivity_ratio", "engagement_score",
                       "career_pace", "training_effectiveness"]


# ══════════════════════════════════════════════════════════════════════
# Custom Transformer 2: Drop non-feature columns
# ══════════════════════════════════════════════════════════════════════
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns that should not be used as features."""

    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_to_drop = [c for c in self.columns if c in df.columns]
        return df.drop(columns=cols_to_drop)


# ══════════════════════════════════════════════════════════════════════
# Pipeline builder
# ══════════════════════════════════════════════════════════════════════
def build_pipeline(classifier=None, cat_features=None, num_features=None):
    """
    Build a full end-to-end sklearn Pipeline.

    Parameters
    ----------
    classifier : sklearn estimator (default: RandomForestClassifier)
    cat_features : list of categorical column names
    num_features : list of numeric column names

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if classifier is None:
        classifier = RandomForestClassifier(**CFG.RF_PARAMS)

    cat_features = cat_features or CFG.CATEGORICAL_FEATURES
    num_features = num_features or CFG.NUMERIC_FEATURES

    # Preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",   # safely ignores any extra columns
    )

    pipeline = Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("drop_cols",        ColumnDropper(columns=CFG.DROP_COLUMNS)),
        ("preprocessor",     preprocessor),
        ("classifier",       classifier),
    ])

    return pipeline


def get_all_pipelines():
    """Return a dict of {name: pipeline} for all models to compare."""
    return {
        "Random Forest"      : build_pipeline(RandomForestClassifier(**CFG.RF_PARAMS)),
        "Gradient Boosting"  : build_pipeline(GradientBoostingClassifier(**CFG.GB_PARAMS)),
        "Logistic Regression": build_pipeline(LogisticRegression(**CFG.LR_PARAMS)),
        "SVM"                : build_pipeline(SVC(**CFG.SVM_PARAMS)),
        "KNN"                : build_pipeline(KNeighborsClassifier(**CFG.KNN_PARAMS)),
    }


if __name__ == "__main__":
    # Quick smoke test
    import pandas as pd
    df = pd.read_csv(CFG.DATA_CSV)
    X = df.drop(columns=[CFG.TARGET_COL])
    y = df[CFG.TARGET_COL]

    pipe = build_pipeline()
    pipe.fit(X[:800], y[:800])
    preds = pipe.predict(X[800:])
    print(f"Pipeline smoke test — {len(preds)} predictions: {set(preds)}")
    print("pipeline.py OK")
