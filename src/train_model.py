"""
train_model.py — Scalable Model Training
==========================================
Upgrades over v1:
  - Uses Pipeline (no manual encode/scale steps)
  - Stratified K-Fold cross-validation for reliable accuracy estimates
  - class_weight='balanced' to handle class imbalance (no SMOTE needed)
  - Optional GridSearchCV for hyperparameter tuning
  - Model versioning: saves with timestamp so old models are never overwritten
  - Saves ONE pipeline.pkl instead of 5 separate files
"""

import os, sys, json, pickle, warnings, time
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from config import CFG, ensure_dirs
from pipeline import build_pipeline, get_all_pipelines


# ── 1. Load & split ──────────────────────────────────────────────────
def load_and_split():
    df = pd.read_csv(CFG.DATA_CSV)
    X  = df.drop(columns=[CFG.TARGET_COL])
    y  = df[CFG.TARGET_COL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CFG.TEST_SIZE,
        random_state = CFG.RANDOM_STATE,
        stratify     = y,
    )
    print(f"  [split]  Train={len(X_train)}  Test={len(X_test)}")
    print(f"  [dist]   {y.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ── 2. Cross-validate all pipelines ─────────────────────────────────
def cross_validate_all(X_train, y_train):
    """
    Run StratifiedKFold CV on every model.
    Returns a sorted DataFrame of results.
    """
    pipelines = get_all_pipelines()
    cv        = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True,
                                random_state=CFG.RANDOM_STATE)
    results   = []

    for name, pipe in pipelines.items():
        t0     = time.time()
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring=CFG.CV_SCORING, n_jobs=-1)
        elapsed = round(time.time() - t0, 1)
        results.append({
            "Model"     : name,
            "CV_Mean_F1": round(scores.mean(), 4),
            "CV_Std_F1" : round(scores.std(),  4),
            "Time_s"    : elapsed,
        })
        print(f"  {name:<25}  CV F1={scores.mean():.4f} ± {scores.std():.4f}  ({elapsed}s)")

    df_res    = pd.DataFrame(results).sort_values("CV_Mean_F1", ascending=False).reset_index(drop=True)
    best_name = df_res.loc[0, "Model"]
    print(f"\n  Best (CV): {best_name}  F1={df_res.loc[0,'CV_Mean_F1']}")
    return df_res, best_name


# ── 3. Optional: tune best model with GridSearchCV ──────────────────
def tune_best_model(X_train, y_train, best_name):
    """
    Runs GridSearchCV on the best model's pipeline.
    Only runs if best_name is Random Forest (most common winner).
    Returns the best fitted pipeline.
    """
    from pipeline import get_all_pipelines
    pipe = get_all_pipelines()[best_name]

    # Only tune RF — other models use their CFG params
    if best_name == "Random Forest":
        print(f"  [tune]  GridSearchCV on {best_name}...")
        cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_STATE)
        gs     = GridSearchCV(pipe, CFG.RF_GRID, cv=cv,
                              scoring=CFG.CV_SCORING, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        print(f"  [tune]  Best params: {gs.best_params_}")
        print(f"  [tune]  Best CV F1:  {gs.best_score_:.4f}")
        return gs.best_estimator_
    else:
        pipe.fit(X_train, y_train)
        return pipe


# ── 4. Evaluate on hold-out test set ────────────────────────────────
def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    classes = sorted(y_test.unique())
    report  = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    print(f"\n  [eval]  Test Accuracy={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))

    return acc, f1, report, cm, classes


# ── 5. Feature importance ────────────────────────────────────────────
def extract_feature_importance(pipe):
    """
    Extract feature importances through the ColumnTransformer.
    Works for tree-based models that expose feature_importances_.
    """
    clf = pipe.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        return None

    # Get feature names after ColumnTransformer
    preprocessor = pipe.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean up prefixes added by ColumnTransformer
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

    fi = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fi.to_csv(CFG.FI_CSV, index=False)
    print(f"\n  Top 5 Features:")
    print(fi.head().to_string(index=False))
    return fi


# ── 6. Save pipeline + metadata ─────────────────────────────────────
def save_pipeline(pipe, model_name, acc, f1, df_cv, report, cm, classes):
    ensure_dirs()

    # Save pipeline
    with open(CFG.PIPELINE_PKL, "wb") as f:
        pickle.dump(pipe, f)

    # Versioned backup with timestamp
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver_dir = os.path.join(CFG.MODELS_DIR, "versions")
    os.makedirs(ver_dir, exist_ok=True)
    with open(os.path.join(ver_dir, f"pipeline_{ts}.pkl"), "wb") as f:
        pickle.dump(pipe, f)

    # Save CV comparison
    df_cv.to_csv(CFG.COMPARISON_CSV, index=False)

    # Save metadata
    meta = {
        "best_model"   : model_name,
        "accuracy"     : round(acc, 4),
        "f1_score"     : round(f1, 4),
        "trained_at"   : ts,
        "n_cv_folds"   : CFG.CV_FOLDS,
        "cv_scoring"   : CFG.CV_SCORING,
        "classes"      : classes,
        "all_cv_results": df_cv.to_dict("records"),
    }
    with open(CFG.METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    # Save classification report
    with open(CFG.REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    with open(CFG.CM_JSON, "w") as f:
        json.dump({"matrix": cm, "classes": classes}, f, indent=2)

    print(f"\n  Pipeline saved  → {CFG.PIPELINE_PKL}")
    print(f"  Version backup  → models/versions/pipeline_{ts}.pkl")
    print(f"  Metadata saved  → {CFG.METADATA_JSON}")


# ── Main ─────────────────────────────────────────────────────────────
def run_training(tune=False):
    print("\n[TRAIN] Loading data...")
    X_train, X_test, y_train, y_test = load_and_split()

    print("\n[TRAIN] Cross-validating all models...")
    df_cv, best_name = cross_validate_all(X_train, y_train)

    if tune:
        print(f"\n[TRAIN] Tuning {best_name}...")
        best_pipe = tune_best_model(X_train, y_train, best_name)
    else:
        print(f"\n[TRAIN] Fitting {best_name} on full train set...")
        from pipeline import get_all_pipelines
        best_pipe = get_all_pipelines()[best_name]
        best_pipe.fit(X_train, y_train)

    print("\n[TRAIN] Evaluating on test set...")
    acc, f1, report, cm, classes = evaluate(best_pipe, X_test, y_test)

    print("\n[TRAIN] Extracting feature importance...")
    extract_feature_importance(best_pipe)

    print("\n[TRAIN] Saving pipeline...")
    save_pipeline(best_pipe, best_name, acc, f1, df_cv, report, cm, classes)

    return best_pipe, best_name, acc, f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV on best model")
    args = parser.parse_args()
    run_training(tune=args.tune)
    print("\n[TRAIN] Done.")
