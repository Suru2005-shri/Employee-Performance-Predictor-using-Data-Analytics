"""
train_model.py
--------------
Trains multiple ML models, compares them, selects the best,
and saves it to models/ with full evaluation metrics.

Run standalone:  python src/train_model.py
Or called from:  main.py
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.metrics       import accuracy_score, classification_report, confusion_matrix, f1_score

# ── Make sure src/ is importable when run as  python src/train_model.py ──
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocess import get_train_test   # always works now


# ── Path helper (same pattern as preprocess.py) ──────────────────────
def _path(relative):
    return os.path.join(ROOT_DIR, relative)


# ── 1. Train ─────────────────────────────────────────────────────────
def train_all_models(X_train, y_train):
    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest'       : RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=150, random_state=42),
        'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
        'KNN'                 : KNeighborsClassifier(n_neighbors=7),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  [trained] {name}")
    return fitted


# ── 2. Evaluate all ──────────────────────────────────────────────────
def evaluate_models(fitted, X_test, y_test, le_target):
    results = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average='weighted')
        results.append({'Model': name, 'Accuracy': round(acc, 4), 'F1_Score': round(f1, 4)})
        print(f"  {name:<25}  Acc={acc:.4f}  F1={f1:.4f}")

    df_res    = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    best_name = df_res.loc[0, 'Model']
    print(f"\n  Best Model: {best_name}  (Acc={df_res.loc[0,'Accuracy']})")
    return df_res, best_name


# ── 3. Detailed report ───────────────────────────────────────────────
def detailed_report(model, X_test, y_test, le_target, model_name):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le_target.classes_,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    out_dir = _path('outputs')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({'matrix': cm, 'classes': list(le_target.classes_)}, f, indent=2)

    print(f"  Reports saved to outputs/")
    return report, cm


# ── 4. Feature importance ────────────────────────────────────────────
def get_feature_importance(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        print(f"  [{model_name}] does not expose feature_importances_ — skipping")
        return None

    fi = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    out_path = _path('outputs/feature_importance.csv')
    fi.to_csv(out_path, index=False)
    print(f"\n  Top 5 Features:\n{fi.head().to_string(index=False)}")
    return fi


# ── 5. Save best model ───────────────────────────────────────────────
def save_best_model(model, model_name, df_results):
    models_dir = _path('models')
    out_dir    = _path('outputs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(out_dir,    exist_ok=True)

    pickle.dump(model, open(os.path.join(models_dir, 'best_model.pkl'), 'wb'))
    df_results.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)

    meta = {
        'best_model': model_name,
        'accuracy'  : float(df_results.loc[0, 'Accuracy']),
        'f1_score'  : float(df_results.loc[0, 'F1_Score']),
        'all_results': df_results.to_dict('records')
    }
    with open(os.path.join(out_dir, 'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Best model saved: models/best_model.pkl")
    print(f"  Comparison CSV  : outputs/model_comparison.csv")


# ── Standalone entry point ───────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  EMPLOYEE PERFORMANCE PREDICTOR - MODEL TRAINING")
    print("=" * 55)

    print("\n[1] Preprocessing data...")
    X_train, X_test, y_train, y_test, le_target, feat_names = get_train_test()

    print("\n[2] Training models...")
    fitted = train_all_models(X_train, y_train)

    print("\n[3] Evaluating models...")
    df_results, best_name = evaluate_models(fitted, X_test, y_test, le_target)
    best_model = fitted[best_name]

    print("\n[4] Detailed classification report...")
    detailed_report(best_model, X_test, y_test, le_target, best_name)

    print("\n[5] Feature importance...")
    get_feature_importance(best_model, feat_names, best_name)

    print("\n[6] Saving best model...")
    save_best_model(best_model, best_name, df_results)

    print("\n  Training complete!")
