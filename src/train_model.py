"""
train_model.py
--------------
Trains multiple ML models, compares them, selects the best,
and saves it with evaluation metrics.
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

from preprocess import get_train_test


def train_all_models(X_train, y_train):
    """Train multiple classifiers and return fitted models dict."""
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
        print(f"  ✔ Trained: {name}")
    return fitted


def evaluate_models(fitted, X_test, y_test, le_target):
    """Evaluate all models; return comparison DataFrame + best model name."""
    results = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average='weighted')
        results.append({'Model': name, 'Accuracy': round(acc, 4), 'F1_Score': round(f1, 4)})
        print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}")

    df_res = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    best_name = df_res.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_name} (Acc={df_res.iloc[0]['Accuracy']})")
    return df_res, best_name


def detailed_report(model, X_test, y_test, le_target, model_name):
    """Print + save classification report and confusion matrix."""
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open('../outputs/confusion_matrix.json', 'w') as f:
        json.dump({'matrix': cm, 'classes': list(le_target.classes_)}, f, indent=2)

    print("✅ Reports saved to outputs/")
    return report, cm


def save_best_model(best_model, best_name, df_results, scaler, le_target, le_dict, feat_names):
    import os
    import pickle

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models")

    os.makedirs(model_path, exist_ok=True)

    # Save model
    pickle.dump(best_model, open(os.path.join(model_path, "best_model.pkl"), 'wb'))

    # Save preprocessing artifacts
    pickle.dump(scaler, open(os.path.join(model_path, "scaler.pkl"), 'wb'))
    pickle.dump(le_target, open(os.path.join(model_path, "le_target.pkl"), 'wb'))
    pickle.dump(le_dict, open(os.path.join(model_path, "le_dict.pkl"), 'wb'))
    pickle.dump(feat_names, open(os.path.join(model_path, "feature_names.pkl"), 'wb'))

    print("✅ All model artifacts saved in models/")


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importances if available."""
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'Feature'   : feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fi.to_csv('../outputs/feature_importance.csv', index=False)
        print("\n🔑 Top 5 Features:")
        print(fi.head())
        return fi
    return None


if __name__ == '__main__':
    print("=" * 55)
    print("  EMPLOYEE PERFORMANCE PREDICTOR — MODEL TRAINING")
    print("=" * 55)

    X_train, X_test, y_train, y_test, scaler, le_target, le_dict, feat_names = get_train_test()

    print("\n[1] Training models...")
    fitted = train_all_models(X_train, y_train)

    print("\n[2] Evaluating models...")
    df_results, best_name = evaluate_models(fitted, X_test, y_test, le_target)

    best_model = fitted[best_name]

    print("\n[3] Detailed report for best model...")
    detailed_report(best_model, X_test, y_test, le_target, best_name)

    print("\n[4] Feature importance...")
    get_feature_importance(best_model, feat_names, best_name)

    print("\n[5] Saving best model...")
    save_best_model(best_model, best_name, df_results)

    print("\n✅ Training complete!")
