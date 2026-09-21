"""
Machine Learning Training & Evaluation Module
=============================================
Academic AI Decision-Support System

Trains, validates, and compares classification models (Random Forest, Logistic Regression,
Decision Tree) to predict donor emergency response likelihood.

Saves the best performing model to `models/donor_response_model.pkl` and evaluation
metrics to `models/model_metrics.pkl`.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import ML_FEATURE_NAMES, calculate_distance
from src.data_generator import generate_synthetic_donors, save_and_init_db


def load_or_generate_dataset(data_path: str = "data/donors.csv") -> pd.DataFrame:
    """Load existing synthetic dataset or generate if missing."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df, _ = save_and_init_db()
    return df


def engineer_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Synthesize realistic distance_km feature for training from hospital reference points.
    """
    df_copy = df.copy()
    
    # Reference hospital: Colombo General (6.9271, 79.8612)
    ref_lat, ref_lon = 6.9271, 79.8612
    if "distance_km" not in df_copy.columns:
        df_copy["distance_km"] = calculate_distance(
            df_copy["latitude"],
            df_copy["longitude"],
            ref_lat,
            ref_lon
        )
    
    X = df_copy[ML_FEATURE_NAMES]
    y = df_copy["target_response"]
    return X, y


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and benchmark multiple classification algorithms.
    """
    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=6,
            random_state=random_state
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("\n" + "="*70)
    print(f"{'MODEL PERFORMANCE BENCHMARKING (Stratified 80/20 & 5-Fold CV)':^70}")
    print("="*70)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} | {'ROC-AUC':<9}")
    print("-" * 70)

    best_model_name = "Random Forest"
    best_f1 = 0.0
    best_fitted_model = None

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions on holdout test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        results[name] = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cv_f1_std": round(float(np.std(cv_scores)), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "y_test": y_test.values.tolist(),
            "y_prob": y_prob.tolist()
        }

        print(f"{name:<22} | {acc:.4f}    | {prec:.4f}    | {rec:.4f}    | {f1:.4f}    | {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_fitted_model = model

    # Feature Importance for Random Forest
    rf_model = models["Random Forest"]
    feature_importances = dict(zip(ML_FEATURE_NAMES, [round(float(v), 4) for v in rf_model.feature_importances_]))
    # Sort descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    results["feature_importances"] = feature_importances
    results["best_model_name"] = best_model_name
    results["feature_names"] = ML_FEATURE_NAMES
    results["test_set_size"] = len(y_test)
    results["train_set_size"] = len(y_train)

    print("-" * 70)
    print(f" Best Model Selected: {best_model_name} (F1 Score: {best_f1:.4f})")
    print("=" * 70 + "\n")

    return results, best_fitted_model


def save_trained_artifacts(
    results: Dict[str, Any],
    best_model: Any,
    models_dir: str = "models"
) -> None:
    """Save best model and comprehensive metrics package."""
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "donor_response_model.pkl")
    joblib.dump(best_model, model_path)
    print(f" Saved best trained model to '{model_path}'")

    # Clean the results dict of non-serializable objects before saving
    metrics_to_save = {}
    for k, v in results.items():
        if k in ["Random Forest", "Logistic Regression", "Decision Tree"]:
            clean_model_dict = {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "roc_auc": v["roc_auc"],
                "cv_f1_mean": v["cv_f1_mean"],
                "cv_f1_std": v["cv_f1_std"],
                "confusion_matrix": v["confusion_matrix"],
                "classification_report": v["classification_report"]
            }
            metrics_to_save[k] = clean_model_dict
        else:
            metrics_to_save[k] = v

    metrics_path = os.path.join(models_dir, "model_metrics.pkl")
    joblib.dump(metrics_to_save, metrics_path)
    print(f" Saved model evaluation benchmark metrics to '{metrics_path}'")


def run_training_pipeline() -> Tuple[Dict[str, Any], Any]:
    """Execute complete model training and benchmarking pipeline."""
    df = load_or_generate_dataset()
    X, y = engineer_training_features(df)
    results, best_model = train_and_evaluate_models(X, y)
    save_trained_artifacts(results, best_model)
    return results, best_model


if __name__ == "__main__":
    run_training_pipeline()
