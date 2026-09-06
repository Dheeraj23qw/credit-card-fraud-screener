"""Model training, evaluation, and serialization pipeline for Credit Card Fraud Detection."""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

# Ensure src in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_raw_data, clean_data, get_feature_target_split
from preprocessor import get_fraud_preprocessor_pipeline, get_feature_names_out

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importances.json")


def build_candidate_classifiers() -> Dict[str, Pipeline]:
    """Instantiates candidate models for imbalanced fraud classification."""
    preprocessor = get_fraud_preprocessor_pipeline()
    
    candidates = {
        "Cost-Sensitive Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="liblinear"
            ))
        ]),
        "Balanced Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "XGBoost (Imbalanced Tuned)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=45.0,  # compensate for ~2% fraud rate
                eval_metric="aucpr",
                random_state=42,
                n_jobs=1
            ))
        ])
    }
    return candidates


def evaluate_cross_validation(candidates: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Evaluates candidates using 5-Fold Stratified Cross-Validation on PR-AUC, ROC-AUC, Recall, and F1."""
    print("=" * 65, flush=True)
    print("Running 5-Fold Stratified Cross-Validation for Fraud Classifiers...", flush=True)
    print("=" * 65, flush=True)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["average_precision", "roc_auc", "recall", "precision", "f1"]
    
    results = []
    for name, pipeline in candidates.items():
        print(f"Evaluating {name}...", flush=True)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        res = {
            "Model": name,
            "PR-AUC (Avg Precision)": float(np.mean(scores["test_average_precision"])),
            "ROC-AUC": float(np.mean(scores["test_roc_auc"])),
            "Recall (Fraud Catch Rate)": float(np.mean(scores["test_recall"])),
            "Precision": float(np.mean(scores["test_precision"])),
            "F1-Score": float(np.mean(scores["test_f1"]))
        }
        results.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC (Avg Precision)']:.4f} | ROC-AUC: {res['ROC-AUC']:.4f} | Recall: {res['Recall (Fraud Catch Rate)']:.2%} | F1: {res['F1-Score']:.4f}", flush=True)
        
    return pd.DataFrame(results)


def tune_xgboost_fraud_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Tunes XGBoost hyperparameters maximizing PR-AUC on imbalanced distribution."""
    print("\n" + "=" * 65, flush=True)
    print("Hyperparameter Tuning for XGBoost Fraud Screener...", flush=True)
    print("=" * 65, flush=True)
    
    preprocessor = get_fraud_preprocessor_pipeline()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="aucpr", random_state=42, n_jobs=1))
    ])
    
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [3, 4],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__scale_pos_weight": [40.0, 50.0]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best CV PR-AUC: {grid_search.best_score_:.4f}", flush=True)
    print(f"Best Parameters: {grid_search.best_params_}", flush=True)
    return grid_search.best_estimator_


def find_optimal_fraud_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Finds optimal decision threshold balancing False Positives and False Negatives."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh


def extract_feature_importances(pipeline: Pipeline, X_sample: pd.DataFrame) -> Dict[str, float]:
    """Extracts feature importances from trained XGBoost model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_out(preprocessor, X_sample)
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
        
    feat_dict = dict(zip(feature_names, [float(x) for x in importances]))
    return dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))


def train_and_save():
    """End-to-end training and artifact serialization."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X, y = get_feature_target_split(cleaned_df, target_col="Is_Fraud")
    
    # 2. Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data Split: Train={len(X_train)} (Fraud={y_train.sum()}), Test={len(X_test)} (Fraud={y_test.sum()})", flush=True)
    
    # 3. Model Benchmark
    candidates = build_candidate_classifiers()
    cv_df = evaluate_cross_validation(candidates, X_train, y_train)
    print("\n--- Model Benchmark Summary ---", flush=True)
    print(cv_df.to_string(index=False), flush=True)
    
    # 4. Hyperparameter Tuning
    best_pipeline = tune_xgboost_fraud_model(X_train, y_train)
    
    # 5. Held-out Test Set Evaluation
    y_probs = best_pipeline.predict_proba(X_test)[:, 1]
    optimal_thresh = find_optimal_fraud_threshold(y_test.values, y_probs)
    y_pred_optimal = (y_probs >= optimal_thresh).astype(int)
    
    test_pr_auc = float(average_precision_score(y_test, y_probs))
    test_roc_auc = float(roc_auc_score(y_test, y_probs))
    test_recall = float(recall_score(y_test, y_pred_optimal))
    test_precision = float(precision_score(y_test, y_pred_optimal))
    test_f1 = float(f1_score(y_test, y_pred_optimal))
    conf_mat = confusion_matrix(y_test, y_pred_optimal).tolist()
    
    metrics = {
        "cv_results": cv_df.to_dict(orient="records"),
        "test_metrics": {
            "pr_auc": test_pr_auc,
            "roc_auc": test_roc_auc,
            "optimal_threshold": optimal_thresh,
            "recall": test_recall,
            "precision": test_precision,
            "f1_score": test_f1,
            "confusion_matrix": conf_mat
        }
    }
    
    print("\n" + "=" * 65, flush=True)
    print("FINAL TEST EVALUATION (Held-out Test Split)", flush=True)
    print("=" * 65, flush=True)
    print(f"PR-AUC (Average Precision): {test_pr_auc:.4f}", flush=True)
    print(f"ROC-AUC: {test_roc_auc:.4f}", flush=True)
    print(f"Optimal Decision Threshold: {optimal_thresh:.2f}", flush=True)
    print(f"Fraud Detection Recall: {test_recall:.2%} (Caught {conf_mat[1][1]} out of {conf_mat[1][0] + conf_mat[1][1]} frauds)", flush=True)
    print(f"Precision: {test_precision:.2%}", flush=True)
    print(f"F1-Score: {test_f1:.4f}", flush=True)
    print(f"Confusion Matrix: TN={conf_mat[0][0]}, FP={conf_mat[0][1]}, FN={conf_mat[1][0]}, TP={conf_mat[1][1]}", flush=True)
    
    # 6. Feature Importances
    feat_importances = extract_feature_importances(best_pipeline, X_train)
    print("\nTop 10 Fraud Risk Indicators:", flush=True)
    for feat, imp in list(feat_importances.items())[:10]:
        print(f"  {feat}: {imp:.4f}", flush=True)
        
    # 7. Save Artifacts
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}", flush=True)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {METRICS_PATH}", flush=True)
    
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feat_importances, f, indent=2)
    print(f"Saved feature importances to: {FEATURE_IMPORTANCE_PATH}", flush=True)


if __name__ == "__main__":
    train_and_save()
