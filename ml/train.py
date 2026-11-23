#!/usr/bin/env python3
"""
ml/train.py
Feature extraction pipeline, 5 machine learning algorithms evaluation,
5-fold stratified CV on TRAIN split, held-out test evaluation, and metrics reporting.
"""

import os
import sys
import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 5 Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.constants import (
    APP_AREAS,
    PRIORITIES,
    LINK_STATUSES,
    DEVICES,
    TEXT_FEATURE,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ALL_FEATURE_COLUMNS,
    LABEL_COLUMN
)

DATA_PATH = os.path.join(BASE_DIR, "data", "app_issues_demo.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "priority_model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")


def load_dataset(csv_path=DATA_PATH, extra_df=None):
    """
    Loads dataset from CSV and optionally appends extra_df (e.g. manager corrections).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}. Please run make_demo_data.py first.")

    df = pd.read_csv(csv_path)

    corrections_count = 0
    if extra_df is not None and not extra_df.empty:
        corrections_count = len(extra_df)
        df = pd.concat([df, extra_df], ignore_index=True)

    # Standardize types and fillna
    df["description"] = df["description"].fillna("").astype(str)
    df["app_area"] = df["app_area"].fillna("Other").astype(str)
    df["link_status"] = df["link_status"].fillna("None").astype(str)
    df["device"] = df["device"].fillna("Desktop").astype(str)
    df["contacts_shared"] = pd.to_numeric(df["contacts_shared"], errors="coerce").fillna(0).astype(float)
    df["hours_since_link_created"] = pd.to_numeric(df["hours_since_link_created"], errors="coerce").fillna(0).astype(float)
    df["issues_last_30d"] = pd.to_numeric(df["issues_last_30d"], errors="coerce").fillna(0).astype(float)
    df["priority"] = df["priority"].astype(str)

    # Filter invalid priorities if any
    df = df[df["priority"].isin(PRIORITIES)].copy()

    # Drop duplicate (description + context) rows before splitting
    context_columns = [
        "description",
        "app_area",
        "link_status",
        "contacts_shared",
        "hours_since_link_created",
        "device",
        "issues_last_30d"
    ]
    df = df.drop_duplicates(subset=context_columns).reset_index(drop=True)

    return df, corrections_count


def build_preprocessor():
    """
    Column transformer:
    - TF-IDF on description (unigram + bigram)
    - OneHotEncoder on categorical features (app_area, link_status, device)
    - MinMaxScaler on numeric features (contacts_shared, hours_since_link_created, issues_last_30d)
      (MinMaxScaler used instead of StandardScaler because MultinomialNB requires non-negative values)
    """
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=4000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True
                ),
                TEXT_FEATURE
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                MinMaxScaler(),
                NUMERIC_FEATURES
            )
        ]
    )


def train_all_models(extra_df=None, csv_path=DATA_PATH):
    """
    Trains and compares 5 ML classifiers.
    Signature: train_all_models(extra_df=None).
    Picks the best model by 5-fold stratified CV macro-F1 on the TRAIN part only.
    Reports metrics on held-out test set and saves best model pipeline and metrics.json.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df, corrections_count = load_dataset(csv_path, extra_df=extra_df)

    feature_cols = [TEXT_FEATURE] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df[LABEL_COLUMN]
    classes = PRIORITIES  # ["High", "Medium", "Low"]

    # Calculate majority-class baseline accuracy
    majority_class_count = y.value_counts().max()
    majority_baseline_accuracy = round(float(majority_class_count / len(y)) * 100, 2)

    # Stratified 80/20 train/test split with fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.5),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=25, random_state=42),
        "Support Vector Machine (LinearSVC)": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42, max_iter=2000, C=1.0)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42)
    }

    # 5-fold Stratified CV on TRAIN split only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = {}
    for name, clf in algorithms.items():
        fold_macro_f1s = []
        fold_high_recalls = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", clf)
            ])
            fold_pipeline.fit(X_fold_train, y_fold_train)
            y_fold_pred = fold_pipeline.predict(X_fold_val)

            fold_macro_f1 = f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0)
            fold_report = classification_report(y_fold_val, y_fold_pred, output_dict=True, zero_division=0)
            high_rec = fold_report.get("High", {}).get("recall", 0.0)

            fold_macro_f1s.append(fold_macro_f1)
            fold_high_recalls.append(high_rec)

        mean_cv_macro_f1 = float(np.mean(fold_macro_f1s))
        mean_cv_high_rec = float(np.mean(fold_high_recalls))
        cv_scores[name] = {
            "cv_macro_f1": round(mean_cv_macro_f1 * 100, 2),
            "cv_high_recall": round(mean_cv_high_rec * 100, 2)
        }

    # Train on full X_train and evaluate on held-out test set
    results = {}
    fitted_pipelines = {}

    for name, clf in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed_sec = round(time.time() - start_time, 4)

        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=classes).tolist()

        per_class_metrics = {}
        for cls_name in classes:
            if cls_name in report:
                per_class_metrics[cls_name] = {
                    "precision": round(report[cls_name]["precision"] * 100, 2),
                    "recall": round(report[cls_name]["recall"] * 100, 2),
                    "f1_score": round(report[cls_name]["f1-score"] * 100, 2),
                    "support": int(report[cls_name]["support"])
                }

        results[name] = {
            "name": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "cv_macro_f1": cv_scores[name]["cv_macro_f1"],
            "cv_high_recall": cv_scores[name]["cv_high_recall"],
            "training_time_sec": elapsed_sec,
            "confusion_matrix": conf_matrix,
            "confusion_matrix_labels": classes,
            "per_class": per_class_metrics,
            "classification_report": report
        }
        fitted_pipelines[name] = pipeline

    # Select best model by 5-fold CV macro-F1 on TRAIN split (tie-breaker: higher CV high recall)
    best_algorithm = max(
        algorithms.keys(),
        key=lambda k: (cv_scores[k]["cv_macro_f1"], cv_scores[k]["cv_high_recall"])
    )
    best_pipeline = fitted_pipelines[best_algorithm]
    best_high_recall = results[best_algorithm]["per_class"].get("High", {}).get("recall", 0.0)

    # Save best pipeline
    joblib.dump(best_pipeline, MODEL_PATH)

    summary_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_used": len(df),
        "corrections_used": corrections_count,
        "dataset_total_samples": len(df),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "best_algorithm": best_algorithm,
        "best_f1_score": results[best_algorithm]["f1_score"],
        "high_recall": best_high_recall,
        "classes": classes,
        "models": results
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    return summary_metrics


if __name__ == "__main__":
    print("Training 5 ML algorithms on App Issue dataset...")
    metrics = train_all_models()
    print(f"\nMajority-class baseline accuracy: {metrics['majority_baseline_accuracy']}%")
    print(f"Total rows used: {metrics['rows_used']} (Train: {metrics['training_samples']}, Test: {metrics['testing_samples']})")
    print(f"Corrections used: {metrics['corrections_used']}")
    print(f"\n--- Model Comparison Summary (Selected Best: {metrics['best_algorithm']}) ---")
    for name, stats in metrics["models"].items():
        is_best = " [BEST]" if name == metrics["best_algorithm"] else ""
        high_rec = stats["per_class"].get("High", {}).get("recall", 0.0)
        print(
            f"{name:35} | CV-F1: {stats['cv_macro_f1']}% | Test-Acc: {stats['accuracy']}% | "
            f"Test-F1: {stats['f1_score']}% | High-Recall: {high_rec}%{is_best}"
        )
    print(f"\nHigh Recall of Best Model: {metrics['high_recall']}%")
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
