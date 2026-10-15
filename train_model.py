"""
Model Training & Evaluation Pipeline
AI-Powered Fake News Detector

Downloads the ISOT Fake News Dataset (44,000+ articles), trains models with
proper train/test splits, computes REAL metrics, and saves everything to disk.

Usage:
    python train_model.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import joblib

# Ensure safe stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing import preprocess_text

# Paths
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
COMBINED_CSV = os.path.join(DATA_DIR, "combined_news.csv")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
PAC_MODEL_FILE = os.path.join(MODELS_DIR, "pac_pipeline.joblib")
LR_MODEL_FILE = os.path.join(MODELS_DIR, "lr_pipeline.joblib")


def download_file(url, target_path, description):
    """Download a file with streaming and progress reporting."""
    import requests
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"    [OK] {description} already cached: {target_path} ({os.path.getsize(target_path) / 1024 / 1024:.1f} MB)")
        return
    print(f"    Downloading {description} from {url}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(target_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 512):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = (downloaded / total) * 100
                    print(f"\r      Progress: {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB ({pct:.1f}%)", end="", flush=True)
    print("\n    [OK] Download complete.")


def download_isot_dataset():
    """
    Download the ISOT Fake News Dataset from public GitHub mirror.
    True.csv (~21,000 real articles) + Fake.csv (~23,000 fake articles).
    """
    print("[*] Downloading ISOT Fake News Dataset...")

    os.makedirs(DATA_DIR, exist_ok=True)

    true_url = "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/True.csv"
    fake_url = "https://raw.githubusercontent.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/main/Fake.csv"

    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")

    try:
        download_file(true_url, true_path, "True.csv (real news)")
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1  # Real = 1
        print(f"    [OK] Loaded {len(true_df)} real articles")

        download_file(fake_url, fake_path, "Fake.csv (fake news)")
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0  # Fake = 0
        print(f"    [OK] Loaded {len(fake_df)} fake articles")

        # Combine title + text into a single 'text' column for richer features
        for df in [true_df, fake_df]:
            df['text'] = df['title'].fillna('') + '. ' + df['text'].fillna('')

        combined = pd.concat([true_df[['text', 'label']], fake_df[['text', 'label']]], ignore_index=True)

        # Shuffle the dataset
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # Remove empty/very short texts
        combined = combined[combined['text'].str.len() > 50].reset_index(drop=True)

        combined.to_csv(COMBINED_CSV, index=False)
        print(f"    [OK] Combined dataset saved: {len(combined)} articles -> {COMBINED_CSV}")
        return combined

    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("    Try manually downloading from:")
        print("    https://github.com/Reinforz/Fake-News-detection-with-ISOT-Dataset")
        sys.exit(1)


def load_dataset():
    """Load cached dataset or download if not present."""
    if os.path.exists(COMBINED_CSV):
        print(f"[*] Loading cached dataset from {COMBINED_CSV}")
        df = pd.read_csv(COMBINED_CSV)
        print(f"    [OK] Loaded {len(df)} articles")
        return df
    else:
        return download_isot_dataset()


def train_and_evaluate():
    """
    Full training pipeline:
    1. Load/download dataset
    2. Preprocess text
    3. Train/test split (75/25, stratified)
    4. Train PAC + LR models
    5. Evaluate with real metrics
    6. 5-fold cross-validation
    7. Save models + metrics to disk
    """
    print("=" * 65)
    print("  AI-Powered Fake News Detector -- Training Pipeline")
    print("  ISOT Dataset | Real Metrics | Proper Evaluation")
    print("=" * 65)

    # Step 1: Load data
    df = load_dataset()
    print(f"\n[*] Dataset shape: {df.shape}")
    print(f"[*] Label distribution:")
    print(f"    Real (1): {(df['label'] == 1).sum()}")
    print(f"    Fake (0): {(df['label'] == 0).sum()}")

    # Step 2: Preprocess
    print("\n[*] Preprocessing text (lemmatization + stopword removal)...")
    start_time = time.time()
    df['clean_text'] = df['text'].apply(preprocess_text)
    preprocess_time = time.time() - start_time
    print(f"    [OK] Preprocessed {len(df)} articles in {preprocess_time:.1f}s")

    # Remove any rows where preprocessing produced empty strings
    df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
    print(f"    [OK] {len(df)} articles retained after filtering empty results")

    X = df['clean_text']
    y = df['label']

    # Step 3: Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n[*] Train/Test split: {len(X_train)} train / {len(X_test)} test (75/25, stratified)")

    # Step 4: Train models
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {
        "dataset": {
            "name": "ISOT Fake News Dataset",
            "total_articles": len(df),
            "real_count": int((df['label'] == 1).sum()),
            "fake_count": int((df['label'] == 0).sum()),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "preprocessing": {
            "lemmatization": True,
            "stopword_removal": True,
            "html_url_stripping": True,
            "preprocess_time_seconds": round(preprocess_time, 2)
        },
        "models": {}
    }

    # --- Model 1: Passive-Aggressive Classifier ---
    print("\n" + "-" * 50)
    print("  Model 1: Passive-Aggressive Classifier")
    print("-" * 50)

    pac_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('clf', PassiveAggressiveClassifier(
            max_iter=1000,
            C=0.5,
            random_state=42
        ))
    ])

    start_time = time.time()
    pac_pipeline.fit(X_train, y_train)
    pac_train_time = time.time() - start_time

    pac_preds = pac_pipeline.predict(X_test)
    pac_acc = accuracy_score(y_test, pac_preds)
    pac_prec = precision_score(y_test, pac_preds, zero_division=0)
    pac_rec = recall_score(y_test, pac_preds, zero_division=0)
    pac_f1 = f1_score(y_test, pac_preds, zero_division=0)
    pac_cm = confusion_matrix(y_test, pac_preds)

    print(f"  Accuracy : {pac_acc * 100:.2f}%")
    print(f"  Precision: {pac_prec * 100:.2f}%")
    print(f"  Recall   : {pac_rec * 100:.2f}%")
    print(f"  F1-Score : {pac_f1 * 100:.2f}%")
    print(f"  Train Time: {pac_train_time:.2f}s")
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={pac_cm[0][0]:5d}  FP={pac_cm[0][1]:5d}]")
    print(f"     [FN={pac_cm[1][0]:5d}  TP={pac_cm[1][1]:5d}]]")

    # 5-fold cross-validation
    print("\n  Running 5-fold cross-validation...")
    pac_cv_scores = cross_val_score(pac_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=1)
    print(f"  CV Accuracy: {pac_cv_scores.mean() * 100:.2f}% (±{pac_cv_scores.std() * 100:.2f}%)")

    joblib.dump(pac_pipeline, PAC_MODEL_FILE)
    print(f"  [OK] Model saved: {PAC_MODEL_FILE}")

    metrics["models"]["passive_aggressive"] = {
        "accuracy": round(pac_acc * 100, 2),
        "precision": round(pac_prec * 100, 2),
        "recall": round(pac_rec * 100, 2),
        "f1_score": round(pac_f1 * 100, 2),
        "confusion_matrix": pac_cm.tolist(),
        "cv_accuracy_mean": round(pac_cv_scores.mean() * 100, 2),
        "cv_accuracy_std": round(pac_cv_scores.std() * 100, 2),
        "train_time_seconds": round(pac_train_time, 2),
        "model_file": "pac_pipeline.joblib"
    }

    # --- Model 2: Logistic Regression ---
    print("\n" + "-" * 50)
    print("  Model 2: Logistic Regression")
    print("-" * 50)

    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ))
    ])

    start_time = time.time()
    lr_pipeline.fit(X_train, y_train)
    lr_train_time = time.time() - start_time

    lr_preds = lr_pipeline.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds, zero_division=0)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)

    print(f"  Accuracy : {lr_acc * 100:.2f}%")
    print(f"  Precision: {lr_prec * 100:.2f}%")
    print(f"  Recall   : {lr_rec * 100:.2f}%")
    print(f"  F1-Score : {lr_f1 * 100:.2f}%")
    print(f"  Train Time: {lr_train_time:.2f}s")
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={lr_cm[0][0]:5d}  FP={lr_cm[0][1]:5d}]")
    print(f"     [FN={lr_cm[1][0]:5d}  TP={lr_cm[1][1]:5d}]]")

    print("\n  Running 5-fold cross-validation...")
    lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=1)
    print(f"  CV Accuracy: {lr_cv_scores.mean() * 100:.2f}% (±{lr_cv_scores.std() * 100:.2f}%)")

    joblib.dump(lr_pipeline, LR_MODEL_FILE)
    print(f"  [OK] Model saved: {LR_MODEL_FILE}")

    metrics["models"]["logistic_regression"] = {
        "accuracy": round(lr_acc * 100, 2),
        "precision": round(lr_prec * 100, 2),
        "recall": round(lr_rec * 100, 2),
        "f1_score": round(lr_f1 * 100, 2),
        "confusion_matrix": lr_cm.tolist(),
        "cv_accuracy_mean": round(lr_cv_scores.mean() * 100, 2),
        "cv_accuracy_std": round(lr_cv_scores.std() * 100, 2),
        "train_time_seconds": round(lr_train_time, 2),
        "model_file": "lr_pipeline.joblib"
    }

    # Full classification reports
    print("\n" + "=" * 65)
    print("  FULL CLASSIFICATION REPORTS")
    print("=" * 65)
    print("\n--- Passive-Aggressive Classifier ---")
    print(classification_report(y_test, pac_preds, target_names=["Fake", "Real"]))
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, lr_preds, target_names=["Fake", "Real"]))

    # Save metrics
    metrics["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[*] Metrics saved: {METRICS_FILE}")

    print("\n" + "=" * 65)
    print("  [OK] TRAINING COMPLETE")
    print(f"  [OK] Dataset: {len(df)} articles (ISOT)")
    print(f"  [OK] PAC Accuracy: {pac_acc * 100:.2f}% | LR Accuracy: {lr_acc * 100:.2f}%")
    print(f"  [OK] Models saved to: {MODELS_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate()
