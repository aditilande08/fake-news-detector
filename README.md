# 🛡️ AI-Powered Fake News & Misinformation Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20TF--IDF-success.svg)](https://www.nltk.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.8%25-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg)]()

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to evaluate article credibility, identify fabricated stories, and detect sensationalist misinformation in real time. 

Built with **Python**, **Scikit-Learn**, **NLTK**, and **Streamlit**, the application features live article scraping, dual-model classification, heuristic anomaly scanning, and interactive credibility scoring.

---

## 📌 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset Overview](#-dataset-overview)
- [Empirical Benchmark Performance](#-empirical-benchmark-performance)
- [Features](#-features)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Use the Application](#-how-to-use-the-application)
- [Running Automated Tests](#-running-automated-tests)
- [Retraining the Models](#-retraining-the-models)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🚀 Key Highlights

- **NLP Classification Pipeline:** Dual-algorithm architecture featuring **Passive-Aggressive Classifier** and **Logistic Regression** trained on **44,881 full-text articles** using **TF-IDF vectorization** with unigram and bigram feature representations.
- **High Empirical Accuracy:** Reached **99.79% test accuracy** and **99.77% 5-fold cross-validation accuracy** with balanced precision (99.74%) and recall (99.81%).
- **Centralized Preprocessing Engine:** Custom NLP cleaning pipeline implementing HTML/URL stripping, regex tokenization, stopword removal, and **NLTK WordNet Lemmatization**.
- **Heuristic Anomaly Detection:** Real-time scanning for clickbait keywords, emotional manipulation, ALL-CAPS shouting, irregular punctuation, and subjectivity indexing.
- **Interactive Streamlit Web Dashboard:** Clean UI enabling real-time URL article scraping, raw text analysis, instant confidence scoring (0–100%), and interactive model benchmark comparisons.
- **Enterprise-Grade Code Quality:** Fully modular code architecture with 22 automated unit tests passing at 100%.

---

## 🏗️ System Architecture

```
                       [ Input: Live News URL or Raw Text ]
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 1. INGESTION & SCRAPING ENGINE                │
                 │    - Beautiful Soup 4 + urllib                │
                 │    - DOM parsing, article body extraction     │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 2. NLP PREPROCESSING PIPELINE (src/)          │
                 │    - HTML tag & URL stripping                 │
                 │    - Alphanumeric regex tokenization          │
                 │    - 179 NLTK English stopwords removal       │
                 │    - NLTK WordNet Lemmatization               │
                 └───────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
┌───────────────────────────────────────┐ ┌─────────────────────────────────────────┐
│ 3. ML CLASSIFICATION PIPELINE         │ │ 4. HEURISTIC & ANOMALY ENGINE           │
│    - TF-IDF Vectorizer (1-2 n-grams,  │ │    - Clickbait & conspiracy phrase scan │
│      50,000 max features, sublinear)  │ │    - ALL-CAPS shouting ratio calculation│
│    - Passive-Aggressive Classifier    │ │    - Sentiment polarity (-1.0 to +1.0)  │
│    - Logistic Regression Classifier   │ │    - Subjectivity index (0.0 to 1.0)    │
│    - Ensemble Voting Logic            │ │    - Excessive punctuation detection    │
└───────────────────┬───────────────────┘ └────────────────────┬────────────────────┘
                    │                                          │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ 5. STREAMLIT VERIFICATION DASHBOARD (app.py)  │
                 │    - Authenticity Badge: REAL vs. FAKE        │
                 │    - Confidence Percentage & Score Gauges     │
                 │    - Anomaly Warning Pills & Sentiment Bar    │
                 │    - Live Confusion Matrix & Metrics Display  │
                 └───────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core programming language |
| **Machine Learning** | Scikit-Learn | Passive-Aggressive Classifier, Logistic Regression, Pipeline, Cross-Validation |
| **Natural Language Processing** | NLTK | Stopword filtering, WordNet Lemmatizer, corpus handling |
| **Feature Extraction** | TF-IDF Vectorizer | Unigram & Bigram feature extraction (50,000 dimensions) |
| **Data Manipulation** | Pandas, NumPy | Dataset combining, splitting, label encoding, and metrics computation |
| **Model Serialization** | Joblib | High-efficiency pipeline serialization and disk caching |
| **Web Scraping** | BeautifulSoup4, urllib | Real-time article retrieval and DOM text extraction from live URLs |
| **Web Dashboard** | Streamlit | Reactive frontend web dashboard and visualization |
| **Testing** | Unittest | 22 automated test cases verifying NLP and model integrity |

---

## 📊 Dataset Overview

The models are trained and evaluated on the **ISOT Fake News Dataset** (University of Victoria):
- **Real News Articles (`True.csv`):** 21,416 genuine news articles sourced from Reuters covering global politics, business, and technology.
- **Fake News Articles (`Fake.csv`):** 23,465 fabricated, misleading, and flagged news stories compiled from unreliable sources and fact-checking archives.
- **Total Dataset Size:** **44,881 verified articles**.
- **Split Configuration:** 75% Training (33,660 articles) and 25% Testing (11,221 articles), stratified by class label.

---

## 📈 Empirical Benchmark Performance

All metrics are computed on the **11,221 held-out test articles** and verified via **5-fold stratified cross-validation** across all 44,881 articles:

| Metric | Passive-Aggressive Classifier | Logistic Regression | Evaluation Notes |
|---|---|---|---|
| **Test Accuracy** | **99.79%** | **99.39%** | Evaluated on 11,221 unseen test articles |
| **Precision** | **99.74%** | **99.27%** | Minimized false accusations of legitimate news |
| **Recall** | **99.81%** | **99.44%** | High sensitivity in catching fabricated stories |
| **F1-Score** | **99.78%** | **99.36%** | Balanced harmonic mean across both classes |
| **5-Fold Cross Validation** | **99.77% (±0.02%)** | **99.25% (±0.08%)** | Consistent generalization across all 5 folds |
| **Training Duration** | ~28.5 seconds | ~41.4 seconds | Multi-threaded Scikit-Learn training |
| **Inference Latency** | **< 15ms** | **< 15ms** | Real-time per-article classification |

### Confusion Matrix Breakdown (Test Set: 11,221 Articles)

#### Passive-Aggressive Classifier:
```
                    Predicted Fake      Predicted Real
  Actual Fake:         5,853 (TN)            14 (FP)
  Actual Real:            10 (FN)         5,344 (TP)
```

#### Logistic Regression:
```
                    Predicted Fake      Predicted Real
  Actual Fake:         5,828 (TN)            39 (FP)
  Actual Real:            30 (FN)         5,324 (TP)
```

---

## ✨ Features

1. **Dual-Model Inference & Ensembling:** Choose between Passive-Aggressive Classifier, Logistic Regression, or an Ensemble Voting mode that combines both models.
2. **Real-Time Web Scraping:** Paste any public article URL (e.g., Reuters, BBC, blogs); the app automatically extracts the headline and body for verification.
3. **Quick Presets:** Built-in verified news benchmarks for instant testing of real economic reports versus viral clickbait scams.
4. **Heuristic Misinformation Signals:**
   - Detects clickbait triggers (*"secret trick"*, *"miracle cure"*, *"they don't want you to know"*).
   - Measures capitalization shouting ratio (`CAPS_RATIO`).
   - Flags sensationalist and repeated punctuation (`???`, `!!!`).
   - Computes text subjectivity and emotional polarity.
5. **Transparency & Metrics:** Live dashboard view showing the exact training dataset size, training timestamp, confusion matrix, and precision/recall curves.

---

## 📁 Project Directory Structure

```
fake-news-detector/
├── app.py                      # Interactive Streamlit Web Application
├── train_model.py              # Training, evaluation & serialization pipeline
├── requirements.txt            # Python dependencies (Streamlit, sklearn, etc.)
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for datasets & virtualenvs
├── data/
│   ├── True.csv                # 21,417 real news articles (ISOT benchmark)
│   ├── Fake.csv                # 23,481 fake news articles (ISOT benchmark)
│   └── combined_news.csv       # Preprocessed & shuffled full corpus
├── models/
│   ├── pac_pipeline.joblib     # Pre-trained Passive-Aggressive pipeline (2.3 MB)
│   ├── lr_pipeline.joblib      # Pre-trained Logistic Regression pipeline (2.3 MB)
│   └── metrics.json            # Empirical evaluation metrics & hyper-parameters
├── src/
│   ├── __init__.py             # Python package indicator
│   └── preprocessing.py        # Centralized NLP cleaning & anomaly engine
└── tests/
    ├── __init__.py             # Tests package indicator
    └── test_preprocessing.py   # 22 automated unit test cases
```

---

## ⚙️ Getting Started & Installation

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your system.
- Git (optional, for repository management).

### 1. Clone or Navigate to the Directory
```powershell
cd C:\Users\aditi\.gemini\antigravity-ide\scratch\fake-news-detector
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (Or on Command Prompt):
# .\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖥️ How to Use the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

The browser will automatically open at:
```
http://localhost:8501
```

### Modes of Operation:
1. **Quick Presets:** Select a preloaded sample in the right column and click **"Load Selected Preset"** to see instant classification.
2. **Custom Text Analysis:** Select **"Paste Raw Text / Article"**, input any headline or article body, and click **"Run Authenticity Analysis"**.
3. **Live URL Scraping:** Select **"Fetch via Live URL"**, paste any article URL, and click **"Fetch Article Content"** to scrape and evaluate live news feeds.
4. **Model Selection:** Use the left sidebar dropdown to switch between **Passive-Aggressive Classifier**, **Logistic Regression**, or **Ensemble Voting**.
5. **Inspect Performance:** Switch to the **"Model & Benchmark Metrics"** tab to view the live confusion matrix and classification reports.

---

## 🧪 Running Automated Tests

A comprehensive unit test suite with 22 test cases is included in `tests/test_preprocessing.py`. To run tests:

```powershell
python -m unittest discover -s tests
```

### What the Tests Verify:
- **Text Preprocessing:** HTML tag stripping, URL removal, lowercase normalization, stopword filtering, short token removal, and punctuation handling.
- **Sentiment & Anomaly Detection:** Clickbait phrase matching, capitalization shouting ratio, punctuation anomaly identification, polarity scoring, and subjectivity bounds.
- **Model Integrity:** Verifies that serialized `.joblib` model pipelines load properly from disk, make consistent predictions, and that `metrics.json` conforms to specification.

---

## 🔄 Retraining the Models

The models are already trained and ready to use in `models/`. However, if you wish to retrain from scratch or with a new dataset:

```powershell
python train_model.py
```

The training script will:
1. Verify or stream-download the ISOT benchmark dataset.
2. Run text preprocessing across all articles.
3. Perform a 75/25 stratified train/test split.
4. Train the TF-IDF + Passive-Aggressive and Logistic Regression pipelines.
5. Compute real test metrics and 5-fold cross-validation scores.
6. Save the resulting `.joblib` pipelines and `metrics.json` to the `models/` folder.

---

## 👤 Author & Acknowledgments

- **Author:** Aditi Ashok Lande
- **Institution:** VIT Bhopal University
- **Specialization:** Computer Science & Engineering
- **GitHub:** [@aditilande08](https://github.com/aditilande08)
- **LinkedIn:** [Aditi Lande](http://linkedin.com/in/aditi-lande-9293a02ba)

### Dataset Reference
- *ISOT Fake News Dataset*, Information Security and Object Technology (ISOT) Research Lab, University of Victoria, Canada.
