"""
AI-Powered Fake News & Misinformation Detector
Streamlit Web Application

Loads pre-trained models from disk (trained on 44,000+ ISOT articles)
and displays REAL computed metrics — nothing hardcoded.
"""

import os
import re
import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

# Import shared preprocessing
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessing import (
    preprocess_text, analyze_sentiment_and_anomalies,
    HAS_NLTK, STOP_WORDS, CLICKBAIT_PHRASES
)

# Optional scraper
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER = True
except Exception:
    HAS_SCRAPER = False

# Paths
PROJECT_ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PAC_MODEL_FILE = os.path.join(MODELS_DIR, "pac_pipeline.joblib")
LR_MODEL_FILE = os.path.join(MODELS_DIR, "lr_pipeline.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 28px; font-weight: 800; color: #0f172a; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }
    .sub-title { font-size: 15px; color: #64748b; margin-bottom: 24px; font-weight: 500; }

    .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .badge-real { background: #dcfce7; color: #166534; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 14px; border: 1px solid #86efac; display: inline-block; }
    .badge-fake { background: #fee2e2; color: #991b1b; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 14px; border: 1px solid #fca5a5; display: inline-block; }
    .badge-warning { background: #fef3c7; color: #92400e; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 14px; border: 1px solid #fcd34d; display: inline-block; }

    .anomaly-tag { background: #fff1f2; color: #be123c; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-right: 6px; margin-bottom: 6px; display: inline-block; border: 1px solid #fecdd3; }
    .feature-tag { background: #f0fdf4; color: #15803d; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-right: 6px; margin-bottom: 6px; display: inline-block; border: 1px solid #bbf7d0; }

    .metric-value { font-size: 32px; font-weight: 800; }
    .metric-label { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_trained_models():
    """
    Load pre-trained models from disk.
    Falls back to a minimal inline model ONLY if .joblib files don't exist yet.
    """
    if os.path.exists(PAC_MODEL_FILE) and os.path.exists(LR_MODEL_FILE):
        pac = joblib.load(PAC_MODEL_FILE)
        lr = joblib.load(LR_MODEL_FILE)
        return pac, lr, True
    else:
        st.warning(
            "⚠️ Pre-trained models not found. Run `python train_model.py` first to train on the full ISOT dataset (44,000+ articles). "
            "Currently using a minimal fallback model — results will be unreliable.",
            icon="⚠️"
        )
        # Minimal fallback so the app doesn't crash
        corpus = [
            ("Stock market rallies as Federal Reserve holds interest rates steady.", 1),
            ("Goldman Sachs expands operations in Bengaluru and Hyderabad.", 1),
            ("Clinical research trial publishes peer-reviewed findings on oncology.", 1),
            ("European Central Bank maintains inflation target.", 1),
            ("Semiconductor manufacturer begins production of 2nm chips.", 1),
            ("BREAKING: Secret government plot to replace currency with monopoly money!", 0),
            ("SHOCKING miracle cure eliminates all debt with one simple trick!", 0),
            ("Celebrity arrested for leaking secret 1000% trading algorithm!", 0),
            ("Alien technology controlling Wall Street servers, insider reveals!", 0),
            ("Drink this kitchen spice to melt 50 pounds overnight!", 0),
        ]
        df = pd.DataFrame(corpus, columns=['text', 'label'])
        df['clean_text'] = df['text'].apply(preprocess_text)

        pac = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=5000)),
            ('clf', PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42))
        ])
        pac.fit(df['clean_text'], df['label'])

        lr = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=5000)),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])
        lr.fit(df['clean_text'], df['label'])

        return pac, lr, False


@st.cache_data
def load_metrics():
    """Load computed metrics from training run."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return None


pac_model, lr_model, models_loaded_from_disk = load_trained_models()
saved_metrics = load_metrics()


# ─── URL Scraper ───────────────────────────────────────────────
def fetch_article_from_url(url: str):
    """Fetch and extract article text from a URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')

        if HAS_SCRAPER:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string if soup.title else "Scraped Article"
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            full_text = " ".join(paragraphs).strip()
            return title.strip(), full_text
        else:
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1) if title_match else "Scraped Article"
            cleaned = re.sub(r'<[^>]+>', ' ', html)
            cleaned = " ".join(cleaned.split())
            return title.strip(), cleaned[:4000]
    except Exception as e:
        return None, f"Error fetching article: {str(e)}"


# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Pipeline Configuration")
selected_model_name = st.sidebar.selectbox(
    "Active Classification Algorithm",
    ["Passive-Aggressive Classifier", "Logistic Regression", "Ensemble (Voting / Joint Decision)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 NLP Pipeline Modules")
st.sidebar.markdown("""
- **Tokenizer:** Regex Alphanumeric + HTML/URL Stripper
- **Stopwords:** NLTK English Corpus (179 tokens)
- **Lemmatizer:** NLTK WordNet Lemmatizer
- **Vectorization:** TF-IDF (Unigram + Bigram)
- **Feature Space:** 50,000 max n-gram dimensions
- **Sublinear TF Scaling:** Enabled (logarithmic TF)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Status")

if models_loaded_from_disk and saved_metrics:
    ds = saved_metrics["dataset"]
    pac_m = saved_metrics["models"]["passive_aggressive"]
    st.sidebar.markdown(f"""
- **Dataset:** ISOT Fake News Dataset
- **Total Articles:** {ds['total_articles']:,}
- **Train / Test:** {ds['train_size']:,} / {ds['test_size']:,}
- **PAC Accuracy:** {pac_m['accuracy']}%
- **PAC F1-Score:** {pac_m['f1_score']}%
- **5-Fold CV:** {pac_m['cv_accuracy_mean']}% (±{pac_m['cv_accuracy_std']}%)
- **Trained:** {saved_metrics.get('trained_at', 'N/A')}
""")
    st.sidebar.success("✅ Models loaded from disk")
else:
    st.sidebar.markdown("""
- **Status:** ⚠️ Using fallback (minimal) model
- **Run:** `python train_model.py` to train on full dataset
""")
    st.sidebar.error("⚠️ Fallback mode — unreliable predictions")


# ─── Main Header ──────────────────────────────────────────────
st.markdown("<div class='main-title'>🛡️ AI-Powered Fake News & Misinformation Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>NLP Classification, Verification Scoring & Anomaly Detection — Trained on 44,000+ ISOT Articles</div>", unsafe_allow_html=True)

tab_input, tab_batch, tab_benchmark, tab_about = st.tabs([
    "🔍 Article Verification",
    "🧪 Batch Test Suite",
    "📈 Model & Benchmark Metrics",
    "ℹ️ Architecture & Methodology"
])


# ─── Tab 1: Article Verification ──────────────────────────────
with tab_input:
    col_input, col_presets = st.columns([2, 1])

    with col_presets:
        st.markdown("#### ⚡ Quick Load Presets")
        preset_choice = st.radio(
            "Select a benchmark sample to test:",
            [
                "Real: Federal Reserve Monetary Policy",
                "Real: Goldman Sachs Financial Expansion",
                "Fake: Miracle Debt Elimination Scam",
                "Fake: Secret Currency Plot Conspiracy"
            ],
            index=0
        )

        preset_texts = {
            "Real: Federal Reserve Monetary Policy": "The Federal Reserve announced it will maintain the benchmark interest rate target following stable quarterly consumer price index and job growth reports across primary industrial sectors.",
            "Real: Goldman Sachs Financial Expansion": "Goldman Sachs operations division expands major engineering and risk hubs in Bengaluru and Hyderabad to streamline multi-asset trade settlement and automated exception workflows.",
            "Fake: Miracle Debt Elimination Scam": "SHOCKING miracle trick discovered that eliminates all credit card debt overnight! Banks and financial gurus are panicking and trying to ban this secret loophole!",
            "Fake: Secret Currency Plot Conspiracy": "BREAKING EXPOSED: Secret shadow government plot uncovered to replace all physical money with monopoly tokens overnight! Alien technology controls Wall Street servers!"
        }

        if st.button("📥 Load Selected Preset", use_container_width=True):
            st.session_state["article_input"] = preset_texts[preset_choice]

    with col_input:
        st.markdown("#### 📝 Input live URL or paste text")
        input_mode = st.radio("Input Source:", ["✍️ Paste Raw Text / Article", "🌐 Fetch via Live URL"], horizontal=True)

        if input_mode == "🌐 Fetch via Live URL":
            url_val = st.text_input("Enter Article URL:", placeholder="https://www.reuters.com/business/...")
            if st.button("🌐 Scrape & Load Article", type="secondary"):
                if url_val.strip():
                    with st.spinner("Fetching and extracting article content..."):
                        title, content = fetch_article_from_url(url_val.strip())
                        if title:
                            st.session_state["article_input"] = content[:3000]
                            st.success(f"Loaded: {title}")
                        else:
                            st.error(content)
                else:
                    st.warning("Please enter a valid URL.")

        default_val = st.session_state.get(
            "article_input",
            "Goldman Sachs operations division expands major engineering and risk hubs in Bengaluru and Hyderabad to streamline multi-asset trade settlement and automated exception workflows."
        )

        user_article = st.text_area(
            "Article / Feed Content for Verification:",
            value=default_val,
            height=160
        )

        verify_clicked = st.button("🔍 Run Full Authenticity & Anomaly Audit", type="primary", use_container_width=True)

    # ─── Verification Pipeline ─────────────────────────────────
    if (verify_clicked or "article_input" in st.session_state) and user_article.strip():
        st.markdown("---")
        cleaned = preprocess_text(user_article)
        nlp_stats = analyze_sentiment_and_anomalies(user_article)

        # Predictions across models
        pac_pred = pac_model.predict([cleaned])[0]
        lr_pred = lr_model.predict([cleaned])[0]

        # Raw decision values
        pac_decision = pac_model.decision_function([cleaned])[0]
        lr_proba = lr_model.predict_proba([cleaned])[0][1]

        # Select prediction based on user selection
        if selected_model_name == "Passive-Aggressive Classifier":
            final_pred = pac_pred
            # Use sigmoid calibration on decision function for confidence
            raw_conf = 1 / (1 + np.exp(-abs(pac_decision) * 1.2))
            base_confidence = float(np.clip(raw_conf * 100, 50.0, 99.0))
        elif selected_model_name == "Logistic Regression":
            final_pred = lr_pred
            base_confidence = float(lr_proba * 100 if lr_pred == 1 else (1 - lr_proba) * 100)
        else:
            # Ensemble: majority vote
            final_pred = 1 if (pac_pred + lr_pred >= 2) else 0
            pac_conf = float(np.clip(1 / (1 + np.exp(-abs(pac_decision))), 0.5, 0.99))
            lr_conf = float(max(lr_proba, 1 - lr_proba))
            base_confidence = (pac_conf * 50) + (lr_conf * 50)

        final_confidence = round(base_confidence, 1)

        # Anomalies are SUPPLEMENTARY signals, not overrides
        anomaly_count = len(nlp_stats["anomalies"])
        anomaly_warning = anomaly_count > 0

        # ─── Results Display ───────────────────────────────────
        st.markdown("### 📊 Verification Audit Summary")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Classification Verdict</div>", unsafe_allow_html=True)
            if final_pred == 1 and not anomaly_warning:
                st.markdown("<div class='badge-real' style='margin-top:8px;'>✅ AUTHENTIC & RELIABLE</div>", unsafe_allow_html=True)
            elif final_pred == 1 and anomaly_warning:
                st.markdown("<div class='badge-warning' style='margin-top:8px;'>⚠️ AUTHENTIC (with anomalies)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-fake' style='margin-top:8px;'>⚠️ SUSPICIOUS / MISINFORMATION</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with m_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Model Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value' style='color:{'#166534' if final_confidence > 70 else '#991b1b'};'>{final_confidence:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with m_col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Sentiment Polarity</div>", unsafe_allow_html=True)
            polarity_val = nlp_stats["polarity"]
            pol_label = "Positive" if polarity_val > 0.1 else ("Negative" if polarity_val < -0.1 else "Neutral")
            st.markdown(f"<div class='metric-value'>{polarity_val:+.2f}</div>", unsafe_allow_html=True)
            st.caption(f"Tone: {pol_label} (Subjectivity: {nlp_stats['subjectivity']:.2f})")
            st.markdown("</div>", unsafe_allow_html=True)

        with m_col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Flagged Anomalies</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value' style='color:{'#991b1b' if anomaly_count > 0 else '#166534'};'>{anomaly_count}</div>", unsafe_allow_html=True)
            st.caption("Clickbait & sensationalist markers")
            st.markdown("</div>", unsafe_allow_html=True)

        # Progress bar
        st.markdown("#### 🎯 Model Confidence Distribution")
        st.progress(final_confidence / 100)

        # Details columns
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("#### 🚩 Flagged Linguistic Anomaly Indicators")
            if nlp_stats["anomalies"]:
                for anomaly in nlp_stats["anomalies"]:
                    st.markdown(f"<span class='anomaly-tag'>⚠️ {anomaly}</span>", unsafe_allow_html=True)
                st.info("ℹ️ Anomalies are supplementary signals displayed alongside the ML model prediction. They do NOT override the model's classification.")
            else:
                st.success("✅ No sensationalist clickbait markers or linguistic anomalies detected.")

            st.markdown("#### 🔠 Preprocessing & Token Statistics")
            st.json({
                "Total Words (Raw)": len(user_article.split()),
                "Cleaned Tokens": len(cleaned.split()),
                "Stopwords Removed": len(user_article.split()) - len(cleaned.split()),
                "Lemmatization Active": HAS_NLTK,
                "ALL-CAPS Ratio": f"{nlp_stats['caps_ratio'] * 100:.1f}%"
            })

        with detail_col2:
            st.markdown("#### 🔍 Model Comparison & Dual-Inference")
            model_comp_df = pd.DataFrame({
                "Algorithm": ["Passive-Aggressive", "Logistic Regression", "Ensemble"],
                "Prediction": [
                    "Authentic" if pac_pred == 1 else "Misinformation",
                    "Authentic" if lr_pred == 1 else "Misinformation",
                    "Authentic" if final_pred == 1 else "Misinformation"
                ],
                "Confidence": [
                    f"{np.clip(1 / (1 + np.exp(-abs(pac_decision))) * 100, 50, 99):.1f}%",
                    f"{max(lr_proba, 1 - lr_proba) * 100:.1f}%",
                    f"{final_confidence:.1f}%"
                ]
            })
            st.dataframe(model_comp_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🔑 Key TF-IDF Features")
            tokens_in_cleaned = cleaned.split()
            if tokens_in_cleaned:
                sample_features = list(set(tokens_in_cleaned))[:10]
                for f in sample_features:
                    st.markdown(f"<span class='feature-tag'>🔹 {f}</span>", unsafe_allow_html=True)


# ─── Tab 2: Batch Test Suite ──────────────────────────────────
with tab_batch:
    st.markdown("### 🧪 Batch Benchmark Test Suite")
    st.markdown("Test the classifier across a diversified mix of real vs fabricated articles:")

    test_suite = [
        {"Category": "Financial News", "Text": "Federal Reserve maintains interest rate corridor at 5.25-5.50% following FOMC meeting.", "Expected": "Authentic"},
        {"Category": "Tech Innovation", "Text": "Semiconductor foundry begins pilot runs for 2nm gate-all-around transistor architecture.", "Expected": "Authentic"},
        {"Category": "Corporate", "Text": "Goldman Sachs announces executive leadership promotions across global transaction banking.", "Expected": "Authentic"},
        {"Category": "Health", "Text": "World Health Organization releases annual guidelines for seasonal influenza vaccination.", "Expected": "Authentic"},
        {"Category": "Science", "Text": "NASA successfully deploys climate monitoring satellite into low Earth orbit for weather analysis.", "Expected": "Authentic"},
        {"Category": "Clickbait Scam", "Text": "SHOCKING: Secret trading loophole guarantees 1000% daily gains, claim your cash now!", "Expected": "Misinformation"},
        {"Category": "Medical Hoax", "Text": "Miracle herbal cure eliminates all chronic ailments overnight, big pharma hates this!", "Expected": "Misinformation"},
        {"Category": "Conspiracy", "Text": "EXPOSED: Alien technology uncovered in underground government bunker controlling elections!", "Expected": "Misinformation"},
        {"Category": "Financial Scam", "Text": "Local janitor discovers free mortgage payoff loophole, banks trying to ban this page!", "Expected": "Misinformation"},
        {"Category": "Health Hoax", "Text": "Drink this one kitchen spice before bed to melt 50 pounds of fat overnight while sleeping!", "Expected": "Misinformation"},
    ]

    results = []
    for item in test_suite:
        clean = preprocess_text(item["Text"])
        pred = pac_model.predict([clean])[0]
        pred_label = "Authentic" if pred == 1 else "Misinformation"
        is_correct = "✅ PASS" if pred_label == item["Expected"] else "❌ FAIL"
        results.append({
            "Category": item["Category"],
            "Sample Text": item["Text"][:80] + "...",
            "Expected": item["Expected"],
            "Predicted": pred_label,
            "Result": is_correct
        })

    results_df = pd.DataFrame(results)
    pass_count = sum(1 for r in results if r["Result"] == "✅ PASS")
    st.metric("Test Suite Accuracy", f"{pass_count}/{len(results)} passed")
    st.dataframe(results_df, use_container_width=True, hide_index=True)


# ─── Tab 3: Real Benchmark Metrics ───────────────────────────
with tab_benchmark:
    st.markdown("### 📈 Model Performance Metrics")

    if saved_metrics:
        ds = saved_metrics["dataset"]
        st.markdown(f"""
        Models were trained and evaluated on the **ISOT Fake News Dataset** comprising
        **{ds['total_articles']:,} articles** ({ds['real_count']:,} real + {ds['fake_count']:,} fake).
        Train/test split: **{ds['train_size']:,} / {ds['test_size']:,}** (75/25, stratified).
        """)

        # Metrics for both models
        for model_key, model_label in [("passive_aggressive", "Passive-Aggressive Classifier"),
                                        ("logistic_regression", "Logistic Regression")]:
            m = saved_metrics["models"][model_key]
            st.markdown(f"#### {model_label}")

            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
            b_col1.metric("Accuracy", f"{m['accuracy']}%")
            b_col2.metric("Precision", f"{m['precision']}%")
            b_col3.metric("Recall", f"{m['recall']}%")
            b_col4.metric("F1-Score", f"{m['f1_score']}%")
            b_col5.metric("5-Fold CV", f"{m['cv_accuracy_mean']}% (±{m['cv_accuracy_std']}%)")

            # Confusion matrix
            cm = m["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                columns=["Predicted Fake", "Predicted Real"],
                index=["Actual Fake", "Actual Real"]
            )
            st.table(cm_df)
            st.caption(f"TN={cm[0][0]:,} | FP={cm[0][1]:,} | FN={cm[1][0]:,} | TP={cm[1][1]:,}")
            st.markdown("---")

        st.caption(f"Training completed: {saved_metrics.get('trained_at', 'N/A')}")
    else:
        st.warning("⚠️ No metrics file found. Run `python train_model.py` first to generate real evaluation metrics.")
        st.code("python train_model.py", language="bash")


# ─── Tab 4: Architecture ─────────────────────────────────────
with tab_about:
    st.markdown("### 🏛️ System Architecture & Methodology")
    st.markdown("""
    #### 1. Data Pipeline
    - **Dataset:** ISOT Fake News Dataset (44,000+ articles from Reuters & various fake news sources)
    - **Split:** 75% train / 25% test (stratified by label)
    - **Validation:** 5-fold cross-validation for robust accuracy estimates

    #### 2. Text Preprocessing
    - **Regex Sanitization:** Strips HTML, URLs, non-alphabetic characters
    - **Stopword Filtering:** NLTK English corpus (179 stop words)
    - **Lemmatization:** WordNet lemmatizer to normalize word forms
    - **Shared Module:** Same preprocessing used in training and inference (`src/preprocessing.py`)

    #### 3. Vectorization
    - **TF-IDF:** Unigram + Bigram with sublinear TF scaling
    - **Feature Space:** 50,000 max n-gram dimensions
    - **Document Frequency Filtering:** min_df=2, max_df=0.95

    #### 4. Classification Models
    - **Passive-Aggressive Classifier:** Online learning, margin-based updates, fast inference
    - **Logistic Regression:** Calibrated probabilities, interpretable confidence scores
    - **Ensemble:** Majority vote across both models

    #### 5. Supplementary Analysis
    - **Anomaly Detection:** Clickbait phrase detection, ALL-CAPS ratio, punctuation analysis
    - **Sentiment Analysis:** Lexicon-based polarity and subjectivity scoring
    - **Note:** Anomalies are displayed as supplementary signals — they do NOT override the ML model's prediction

    #### 6. Model Persistence
    - Trained models saved as `.joblib` files for fast loading
    - Metrics saved as `metrics.json` with full evaluation results
    - No retraining needed at app startup
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 13px;'>Built by <b>Aditi Ashok Lande</b> | AI Fake News & Misinformation Detector | Python, Scikit-Learn, Streamlit, NLP</div>", unsafe_allow_html=True)
