# 🛡️ AI-Powered Fake News & Data Authenticity Detector

An end-to-end Machine Learning & NLP pipeline that analyzes news articles, press releases, and textual data feeds to evaluate authenticity, assign credibility scores, and flag potential misinformation in real time.

## 🚀 Key Features
- **NLP Text Preprocessing:** Cleans and tokenizes text, removes stopwords, and performs linguistic feature extraction using NLTK and regular expressions.
- **Machine Learning Classification:** Utilizes TF-IDF (Unigram + Bigram) vectorization paired with a Passive-Aggressive Classifier to achieve high-precision credibility predictions (94.8% accuracy).
- **Interactive Streamlit Dashboard:** User-friendly web interface to paste articles, inspect real-time credibility confidence scores (0–100%), and view linguistic anomaly breakdowns.
- **Explainable Anomaly Alerts:** Flags sensationalist keywords, clickbait indicators, and unverified source claims for transparent audit trails.

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Machine Learning & NLP:** Scikit-Learn, NLTK, TF-IDF
- **Data Manipulation:** Pandas, NumPy
- **Web App / UI:** Streamlit

## 📦 Installation & Setup

1. **Clone the repository:**
   `ash
   git clone https://github.com/aditilande08/fake-news-detector.git
   cd fake-news-detector
   `

2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

3. **Run the Streamlit app:**
   `ash
   streamlit run app.py
   `

## 📊 Evaluation & Metrics
- **Accuracy:** 94.8% on benchmark news datasets
- **Inference Time:** < 50ms per article feed
- **Decision Boundary:** Calibrated statistical confidence scoring with feature importance ranking.

## 👤 Author
- **Aditi Ashok Lande** (VIT Bhopal University)
- [LinkedIn](http://linkedin.com/in/aditi-lande-9293a02ba) | [GitHub](https://github.com/aditilande08)
