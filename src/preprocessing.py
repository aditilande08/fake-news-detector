"""
Text Preprocessing Module
AI-Powered Fake News Detector

Centralized NLP preprocessing pipeline used by both training and inference.
Ensures consistent text cleaning across the entire project.
"""

import re

# Optional NLTK with safe fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data quietly if not available
    for resource, name in [('corpora/stopwords', 'stopwords'),
                           ('corpora/wordnet', 'wordnet'),
                           ('corpora/omw-1.4', 'omw-1.4')]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)

    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    STOP_WORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
        'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
        'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
        'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
        'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
        'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
        'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself'
    }
    LEMMATIZER = None


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for TF-IDF vectorization.

    Pipeline:
    1. Strip HTML tags
    2. Strip URLs
    3. Remove non-alphabetic characters
    4. Lowercase and tokenize
    5. Remove stopwords (tokens <= 2 chars)
    6. Lemmatize (if NLTK available)

    Args:
        text: Raw article text

    Returns:
        Cleaned, space-joined token string
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Strip non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase & Tokenize
    tokens = text.lower().split()
    # Filter stopwords & lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            if LEMMATIZER:
                token = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


# Lexicon-based Sentiment & Anomaly Analysis
POSITIVE_WORDS = {
    'benefit', 'growth', 'progress', 'stable', 'positive', 'gain', 'success', 'reliable', 'certified',
    'verified', 'confirmed', 'record', 'improve', 'strengthen', 'advance', 'boost', 'expand', 'prosper',
    'innovative', 'robust', 'sustainable', 'effective', 'efficient', 'breakthrough', 'constructive',
    'achievement', 'optimistic', 'thriving', 'resilient', 'favorable', 'promising', 'leading'
}

NEGATIVE_WORDS = {
    'decline', 'drop', 'crisis', 'crash', 'loss', 'danger', 'fraud', 'scam', 'corrupt', 'arrest',
    'illegal', 'risk', 'fail', 'threat', 'panic', 'plunge', 'collapse', 'conspiracy', 'scandal',
    'severe', 'fatal', 'destructive', 'damaging', 'hostile', 'disaster', 'violation',
    'devastating', 'alarming', 'terrifying', 'catastrophic', 'outrage', 'shocking', 'horrific'
}

CLICKBAIT_PHRASES = [
    'shocking', 'miracle cure', 'miracle', 'secret trick', 'secret plot', 'guaranteed profit',
    'guaranteed', 'banned by government', 'doctors hate', 'exposed', 'conspiracy', 'alien technology',
    'overnight million', '1000%', 'hidden truth', "they don't want you to know", 'unbelievable trick',
    'leaked video', 'urgent warning', 'act now', 'monopoly money', 'instant cure', 'free money',
    'you won\'t believe', 'this one trick', 'big pharma', 'what they don\'t tell you',
    'click here', 'share before deleted', 'exposed the truth'
]


def analyze_sentiment_and_anomalies(raw_text: str) -> dict:
    """
    Perform lexicon-based sentiment analysis and detect clickbait anomalies.

    Returns dict with polarity, subjectivity, anomaly list, caps ratio, token count.
    """
    tokens = re.findall(r'\b[a-zA-Z]+\b', raw_text.lower())
    total_tokens = max(1, len(tokens))

    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)

    # Polarity Score (-1.0 to +1.0)
    if pos_count == 0 and neg_count == 0:
        polarity = 0.0
    else:
        polarity = (pos_count - neg_count) / max(1, (pos_count + neg_count))

    # Subjectivity Score (0.0 = Objective, 1.0 = Highly Subjective)
    subjectivity = min(1.0, (pos_count + neg_count) / total_tokens * 8)

    # Anomaly Detection
    detected_anomalies = []
    text_lower = raw_text.lower()
    for phrase in CLICKBAIT_PHRASES:
        if phrase in text_lower:
            detected_anomalies.append(phrase.title())

    # Uppercase anomaly (shouting detection)
    raw_words = raw_text.split()
    caps_words = [w for w in raw_words if w.isupper() and len(w) > 2 and w.isalpha()]
    caps_ratio = len(caps_words) / max(1, len(raw_words))

    # Punctuation anomalies
    exclamations = raw_text.count('!')
    questions = raw_text.count('?')

    if caps_ratio > 0.15:
        detected_anomalies.append(f"Excessive Capitalization ({int(caps_ratio * 100)}% ALL-CAPS words)")
    if exclamations >= 3:
        detected_anomalies.append(f"Sensational Punctuation ({exclamations} Exclamations '!')")
    if questions >= 3:
        detected_anomalies.append(f"Loaded Question Formatting ({questions} Questions '?')")

    return {
        "polarity": polarity,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "subjectivity": subjectivity,
        "anomalies": detected_anomalies,
        "caps_ratio": caps_ratio,
        "token_count": total_tokens
    }
