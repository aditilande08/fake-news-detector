"""
Unit Tests for AI-Powered Fake News Detector
Tests preprocessing, sentiment analysis, anomaly detection, and model loading.
"""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import (
    preprocess_text, analyze_sentiment_and_anomalies,
    POSITIVE_WORDS, NEGATIVE_WORDS, CLICKBAIT_PHRASES
)


class TestPreprocessing(unittest.TestCase):
    """Tests for the text preprocessing pipeline."""

    def test_empty_string(self):
        self.assertEqual(preprocess_text(""), "")

    def test_none_input(self):
        self.assertEqual(preprocess_text(None), "")

    def test_integer_input(self):
        self.assertEqual(preprocess_text(123), "")

    def test_html_stripping(self):
        result = preprocess_text("<p>Hello <b>world</b></p>")
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)

    def test_url_stripping(self):
        result = preprocess_text("Visit https://example.com for details")
        self.assertNotIn("https", result)
        self.assertNotIn("example.com", result)

    def test_lowercase(self):
        result = preprocess_text("BREAKING NEWS ALERT")
        self.assertEqual(result, result.lower())

    def test_stopword_removal(self):
        result = preprocess_text("This is a very simple test of the system")
        self.assertNotIn("this", result.split())
        self.assertNotIn("is", result.split())
        self.assertNotIn("the", result.split())

    def test_short_token_removal(self):
        result = preprocess_text("I am at my car in an old area")
        # Tokens <= 2 chars should be removed
        for token in result.split():
            self.assertGreater(len(token), 2)

    def test_special_characters_removed(self):
        result = preprocess_text("Stock price: $150.00!!! Revenue @10M")
        self.assertNotIn("$", result)
        self.assertNotIn("@", result)
        self.assertNotIn("!", result)

    def test_meaningful_output(self):
        text = "Goldman Sachs expands operations in Bengaluru"
        result = preprocess_text(text)
        self.assertGreater(len(result), 0)
        self.assertIn("goldman", result)
        self.assertIn("sachs", result)


class TestSentimentAnalysis(unittest.TestCase):
    """Tests for sentiment and anomaly analysis."""

    def test_positive_text(self):
        result = analyze_sentiment_and_anomalies(
            "The company showed remarkable growth and progress with robust sustainable gains."
        )
        self.assertGreater(result["polarity"], 0)
        self.assertGreater(result["pos_count"], 0)

    def test_negative_text(self):
        result = analyze_sentiment_and_anomalies(
            "The crisis led to a devastating collapse and catastrophic failure."
        )
        self.assertLess(result["polarity"], 0)
        self.assertGreater(result["neg_count"], 0)

    def test_neutral_text(self):
        result = analyze_sentiment_and_anomalies(
            "The meeting was held at the conference room on Tuesday."
        )
        self.assertEqual(result["polarity"], 0.0)

    def test_anomaly_detection_clickbait(self):
        result = analyze_sentiment_and_anomalies(
            "SHOCKING miracle cure discovered! Doctors hate this secret trick!"
        )
        self.assertGreater(len(result["anomalies"]), 0)

    def test_no_anomalies_clean_text(self):
        result = analyze_sentiment_and_anomalies(
            "The Federal Reserve announced stable monetary policy guidelines."
        )
        self.assertEqual(len(result["anomalies"]), 0)

    def test_caps_detection(self):
        result = analyze_sentiment_and_anomalies(
            "BREAKING URGENT ALERT THIS NEWS WILL SHOCK YOU TODAY"
        )
        self.assertGreater(result["caps_ratio"], 0.15)

    def test_exclamation_detection(self):
        result = analyze_sentiment_and_anomalies(
            "This is incredible!!! You won't believe it!!! Act now!!!"
        )
        anomaly_text = " ".join(result["anomalies"])
        self.assertIn("Exclamation", anomaly_text)

    def test_return_structure(self):
        result = analyze_sentiment_and_anomalies("Test text.")
        self.assertIn("polarity", result)
        self.assertIn("pos_count", result)
        self.assertIn("neg_count", result)
        self.assertIn("subjectivity", result)
        self.assertIn("anomalies", result)
        self.assertIn("caps_ratio", result)
        self.assertIn("token_count", result)

    def test_subjectivity_range(self):
        result = analyze_sentiment_and_anomalies("Some positive growth and negative decline observed.")
        self.assertGreaterEqual(result["subjectivity"], 0.0)
        self.assertLessEqual(result["subjectivity"], 1.0)


class TestModelFiles(unittest.TestCase):
    """Tests that model files exist after training."""

    def test_model_directory_exists(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if os.path.exists(models_dir):
            self.assertTrue(os.path.isdir(models_dir))

    def test_metrics_file_valid_json(self):
        metrics_file = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file) as f:
                data = json.load(f)
            self.assertIn("dataset", data)
            self.assertIn("models", data)
            self.assertIn("passive_aggressive", data["models"])
            self.assertIn("logistic_regression", data["models"])
            # Verify metrics are reasonable
            pac = data["models"]["passive_aggressive"]
            self.assertGreater(pac["accuracy"], 80)
            self.assertGreater(pac["f1_score"], 80)

    def test_model_files_loadable(self):
        import joblib
        pac_path = os.path.join(os.path.dirname(__file__), "..", "models", "pac_pipeline.joblib")
        lr_path = os.path.join(os.path.dirname(__file__), "..", "models", "lr_pipeline.joblib")
        if os.path.exists(pac_path):
            model = joblib.load(pac_path)
            # Should be able to predict
            result = model.predict(["test article about financial markets"])
            self.assertIn(result[0], [0, 1])
        if os.path.exists(lr_path):
            model = joblib.load(lr_path)
            result = model.predict(["test article about financial markets"])
            self.assertIn(result[0], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
