#!/usr/bin/env python3
"""
Make predictions using the trained sentiment model.
Can be used standalone or imported by other scripts.
"""

import argparse
import logging
import sys
import os
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_model(model_path: str = "sentiment_model.pkl"):
    """Load the trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def predict_sentiment(text: str, model_path: str = "sentiment_model.pkl") -> dict:
    """
    Predict sentiment for a given text.
    Returns a dict with text, sentiment, and confidence.
    """
    model = load_model(model_path)
    
    # Make prediction
    prediction = model.predict([text])[0]
    
    # Get prediction probabilities
    try:
        proba = model.predict_proba([text])[0]
        confidence = max(proba)
    except AttributeError:
        confidence = 1.0  # If model doesn't support predict_proba
    
    return {
        'text': text,
        'sentiment': prediction,
        'confidence': float(confidence)
    }


def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for text')
    parser.add_argument('--text', '-t', required=True, help='Text to classify')
    parser.add_argument('--model', '-m', default='sentiment_model.pkl',
                        help='Path to the trained model')
    
    args = parser.parse_args()
    
    try:
        result = predict_sentiment(args.text, args.model)
        logger.info("Prediction: %s (confidence: %.2f)", 
                   result['sentiment'], result['confidence'])
        print(result)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
