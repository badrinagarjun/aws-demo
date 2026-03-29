#!/usr/bin/env python3
"""
Train a simple text sentiment classifier.
This script trains a model on sample data and saves it for use in predictions.
"""

import argparse
import logging
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Sample training data - simple sentiment analysis
SAMPLE_DATA = [
    # Positive examples
    ("This is great and wonderful", "positive"),
    ("I love this amazing product", "positive"),
    ("Excellent work very good", "positive"),
    ("Fantastic experience highly recommend", "positive"),
    ("Best thing ever so happy", "positive"),
    ("Really pleased with this", "positive"),
    ("Outstanding quality superb", "positive"),
    ("Wonderful service great", "positive"),
    ("Very satisfied excellent", "positive"),
    ("Impressive results amazing", "positive"),
    ("Perfect exactly what I needed", "positive"),
    ("Brilliant absolutely love it", "positive"),
    ("Awesome product very happy", "positive"),
    ("Delighted with the outcome", "positive"),
    ("Superb quality great value", "positive"),
    
    # Negative examples
    ("This is terrible and awful", "negative"),
    ("I hate this bad product", "negative"),
    ("Poor quality very bad", "negative"),
    ("Horrible experience not recommend", "negative"),
    ("Worst thing ever so disappointed", "negative"),
    ("Really unhappy with this", "negative"),
    ("Terrible service awful", "negative"),
    ("Very dissatisfied poor", "negative"),
    ("Disappointing results bad", "negative"),
    ("Useless waste of time", "negative"),
    ("Awful terrible quality", "negative"),
    ("Horrible not worth it", "negative"),
    ("Disaster complete waste", "negative"),
    ("Unacceptable very poor", "negative"),
    ("Dreadful experience terrible", "negative"),
    
    # Neutral examples
    ("This is okay average product", "neutral"),
    ("It works as expected", "neutral"),
    ("Standard quality nothing special", "neutral"),
    ("Acceptable meets requirements", "neutral"),
    ("Fair product reasonable", "neutral"),
]


def train_model(output_path: str = "sentiment_model.pkl"):
    """Train a sentiment classification model and save it."""
    logger.info("Preparing training data...")
    
    texts = [text for text, _ in SAMPLE_DATA]
    labels = [label for _, label in SAMPLE_DATA]
    
    logger.info("Training model with %d samples...", len(texts))
    
    # Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
    # max_features=1000: Limit vocabulary size to most common 1000 words
    # ngram_range=(1, 2): Use both single words and word pairs
    # alpha=0.1: Smoothing parameter for Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    # Train the model
    model.fit(texts, labels)
    
    # Save the model
    joblib.dump(model, output_path)
    logger.info("Model saved to %s", output_path)
    
    # Quick validation
    test_texts = [
        "This is amazing and wonderful",
        "This is terrible and horrible",
        "This is okay"
    ]
    
    logger.info("\nQuick validation:")
    for text in test_texts:
        prediction = model.predict([text])[0]
        logger.info("  '%s' -> %s", text, prediction)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classification model')
    parser.add_argument('--output', '-o', default='sentiment_model.pkl',
                        help='Output path for the trained model')
    
    args = parser.parse_args()
    
    try:
        train_model(args.output)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
