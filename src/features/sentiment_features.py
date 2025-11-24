"""
Sentiment feature extraction module.

Extracts VADER sentiment scores (compound, pos, neu, neg) as features for model input.
This implements the paper's "feature-level sentiment analysis" approach.
"""

import numpy as np
from typing import List, Dict
import logging

from src.analysis.sentiment import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFeatureExtractor:
    """Extract VADER sentiment features for model input."""
    
    def __init__(self):
        """Initialize sentiment feature extractor."""
        self.analyzer = SentimentAnalyzer()
        logger.info("SentimentFeatureExtractor initialized")
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract sentiment features from texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Numpy array of shape (n_samples, 4) with columns:
            - compound: Overall sentiment score (-1 to +1)
            - pos: Positive sentiment score (0 to 1)
            - neu: Neutral sentiment score (0 to 1)
            - neg: Negative sentiment score (0 to 1)
        """
        logger.info(f"Extracting sentiment features for {len(texts)} texts")
        
        sentiment_scores = self.analyzer.analyze_batch(texts)
        
        # Extract 4 features per text
        features = np.array([
            [
                scores['compound'],
                scores['pos'],
                scores['neu'],
                scores['neg']
            ]
            for scores in sentiment_scores
        ])
        
        logger.info(f"Extracted sentiment features: shape {features.shape}")
        logger.debug(f"Feature statistics - Compound: mean={features[:, 0].mean():.4f}, "
                    f"Pos: mean={features[:, 1].mean():.4f}, "
                    f"Neu: mean={features[:, 2].mean():.4f}, "
                    f"Neg: mean={features[:, 3].mean():.4f}")
        
        return features
    
    def extract_features_batch(self, texts: List[str], batch_size: int = 1000) -> np.ndarray:
        """
        Extract sentiment features in batches (for large datasets).
        
        Args:
            texts: List of preprocessed text strings
            batch_size: Number of texts to process per batch
            
        Returns:
            Numpy array of shape (n_samples, 4) with sentiment features
        """
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_features = self.extract_features(batch)
            all_features.append(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
        
        return np.vstack(all_features)


def extract_sentiment_features(texts: List[str]) -> np.ndarray:
    """
    Convenience function to extract sentiment features.
    
    Args:
        texts: List of preprocessed text strings
        
    Returns:
        Numpy array of shape (n_samples, 4) with sentiment features
    """
    extractor = SentimentFeatureExtractor()
    return extractor.extract_features(texts)

