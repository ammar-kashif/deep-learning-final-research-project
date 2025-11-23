"""
TF-IDF feature extraction module.

Extracts TF-IDF features from preprocessed text with configurable parameters.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple
import logging
import pickle
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFExtractor:
    """TF-IDF feature extractor."""
    
    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF,
    ):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features (20k-50k range)
            ngram_range: Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as proportion)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            sublinear_tf=True,  # Apply sublinear TF scaling
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        logger.info(f"Parameters: max_features={self.max_features}, ngram_range={self.ngram_range}")
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        logger.info(f"TF-IDF vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            TF-IDF feature matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming {len(texts)} texts to TF-IDF features")
        features = self.vectorizer.transform(texts)
        logger.info(f"Feature matrix shape: {features.shape}")
        
        return features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            TF-IDF feature matrix (n_samples, n_features)
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary terms).
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first.")
        
        return list(self.vectorizer.get_feature_names_out())
    
    def save(self, filepath: Path):
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer to {filepath}")
    
    def load(self, filepath: Path) -> 'TFIDFExtractor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to load the vectorizer from
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF vectorizer from {filepath}")
        return self
