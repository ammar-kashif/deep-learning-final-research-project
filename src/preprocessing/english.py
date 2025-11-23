"""
English text preprocessing module.

Uses NLTK for tokenization, stopword removal, and Porter stemming.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from typing import List, Optional
import logging

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data resources."""
    # Download punkt (older NLTK versions)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    
    # Download punkt_tab (newer NLTK versions - required for word_tokenize)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download punkt_tab: {e}")
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download stopwords: {e}")

# Download NLTK data on import
download_nltk_data()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnglishPreprocessor:
    """Preprocessor for English text using NLTK."""
    
    def __init__(self, remove_stopwords: bool = True, stem: bool = True):
        """
        Initialize English preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            stem: Whether to apply Porter stemming
        """
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if stem:
            self.stemmer = PorterStemmer()
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by removing special characters and converting to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"Tokenization error: {e}, using simple split")
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of tokens with stopwords removed
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of stemmed tokens
        """
        if not self.stem:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline: normalize, tokenize, remove stopwords, stem.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string (space-separated tokens)
        """
        # Normalize
        text = self.normalize(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_func(tokens)
        
        # Stem
        tokens = self.stem_tokens(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess(text) for text in texts]


def preprocess_english_text(text: str, remove_stopwords: bool = True, stem: bool = True) -> str:
    """
    Convenience function to preprocess a single English text.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        stem: Whether to apply Porter stemming
        
    Returns:
        Preprocessed text string
    """
    preprocessor = EnglishPreprocessor(remove_stopwords=remove_stopwords, stem=stem)
    return preprocessor.preprocess(text)


def preprocess_english_batch(texts: List[str], remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """
    Convenience function to preprocess a batch of English texts.
    
    Args:
        texts: List of input text strings
        remove_stopwords: Whether to remove stopwords
        stem: Whether to apply Porter stemming
        
    Returns:
        List of preprocessed text strings
    """
    preprocessor = EnglishPreprocessor(remove_stopwords=remove_stopwords, stem=stem)
    return preprocessor.preprocess_batch(texts)
