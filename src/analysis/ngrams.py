"""
N-gram extraction and analysis module.

Extracts top N bigrams and trigrams from preprocessed Chinese tokens.
"""

from collections import Counter
from typing import List, Dict, Tuple
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import TOP_N_BIGRAMS, TOP_N_TRIGRAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NGramExtractor:
    """Extractor for N-grams from tokenized text."""
    
    def __init__(self):
        """Initialize N-gram extractor."""
        pass
    
    def extract_bigrams(self, tokens: List[List[str]]) -> List[Tuple[str, str]]:
        """
        Extract bigrams from tokenized texts.
        
        Args:
            tokens: List of token lists (each inner list is a preprocessed text)
            
        Returns:
            List of bigram tuples
        """
        bigrams = []
        for token_list in tokens:
            if len(token_list) < 2:
                continue
            for i in range(len(token_list) - 1):
                bigram = (token_list[i], token_list[i + 1])
                bigrams.append(bigram)
        return bigrams
    
    def extract_trigrams(self, tokens: List[List[str]]) -> List[Tuple[str, str, str]]:
        """
        Extract trigrams from tokenized texts.
        
        Args:
            tokens: List of token lists (each inner list is a preprocessed text)
            
        Returns:
            List of trigram tuples
        """
        trigrams = []
        for token_list in tokens:
            if len(token_list) < 3:
                continue
            for i in range(len(token_list) - 2):
                trigram = (token_list[i], token_list[i + 1], token_list[i + 2])
                trigrams.append(trigram)
        return trigrams
    
    def get_top_bigrams(
        self,
        tokens: List[List[str]],
        top_n: int = TOP_N_BIGRAMS,
    ) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get top N most frequent bigrams.
        
        Args:
            tokens: List of token lists
            top_n: Number of top bigrams to return
            
        Returns:
            List of (bigram, count) tuples, sorted by frequency
        """
        bigrams = self.extract_bigrams(tokens)
        bigram_counts = Counter(bigrams)
        top_bigrams = bigram_counts.most_common(top_n)
        return top_bigrams
    
    def get_top_trigrams(
        self,
        tokens: List[List[str]],
        top_n: int = TOP_N_TRIGRAMS,
    ) -> List[Tuple[Tuple[str, str, str], int]]:
        """
        Get top N most frequent trigrams.
        
        Args:
            tokens: List of token lists
            top_n: Number of top trigrams to return
            
        Returns:
            List of (trigram, count) tuples, sorted by frequency
        """
        trigrams = self.extract_trigrams(tokens)
        trigram_counts = Counter(trigrams)
        top_trigrams = trigram_counts.most_common(top_n)
        return top_trigrams
    
    def analyze_ngrams(
        self,
        tokens: List[List[str]],
        top_n_bigrams: int = TOP_N_BIGRAMS,
        top_n_trigrams: int = TOP_N_TRIGRAMS,
    ) -> Dict:
        """
        Analyze N-grams and return top bigrams and trigrams.
        
        Args:
            tokens: List of token lists
            top_n_bigrams: Number of top bigrams to extract
            top_n_trigrams: Number of top trigrams to extract
            
        Returns:
            Dictionary with:
            - top_bigrams: List of (bigram, count) tuples
            - top_trigrams: List of (trigram, count) tuples
            - total_bigrams: Total number of unique bigrams
            - total_trigrams: Total number of unique trigrams
        """
        logger.info(f"Extracting top {top_n_bigrams} bigrams and top {top_n_trigrams} trigrams")
        
        top_bigrams = self.get_top_bigrams(tokens, top_n_bigrams)
        top_trigrams = self.get_top_trigrams(tokens, top_n_trigrams)
        
        # Count unique N-grams
        all_bigrams = self.extract_bigrams(tokens)
        all_trigrams = self.extract_trigrams(tokens)
        unique_bigrams = len(set(all_bigrams))
        unique_trigrams = len(set(all_trigrams))
        
        result = {
            'top_bigrams': top_bigrams,
            'top_trigrams': top_trigrams,
            'total_unique_bigrams': unique_bigrams,
            'total_unique_trigrams': unique_trigrams,
            'total_bigram_count': len(all_bigrams),
            'total_trigram_count': len(all_trigrams),
        }
        
        logger.info(f"Found {unique_bigrams} unique bigrams and {unique_trigrams} unique trigrams")
        
        return result
