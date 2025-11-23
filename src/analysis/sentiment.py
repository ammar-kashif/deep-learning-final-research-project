"""
Sentiment analysis module using NLTK VADER.

Analyzes sentiment of English text and reports compound scores (-1 to +1).
"""

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, List, Optional

# Download VADER lexicon if needed
try:
    import nltk
    nltk.data.find('vader_lexicon')
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer using NLTK VADER."""
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER SentimentIntensityAnalyzer initialized")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment scores:
            - compound: Overall sentiment score (-1 to +1)
            - pos: Positive sentiment score (0 to 1)
            - neu: Neutral sentiment score (0 to 1)
            - neg: Negative sentiment score (0 to 1)
        """
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of sentiment score dictionaries
        """
        return [self.analyze(text) for text in texts]
    
    def get_compound_score(self, text: str) -> float:
        """
        Get compound sentiment score for a text.
        
        Args:
            text: Input text string
            
        Returns:
            Compound sentiment score (-1 to +1)
        """
        scores = self.analyze(text)
        return scores['compound']
    
    def get_compound_scores(self, texts: List[str]) -> List[float]:
        """
        Get compound sentiment scores for a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of compound sentiment scores
        """
        return [self.get_compound_score(text) for text in texts]


def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    text_column: str = "review_en",
    analyzer: Optional[SentimentAnalyzer] = None,
) -> pd.DataFrame:
    """
    Analyze sentiment for all texts in a DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of column containing text to analyze
        analyzer: Optional pre-initialized analyzer
        
    Returns:
        DataFrame with added sentiment columns:
        - sentiment_compound: Overall sentiment score
        - sentiment_pos: Positive score
        - sentiment_neu: Neutral score
        - sentiment_neg: Negative score
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()
    
    logger.info(f"Analyzing sentiment for {len(df)} texts from column '{text_column}'")
    
    texts = df[text_column].astype(str).tolist()
    sentiment_scores = analyzer.analyze_batch(texts)
    
    # Add sentiment columns to dataframe
    df_result = df.copy()
    df_result['sentiment_compound'] = [s['compound'] for s in sentiment_scores]
    df_result['sentiment_pos'] = [s['pos'] for s in sentiment_scores]
    df_result['sentiment_neu'] = [s['neu'] for s in sentiment_scores]
    df_result['sentiment_neg'] = [s['neg'] for s in sentiment_scores]
    
    logger.info("Sentiment analysis completed")
    
    return df_result


def get_sentiment_statistics(df: pd.DataFrame, sentiment_column: str = "sentiment_compound") -> Dict:
    """
    Get descriptive statistics for sentiment scores.
    
    Args:
        df: DataFrame with sentiment scores
        sentiment_column: Name of sentiment score column
        
    Returns:
        Dictionary with sentiment statistics
    """
    stats = {
        'mean': df[sentiment_column].mean(),
        'median': df[sentiment_column].median(),
        'std': df[sentiment_column].std(),
        'min': df[sentiment_column].min(),
        'max': df[sentiment_column].max(),
        'positive_count': (df[sentiment_column] > 0).sum(),
        'negative_count': (df[sentiment_column] < 0).sum(),
        'neutral_count': (df[sentiment_column] == 0).sum(),
        'positive_percentage': (df[sentiment_column] > 0).mean() * 100,
        'negative_percentage': (df[sentiment_column] < 0).mean() * 100,
    }
    
    return stats
