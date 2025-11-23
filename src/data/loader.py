"""
Data loading module for sentiment analysis replication.

Loads datasets from local directories:
- Dataset A: datasetA/Dataset A - Chinese.csv
- Dataset B: datasetB/Online Shopping 10 Cats.csv

Also checks for existing translated files in datasetA/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATASET_A_CHINESE,
    DATASET_B_FILE,
    DATASET_A_TRANSLATED,
    DATASET_B_TRANSLATED,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_csv_with_encoding(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Read CSV file with automatic encoding detection.
    Tries multiple encodings in order until one works.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame
    """
    # Common encodings to try in order (most common first)
    # For Chinese/English mixed content, try these encodings
    encodings_to_try = [
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'latin-1',    # ISO-8859-1, handles most Western European characters
        'cp1252',     # Windows-1252 (common on Windows)
        'iso-8859-1', # Latin-1
        'gbk',        # Chinese encoding
        'gb2312',     # Chinese encoding
        'big5',       # Traditional Chinese
    ]
    
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            if enc != 'utf-8':
                logger.debug(f"Successfully read {filepath} with encoding: {enc}")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            logger.debug(f"Failed to read with encoding {enc}: {str(e)[:100]}")
            continue
        except Exception as e:
            # For other errors (like file not found), raise immediately
            raise
    
    # If all encodings fail, try with errors='replace' to replace invalid chars
    logger.warning(f"All standard encodings failed for {filepath}, trying with errors='replace'")
    try:
        return pd.read_csv(filepath, encoding='utf-8', errors='replace', **kwargs)
    except Exception as e:
        logger.error(f"Failed to read CSV file {filepath} even with error replacement: {e}")
        if last_error:
            logger.error(f"Last encoding error: {last_error}")
        raise


def load_dataset_a() -> pd.DataFrame:
    """
    Load Dataset A from local directory.
    
    Returns:
        DataFrame with columns: cat, label, review
    """
    logger.info(f"Loading Dataset A from {DATASET_A_CHINESE}")
    
    if not DATASET_A_CHINESE.exists():
        raise FileNotFoundError(f"Dataset A not found at {DATASET_A_CHINESE}")
    
    df = read_csv_with_encoding(DATASET_A_CHINESE)
    
    # Validate structure
    required_columns = ['cat', 'label', 'review']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset A missing required columns: {missing_columns}")
    
    # Remove rows with missing values
    initial_count = len(df)
    df = df.dropna(subset=['cat', 'label', 'review'])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with missing values from Dataset A")
    
    # Ensure label is binary (0 or 1)
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"Loaded Dataset A: {len(df)} rows, {df['cat'].nunique()} categories")
    logger.info(f"Category distribution:\n{df['cat'].value_counts()}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def load_dataset_b() -> pd.DataFrame:
    """
    Load Dataset B from local directory.
    
    Returns:
        DataFrame with columns: cat, label, review
    """
    logger.info(f"Loading Dataset B from {DATASET_B_FILE}")
    
    if not DATASET_B_FILE.exists():
        raise FileNotFoundError(f"Dataset B not found at {DATASET_B_FILE}")
    
    df = read_csv_with_encoding(DATASET_B_FILE)
    
    # Validate structure
    required_columns = ['cat', 'label', 'review']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset B missing required columns: {missing_columns}")
    
    # Remove rows with missing values
    initial_count = len(df)
    df = df.dropna(subset=['cat', 'label', 'review'])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with missing values from Dataset B")
    
    # Ensure label is binary (0 or 1)
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"Loaded Dataset B: {len(df)} rows, {df['cat'].nunique()} categories")
    logger.info(f"Category distribution:\n{df['cat'].value_counts()}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def check_translated_files() -> Dict[str, bool]:
    """
    Check for existing translated files in datasetA/ directory.
    
    Returns:
        Dictionary indicating which translated files exist
    """
    translated_files = {
        'dataset_a_translated': DATASET_A_TRANSLATED.exists(),
        'dataset_b_translated': DATASET_B_TRANSLATED.exists(),
    }
    
    if translated_files['dataset_a_translated']:
        logger.info(f"Found existing translation for Dataset A: {DATASET_A_TRANSLATED}")
    if translated_files['dataset_b_translated']:
        logger.info(f"Found existing translation for Dataset B: {DATASET_B_TRANSLATED}")
    
    return translated_files


def load_translated_dataset_a() -> Optional[pd.DataFrame]:
    """
    Load pre-translated Dataset A if it exists.
    
    Returns:
        DataFrame with translated reviews, or None if file doesn't exist
    """
    if not DATASET_A_TRANSLATED.exists():
        return None
    
    logger.info(f"Loading pre-translated Dataset A from {DATASET_A_TRANSLATED}")
    df = read_csv_with_encoding(DATASET_A_TRANSLATED)
    
    # Validate structure - should have original columns plus translation
    if 'review' not in df.columns:
        logger.warning("Translated Dataset A missing 'review' column")
        return None
    
    logger.info(f"Loaded {len(df)} translated reviews for Dataset A")
    return df


def load_translated_dataset_b() -> Optional[pd.DataFrame]:
    """
    Load pre-translated Dataset B if it exists.
    
    Returns:
        DataFrame with translated reviews, or None if file doesn't exist
    """
    if not DATASET_B_TRANSLATED.exists():
        return None
    
    logger.info(f"Loading pre-translated Dataset B from {DATASET_B_TRANSLATED}")
    df = read_csv_with_encoding(DATASET_B_TRANSLATED)
    
    # Validate structure
    if 'review' not in df.columns:
        logger.warning("Translated Dataset B missing 'review' column")
        return None
    
    logger.info(f"Loaded {len(df)} translated reviews for Dataset B")
    return df


def get_dataset_info(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Get summary statistics for a dataset.
    
    Args:
        df: DataFrame with dataset
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'name': dataset_name,
        'total_rows': len(df),
        'num_categories': df['cat'].nunique(),
        'categories': df['cat'].unique().tolist(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'category_distribution': df['cat'].value_counts().to_dict(),
        'avg_review_length': df['review'].str.len().mean(),
        'median_review_length': df['review'].str.len().median(),
    }
    
    return info
