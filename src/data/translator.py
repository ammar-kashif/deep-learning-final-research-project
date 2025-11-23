"""
Chinese-to-English translation module using MarianMT.

Uses Helsinki-NLP/opus-mt-zh-en model for batch translation.
Implements chunking (16-32 sentences per batch) and error handling.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Optional
import logging
import time
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TRANSLATION_MODEL,
    TRANSLATION_BATCH_SIZE,
    DATASET_A_TRANSLATED,
    DATASET_B_TRANSLATED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTranslator:
    """Translator for Chinese text to English using MarianMT."""
    
    def __init__(self, model_name: str = TRANSLATION_MODEL, device: Optional[str] = None):
        """
        Initialize the translator.
        
        Args:
            model_name: Name of the MarianMT model
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        logger.info(f"Loading translation model: {model_name}")
        
        # Auto-detect device
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        logger.info("Translation model loaded successfully")
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of Chinese texts to English.
        
        Args:
            texts: List of Chinese text strings
            
        Returns:
            List of translated English text strings
        """
        if not texts:
            return []
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Translate
        with torch.no_grad():
            translated = self.model.generate(**encoded)
        
        # Decode
        translated_texts = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translated_texts
    
    def translate(self, texts: List[str], batch_size: int = TRANSLATION_BATCH_SIZE) -> List[str]:
        """
        Translate a list of Chinese texts to English in batches.
        
        Args:
            texts: List of Chinese text strings
            batch_size: Number of texts to translate per batch
            
        Returns:
            List of translated English text strings
        """
        if not texts:
            return []
        
        logger.info(f"Translating {len(texts)} texts in batches of {batch_size}")
        
        translated_texts = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                batch_translated = self.translate_batch(batch)
                translated_texts.extend(batch_translated)
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    elapsed = time.time() - start_time
                    progress = (batch_num / total_batches) * 100
                    logger.info(
                        f"Translated batch {batch_num}/{total_batches} "
                        f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s"
                    )
            
            except Exception as e:
                logger.error(f"Error translating batch {batch_num}: {e}")
                # Fill with placeholder on error
                translated_texts.extend([f"[Translation Error: {str(e)}]"] * len(batch))
        
        elapsed = time.time() - start_time
        logger.info(f"Translation completed in {elapsed:.2f} seconds")
        
        return translated_texts
    
    def translate_dataframe(
        self,
        df: pd.DataFrame,
        source_column: str = "review",
        target_column: str = "review_en",
        batch_size: int = TRANSLATION_BATCH_SIZE,
    ) -> pd.DataFrame:
        """
        Translate reviews in a DataFrame.
        
        Args:
            df: DataFrame with Chinese reviews
            source_column: Column name containing Chinese text
            target_column: Column name to store translated English text
            batch_size: Batch size for translation
            
        Returns:
            DataFrame with added translation column
        """
        logger.info(f"Translating {len(df)} reviews from column '{source_column}'")
        
        # Extract texts
        texts = df[source_column].astype(str).tolist()
        
        # Translate
        translated_texts = self.translate(texts, batch_size=batch_size)
        
        # Add to dataframe
        df = df.copy()
        df[target_column] = translated_texts
        
        return df


def translate_dataset_a(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    use_existing: bool = False,
) -> pd.DataFrame:
    """
    Translate Dataset A reviews from Chinese to English.
    
    Args:
        df: DataFrame with Dataset A
        save_path: Path to save translated dataset (default: DATASET_A_TRANSLATED)
        use_existing: If True and file exists, load existing translation
        
    Returns:
        DataFrame with translated reviews
    """
    if save_path is None:
        save_path = DATASET_A_TRANSLATED
    
    # Check if translation already exists
    if use_existing and save_path.exists():
        logger.info(f"Loading existing translation from {save_path}")
        return pd.read_csv(save_path, encoding='utf-8')
    
    # Translate
    translator = ChineseTranslator()
    df_translated = translator.translate_dataframe(df, source_column="review", target_column="review_en")
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_translated.to_csv(save_path, index=False, encoding='utf-8')
    logger.info(f"Saved translated Dataset A to {save_path}")
    
    return df_translated


def translate_dataset_b(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    use_existing: bool = False,
) -> pd.DataFrame:
    """
    Translate Dataset B reviews from Chinese to English.
    
    Args:
        df: DataFrame with Dataset B
        save_path: Path to save translated dataset (default: DATASET_B_TRANSLATED)
        use_existing: If True and file exists, load existing translation
        
    Returns:
        DataFrame with translated reviews
    """
    if save_path is None:
        save_path = DATASET_B_TRANSLATED
    
    # Check if translation already exists
    if use_existing and save_path.exists():
        logger.info(f"Loading existing translation from {save_path}")
        return pd.read_csv(save_path, encoding='utf-8')
    
    # Translate
    translator = ChineseTranslator()
    df_translated = translator.translate_dataframe(df, source_column="review", target_column="review_en")
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_translated.to_csv(save_path, index=False, encoding='utf-8')
    logger.info(f"Saved translated Dataset B to {save_path}")
    
    return df_translated
