"""
Transformer tokenization module for BERT and ELECTRA.

Handles tokenization with max_length, padding, and truncation.
"""

import torch
from transformers import AutoTokenizer, BertTokenizer, ElectraTokenizer
from typing import List, Optional, Dict, Union
import logging
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_PADDING,
    TRANSFORMER_TRUNCATION,
    BERT_MODEL,
    ELECTRA_MODEL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerTokenizer:
    """Tokenizer for transformer models (BERT, ELECTRA)."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = TRANSFORMER_MAX_LENGTH,
        padding: bool = TRANSFORMER_PADDING,
        truncation: bool = TRANSFORMER_TRUNCATION,
    ):
        """
        Initialize transformer tokenizer.
        
        Args:
            model_name: Name of the transformer model
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.
        
        Args:
            texts: Single text string or list of text strings
            return_tensors: Format of returned tensors ('pt', 'tf', 'np', None)
            
        Returns:
            Dictionary with tokenized inputs:
            - input_ids: Token IDs
            - attention_mask: Attention masks
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors or "pt",
        )
        
        return encoded
    
    def tokenize_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a large batch of texts in smaller chunks.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for tokenization
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary with concatenated tokenized inputs
        """
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenize(batch, return_tensors=return_tensors)
            
            if return_tensors == "pt":
                all_input_ids.append(encoded['input_ids'])
                all_attention_masks.append(encoded['attention_mask'])
            else:
                all_input_ids.extend(encoded['input_ids'])
                all_attention_masks.extend(encoded['attention_mask'])
        
        if return_tensors == "pt":
            result = {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
            }
        else:
            result = {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_masks,
            }
        
        return result
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs or tensor
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class BERTTokenizer(TransformerTokenizer):
    """BERT-specific tokenizer."""
    
    def __init__(
        self,
        model_name: str = BERT_MODEL,
        max_length: int = TRANSFORMER_MAX_LENGTH,
        padding: bool = TRANSFORMER_PADDING,
        truncation: bool = TRANSFORMER_TRUNCATION,
    ):
        """
        Initialize BERT tokenizer.
        
        Args:
            model_name: BERT model name (default: bert-base-uncased)
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        """
        super().__init__(model_name, max_length, padding, truncation)


class ELECTRATokenizer(TransformerTokenizer):
    """ELECTRA-specific tokenizer."""
    
    def __init__(
        self,
        model_name: str = ELECTRA_MODEL,
        max_length: int = TRANSFORMER_MAX_LENGTH,
        padding: bool = TRANSFORMER_PADDING,
        truncation: bool = TRANSFORMER_TRUNCATION,
    ):
        """
        Initialize ELECTRA tokenizer.
        
        Args:
            model_name: ELECTRA model name (default: google/electra-base-discriminator)
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        """
        super().__init__(model_name, max_length, padding, truncation)
