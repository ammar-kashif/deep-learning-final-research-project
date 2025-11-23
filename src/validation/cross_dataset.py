"""
Cross-dataset validation module.

Measures translation fidelity via cosine similarity and ELECTRA-BERT agreement.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    SENTENCE_TRANSFORMER_MODEL,
    CROSS_DATASET_SAMPLE_SIZE,
    METRICS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossDatasetValidator:
    """Validator for cross-dataset analysis."""
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initialize cross-dataset validator.
        
        Args:
            model_name: Name of sentence transformer model for embeddings
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("Sentence transformer model loaded")
    
    def compute_translation_fidelity(
        self,
        chinese_texts: List[str],
        english_texts: List[str],
    ) -> Dict:
        """
        Compute translation fidelity using cosine similarity of embeddings.
        
        Args:
            chinese_texts: List of original Chinese texts
            english_texts: List of translated English texts
            
        Returns:
            Dictionary with fidelity metrics
        """
        logger.info(f"Computing translation fidelity for {len(chinese_texts)} texts")
        
        # Get embeddings
        chinese_embeddings = self.embedding_model.encode(chinese_texts, show_progress_bar=True)
        english_embeddings = self.embedding_model.encode(english_texts, show_progress_bar=True)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(chinese_embeddings, english_embeddings)
        
        # Get diagonal (each text with its translation)
        diagonal_similarities = np.diag(similarities)
        
        # Compute statistics
        fidelity_stats = {
            'mean_similarity': float(np.mean(diagonal_similarities)),
            'median_similarity': float(np.median(diagonal_similarities)),
            'std_similarity': float(np.std(diagonal_similarities)),
            'min_similarity': float(np.min(diagonal_similarities)),
            'max_similarity': float(np.max(diagonal_similarities)),
            'similarities': diagonal_similarities.tolist(),
        }
        
        logger.info(f"Translation fidelity - Mean: {fidelity_stats['mean_similarity']:.4f}, "
                   f"Median: {fidelity_stats['median_similarity']:.4f}")
        
        return fidelity_stats
    
    def compute_model_agreement(
        self,
        predictions_model1: np.ndarray,
        predictions_model2: np.ndarray,
    ) -> Dict:
        """
        Compute agreement between two models.
        
        Args:
            predictions_model1: Predictions from first model
            predictions_model2: Predictions from second model
            
        Returns:
            Dictionary with agreement metrics
        """
        logger.info("Computing model agreement")
        
        # Agreement rate
        agreement_mask = predictions_model1 == predictions_model2
        agreement_rate = float(np.mean(agreement_mask))
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(predictions_model1, predictions_model2)
        
        # Per-class agreement
        unique_labels = sorted(np.unique(np.concatenate([predictions_model1, predictions_model2])))
        per_class_agreement = {}
        
        for label in unique_labels:
            mask = (predictions_model1 == label) | (predictions_model2 == label)
            if mask.sum() > 0:
                label_agreement = (predictions_model1[mask] == predictions_model2[mask]).mean()
                per_class_agreement[f"class_{label}"] = float(label_agreement)
        
        agreement_stats = {
            'agreement_rate': agreement_rate,
            'cohens_kappa': float(kappa),
            'total_samples': len(predictions_model1),
            'agreed_samples': int(agreement_mask.sum()),
            'disagreed_samples': int((~agreement_mask).sum()),
            'per_class_agreement': per_class_agreement,
        }
        
        logger.info(f"Model agreement - Rate: {agreement_rate:.4f}, Kappa: {kappa:.4f}")
        
        return agreement_stats
    
    def cross_dataset_evaluation(
        self,
        model_trainer,
        X_test: List[str],
        y_test: np.ndarray,
        dataset_name: str = "cross_dataset",
    ) -> Dict:
        """
        Evaluate model on cross-dataset test set.
        
        Args:
            model_trainer: Trained model trainer object
            X_test: Test texts
            y_test: Test labels
            dataset_name: Name of the test dataset
            
        Returns:
            Dictionary with cross-dataset evaluation results
        """
        logger.info(f"Cross-dataset evaluation on {dataset_name} ({len(X_test)} samples)")
        
        # Make predictions
        predictions = model_trainer.predict(X_test)
        
        # Compute metrics
        from src.evaluation.metrics import evaluate_model
        
        results = evaluate_model(
            y_test,
            predictions,
            training_time=model_trainer.training_time,
            model_name=f"{model_trainer.model_name}_{dataset_name}",
        )
        
        return results


def validate_translation_fidelity(
    df_chinese: pd.DataFrame,
    df_english: pd.DataFrame,
    chinese_column: str = "review",
    english_column: str = "review_en",
    sample_size: Optional[int] = None,
) -> Dict:
    """
    Validate translation fidelity between Chinese and English datasets.
    
    Args:
        df_chinese: DataFrame with Chinese texts
        df_english: DataFrame with English translations
        chinese_column: Column name with Chinese text
        english_column: Column name with English text
        sample_size: Optional sample size (for large datasets)
        
    Returns:
        Dictionary with fidelity metrics
    """
    validator = CrossDatasetValidator()
    
    # Align datasets (assuming same order or matching by index)
    chinese_texts = df_chinese[chinese_column].astype(str).tolist()
    english_texts = df_english[english_column].astype(str).tolist()
    
    # Sample if needed
    if sample_size is not None and len(chinese_texts) > sample_size:
        indices = np.random.choice(len(chinese_texts), sample_size, replace=False)
        chinese_texts = [chinese_texts[i] for i in indices]
        english_texts = [english_texts[i] for i in indices]
        logger.info(f"Sampled {sample_size} texts for translation fidelity analysis")
    
    fidelity_stats = validator.compute_translation_fidelity(chinese_texts, english_texts)
    
    return fidelity_stats


def validate_model_agreement(
    predictions_electra: np.ndarray,
    predictions_bert: np.ndarray,
    sample_size: Optional[int] = CROSS_DATASET_SAMPLE_SIZE,
) -> Dict:
    """
    Validate agreement between ELECTRA and BERT models.
    
    Args:
        predictions_electra: Predictions from ELECTRA model
        predictions_bert: Predictions from BERT model
        sample_size: Optional sample size (for large datasets)
        
    Returns:
        Dictionary with agreement metrics
    """
    validator = CrossDatasetValidator()
    
    # Sample if needed
    if sample_size is not None and len(predictions_electra) > sample_size:
        indices = np.random.choice(len(predictions_electra), sample_size, replace=False)
        predictions_electra = predictions_electra[indices]
        predictions_bert = predictions_bert[indices]
        logger.info(f"Sampled {sample_size} predictions for model agreement analysis")
    
    agreement_stats = validator.compute_model_agreement(predictions_electra, predictions_bert)
    
    return agreement_stats


def perform_cross_dataset_validation(
    model_electra,
    model_bert,
    X_test_dataset_b: List[str],
    y_test_dataset_b: np.ndarray,
    df_chinese: Optional[pd.DataFrame] = None,
    df_english: Optional[pd.DataFrame] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Perform comprehensive cross-dataset validation.
    
    Args:
        model_electra: Trained ELECTRA model trainer
        model_bert: Trained BERT model trainer
        X_test_dataset_b: Test texts from Dataset B
        y_test_dataset_b: Test labels from Dataset B
        df_chinese: Optional DataFrame with Chinese texts
        df_english: Optional DataFrame with English translations
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary with all validation results
    """
    logger.info("Performing comprehensive cross-dataset validation")
    
    results = {}
    
    # Translation fidelity
    if df_chinese is not None and df_english is not None:
        logger.info("Computing translation fidelity")
        fidelity_stats = validate_translation_fidelity(df_chinese, df_english)
        results['translation_fidelity'] = fidelity_stats
    
    # Model agreement on Dataset B
    logger.info("Computing ELECTRA-BERT agreement on Dataset B")
    predictions_electra = model_electra.predict(X_test_dataset_b)
    predictions_bert = model_bert.predict(X_test_dataset_b)
    
    agreement_stats = validate_model_agreement(predictions_electra, predictions_bert)
    results['model_agreement'] = agreement_stats
    
    # Cross-dataset performance
    logger.info("Evaluating models on cross-dataset (Dataset B)")
    validator = CrossDatasetValidator()
    
    results_electra = validator.cross_dataset_evaluation(
        model_electra,
        X_test_dataset_b,
        y_test_dataset_b,
        "dataset_b",
    )
    results['electra_cross_dataset'] = results_electra
    
    results_bert = validator.cross_dataset_evaluation(
        model_bert,
        X_test_dataset_b,
        y_test_dataset_b,
        "dataset_b",
    )
    results['bert_cross_dataset'] = results_bert
    
    # Save results
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(save_dir / "cross_dataset_validation.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved cross-dataset validation results to {save_dir}")
    
    return results
