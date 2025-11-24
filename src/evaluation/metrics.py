"""
Evaluation metrics module.

Computes accuracy, precision, recall, F1-score, and training time.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Optional
import logging
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import METRICS_DIR, METRICS_AVERAGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = METRICS_AVERAGE,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary' for binary classification, 'macro', 'micro', 'weighted' for multi-class)
        
    Returns:
        Dictionary with metrics
    """
    # Detect number of unique classes in predictions and true labels
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)
    n_classes_pred = len(unique_pred)
    n_classes_true = len(unique_true)
    
    # If 'binary' average is specified but we have more than 2 classes, adjust
    if average == 'binary':
        if n_classes_pred > 2 or n_classes_true > 2:
            logger.warning(f"Detected {n_classes_pred} unique predictions and {n_classes_true} unique labels, "
                          f"but 'binary' average specified. Switching to 'macro' average.")
            average = 'macro'
    
    # For binary classification, use 'binary' average
    # For multi-class, use the specified average
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names (defaults to ['negative', 'positive'] for binary)
        
    Returns:
        Dictionary with per-class metrics
    """
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    if class_names is None:
        # Default to binary class names if 2 classes
        if len(unique_classes) == 2:
            class_names = ['negative', 'positive']
        else:
            class_names = [f"Class_{c}" for c in unique_classes]
    
    # Per-class precision, recall, F1
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
            }
    
    # Macro averages (or binary average for binary classification)
    avg_type = 'binary' if len(unique_classes) == 2 else 'macro'
    per_class_metrics['macro_avg'] = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1),
    }
    
    return per_class_metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the matrix
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time: float = 0.0,
    model_name: str = "model",
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        training_time: Training time in seconds
        model_name: Name of the model
        class_names: Optional class names
        
    Returns:
        Dictionary with all evaluation results
    """
    logger.info(f"Evaluating {model_name}")
    
    # Overall metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics['training_time'] = training_time
    
    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, class_names)
    
    # Classification report
    report = get_classification_report(y_true, y_pred, class_names)
    
    # Confusion matrix
    cm = get_confusion_matrix(y_true, y_pred)
    cm_normalized = get_confusion_matrix(y_true, y_pred, normalize=True)
    
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
    }
    
    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return results


def save_metrics(
    results: Dict,
    filepath: Path,
    format: str = 'json',
):
    """
    Save evaluation metrics to file.
    
    Args:
        results: Evaluation results dictionary
        filepath: Path to save file
        format: File format ('json', 'csv')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        # Save main metrics as CSV
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved metrics to {filepath}")


def compare_models(
    results_list: List[Dict],
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results_list: List of evaluation results dictionaries
        save_path: Optional path to save comparison
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for results in results_list:
        model_name = results['model_name']
        metrics = results['metrics']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Training Time (s)': metrics['training_time'],
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved model comparison to {save_path}")
    
    return df
