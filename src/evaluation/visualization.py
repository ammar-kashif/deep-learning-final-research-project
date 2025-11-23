"""
Visualization module for training curves, confusion matrices, and performance comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PLOTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_curves(
    history: Dict,
    model_name: str,
    save_path: Optional[Path] = None,
):
    """
    Plot training curves (loss and accuracy).
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Optional class names
        normalize: Whether matrix is normalized
        model_name: Name of the model
        save_path: Optional path to save plot
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    
    if normalize:
        fmt = '.2f'
        title = f'{model_name} - Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = f'{model_name} - Confusion Matrix'
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """
    Plot model comparison bar chart.
    
    Args:
        comparison_df: DataFrame with model comparison data
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        bars = ax.barh(comparison_df['Model'], comparison_df[metric], color='steelblue')
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")
    
    plt.close()


def plot_training_time_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """
    Plot training time comparison.
    
    Args:
        comparison_df: DataFrame with model comparison data
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    bars = plt.barh(comparison_df['Model'], comparison_df['Training Time (s)'], color='coral')
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}s', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training time comparison to {save_path}")
    
    plt.close()


def create_comprehensive_report(
    results_list: List[Dict],
    save_dir: Path,
):
    """
    Create comprehensive visualization report for all models.
    
    Args:
        results_list: List of evaluation results dictionaries
        save_dir: Directory to save all plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating comprehensive visualization report")
    
    # Model comparison
    from src.evaluation.metrics import compare_models
    comparison_df = compare_models(results_list, save_dir / "model_comparison.csv")
    
    # Plot comparisons
    plot_model_comparison(comparison_df, save_dir / "model_comparison.png")
    plot_training_time_comparison(comparison_df, save_dir / "training_time_comparison.png")
    
    # Individual model plots
    for results in results_list:
        model_name = results['model_name']
        safe_name = model_name.lower().replace(' ', '_')
        
        # Confusion matrix
        cm = np.array(results['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            model_name=model_name,
            save_path=save_dir / f"{safe_name}_confusion_matrix.png",
        )
        
        # Normalized confusion matrix
        cm_norm = np.array(results['confusion_matrix_normalized'])
        plot_confusion_matrix(
            cm_norm,
            normalize=True,
            model_name=model_name,
            save_path=save_dir / f"{safe_name}_confusion_matrix_normalized.png",
        )
    
    logger.info(f"Comprehensive report saved to {save_dir}")
