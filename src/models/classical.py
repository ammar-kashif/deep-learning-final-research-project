"""
Classical machine learning models for text classification.

Includes: Linear Regression, SVC with SGD, Random Forest, XGBoost.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Optional, List
import logging
import pickle
from pathlib import Path
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CLASSICAL_MODELS, RANDOM_SEED, MODELS_DIR, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalModelTrainer:
    """Trainer for classical ML models."""
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize classical model trainer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.models = {}
        self.training_times = {}
    
    def train_linear_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> LinearRegression:
        """
        Train Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained LinearRegression model
        """
        logger.info("Training Linear Regression model")
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['linear_regression'] = training_time
        self.models['linear_regression'] = model
        
        logger.info(f"Linear Regression trained in {training_time:.2f} seconds")
        
        return model
    
    def train_svc_sgd(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> SGDClassifier:
        """
        Train SVC with SGD optimizer.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for SGDClassifier
            
        Returns:
            Trained SGDClassifier model
        """
        logger.info("Training SVC with SGD")
        start_time = time.time()
        
        params = CLASSICAL_MODELS['svc_sgd'].copy()
        params.update(kwargs)
        params['random_state'] = self.random_seed
        
        model = SGDClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['svc_sgd'] = training_time
        self.models['svc_sgd'] = model
        
        logger.info(f"SVC with SGD trained in {training_time:.2f} seconds")
        
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest model")
        start_time = time.time()
        
        params = CLASSICAL_MODELS['random_forest'].copy()
        params.update(kwargs)
        params['random_state'] = self.random_seed
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['random_forest'] = training_time
        self.models['random_forest'] = model
        
        logger.info(f"Random Forest trained in {training_time:.2f} seconds")
        
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            Trained XGBClassifier model
        """
        logger.info("Training XGBoost model")
        start_time = time.time()
        
        params = CLASSICAL_MODELS['xgboost'].copy()
        params.update(kwargs)
        params['random_state'] = self.random_seed
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['xgboost'] = training_time
        self.models['xgboost'] = model
        
        logger.info(f"XGBoost trained in {training_time:.2f} seconds")
        
        return model
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Train all classical models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train (None = all)
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['linear_regression', 'svc_sgd', 'random_forest', 'xgboost']
        
        logger.info(f"Training {len(models_to_train)} classical models")
        
        for model_name in models_to_train:
            if model_name == 'linear_regression':
                self.train_linear_regression(X_train, y_train)
            elif model_name == 'svc_sgd':
                self.train_svc_sgd(X_train, y_train)
            elif model_name == 'random_forest':
                self.train_random_forest(X_train, y_train)
            elif model_name == 'xgboost':
                self.train_xgboost(X_train, y_train)
            else:
                logger.warning(f"Unknown model: {model_name}")
        
        return self.models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a trained model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name]
        
        # Linear Regression outputs continuous values, need to convert to valid class range
        if isinstance(model, LinearRegression):
            predictions = model.predict(X)
            # Clip to valid class range [0, num_classes-1] and round to nearest integer
            # For binary classification (NUM_CLASSES=2), range is [0, 1]
            predictions = np.clip(np.round(predictions), 0, NUM_CLASSES - 1).astype(int)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def evaluate(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(model_name, X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall': recall_score(y_test, predictions, average='macro', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='macro', zero_division=0),
            'training_time': self.training_times.get(model_name, 0.0),
        }
        
        return metrics
    
    def save_model(self, model_name: str, filepath: Path):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        logger.info(f"Saved {model_name} to {filepath}")
    
    def load_model(self, model_name: str, filepath: Path):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        logger.info(f"Loaded {model_name} from {filepath}")
