"""
Transformer models for text classification: BERT and ELECTRA.

Fine-tuning with 50 epochs and configurable hyperparameters.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    BERT_MODEL,
    ELECTRA_MODEL,
    NUM_EPOCHS,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_WEIGHT_DECAY,
    TRANSFORMER_WARMUP_STEPS,
    TRANSFORMER_MAX_LENGTH,
    RANDOM_SEED,
    USE_MPS,
    MODELS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class TransformerDataset(Dataset):
    """Dataset for transformer models."""
    
    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None, tokenizer=None, max_length: int = TRANSFORMER_MAX_LENGTH):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: Array of labels (None for inference)
            tokenizer: Tokenizer object
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class TransformerTrainer:
    """Trainer for transformer models (BERT, ELECTRA)."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize transformer trainer.
        
        Args:
            model_name: Name of the model (BERT or ELECTRA)
            num_classes: Number of output classes
            device: Device to use
        """
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            if USE_MPS and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading {model_name} tokenizer and model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if 'bert' in model_name.lower():
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
            )
        elif 'electra' in model_name.lower():
            self.model = ElectraForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
            )
        else:
            # Fallback to AutoModel
            self.model = AutoModel.from_pretrained(model_name)
            # Add classification head
            self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        
        self.model.to(self.device)
        logger.info(f"Loaded {model_name} model")
        
        self.training_time = 0.0
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = NUM_EPOCHS,
        batch_size: int = TRANSFORMER_BATCH_SIZE,
        learning_rate: float = TRANSFORMER_LEARNING_RATE,
        weight_decay: float = TRANSFORMER_WEIGHT_DECAY,
        warmup_steps: int = TRANSFORMER_WARMUP_STEPS,
    ) -> Dict:
        """
        Train transformer model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps for learning rate scheduler
            
        Returns:
            Training history
        """
        # Create datasets
        train_dataset = TransformerDataset(X_train, y_train, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TransformerDataset(X_val, y_val, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        logger.info(f"Training {self.model_name} for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                train_total += labels.size(0)
                train_correct += (predictions == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        
                        val_loss += outputs.loss.item()
                        predictions = torch.argmax(outputs.logits, dim=1)
                        val_total += labels.size(0)
                        val_correct += (predictions == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                if val_loader is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                logger.info(log_msg)
        
        self.training_time = time.time() - start_time
        logger.info(f"{self.model_name} training completed in {self.training_time:.2f} seconds")
        
        return history
    
    def predict(
        self,
        X: List[str],
        batch_size: int = TRANSFORMER_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: List of text strings
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        self.model.eval()
        dataset = TransformerDataset(X, None, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, filepath: Path):
        """Save model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"Saved {self.model_name} model to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load model from disk."""
        if 'bert' in self.model_name.lower():
            self.model = BertForSequenceClassification.from_pretrained(filepath)
        elif 'electra' in self.model_name.lower():
            self.model = ElectraForSequenceClassification.from_pretrained(filepath)
        else:
            self.model = AutoModel.from_pretrained(filepath)
        
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.model.to(self.device)
        logger.info(f"Loaded {self.model_name} model from {filepath}")


class BERTTrainer(TransformerTrainer):
    """BERT-specific trainer."""
    
    def __init__(self, num_classes: int = 2, device: Optional[str] = None):
        """Initialize BERT trainer."""
        super().__init__(BERT_MODEL, num_classes, device)


class ELECTRATrainer(TransformerTrainer):
    """ELECTRA-specific trainer."""
    
    def __init__(self, num_classes: int = 2, device: Optional[str] = None):
        """Initialize ELECTRA trainer."""
        super().__init__(ELECTRA_MODEL, num_classes, device)
