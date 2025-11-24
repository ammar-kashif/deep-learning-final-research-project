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
    EARLY_STOPPING_PATIENCE_TRANSFORMER,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MONITOR,
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
    
    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None, tokenizer=None, max_length: int = TRANSFORMER_MAX_LENGTH, sentiment_features: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: Array of labels (None for inference)
            tokenizer: Tokenizer object
            max_length: Maximum sequence length
            sentiment_features: Optional sentiment features array (n_samples, 4)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiment_features = sentiment_features
    
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
        
        if self.sentiment_features is not None:
            item['sentiment_features'] = torch.tensor(self.sentiment_features[idx], dtype=torch.float)
        
        return item


class TransformerTrainer:
    """Trainer for transformer models (BERT, ELECTRA)."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,  # Binary sentiment classification
        device: Optional[str] = None,
        use_sentiment_features: bool = False,
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
        self.use_sentiment_features = use_sentiment_features
        
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
        
        # Modify classification head to accept sentiment features if enabled
        if use_sentiment_features:
            # Get the hidden size
            hidden_size = self.model.config.hidden_size
            # Replace classifier with one that accepts sentiment features (4 additional dims)
            self.model.classifier = nn.Linear(hidden_size + 4, num_classes)
        
        self.model.to(self.device)
        logger.info(f"Loaded {model_name} model")
        
        self.training_time = 0.0
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        X_train_sentiment: Optional[np.ndarray] = None,
        X_val_sentiment: Optional[np.ndarray] = None,
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
        train_dataset = TransformerDataset(X_train, y_train, self.tokenizer, sentiment_features=X_train_sentiment)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TransformerDataset(X_val, y_val, self.tokenizer, sentiment_features=X_val_sentiment)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        total_steps = len(train_loader) * epochs
        # Calculate warmup steps as 10% of total steps, but ensure it's at least 1
        # This prevents warmup_steps from exceeding total_steps (critical for small datasets)
        calculated_warmup = max(1, int(0.1 * total_steps))
        # Use the minimum of configured warmup and calculated warmup to prevent issues
        actual_warmup_steps = min(warmup_steps, calculated_warmup) if warmup_steps > 0 else calculated_warmup
        
        if actual_warmup_steps >= total_steps:
            logger.warning(f"Warmup steps ({actual_warmup_steps}) >= total steps ({total_steps}). "
                          f"Using {calculated_warmup} warmup steps (10% of total).")
            actual_warmup_steps = calculated_warmup
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=actual_warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(f"Training for {epochs} epochs, {total_steps} total steps, {actual_warmup_steps} warmup steps")
        
        # Early stopping
        class EarlyStopping:
            def __init__(self, patience=EARLY_STOPPING_PATIENCE_TRANSFORMER, min_delta=EARLY_STOPPING_MIN_DELTA, monitor=EARLY_STOPPING_MONITOR):
                self.patience = patience
                self.min_delta = min_delta
                self.monitor = monitor
                self.best_score = float('inf') if monitor == 'val_loss' else float('-inf')
                self.counter = 0
                self.best_weights = None
                self.early_stop = False
            
            def __call__(self, score, model):
                if self.monitor == 'val_loss':
                    is_better = score < self.best_score - self.min_delta
                else:  # val_acc or val_f1
                    is_better = score > self.best_score + self.min_delta
                
                if is_better:
                    self.best_score = score
                    self.counter = 0
                    self.best_weights = model.state_dict().copy()
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                
                return self.early_stop
        
        early_stopping = EarlyStopping() if val_loader is not None else None
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        logger.info(f"Training {self.model_name} for up to {epochs} epochs (early stopping enabled)")
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
                
                # Handle sentiment features if enabled
                if self.use_sentiment_features and 'sentiment_features' in batch:
                    sentiment_features = batch['sentiment_features'].to(self.device)
                    # Get base model outputs (without classification head)
                    # For BERT/ELECTRA sequence classification models, access the base model
                    if hasattr(self.model, 'bert'):
                        # Access BERT base model
                        base_outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                        # Extract [CLS] token (first token) from last hidden state
                        if hasattr(base_outputs, 'last_hidden_state'):
                            pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                        elif isinstance(base_outputs, tuple):
                            pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                        else:
                            # Fallback: try to get pooled output
                            pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                    elif hasattr(self.model, 'electra'):
                        # Access ELECTRA base model
                        base_outputs = self.model.electra(input_ids=input_ids, attention_mask=attention_mask)
                        # Extract [CLS] token (first token) from last hidden state
                        if hasattr(base_outputs, 'last_hidden_state'):
                            pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                        elif isinstance(base_outputs, tuple):
                            pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                        else:
                            # Fallback: try to get pooled output
                            pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                    else:
                        # Fallback: use the model directly and extract hidden states
                        outputs_base = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(outputs_base, 'hidden_states') and outputs_base.hidden_states is not None:
                            pooled_output = outputs_base.hidden_states[-1][:, 0]
                        elif hasattr(outputs_base, 'last_hidden_state'):
                            pooled_output = outputs_base.last_hidden_state[:, 0]
                        else:
                            # Use logits as pooled output (not ideal but works)
                            pooled_output = outputs_base.logits
                    
                    # Concatenate with sentiment features
                    combined_features = torch.cat([pooled_output, sentiment_features], dim=1)
                    # Pass through classifier
                    logits = self.model.classifier(combined_features)
                    # Compute loss
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    outputs = type('obj', (object,), {'loss': loss, 'logits': logits})()
                else:
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
                        
                        if self.use_sentiment_features and 'sentiment_features' in batch:
                            sentiment_features = batch['sentiment_features'].to(self.device)
                            # Get base model outputs (without classification head)
                            if hasattr(self.model, 'bert'):
                                base_outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                                if hasattr(base_outputs, 'last_hidden_state'):
                                    pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                                elif isinstance(base_outputs, tuple):
                                    pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                                else:
                                    pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                            elif hasattr(self.model, 'electra'):
                                base_outputs = self.model.electra(input_ids=input_ids, attention_mask=attention_mask)
                                if hasattr(base_outputs, 'last_hidden_state'):
                                    pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                                elif isinstance(base_outputs, tuple):
                                    pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                                else:
                                    pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                            else:
                                # Fallback
                                outputs_base = self.model(input_ids=input_ids, attention_mask=attention_mask)
                                if hasattr(outputs_base, 'hidden_states') and outputs_base.hidden_states is not None:
                                    pooled_output = outputs_base.hidden_states[-1][:, 0]
                                elif hasattr(outputs_base, 'last_hidden_state'):
                                    pooled_output = outputs_base.last_hidden_state[:, 0]
                                else:
                                    pooled_output = outputs_base.logits
                            
                            # Concatenate with sentiment features
                            combined_features = torch.cat([pooled_output, sentiment_features], dim=1)
                            # Pass through classifier
                            logits = self.model.classifier(combined_features)
                            # Compute loss
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(logits, labels)
                            outputs = type('obj', (object,), {'loss': loss, 'logits': logits})()
                        else:
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
                
                # Early stopping check
                if early_stopping is not None:
                    monitor_score = val_loss if EARLY_STOPPING_MONITOR == 'val_loss' else val_acc
                    if early_stopping(monitor_score, self.model):
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        self.model.load_state_dict(early_stopping.best_weights)
                        break
            
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
        X_test_sentiment: Optional[np.ndarray] = None,
        batch_size: int = TRANSFORMER_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: List of text strings
            X_test_sentiment: Optional sentiment features array (n_samples, 4)
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        # If model requires sentiment features but they're not provided, extract them
        if self.use_sentiment_features and X_test_sentiment is None:
            logger.info("Model requires sentiment features but none provided. Extracting sentiment features...")
            try:
                from src.features.sentiment_features import SentimentFeatureExtractor
                sentiment_extractor = SentimentFeatureExtractor()
                X_test_sentiment = sentiment_extractor.extract_features(X)
                logger.info(f"Extracted sentiment features: shape {X_test_sentiment.shape}")
            except Exception as e:
                logger.warning(f"Failed to extract sentiment features: {e}. Using zero features.")
                # Fallback: use zero features
                X_test_sentiment = np.zeros((len(X), 4), dtype=np.float32)
        
        self.model.eval()
        dataset = TransformerDataset(X, None, self.tokenizer, sentiment_features=X_test_sentiment)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_sentiment_features and 'sentiment_features' in batch:
                    sentiment_features = batch['sentiment_features'].to(self.device)
                    # Get base model outputs (without classification head)
                    if hasattr(self.model, 'bert'):
                        base_outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(base_outputs, 'last_hidden_state'):
                            pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                        elif isinstance(base_outputs, tuple):
                            pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                        else:
                            pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                    elif hasattr(self.model, 'electra'):
                        base_outputs = self.model.electra(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(base_outputs, 'last_hidden_state'):
                            pooled_output = base_outputs.last_hidden_state[:, 0]  # [CLS] token
                        elif isinstance(base_outputs, tuple):
                            pooled_output = base_outputs[0][:, 0]  # First element is hidden states
                        else:
                            pooled_output = base_outputs[0][:, 0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state[:, 0]
                    else:
                        # Fallback
                        outputs_base = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(outputs_base, 'hidden_states') and outputs_base.hidden_states is not None:
                            pooled_output = outputs_base.hidden_states[-1][:, 0]
                        elif hasattr(outputs_base, 'last_hidden_state'):
                            pooled_output = outputs_base.last_hidden_state[:, 0]
                        else:
                            pooled_output = outputs_base.logits
                    
                    # Concatenate with sentiment features
                    combined_features = torch.cat([pooled_output, sentiment_features], dim=1)
                    # Pass through classifier
                    logits = self.model.classifier(combined_features)
                    preds = torch.argmax(logits, dim=1)
                    predictions.extend(preds.cpu().numpy())
                else:
                    # Model doesn't use sentiment features, use standard forward pass
                    # But if classifier was modified (shouldn't happen if use_sentiment_features=False), handle it
                    if self.use_sentiment_features:
                        # This shouldn't happen if sentiment features were extracted above
                        # But handle it just in case
                        logger.warning("Model requires sentiment features but batch doesn't contain them. Using zero features.")
                        # Get base model outputs
                        if hasattr(self.model, 'bert'):
                            base_outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                            pooled_output = base_outputs.last_hidden_state[:, 0] if hasattr(base_outputs, 'last_hidden_state') else base_outputs[0][:, 0]
                        elif hasattr(self.model, 'electra'):
                            base_outputs = self.model.electra(input_ids=input_ids, attention_mask=attention_mask)
                            pooled_output = base_outputs.last_hidden_state[:, 0] if hasattr(base_outputs, 'last_hidden_state') else base_outputs[0][:, 0]
                        else:
                            outputs_base = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            pooled_output = outputs_base.hidden_states[-1][:, 0] if hasattr(outputs_base, 'hidden_states') else outputs_base.last_hidden_state[:, 0]
                        
                        # Use zero sentiment features
                        batch_size = pooled_output.size(0)
                        zero_sentiment = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)
                        combined_features = torch.cat([pooled_output, zero_sentiment], dim=1)
                        logits = self.model.classifier(combined_features)
                        preds = torch.argmax(logits, dim=1)
                        predictions.extend(preds.cpu().numpy())
                    else:
                        # Standard forward pass
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
    
    def __init__(self, num_classes: int = 2, device: Optional[str] = None, use_sentiment_features: bool = False):  # Binary sentiment classification
        """Initialize BERT trainer."""
        super().__init__(BERT_MODEL, num_classes, device, use_sentiment_features)


class ELECTRATrainer(TransformerTrainer):
    """ELECTRA-specific trainer."""
    
    def __init__(self, num_classes: int = 2, device: Optional[str] = None, use_sentiment_features: bool = False):  # Binary sentiment classification
        """Initialize ELECTRA trainer."""
        super().__init__(ELECTRA_MODEL, num_classes, device, use_sentiment_features)
