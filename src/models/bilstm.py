"""
BiLSTM model for text classification.

Configurable architecture with embedding and hidden layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
    BILSTM_EMBEDDING_DIM,
    BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS,
    BILSTM_BATCH_SIZE,
    BILSTM_LEARNING_RATE,
    BILSTM_DROPOUT,
    BILSTM_BIDIRECTIONAL,
    NUM_EPOCHS,
    RANDOM_SEED,
    USE_MPS,
    MODELS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class TextDataset(Dataset):
    """Dataset class for text classification."""
    
    def __init__(self, texts: np.ndarray, labels: np.ndarray, vocab_size: int, max_length: int):
        """
        Initialize dataset.
        
        Args:
            texts: Array of tokenized texts (each text is a list of token indices)
            labels: Array of labels
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Pad or truncate to max_length
        if len(text) > self.max_length:
            text = text[:self.max_length]
        else:
            text = np.pad(text, (0, self.max_length - len(text)), mode='constant')
        
        return torch.LongTensor(text), torch.LongTensor([label])


class BiLSTMClassifier(nn.Module):
    """BiLSTM model for text classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = BILSTM_EMBEDDING_DIM,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        num_classes: int = 2,
        dropout: float = BILSTM_DROPOUT,
        bidirectional: bool = BILSTM_BIDIRECTIONAL,
    ):
        """
        Initialize BiLSTM model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and fully connected layer
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output


class BiLSTMTrainer:
    """Trainer for BiLSTM model."""
    
    def __init__(
        self,
        vocab_size: int,
        device: Optional[str] = None,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Initialize BiLSTM trainer.
        
        Args:
            vocab_size: Vocabulary size
            device: Device to use ('cpu', 'cuda', 'mps')
            random_seed: Random seed
        """
        self.vocab_size = vocab_size
        
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
        
        self.model = None
        self.training_time = 0.0
    
    def build_model(
        self,
        embedding_dim: int = BILSTM_EMBEDDING_DIM,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        num_classes: int = 2,
        dropout: float = BILSTM_DROPOUT,
        bidirectional: bool = BILSTM_BIDIRECTIONAL,
    ) -> BiLSTMClassifier:
        """
        Build BiLSTM model.
        
        Args:
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            
        Returns:
            BiLSTM model
        """
        model = BiLSTMClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        model.to(self.device)
        self.model = model
        
        logger.info(f"Built BiLSTM model: {model}")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = NUM_EPOCHS,
        batch_size: int = BILSTM_BATCH_SIZE,
        learning_rate: float = BILSTM_LEARNING_RATE,
        max_length: int = 128,
    ) -> Dict:
        """
        Train BiLSTM model.
        
        Args:
            X_train: Training tokenized texts (list of token index lists)
            y_train: Training labels
            X_val: Validation tokenized texts (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model()
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train, self.vocab_size, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TextDataset(X_val, y_val, self.vocab_size, max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        logger.info(f"Training BiLSTM for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for texts, labels in train_loader:
                texts = texts.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
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
                    for texts, labels in val_loader:
                        texts = texts.to(self.device)
                        labels = labels.squeeze().to(self.device)
                        
                        outputs = self.model(texts)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
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
        logger.info(f"BiLSTM training completed in {self.training_time:.2f} seconds")
        
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = BILSTM_BATCH_SIZE, max_length: int = 128) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Tokenized texts
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        dataset = TextDataset(X, np.zeros(len(X)), self.vocab_size, max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for texts, _ in loader:
                texts = texts.to(self.device)
                outputs = self.model(texts)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, filepath: Path):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
        }, filepath)
        logger.info(f"Saved BiLSTM model to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.vocab_size = checkpoint['vocab_size']
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded BiLSTM model from {filepath}")
