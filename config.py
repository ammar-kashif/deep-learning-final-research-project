"""
Configuration file for Sentiment Analysis Research Paper Replication

This module centralizes all hyperparameters, model settings, and implementation choices.
All decisions where the paper left specifications ambiguous are documented here.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_A_DIR = PROJECT_ROOT / "datasetA"
DATASET_B_DIR = PROJECT_ROOT / "datasetB"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Dataset file paths
DATASET_A_CHINESE = DATASET_A_DIR / "Dataset A - Chinese.csv"
DATASET_A_TRANSLATED = DATASET_A_DIR / "Dataset A Translated.csv"
DATASET_B_TRANSLATED = DATASET_A_DIR / "Dataset B Translated.csv"
DATASET_B_FILE = DATASET_B_DIR / "Online Shopping 10 Cats.csv"

# Reproducibility
RANDOM_SEED = 42

# Data splitting
# Paper did not specify exact split ratio - using train/val/test split for early stopping
# Split: 70% train, 15% validation, 15% test (stratified to maintain class balance)
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)  # (train, val, test) proportions
STRATIFIED_SPLIT = True

# Translation settings
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-zh-en"
TRANSLATION_BATCH_SIZE = 24  # Balance between speed and memory (16-32 range)
TRANSLATION_DEVICE = "cpu"  # Will be set to MPS if available

# Text preprocessing
# English preprocessing uses NLTK
ENGLISH_STOPWORDS = True
ENGLISH_STEMMING = True  # Porter stemmer

# Chinese preprocessing uses Jieba
CHINESE_STOPWORDS = True
CHINESE_STOPWORDS_SOURCE = "open-source"  # Using open-source Chinese stopwords

# Feature extraction - TF-IDF
# Paper did not specify max_features - using 30,000 as default
# This balances performance and memory usage (configurable 20k-50k)
TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Transformer tokenization
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_PADDING = True
TRANSFORMER_TRUNCATION = True

# Model checkpoints
BERT_MODEL = "bert-base-uncased"
ELECTRA_MODEL = "google/electra-base-discriminator"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Training hyperparameters
NUM_EPOCHS = 50  # Maximum epochs (early stopping will terminate earlier if validation doesn't improve)

# Early stopping
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before stopping (BiLSTM)
EARLY_STOPPING_PATIENCE_TRANSFORMER = 3  # Patience for transformers (converge faster)
EARLY_STOPPING_MIN_DELTA = 0.0  # Minimum change to qualify as improvement
EARLY_STOPPING_MONITOR = "val_loss"  # Metric to monitor: "val_loss" or "val_f1"

# Transformer models (BERT, ELECTRA)
TRANSFORMER_BATCH_SIZE = 32  # Memory efficient for transformers
TRANSFORMER_LEARNING_RATE = 2e-5  # Standard learning rate for transformers
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_STEPS = 500

# BiLSTM model
BILSTM_EMBEDDING_DIM = 128
BILSTM_HIDDEN_DIM = 256
BILSTM_NUM_LAYERS = 2
BILSTM_BATCH_SIZE = 64
BILSTM_LEARNING_RATE = 1e-3
BILSTM_DROPOUT = 0.3
BILSTM_BIDIRECTIONAL = True

# Classical ML models
CLASSICAL_MODELS = {
    "linear_regression": {},
    "svc_sgd": {
        "loss": "hinge",
        "penalty": "l2",
        "alpha": 0.0001,
        "max_iter": 1000,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
    },
}

# Hardware acceleration
# Use MPS (Metal Performance Shaders) for Mac M3 optimization
USE_MPS = True
DEVICE = "mps"  # Will fallback to "cuda" or "cpu" if MPS not available

# Analysis settings
# Sentiment analysis
SENTIMENT_ANALYZER = "vader"  # NLTK VADER SentimentIntensityAnalyzer

# N-gram analysis
TOP_N_BIGRAMS = 25
TOP_N_TRIGRAMS = 25

# Network analysis
NETWORK_MIN_COOCCURRENCE = 2  # Minimum co-occurrence frequency for edges
NETWORK_TOP_NODES = 100  # Top nodes by degree for visualization

# Cross-dataset validation
CROSS_DATASET_SAMPLE_SIZE = 10000  # Number of reviews for model agreement analysis

# Classification task
# The paper performs binary sentiment classification (positive/negative)
# Using label column: 0=negative, 1=positive
NUM_CLASSES = 2  # Binary sentiment classification

# Feature engineering
# Paper uses "feature-level sentiment analysis" - integrate VADER sentiment scores as features
USE_SENTIMENT_FEATURES = True  # Enable sentiment feature integration with TF-IDF/embeddings

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1_score"]
METRICS_AVERAGE = "binary"  # Binary classification metrics

# Output settings
SAVE_MODELS = True
SAVE_PREDICTIONS = True
SAVE_METRICS = True
SAVE_PLOTS = True

# Logging
LOG_LEVEL = "INFO"
VERBOSE = True
