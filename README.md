# Sentiment Analysis Research Paper Replication

Complete, reproducible Python implementation for replicating the research paper on sentiment analysis and text classification of Chinese product reviews.

## Overview

This project implements a comprehensive binary sentiment classification pipeline that:
- Loads Chinese product review datasets
- Translates Chinese reviews to English using MarianMT
- Preprocesses text in both languages
- Performs sentiment analysis using VADER
- Extracts N-grams and builds co-occurrence networks
- Trains multiple models (classical ML, BiLSTM, BERT, ELECTRA) for **binary sentiment classification**
- Integrates VADER sentiment features as model inputs (feature-level sentiment analysis)
- Evaluates models with comprehensive metrics
- Performs cross-dataset validation

## Classification Task

The primary task is **binary sentiment classification** (positive/negative) from review text, matching the paper's methodology.

**Labels:**
- `0` = Negative sentiment
- `1` = Positive sentiment

**Expected Performance (from paper):**
- ELECTRA: ~98% accuracy, F1: ~0.97
- BiLSTM: ~96% accuracy, F1: ~0.96
- BERT: ~95% accuracy, F1: ~0.95
- Random Forest: ~82% accuracy
- SVC with SGD: ~89% accuracy

## Feature-Level Sentiment Analysis

The paper uses "feature-level sentiment analysis" where VADER sentiment scores (compound, pos, neu, neg) are integrated as additional features alongside TF-IDF and transformer embeddings. This is enabled by default via `USE_SENTIMENT_FEATURES = True` in `config.py`.

## Project Structure

```
PROJ/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── config.py                     # Centralized configuration
├── main.py                       # Main execution script
├── src/
│   ├── data/
│   │   ├── loader.py            # Dataset loading
│   │   └── translator.py       # Chinese-to-English translation
│   ├── preprocessing/
│   │   ├── english.py          # English text preprocessing
│   │   └── chinese.py          # Chinese text preprocessing
│   ├── analysis/
│   │   ├── sentiment.py        # VADER sentiment analysis
│   │   ├── ngrams.py           # N-gram extraction
│   │   └── network.py          # Network graph construction
│   ├── features/
│   │   ├── tfidf.py            # TF-IDF feature extraction
│   │   └── transformers.py     # Transformer tokenization
│   ├── models/
│   │   ├── classical.py        # Classical ML models
│   │   ├── bilstm.py           # BiLSTM model
│   │   └── transformers.py     # BERT and ELECTRA models
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualization.py    # Visualization functions
│   └── validation/
│       └── cross_dataset.py     # Cross-dataset validation
├── results/
│   ├── models/                  # Saved model checkpoints
│   ├── predictions/             # Prediction outputs
│   ├── metrics/                 # Evaluation metrics (CSV/JSON)
│   └── plots/                   # Visualization outputs
├── datasetA/                    # Dataset A (already downloaded)
│   ├── Dataset A - Chinese.csv
│   ├── Dataset A Translated.csv
│   └── Dataset B Translated.csv
└── datasetB/                     # Dataset B (already downloaded)
    └── Online Shopping 10 Cats.csv
```

## Installation

### For macOS (Apple Silicon - M1/M2/M3)

**Recommended: Use the automated installation script:**
```bash
./install_dependencies.sh
```

**Manual installation:**
1. Install PyTorch first (required for MPS support):
   ```bash
   pip install torch torchvision torchaudio
   ```

2. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify MPS is working:
   ```bash
   python3 verify_mps.py
   ```

### For Other Systems

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
   ```

**Note:** For detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

## Usage

### Running the Complete Pipeline

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. Load datasets from `datasetA/` and `datasetB/`
2. Translate Chinese reviews to English (or use existing translations)
3. Preprocess text in both languages
4. Perform sentiment analysis on Dataset A
5. Extract Chinese N-grams and build network graphs
6. Extract features (TF-IDF and transformer tokens)
7. Train all models (50 epochs for deep learning models)
8. Evaluate all models
9. Perform cross-dataset validation
10. Generate comprehensive reports

### Output

Results are saved in the `results/` directory:
- **`results/metrics/`**: Evaluation metrics in JSON/CSV format
- **`results/plots/`**: Visualization plots (training curves, confusion matrices, etc.)
- **`results/models/`**: Saved model checkpoints
- **`results/predictions/`**: Prediction outputs

## Configuration

All hyperparameters and settings are centralized in `config.py`. Key configuration options:

- **Train/Test Split**: 80/20 stratified (ensures class balance)
- **TF-IDF max_features**: 30,000 (configurable 20k-50k)
- **Random Seed**: 42 (for reproducibility)
- **Batch Sizes**: 32 for transformers, 64 for BiLSTM
- **Learning Rates**: 2e-5 for transformers, 1e-3 for BiLSTM
- **Epochs**: 50 for all deep learning models
- **Model Checkpoints**:
  - BERT: `bert-base-uncased`
  - ELECTRA: `google/electra-base-discriminator`

## Models Implemented

### Classical Machine Learning Models
- **Linear Regression**: Baseline regression model
- **SVC with SGD**: Support Vector Classifier with Stochastic Gradient Descent
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework

### Deep Learning Models
- **BiLSTM**: Bidirectional LSTM with configurable architecture
  - Embedding dimension: 128
  - Hidden dimension: 256
  - Number of layers: 2
- **BERT**: Fine-tuned BERT-base-uncased
- **ELECTRA**: Fine-tuned ELECTRA-base-discriminator

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1-Score**: Macro-averaged F1-score
- **Training Time**: Time taken to train the model

## Cross-Dataset Validation

The pipeline includes cross-dataset validation:
- **Translation Fidelity**: Cosine similarity between original Chinese and translated English embeddings
- **Model Agreement**: ELECTRA-BERT agreement rate and Cohen's Kappa
- **Cross-Dataset Performance**: Model performance on Dataset B when trained on Dataset A

## Implementation Decisions

Where the paper left specifications ambiguous, the following decisions were made:

1. **Train/Val/Test Split**: 70/15/15 stratified split (enables early stopping and validation)
2. **TF-IDF max_features**: 30,000 (balance between performance and memory)
3. **Random Seed**: 42 (for reproducibility across runs)
4. **Batch Sizes**: 32 for transformers (memory efficient), 64 for BiLSTM
5. **Learning Rates**: 2e-5 for transformers (standard), 1e-3 for BiLSTM
6. **BiLSTM Architecture**: 2 layers, 256 hidden units, 128 embedding dim
7. **Translation Chunking**: 24 sentences per batch (balance between speed and memory)
8. **MPS Acceleration**: Uses PyTorch MPS backend for Mac M3 optimization
9. **Evaluation Metrics**: Binary classification metrics (accuracy, precision, recall, F1)
10. **Early Stopping**: Enabled for BiLSTM (patience=5) and transformers (patience=3) to prevent overfitting
11. **Sentiment Features**: VADER scores (compound, pos, neu, neg) integrated as 4 additional features per text

## Hardware Requirements

- **Recommended**: MacBook M3 Air or similar with MPS acceleration support
- **Minimum**: CPU with 8GB RAM (training will be slower)
- **Storage**: ~5GB for models and datasets

## Notes

- Translation models are downloaded automatically on first use
- Pre-trained transformer models are downloaded automatically
- Existing translations in `datasetA/` are used if available (can be overridden)
- All random seeds are set for reproducibility

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes in `config.py`
2. **Translation Timeout**: Increase `TRANSLATION_BATCH_SIZE` or use existing translations
3. **NLTK Data Missing**: Run the NLTK download commands in the Installation section
4. **MPS Not Available**: The code will automatically fallback to CPU

## Citation

If you use this implementation, please cite the original research paper.

## License

This implementation is provided for research and educational purposes.
