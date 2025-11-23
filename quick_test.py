#!/usr/bin/env python3
"""
Quick test script for end-to-end pipeline validation.

Runs the complete pipeline with reduced parameters:
- 1 epoch instead of 50
- Smaller sample sizes for faster execution
- All stages are executed to verify the pipeline works
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Temporarily override config values for quick testing
import config
config.NUM_EPOCHS = 1  # Single epoch for quick test
config.CROSS_DATASET_SAMPLE_SIZE = 100  # Reduced from 10000
config.TRANSFORMER_WARMUP_STEPS = 10  # Reduced warmup steps
config.TOP_N_BIGRAMS = 10  # Reduced from 25
config.TOP_N_TRIGRAMS = 10  # Reduced from 25
config.NETWORK_TOP_NODES = 50  # Reduced from 100

from config import (
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    STRATIFIED_SPLIT,
    RESULTS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    NUM_EPOCHS,  # Now 1
)

# Import modules
from src.data import loader
from src.data import translator
from src.preprocessing import english as eng_prep
from src.preprocessing import chinese as chn_prep
from src.analysis import sentiment
from src.analysis import ngrams
from src.analysis import network
from src.features import tfidf
from src.features import transformers as trans_features
from src.models import classical
from src.models import bilstm
from src.models import transformers as trans_models
from src.evaluation import metrics
from src.evaluation import visualization
from src.validation import cross_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(RANDOM_SEED)


def main():
    """Quick test of the complete pipeline."""
    logger.info("=" * 80)
    logger.info("QUICK TEST MODE - Running pipeline with 1 epoch")
    logger.info("=" * 80)
    
    # Create results directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # ============================================================================
        # Step 1: Load datasets
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Loading Datasets")
        logger.info("=" * 80)
        
        df_a = loader.load_dataset_a()
        df_b = loader.load_dataset_b()
        
        # Use smaller sample for quick test
        logger.info(f"Using sample of 1000 rows for quick test (full: {len(df_a)} rows)")
        df_a = df_a.sample(n=min(1000, len(df_a)), random_state=RANDOM_SEED).reset_index(drop=True)
        df_b = df_b.sample(n=min(500, len(df_b)), random_state=RANDOM_SEED).reset_index(drop=True)
        
        logger.info(f"Dataset A (sample): {len(df_a)} samples")
        logger.info(f"Dataset B (sample): {len(df_b)} samples")
        
        # ============================================================================
        # Step 2: Translate Chinese to English
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Translating Chinese to English")
        logger.info("=" * 80)
        
        # Check for existing translations
        translated_files = loader.check_translated_files()
        
        # For quick test, use existing translations if available, otherwise skip translation
        if translated_files['dataset_a_translated']:
            logger.info("Using existing translation for Dataset A")
            df_a_translated = loader.load_translated_dataset_a()
            if df_a_translated is None or 'review_en' not in df_a_translated.columns:
                logger.info("Skipping translation for quick test (use existing or run full pipeline)")
                df_a_translated = df_a.copy()
                df_a_translated['review_en'] = df_a['review']  # Use Chinese as placeholder
        else:
            logger.info("Skipping translation for quick test (use existing or run full pipeline)")
            df_a_translated = df_a.copy()
            df_a_translated['review_en'] = df_a['review']  # Use Chinese as placeholder
        
        if translated_files['dataset_b_translated']:
            logger.info("Using existing translation for Dataset B")
            df_b_translated = loader.load_translated_dataset_b()
            if df_b_translated is None or 'review_en' not in df_b_translated.columns:
                df_b_translated = df_b.copy()
                df_b_translated['review_en'] = df_b['review']
        else:
            df_b_translated = df_b.copy()
            df_b_translated['review_en'] = df_b['review']
        
        # ============================================================================
        # Step 3: Preprocess text (both languages)
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Preprocessing Text")
        logger.info("=" * 80)
        
        # English preprocessing
        logger.info("Preprocessing English text (Dataset A)")
        eng_preprocessor = eng_prep.EnglishPreprocessor()
        df_a_translated['review_en_processed'] = df_a_translated['review_en'].apply(
            eng_preprocessor.preprocess
        )
        
        logger.info("Preprocessing English text (Dataset B)")
        df_b_translated['review_en_processed'] = df_b_translated['review_en'].apply(
            eng_preprocessor.preprocess
        )
        
        # Chinese preprocessing
        logger.info("Preprocessing Chinese text (Dataset A)")
        chn_preprocessor = chn_prep.ChinesePreprocessor()
        df_a['review_tokens'] = df_a['review'].apply(chn_preprocessor.preprocess)
        
        logger.info("Preprocessing Chinese text (Dataset B)")
        df_b['review_tokens'] = df_b['review'].apply(chn_preprocessor.preprocess)
        
        # ============================================================================
        # Step 4: Perform sentiment analysis on Dataset A
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Sentiment Analysis on Dataset A")
        logger.info("=" * 80)
        
        sentiment_analyzer = sentiment.SentimentAnalyzer()
        df_a_sentiment = sentiment.analyze_sentiment_dataframe(
            df_a_translated,
            text_column="review_en",
            analyzer=sentiment_analyzer,
        )
        
        # Get sentiment statistics
        sentiment_stats = sentiment.get_sentiment_statistics(df_a_sentiment)
        logger.info(f"Sentiment Statistics:")
        logger.info(f"  Mean: {sentiment_stats['mean']:.4f}")
        
        # ============================================================================
        # Step 5: Extract Chinese N-grams and build network graphs
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Chinese N-gram and Network Analysis")
        logger.info("=" * 80)
        
        # N-gram extraction
        logger.info("Extracting N-grams from Dataset A")
        ngram_extractor = ngrams.NGramExtractor()
        chinese_tokens = df_a['review_tokens'].tolist()
        ngram_results = ngram_extractor.analyze_ngrams(chinese_tokens)
        
        logger.info(f"Top 3 Bigrams: {ngram_results['top_bigrams'][:3]}")
        
        # Network analysis (simplified for quick test)
        logger.info("Building co-occurrence network (simplified)")
        network_analyzer = network.CoOccurrenceNetwork()
        G = network_analyzer.build_graph(chinese_tokens, top_n_nodes=50)
        
        logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # ============================================================================
        # Step 6: Extract features (TF-IDF and transformer tokens)
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Feature Extraction")
        logger.info("=" * 80)
        
        # Prepare data for classification
        X_texts = df_a_translated['review_en_processed'].tolist()
        y_labels = df_a_translated['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_texts,
            y_labels,
            test_size=1 - TRAIN_TEST_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_labels if STRATIFIED_SPLIT else None,
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # TF-IDF features
        logger.info("Extracting TF-IDF features")
        tfidf_extractor = tfidf.TFIDFExtractor(max_features=5000)  # Reduced for quick test
        X_train_tfidf = tfidf_extractor.fit_transform(X_train)
        X_test_tfidf = tfidf_extractor.transform(X_test)
        
        # Transformer tokenization
        logger.info("Tokenizing for transformer models")
        bert_tokenizer = trans_features.BERTTokenizer()
        electra_tokenizer = trans_features.ELECTRATokenizer()
        
        # ============================================================================
        # Step 7: Train all models (1 epoch for quick test)
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Training Models (1 epoch for quick test)")
        logger.info("=" * 80)
        
        all_results = []
        
        # Classical models
        logger.info("Training classical ML models")
        classical_trainer = classical.ClassicalModelTrainer()
        classical_trainer.train_all(X_train_tfidf, y_train)
        
        # Evaluate classical models
        for model_name in ['linear_regression', 'svc_sgd', 'random_forest', 'xgboost']:
            predictions = classical_trainer.predict(model_name, X_test_tfidf)
            result = metrics.evaluate_model(
                y_test,
                predictions,
                training_time=classical_trainer.training_times[model_name],
                model_name=model_name,
            )
            all_results.append(result)
        
        # BiLSTM
        logger.info("Training BiLSTM model (1 epoch)")
        from collections import Counter
        all_tokens = []
        for text in X_train:
            all_tokens.extend(text.split())
        vocab = Counter(all_tokens)
        vocab_size = len(vocab) + 1
        
        word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab.keys())}
        
        def text_to_indices(text):
            return [word_to_idx.get(word, 0) for word in text.split() if word in word_to_idx]
        
        X_train_bilstm = [text_to_indices(text) for text in X_train]
        X_test_bilstm = [text_to_indices(text) for text in X_test]
        
        bilstm_trainer = bilstm.BiLSTMTrainer(vocab_size=vocab_size)
        bilstm_trainer.build_model()
        bilstm_history = bilstm_trainer.train(
            np.array(X_train_bilstm, dtype=object),
            y_train,
            epochs=1,  # Single epoch
        )
        
        predictions_bilstm = bilstm_trainer.predict(np.array(X_test_bilstm, dtype=object))
        result_bilstm = metrics.evaluate_model(
            y_test,
            predictions_bilstm,
            training_time=bilstm_trainer.training_time,
            model_name="BiLSTM",
        )
        all_results.append(result_bilstm)
        
        # BERT
        logger.info("Training BERT model (1 epoch)")
        bert_trainer = trans_models.BERTTrainer()
        bert_history = bert_trainer.train(X_train, y_train, epochs=1)
        
        predictions_bert = bert_trainer.predict(X_test)
        result_bert = metrics.evaluate_model(
            y_test,
            predictions_bert,
            training_time=bert_trainer.training_time,
            model_name="BERT",
        )
        all_results.append(result_bert)
        
        # ELECTRA
        logger.info("Training ELECTRA model (1 epoch)")
        electra_trainer = trans_models.ELECTRATrainer()
        electra_history = electra_trainer.train(X_train, y_train, epochs=1)
        
        predictions_electra = electra_trainer.predict(X_test)
        result_electra = metrics.evaluate_model(
            y_test,
            predictions_electra,
            training_time=electra_trainer.training_time,
            model_name="ELECTRA",
        )
        all_results.append(result_electra)
        
        # ============================================================================
        # Step 8: Evaluate all models
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Model Evaluation")
        logger.info("=" * 80)
        
        # Save all metrics with _test_ prefix to avoid conflicts
        for result in all_results:
            model_name = result['model_name']
            safe_name = model_name.lower().replace(' ', '_')
            metrics.save_metrics(result, METRICS_DIR / f"{safe_name}_test_metrics.json")
        
        # Create comparison
        comparison_df = metrics.compare_models(all_results, METRICS_DIR / "model_comparison_test.csv")
        logger.info("\nModel Comparison (Quick Test):")
        logger.info(comparison_df.to_string())
        
        # ============================================================================
        # Step 9: Cross-dataset validation (simplified)
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("Step 9: Cross-Dataset Validation (Simplified)")
        logger.info("=" * 80)
        
        # Prepare Dataset B for cross-dataset validation
        X_test_b = df_b_translated['review_en_processed'].tolist()[:100]  # Small sample
        y_test_b = df_b_translated['label'].values[:100]
        
        logger.info("Computing ELECTRA-BERT agreement on small sample")
        predictions_electra_b = electra_trainer.predict(X_test_b)
        predictions_bert_b = bert_trainer.predict(X_test_b)
        
        agreement_stats = cross_dataset.validate_model_agreement(
            predictions_electra_b,
            predictions_bert_b,
            sample_size=100,
        )
        logger.info(f"Model Agreement: {agreement_stats['agreement_rate']:.4f}")
        
        # ============================================================================
        # Step 10: Summary
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("QUICK TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nAll pipeline stages executed without errors.")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info("\nTo run the full pipeline with 50 epochs, use: python3 main.py")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during quick test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
