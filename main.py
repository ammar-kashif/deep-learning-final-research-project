"""
Main execution script for sentiment analysis research paper replication.

Orchestrates the complete pipeline from data loading to final evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    STRATIFIED_SPLIT,
    RESULTS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
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
    """Main execution pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Sentiment Analysis Research Paper Replication")
    logger.info("=" * 80)
    
    # Create results directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Step 1: Load datasets
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Datasets")
    logger.info("=" * 80)
    
    df_a = loader.load_dataset_a()
    df_b = loader.load_dataset_b()
    
    logger.info(f"Dataset A: {len(df_a)} samples")
    logger.info(f"Dataset B: {len(df_b)} samples")
    
    # ============================================================================
    # Step 2: Translate Chinese to English
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Translating Chinese to English")
    logger.info("=" * 80)
    
    # Check for existing translations
    translated_files = loader.check_translated_files()
    
    # Translate Dataset A
    if translated_files['dataset_a_translated']:
        logger.info("Using existing translation for Dataset A")
        df_a_translated = loader.load_translated_dataset_a()
        if df_a_translated is None or 'review_en' not in df_a_translated.columns:
            logger.info("Re-translating Dataset A")
            df_a_translated = translator.translate_dataset_a(df_a, use_existing=False)
    else:
        logger.info("Translating Dataset A")
        df_a_translated = translator.translate_dataset_a(df_a, use_existing=False)
    
    # Translate Dataset B
    if translated_files['dataset_b_translated']:
        logger.info("Using existing translation for Dataset B")
        df_b_translated = loader.load_translated_dataset_b()
        if df_b_translated is None or 'review_en' not in df_b_translated.columns:
            logger.info("Re-translating Dataset B")
            df_b_translated = translator.translate_dataset_b(df_b, use_existing=False)
    else:
        logger.info("Translating Dataset B")
        df_b_translated = translator.translate_dataset_b(df_b, use_existing=False)
    
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
    logger.info(f"  Positive: {sentiment_stats['positive_percentage']:.2f}%")
    logger.info(f"  Negative: {sentiment_stats['negative_percentage']:.2f}%")
    
    # Save sentiment analysis results
    df_a_sentiment.to_csv(RESULTS_DIR / "dataset_a_sentiment.csv", index=False)
    
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
    
    logger.info(f"Top 5 Bigrams: {ngram_results['top_bigrams'][:5]}")
    logger.info(f"Top 5 Trigrams: {ngram_results['top_trigrams'][:5]}")
    
    # Network analysis
    logger.info("Building co-occurrence network")
    network_analyzer = network.CoOccurrenceNetwork()
    G = network_analyzer.build_graph(chinese_tokens, top_n_nodes=100)
    
    network_stats = network_analyzer.get_network_statistics(G)
    logger.info(f"Network Statistics: {network_stats}")
    
    # Visualize network
    logger.info("Visualizing network")
    fig = network_analyzer.visualize(G, save_path=PLOTS_DIR / "cooccurrence_network.html")
    
    # ============================================================================
    # Step 6: Extract features (TF-IDF and transformer tokens)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Feature Extraction")
    logger.info("=" * 80)
    
    # Prepare data for classification (using category as target)
    # Note: The paper uses category prediction, but we'll use sentiment labels
    # as specified in the task description
    
    # Split Dataset A for training
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
    tfidf_extractor = tfidf.TFIDFExtractor()
    X_train_tfidf = tfidf_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_extractor.transform(X_test)
    
    # Transformer tokenization (for BERT/ELECTRA)
    logger.info("Tokenizing for transformer models")
    bert_tokenizer = trans_features.BERTTokenizer()
    electra_tokenizer = trans_features.ELECTRATokenizer()
    
    # ============================================================================
    # Step 7: Train all models (50 epochs)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Training Models")
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
    logger.info("Training BiLSTM model")
    # For BiLSTM, we need to create vocabulary and tokenize
    # This is simplified - in practice, you'd build a proper vocabulary
    from collections import Counter
    all_tokens = []
    for text in X_train:
        all_tokens.extend(text.split())
    vocab = Counter(all_tokens)
    vocab_size = len(vocab) + 1  # +1 for padding
    
    # Create token indices (simplified)
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
        epochs=50,
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
    logger.info("Training BERT model")
    bert_trainer = trans_models.BERTTrainer()
    bert_history = bert_trainer.train(X_train, y_train, epochs=50)
    
    predictions_bert = bert_trainer.predict(X_test)
    result_bert = metrics.evaluate_model(
        y_test,
        predictions_bert,
        training_time=bert_trainer.training_time,
        model_name="BERT",
    )
    all_results.append(result_bert)
    
    # ELECTRA
    logger.info("Training ELECTRA model")
    electra_trainer = trans_models.ELECTRATrainer()
    electra_history = electra_trainer.train(X_train, y_train, epochs=50)
    
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
    
    # Save all metrics
    for result in all_results:
        model_name = result['model_name']
        safe_name = model_name.lower().replace(' ', '_')
        metrics.save_metrics(result, METRICS_DIR / f"{safe_name}_metrics.json")
    
    # Create comparison
    comparison_df = metrics.compare_models(all_results, METRICS_DIR / "model_comparison.csv")
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())
    
    # Create visualizations
    visualization.create_comprehensive_report(all_results, PLOTS_DIR)
    
    # ============================================================================
    # Step 9: Cross-dataset validation
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 9: Cross-Dataset Validation")
    logger.info("=" * 80)
    
    # Prepare Dataset B for cross-dataset validation
    X_test_b = df_b_translated['review_en_processed'].tolist()
    y_test_b = df_b_translated['label'].values
    
    # Sample for validation if needed
    if len(X_test_b) > 10000:
        indices = np.random.choice(len(X_test_b), 10000, replace=False)
        X_test_b = [X_test_b[i] for i in indices]
        y_test_b = y_test_b[indices]
    
    cross_val_results = cross_dataset.perform_cross_dataset_validation(
        electra_trainer,
        bert_trainer,
        X_test_b,
        y_test_b,
        df_a,
        df_a_translated,
        save_dir=METRICS_DIR,
    )
    
    logger.info("Cross-dataset validation completed")
    
    # ============================================================================
    # Step 10: Generate comprehensive reports
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 10: Generating Reports")
    logger.info("=" * 80)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("\nModel Performance (Dataset A Test Set):")
    for result in all_results:
        m = result['metrics']
        logger.info(
            f"{result['model_name']:20s} - "
            f"Accuracy: {m['accuracy']:.4f}, "
            f"F1: {m['f1_score']:.4f}, "
            f"Time: {m['training_time']:.2f}s"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
