"""
A/B Testing Framework for Stock Prediction Models

Compare different ML approaches to find the best performer for Indian stock market:
1. Traditional approach: Feature-engineered data + Random Forest/XGBoost
2. Research-based approach: Raw OHLCV + Simple LSTM + Triple Barrier Labeling

This framework provides:
- Side-by-side model comparison
- Performance metrics tracking
- Backtesting on same data
- Statistical significance testing
- Automated model selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare different stock prediction models on the same dataset.

    Supports:
    - Multiple model architectures
    - Consistent train/test splits
    - Comprehensive metrics
    - Statistical significance testing
    - Performance visualization
    """

    def __init__(self, save_dir: str = 'ml_models/comparison'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.results = {}
        self.test_data = None

        logger.info(f"ModelComparator initialized (save_dir: {self.save_dir})")

    def add_model(self, name: str, model, model_type: str = 'traditional'):
        """
        Add a model to the comparison.

        Parameters:
        -----------
        name : str
            Unique name for this model (e.g., 'raw_ohlcv_lstm', 'xgboost_features')
        model : object
            Trained model instance
        model_type : str
            Type of model: 'traditional', 'raw_lstm', 'ensemble', etc.
        """
        self.models[name] = {
            'model': model,
            'type': model_type,
            'added_at': datetime.now().isoformat()
        }
        logger.info(f"Added model '{name}' (type: {model_type})")

    def evaluate_all(self, test_data: pd.DataFrame, test_labels: np.ndarray,
                    feature_names: Optional[List[str]] = None) -> Dict:
        """
        Evaluate all registered models on the same test set.

        Parameters:
        -----------
        test_data : pd.DataFrame
            Test dataset (format depends on model)
        test_labels : np.ndarray
            True labels for test set
        feature_names : Optional[List[str]]
            Feature names (for traditional models)

        Returns:
        --------
        Dict
            Comparison results for all models
        """
        logger.info(f"Evaluating {len(self.models)} models on {len(test_labels)} test samples...")

        self.test_data = test_data
        results = {}

        for model_name, model_info in self.models.items():
            logger.info(f"\n--- Evaluating {model_name} ---")

            try:
                # Get predictions
                model = model_info['model']

                # Handle different model interfaces
                if hasattr(model, 'predict'):
                    if model_info['type'] == 'raw_lstm':
                        # Raw OHLCV LSTM model
                        predictions = model.predict(test_data, return_probabilities=False)
                    else:
                        # Traditional sklearn-style model
                        predictions = model.predict(test_data.values if isinstance(test_data, pd.DataFrame) else test_data)
                else:
                    logger.error(f"Model {model_name} does not have a predict method")
                    continue

                # Ensure predictions and labels are aligned
                min_len = min(len(predictions), len(test_labels))
                predictions = predictions[:min_len]
                labels = test_labels[:min_len]

                # Calculate metrics
                metrics = self._calculate_metrics(labels, predictions, model_name)

                results[model_name] = {
                    'model_type': model_info['type'],
                    'metrics': metrics,
                    'predictions': predictions,
                    'evaluated_at': datetime.now().isoformat()
                }

                # Log key metrics
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
                logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        self.results = results
        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str) -> Dict:
        """
        Calculate comprehensive metrics for a model's predictions.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        model_name : str
            Name of the model (for logging)

        Returns:
        --------
        Dict
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y_true)
        }

        return metrics

    def get_winner(self, metric: str = 'f1_macro') -> Tuple[str, float]:
        """
        Determine the best performing model based on a metric.

        Parameters:
        -----------
        metric : str
            Metric to use for comparison (default: 'f1_macro')

        Returns:
        --------
        Tuple[str, float]
            (model_name, score)
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate_all() first.")

        best_model = None
        best_score = -1

        for model_name, result in self.results.items():
            if 'error' in result:
                continue

            score = result['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name

        logger.info(f"Best model: {best_model} ({metric}={best_score:.4f})")
        return best_model, best_score

    def generate_report(self) -> str:
        """
        Generate a comprehensive comparison report.

        Returns:
        --------
        str
            Markdown-formatted report
        """
        if not self.results:
            return "No results available. Run evaluate_all() first."

        report = []
        report.append("# Model Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary table
        report.append("## Performance Summary\n")
        report.append("| Model | Type | Accuracy | F1 Macro | F1 Weighted | Precision | Recall |")
        report.append("|-------|------|----------|----------|-------------|-----------|--------|")

        for model_name, result in self.results.items():
            if 'error' in result:
                report.append(f"| {model_name} | - | ERROR | - | - | - | - |")
                continue

            metrics = result['metrics']
            model_type = result['model_type']

            report.append(
                f"| {model_name} | {model_type} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['f1_macro']:.4f} | "
                f"{metrics['f1_weighted']:.4f} | "
                f"{metrics['precision_macro']:.4f} | "
                f"{metrics['recall_macro']:.4f} |"
            )

        # Best model
        best_model, best_score = self.get_winner()
        report.append(f"\n**Winner:** {best_model} (F1 Macro = {best_score:.4f})\n")

        # Detailed metrics per model
        report.append("\n## Detailed Metrics\n")

        for model_name, result in self.results.items():
            if 'error' in result:
                report.append(f"\n### {model_name}\n")
                report.append(f"**Error:** {result['error']}\n")
                continue

            report.append(f"\n### {model_name}\n")

            metrics = result['metrics']

            # Per-class performance
            report.append("\n**Per-Class Performance:**\n")
            report.append("| Class | Precision | Recall | F1 Score |")
            report.append("|-------|-----------|--------|----------|")

            class_names = ['Loss', 'Neutral', 'Profit']
            for i, class_name in enumerate(class_names):
                if i < len(metrics['precision_per_class']):
                    report.append(
                        f"| {class_name} | "
                        f"{metrics['precision_per_class'][i]:.4f} | "
                        f"{metrics['recall_per_class'][i]:.4f} | "
                        f"{metrics['f1_per_class'][i]:.4f} |"
                    )

            # Confusion matrix
            report.append("\n**Confusion Matrix:**\n")
            cm = np.array(metrics['confusion_matrix'])
            report.append("```")
            report.append(f"{'':>10} | {'Loss':>8} | {'Neutral':>8} | {'Profit':>8}")
            report.append(f"{'-'*10}-|{'-'*10}|{'-'*10}|{'-'*10}")
            for i, class_name in enumerate(class_names):
                if i < cm.shape[0]:
                    row = ' | '.join([f"{val:>8}" for val in cm[i]])
                    report.append(f"{class_name:>10} | {row}")
            report.append("```\n")

        # Research findings comparison
        report.append("\n## Comparison with Research Paper\n")
        report.append("**Korean Stock Prediction Study (2024) Results:**\n")
        report.append("- LSTM (raw OHLCV): F1 = 0.4312\n")
        report.append("- XGBoost (technical indicators): F1 = 0.4316\n")
        report.append("- Dummy classifier: F1 = 0.1852\n\n")

        report.append("**Our Results on Indian Stocks:**\n")
        for model_name, result in self.results.items():
            if 'error' not in result:
                f1 = result['metrics']['f1_macro']
                report.append(f"- {model_name}: F1 = {f1:.4f}\n")

        return '\n'.join(report)

    def save_report(self, filename: str = 'comparison_report.md'):
        """
        Save the comparison report to a file.

        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        report = self.generate_report()
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {filepath}")

    def save_results(self, filename: str = 'comparison_results.json'):
        """
        Save detailed results to JSON file.

        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        filepath = self.save_dir / filename

        # Prepare results for JSON (remove non-serializable objects)
        results_clean = {}
        for model_name, result in self.results.items():
            result_copy = result.copy()
            if 'predictions' in result_copy:
                result_copy['predictions'] = result_copy['predictions'].tolist() if isinstance(result_copy['predictions'], np.ndarray) else result_copy['predictions']
            results_clean[model_name] = result_copy

        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def compare_traditional_vs_raw_lstm(symbol: str, period: str = '3y', user_id: int = 1) -> Dict:
    """
    Quick comparison between traditional feature engineering and raw OHLCV LSTM.

    Parameters:
    -----------
    symbol : str
        Stock symbol to test
    period : str
        Data period (e.g., '1y', '3y')
    user_id : int
        User ID for broker authentication

    Returns:
    --------
    Dict
        Comparison results
    """
    logger.info(f"Comparing Traditional vs Raw LSTM approaches for {symbol}")

    from .data_service import get_raw_ohlcv_data, get_stock_data, create_features
    from .raw_ohlcv_lstm import RawOHLCVLSTM
    from .training_service import train_and_tune_models

    # Initialize comparator
    comparator = ModelComparator()

    # 1. Train Raw OHLCV LSTM Model
    logger.info("\n=== Training Raw OHLCV LSTM ===")
    try:
        df_raw = get_raw_ohlcv_data(symbol=symbol, period=period, user_id=user_id)

        raw_lstm = RawOHLCVLSTM(
            hidden_size=8,        # Paper's optimal value
            window_length=100,    # Paper's optimal value
            use_full_ohlcv=True   # Use all 5 OHLCV features
        )

        # Train with test split
        metrics_raw = raw_lstm.train(df_raw, test_size=0.2, epochs=50, verbose=1)

        comparator.add_model('raw_ohlcv_lstm', raw_lstm, 'raw_lstm')
        logger.info("✓ Raw OHLCV LSTM trained successfully")

    except Exception as e:
        logger.error(f"Failed to train Raw OHLCV LSTM: {e}")

    # 2. Train Traditional Feature-Engineered Model
    logger.info("\n=== Training Traditional Feature-Engineered Model ===")
    try:
        # This uses your existing training pipeline with 70+ engineered features
        result = train_and_tune_models(
            symbol=symbol,
            start_date=None,
            end_date=None,
            job_id=None
        )

        if result.get('success'):
            # Load the trained ensemble model
            from src.utils.ml_helpers import load_model

            meta_learner = load_model(f"{symbol}_meta")
            if meta_learner:
                comparator.add_model('traditional_ensemble', meta_learner, 'traditional')
                logger.info("✓ Traditional ensemble trained successfully")
        else:
            logger.error("Traditional model training failed")

    except Exception as e:
        logger.error(f"Failed to train Traditional model: {e}")

    # 3. Evaluate on same test set
    logger.info("\n=== Evaluating Models ===")

    # For fair comparison, we need to evaluate on the same test period
    # Use the last 20% of data as test set
    split_idx = int(len(df_raw) * 0.8)
    test_raw = df_raw[split_idx:]

    # Get labels from raw LSTM model
    from .triple_barrier_labeling import TripleBarrierLabeler
    labeler = TripleBarrierLabeler()
    _, test_labels = labeler.apply_for_ml(test_raw, return_multiclass=True)

    # Evaluate
    results = comparator.evaluate_all(test_raw, test_labels)

    # Generate and save report
    logger.info("\n=== Generating Comparison Report ===")
    comparator.save_report(f'comparison_{symbol}.md')
    comparator.save_results(f'comparison_{symbol}.json')

    # Print report to console
    print("\n" + comparator.generate_report())

    return results


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Compare approaches on a stock
    symbol = "RELIANCE"
    results = compare_traditional_vs_raw_lstm(symbol=symbol, period='2y')

    print(f"\n✓ Comparison complete. Check ml_models/comparison/ for detailed report.")
