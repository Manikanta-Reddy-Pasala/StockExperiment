#!/usr/bin/env python3
"""
Train Raw OHLCV LSTM Model - Research-Based Approach

This script implements the methodology from the Korean stock prediction research (2024)
which showed that simple LSTMs on raw OHLCV data can match complex feature-engineered models.

Usage:
    # Train on a single stock
    python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE

    # Train with custom parameters
    python tools/train_raw_ohlcv_lstm.py --symbol TCS --hidden-size 8 --window 100 --epochs 50

    # Compare with traditional approach
    python tools/train_raw_ohlcv_lstm.py --symbol INFY --compare

    # Grid search for optimal hyperparameters
    python tools/train_raw_ohlcv_lstm.py --symbol HDFCBANK --grid-search
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.ml.data_service import get_raw_ohlcv_data
from src.services.ml.raw_ohlcv_lstm import RawOHLCVLSTM, grid_search_lstm
from src.services.ml.model_comparison import compare_traditional_vs_raw_lstm
from src.services.ml.triple_barrier_labeling import create_balanced_barriers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/raw_ohlcv_lstm_training.log')
    ]
)

logger = logging.getLogger(__name__)


def train_single_stock(args):
    """Train raw OHLCV LSTM on a single stock."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Raw OHLCV LSTM for {args.symbol}")
    logger.info(f"{'='*60}\n")

    # Fetch data
    logger.info("Fetching raw OHLCV data...")
    df = get_raw_ohlcv_data(
        symbol=args.symbol,
        period=args.period,
        user_id=args.user_id
    )

    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    # Auto-tune barriers if requested
    if args.auto_tune_barriers:
        logger.info("\nAuto-tuning triple barrier parameters...")
        upper, lower, horizon = create_balanced_barriers(df)
        logger.info(f"Optimal barriers: upper={upper}%, lower={lower}%, horizon={horizon} days")
    else:
        logger.info(f"Using default barriers: 9%, 9%, 29 days")

    # Create and train model
    logger.info("\nInitializing Raw OHLCV LSTM model...")
    model = RawOHLCVLSTM(
        hidden_size=args.hidden_size,
        window_length=args.window,
        use_full_ohlcv=args.use_full_ohlcv,
        dropout=args.dropout
    )

    logger.info("\nTraining model...")
    metrics = model.train(
        df,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        early_stopping_patience=args.patience,
        verbose=1
    )

    # Save model
    logger.info("\nSaving model...")
    model_dir = f'ml_models/raw_ohlcv_lstm/{args.symbol}'
    model.save(model_dir)

    # Print results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nModel: Raw OHLCV LSTM")
    print(f"Symbol: {args.symbol}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"\nModel saved to: {model_dir}")
    print(f"\nResearch Paper Benchmark:")
    print(f"  LSTM (Korean stocks):    F1 = 0.4312")
    print(f"  XGBoost (Korean stocks): F1 = 0.4316")
    print(f"\n{'='*60}\n")

    return metrics


def run_grid_search(args):
    """Run grid search to find optimal hyperparameters."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Grid Search for {args.symbol}")
    logger.info(f"{'='*60}\n")

    # Fetch data
    df = get_raw_ohlcv_data(
        symbol=args.symbol,
        period=args.period,
        user_id=args.user_id
    )

    # Run grid search
    logger.info("Starting grid search...")
    results = grid_search_lstm(
        df,
        hidden_sizes=[4, 8, 16, 32],
        window_lengths=[50, 100, 150],
        use_full_ohlcv_options=[True, False],
        n_trials=args.max_trials
    )

    # Print results
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest Configuration:")
    print(f"  Hidden Size:  {results['best_config']['hidden_size']}")
    print(f"  Window Length: {results['best_config']['window_length']}")
    print(f"  Full OHLCV:   {results['best_config']['use_full_ohlcv']}")
    print(f"\nBest Performance:")
    print(f"  F1 Macro:    {results['best_config']['f1_macro']:.4f}")
    print(f"  Accuracy:    {results['best_config']['accuracy']:.4f}")
    print(f"\n{'='*60}\n")

    # Save results
    results_file = f'ml_models/raw_ohlcv_lstm/grid_search_{args.symbol}.csv'
    results['all_results'].to_csv(results_file, index=False)
    logger.info(f"Full results saved to: {results_file}")

    return results


def run_comparison(args):
    """Compare traditional approach vs raw OHLCV LSTM."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing Approaches for {args.symbol}")
    logger.info(f"{'='*60}\n")

    results = compare_traditional_vs_raw_lstm(
        symbol=args.symbol,
        period=args.period,
        user_id=args.user_id
    )

    logger.info("Comparison complete! Check ml_models/comparison/ for detailed report.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train Raw OHLCV LSTM model based on Korean stock prediction research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE

  # Custom parameters (matching research paper)
  python tools/train_raw_ohlcv_lstm.py --symbol TCS --hidden-size 8 --window 100

  # Compare with traditional approach
  python tools/train_raw_ohlcv_lstm.py --symbol INFY --compare

  # Find optimal hyperparameters
  python tools/train_raw_ohlcv_lstm.py --symbol HDFCBANK --grid-search
        """
    )

    # Required arguments
    parser.add_argument('--symbol', type=str, required=True,
                       help='Stock symbol (e.g., RELIANCE, TCS, INFY)')

    # Mode selection
    parser.add_argument('--compare', action='store_true',
                       help='Compare with traditional feature-engineered approach')
    parser.add_argument('--grid-search', action='store_true',
                       help='Run grid search for optimal hyperparameters')

    # Data parameters
    parser.add_argument('--period', type=str, default='3y',
                       help='Data period (default: 3y)')
    parser.add_argument('--user-id', type=int, default=1,
                       help='User ID for broker authentication (default: 1)')

    # Model parameters (from research paper)
    parser.add_argument('--hidden-size', type=int, default=8,
                       help='LSTM hidden size (paper optimal: 8)')
    parser.add_argument('--window', type=int, default=100,
                       help='Window length in days (paper optimal: 100)')
    parser.add_argument('--use-full-ohlcv', type=bool, default=True,
                       help='Use all OHLCV features (default: True)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')

    # Triple barrier parameters
    parser.add_argument('--auto-tune-barriers', action='store_true',
                       help='Auto-tune triple barrier parameters')

    # Grid search parameters
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Max trials for grid search (default: all combinations)')

    args = parser.parse_args()

    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('ml_models/raw_ohlcv_lstm', exist_ok=True)
    os.makedirs('ml_models/comparison', exist_ok=True)

    # Run appropriate mode
    try:
        if args.grid_search:
            results = run_grid_search(args)
        elif args.compare:
            results = run_comparison(args)
        else:
            results = train_single_stock(args)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
