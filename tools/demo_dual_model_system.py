#!/usr/bin/env python3
"""
Dual Model System Demonstration

This script demonstrates the dual-model approach by:
1. Training raw LSTM models for a few top stocks
2. Generating predictions from both traditional and raw LSTM models
3. Saving both to daily_suggested_stocks table
4. Showing side-by-side comparison

Usage:
    python tools/demo_dual_model_system.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime
from sqlalchemy import text

from src.models.database import get_database_manager
from src.services.ml.raw_ohlcv_lstm import RawOHLCVLSTM
from src.services.ml.data_service import get_raw_ohlcv_data
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_top_stocks_with_data(limit=5):
    """Get top stocks that have sufficient historical data."""
    logger.info(f"Fetching top {limit} stocks with sufficient data...")

    db = get_database_manager()

    with db.get_session() as session:
        query = text("""
            SELECT s.symbol, s.name as stock_name, s.market_cap, COUNT(h.date) as data_points
            FROM stocks s
            JOIN historical_data h ON s.symbol = h.symbol
            WHERE s.market_cap > 10000
              AND s.symbol LIKE 'NSE:%'
            GROUP BY s.symbol, s.name, s.market_cap
            HAVING COUNT(h.date) >= 200
            ORDER BY s.market_cap DESC
            LIMIT :limit
        """)

        result = session.execute(query, {'limit': limit})
        stocks = [dict(row._mapping) for row in result]

    logger.info(f"Found {len(stocks)} stocks with sufficient data")
    for stock in stocks:
        logger.info(f"  {stock['symbol']}: {stock['data_points']} days, MCap: ₹{stock['market_cap']:.0f}Cr")

    return [s['symbol'] for s in stocks]


def train_raw_lstm_models(symbols):
    """Train raw LSTM models for the given symbols."""
    logger.info("="*70)
    logger.info("TRAINING RAW LSTM MODELS")
    logger.info("="*70)

    trained_models = []
    model_dir = Path('ml_models/raw_ohlcv_lstm')
    model_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        logger.info(f"\n📊 Training {symbol}...")

        try:
            # Get data
            df = get_raw_ohlcv_data(symbol, period='3y', user_id=1)

            if len(df) < 200:
                logger.warning(f"  ⚠️  Insufficient data: {len(df)} days (need 200+)")
                continue

            logger.info(f"  ✓ Data loaded: {len(df)} days")

            # Determine window length based on available data
            window_length = 100 if len(df) >= 300 else 50

            # Create and train model
            model = RawOHLCVLSTM(
                hidden_size=8,
                window_length=window_length,
                use_full_ohlcv=True
            )

            logger.info(f"  Training with window_length={window_length}...")
            metrics = model.train(df, epochs=30, verbose=0)

            # Save model
            model_path = model_dir / symbol
            model.save(str(model_path))

            logger.info(f"  ✓ Training complete!")
            logger.info(f"    Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"    F1 Score: {metrics['f1_macro']:.4f}")
            logger.info(f"    Model saved: {model_path}")

            trained_models.append({
                'symbol': symbol,
                'metrics': metrics,
                'model_path': str(model_path)
            })

        except Exception as e:
            logger.error(f"  ✗ Failed to train {symbol}: {e}")

    logger.info(f"\n✓ Trained {len(trained_models)} models successfully")
    return trained_models


def generate_raw_lstm_predictions(symbols):
    """Generate predictions from raw LSTM models."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING RAW LSTM PREDICTIONS")
    logger.info("="*70)

    service = get_raw_lstm_prediction_service()

    # Batch predict
    predictions = service.batch_predict(symbols, user_id=1)

    if not predictions:
        logger.warning("No predictions generated!")
        return []

    # Sort by prediction score
    predictions.sort(key=lambda x: x['ml_prediction_score'], reverse=True)

    # Save to database
    logger.info(f"\nSaving {len(predictions)} raw LSTM predictions to database...")
    service.save_to_suggested_stocks(predictions, strategy='raw_lstm')

    logger.info("\n📈 Top Raw LSTM Predictions:")
    for i, pred in enumerate(predictions[:5], 1):
        logger.info(f"\n{i}. {pred['symbol']} - {pred['stock_name']}")
        logger.info(f"   Score: {pred['ml_prediction_score']:.4f} | Confidence: {pred['ml_confidence']:.4f}")
        logger.info(f"   Target: ₹{pred['ml_price_target']:.2f} | Recommendation: {pred['recommendation']}")
        logger.info(f"   Probabilities: Loss={pred['class_probabilities']['loss']:.2%}, "
                   f"Neutral={pred['class_probabilities']['neutral']:.2%}, "
                   f"Profit={pred['class_probabilities']['profit']:.2%}")

    return predictions


def show_model_comparison():
    """Show side-by-side comparison of both models."""
    logger.info("\n" + "="*70)
    logger.info("DUAL MODEL COMPARISON")
    logger.info("="*70)

    db = get_database_manager()
    today = datetime.now().date()

    with db.get_session() as session:
        # Query for comparison
        query = text("""
            SELECT
                t.symbol,
                COALESCE(t.stock_name, s.name) as stock_name,
                t.current_price,
                t.ml_prediction_score as traditional_score,
                r.ml_prediction_score as raw_lstm_score,
                t.ml_confidence as traditional_confidence,
                r.ml_confidence as raw_lstm_confidence,
                t.recommendation as traditional_rec,
                r.recommendation as raw_lstm_rec,
                t.ml_price_target as traditional_target,
                r.ml_price_target as raw_lstm_target,
                ABS(t.ml_prediction_score - r.ml_prediction_score) as score_diff
            FROM daily_suggested_stocks t
            LEFT JOIN daily_suggested_stocks r
                ON t.symbol = r.symbol
                AND t.date = r.date
                AND r.model_type = 'raw_lstm'
            LEFT JOIN stocks s ON t.symbol = s.symbol
            WHERE t.date = :date
              AND t.model_type = 'traditional'
              AND r.symbol IS NOT NULL  -- Only show stocks with both predictions
            ORDER BY (t.ml_prediction_score + r.ml_prediction_score) DESC
            LIMIT 10
        """)

        result = session.execute(query, {'date': today})
        comparisons = [dict(row._mapping) for row in result]

    if not comparisons:
        logger.warning("No overlapping predictions found between models!")
        logger.info("This is expected if you haven't run traditional model predictions today.")

        # Show just raw LSTM predictions
        with db.get_session() as session:
            query = text("""
                SELECT
                    d.symbol, COALESCE(d.stock_name, s.name) as stock_name, d.current_price,
                    d.ml_prediction_score, d.ml_confidence,
                    d.recommendation, d.ml_price_target,
                    d.model_type
                FROM daily_suggested_stocks d
                LEFT JOIN stocks s ON d.symbol = s.symbol
                WHERE d.date = :date
                  AND d.model_type = 'raw_lstm'
                ORDER BY d.ml_prediction_score DESC
                LIMIT 5
            """)

            result = session.execute(query, {'date': today})
            raw_predictions = [dict(row._mapping) for row in result]

        if raw_predictions:
            logger.info("\n📊 Raw LSTM Predictions (Traditional model not run today):")
            for i, pred in enumerate(raw_predictions, 1):
                logger.info(f"\n{i}. {pred['symbol']} - {pred['stock_name']}")
                logger.info(f"   Price: ₹{pred['current_price']:.2f} | Target: ₹{pred['ml_price_target']:.2f}")
                logger.info(f"   Score: {pred['ml_prediction_score']:.4f} | Confidence: {pred['ml_confidence']:.4f}")
                logger.info(f"   Recommendation: {pred['recommendation']}")

        return

    logger.info(f"\n📊 Found {len(comparisons)} stocks with predictions from BOTH models:\n")

    for i, comp in enumerate(comparisons, 1):
        logger.info(f"{i}. {comp['symbol']} - {comp['stock_name']}")
        logger.info(f"   Current Price: ₹{comp['current_price']:.2f}")
        logger.info(f"   ")
        logger.info(f"   Traditional Model:")
        logger.info(f"     Score: {comp['traditional_score']:.4f} | Confidence: {comp['traditional_confidence']:.4f}")
        logger.info(f"     Target: ₹{comp['traditional_target']:.2f} | Rec: {comp['traditional_rec']}")
        logger.info(f"   ")
        logger.info(f"   Raw LSTM Model:")
        logger.info(f"     Score: {comp['raw_lstm_score']:.4f} | Confidence: {comp['raw_lstm_confidence']:.4f}")
        logger.info(f"     Target: ₹{comp['raw_lstm_target']:.2f} | Rec: {comp['raw_lstm_rec']}")
        logger.info(f"   ")

        # Agreement analysis
        if comp['traditional_rec'] == comp['raw_lstm_rec']:
            logger.info(f"   ✅ MODELS AGREE on {comp['traditional_rec']}")
        else:
            logger.info(f"   ⚠️  Models disagree: Traditional={comp['traditional_rec']}, Raw LSTM={comp['raw_lstm_rec']}")

        logger.info(f"   Score Difference: {comp['score_diff']:.4f}")
        logger.info("")


def show_database_stats():
    """Show statistics about the dual model system."""
    logger.info("\n" + "="*70)
    logger.info("DATABASE STATISTICS")
    logger.info("="*70)

    db = get_database_manager()
    today = datetime.now().date()

    with db.get_session() as session:
        # Count by model type
        query = text("""
            SELECT
                model_type,
                COUNT(*) as total,
                COUNT(DISTINCT symbol) as unique_stocks,
                AVG(ml_prediction_score) as avg_score,
                AVG(ml_confidence) as avg_confidence
            FROM daily_suggested_stocks
            WHERE date = :date
            GROUP BY model_type
        """)

        result = session.execute(query, {'date': today})
        stats = [dict(row._mapping) for row in result]

    logger.info(f"\nPredictions for {today}:\n")
    for stat in stats:
        logger.info(f"{stat['model_type'].upper()} Model:")
        logger.info(f"  Total predictions: {stat['total']}")
        logger.info(f"  Unique stocks: {stat['unique_stocks']}")
        logger.info(f"  Average score: {stat['avg_score']:.4f}")
        logger.info(f"  Average confidence: {stat['avg_confidence']:.4f}")
        logger.info("")


def main():
    """Main demonstration workflow."""
    logger.info("="*70)
    logger.info("DUAL MODEL SYSTEM DEMONSTRATION")
    logger.info("="*70)
    logger.info("")
    logger.info("This demo will:")
    logger.info("  1. Find top stocks with sufficient data")
    logger.info("  2. Train raw LSTM models for these stocks")
    logger.info("  3. Generate predictions from raw LSTM")
    logger.info("  4. Show side-by-side comparison with traditional model")
    logger.info("  5. Display database statistics")
    logger.info("")

    try:
        # Step 1: Get top stocks
        symbols = get_top_stocks_with_data(limit=5)

        if not symbols:
            logger.error("No stocks found with sufficient data!")
            return 1

        # Step 2: Train raw LSTM models
        trained = train_raw_lstm_models(symbols)

        if not trained:
            logger.error("No models trained successfully!")
            return 1

        # Step 3: Generate predictions
        trained_symbols = [m['symbol'] for m in trained]
        predictions = generate_raw_lstm_predictions(trained_symbols)

        if not predictions:
            logger.error("No predictions generated!")
            return 1

        # Step 4: Show comparison
        show_model_comparison()

        # Step 5: Show stats
        show_database_stats()

        logger.info("\n" + "="*70)
        logger.info("✓ DEMONSTRATION COMPLETE!")
        logger.info("="*70)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Check daily_suggested_stocks table to see both model types")
        logger.info("  2. Update scheduler.py to train both models daily")
        logger.info("  3. Create comparison UI to show both predictions")
        logger.info("  4. Monitor performance of both models over time")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
