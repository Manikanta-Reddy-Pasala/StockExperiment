#!/usr/bin/env python3
"""
Quick Dual Model Prediction Demo

Uses existing trained models to demonstrate dual-model system.
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
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trained_models():
    """Get list of symbols with trained raw LSTM models."""
    model_dir = Path('ml_models/raw_ohlcv_lstm')
    if not model_dir.exists():
        return []

    models = [d.name for d in model_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(models)} trained raw LSTM models: {models}")
    return models


def generate_raw_lstm_predictions():
    """Generate predictions using existing models."""
    logger.info("="*70)
    logger.info("GENERATING RAW LSTM PREDICTIONS")
    logger.info("="*70)

    symbols = get_trained_models()

    if not symbols:
        logger.error("No trained models found! Train models first using:")
        logger.error("  python tools/train_raw_ohlcv_lstm.py --symbol NSE:ADANIPOWER-EQ")
        return []

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

    logger.info("\n📈 Raw LSTM Predictions:")
    for i, pred in enumerate(predictions, 1):
        logger.info(f"\n{i}. {pred['symbol']} - {pred['stock_name']}")
        logger.info(f"   Score: {pred['ml_prediction_score']:.4f} | Confidence: {pred['ml_confidence']:.4f}")
        logger.info(f"   Target: ₹{pred['ml_price_target']:.2f} | Recommendation: {pred['recommendation']}")
        logger.info(f"   Probabilities: Loss={pred['class_probabilities']['loss']:.2%}, "
                   f"Neutral={pred['class_probabilities']['neutral']:.2%}, "
                   f"Profit={pred['class_probabilities']['profit']:.2%}")

    return predictions


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
                AVG(ml_confidence) as avg_confidence,
                COUNT(CASE WHEN recommendation = 'BUY' THEN 1 END) as buy_count,
                COUNT(CASE WHEN recommendation = 'HOLD' THEN 1 END) as hold_count,
                COUNT(CASE WHEN recommendation = 'SELL' THEN 1 END) as sell_count
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
        logger.info(f"  Recommendations: BUY={stat['buy_count']}, HOLD={stat['hold_count']}, SELL={stat['sell_count']}")
        logger.info("")


def show_raw_lstm_vs_traditional():
    """Show comparison between models (if traditional exists)."""
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON (Raw LSTM vs Traditional)")
    logger.info("="*70)

    db = get_database_manager()
    today = datetime.now().date()

    with db.get_session() as session:
        # Check if traditional model has run today
        query = text("""
            SELECT COUNT(*) as count
            FROM daily_suggested_stocks
            WHERE date = :date
              AND model_type = 'traditional'
        """)

        result = session.execute(query, {'date': today}).fetchone()

        if result.count == 0:
            logger.info("\nTraditional model hasn't run today.")
            logger.info("Only showing Raw LSTM predictions for now.")
            logger.info("\nTo see side-by-side comparison:")
            logger.info("  1. Run your traditional prediction pipeline")
            logger.info("  2. Re-run this demo to see both models compared")
            return

        # Show comparison
        query = text("""
            SELECT
                t.symbol,
                COALESCE(t.stock_name, s.name) as stock_name,
                t.current_price,
                t.ml_prediction_score as traditional_score,
                r.ml_prediction_score as raw_lstm_score,
                t.recommendation as traditional_rec,
                r.recommendation as raw_lstm_rec,
                ABS(t.ml_prediction_score - r.ml_prediction_score) as score_diff
            FROM daily_suggested_stocks t
            INNER JOIN daily_suggested_stocks r
                ON t.symbol = r.symbol
                AND t.date = r.date
                AND r.model_type = 'raw_lstm'
            LEFT JOIN stocks s ON t.symbol = s.symbol
            WHERE t.date = :date
              AND t.model_type = 'traditional'
            ORDER BY (t.ml_prediction_score + r.ml_prediction_score) DESC
            LIMIT 10
        """)

        result = session.execute(query, {'date': today})
        comparisons = [dict(row._mapping) for row in result]

    if not comparisons:
        logger.info("\nNo overlapping predictions between models.")
        return

    logger.info(f"\nFound {len(comparisons)} stocks with predictions from BOTH models:\n")

    for i, comp in enumerate(comparisons, 1):
        logger.info(f"{i}. {comp['symbol']} - {comp['stock_name']}")
        logger.info(f"   Price: ₹{comp['current_price']:.2f}")
        logger.info(f"   Traditional: Score={comp['traditional_score']:.4f}, Rec={comp['traditional_rec']}")
        logger.info(f"   Raw LSTM:    Score={comp['raw_lstm_score']:.4f}, Rec={comp['raw_lstm_rec']}")

        if comp['traditional_rec'] == comp['raw_lstm_rec']:
            logger.info(f"   ✅ AGREE on {comp['traditional_rec']}")
        else:
            logger.info(f"   ⚠️  DISAGREE: Traditional={comp['traditional_rec']}, LSTM={comp['raw_lstm_rec']}")

        logger.info("")


def main():
    """Main demo workflow."""
    logger.info("="*70)
    logger.info("DUAL MODEL PREDICTION DEMONSTRATION")
    logger.info("="*70)
    logger.info("")

    try:
        # Generate predictions from existing models
        predictions = generate_raw_lstm_predictions()

        if not predictions:
            logger.error("\nNo predictions generated. Make sure you have trained models.")
            return 1

        # Show database stats
        show_database_stats()

        # Show comparison
        show_raw_lstm_vs_traditional()

        logger.info("\n" + "="*70)
        logger.info("✓ DEMONSTRATION COMPLETE!")
        logger.info("="*70)
        logger.info("")
        logger.info("What's been done:")
        logger.info("  ✓ Database has 'model_type' column for dual models")
        logger.info("  ✓ Raw LSTM predictions saved to daily_suggested_stocks")
        logger.info("  ✓ Both models can now coexist in the same table")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Train more raw LSTM models: python tools/train_raw_ohlcv_lstm.py --symbol NSE:XXX-EQ")
        logger.info("  2. Update scheduler to train both models daily")
        logger.info("  3. Create UI to show side-by-side comparison")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
