#!/usr/bin/env python3
"""
Generate Raw LSTM Predictions for ALL Risk Strategies
Generates both default_risk and high_risk predictions for Raw LSTM model
to complete the 4th variant of the dual model view.
"""

import sys
import logging
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trained_models():
    """Get list of symbols that have trained Raw LSTM models."""
    model_dir = Path('ml_models/raw_ohlcv_lstm')

    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return []

    # Find all symbol directories
    symbols = []
    for symbol_dir in model_dir.iterdir():
        if symbol_dir.is_dir():
            # Check if model file exists (try both filenames)
            model_file = symbol_dir / 'lstm_model.h5'
            if not model_file.exists():
                model_file = symbol_dir / 'model.h5'
            if model_file.exists():
                symbols.append(symbol_dir.name)

    logger.info(f"Found {len(symbols)} trained Raw LSTM models")
    return symbols


def generate_raw_lstm_predictions_for_strategy(symbols, strategy='default_risk'):
    """Generate Raw LSTM predictions for a specific risk strategy."""
    logger.info("\n" + "="*100)
    logger.info(f"GENERATING RAW LSTM PREDICTIONS - {strategy.upper()}")
    logger.info("="*100)

    try:
        service = get_raw_lstm_prediction_service()

        logger.info(f"\n🎯 Generating predictions for {len(symbols)} stocks...")
        logger.info(f"   Strategy: {strategy}")
        logger.info(f"   Model: Raw LSTM (OHLCV-based)")
        logger.info("")

        # Batch predict with strategy
        predictions = service.batch_predict(symbols, user_id=1, strategy=strategy)

        if not predictions:
            logger.warning("No predictions generated!")
            return []

        # Sort by prediction score
        predictions.sort(key=lambda x: x['ml_prediction_score'], reverse=True)

        # Save to database
        logger.info(f"\nSaving {len(predictions)} predictions to database...")
        service.save_to_suggested_stocks(predictions, strategy=strategy)

        logger.info("\n" + "="*100)
        logger.info(f"✅ {strategy.upper()} PREDICTIONS GENERATED SUCCESSFULLY!")
        logger.info("="*100)
        logger.info(f"  Stocks predicted: {len(predictions)}")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Model: raw_lstm")
        logger.info("")

        # Show top 5 predictions
        logger.info("  Top 5 Predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            logger.info(f"\n  {i}. {pred['symbol']} - {pred['stock_name']}")
            logger.info(f"     Price: ₹{pred['current_price']:.2f} → Target: ₹{pred['ml_price_target']:.2f}")
            logger.info(f"     Score: {pred['ml_prediction_score']:.4f} | Confidence: {pred['ml_confidence']:.4f}")
            logger.info(f"     Recommendation: {pred['recommendation']}")
            if pred.get('stop_loss'):
                logger.info(f"     Stop Loss: ₹{pred['stop_loss']:.2f}")

        logger.info("\n" + "="*100)

        return predictions

    except Exception as e:
        logger.error(f"❌ Failed to generate {strategy} predictions: {e}", exc_info=True)
        return []


def check_current_data():
    """Check what data we currently have in the database."""
    logger.info("\n" + "="*100)
    logger.info("CURRENT DATABASE STATUS (Before Generation)")
    logger.info("="*100)

    try:
        db = get_database_manager()

        with db.get_session() as session:
            query = text("""
                SELECT
                    model_type,
                    strategy,
                    COUNT(*) as count,
                    ROUND(AVG(ml_prediction_score)::numeric, 4) as avg_score
                FROM daily_suggested_stocks
                WHERE date = CURRENT_DATE
                GROUP BY model_type, strategy
                ORDER BY model_type, strategy
            """)

            result = session.execute(query)
            rows = result.fetchall()

            if not rows:
                logger.info("  ❌ No predictions for today yet")
            else:
                logger.info(f"\n  {'Model':<15} {'Strategy':<20} {'Count':<8} {'Avg Score'}")
                logger.info("  " + "-" * 60)
                for row in rows:
                    model = row[0]
                    strategy = row[1]
                    count = row[2]
                    score = row[3] if row[3] else 0
                    logger.info(f"  {model:<15} {strategy:<20} {count:<8} {score}")

            logger.info("\n" + "="*100)

    except Exception as e:
        logger.error(f"Failed to check current data: {e}")


def verify_all_combinations():
    """Verify all 4 model/risk combinations are present."""
    logger.info("\n" + "="*100)
    logger.info("FINAL VERIFICATION - ALL 4 COMBINATIONS")
    logger.info("="*100)

    try:
        db = get_database_manager()

        with db.get_session() as session:
            query = text("""
                SELECT
                    model_type,
                    strategy,
                    COUNT(*) as count,
                    ROUND(AVG(ml_prediction_score)::numeric, 4) as avg_score,
                    STRING_AGG(DISTINCT symbol, ', ' ORDER BY symbol) as sample_symbols
                FROM daily_suggested_stocks
                WHERE date = CURRENT_DATE
                GROUP BY model_type, strategy
                ORDER BY model_type, strategy
            """)

            result = session.execute(query)
            rows = result.fetchall()

            logger.info(f"\n  {'Model':<15} {'Strategy':<20} {'Count':<8} {'Avg Score':<12} Sample Symbols")
            logger.info("  " + "-" * 120)

            for row in rows:
                model = row[0]
                strategy = row[1]
                count = row[2]
                score = row[3] if row[3] else 0
                symbols = row[4][:50] + "..." if row[4] and len(row[4]) > 50 else row[4]
                logger.info(f"  {model:<15} {strategy:<20} {count:<8} {score:<12} {symbols}")

            # Check for all 4 combinations
            logger.info("\n  " + "="*80)
            logger.info("  DUAL MODEL VIEW COMBINATIONS:")
            logger.info("  " + "="*80)

            expected = [
                ('traditional', 'default_risk', 'Traditional + Default Risk'),
                ('traditional', 'high_risk', 'Traditional + High Risk'),
                ('raw_lstm', 'default_risk', 'Raw LSTM + Default Risk'),
                ('raw_lstm', 'high_risk', 'Raw LSTM + High Risk')
            ]

            existing = {(row[0], row[1]): row[2] for row in rows}

            all_present = True
            for model, strategy, label in expected:
                count = existing.get((model, strategy), 0)
                status = "✅" if count > 0 else "❌"
                logger.info(f"  {status} {label:<35} : {count} stocks")
                if count == 0:
                    all_present = False

            logger.info("\n" + "="*100)

            if all_present:
                logger.info("🎉 SUCCESS! All 4 combinations are now available!")
                logger.info("   Your dual model view page will show complete data.")
            else:
                logger.warning("⚠️  Some combinations are still missing")

            logger.info("="*100)

    except Exception as e:
        logger.error(f"Failed to verify data: {e}")


def main():
    """Main execution flow."""
    logger.info("\n\n")
    logger.info("█" * 100)
    logger.info("█" + " " * 98 + "█")
    logger.info("█" + " " * 15 + "RAW LSTM ALL STRATEGIES GENERATOR - Complete Dual Model View" + " " * 18 + "█")
    logger.info("█" + " " * 98 + "█")
    logger.info("█" * 100)
    logger.info("")

    # Step 1: Check current data
    check_current_data()

    # Step 2: Get trained models
    logger.info("\n" + "="*100)
    logger.info("FINDING TRAINED RAW LSTM MODELS")
    logger.info("="*100)

    symbols = get_trained_models()

    if not symbols:
        logger.error("❌ No trained Raw LSTM models found!")
        logger.info("\nYou need to train Raw LSTM models first:")
        logger.info("  python3 tools/demo_dual_model_system.py")
        return 1

    logger.info(f"\nFound trained models for {len(symbols)} symbols:")
    for symbol in symbols:
        logger.info(f"  • {symbol}")

    # Step 3: Generate default_risk predictions
    default_predictions = generate_raw_lstm_predictions_for_strategy(symbols, 'default_risk')

    # Step 4: Generate high_risk predictions
    high_predictions = generate_raw_lstm_predictions_for_strategy(symbols, 'high_risk')

    # Step 5: Verify all combinations
    verify_all_combinations()

    # Summary
    logger.info("\n" + "="*100)
    logger.info("📊 SUMMARY")
    logger.info("="*100)
    logger.info(f"  Default Risk Predictions: {len(default_predictions)} stocks")
    logger.info(f"  High Risk Predictions:    {len(high_predictions)} stocks")
    logger.info(f"  Total Predictions:        {len(default_predictions) + len(high_predictions)}")
    logger.info("")

    if default_predictions and high_predictions:
        logger.info("✅ Raw LSTM predictions generated for BOTH strategies!")
        logger.info("   Combined with Traditional model, you now have ALL 4 combinations:")
        logger.info("")
        logger.info("   1. ✅ Traditional + default_risk")
        logger.info("   2. ✅ Traditional + high_risk")
        logger.info("   3. ✅ Raw LSTM + default_risk")
        logger.info("   4. ✅ Raw LSTM + high_risk")
        logger.info("")
        logger.info("   Navigate to Suggested Stocks page to see the complete dual model view!")
        logger.info("="*100)
        return 0
    else:
        logger.error("❌ Some predictions failed. Check logs above.")
        logger.info("="*100)
        return 1


if __name__ == '__main__':
    exit(main())
