#!/usr/bin/env python3
"""
Generate Predictions for ALL 3 Models × 2 Strategies = 6 Combinations

This script generates a complete set of predictions for:
1. Traditional ML (RF + XGBoost) → DEFAULT_RISK + HIGH_RISK
2. Raw LSTM (Deep Learning) → default_risk + high_risk
3. Kronos (Pattern-based) → DEFAULT_RISK + HIGH_RISK

Ensures all 6 combinations are available for the triple model view.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, date
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service
from src.services.ml.kronos_prediction_service import get_kronos_prediction_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_current_status():
    """Check current predictions in database."""
    logger.info("\n" + "="*100)
    logger.info("CURRENT DATABASE STATUS")
    logger.info("="*100)

    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT
                model_type,
                strategy,
                COUNT(*) as count,
                MAX(date) as latest_date,
                ROUND(AVG(ml_prediction_score)::numeric, 3) as avg_score
            FROM daily_suggested_stocks
            GROUP BY model_type, strategy
            ORDER BY model_type, strategy
        """)

        result = session.execute(query)
        rows = result.fetchall()

        if not rows:
            logger.info("  ❌ No predictions found in database")
        else:
            logger.info(f"\n  {'Model':<15} {'Strategy':<20} {'Count':<8} {'Latest Date':<15} {'Avg Score'}")
            logger.info("  " + "-"*80)
            for row in rows:
                logger.info(f"  {row[0]:<15} {row[1]:<20} {row[2]:<8} {str(row[3]):<15} {row[4]}")

    logger.info("\n" + "="*100)


def generate_traditional_ml(strategies=['DEFAULT_RISK', 'HIGH_RISK']):
    """Generate Traditional ML predictions using saga."""
    logger.info("\n" + "="*100)
    logger.info("🤖 GENERATING TRADITIONAL ML PREDICTIONS (RF + XGBoost)")
    logger.info("="*100)

    for strategy in strategies:
        logger.info(f"\n📊 Strategy: {strategy}")
        logger.info("-"*80)

        try:
            orchestrator = SuggestedStocksSagaOrchestrator()

            result = orchestrator.execute_suggested_stocks_saga(
                user_id=1,
                strategies=[strategy],
                limit=50  # Generate top 50 per strategy
            )

            if result['status'] == 'completed':
                final_count = result['summary'].get('final_result_count', 0)
                logger.info(f"✅ {strategy}: Generated {final_count} predictions")
            else:
                logger.error(f"❌ {strategy}: Failed - {result.get('errors', [])}")

        except Exception as e:
            logger.error(f"❌ {strategy}: Exception - {e}", exc_info=True)

    logger.info("\n" + "="*100)


def generate_raw_lstm(strategies=['default_risk', 'high_risk']):
    """Generate Raw LSTM predictions."""
    logger.info("\n" + "="*100)
    logger.info("🧠 GENERATING RAW LSTM PREDICTIONS (Deep Learning)")
    logger.info("="*100)

    # Get trained models
    model_dir = Path('ml_models/raw_ohlcv_lstm')
    if not model_dir.exists():
        logger.error("❌ Raw LSTM model directory not found")
        return

    symbols = []
    for symbol_dir in model_dir.iterdir():
        if symbol_dir.is_dir():
            model_file = symbol_dir / 'lstm_model.h5'
            if not model_file.exists():
                model_file = symbol_dir / 'model.h5'
            if model_file.exists():
                symbols.append(symbol_dir.name)

    logger.info(f"Found {len(symbols)} trained Raw LSTM models")

    if not symbols:
        logger.error("❌ No trained models found. Run training first.")
        return

    service = get_raw_lstm_prediction_service()

    for strategy in strategies:
        logger.info(f"\n📊 Strategy: {strategy}")
        logger.info("-"*80)

        try:
            predictions = service.batch_predict(
                symbols=symbols,
                user_id=1,
                strategy=strategy
            )

            if predictions:
                logger.info(f"✅ {strategy}: Generated {len(predictions)} predictions")

                # Save to database
                service.save_to_suggested_stocks(
                    predictions=predictions,
                    strategy=strategy,
                    prediction_date=datetime.now().date()
                )
            else:
                logger.warning(f"⚠️  {strategy}: No predictions generated")

        except Exception as e:
            logger.error(f"❌ {strategy}: Exception - {e}", exc_info=True)

    logger.info("\n" + "="*100)


def generate_kronos(strategies=['DEFAULT_RISK', 'HIGH_RISK']):
    """Generate Kronos predictions."""
    logger.info("\n" + "="*100)
    logger.info("🔮 GENERATING KRONOS PREDICTIONS (Pattern-based)")
    logger.info("="*100)

    # Get all active symbols
    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT symbol
            FROM stocks
            WHERE is_active = true
            ORDER BY symbol
        """)
        result = session.execute(query)
        all_symbols = [row[0] for row in result]

    logger.info(f"Found {len(all_symbols)} active stocks")

    service = get_kronos_prediction_service()

    for strategy in strategies:
        logger.info(f"\n📊 Strategy: {strategy}")
        logger.info("-"*80)

        try:
            predictions = service.batch_predict(
                symbols=all_symbols,
                user_id=1,
                max_workers=1,
                strategy=strategy
            )

            if predictions:
                logger.info(f"✅ {strategy}: Generated {len(predictions)} predictions")

                # Save to database
                service.save_to_suggested_stocks(
                    predictions=predictions,
                    strategy=strategy,
                    prediction_date=datetime.now().date()
                )
            else:
                logger.warning(f"⚠️  {strategy}: No predictions generated")

        except Exception as e:
            logger.error(f"❌ {strategy}: Exception - {e}", exc_info=True)

    logger.info("\n" + "="*100)


def verify_all_combinations():
    """Verify all 6 model × strategy combinations exist."""
    logger.info("\n" + "="*100)
    logger.info("✅ FINAL VERIFICATION - ALL 6 COMBINATIONS")
    logger.info("="*100)

    today = datetime.now().date()

    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT
                model_type,
                strategy,
                COUNT(*) as count,
                MAX(date) as latest_date
            FROM daily_suggested_stocks
            WHERE date = :today
            GROUP BY model_type, strategy
            ORDER BY model_type, strategy
        """)

        result = session.execute(query, {'today': today})
        rows = result.fetchall()

        # Expected combinations
        expected = [
            ('traditional', 'DEFAULT_RISK'),
            ('traditional', 'HIGH_RISK'),
            ('raw_lstm', 'default_risk'),
            ('raw_lstm', 'high_risk'),
            ('kronos', 'DEFAULT_RISK'),
            ('kronos', 'HIGH_RISK')
        ]

        existing = {(row[0], row[1]): row[2] for row in rows}

        logger.info(f"\nPredictions for {today}:")
        logger.info("\n  " + "-"*80)
        logger.info(f"  {'Model':<15} {'Strategy':<20} {'Count':<10} {'Status'}")
        logger.info("  " + "-"*80)

        all_present = True
        total_stocks = 0

        for model, strategy in expected:
            count = existing.get((model, strategy), 0)
            status = "✅" if count > 0 else "❌ MISSING"
            logger.info(f"  {model:<15} {strategy:<20} {count:<10} {status}")

            if count == 0:
                all_present = False
            else:
                total_stocks += count

        logger.info("  " + "-"*80)
        logger.info(f"  {'TOTAL':<15} {'':<20} {total_stocks:<10}")

        logger.info("\n" + "="*100)

        if all_present:
            logger.info("🎉 SUCCESS! All 6 combinations available!")
            logger.info(f"   Total stocks across all models: {total_stocks}")
            logger.info("\n   ✅ Traditional ML → DEFAULT_RISK, HIGH_RISK")
            logger.info("   ✅ Raw LSTM → default_risk, high_risk")
            logger.info("   ✅ Kronos → DEFAULT_RISK, HIGH_RISK")
            logger.info("\n   Navigate to Suggested Stocks page to see the complete triple model view!")
        else:
            logger.warning("⚠️  Some combinations are still missing!")

        logger.info("="*100)

        return all_present


def main():
    """Main execution."""
    logger.info("\n\n")
    logger.info("█" * 100)
    logger.info("█" + " "*98 + "█")
    logger.info("█" + " "*10 + "TRIPLE MODEL PREDICTION GENERATOR - All 3 Models × 2 Strategies = 6 Combinations" + " "*10 + "█")
    logger.info("█" + " "*98 + "█")
    logger.info("█" * 100)

    # Step 1: Check current status
    check_current_status()

    # Step 2: Generate Traditional ML (DEFAULT_RISK + HIGH_RISK)
    logger.info("\n📝 Step 1/3: Traditional ML (RF + XGBoost)")
    generate_traditional_ml()

    # Step 3: Generate Raw LSTM (default_risk + high_risk)
    logger.info("\n📝 Step 2/3: Raw LSTM (Deep Learning)")
    generate_raw_lstm()

    # Step 4: Generate Kronos (DEFAULT_RISK + HIGH_RISK)
    logger.info("\n📝 Step 3/3: Kronos (Pattern-based)")
    generate_kronos()

    # Step 5: Verify all combinations
    all_present = verify_all_combinations()

    # Summary
    logger.info("\n" + "="*100)
    logger.info("📊 GENERATION COMPLETE")
    logger.info("="*100)

    if all_present:
        logger.info("\n✅ All 3 models × 2 strategies = 6 combinations generated successfully!")
        logger.info("   Your triple model view is now fully populated.")
        return 0
    else:
        logger.error("\n❌ Some predictions are missing. Check logs above.")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
