"""
Generate Kronos predictions for all strategies (DEFAULT_RISK + HIGH_RISK)

This tool generates daily stock predictions using the Kronos foundation model
and saves them to the daily_suggested_stocks table.

Usage:
    python tools/generate_kronos_predictions.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.database import get_database_manager
from src.services.ml.kronos_prediction_service import get_kronos_prediction_service
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_active_symbols():
    """Get all active stock symbols from database."""
    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT symbol
            FROM stocks
            WHERE is_active = true
            ORDER BY symbol
        """)
        result = session.execute(query)
        symbols = [row[0] for row in result]

    logger.info(f"Found {len(symbols)} active stocks")
    return symbols


def generate_kronos_predictions_all_strategies():
    """Generate Kronos predictions for both DEFAULT_RISK and HIGH_RISK strategies."""

    logger.info("="*60)
    logger.info("KRONOS PREDICTION GENERATION - ALL STRATEGIES")
    logger.info("="*60)

    # Get Kronos service
    kronos_service = get_kronos_prediction_service()

    # Get all active symbols
    all_symbols = get_all_active_symbols()

    if not all_symbols:
        logger.error("No active symbols found!")
        return

    # Strategy configurations
    strategies = [
        {
            'name': 'DEFAULT_RISK',
            'description': 'Conservative (Large-cap, 7% target, 5% stop loss)'
        },
        {
            'name': 'HIGH_RISK',
            'description': 'Aggressive (Small/Mid-cap, 12% target, 10% stop loss)'
        }
    ]

    today = datetime.now().date()

    for strategy_config in strategies:
        strategy_name = strategy_config['name']
        strategy_desc = strategy_config['description']

        logger.info("\n" + "="*60)
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Description: {strategy_desc}")
        logger.info("="*60)

        # Generate predictions for this strategy
        try:
            logger.info(f"Generating Kronos predictions for {strategy_name}...")

            predictions = kronos_service.batch_predict(
                symbols=all_symbols,
                user_id=1,
                max_workers=1,  # Sequential for now
                strategy=strategy_name
            )

            if predictions:
                logger.info(f"✅ Generated {len(predictions)} predictions for {strategy_name}")

                # Display top 10
                logger.info(f"\nTop 10 stocks for {strategy_name}:")
                logger.info("-" * 80)
                logger.info(f"{'Rank':<6} {'Symbol':<20} {'Score':<8} {'Target':<10} {'Rec':<6}")
                logger.info("-" * 80)

                for i, pred in enumerate(predictions[:10], 1):
                    score_pct = pred['ml_prediction_score'] * 100
                    logger.info(
                        f"{i:<6} {pred['symbol']:<20} {score_pct:>6.2f}%  "
                        f"₹{pred['ml_price_target']:>8.2f}  {pred['recommendation']:<6}"
                    )

                # Save to database
                logger.info(f"\nSaving Kronos predictions to database for {strategy_name}...")
                kronos_service.save_to_suggested_stocks(
                    predictions=predictions,
                    strategy=strategy_name,
                    prediction_date=today
                )

                logger.info(f"✅ {strategy_name} predictions saved successfully!")

            else:
                logger.warning(f"⚠️  No predictions generated for {strategy_name}")

        except Exception as e:
            logger.error(f"❌ Error generating predictions for {strategy_name}: {e}", exc_info=True)

    logger.info("\n" + "="*60)
    logger.info("KRONOS PREDICTION GENERATION COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    try:
        generate_kronos_predictions_all_strategies()
        logger.info("\n✅ All Kronos predictions generated successfully!")

    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
