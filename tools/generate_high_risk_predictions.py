#!/usr/bin/env python3
"""
Generate High Risk Predictions - One-Time Script
Generates high_risk strategy predictions for both Traditional and Raw LSTM models
to populate the dual model view immediately (without waiting for scheduler).
"""

import sys
import logging
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_high_risk_predictions():
    """Generate high_risk predictions for Traditional model."""
    logger.info("=" * 100)
    logger.info("GENERATING HIGH RISK PREDICTIONS - Traditional Model")
    logger.info("=" * 100)

    try:
        orchestrator = SuggestedStocksSagaOrchestrator()

        logger.info("\n🎯 Running HIGH_RISK strategy for Traditional model...")
        logger.info(f"   Date: {date.today()}")
        logger.info(f"   Limit: 50 stocks per strategy")
        logger.info(f"   Model: Traditional (Feature-Engineered Ensemble)")
        logger.info("")

        # Run saga for high_risk strategy
        result = orchestrator.execute_suggested_stocks_saga(
            user_id=1,
            strategies=['high_risk'],  # Only high_risk
            limit=50
        )

        if result['status'] == 'completed':
            stocks_count = result['summary'].get('final_result_count', 0)

            logger.info("\n" + "=" * 100)
            logger.info("✅ HIGH RISK PREDICTIONS GENERATED SUCCESSFULLY!")
            logger.info("=" * 100)
            logger.info(f"  Stocks found: {stocks_count}")
            logger.info("")

            # Show ML prediction stats
            ml_step = next((s for s in result['summary']['step_summary']
                          if s['step_id'] == 'step6_ml_prediction'), None)
            if ml_step and ml_step['status'] == 'completed':
                metadata = ml_step.get('metadata', {})
                logger.info("  ML Predictions:")
                logger.info(f"    Average Score: {metadata.get('avg_prediction_score', 0):.4f}")
                logger.info(f"    Average Confidence: {metadata.get('avg_confidence', 0):.4f}")
                logger.info(f"    Average Risk Score: {metadata.get('avg_risk_score', 0):.4f}")
                logger.info("")

            # Show snapshot save stats
            snapshot_step = next((s for s in result['summary']['step_summary']
                                if s['step_id'] == 'step7_daily_snapshot'), None)
            if snapshot_step and snapshot_step['status'] == 'completed':
                metadata = snapshot_step.get('metadata', {})
                logger.info("  Database Snapshot:")
                logger.info(f"    Inserted: {metadata.get('inserted', 0)} new records")
                logger.info(f"    Updated: {metadata.get('updated', 0)} existing records")
                logger.info(f"    Skipped: {metadata.get('skipped', 0)} duplicates")
                logger.info("")

            # Show summary
            logger.info("  Pipeline Summary:")
            for step in result['summary']['step_summary']:
                status_icon = "✅" if step['status'] == 'completed' else "❌"
                logger.info(f"    {status_icon} {step['name']}: {step['output_count']} items ({step['duration_seconds']:.2f}s)")

            logger.info("")
            logger.info("=" * 100)
            logger.info("🎉 DONE! High risk predictions are now available in the database.")
            logger.info("   You can now view them in the Suggested Stocks page.")
            logger.info("=" * 100)

            return True

        else:
            logger.error("=" * 100)
            logger.error("❌ HIGH RISK PREDICTIONS FAILED")
            logger.error("=" * 100)
            logger.error(f"Errors: {result.get('errors', [])}")
            logger.error("=" * 100)
            return False

    except Exception as e:
        logger.error("=" * 100)
        logger.error("❌ SCRIPT FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 100)
        return False


def check_current_data():
    """Check what data we currently have in the database."""
    logger.info("\n" + "=" * 100)
    logger.info("CURRENT DATABASE STATUS (Before Generation)")
    logger.info("=" * 100)

    try:
        from sqlalchemy import text
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

            logger.info("\n" + "=" * 100)

    except Exception as e:
        logger.error(f"Failed to check current data: {e}")


def verify_after_generation():
    """Verify data after generation."""
    logger.info("\n" + "=" * 100)
    logger.info("VERIFICATION (After Generation)")
    logger.info("=" * 100)

    try:
        from sqlalchemy import text
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

            # Check if we have all 4 combinations (for dual model view)
            logger.info("\n  " + "=" * 80)
            logger.info("  DUAL MODEL VIEW STATUS:")
            logger.info("  " + "=" * 80)

            expected = [
                ('traditional', 'default_risk', 'Traditional + Default Risk'),
                ('traditional', 'high_risk', 'Traditional + High Risk'),
                ('raw_lstm', 'default_risk', 'Raw LSTM + Default Risk'),
                ('raw_lstm', 'high_risk', 'Raw LSTM + High Risk')
            ]

            existing = {(row[0], row[1]): row[2] for row in rows}

            for model, strategy, label in expected:
                count = existing.get((model, strategy), 0)
                status = "✅" if count > 0 else "❌"
                logger.info(f"  {status} {label:<35} : {count} stocks")

            logger.info("\n" + "=" * 100)

    except Exception as e:
        logger.error(f"Failed to verify data: {e}")


if __name__ == '__main__':
    logger.info("\n\n")
    logger.info("█" * 100)
    logger.info("█" + " " * 98 + "█")
    logger.info("█" + " " * 20 + "HIGH RISK PREDICTIONS GENERATOR" + " " * 47 + "█")
    logger.info("█" + " " * 98 + "█")
    logger.info("█" * 100)
    logger.info("")

    # Step 1: Check current data
    check_current_data()

    # Step 2: Generate high risk predictions
    success = generate_high_risk_predictions()

    # Step 3: Verify after generation
    if success:
        verify_after_generation()

    logger.info("\n")

    if success:
        logger.info("✅ Script completed successfully!")
        logger.info("   Next: Navigate to Suggested Stocks page to see all 4 model/risk combinations")
    else:
        logger.info("❌ Script failed. Check errors above.")

    logger.info("\n")
