#!/usr/bin/env python3
"""
Generate ML predictions for all remaining model/strategy combinations
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_predictions(model_type, strategy):
    """Generate predictions for a specific model/strategy combination."""
    try:
        from src.services.data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
        from src.models.database import get_database_manager
        from sqlalchemy import text

        logger.info(f"\n{'='*80}")
        logger.info(f"Generating: {model_type} + {strategy}")
        logger.info("="*80)

        orchestrator = get_suggested_stocks_saga_orchestrator()
        result = orchestrator.execute_suggested_stocks_saga(
            user_id=1,
            strategies=[strategy],
            limit=50,
            model_type=model_type
        )

        logger.info(f"Status: {result['status']}")
        logger.info(f"Final Result Count: {result['summary']['final_result_count']}")

        # Verify in database
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            count_result = session.execute(text(
                f"SELECT COUNT(*) FROM daily_suggested_stocks WHERE model_type='{model_type}' AND strategy='{strategy}'"
            ))
            count = count_result.scalar()
            logger.info(f"Database Records: {count}")

        return count > 0

    except Exception as e:
        logger.error(f"Failed {model_type}/{strategy}: {e}")
        return False

def main():
    logger.info("="*80)
    logger.info("GENERATING ALL ML PREDICTIONS")
    logger.info("="*80)

    # Define all combinations (excluding Traditional/default_risk which is already done)
    combinations = [
        ("traditional", "high_risk"),
        ("raw_lstm", "default_risk"),
        ("raw_lstm", "high_risk"),
        ("kronos", "default_risk"),
        ("kronos", "high_risk"),
    ]

    success_count = 0
    for model_type, strategy in combinations:
        if generate_predictions(model_type, strategy):
            success_count += 1
            logger.info(f"✅ {model_type}/{strategy} completed")
        else:
            logger.error(f"❌ {model_type}/{strategy} failed")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    from src.models.database import get_database_manager
    from sqlalchemy import text

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        # Total records
        total_result = session.execute(text("SELECT COUNT(*) FROM daily_suggested_stocks"))
        total = total_result.scalar()

        # By model/strategy
        breakdown_result = session.execute(text(
            "SELECT model_type, strategy, COUNT(*) FROM daily_suggested_stocks GROUP BY model_type, strategy ORDER BY model_type, strategy"
        ))

        logger.info(f"\nTotal Records: {total}")
        logger.info("\nBreakdown:")
        for row in breakdown_result:
            logger.info(f"  {row[0]:15} {row[1]:15} {row[2]} stocks")

    logger.info(f"\n✅ Successfully generated {success_count}/{len(combinations)} combinations")
    logger.info("="*80)

if __name__ == '__main__':
    main()
