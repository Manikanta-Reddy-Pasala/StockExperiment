#!/usr/bin/env python3
"""
Test script to generate ML predictions for a single model/strategy
"""
import sys
import logging
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=" * 80)
        logger.info("TEST: Generating ML predictions for Traditional ML + DEFAULT_RISK")
        logger.info("=" * 80)

        from src.services.data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
        from src.models.database import get_database_manager

        # Initialize orchestrator
        orchestrator = get_suggested_stocks_saga_orchestrator()

        # Execute saga for traditional model with default_risk strategy
        logger.info("\nExecuting saga...")
        result = orchestrator.execute_suggested_stocks_saga(
            user_id=1,
            strategies=['default_risk'],
            limit=50,
            model_type='traditional'
        )

        logger.info("\n" + "=" * 80)
        logger.info("SAGA RESULT")
        logger.info("=" * 80)
        logger.info(f"Status: {result['status']}")
        logger.info(f"Total Steps: {result['summary']['total_steps']}")
        logger.info(f"Completed Steps: {result['summary']['completed_steps']}")
        logger.info(f"Failed Steps: {result['summary']['failed_steps']}")
        logger.info(f"Final Result Count: {result['summary']['final_result_count']}")

        # Check database
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE VERIFICATION")
        logger.info("=" * 80)

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            from sqlalchemy import text

            # Count records
            count_result = session.execute(text(
                "SELECT COUNT(*) FROM daily_suggested_stocks WHERE model_type='traditional' AND strategy='default_risk'"
            ))
            count = count_result.scalar()
            logger.info(f"Records in database: {count}")

            if count > 0:
                # Show sample record
                sample_result = session.execute(text(
                    "SELECT symbol, current_price, ml_prediction_score, ml_price_target FROM daily_suggested_stocks WHERE model_type='traditional' AND strategy='default_risk' LIMIT 5"
                ))
                logger.info("\nSample Records:")
                for row in sample_result:
                    logger.info(f"  {row[0]}: Price={row[1]}, Score={row[2]}, Target={row[3]}")

        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
