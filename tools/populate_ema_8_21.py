#!/usr/bin/env python3
"""
Populate EMA 8, EMA 21, and DeMarker columns in technical_indicators table.

This script calculates the required indicators for the 8-21 EMA strategy
and stores them in the database.

Usage:
    python tools/populate_ema_8_21.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.data.technical_indicators_service import TechnicalIndicatorsService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def populate_ema_indicators():
    """
    Calculate and populate EMA 8, EMA 21, and DeMarker for all stocks.
    """
    try:
        logger.info("=" * 80)
        logger.info("Populating EMA 8, EMA 21, and DeMarker indicators")
        logger.info("=" * 80)

        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Get all unique symbols from historical_data
            from sqlalchemy import text

            result = session.execute(text("""
                SELECT DISTINCT symbol
                FROM historical_data
                WHERE date >= CURRENT_DATE - INTERVAL '400 days'
                ORDER BY symbol
            """))

            symbols = [row[0] for row in result.fetchall()]

            logger.info(f"Found {len(symbols)} symbols to process")

            if not symbols:
                logger.warning("No symbols found in historical_data table")
                return

        # Get calculator service (creates its own session)
        calculator = TechnicalIndicatorsService()

        # Calculate indicators for all stocks
        logger.info("Calculating 8-21 EMA strategy indicators for all stocks...")
        results = calculator.calculate_indicators_bulk(
            symbols=symbols,
            max_symbols=len(symbols)
        )

        logger.info(f"Successfully calculated indicators for {results.get('records_updated', 0)} stocks")

        logger.info("=" * 80)
        logger.info(f"✅ Successfully populated indicators:")
        logger.info(f"   - Records processed: {results.get('records_updated', 0)}")
        logger.info(f"   - Symbols processed: {results.get('symbols_processed', 0)}")
        logger.info("=" * 80)

        # Verify data
        with db_manager.get_session() as session:
            from sqlalchemy import text

            result = session.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(ema_8) as has_ema8,
                    COUNT(ema_21) as has_ema21,
                    COUNT(demarker) as has_demarker
                FROM technical_indicators
                WHERE date >= CURRENT_DATE - INTERVAL '365 days'
            """))

            row = result.fetchone()
            logger.info("Verification:")
            logger.info(f"  Total rows: {row[0]}")
            logger.info(f"  Has EMA 8: {row[1]} ({row[1]*100/row[0]:.1f}%)")
            logger.info(f"  Has EMA 21: {row[2]} ({row[2]*100/row[0]:.1f}%)")
            logger.info(f"  Has DeMarker: {row[3]} ({row[3]*100/row[0]:.1f}%)")

    except Exception as e:
        logger.error(f"Error populating indicators: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    import pandas as pd
    populate_ema_indicators()
