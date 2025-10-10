#!/usr/bin/env python3
"""
Database Migration: Add model_type column to daily_suggested_stocks

This adds a column to track which ML model generated each stock suggestion:
- 'traditional': Feature-engineered ensemble (RF + XGB + LSTM)
- 'raw_lstm': Raw OHLCV simple LSTM with triple barrier

Usage:
    python tools/migrate_add_model_type.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.database import get_database_manager
from sqlalchemy import text
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_column_exists(session, table_name, column_name):
    """Check if a column exists in a table."""
    query = text(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table
        AND column_name = :column
    """)

    result = session.execute(query, {'table': table_name, 'column': column_name})
    return result.fetchone() is not None


def main():
    logger.info("="*60)
    logger.info("Database Migration: Add model_type to daily_suggested_stocks")
    logger.info("="*60)

    db = get_database_manager()

    with db.get_session() as session:
        # Check if column already exists
        if check_column_exists(session, 'daily_suggested_stocks', 'model_type'):
            logger.warning("Column 'model_type' already exists. Skipping migration.")
            return 0

        logger.info("\nAdding 'model_type' column...")

        try:
            # Add column with default value
            session.execute(text("""
                ALTER TABLE daily_suggested_stocks
                ADD COLUMN model_type VARCHAR(20) DEFAULT 'traditional'
            """))

            session.commit()
            logger.info("✓ Column added successfully")

            # Update existing records to have 'traditional' model type
            logger.info("\nUpdating existing records...")
            result = session.execute(text("""
                UPDATE daily_suggested_stocks
                SET model_type = 'traditional'
                WHERE model_type IS NULL
            """))

            session.commit()
            logger.info(f"✓ Updated {result.rowcount} existing records")

            # Verify
            logger.info("\nVerifying migration...")
            result = session.execute(text("""
                SELECT COUNT(*) as total, model_type
                FROM daily_suggested_stocks
                GROUP BY model_type
            """))

            for row in result:
                logger.info(f"  {row.model_type}: {row.total} records")

            logger.info("\n" + "="*60)
            logger.info("✓ Migration completed successfully!")
            logger.info("="*60)

            return 0

        except Exception as e:
            session.rollback()
            logger.error(f"✗ Migration failed: {e}")
            return 1


if __name__ == '__main__':
    exit(main())
