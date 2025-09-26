"""
Minimal Stock Data Repository

Provides essential data access abstraction for stock-related database operations.
"""

import logging
from typing import List, Any
from .screening_config import DatabaseQueryConfig, QueryStrategy

logger = logging.getLogger(__name__)


class StockDataRepository:
    """Repository for essential stock data access operations."""

    def __init__(self, db_manager):
        """Initialize the stock data repository."""
        self.db_manager = db_manager

    def get_all_stocks(self) -> List[Any]:
        """Retrieve all stocks from the database."""
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks from database")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving all stocks: {e}")
            return []

    def get_stocks_with_config(self, config: DatabaseQueryConfig) -> List[Any]:
        """Retrieve stocks based on configuration."""
        try:
            if config.strategy == QueryStrategy.ALL_STOCKS:
                stocks = self.get_all_stocks()

                # Apply simple limit if specified
                if config.limit and config.limit > 0:
                    stocks = stocks[:config.limit]

                return stocks

            elif config.strategy == QueryStrategy.WITH_FILTERS:
                return self._get_stocks_with_filters(config)

            else:
                # Fallback to all stocks
                return self.get_all_stocks()

        except Exception as e:
            logger.error(f"Error retrieving stocks with config: {e}")
            return []

    def _get_stocks_with_filters(self, config: DatabaseQueryConfig) -> List[Any]:
        """Get stocks with basic database-level filters."""
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                query = session.query(Stock)

                # Apply sector filter at database level
                if config.sectors:
                    query = query.filter(Stock.sector.in_(config.sectors))

                # Apply active filter
                if not config.include_inactive:
                    query = query.filter(Stock.is_active == True)

                # Apply limit and offset
                if config.offset:
                    query = query.offset(config.offset)
                if config.limit:
                    query = query.limit(config.limit)

                stocks = query.all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} filtered stocks from database")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving filtered stocks: {e}")
            return []