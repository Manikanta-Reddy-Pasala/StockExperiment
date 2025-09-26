"""
Stock Data Repository

Provides data access abstraction for stock-related database operations.
Centralizes all database queries and makes them testable.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

logger = logging.getLogger(__name__)


class StockDataRepository:
    """
    Repository for stock data access operations.

    This class encapsulates all database queries related to stock data,
    providing a clean interface for data access and making it easier to
    mock or replace the data source for testing.
    """

    def __init__(self, db_manager):
        """
        Initialize the stock data repository.

        Args:
            db_manager: Database manager instance for session management
        """
        self.db_manager = db_manager

    def get_all_stocks(self) -> List[Any]:
        """
        Retrieve all stocks from the database.

        Returns:
            List of all stock objects
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).all()
                # Detach from session to avoid lazy loading issues
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks from database")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving all stocks: {e}")
            return []

    def get_tradeable_stocks(self) -> List[Any]:
        """
        Retrieve only tradeable and active stocks from the database.

        Returns:
            List of tradeable stock objects
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).filter(
                    and_(
                        Stock.is_tradeable == True,
                        Stock.is_active == True
                    )
                ).all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} tradeable stocks from database")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving tradeable stocks: {e}")
            return []

    def get_stocks_by_symbols(self, symbols: List[str]) -> List[Any]:
        """
        Retrieve specific stocks by their symbols.

        Args:
            symbols: List of stock symbols to retrieve

        Returns:
            List of stock objects matching the symbols
        """
        try:
            from src.models.stock_models import Stock

            if not symbols:
                return []

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).filter(
                    Stock.symbol.in_(symbols)
                ).all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks for {len(symbols)} symbols")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving stocks by symbols: {e}")
            return []

    def get_stocks_by_sector(self, sector: str) -> List[Any]:
        """
        Retrieve stocks from a specific sector.

        Args:
            sector: Sector name to filter by

        Returns:
            List of stock objects in the specified sector
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).filter(
                    Stock.sector == sector
                ).all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks in sector: {sector}")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving stocks by sector: {e}")
            return []

    def get_stocks_by_market_cap_category(self, category: str) -> List[Any]:
        """
        Retrieve stocks by market cap category.

        Args:
            category: Market cap category (large_cap, mid_cap, small_cap)

        Returns:
            List of stock objects in the specified category
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stocks = session.query(Stock).filter(
                    Stock.market_cap_category == category
                ).all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks in category: {category}")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving stocks by market cap category: {e}")
            return []

    def get_stocks_with_filters(self, filters: Dict[str, Any]) -> List[Any]:
        """
        Retrieve stocks with multiple filter criteria.

        Args:
            filters: Dictionary of filter criteria
                - is_tradeable: bool
                - is_active: bool
                - min_price: float
                - max_price: float
                - min_volume: int
                - sectors: List[str]
                - market_cap_categories: List[str]

        Returns:
            List of stock objects matching all filter criteria
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                query = session.query(Stock)

                # Apply filters dynamically
                if filters.get('is_tradeable') is not None:
                    query = query.filter(Stock.is_tradeable == filters['is_tradeable'])

                if filters.get('is_active') is not None:
                    query = query.filter(Stock.is_active == filters['is_active'])

                if filters.get('min_price') is not None:
                    query = query.filter(Stock.current_price >= filters['min_price'])

                if filters.get('max_price') is not None:
                    query = query.filter(Stock.current_price <= filters['max_price'])

                if filters.get('min_volume') is not None:
                    query = query.filter(Stock.volume >= filters['min_volume'])

                if filters.get('sectors'):
                    query = query.filter(Stock.sector.in_(filters['sectors']))

                if filters.get('market_cap_categories'):
                    query = query.filter(
                        Stock.market_cap_category.in_(filters['market_cap_categories'])
                    )

                stocks = query.all()
                session.expunge_all()
                logger.info(f"Retrieved {len(stocks)} stocks with filters: {filters}")
                return stocks

        except Exception as e:
            logger.error(f"Error retrieving stocks with filters: {e}")
            return []

    def get_stocks_paginated(self, page: int = 1,
                           page_size: int = 50,
                           filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve stocks with pagination support.

        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            filters: Optional filter criteria

        Returns:
            Dictionary with paginated results and metadata
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                # Base query
                query = session.query(Stock)

                # Apply filters if provided
                if filters:
                    if filters.get('is_tradeable') is not None:
                        query = query.filter(Stock.is_tradeable == filters['is_tradeable'])
                    if filters.get('is_active') is not None:
                        query = query.filter(Stock.is_active == filters['is_active'])

                # Get total count
                total_count = query.count()

                # Apply pagination
                offset = (page - 1) * page_size
                stocks = query.limit(page_size).offset(offset).all()
                session.expunge_all()

                return {
                    'stocks': stocks,
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size,
                    'has_next': page * page_size < total_count,
                    'has_previous': page > 1
                }

        except Exception as e:
            logger.error(f"Error retrieving paginated stocks: {e}")
            return {
                'stocks': [],
                'page': page,
                'page_size': page_size,
                'total_count': 0,
                'total_pages': 0,
                'has_next': False,
                'has_previous': False
            }

    def get_stock_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics about stocks in the database.

        Returns:
            Dictionary containing stock statistics
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                total_stocks = session.query(func.count(Stock.id)).scalar()
                tradeable_stocks = session.query(func.count(Stock.id)).filter(
                    Stock.is_tradeable == True
                ).scalar()
                active_stocks = session.query(func.count(Stock.id)).filter(
                    Stock.is_active == True
                ).scalar()

                # Get sector distribution
                sector_counts = session.query(
                    Stock.sector,
                    func.count(Stock.id)
                ).group_by(Stock.sector).all()

                # Get market cap distribution
                market_cap_counts = session.query(
                    Stock.market_cap_category,
                    func.count(Stock.id)
                ).group_by(Stock.market_cap_category).all()

                return {
                    'total_stocks': total_stocks,
                    'tradeable_stocks': tradeable_stocks,
                    'active_stocks': active_stocks,
                    'sector_distribution': dict(sector_counts),
                    'market_cap_distribution': dict(market_cap_counts),
                    'last_updated': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting stock statistics: {e}")
            return {}

    def update_stock_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Update stock data for a specific symbol.

        Args:
            symbol: Stock symbol to update
            data: Dictionary of fields to update

        Returns:
            True if update was successful, False otherwise
        """
        try:
            from src.models.stock_models import Stock

            with self.db_manager.get_session() as session:
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()

                if not stock:
                    logger.warning(f"Stock not found for symbol: {symbol}")
                    return False

                # Update fields
                for key, value in data.items():
                    if hasattr(stock, key):
                        setattr(stock, key, value)

                stock.last_updated = datetime.now()
                session.commit()

                logger.info(f"Updated stock data for {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error updating stock data for {symbol}: {e}")
            return False

    def bulk_update_stocks(self, updates: List[Dict[str, Any]]) -> int:
        """
        Perform bulk update of stock data.

        Args:
            updates: List of dictionaries containing 'symbol' and data to update

        Returns:
            Number of stocks successfully updated
        """
        try:
            from src.models.stock_models import Stock

            updated_count = 0
            with self.db_manager.get_session() as session:
                for update in updates:
                    symbol = update.get('symbol')
                    if not symbol:
                        continue

                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    if not stock:
                        continue

                    # Update fields
                    for key, value in update.items():
                        if key != 'symbol' and hasattr(stock, key):
                            setattr(stock, key, value)

                    stock.last_updated = datetime.now()
                    updated_count += 1

                session.commit()
                logger.info(f"Bulk updated {updated_count} stocks")

            return updated_count

        except Exception as e:
            logger.error(f"Error in bulk update: {e}")
            return 0

    def get_stocks_with_config(self, config) -> List[Any]:
        """
        Retrieve stocks based on configuration parameters.

        Args:
            config: DatabaseQueryConfig object with query parameters

        Returns:
            List of stock objects based on configuration
        """
        try:
            from src.services.stock_filtering.screening_config import QueryStrategy

            # Map strategy to appropriate method
            strategy_map = {
                QueryStrategy.ALL_STOCKS: lambda: self.get_all_stocks(),
                QueryStrategy.TRADEABLE_ONLY: lambda: self.get_tradeable_stocks(),
                QueryStrategy.BY_SYMBOLS: lambda: self.get_stocks_by_symbols(config.symbols),
                QueryStrategy.BY_SECTOR: lambda: self._get_stocks_by_sectors_config(config),
                QueryStrategy.BY_MARKET_CAP: lambda: self._get_stocks_by_market_cap_config(config),
                QueryStrategy.WITH_FILTERS: lambda: self.get_stocks_with_filters(config.filters),
                QueryStrategy.PAGINATED: lambda: self._get_stocks_paginated_config(config),
                QueryStrategy.CUSTOM: lambda: self._execute_custom_query(config)
            }

            # Get stocks based on strategy
            if config.strategy in strategy_map:
                stocks = strategy_map[config.strategy]()
            else:
                logger.warning(f"Unknown query strategy: {config.strategy}, using ALL_STOCKS")
                stocks = self.get_all_stocks()

            # Apply additional configuration parameters
            stocks = self._apply_query_config_params(stocks, config)

            return stocks

        except Exception as e:
            logger.error(f"Error getting stocks with config: {e}")
            return []

    def _get_stocks_by_sectors_config(self, config) -> List[Any]:
        """Get stocks by sectors from config."""
        all_stocks = []
        if config.sectors:
            for sector in config.sectors:
                stocks = self.get_stocks_by_sector(sector)
                all_stocks.extend(stocks)
        return all_stocks

    def _get_stocks_by_market_cap_config(self, config) -> List[Any]:
        """Get stocks by market cap categories from config."""
        all_stocks = []
        if config.market_cap_categories:
            for category in config.market_cap_categories:
                stocks = self.get_stocks_by_market_cap_category(category)
                all_stocks.extend(stocks)
        return all_stocks

    def _get_stocks_paginated_config(self, config) -> List[Any]:
        """Get paginated stocks from config."""
        result = self.get_stocks_paginated(
            page=config.page,
            page_size=config.page_size,
            filters=config.filters
        )
        return result.get('stocks', [])

    def _execute_custom_query(self, config) -> List[Any]:
        """Execute custom query function from config."""
        if config.custom_query_func and callable(config.custom_query_func):
            return config.custom_query_func(self.db_manager)
        return []

    def _apply_query_config_params(self, stocks: List[Any], config) -> List[Any]:
        """Apply additional query configuration parameters."""
        try:
            # Apply ordering if specified
            if config.order_by and hasattr(stocks[0] if stocks else None, config.order_by):
                stocks = sorted(
                    stocks,
                    key=lambda x: getattr(x, config.order_by, 0),
                    reverse=config.order_desc
                )

            # Apply offset
            if config.offset and config.offset > 0:
                stocks = stocks[config.offset:]

            # Apply limit
            if config.limit and config.limit > 0:
                stocks = stocks[:config.limit]

            # Filter out inactive if not included
            if not config.include_inactive:
                stocks = [s for s in stocks if getattr(s, 'is_active', True)]

            return stocks

        except Exception as e:
            logger.error(f"Error applying query config params: {e}")
            return stocks