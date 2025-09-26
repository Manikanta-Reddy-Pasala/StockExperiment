"""
Stock Filtering Service

Provides centralized filtering logic for stock screening operations.
Separates filtering concerns from data access and transformation.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Types of filters that can be applied to stocks."""
    TRADEABLE = "tradeable"
    MARKET_CAP = "market_cap"
    VOLUME = "volume"
    PRICE = "price"
    SECTOR = "sector"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    CUSTOM = "custom"


class StockFilteringService:
    """
    Service for applying various filters to stock data.

    This service encapsulates all stock filtering logic, making it
    reusable and testable independently of data access or UI concerns.
    """

    def __init__(self):
        """Initialize the stock filtering service."""
        self.applied_filters: List[Dict[str, Any]] = []
        self.filter_stats: Dict[str, Any] = {}

    def filter_tradeable_stocks(self, stocks: List[Any]) -> List[Any]:
        """
        Filter stocks to include only tradeable and active ones.

        Args:
            stocks: List of stock objects to filter

        Returns:
            List of tradeable and active stocks
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            # Apply tradeable filter
            filtered_stocks = [
                stock for stock in stocks
                if self._is_tradeable(stock)
            ]

            # Record filter statistics
            self._record_filter_stats(
                FilterType.TRADEABLE,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"Tradeable filter: {initial_count} -> {len(filtered_stocks)} "
                f"({initial_count - len(filtered_stocks)} removed)"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering tradeable stocks: {e}")
            return stocks  # Return original list on error

    def filter_by_market_cap(self, stocks: List[Any],
                            min_cap: Optional[float] = None,
                            max_cap: Optional[float] = None,
                            categories: Optional[List[str]] = None) -> List[Any]:
        """
        Filter stocks by market capitalization.

        Args:
            stocks: List of stock objects to filter
            min_cap: Minimum market cap (optional)
            max_cap: Maximum market cap (optional)
            categories: List of market cap categories to include (optional)

        Returns:
            List of stocks matching market cap criteria
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            filtered_stocks = []

            for stock in stocks:
                if self._meets_market_cap_criteria(stock, min_cap, max_cap, categories):
                    filtered_stocks.append(stock)

            self._record_filter_stats(
                FilterType.MARKET_CAP,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"Market cap filter: {initial_count} -> {len(filtered_stocks)} "
                f"(min: {min_cap}, max: {max_cap}, categories: {categories})"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering by market cap: {e}")
            return stocks

    def filter_by_volume(self, stocks: List[Any],
                        min_volume: Optional[int] = None,
                        min_avg_volume: Optional[int] = None) -> List[Any]:
        """
        Filter stocks by trading volume.

        Args:
            stocks: List of stock objects to filter
            min_volume: Minimum current volume
            min_avg_volume: Minimum average volume

        Returns:
            List of stocks meeting volume criteria
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            filtered_stocks = []

            for stock in stocks:
                if self._meets_volume_criteria(stock, min_volume, min_avg_volume):
                    filtered_stocks.append(stock)

            self._record_filter_stats(
                FilterType.VOLUME,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"Volume filter: {initial_count} -> {len(filtered_stocks)} "
                f"(min_volume: {min_volume}, min_avg_volume: {min_avg_volume})"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering by volume: {e}")
            return stocks

    def filter_by_price_range(self, stocks: List[Any],
                             min_price: Optional[float] = None,
                             max_price: Optional[float] = None) -> List[Any]:
        """
        Filter stocks by price range.

        Args:
            stocks: List of stock objects to filter
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of stocks within price range
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            filtered_stocks = []

            for stock in stocks:
                if self._meets_price_criteria(stock, min_price, max_price):
                    filtered_stocks.append(stock)

            self._record_filter_stats(
                FilterType.PRICE,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"Price filter: {initial_count} -> {len(filtered_stocks)} "
                f"(range: {min_price} - {max_price})"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering by price: {e}")
            return stocks

    def filter_by_sectors(self, stocks: List[Any],
                         included_sectors: Optional[List[str]] = None,
                         excluded_sectors: Optional[List[str]] = None) -> List[Any]:
        """
        Filter stocks by sector inclusion/exclusion.

        Args:
            stocks: List of stock objects to filter
            included_sectors: Sectors to include (if specified, only these are included)
            excluded_sectors: Sectors to exclude

        Returns:
            List of stocks matching sector criteria
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            filtered_stocks = []

            for stock in stocks:
                if self._meets_sector_criteria(stock, included_sectors, excluded_sectors):
                    filtered_stocks.append(stock)

            self._record_filter_stats(
                FilterType.SECTOR,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                f"Sector filter: {initial_count} -> {len(filtered_stocks)} "
                f"(included: {included_sectors}, excluded: {excluded_sectors})"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering by sectors: {e}")
            return stocks

    def apply_custom_filter(self, stocks: List[Any],
                           filter_func: callable,
                           filter_name: str = "custom") -> List[Any]:
        """
        Apply a custom filter function to stocks.

        Args:
            stocks: List of stock objects to filter
            filter_func: Custom filter function that returns True/False for each stock
            filter_name: Name of the custom filter for logging

        Returns:
            List of stocks passing the custom filter
        """
        start_time = datetime.now()
        initial_count = len(stocks)

        try:
            filtered_stocks = [stock for stock in stocks if filter_func(stock)]

            self._record_filter_stats(
                FilterType.CUSTOM,
                initial_count,
                len(filtered_stocks),
                (datetime.now() - start_time).total_seconds(),
                custom_name=filter_name
            )

            logger.info(
                f"Custom filter '{filter_name}': {initial_count} -> {len(filtered_stocks)}"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error applying custom filter '{filter_name}': {e}")
            return stocks

    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about applied filters.

        Returns:
            Dictionary containing filter statistics
        """
        return {
            'total_filters_applied': len(self.applied_filters),
            'filters': self.applied_filters,
            'filter_stats': self.filter_stats,
            'total_execution_time': sum(
                f.get('execution_time', 0) for f in self.applied_filters
            )
        }

    def reset_statistics(self):
        """Reset filter statistics for a new filtering session."""
        self.applied_filters = []
        self.filter_stats = {}
        logger.debug("Filter statistics reset")

    # Private helper methods

    def _is_tradeable(self, stock: Any) -> bool:
        """Check if a stock is tradeable and active."""
        return (
            hasattr(stock, 'is_tradeable') and stock.is_tradeable and
            hasattr(stock, 'is_active') and stock.is_active
        )

    def _meets_market_cap_criteria(self, stock: Any,
                                   min_cap: Optional[float],
                                   max_cap: Optional[float],
                                   categories: Optional[List[str]]) -> bool:
        """Check if stock meets market cap criteria."""
        if not hasattr(stock, 'market_cap'):
            return False

        market_cap = stock.market_cap
        if market_cap is None:
            return False

        # Check min/max bounds
        if min_cap is not None and market_cap < min_cap:
            return False
        if max_cap is not None and market_cap > max_cap:
            return False

        # Check categories
        if categories and hasattr(stock, 'market_cap_category'):
            if stock.market_cap_category not in categories:
                return False

        return True

    def _meets_volume_criteria(self, stock: Any,
                              min_volume: Optional[int],
                              min_avg_volume: Optional[int]) -> bool:
        """Check if stock meets volume criteria."""
        if min_volume is not None:
            if not hasattr(stock, 'volume') or stock.volume is None:
                return False
            if stock.volume < min_volume:
                return False

        if min_avg_volume is not None:
            if not hasattr(stock, 'avg_daily_volume_20d') or stock.avg_daily_volume_20d is None:
                return False
            if stock.avg_daily_volume_20d < min_avg_volume:
                return False

        return True

    def _meets_price_criteria(self, stock: Any,
                            min_price: Optional[float],
                            max_price: Optional[float]) -> bool:
        """Check if stock meets price criteria."""
        if not hasattr(stock, 'current_price') or stock.current_price is None:
            return False

        price = stock.current_price

        if min_price is not None and price < min_price:
            return False
        if max_price is not None and price > max_price:
            return False

        return True

    def _meets_sector_criteria(self, stock: Any,
                             included_sectors: Optional[List[str]],
                             excluded_sectors: Optional[List[str]]) -> bool:
        """Check if stock meets sector criteria."""
        if not hasattr(stock, 'sector'):
            return False

        sector = stock.sector

        # If included sectors are specified, stock must be in one of them
        if included_sectors and sector not in included_sectors:
            return False

        # Stock must not be in excluded sectors
        if excluded_sectors and sector in excluded_sectors:
            return False

        return True

    def _record_filter_stats(self, filter_type: FilterType,
                           initial_count: int,
                           final_count: int,
                           execution_time: float,
                           **kwargs):
        """Record statistics for a filter operation."""
        filter_record = {
            'type': filter_type.value,
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_count': initial_count - final_count,
            'retention_rate': final_count / initial_count if initial_count > 0 else 0,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        self.applied_filters.append(filter_record)

        # Update aggregate stats
        if filter_type.value not in self.filter_stats:
            self.filter_stats[filter_type.value] = {
                'total_applications': 0,
                'total_removed': 0,
                'total_execution_time': 0
            }

        stats = self.filter_stats[filter_type.value]
        stats['total_applications'] += 1
        stats['total_removed'] += filter_record['removed_count']
        stats['total_execution_time'] += execution_time

    def apply_filters_with_config(self, stocks: List[Any], config) -> List[Any]:
        """
        Apply filters based on configuration parameters.

        Args:
            stocks: List of stock objects to filter
            config: FilteringConfig object with filtering parameters

        Returns:
            List of stocks after applying configured filters
        """
        try:
            from src.services.stock_filtering.screening_config import FilterStrategy
            from concurrent.futures import ThreadPoolExecutor
            import time

            start_time = time.time()
            filtered_stocks = stocks
            initial_count = len(stocks)

            # Reset statistics at the beginning
            if config.enable_statistics if hasattr(config, 'enable_statistics') else True:
                self.reset_statistics()

            # Map strategies to filter methods
            strategy_map = {
                FilterStrategy.BASIC_TRADEABLE: self._apply_tradeable_filter_config,
                FilterStrategy.MARKET_CAP_BASED: self._apply_market_cap_filter_config,
                FilterStrategy.VOLUME_BASED: self._apply_volume_filter_config,
                FilterStrategy.PRICE_RANGE: self._apply_price_filter_config,
                FilterStrategy.SECTOR_BASED: self._apply_sector_filter_config,
                FilterStrategy.FUNDAMENTAL: self._apply_fundamental_filter_config,
                FilterStrategy.TECHNICAL: self._apply_technical_filter_config,
                FilterStrategy.CUSTOM: self._apply_custom_filters_config
            }

            # Apply filters based on chain operator
            if config.filter_chain_operator == "AND":
                # Apply filters sequentially (AND operation)
                for strategy in config.strategies:
                    if strategy in strategy_map:
                        filtered_stocks = strategy_map[strategy](filtered_stocks, config)

                        # Stop if empty and stop_on_empty is True
                        if not filtered_stocks and config.stop_on_empty:
                            logger.warning(f"Stopping filter chain: {strategy} returned empty")
                            break

            elif config.filter_chain_operator == "OR":
                # Apply filters in parallel and combine results (OR operation)
                all_results = set()

                if config.parallel_processing:
                    # Parallel processing for OR filters
                    with ThreadPoolExecutor(max_workers=len(config.strategies)) as executor:
                        futures = []
                        for strategy in config.strategies:
                            if strategy in strategy_map:
                                future = executor.submit(
                                    strategy_map[strategy], stocks, config
                                )
                                futures.append(future)

                        for future in futures:
                            result = future.result()
                            all_results.update(result)
                else:
                    # Sequential processing for OR filters
                    for strategy in config.strategies:
                        if strategy in strategy_map:
                            result = strategy_map[strategy](stocks, config)
                            all_results.update(result)

                filtered_stocks = list(all_results)

            # Log overall filtering results
            execution_time = time.time() - start_time
            logger.info(
                f"Config-based filtering complete: {initial_count} -> {len(filtered_stocks)} "
                f"stocks in {execution_time:.2f}s"
            )

            return filtered_stocks

        except Exception as e:
            logger.error(f"Error applying filters with config: {e}")
            return stocks  # Return original list on error

    def _apply_tradeable_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply tradeable filter based on config."""
        if config.apply_tradeable_filter:
            return self.filter_tradeable_stocks(stocks)
        return stocks

    def _apply_market_cap_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply market cap filter based on config."""
        return self.filter_by_market_cap(
            stocks,
            min_cap=config.min_market_cap,
            max_cap=config.max_market_cap,
            categories=config.market_cap_categories
        )

    def _apply_volume_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply volume filter based on config."""
        return self.filter_by_volume(
            stocks,
            min_volume=config.min_volume,
            min_avg_volume=config.min_avg_volume
        )

    def _apply_price_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply price filter based on config."""
        return self.filter_by_price_range(
            stocks,
            min_price=config.min_price,
            max_price=config.max_price
        )

    def _apply_sector_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply sector filter based on config."""
        return self.filter_by_sectors(
            stocks,
            included_sectors=config.included_sectors,
            excluded_sectors=config.excluded_sectors
        )

    def _apply_fundamental_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply fundamental filters based on config."""
        filtered_stocks = stocks

        # Filter by P/E ratio
        if config.min_pe_ratio is not None or config.max_pe_ratio is not None:
            filtered_stocks = self._filter_by_pe_ratio(
                filtered_stocks,
                min_pe=config.min_pe_ratio,
                max_pe=config.max_pe_ratio
            )

        # Filter by dividend yield
        if config.min_dividend_yield is not None:
            filtered_stocks = self._filter_by_dividend_yield(
                filtered_stocks,
                min_yield=config.min_dividend_yield
            )

        return filtered_stocks

    def _apply_technical_filter_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply technical filters based on config."""
        filtered_stocks = stocks

        # Filter by volatility
        if config.max_volatility is not None:
            filtered_stocks = self._filter_by_volatility(
                filtered_stocks,
                max_volatility=config.max_volatility
            )

        return filtered_stocks

    def _apply_custom_filters_config(self, stocks: List[Any], config) -> List[Any]:
        """Apply custom filters based on config."""
        filtered_stocks = stocks

        if config.custom_filters:
            for i, filter_func in enumerate(config.custom_filters):
                if callable(filter_func):
                    filtered_stocks = self.apply_custom_filter(
                        filtered_stocks,
                        filter_func,
                        f"custom_filter_{i}"
                    )

        return filtered_stocks

    def _filter_by_pe_ratio(self, stocks: List[Any],
                           min_pe: Optional[float] = None,
                           max_pe: Optional[float] = None) -> List[Any]:
        """Filter stocks by P/E ratio."""
        filtered = []
        for stock in stocks:
            pe_ratio = getattr(stock, 'pe_ratio', None)
            if pe_ratio is not None:
                if min_pe is not None and pe_ratio < min_pe:
                    continue
                if max_pe is not None and pe_ratio > max_pe:
                    continue
                filtered.append(stock)
        return filtered

    def _filter_by_dividend_yield(self, stocks: List[Any],
                                 min_yield: float) -> List[Any]:
        """Filter stocks by dividend yield."""
        filtered = []
        for stock in stocks:
            dividend_yield = getattr(stock, 'dividend_yield', 0)
            if dividend_yield >= min_yield:
                filtered.append(stock)
        return filtered

    def _filter_by_volatility(self, stocks: List[Any],
                            max_volatility: float) -> List[Any]:
        """Filter stocks by volatility."""
        filtered = []
        for stock in stocks:
            volatility = getattr(stock, 'volatility', 0)
            if volatility <= max_volatility:
                filtered.append(stock)
        return filtered