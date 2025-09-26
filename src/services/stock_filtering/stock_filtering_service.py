"""
Minimal Stock Filtering Service

Provides essential stock filtering functionality with configurable parameters.
"""

import logging
from typing import List, Any, Dict
from .screening_config import FilteringConfig, FilterStrategy

logger = logging.getLogger(__name__)


class StockFilteringService:
    """Service for essential stock filtering operations."""

    def __init__(self):
        """Initialize the filtering service."""
        self.filter_stats = {
            'total_filters_applied': 0,
            'filters': [],
            'filter_stats': {},
            'total_execution_time': 0
        }

    def reset_statistics(self):
        """Reset filter statistics."""
        self.filter_stats = {
            'total_filters_applied': 0,
            'filters': [],
            'filter_stats': {},
            'total_execution_time': 0
        }

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get current filter statistics."""
        return self.filter_stats.copy()

    def filter_tradeable_stocks(self, stocks: List[Any]) -> List[Any]:
        """Filter stocks to only include tradeable ones."""
        try:
            tradeable_stocks = [stock for stock in stocks
                              if stock.is_tradeable and stock.is_active]

            self.filter_stats['total_filters_applied'] += 1
            self.filter_stats['filters'].append('tradeable_filter')
            self.filter_stats['filter_stats']['tradeable'] = {
                'input_count': len(stocks),
                'output_count': len(tradeable_stocks),
                'filtered_out': len(stocks) - len(tradeable_stocks)
            }

            logger.info(f"Tradeable filter: {len(stocks)} -> {len(tradeable_stocks)} stocks")
            return tradeable_stocks

        except Exception as e:
            logger.error(f"Error in tradeable filter: {e}")
            return stocks

    def apply_filters_with_config(self, stocks: List[Any],
                                config: FilteringConfig) -> List[Any]:
        """Apply filters based on configuration."""
        try:
            filtered_stocks = stocks

            # Always apply tradeable filter if enabled
            if config.apply_tradeable_filter:
                filtered_stocks = self.filter_tradeable_stocks(filtered_stocks)

            # Apply additional filters based on strategies
            for strategy in config.strategies:
                if strategy == FilterStrategy.BASIC_TRADEABLE:
                    # Already applied above
                    continue
                elif strategy == FilterStrategy.PRICE_RANGE:
                    filtered_stocks = self._apply_price_range_filter(
                        filtered_stocks, config)
                elif strategy == FilterStrategy.VOLUME_BASED:
                    filtered_stocks = self._apply_volume_filter(
                        filtered_stocks, config)
                elif strategy == FilterStrategy.SECTOR_BASED:
                    filtered_stocks = self._apply_sector_filter(
                        filtered_stocks, config)

            logger.info(f"Applied {len(config.strategies)} filter strategies: "
                       f"{len(stocks)} -> {len(filtered_stocks)} stocks")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error applying filters with config: {e}")
            return stocks

    def _apply_price_range_filter(self, stocks: List[Any],
                                 config: FilteringConfig) -> List[Any]:
        """Apply price range filter."""
        try:
            filtered_stocks = []
            for stock in stocks:
                price = stock.current_price or 0

                # Check price range
                if config.min_price and price < config.min_price:
                    continue
                if config.max_price and price > config.max_price:
                    continue

                filtered_stocks.append(stock)

            self.filter_stats['filter_stats']['price_range'] = {
                'input_count': len(stocks),
                'output_count': len(filtered_stocks),
                'min_price': config.min_price,
                'max_price': config.max_price
            }

            logger.info(f"Price range filter: {len(stocks)} -> {len(filtered_stocks)} stocks")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error in price range filter: {e}")
            return stocks

    def _apply_volume_filter(self, stocks: List[Any],
                           config: FilteringConfig) -> List[Any]:
        """Apply volume filter."""
        try:
            filtered_stocks = []
            for stock in stocks:
                volume = stock.volume or 0

                if config.min_volume and volume < config.min_volume:
                    continue

                filtered_stocks.append(stock)

            self.filter_stats['filter_stats']['volume'] = {
                'input_count': len(stocks),
                'output_count': len(filtered_stocks),
                'min_volume': config.min_volume
            }

            logger.info(f"Volume filter: {len(stocks)} -> {len(filtered_stocks)} stocks")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error in volume filter: {e}")
            return stocks

    def _apply_sector_filter(self, stocks: List[Any],
                           config: FilteringConfig) -> List[Any]:
        """Apply sector filter."""
        try:
            if not config.included_sectors:
                return stocks

            filtered_stocks = []
            for stock in stocks:
                sector = getattr(stock, 'sector', None)

                if sector and sector in config.included_sectors:
                    filtered_stocks.append(stock)

            self.filter_stats['filter_stats']['sector'] = {
                'input_count': len(stocks),
                'output_count': len(filtered_stocks),
                'included_sectors': config.included_sectors
            }

            logger.info(f"Sector filter: {len(stocks)} -> {len(filtered_stocks)} stocks")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error in sector filter: {e}")
            return stocks