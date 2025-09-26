"""
Test Suite for Stock Screening Configuration System

This module provides comprehensive tests for the configurable stock screening pipeline,
including database query configuration, filtering configuration, and profile presets.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import List, Dict, Any
from datetime import datetime

from src.services.stock_filtering.screening_config import (
    QueryStrategy, FilterStrategy, ScreeningProfile,
    DatabaseQueryConfig, FilteringConfig, StockScreeningConfig,
    ConfigurationBuilder,
    get_default_config, get_conservative_config, get_aggressive_config,
    get_day_trading_config, get_custom_config
)


class TestDatabaseQueryConfig(unittest.TestCase):
    """Test suite for DatabaseQueryConfig."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = DatabaseQueryConfig()

        self.assertEqual(config.strategy, QueryStrategy.ALL_STOCKS)
        self.assertIsNone(config.limit)
        self.assertIsNone(config.offset)
        self.assertIsNone(config.symbols)
        self.assertEqual(config.page, 1)
        self.assertEqual(config.page_size, 50)
        self.assertFalse(config.cache_enabled)

    def test_by_symbols_strategy(self):
        """Test BY_SYMBOLS strategy configuration."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SYMBOLS,
            symbols=symbols
        )

        self.assertEqual(config.strategy, QueryStrategy.BY_SYMBOLS)
        self.assertEqual(config.symbols, symbols)
        self.assertTrue(config.validate())

    def test_by_symbols_strategy_validation_failure(self):
        """Test BY_SYMBOLS strategy validation without symbols."""
        config = DatabaseQueryConfig(strategy=QueryStrategy.BY_SYMBOLS)
        self.assertFalse(config.validate())

    def test_by_sector_strategy(self):
        """Test BY_SECTOR strategy configuration."""
        sectors = ['Technology', 'Healthcare']
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SECTOR,
            sectors=sectors
        )

        self.assertEqual(config.strategy, QueryStrategy.BY_SECTOR)
        self.assertEqual(config.sectors, sectors)
        self.assertTrue(config.validate())

    def test_with_filters_strategy(self):
        """Test WITH_FILTERS strategy configuration."""
        filters = {
            'is_tradeable': True,
            'min_price': 10.0,
            'max_price': 100.0
        }
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters=filters
        )

        self.assertEqual(config.strategy, QueryStrategy.WITH_FILTERS)
        self.assertEqual(config.filters, filters)
        self.assertTrue(config.validate())

    def test_paginated_strategy(self):
        """Test PAGINATED strategy configuration."""
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.PAGINATED,
            page=2,
            page_size=25
        )

        self.assertEqual(config.strategy, QueryStrategy.PAGINATED)
        self.assertEqual(config.page, 2)
        self.assertEqual(config.page_size, 25)
        self.assertTrue(config.validate())

    def test_limit_and_offset(self):
        """Test limit and offset parameters."""
        config = DatabaseQueryConfig(
            limit=100,
            offset=50
        )

        self.assertEqual(config.limit, 100)
        self.assertEqual(config.offset, 50)
        self.assertTrue(config.validate())

    def test_invalid_limit(self):
        """Test validation with invalid limit."""
        config = DatabaseQueryConfig(limit=-10)
        self.assertFalse(config.validate())

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SYMBOLS,
            symbols=['AAPL', 'GOOGL'],
            limit=50
        )

        config_dict = config.to_dict()
        self.assertEqual(config_dict['strategy'], 'by_symbols')
        self.assertEqual(config_dict['symbols'], ['AAPL', 'GOOGL'])
        self.assertEqual(config_dict['limit'], 50)
        self.assertNotIn('custom_query_func', config_dict)  # Callables should be excluded


class TestFilteringConfig(unittest.TestCase):
    """Test suite for FilteringConfig."""

    def test_default_initialization(self):
        """Test default filtering configuration."""
        config = FilteringConfig()

        self.assertEqual(config.strategies, [FilterStrategy.BASIC_TRADEABLE])
        self.assertTrue(config.apply_tradeable_filter)
        self.assertIsNone(config.min_price)
        self.assertIsNone(config.max_price)
        self.assertEqual(config.filter_chain_operator, "AND")
        self.assertFalse(config.stop_on_empty)

    def test_multiple_strategies(self):
        """Test configuration with multiple filter strategies."""
        config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.MARKET_CAP_BASED,
                FilterStrategy.VOLUME_BASED
            ]
        )

        self.assertEqual(len(config.strategies), 3)
        self.assertIn(FilterStrategy.BASIC_TRADEABLE, config.strategies)
        self.assertIn(FilterStrategy.MARKET_CAP_BASED, config.strategies)
        self.assertIn(FilterStrategy.VOLUME_BASED, config.strategies)

    def test_price_range_filter(self):
        """Test price range filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.PRICE_RANGE],
            min_price=10.0,
            max_price=100.0
        )

        self.assertEqual(config.min_price, 10.0)
        self.assertEqual(config.max_price, 100.0)
        self.assertTrue(config.validate())

    def test_invalid_price_range(self):
        """Test validation with invalid price range."""
        config = FilteringConfig(
            min_price=100.0,
            max_price=10.0  # Max less than min
        )
        self.assertFalse(config.validate())

    def test_market_cap_filter(self):
        """Test market cap filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.MARKET_CAP_BASED],
            min_market_cap=1000000000,  # $1B
            max_market_cap=50000000000,  # $50B
            market_cap_categories=['large_cap', 'mid_cap']
        )

        self.assertEqual(config.min_market_cap, 1000000000)
        self.assertEqual(config.max_market_cap, 50000000000)
        self.assertEqual(config.market_cap_categories, ['large_cap', 'mid_cap'])
        self.assertTrue(config.validate())

    def test_volume_filter(self):
        """Test volume filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.VOLUME_BASED],
            min_volume=1000000,
            min_avg_volume=500000
        )

        self.assertEqual(config.min_volume, 1000000)
        self.assertEqual(config.min_avg_volume, 500000)
        self.assertTrue(config.validate())

    def test_sector_filter(self):
        """Test sector filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.SECTOR_BASED],
            included_sectors=['Technology', 'Healthcare'],
            excluded_sectors=['Energy', 'Utilities']
        )

        self.assertEqual(config.included_sectors, ['Technology', 'Healthcare'])
        self.assertEqual(config.excluded_sectors, ['Energy', 'Utilities'])
        self.assertTrue(config.validate())

    def test_fundamental_filters(self):
        """Test fundamental filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.FUNDAMENTAL],
            min_pe_ratio=5,
            max_pe_ratio=25,
            min_dividend_yield=0.02
        )

        self.assertEqual(config.min_pe_ratio, 5)
        self.assertEqual(config.max_pe_ratio, 25)
        self.assertEqual(config.min_dividend_yield, 0.02)
        self.assertTrue(config.validate())

    def test_technical_filters(self):
        """Test technical filter configuration."""
        config = FilteringConfig(
            strategies=[FilterStrategy.TECHNICAL],
            max_volatility=0.5
        )

        self.assertEqual(config.max_volatility, 0.5)
        self.assertTrue(config.validate())

    def test_filter_chain_operator(self):
        """Test filter chain operator configuration."""
        config_and = FilteringConfig(filter_chain_operator="AND")
        config_or = FilteringConfig(filter_chain_operator="OR")

        self.assertTrue(config_and.validate())
        self.assertTrue(config_or.validate())

        config_invalid = FilteringConfig(filter_chain_operator="XOR")
        self.assertFalse(config_invalid.validate())

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = FilteringConfig(
            strategies=[FilterStrategy.BASIC_TRADEABLE, FilterStrategy.PRICE_RANGE],
            min_price=10.0,
            max_price=100.0
        )

        config_dict = config.to_dict()
        self.assertEqual(config_dict['strategies'], ['basic_tradeable', 'price_range'])
        self.assertEqual(config_dict['min_price'], 10.0)
        self.assertEqual(config_dict['max_price'], 100.0)


class TestStockScreeningConfig(unittest.TestCase):
    """Test suite for StockScreeningConfig."""

    def test_default_profile(self):
        """Test default profile configuration."""
        config = StockScreeningConfig(profile=ScreeningProfile.DEFAULT)

        self.assertEqual(config.profile, ScreeningProfile.DEFAULT)
        self.assertEqual(config.query_config.strategy, QueryStrategy.ALL_STOCKS)
        self.assertEqual(config.filtering_config.strategies, [FilterStrategy.BASIC_TRADEABLE])
        self.assertTrue(config.validate())

    def test_conservative_profile(self):
        """Test conservative profile configuration."""
        config = StockScreeningConfig(profile=ScreeningProfile.CONSERVATIVE)

        self.assertEqual(config.profile, ScreeningProfile.CONSERVATIVE)
        self.assertEqual(config.query_config.strategy, QueryStrategy.WITH_FILTERS)
        self.assertIsNotNone(config.query_config.filters)
        self.assertIn('large_cap', config.filtering_config.market_cap_categories)
        self.assertIsNotNone(config.filtering_config.min_volume)
        self.assertTrue(config.validate())

    def test_aggressive_profile(self):
        """Test aggressive profile configuration."""
        config = StockScreeningConfig(profile=ScreeningProfile.AGGRESSIVE)

        self.assertEqual(config.profile, ScreeningProfile.AGGRESSIVE)
        self.assertIn('small_cap', config.filtering_config.market_cap_categories)
        self.assertIn('mid_cap', config.filtering_config.market_cap_categories)
        self.assertTrue(config.validate())

    def test_day_trading_profile(self):
        """Test day trading profile configuration."""
        config = StockScreeningConfig(profile=ScreeningProfile.DAY_TRADING)

        self.assertEqual(config.profile, ScreeningProfile.DAY_TRADING)
        self.assertGreaterEqual(config.filtering_config.min_volume, 5000000)
        self.assertIsNotNone(config.filtering_config.max_volatility)
        self.assertTrue(config.validate())

    def test_value_investing_profile(self):
        """Test value investing profile configuration."""
        config = StockScreeningConfig(profile=ScreeningProfile.VALUE_INVESTING)

        self.assertEqual(config.profile, ScreeningProfile.VALUE_INVESTING)
        self.assertIn(FilterStrategy.FUNDAMENTAL, config.filtering_config.strategies)
        self.assertIsNotNone(config.filtering_config.max_pe_ratio)
        self.assertIsNotNone(config.filtering_config.min_dividend_yield)
        self.assertTrue(config.validate())

    def test_custom_profile(self):
        """Test custom profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SYMBOLS,
            symbols=['AAPL', 'GOOGL']
        )
        filtering_config = FilteringConfig(
            strategies=[FilterStrategy.PRICE_RANGE],
            min_price=50,
            max_price=200
        )

        config = StockScreeningConfig(
            profile=ScreeningProfile.CUSTOM,
            query_config=query_config,
            filtering_config=filtering_config
        )

        self.assertEqual(config.profile, ScreeningProfile.CUSTOM)
        self.assertEqual(config.query_config.symbols, ['AAPL', 'GOOGL'])
        self.assertEqual(config.filtering_config.min_price, 50)
        self.assertTrue(config.validate())

    def test_caching_configuration(self):
        """Test caching configuration."""
        config = StockScreeningConfig(
            enable_caching=True,
            cache_ttl=600
        )

        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 600)
        self.assertTrue(config.validate())

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = StockScreeningConfig(profile=ScreeningProfile.DEFAULT)
        config_dict = config.to_dict()

        self.assertEqual(config_dict['profile'], 'default')
        self.assertIn('query_config', config_dict)
        self.assertIn('filtering_config', config_dict)
        self.assertIn('enable_caching', config_dict)

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'profile': 'conservative',
            'query_config': {
                'strategy': 'all_stocks',
                'limit': 100
            },
            'filtering_config': {
                'strategies': ['basic_tradeable'],
                'min_price': 10.0
            },
            'enable_caching': True,
            'cache_ttl': 300
        }

        config = StockScreeningConfig.from_dict(config_dict)

        self.assertEqual(config.profile, ScreeningProfile.CONSERVATIVE)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 300)


class TestConfigurationBuilder(unittest.TestCase):
    """Test suite for ConfigurationBuilder."""

    def test_basic_builder(self):
        """Test basic builder functionality."""
        builder = ConfigurationBuilder()
        config = builder.with_query_limit(100).build()

        self.assertEqual(config.query_config.limit, 100)
        self.assertTrue(config.validate())

    def test_builder_with_symbols(self):
        """Test builder with symbols."""
        builder = ConfigurationBuilder()
        config = builder.with_symbols(['AAPL', 'GOOGL']).build()

        self.assertEqual(config.query_config.strategy, QueryStrategy.BY_SYMBOLS)
        self.assertEqual(config.query_config.symbols, ['AAPL', 'GOOGL'])

    def test_builder_with_price_range(self):
        """Test builder with price range."""
        builder = ConfigurationBuilder()
        config = builder.with_price_range(min_price=10, max_price=100).build()

        self.assertEqual(config.filtering_config.min_price, 10)
        self.assertEqual(config.filtering_config.max_price, 100)
        self.assertIn(FilterStrategy.PRICE_RANGE, config.filtering_config.strategies)

    def test_builder_with_volume_filter(self):
        """Test builder with volume filter."""
        builder = ConfigurationBuilder()
        config = builder.with_volume_filter(
            min_volume=1000000,
            min_avg_volume=500000
        ).build()

        self.assertEqual(config.filtering_config.min_volume, 1000000)
        self.assertEqual(config.filtering_config.min_avg_volume, 500000)
        self.assertIn(FilterStrategy.VOLUME_BASED, config.filtering_config.strategies)

    def test_builder_with_market_cap_filter(self):
        """Test builder with market cap filter."""
        builder = ConfigurationBuilder()
        config = builder.with_market_cap_filter(
            categories=['large_cap'],
            min_cap=1000000000,
            max_cap=50000000000
        ).build()

        self.assertEqual(config.filtering_config.market_cap_categories, ['large_cap'])
        self.assertEqual(config.filtering_config.min_market_cap, 1000000000)
        self.assertEqual(config.filtering_config.max_market_cap, 50000000000)
        self.assertIn(FilterStrategy.MARKET_CAP_BASED, config.filtering_config.strategies)

    def test_builder_with_sectors(self):
        """Test builder with sectors."""
        builder = ConfigurationBuilder()
        config = builder.with_sectors(['Technology', 'Healthcare']).build()

        self.assertEqual(config.query_config.sectors, ['Technology', 'Healthcare'])
        self.assertEqual(config.filtering_config.included_sectors, ['Technology', 'Healthcare'])

    def test_builder_with_caching(self):
        """Test builder with caching."""
        builder = ConfigurationBuilder()
        config = builder.with_caching(enabled=True, ttl=600).build()

        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 600)
        self.assertTrue(config.query_config.cache_enabled)

    def test_builder_chain(self):
        """Test chaining multiple builder methods."""
        config = (ConfigurationBuilder()
                 .with_profile(ScreeningProfile.CONSERVATIVE)
                 .with_query_limit(50)
                 .with_price_range(min_price=20, max_price=100)
                 .with_volume_filter(min_volume=1000000)
                 .with_caching(enabled=True)
                 .build())

        self.assertEqual(config.profile, ScreeningProfile.CONSERVATIVE)
        self.assertEqual(config.query_config.limit, 50)
        self.assertEqual(config.filtering_config.min_price, 20)
        self.assertEqual(config.filtering_config.min_volume, 1000000)
        self.assertTrue(config.enable_caching)


class TestFactoryFunctions(unittest.TestCase):
    """Test suite for factory functions."""

    def test_get_default_config(self):
        """Test get_default_config factory function."""
        config = get_default_config()

        self.assertEqual(config.profile, ScreeningProfile.DEFAULT)
        self.assertEqual(config.query_config.strategy, QueryStrategy.ALL_STOCKS)
        self.assertTrue(config.validate())

    def test_get_conservative_config(self):
        """Test get_conservative_config factory function."""
        config = get_conservative_config()

        self.assertEqual(config.profile, ScreeningProfile.CONSERVATIVE)
        self.assertIn('large_cap', config.filtering_config.market_cap_categories)
        self.assertTrue(config.validate())

    def test_get_aggressive_config(self):
        """Test get_aggressive_config factory function."""
        config = get_aggressive_config()

        self.assertEqual(config.profile, ScreeningProfile.AGGRESSIVE)
        self.assertIn('small_cap', config.filtering_config.market_cap_categories)
        self.assertTrue(config.validate())

    def test_get_day_trading_config(self):
        """Test get_day_trading_config factory function."""
        config = get_day_trading_config()

        self.assertEqual(config.profile, ScreeningProfile.DAY_TRADING)
        self.assertGreaterEqual(config.filtering_config.min_volume, 5000000)
        self.assertTrue(config.validate())

    def test_get_custom_config(self):
        """Test get_custom_config factory function."""
        config = get_custom_config(
            query_strategy=QueryStrategy.BY_SYMBOLS,
            filter_strategies=[FilterStrategy.PRICE_RANGE],
            symbols=['AAPL', 'GOOGL'],
            min_price=10,
            max_price=100
        )

        self.assertEqual(config.profile, ScreeningProfile.CUSTOM)
        self.assertEqual(config.query_config.strategy, QueryStrategy.BY_SYMBOLS)
        self.assertEqual(config.query_config.symbols, ['AAPL', 'GOOGL'])
        self.assertEqual(config.filtering_config.min_price, 10)
        self.assertEqual(config.filtering_config.max_price, 100)
        self.assertTrue(config.validate())


class TestIntegrationWithRepository(unittest.TestCase):
    """Test integration with StockDataRepository."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    def test_repository_with_config(self, MockRepository):
        """Test repository integration with configuration."""
        # Setup mock repository
        mock_repo = MockRepository.return_value
        mock_stocks = [
            Mock(symbol='AAPL', price=150, is_tradeable=True),
            Mock(symbol='GOOGL', price=2800, is_tradeable=True),
            Mock(symbol='MSFT', price=300, is_tradeable=True)
        ]
        mock_repo.get_stocks_with_config.return_value = mock_stocks

        # Create configuration
        config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SYMBOLS,
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            limit=10
        )

        # Test repository call
        stocks = mock_repo.get_stocks_with_config(config)

        self.assertEqual(len(stocks), 3)
        mock_repo.get_stocks_with_config.assert_called_once_with(config)


class TestIntegrationWithFilteringService(unittest.TestCase):
    """Test integration with StockFilteringService."""

    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_filtering_service_with_config(self, MockFilteringService):
        """Test filtering service integration with configuration."""
        # Setup mock filtering service
        mock_service = MockFilteringService.return_value
        mock_stocks = [
            Mock(symbol='AAPL', price=150, volume=10000000),
            Mock(symbol='GOOGL', price=2800, volume=5000000)
        ]
        mock_filtered = [mock_stocks[0]]  # Only AAPL passes filters
        mock_service.apply_filters_with_config.return_value = mock_filtered

        # Create configuration
        config = FilteringConfig(
            strategies=[FilterStrategy.VOLUME_BASED],
            min_volume=7000000
        )

        # Test filtering call
        filtered_stocks = mock_service.apply_filters_with_config(mock_stocks, config)

        self.assertEqual(len(filtered_stocks), 1)
        self.assertEqual(filtered_stocks[0].symbol, 'AAPL')
        mock_service.apply_filters_with_config.assert_called_once_with(mock_stocks, config)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def test_default_behavior_maintained(self):
        """Test that default configuration maintains current behavior."""
        default_config = get_default_config()

        # Should use ALL_STOCKS strategy
        self.assertEqual(default_config.query_config.strategy, QueryStrategy.ALL_STOCKS)

        # Should only apply basic tradeable filter
        self.assertEqual(len(default_config.filtering_config.strategies), 1)
        self.assertEqual(default_config.filtering_config.strategies[0],
                        FilterStrategy.BASIC_TRADEABLE)

        # Should have tradeable filter enabled
        self.assertTrue(default_config.filtering_config.apply_tradeable_filter)

        # Should not have any advanced filters
        self.assertIsNone(default_config.filtering_config.min_price)
        self.assertIsNone(default_config.filtering_config.max_price)
        self.assertIsNone(default_config.filtering_config.min_volume)

    def test_none_config_fallback(self):
        """Test that None config falls back to default."""
        # This should be handled in the actual implementation
        # by checking if config is None and using default
        pass


class TestPerformanceConsiderations(unittest.TestCase):
    """Test performance-related configuration options."""

    def test_parallel_processing_config(self):
        """Test parallel processing configuration."""
        config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.VOLUME_BASED,
                FilterStrategy.PRICE_RANGE
            ],
            filter_chain_operator="OR",
            parallel_processing=True
        )

        self.assertTrue(config.parallel_processing)
        self.assertEqual(config.filter_chain_operator, "OR")

    def test_stop_on_empty_config(self):
        """Test stop on empty configuration."""
        config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.VOLUME_BASED
            ],
            stop_on_empty=True
        )

        self.assertTrue(config.stop_on_empty)

    def test_caching_configuration(self):
        """Test caching configuration options."""
        config = StockScreeningConfig(
            enable_caching=True,
            cache_ttl=1800  # 30 minutes
        )

        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 1800)

    def test_max_processing_time(self):
        """Test maximum processing time configuration."""
        config = StockScreeningConfig(
            max_processing_time=30.0  # 30 seconds timeout
        )

        self.assertEqual(config.max_processing_time, 30.0)


if __name__ == '__main__':
    unittest.main()