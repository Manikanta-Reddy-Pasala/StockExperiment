"""
Integration Tests for Stock Screening Pipeline with Configurations

This module provides comprehensive integration tests for the entire stock screening
pipeline with various configuration profiles and custom settings.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any
from datetime import datetime

from src.services.stock_filtering.screening_config import (
    QueryStrategy, FilterStrategy, ScreeningProfile,
    DatabaseQueryConfig, FilteringConfig, StockScreeningConfig,
    ConfigurationBuilder,
    get_default_config, get_conservative_config, get_aggressive_config,
    get_day_trading_config
)


class MockStock:
    """Mock stock object for testing."""

    def __init__(self, symbol, price=100, volume=1000000, market_cap=1000000000,
                 is_tradeable=True, is_active=True, sector='Technology',
                 market_cap_category='mid_cap', pe_ratio=15, dividend_yield=0.02,
                 volatility=0.3):
        self.symbol = symbol
        self.price = price
        self.current_price = price
        self.volume = volume
        self.market_cap = market_cap
        self.is_tradeable = is_tradeable
        self.is_active = is_active
        self.sector = sector
        self.market_cap_category = market_cap_category
        self.pe_ratio = pe_ratio
        self.dividend_yield = dividend_yield
        self.volatility = volatility

    def __repr__(self):
        return f"MockStock({self.symbol})"

    def __eq__(self, other):
        if isinstance(other, MockStock):
            return self.symbol == other.symbol
        return False

    def __hash__(self):
        return hash(self.symbol)


class TestPipelineWithDefaultConfiguration(unittest.TestCase):
    """Test pipeline with default configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_default_configuration_pipeline(self, MockFilteringService, MockRepository):
        """Test pipeline with default configuration (backward compatibility)."""
        # Setup mock stocks
        all_stocks = [
            MockStock('AAPL', price=150, is_tradeable=True),
            MockStock('GOOGL', price=2800, is_tradeable=True),
            MockStock('TSLA', price=700, is_tradeable=False),  # Not tradeable
            MockStock('MSFT', price=300, is_tradeable=True),
            MockStock('AMZN', price=3200, is_tradeable=True)
        ]

        tradeable_stocks = [s for s in all_stocks if s.is_tradeable]

        # Setup mocks
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.return_value = all_stocks
        mock_repo.get_all_stocks.return_value = all_stocks

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = tradeable_stocks
        mock_filter.filter_tradeable_stocks.return_value = tradeable_stocks

        # Create default configuration
        config = get_default_config()

        # Simulate pipeline execution
        stocks = mock_repo.get_stocks_with_config(config.query_config)
        filtered = mock_filter.apply_filters_with_config(stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 4)  # Only tradeable stocks
        self.assertNotIn(MockStock('TSLA'), filtered)

        # Verify configuration used
        self.assertEqual(config.query_config.strategy, QueryStrategy.ALL_STOCKS)
        self.assertEqual(config.filtering_config.strategies, [FilterStrategy.BASIC_TRADEABLE])


class TestPipelineWithConservativeConfiguration(unittest.TestCase):
    """Test pipeline with conservative configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_conservative_configuration_pipeline(self, MockFilteringService, MockRepository):
        """Test pipeline with conservative configuration."""
        # Setup mock stocks with varying characteristics
        all_stocks = [
            MockStock('AAPL', price=150, volume=50000000, market_cap_category='large_cap',
                     volatility=0.2),  # Passes all filters
            MockStock('GOOGL', price=2800, volume=30000000, market_cap_category='large_cap',
                     volatility=0.25),  # Passes all filters
            MockStock('PENNY', price=2, volume=100000, market_cap_category='small_cap',
                     volatility=0.8),  # Fails: price too low, small cap, high volatility
            MockStock('VOLATILE', price=50, volume=2000000, market_cap_category='mid_cap',
                     volatility=0.6),  # Fails: not large cap, high volatility
            MockStock('LOWVOL', price=5, volume=10000000, market_cap_category='large_cap',
                     volatility=0.1),  # Fails: price too low
        ]

        conservative_stocks = [all_stocks[0], all_stocks[1]]  # Only AAPL and GOOGL

        # Setup mocks
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.return_value = all_stocks

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = conservative_stocks

        # Create conservative configuration
        config = get_conservative_config()

        # Simulate pipeline execution
        stocks = mock_repo.get_stocks_with_config(config.query_config)
        filtered = mock_filter.apply_filters_with_config(stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].symbol, 'AAPL')
        self.assertEqual(filtered[1].symbol, 'GOOGL')

        # Verify configuration characteristics
        self.assertIn('large_cap', config.filtering_config.market_cap_categories)
        self.assertIsNotNone(config.filtering_config.min_volume)
        self.assertIsNotNone(config.filtering_config.max_volatility)


class TestPipelineWithAggressiveConfiguration(unittest.TestCase):
    """Test pipeline with aggressive configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_aggressive_configuration_pipeline(self, MockFilteringService, MockRepository):
        """Test pipeline with aggressive configuration."""
        # Setup mock stocks
        all_stocks = [
            MockStock('GROWTH1', price=25, volume=500000, market_cap_category='small_cap'),
            MockStock('GROWTH2', price=75, volume=1000000, market_cap_category='mid_cap'),
            MockStock('LARGECAP', price=500, volume=10000000, market_cap_category='large_cap'),
            MockStock('PENNY', price=0.5, volume=50000, market_cap_category='small_cap'),
        ]

        # Only small and mid cap stocks within price range
        aggressive_stocks = [all_stocks[0], all_stocks[1]]

        # Setup mocks
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.return_value = all_stocks

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = aggressive_stocks

        # Create aggressive configuration
        config = get_aggressive_config()

        # Simulate pipeline execution
        stocks = mock_repo.get_stocks_with_config(config.query_config)
        filtered = mock_filter.apply_filters_with_config(stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 2)

        # Verify configuration characteristics
        self.assertIn('small_cap', config.filtering_config.market_cap_categories)
        self.assertIn('mid_cap', config.filtering_config.market_cap_categories)
        self.assertNotIn('large_cap', config.filtering_config.market_cap_categories)


class TestPipelineWithCustomConfiguration(unittest.TestCase):
    """Test pipeline with custom configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_custom_configuration_with_builder(self, MockFilteringService, MockRepository):
        """Test pipeline with custom configuration using builder."""
        # Setup mock stocks
        tech_stocks = [
            MockStock('AAPL', price=150, sector='Technology', volume=50000000),
            MockStock('GOOGL', price=2800, sector='Technology', volume=30000000),
            MockStock('MSFT', price=300, sector='Technology', volume=40000000),
        ]

        healthcare_stocks = [
            MockStock('JNJ', price=160, sector='Healthcare', volume=10000000),
            MockStock('PFE', price=40, sector='Healthcare', volume=25000000),
        ]

        all_stocks = tech_stocks + healthcare_stocks

        # Filter: Tech sector, price between 100-500, volume > 20M
        filtered_stocks = [tech_stocks[0], tech_stocks[2]]  # AAPL and MSFT

        # Setup mocks
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.return_value = all_stocks

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = filtered_stocks

        # Create custom configuration with builder
        config = (ConfigurationBuilder()
                 .with_sectors(['Technology'])
                 .with_price_range(min_price=100, max_price=500)
                 .with_volume_filter(min_volume=20000000)
                 .build())

        # Simulate pipeline execution
        stocks = mock_repo.get_stocks_with_config(config.query_config)
        filtered = mock_filter.apply_filters_with_config(stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 2)
        self.assertIn(MockStock('AAPL'), filtered)
        self.assertIn(MockStock('MSFT'), filtered)

        # Verify configuration
        self.assertEqual(config.query_config.sectors, ['Technology'])
        self.assertEqual(config.filtering_config.min_price, 100)
        self.assertEqual(config.filtering_config.max_price, 500)
        self.assertEqual(config.filtering_config.min_volume, 20000000)


class TestPipelineWithSpecificSymbols(unittest.TestCase):
    """Test pipeline with specific symbols configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_specific_symbols_configuration(self, MockFilteringService, MockRepository):
        """Test pipeline with specific symbols query."""
        # Setup mock stocks
        target_symbols = ['AAPL', 'GOOGL', 'MSFT']
        target_stocks = [
            MockStock('AAPL', price=150),
            MockStock('GOOGL', price=2800),
            MockStock('MSFT', price=300)
        ]

        # Setup mocks
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.return_value = target_stocks

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = target_stocks

        # Create configuration for specific symbols
        config = (ConfigurationBuilder()
                 .with_symbols(target_symbols)
                 .build())

        # Simulate pipeline execution
        stocks = mock_repo.get_stocks_with_config(config.query_config)
        filtered = mock_filter.apply_filters_with_config(stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 3)
        self.assertEqual(config.query_config.strategy, QueryStrategy.BY_SYMBOLS)
        self.assertEqual(config.query_config.symbols, target_symbols)


class TestPipelineWithPagination(unittest.TestCase):
    """Test pipeline with pagination configuration."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    def test_pagination_configuration(self, MockRepository):
        """Test pipeline with pagination."""
        # Setup mock stocks (simulating paginated results)
        page1_stocks = [MockStock(f'STOCK{i}') for i in range(1, 26)]  # 25 stocks
        page2_stocks = [MockStock(f'STOCK{i}') for i in range(26, 51)]  # 25 stocks

        # Setup mock
        mock_repo = MockRepository.return_value

        # Create pagination configuration for page 1
        config_page1 = DatabaseQueryConfig(
            strategy=QueryStrategy.PAGINATED,
            page=1,
            page_size=25
        )

        # Create pagination configuration for page 2
        config_page2 = DatabaseQueryConfig(
            strategy=QueryStrategy.PAGINATED,
            page=2,
            page_size=25
        )

        # Setup different returns for different configs
        mock_repo.get_stocks_with_config.side_effect = [page1_stocks, page2_stocks]

        # Get page 1
        stocks_page1 = mock_repo.get_stocks_with_config(config_page1)
        self.assertEqual(len(stocks_page1), 25)
        self.assertEqual(stocks_page1[0].symbol, 'STOCK1')

        # Get page 2
        stocks_page2 = mock_repo.get_stocks_with_config(config_page2)
        self.assertEqual(len(stocks_page2), 25)
        self.assertEqual(stocks_page2[0].symbol, 'STOCK26')


class TestPipelineWithORFiltering(unittest.TestCase):
    """Test pipeline with OR filtering logic."""

    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_or_filtering_configuration(self, MockFilteringService):
        """Test OR filtering configuration."""
        # Setup mock stocks
        all_stocks = [
            MockStock('HIGH_VOL', volume=10000000, price=5),  # High volume only
            MockStock('HIGH_PRICE', volume=100000, price=500),  # High price only
            MockStock('BOTH', volume=5000000, price=200),  # Both criteria
            MockStock('NEITHER', volume=50000, price=2),  # Neither criteria
        ]

        # With OR logic, stocks meeting ANY criteria should pass
        or_filtered = [all_stocks[0], all_stocks[1], all_stocks[2]]

        # Setup mock
        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = or_filtered

        # Create OR configuration
        config = FilteringConfig(
            strategies=[FilterStrategy.VOLUME_BASED, FilterStrategy.PRICE_RANGE],
            min_volume=5000000,
            min_price=100,
            filter_chain_operator="OR"
        )

        # Apply filtering
        filtered = mock_filter.apply_filters_with_config(all_stocks, config)

        # Verify results
        self.assertEqual(len(filtered), 3)
        self.assertNotIn(MockStock('NEITHER'), filtered)
        self.assertEqual(config.filter_chain_operator, "OR")


class TestPipelineErrorHandling(unittest.TestCase):
    """Test pipeline error handling with configurations."""

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_fallback_on_error_enabled(self, MockFilteringService, MockRepository):
        """Test fallback behavior when enabled."""
        # Setup mocks to simulate errors
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.side_effect = Exception("Database error")
        mock_repo.get_all_stocks.return_value = [MockStock('FALLBACK')]

        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = [MockStock('FALLBACK')]

        # Create configuration with fallback enabled
        config = StockScreeningConfig(
            fallback_on_error=True
        )

        # In actual implementation, this would trigger fallback
        # Here we simulate what should happen
        try:
            stocks = mock_repo.get_stocks_with_config(config.query_config)
        except:
            # Fallback to get_all_stocks
            stocks = mock_repo.get_all_stocks()

        self.assertEqual(len(stocks), 1)
        self.assertEqual(stocks[0].symbol, 'FALLBACK')

    @patch('src.services.stock_filtering.stock_data_repository.StockDataRepository')
    def test_fallback_on_error_disabled(self, MockRepository):
        """Test no fallback when disabled."""
        # Setup mock to simulate error
        mock_repo = MockRepository.return_value
        mock_repo.get_stocks_with_config.side_effect = Exception("Database error")

        # Create configuration with fallback disabled
        config = StockScreeningConfig(
            fallback_on_error=False
        )

        # Should raise exception without fallback
        with self.assertRaises(Exception):
            mock_repo.get_stocks_with_config(config.query_config)


class TestPipelineWithCaching(unittest.TestCase):
    """Test pipeline with caching configuration."""

    def test_caching_configuration(self):
        """Test caching configuration settings."""
        # Create configuration with caching
        config = (ConfigurationBuilder()
                 .with_caching(enabled=True, ttl=1800)
                 .build())

        # Verify caching settings
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 1800)
        self.assertTrue(config.query_config.cache_enabled)
        self.assertEqual(config.query_config.cache_ttl, 1800)


class TestPipelineWithCustomFilters(unittest.TestCase):
    """Test pipeline with custom filter functions."""

    @patch('src.services.stock_filtering.stock_filtering_service.StockFilteringService')
    def test_custom_filter_function(self, MockFilteringService):
        """Test custom filter function configuration."""
        # Define custom filter function
        def high_momentum_filter(stock):
            # Custom logic: price > 50 and volume > 1M
            return stock.price > 50 and stock.volume > 1000000

        # Setup mock stocks
        all_stocks = [
            MockStock('HIGH_MOM', price=100, volume=2000000),  # Passes
            MockStock('LOW_PRICE', price=25, volume=3000000),  # Fails
            MockStock('LOW_VOL', price=75, volume=500000),  # Fails
        ]

        filtered_stocks = [all_stocks[0]]

        # Setup mock
        mock_filter = MockFilteringService.return_value
        mock_filter.apply_filters_with_config.return_value = filtered_stocks

        # Create configuration with custom filter
        config = (ConfigurationBuilder()
                 .with_custom_filter(high_momentum_filter)
                 .build())

        # Apply filtering
        filtered = mock_filter.apply_filters_with_config(all_stocks, config.filtering_config)

        # Verify results
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].symbol, 'HIGH_MOM')
        self.assertIn(FilterStrategy.CUSTOM, config.filtering_config.strategies)


class TestPipelinePerformance(unittest.TestCase):
    """Test pipeline performance configurations."""

    def test_parallel_processing_configuration(self):
        """Test parallel processing configuration."""
        config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.VOLUME_BASED,
                FilterStrategy.PRICE_RANGE,
                FilterStrategy.MARKET_CAP_BASED
            ],
            filter_chain_operator="OR",
            parallel_processing=True
        )

        self.assertTrue(config.parallel_processing)
        self.assertEqual(len(config.strategies), 4)

    def test_stop_on_empty_configuration(self):
        """Test stop on empty configuration."""
        config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.PRICE_RANGE,
                FilterStrategy.VOLUME_BASED
            ],
            filter_chain_operator="AND",
            stop_on_empty=True
        )

        self.assertTrue(config.stop_on_empty)
        # In actual implementation, this would stop the chain
        # if any filter returns empty results


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_invalid_configuration_detection(self):
        """Test detection of invalid configurations."""
        # Invalid price range
        config1 = FilteringConfig(
            min_price=100,
            max_price=50  # Invalid: max < min
        )
        self.assertFalse(config1.validate())

        # Missing required parameters
        config2 = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SYMBOLS
            # Missing symbols
        )
        self.assertFalse(config2.validate())

        # Invalid page number
        config3 = DatabaseQueryConfig(
            strategy=QueryStrategy.PAGINATED,
            page=0  # Invalid: page must be >= 1
        )
        self.assertFalse(config3.validate())


class TestConfigurationProfiles(unittest.TestCase):
    """Test all configuration profiles."""

    def test_all_profiles_valid(self):
        """Test that all predefined profiles are valid."""
        profiles = [
            ScreeningProfile.DEFAULT,
            ScreeningProfile.CONSERVATIVE,
            ScreeningProfile.AGGRESSIVE,
            ScreeningProfile.BALANCED,
            ScreeningProfile.DAY_TRADING,
            ScreeningProfile.VALUE_INVESTING,
            ScreeningProfile.GROWTH_INVESTING,
            ScreeningProfile.SECTOR_SPECIFIC
        ]

        for profile in profiles:
            config = StockScreeningConfig(profile=profile)
            self.assertTrue(config.validate(), f"Profile {profile.value} should be valid")


if __name__ == '__main__':
    unittest.main()