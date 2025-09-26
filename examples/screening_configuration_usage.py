"""
Usage Examples for Stock Screening Configuration System

This module demonstrates various ways to use the configurable stock screening pipeline.
"""

from src.services.stock_filtering.screening_config import (
    QueryStrategy, FilterStrategy, ScreeningProfile,
    DatabaseQueryConfig, FilteringConfig, StockScreeningConfig,
    ConfigurationBuilder,
    get_default_config, get_conservative_config, get_aggressive_config,
    get_day_trading_config, get_value_investing_config, get_custom_config
)
from src.services.implementations.fyers_suggested_stocks_provider import FyersSuggestedStocksProvider


def example_default_configuration():
    """Example: Using default configuration (backward compatible)."""
    print("\n=== EXAMPLE 1: Default Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Use default configuration (same as current behavior)
    config = get_default_config()

    print(f"Profile: {config.profile.value}")
    print(f"Query Strategy: {config.query_config.strategy.value}")
    print(f"Filter Strategies: {[s.value for s in config.filtering_config.strategies]}")

    # Get suggested stocks with default configuration
    # This maintains backward compatibility - no config means default behavior
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=10
        # config parameter is optional, defaults to None
    )

    return results


def example_conservative_configuration():
    """Example: Using conservative profile for risk-averse investors."""
    print("\n=== EXAMPLE 2: Conservative Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Use conservative configuration
    config = get_conservative_config()

    print(f"Profile: {config.profile.value}")
    print(f"Market Cap Categories: {config.filtering_config.market_cap_categories}")
    print(f"Min Volume: {config.filtering_config.min_volume}")
    print(f"Max Volatility: {config.filtering_config.max_volatility}")

    # Get suggested stocks with conservative filters
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=10,
        config=config
    )

    return results


def example_aggressive_configuration():
    """Example: Using aggressive profile for growth-focused traders."""
    print("\n=== EXAMPLE 3: Aggressive Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Use aggressive configuration
    config = get_aggressive_config()

    print(f"Profile: {config.profile.value}")
    print(f"Market Cap Categories: {config.filtering_config.market_cap_categories}")
    print(f"Price Range: ${config.filtering_config.min_price} - ${config.filtering_config.max_price}")

    # Get suggested stocks with aggressive filters
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=20,
        config=config
    )

    return results


def example_day_trading_configuration():
    """Example: Using day trading profile for active traders."""
    print("\n=== EXAMPLE 4: Day Trading Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Use day trading configuration
    config = get_day_trading_config()

    print(f"Profile: {config.profile.value}")
    print(f"Min Volume: {config.filtering_config.min_volume}")
    print(f"Min Avg Volume: {config.filtering_config.min_avg_volume}")
    print(f"Price Range: ${config.filtering_config.min_price} - ${config.filtering_config.max_price}")

    # Get suggested stocks for day trading
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=15,
        config=config
    )

    return results


def example_custom_configuration_with_builder():
    """Example: Creating custom configuration with builder pattern."""
    print("\n=== EXAMPLE 5: Custom Configuration with Builder ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Build custom configuration
    config = (ConfigurationBuilder()
             .with_sectors(['Technology', 'Healthcare'])  # Focus on specific sectors
             .with_price_range(min_price=50, max_price=500)  # Price constraints
             .with_volume_filter(min_volume=2000000)  # Minimum liquidity
             .with_market_cap_filter(
                 categories=['large_cap', 'mid_cap'],
                 min_cap=5000000000  # $5B minimum
             )
             .with_caching(enabled=True, ttl=600)  # Enable 10-minute cache
             .build())

    print(f"Sectors: {config.query_config.sectors}")
    print(f"Price Range: ${config.filtering_config.min_price} - ${config.filtering_config.max_price}")
    print(f"Min Volume: {config.filtering_config.min_volume}")
    print(f"Market Cap Categories: {config.filtering_config.market_cap_categories}")
    print(f"Caching Enabled: {config.enable_caching} (TTL: {config.cache_ttl}s)")

    # Get suggested stocks with custom configuration
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=25,
        config=config
    )

    return results


def example_specific_symbols_configuration():
    """Example: Screening specific symbols only."""
    print("\n=== EXAMPLE 6: Specific Symbols Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Build configuration for specific symbols
    watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']

    config = (ConfigurationBuilder()
             .with_symbols(watchlist)  # Only analyze these symbols
             .with_price_range(min_price=100, max_price=3000)
             .with_volume_filter(min_volume=10000000)  # High liquidity only
             .build())

    print(f"Symbols to analyze: {config.query_config.symbols}")
    print(f"Query Strategy: {config.query_config.strategy.value}")

    # Get analysis for specific symbols
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=10,
        config=config
    )

    return results


def example_value_investing_configuration():
    """Example: Value investing configuration focusing on fundamentals."""
    print("\n=== EXAMPLE 7: Value Investing Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Use value investing configuration
    config = get_value_investing_config()

    print(f"Profile: {config.profile.value}")
    print(f"Max P/E Ratio: {config.filtering_config.max_pe_ratio}")
    print(f"Min Dividend Yield: {config.filtering_config.min_dividend_yield * 100}%")
    print(f"Min Market Cap: ${config.filtering_config.min_market_cap:,.0f}")

    # Get value stocks
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=20,
        config=config
    )

    return results


def example_sector_specific_configuration():
    """Example: Sector-specific screening."""
    print("\n=== EXAMPLE 8: Sector-Specific Configuration ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Create configuration for technology sector
    config = get_custom_config(
        query_strategy=QueryStrategy.BY_SECTOR,
        filter_strategies=[
            FilterStrategy.BASIC_TRADEABLE,
            FilterStrategy.SECTOR_BASED,
            FilterStrategy.VOLUME_BASED,
            FilterStrategy.PRICE_RANGE
        ],
        sectors=['Technology'],
        included_sectors=['Technology'],
        min_volume=1000000,
        min_price=20,
        max_price=1000
    )

    print(f"Target Sectors: {config.query_config.sectors}")
    print(f"Filter Strategies: {[s.value for s in config.filtering_config.strategies]}")

    # Get technology sector stocks
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=15,
        config=config
    )

    return results


def example_custom_filter_function():
    """Example: Using custom filter functions."""
    print("\n=== EXAMPLE 9: Custom Filter Function ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Define custom filter functions
    def momentum_filter(stock):
        """Filter for stocks with strong momentum."""
        # Custom logic: high volume and price above certain threshold
        return (hasattr(stock, 'volume') and stock.volume > 5000000 and
                hasattr(stock, 'price') and stock.price > 50)

    def volatility_filter(stock):
        """Filter for stocks with moderate volatility."""
        # Custom logic: volatility in specific range
        return (hasattr(stock, 'volatility') and
                0.2 <= getattr(stock, 'volatility', 0) <= 0.5)

    # Build configuration with custom filters
    config = (ConfigurationBuilder()
             .with_custom_filter(momentum_filter)
             .with_custom_filter(volatility_filter)
             .build())

    print(f"Number of custom filters: {len(config.filtering_config.custom_filters)}")
    print(f"Filter Strategies: {[s.value for s in config.filtering_config.strategies]}")

    # Get stocks matching custom criteria
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=10,
        config=config
    )

    return results


def example_or_filtering_logic():
    """Example: Using OR logic for filters instead of AND."""
    print("\n=== EXAMPLE 10: OR Filtering Logic ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Create configuration with OR logic
    # Stocks need to meet ANY of the criteria, not all
    config = FilteringConfig(
        strategies=[
            FilterStrategy.VOLUME_BASED,
            FilterStrategy.PRICE_RANGE,
            FilterStrategy.MARKET_CAP_BASED
        ],
        min_volume=10000000,  # High volume stocks
        min_price=500,  # OR expensive stocks
        market_cap_categories=['large_cap'],  # OR large cap stocks
        filter_chain_operator="OR"  # Use OR logic
    )

    full_config = StockScreeningConfig(
        profile=ScreeningProfile.CUSTOM,
        filtering_config=config
    )

    print(f"Filter Chain Operator: {config.filter_chain_operator}")
    print("Stocks meeting ANY of these criteria will pass:")
    print(f"  - Volume > {config.min_volume}")
    print(f"  - Price > ${config.min_price}")
    print(f"  - Market Cap: {config.market_cap_categories}")

    # Get stocks meeting any criteria
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=30,
        config=full_config
    )

    return results


def example_paginated_screening():
    """Example: Paginated screening for large datasets."""
    print("\n=== EXAMPLE 11: Paginated Screening ===")

    # Create provider instance
    provider = FyersSuggestedStocksProvider()

    # Create paginated configuration
    page_size = 25
    current_page = 1

    config = StockScreeningConfig(
        profile=ScreeningProfile.CUSTOM,
        query_config=DatabaseQueryConfig(
            strategy=QueryStrategy.PAGINATED,
            page=current_page,
            page_size=page_size
        ),
        filtering_config=FilteringConfig(
            strategies=[FilterStrategy.BASIC_TRADEABLE],
            min_price=10,
            max_price=500
        )
    )

    print(f"Page: {config.query_config.page}")
    print(f"Page Size: {config.query_config.page_size}")
    print(f"Query Strategy: {config.query_config.strategy.value}")

    # Get first page of results
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=page_size,
        config=config
    )

    # To get next page, update configuration
    config.query_config.page = 2
    next_page_results = provider.get_suggested_stocks(
        user_id=1,
        limit=page_size,
        config=config
    )

    return results


def example_configuration_from_dict():
    """Example: Creating configuration from dictionary (e.g., from JSON/database)."""
    print("\n=== EXAMPLE 12: Configuration from Dictionary ===")

    # Configuration dictionary (could be from JSON, database, or API)
    config_dict = {
        'profile': 'custom',
        'query_config': {
            'strategy': 'with_filters',
            'filters': {
                'is_tradeable': True,
                'is_active': True,
                'min_price': 20.0,
                'max_price': 200.0
            },
            'limit': 100
        },
        'filtering_config': {
            'strategies': ['basic_tradeable', 'price_range', 'volume_based'],
            'min_price': 20.0,
            'max_price': 200.0,
            'min_volume': 500000,
            'apply_tradeable_filter': True
        },
        'enable_caching': True,
        'cache_ttl': 300,
        'enable_statistics': True,
        'description': 'Custom configuration from dictionary'
    }

    # Create configuration from dictionary
    config = StockScreeningConfig.from_dict(config_dict)

    print(f"Configuration loaded: {config.description}")
    print(f"Profile: {config.profile.value}")
    print(f"Query Strategy: {config.query_config.strategy.value}")
    print(f"Filter Strategies: {[s.value for s in config.filtering_config.strategies]}")

    # Use the configuration
    provider = FyersSuggestedStocksProvider()
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=20,
        config=config
    )

    return results


def example_performance_optimized_configuration():
    """Example: Performance-optimized configuration."""
    print("\n=== EXAMPLE 13: Performance-Optimized Configuration ===")

    # Create configuration optimized for performance
    config = StockScreeningConfig(
        profile=ScreeningProfile.CUSTOM,
        query_config=DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters={
                'is_tradeable': True,
                'is_active': True,
                'min_volume': 1000000  # Pre-filter at database level
            },
            limit=500,  # Limit initial query
            cache_enabled=True,
            cache_ttl=600
        ),
        filtering_config=FilteringConfig(
            strategies=[
                FilterStrategy.PRICE_RANGE,
                FilterStrategy.VOLUME_BASED
            ],
            min_price=10,
            max_price=1000,
            min_volume=2000000,
            filter_chain_operator="AND",
            stop_on_empty=True,  # Stop early if no results
            parallel_processing=False  # Use sequential for AND logic
        ),
        enable_caching=True,
        cache_ttl=600,
        max_processing_time=30.0,  # 30 second timeout
        fallback_on_error=True
    )

    print("Performance Optimizations:")
    print(f"  - Database pre-filtering: Yes")
    print(f"  - Query limit: {config.query_config.limit}")
    print(f"  - Caching enabled: {config.enable_caching} (TTL: {config.cache_ttl}s)")
    print(f"  - Stop on empty: {config.filtering_config.stop_on_empty}")
    print(f"  - Max processing time: {config.max_processing_time}s")
    print(f"  - Fallback on error: {config.fallback_on_error}")

    # Use optimized configuration
    provider = FyersSuggestedStocksProvider()
    results = provider.get_suggested_stocks(
        user_id=1,
        limit=50,
        config=config
    )

    return results


def main():
    """Run all examples."""
    print("=" * 70)
    print("STOCK SCREENING CONFIGURATION SYSTEM - USAGE EXAMPLES")
    print("=" * 70)

    # Note: These examples show the configuration setup
    # Actual execution would require database connection and data

    examples = [
        ("Default Configuration", example_default_configuration),
        ("Conservative Configuration", example_conservative_configuration),
        ("Aggressive Configuration", example_aggressive_configuration),
        ("Day Trading Configuration", example_day_trading_configuration),
        ("Custom Builder Configuration", example_custom_configuration_with_builder),
        ("Specific Symbols", example_specific_symbols_configuration),
        ("Value Investing", example_value_investing_configuration),
        ("Sector-Specific", example_sector_specific_configuration),
        ("Custom Filter Functions", example_custom_filter_function),
        ("OR Filtering Logic", example_or_filtering_logic),
        ("Paginated Screening", example_paginated_screening),
        ("Configuration from Dict", example_configuration_from_dict),
        ("Performance Optimized", example_performance_optimized_configuration)
    ]

    for name, example_func in examples:
        try:
            # Note: These would actually execute if database is available
            # For demonstration, we just show the configuration setup
            example_func()
            print(f"✅ {name} configuration created successfully\n")
        except Exception as e:
            print(f"❌ {name} example failed: {e}\n")

    print("=" * 70)
    print("All configuration examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()