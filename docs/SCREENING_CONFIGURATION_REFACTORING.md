# Stock Screening Configuration System - Refactoring Documentation

## Overview

This document describes the comprehensive refactoring of Stage 1 (database query) and Stage 2 (filtering) in the stock filtering pipeline to make them configurable parameters. The refactoring maintains 100% backward compatibility while adding powerful configuration capabilities.

## Key Changes

### 1. New Configuration Module (`screening_config.py`)

Created a comprehensive configuration system with the following components:

- **`DatabaseQueryConfig`**: Configures Stage 1 (database queries)
- **`FilteringConfig`**: Configures Stage 2 (filtering operations)
- **`StockScreeningConfig`**: Main configuration class combining both stages
- **`ConfigurationBuilder`**: Fluent API for building configurations
- **Pre-defined Profiles**: Conservative, Aggressive, Day Trading, Value Investing, etc.

### 2. Extended Repository (`stock_data_repository.py`)

Added new method `get_stocks_with_config()` that:
- Accepts `DatabaseQueryConfig` parameter
- Routes to appropriate query method based on strategy
- Supports multiple query strategies (ALL_STOCKS, BY_SYMBOLS, BY_SECTOR, etc.)
- Applies additional configuration parameters (limit, offset, ordering)

### 3. Enhanced Filtering Service (`stock_filtering_service.py`)

Added new method `apply_filters_with_config()` that:
- Accepts `FilteringConfig` parameter
- Applies multiple filter strategies in sequence or parallel
- Supports AND/OR logic for filter combinations
- Includes new filter types (fundamental, technical, custom)

### 4. Updated Main Provider (`fyers_suggested_stocks_provider.py`)

Modified methods to accept optional configuration:
- `_get_filtered_stocks_from_database(user_id, config=None)`
- `get_suggested_stocks(..., config=None)`
- Falls back to default configuration when config is None
- Maintains complete backward compatibility

## Configuration Strategies

### Database Query Strategies (Stage 1)

1. **ALL_STOCKS**: Fetch all stocks from database (default)
2. **TRADEABLE_ONLY**: Fetch only tradeable stocks
3. **BY_SYMBOLS**: Fetch specific symbols
4. **BY_SECTOR**: Fetch stocks from specific sectors
5. **BY_MARKET_CAP**: Fetch by market cap categories
6. **WITH_FILTERS**: Apply filters at database level
7. **PAGINATED**: Support pagination for large datasets
8. **CUSTOM**: Execute custom query function

### Filtering Strategies (Stage 2)

1. **BASIC_TRADEABLE**: Filter for tradeable stocks (default)
2. **MARKET_CAP_BASED**: Filter by market capitalization
3. **VOLUME_BASED**: Filter by trading volume
4. **PRICE_RANGE**: Filter by price range
5. **SECTOR_BASED**: Filter by sectors
6. **FUNDAMENTAL**: Filter by P/E ratio, dividend yield
7. **TECHNICAL**: Filter by volatility and technical indicators
8. **CUSTOM**: Apply custom filter functions

## Pre-Configured Profiles

### 1. DEFAULT
- **Purpose**: Maintain current behavior
- **Query**: All stocks
- **Filters**: Basic tradeable filter only

### 2. CONSERVATIVE
- **Purpose**: Risk-averse investors
- **Query**: Large cap stocks only
- **Filters**: High volume, low volatility, minimum price

### 3. AGGRESSIVE
- **Purpose**: Growth-focused traders
- **Query**: Small/mid cap stocks
- **Filters**: Lower volume threshold, wider price range

### 4. DAY_TRADING
- **Purpose**: Active day traders
- **Query**: High volume stocks
- **Filters**: Very high liquidity, moderate price range

### 5. VALUE_INVESTING
- **Purpose**: Long-term value investors
- **Query**: All stocks
- **Filters**: Low P/E, dividend yield, large market cap

### 6. GROWTH_INVESTING
- **Purpose**: Growth stock investors
- **Query**: Mid/large cap stocks
- **Filters**: Higher P/E acceptable, growth metrics

### 7. SECTOR_SPECIFIC
- **Purpose**: Sector-focused strategies
- **Query**: Specific sectors
- **Filters**: Sector-based filtering

## Usage Examples

### Using Default Configuration (Backward Compatible)
```python
# No configuration = default behavior
provider = FyersSuggestedStocksProvider()
results = provider.get_suggested_stocks(user_id=1, limit=50)
```

### Using Pre-Configured Profile
```python
from src.services.stock_filtering.screening_config import get_conservative_config

config = get_conservative_config()
results = provider.get_suggested_stocks(user_id=1, limit=50, config=config)
```

### Building Custom Configuration
```python
from src.services.stock_filtering.screening_config import ConfigurationBuilder

config = (ConfigurationBuilder()
         .with_sectors(['Technology', 'Healthcare'])
         .with_price_range(min_price=50, max_price=500)
         .with_volume_filter(min_volume=1000000)
         .with_market_cap_filter(categories=['large_cap'])
         .build())

results = provider.get_suggested_stocks(user_id=1, limit=50, config=config)
```

### Configuration from Dictionary
```python
config_dict = {
    'profile': 'custom',
    'query_config': {
        'strategy': 'by_symbols',
        'symbols': ['AAPL', 'GOOGL', 'MSFT']
    },
    'filtering_config': {
        'strategies': ['price_range', 'volume_based'],
        'min_price': 100,
        'max_price': 500,
        'min_volume': 1000000
    }
}

config = StockScreeningConfig.from_dict(config_dict)
```

## Advanced Features

### 1. Filter Chaining
- **AND Logic**: All filters must pass (default)
- **OR Logic**: Any filter can pass
- Configurable via `filter_chain_operator`

### 2. Performance Optimization
- **Parallel Processing**: For OR filters
- **Stop on Empty**: Early termination
- **Caching**: Query and result caching
- **Database Pre-filtering**: Reduce data transfer

### 3. Error Handling
- **Fallback on Error**: Use default behavior on failure
- **Validation**: Configuration validation before execution
- **Logging**: Detailed logging of configuration usage

### 4. Custom Extensions
- **Custom Query Functions**: Inject custom database queries
- **Custom Filter Functions**: Add domain-specific filters
- **Configuration Profiles**: Create reusable configurations

## Testing

### Test Coverage
- **Unit Tests**: 50+ tests for configuration classes
- **Integration Tests**: 15+ tests for pipeline integration
- **Backward Compatibility**: Verified default behavior unchanged
- **Profile Validation**: All profiles tested and validated

### Test Files
- `test_screening_configuration.py`: Configuration system tests
- `test_screening_pipeline_integration.py`: Integration tests

## Migration Guide

### For Existing Code
No changes required! The system is 100% backward compatible:
- Calls without config parameter use default configuration
- Default configuration maintains exact current behavior

### For New Features
To use new configuration features:
1. Import configuration classes
2. Create or select configuration
3. Pass configuration to `get_suggested_stocks()`

## Benefits

### 1. Flexibility
- Configure database queries without code changes
- Adjust filtering criteria dynamically
- Support different trading strategies

### 2. Performance
- Pre-filter at database level
- Parallel filter processing
- Caching support
- Early termination optimization

### 3. Maintainability
- Clear separation of concerns
- Configuration validation
- Comprehensive logging
- Easy to test and mock

### 4. Extensibility
- Add new query strategies easily
- Create custom filter functions
- Define new configuration profiles
- Support configuration from external sources

## Future Enhancements

### Potential Improvements
1. **Configuration Persistence**: Save/load configurations from database
2. **User Preferences**: User-specific configuration profiles
3. **Dynamic Configuration**: Runtime configuration adjustments
4. **Configuration UI**: Web interface for configuration management
5. **Performance Metrics**: Track configuration performance
6. **A/B Testing**: Compare different configurations

### API Extensions
1. REST endpoint for configuration management
2. Configuration templates library
3. Configuration recommendation engine
4. Historical configuration tracking

## Conclusion

This refactoring successfully transforms the hardcoded Stage 1 and Stage 2 operations into a flexible, configurable system while maintaining complete backward compatibility. The new configuration system provides powerful customization options for different trading strategies and use cases, with comprehensive testing ensuring reliability and correctness.