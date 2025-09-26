"""
Stock Screening Configuration Module

Provides configurable parameters for the stock screening pipeline, including
database query configuration and filtering criteria configuration.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Database query strategies for fetching stocks."""
    ALL_STOCKS = "all_stocks"
    TRADEABLE_ONLY = "tradeable_only"
    BY_SYMBOLS = "by_symbols"
    BY_SECTOR = "by_sector"
    BY_MARKET_CAP = "by_market_cap"
    WITH_FILTERS = "with_filters"
    PAGINATED = "paginated"
    CUSTOM = "custom"


class FilterStrategy(Enum):
    """Filtering strategies for stock selection."""
    BASIC_TRADEABLE = "basic_tradeable"
    MARKET_CAP_BASED = "market_cap_based"
    VOLUME_BASED = "volume_based"
    PRICE_RANGE = "price_range"
    SECTOR_BASED = "sector_based"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class ScreeningProfile(Enum):
    """Pre-configured screening profiles for different trading styles."""
    DEFAULT = "default"  # Current behavior - all stocks with tradeable filter
    CONSERVATIVE = "conservative"  # Large cap, high volume, established companies
    AGGRESSIVE = "aggressive"  # Small/mid cap, growth potential
    BALANCED = "balanced"  # Mix of market caps with moderate filters
    DAY_TRADING = "day_trading"  # High volume, liquid stocks
    VALUE_INVESTING = "value_investing"  # Focus on fundamentals
    GROWTH_INVESTING = "growth_investing"  # Focus on growth metrics
    SECTOR_SPECIFIC = "sector_specific"  # Focus on specific sectors
    CUSTOM = "custom"  # User-defined configuration


@dataclass
class DatabaseQueryConfig:
    """
    Configuration for Stage 1: Database Query.

    Attributes:
        strategy: Query strategy to use
        limit: Maximum number of stocks to fetch (None = no limit)
        offset: Number of stocks to skip (for pagination)
        symbols: Specific symbols to fetch (for BY_SYMBOLS strategy)
        sectors: Sectors to include (for BY_SECTOR strategy)
        market_cap_categories: Market cap categories to include
        filters: Additional filters for WITH_FILTERS strategy
        page: Page number for PAGINATED strategy
        page_size: Items per page for PAGINATED strategy
        order_by: Field to order results by
        order_desc: Whether to order in descending order
        include_inactive: Whether to include inactive stocks
        custom_query_func: Custom query function for CUSTOM strategy
        cache_enabled: Whether to enable query result caching
        cache_ttl: Cache time-to-live in seconds
    """
    strategy: QueryStrategy = QueryStrategy.ALL_STOCKS
    limit: Optional[int] = None
    offset: Optional[int] = None
    symbols: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    market_cap_categories: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 50
    order_by: Optional[str] = None
    order_desc: bool = False
    include_inactive: bool = False
    custom_query_func: Optional[Callable] = None
    cache_enabled: bool = False
    cache_ttl: int = 300  # 5 minutes default

    def validate(self) -> bool:
        """Validate the configuration parameters."""
        try:
            # Validate strategy-specific requirements
            if self.strategy == QueryStrategy.BY_SYMBOLS and not self.symbols:
                raise ValueError("BY_SYMBOLS strategy requires symbols list")

            if self.strategy == QueryStrategy.BY_SECTOR and not self.sectors:
                raise ValueError("BY_SECTOR strategy requires sectors list")

            if self.strategy == QueryStrategy.BY_MARKET_CAP and not self.market_cap_categories:
                raise ValueError("BY_MARKET_CAP strategy requires market_cap_categories")

            if self.strategy == QueryStrategy.WITH_FILTERS and not self.filters:
                raise ValueError("WITH_FILTERS strategy requires filters dictionary")

            if self.strategy == QueryStrategy.CUSTOM and not self.custom_query_func:
                raise ValueError("CUSTOM strategy requires custom_query_func")

            # Validate pagination parameters
            if self.page < 1:
                raise ValueError("Page number must be >= 1")

            if self.page_size < 1:
                raise ValueError("Page size must be >= 1")

            # Validate limit and offset
            if self.limit is not None and self.limit < 0:
                raise ValueError("Limit must be >= 0")

            if self.offset is not None and self.offset < 0:
                raise ValueError("Offset must be >= 0")

            return True

        except Exception as e:
            logger.error(f"Query configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        config_dict['strategy'] = self.strategy.value
        # Remove None values and callable functions
        return {k: v for k, v in config_dict.items()
                if v is not None and not callable(v)}


@dataclass
class FilteringConfig:
    """
    Configuration for Stage 2: Filtering.

    Attributes:
        strategies: List of filtering strategies to apply
        apply_tradeable_filter: Whether to apply tradeable filter
        min_price: Minimum stock price
        max_price: Maximum stock price
        min_volume: Minimum trading volume
        min_avg_volume: Minimum average volume
        min_market_cap: Minimum market capitalization
        max_market_cap: Maximum market capitalization
        market_cap_categories: Allowed market cap categories
        included_sectors: Sectors to include
        excluded_sectors: Sectors to exclude
        min_pe_ratio: Minimum P/E ratio
        max_pe_ratio: Maximum P/E ratio
        min_dividend_yield: Minimum dividend yield
        max_volatility: Maximum volatility threshold
        custom_filters: List of custom filter functions
        filter_chain_operator: How to combine filters ('AND' or 'OR')
        stop_on_empty: Stop filtering if any filter returns empty
        parallel_processing: Enable parallel filter processing
    """
    strategies: List[FilterStrategy] = field(
        default_factory=lambda: [FilterStrategy.BASIC_TRADEABLE]
    )
    apply_tradeable_filter: bool = True
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    min_avg_volume: Optional[int] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    market_cap_categories: Optional[List[str]] = None
    included_sectors: Optional[List[str]] = None
    excluded_sectors: Optional[List[str]] = None
    min_pe_ratio: Optional[float] = None
    max_pe_ratio: Optional[float] = None
    min_dividend_yield: Optional[float] = None
    max_volatility: Optional[float] = None
    custom_filters: Optional[List[Callable]] = None
    filter_chain_operator: str = "AND"  # AND or OR
    stop_on_empty: bool = False
    parallel_processing: bool = False

    def validate(self) -> bool:
        """Validate the configuration parameters."""
        try:
            # Validate price range
            if (self.min_price is not None and self.max_price is not None
                and self.min_price > self.max_price):
                raise ValueError("min_price cannot be greater than max_price")

            # Validate market cap range
            if (self.min_market_cap is not None and self.max_market_cap is not None
                and self.min_market_cap > self.max_market_cap):
                raise ValueError("min_market_cap cannot be greater than max_market_cap")

            # Validate PE ratio range
            if (self.min_pe_ratio is not None and self.max_pe_ratio is not None
                and self.min_pe_ratio > self.max_pe_ratio):
                raise ValueError("min_pe_ratio cannot be greater than max_pe_ratio")

            # Validate filter chain operator
            if self.filter_chain_operator not in ["AND", "OR"]:
                raise ValueError("filter_chain_operator must be 'AND' or 'OR'")

            # Validate numeric values
            for attr in ['min_price', 'max_price', 'min_volume', 'min_avg_volume',
                        'min_market_cap', 'max_market_cap', 'min_pe_ratio',
                        'max_pe_ratio', 'min_dividend_yield', 'max_volatility']:
                value = getattr(self, attr)
                if value is not None and value < 0:
                    raise ValueError(f"{attr} must be >= 0")

            return True

        except Exception as e:
            logger.error(f"Filtering configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        config_dict['strategies'] = [s.value for s in self.strategies]
        # Remove None values and callable functions
        return {k: v for k, v in config_dict.items()
                if v is not None and not callable(v)}


@dataclass
class StockScreeningConfig:
    """
    Main configuration class for the entire stock screening pipeline.

    Attributes:
        profile: Pre-configured screening profile to use
        query_config: Configuration for database query stage
        filtering_config: Configuration for filtering stage
        enable_caching: Enable caching for the entire pipeline
        cache_ttl: Cache time-to-live in seconds
        enable_logging: Enable detailed logging
        enable_statistics: Enable collection of screening statistics
        max_processing_time: Maximum time allowed for screening (seconds)
        fallback_on_error: Use fallback behavior on errors
        description: Human-readable description of this configuration
    """
    profile: ScreeningProfile = ScreeningProfile.DEFAULT
    query_config: DatabaseQueryConfig = field(default_factory=DatabaseQueryConfig)
    filtering_config: FilteringConfig = field(default_factory=FilteringConfig)
    enable_caching: bool = False
    cache_ttl: int = 300
    enable_logging: bool = True
    enable_statistics: bool = True
    max_processing_time: Optional[float] = None
    fallback_on_error: bool = True
    description: str = "Default screening configuration"

    def __post_init__(self):
        """Apply profile settings after initialization."""
        if self.profile != ScreeningProfile.CUSTOM:
            self._apply_profile_settings()

    def _apply_profile_settings(self):
        """Apply pre-configured profile settings."""
        profile_configs = {
            ScreeningProfile.DEFAULT: self._get_default_profile,
            ScreeningProfile.CONSERVATIVE: self._get_conservative_profile,
            ScreeningProfile.AGGRESSIVE: self._get_aggressive_profile,
            ScreeningProfile.BALANCED: self._get_balanced_profile,
            ScreeningProfile.DAY_TRADING: self._get_day_trading_profile,
            ScreeningProfile.VALUE_INVESTING: self._get_value_investing_profile,
            ScreeningProfile.GROWTH_INVESTING: self._get_growth_investing_profile,
            ScreeningProfile.SECTOR_SPECIFIC: self._get_sector_specific_profile,
        }

        if self.profile in profile_configs:
            config_func = profile_configs[self.profile]
            query_config, filtering_config = config_func()
            self.query_config = query_config
            self.filtering_config = filtering_config
            self.description = f"{self.profile.value} screening profile"

    def _get_default_profile(self) -> tuple:
        """Get default profile configuration (current behavior)."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.ALL_STOCKS
        )
        filtering_config = FilteringConfig(
            strategies=[FilterStrategy.BASIC_TRADEABLE],
            apply_tradeable_filter=True
        )
        return query_config, filtering_config

    def _get_conservative_profile(self) -> tuple:
        """Get conservative profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters={
                'is_tradeable': True,
                'is_active': True,
                'market_cap_categories': ['large_cap']
            }
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.MARKET_CAP_BASED,
                FilterStrategy.VOLUME_BASED
            ],
            apply_tradeable_filter=True,
            min_volume=1000000,
            min_avg_volume=500000,
            market_cap_categories=['large_cap'],
            max_volatility=0.3,
            min_price=10.0
        )
        return query_config, filtering_config

    def _get_aggressive_profile(self) -> tuple:
        """Get aggressive profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters={
                'is_tradeable': True,
                'is_active': True,
                'market_cap_categories': ['small_cap', 'mid_cap']
            }
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.MARKET_CAP_BASED,
                FilterStrategy.VOLUME_BASED
            ],
            apply_tradeable_filter=True,
            min_volume=100000,
            market_cap_categories=['small_cap', 'mid_cap'],
            min_price=1.0,
            max_price=100.0
        )
        return query_config, filtering_config

    def _get_balanced_profile(self) -> tuple:
        """Get balanced profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.ALL_STOCKS
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.MARKET_CAP_BASED,
                FilterStrategy.VOLUME_BASED,
                FilterStrategy.PRICE_RANGE
            ],
            apply_tradeable_filter=True,
            min_volume=500000,
            min_avg_volume=250000,
            min_price=5.0,
            max_price=500.0,
            max_volatility=0.5
        )
        return query_config, filtering_config

    def _get_day_trading_profile(self) -> tuple:
        """Get day trading profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters={
                'is_tradeable': True,
                'is_active': True,
                'min_volume': 5000000
            }
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.VOLUME_BASED,
                FilterStrategy.PRICE_RANGE
            ],
            apply_tradeable_filter=True,
            min_volume=5000000,
            min_avg_volume=3000000,
            min_price=10.0,
            max_price=200.0,
            max_volatility=1.0  # Higher volatility acceptable for day trading
        )
        return query_config, filtering_config

    def _get_value_investing_profile(self) -> tuple:
        """Get value investing profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.ALL_STOCKS
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.FUNDAMENTAL,
                FilterStrategy.MARKET_CAP_BASED
            ],
            apply_tradeable_filter=True,
            min_market_cap=1000000000,  # $1B minimum
            max_pe_ratio=20,
            min_dividend_yield=0.02,  # 2% minimum dividend
            min_price=5.0
        )
        return query_config, filtering_config

    def _get_growth_investing_profile(self) -> tuple:
        """Get growth investing profile configuration."""
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.WITH_FILTERS,
            filters={
                'is_tradeable': True,
                'is_active': True,
                'market_cap_categories': ['mid_cap', 'large_cap']
            }
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.MARKET_CAP_BASED,
                FilterStrategy.TECHNICAL
            ],
            apply_tradeable_filter=True,
            market_cap_categories=['mid_cap', 'large_cap'],
            min_pe_ratio=15,
            max_pe_ratio=50,
            min_price=20.0
        )
        return query_config, filtering_config

    def _get_sector_specific_profile(self) -> tuple:
        """Get sector-specific profile configuration."""
        # Default to technology sector, can be customized
        query_config = DatabaseQueryConfig(
            strategy=QueryStrategy.BY_SECTOR,
            sectors=['Technology', 'Healthcare', 'Finance']
        )
        filtering_config = FilteringConfig(
            strategies=[
                FilterStrategy.BASIC_TRADEABLE,
                FilterStrategy.SECTOR_BASED,
                FilterStrategy.VOLUME_BASED
            ],
            apply_tradeable_filter=True,
            included_sectors=['Technology', 'Healthcare', 'Finance'],
            min_volume=500000
        )
        return query_config, filtering_config

    def validate(self) -> bool:
        """Validate the entire configuration."""
        try:
            # Validate sub-configurations
            if not self.query_config.validate():
                raise ValueError("Query configuration validation failed")

            if not self.filtering_config.validate():
                raise ValueError("Filtering configuration validation failed")

            # Validate main configuration
            if self.cache_ttl < 0:
                raise ValueError("cache_ttl must be >= 0")

            if self.max_processing_time is not None and self.max_processing_time <= 0:
                raise ValueError("max_processing_time must be > 0")

            logger.info(f"Configuration validated successfully: {self.description}")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'profile': self.profile.value,
            'query_config': self.query_config.to_dict(),
            'filtering_config': self.filtering_config.to_dict(),
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'enable_logging': self.enable_logging,
            'enable_statistics': self.enable_statistics,
            'max_processing_time': self.max_processing_time,
            'fallback_on_error': self.fallback_on_error,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StockScreeningConfig':
        """Create configuration from dictionary."""
        try:
            # Parse profile
            profile = ScreeningProfile(config_dict.get('profile', 'default'))

            # Parse query config
            query_dict = config_dict.get('query_config', {})
            if 'strategy' in query_dict:
                query_dict['strategy'] = QueryStrategy(query_dict['strategy'])
            query_config = DatabaseQueryConfig(**query_dict)

            # Parse filtering config
            filtering_dict = config_dict.get('filtering_config', {})
            if 'strategies' in filtering_dict:
                filtering_dict['strategies'] = [
                    FilterStrategy(s) for s in filtering_dict['strategies']
                ]
            filtering_config = FilteringConfig(**filtering_dict)

            # Create main config
            return cls(
                profile=profile,
                query_config=query_config,
                filtering_config=filtering_config,
                enable_caching=config_dict.get('enable_caching', False),
                cache_ttl=config_dict.get('cache_ttl', 300),
                enable_logging=config_dict.get('enable_logging', True),
                enable_statistics=config_dict.get('enable_statistics', True),
                max_processing_time=config_dict.get('max_processing_time'),
                fallback_on_error=config_dict.get('fallback_on_error', True),
                description=config_dict.get('description', 'Custom configuration')
            )

        except Exception as e:
            logger.error(f"Error creating configuration from dict: {e}")
            raise


class ConfigurationBuilder:
    """Builder class for creating stock screening configurations."""

    def __init__(self):
        """Initialize the configuration builder."""
        self.config = StockScreeningConfig(profile=ScreeningProfile.CUSTOM)

    def with_profile(self, profile: ScreeningProfile) -> 'ConfigurationBuilder':
        """Set screening profile."""
        self.config = StockScreeningConfig(profile=profile)
        return self

    def with_query_strategy(self, strategy: QueryStrategy) -> 'ConfigurationBuilder':
        """Set query strategy."""
        self.config.query_config.strategy = strategy
        return self

    def with_query_limit(self, limit: int) -> 'ConfigurationBuilder':
        """Set query limit."""
        self.config.query_config.limit = limit
        return self

    def with_symbols(self, symbols: List[str]) -> 'ConfigurationBuilder':
        """Set specific symbols to query."""
        self.config.query_config.symbols = symbols
        self.config.query_config.strategy = QueryStrategy.BY_SYMBOLS
        return self

    def with_sectors(self, sectors: List[str]) -> 'ConfigurationBuilder':
        """Set sectors to query."""
        self.config.query_config.sectors = sectors
        if not self.config.filtering_config.included_sectors:
            self.config.filtering_config.included_sectors = sectors
        return self

    def with_price_range(self, min_price: float = None,
                        max_price: float = None) -> 'ConfigurationBuilder':
        """Set price range filter."""
        self.config.filtering_config.min_price = min_price
        self.config.filtering_config.max_price = max_price
        if FilterStrategy.PRICE_RANGE not in self.config.filtering_config.strategies:
            self.config.filtering_config.strategies.append(FilterStrategy.PRICE_RANGE)
        return self

    def with_volume_filter(self, min_volume: int = None,
                          min_avg_volume: int = None) -> 'ConfigurationBuilder':
        """Set volume filter."""
        self.config.filtering_config.min_volume = min_volume
        self.config.filtering_config.min_avg_volume = min_avg_volume
        if FilterStrategy.VOLUME_BASED not in self.config.filtering_config.strategies:
            self.config.filtering_config.strategies.append(FilterStrategy.VOLUME_BASED)
        return self

    def with_market_cap_filter(self, categories: List[str] = None,
                              min_cap: float = None,
                              max_cap: float = None) -> 'ConfigurationBuilder':
        """Set market cap filter."""
        self.config.filtering_config.market_cap_categories = categories
        self.config.filtering_config.min_market_cap = min_cap
        self.config.filtering_config.max_market_cap = max_cap
        if FilterStrategy.MARKET_CAP_BASED not in self.config.filtering_config.strategies:
            self.config.filtering_config.strategies.append(FilterStrategy.MARKET_CAP_BASED)
        return self

    def with_caching(self, enabled: bool = True, ttl: int = 300) -> 'ConfigurationBuilder':
        """Enable caching with TTL."""
        self.config.enable_caching = enabled
        self.config.cache_ttl = ttl
        self.config.query_config.cache_enabled = enabled
        self.config.query_config.cache_ttl = ttl
        return self

    def with_custom_filter(self, filter_func: Callable) -> 'ConfigurationBuilder':
        """Add custom filter function."""
        if self.config.filtering_config.custom_filters is None:
            self.config.filtering_config.custom_filters = []
        self.config.filtering_config.custom_filters.append(filter_func)
        if FilterStrategy.CUSTOM not in self.config.filtering_config.strategies:
            self.config.filtering_config.strategies.append(FilterStrategy.CUSTOM)
        return self

    def build(self) -> StockScreeningConfig:
        """Build and validate the configuration."""
        if self.config.validate():
            return self.config
        else:
            raise ValueError("Configuration validation failed")


# Convenience factory functions for common configurations
def get_default_config() -> StockScreeningConfig:
    """Get default screening configuration."""
    return StockScreeningConfig(profile=ScreeningProfile.DEFAULT)


def get_conservative_config() -> StockScreeningConfig:
    """Get conservative screening configuration."""
    return StockScreeningConfig(profile=ScreeningProfile.CONSERVATIVE)


def get_aggressive_config() -> StockScreeningConfig:
    """Get aggressive screening configuration."""
    return StockScreeningConfig(profile=ScreeningProfile.AGGRESSIVE)


def get_day_trading_config() -> StockScreeningConfig:
    """Get day trading screening configuration."""
    return StockScreeningConfig(profile=ScreeningProfile.DAY_TRADING)


def get_custom_config(query_strategy: QueryStrategy = QueryStrategy.ALL_STOCKS,
                     filter_strategies: List[FilterStrategy] = None,
                     **kwargs) -> StockScreeningConfig:
    """
    Create a custom screening configuration.

    Args:
        query_strategy: Database query strategy
        filter_strategies: List of filtering strategies
        **kwargs: Additional configuration parameters

    Returns:
        Custom screening configuration
    """
    if filter_strategies is None:
        filter_strategies = [FilterStrategy.BASIC_TRADEABLE]

    query_config = DatabaseQueryConfig(strategy=query_strategy)
    filtering_config = FilteringConfig(strategies=filter_strategies)

    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(query_config, key):
            setattr(query_config, key, value)
        elif hasattr(filtering_config, key):
            setattr(filtering_config, key, value)

    config = StockScreeningConfig(
        profile=ScreeningProfile.CUSTOM,
        query_config=query_config,
        filtering_config=filtering_config,
        description="Custom screening configuration"
    )

    return config