"""
Minimal Stock Screening Configuration Module

Provides only the essential configurable parameters for the stock screening pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Database query strategies for fetching stocks."""
    ALL_STOCKS = "all_stocks"
    WITH_FILTERS = "with_filters"


class FilterStrategy(Enum):
    """Filtering strategies for stock selection."""
    BASIC_TRADEABLE = "basic_tradeable"
    MARKET_CAP_BASED = "market_cap_based"
    VOLUME_BASED = "volume_based"
    PRICE_RANGE = "price_range"
    SECTOR_BASED = "sector_based"


class ScreeningProfile(Enum):
    """Pre-configured screening profiles for different trading styles."""
    DEFAULT = "default"  # Current behavior - all stocks with tradeable filter
    CUSTOM = "custom"   # Custom configuration


@dataclass
class DatabaseQueryConfig:
    """Configuration for database query operations."""
    strategy: QueryStrategy = QueryStrategy.ALL_STOCKS
    limit: Optional[int] = None
    offset: Optional[int] = None
    sectors: Optional[List[str]] = None
    include_inactive: bool = False

    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
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
        return {k: v for k, v in config_dict.items() if v is not None}


@dataclass
class FilteringConfig:
    """Configuration for stock filtering operations."""
    strategies: List[FilterStrategy] = field(default_factory=lambda: [FilterStrategy.BASIC_TRADEABLE])
    apply_tradeable_filter: bool = True
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    included_sectors: Optional[List[str]] = None
    filter_chain_operator: str = "AND"

    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate price range
            if (self.min_price is not None and self.max_price is not None
                and self.min_price > self.max_price):
                raise ValueError("min_price cannot be greater than max_price")

            # Validate filter chain operator
            if self.filter_chain_operator not in ["AND", "OR"]:
                raise ValueError("filter_chain_operator must be 'AND' or 'OR'")

            # Validate numeric values
            for attr in ['min_price', 'max_price', 'min_volume']:
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
        return {k: v for k, v in config_dict.items() if v is not None}


@dataclass
class StockScreeningConfig:
    """Main configuration class for the stock screening pipeline."""
    profile: ScreeningProfile = ScreeningProfile.DEFAULT
    query_config: DatabaseQueryConfig = field(default_factory=DatabaseQueryConfig)
    filtering_config: FilteringConfig = field(default_factory=FilteringConfig)
    enable_logging: bool = True
    enable_statistics: bool = True
    fallback_on_error: bool = True
    description: str = "default screening profile"

    def validate(self) -> bool:
        """Validate the entire configuration."""
        return (self.query_config.validate() and
                self.filtering_config.validate())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'profile': self.profile.value,
            'query_config': self.query_config.to_dict(),
            'filtering_config': self.filtering_config.to_dict(),
            'enable_logging': self.enable_logging,
            'enable_statistics': self.enable_statistics,
            'fallback_on_error': self.fallback_on_error,
            'description': self.description
        }


class ConfigurationBuilder:
    """Builder class for creating custom configurations."""

    def __init__(self):
        self._query_config = DatabaseQueryConfig()
        self._filtering_config = FilteringConfig()
        self._profile = ScreeningProfile.CUSTOM
        self._enable_logging = True
        self._enable_statistics = True
        self._fallback_on_error = True

    def with_sectors(self, sectors: List[str]) -> 'ConfigurationBuilder':
        """Set sectors filter."""
        self._filtering_config.included_sectors = sectors
        if FilterStrategy.SECTOR_BASED not in self._filtering_config.strategies:
            self._filtering_config.strategies.append(FilterStrategy.SECTOR_BASED)
        return self

    def with_price_range(self, min_price: float, max_price: float) -> 'ConfigurationBuilder':
        """Set price range filter."""
        self._filtering_config.min_price = min_price
        self._filtering_config.max_price = max_price
        if FilterStrategy.PRICE_RANGE not in self._filtering_config.strategies:
            self._filtering_config.strategies.append(FilterStrategy.PRICE_RANGE)
        return self

    def with_volume_filter(self, min_volume: int) -> 'ConfigurationBuilder':
        """Set minimum volume filter."""
        self._filtering_config.min_volume = min_volume
        if FilterStrategy.VOLUME_BASED not in self._filtering_config.strategies:
            self._filtering_config.strategies.append(FilterStrategy.VOLUME_BASED)
        return self

    def with_query_limit(self, limit: int) -> 'ConfigurationBuilder':
        """Set query limit."""
        self._query_config.limit = limit
        return self

    def build(self) -> StockScreeningConfig:
        """Build the final configuration."""
        return StockScreeningConfig(
            profile=self._profile,
            query_config=self._query_config,
            filtering_config=self._filtering_config,
            enable_logging=self._enable_logging,
            enable_statistics=self._enable_statistics,
            fallback_on_error=self._fallback_on_error,
            description=f"{self._profile.value} screening profile"
        )


def get_default_config() -> StockScreeningConfig:
    """Get the default configuration (maintains current behavior)."""
    return StockScreeningConfig(
        profile=ScreeningProfile.DEFAULT,
        query_config=DatabaseQueryConfig(strategy=QueryStrategy.ALL_STOCKS),
        filtering_config=FilteringConfig(strategies=[FilterStrategy.BASIC_TRADEABLE]),
        description="default screening profile"
    )