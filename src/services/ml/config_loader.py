"""
Configuration Loader for Stock Filters

This module provides a centralized way to load and access stock filtering
configuration from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeabilityConfig:
    """Configuration for stock tradeability criteria."""
    minimum_price: float = 5.0
    maximum_price: float = 10000.0
    minimum_volume: int = 10000
    minimum_liquidity_score: float = 0.3


@dataclass
class MarketCapConfig:
    """Configuration for market cap category."""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    label: str = ""


@dataclass
class LiquidityScoringConfig:
    """Configuration for liquidity scoring."""
    volume_normalization: int = 1000000
    volume_weight: float = 0.7
    spread_weight: float = 0.3
    spread_multiplier: int = 10

    # Database scoring
    db_volume_weight: float = 0.8
    db_price_stability_weight: float = 0.2
    stable_price_min: float = 100
    stable_price_max: float = 1000
    stable_price_score: float = 0.8
    unstable_price_score: float = 0.6


@dataclass
class SharesEstimationConfig:
    """Configuration for shares outstanding estimation."""
    high_price_threshold: float = 1500
    high_price_shares_min: int = 50000000
    high_price_shares_max: int = 300000000

    mid_high_price_threshold: float = 500
    mid_high_shares_min: int = 100000000
    mid_high_shares_max: int = 800000000

    medium_price_threshold: float = 100
    medium_shares_min: int = 200000000
    medium_shares_max: int = 1500000000

    low_shares_min: int = 500000000
    low_shares_max: int = 5000000000

    # Volume adjustments
    high_volume_threshold: int = 200000
    high_volume_mult_min: float = 0.8
    high_volume_mult_max: float = 1.2

    medium_volume_threshold: int = 50000
    medium_volume_mult_min: float = 0.9
    medium_volume_mult_max: float = 1.3

    low_volume_mult_min: float = 1.0
    low_volume_mult_max: float = 2.0


@dataclass
class StockFilterConfig:
    """Main configuration class for stock filters."""
    tradeability: TradeabilityConfig = field(default_factory=TradeabilityConfig)
    market_cap_categories: Dict[str, MarketCapConfig] = field(default_factory=dict)
    liquidity_scoring: LiquidityScoringConfig = field(default_factory=LiquidityScoringConfig)
    shares_estimation: SharesEstimationConfig = field(default_factory=SharesEstimationConfig)

    cache_duration: int = 3600
    screening_limit: int = 1000
    quote_batch_size: int = 20
    quote_rate_limit: float = 0.5

    sector_keywords: Dict[str, list] = field(default_factory=dict)

    enable_discovery_summary: bool = True
    top_stocks_per_category: int = 5


class ConfigLoader:
    """Singleton configuration loader for stock filters."""

    _instance: Optional['ConfigLoader'] = None
    _config: Optional[StockFilterConfig] = None
    _config_path: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            # Find the config file
            config_paths = [
                Path(__file__).parent.parent.parent.parent / "config" / "stock_filters.yaml",
                Path(os.getcwd()) / "config" / "stock_filters.yaml",
                Path.home() / ".stockexperiment" / "stock_filters.yaml"
            ]

            config_data = None
            for path in config_paths:
                if path.exists():
                    self._config_path = path
                    logger.info(f"Loading stock filter configuration from: {path}")
                    with open(path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    break

            if config_data is None:
                logger.warning("No configuration file found, using defaults")
                self._config = StockFilterConfig()
                return

            # Parse configuration
            self._config = self._parse_config(config_data)
            logger.info("Stock filter configuration loaded successfully")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            self._config = StockFilterConfig()

    def _parse_config(self, data: Dict[str, Any]) -> StockFilterConfig:
        """Parse YAML data into configuration objects."""
        config = StockFilterConfig()

        # Stage 1 filters
        if 'stage_1_filters' in data:
            stage1 = data['stage_1_filters']

            # Tradeability
            if 'tradeability' in stage1:
                trade = stage1['tradeability']
                config.tradeability = TradeabilityConfig(
                    minimum_price=trade.get('minimum_price', 5.0),
                    maximum_price=trade.get('maximum_price', 10000.0),
                    minimum_volume=trade.get('minimum_volume', 10000),
                    minimum_liquidity_score=trade.get('minimum_liquidity_score', 0.3)
                )

            # Market cap categories
            if 'market_cap_categories' in stage1:
                for cap_type, cap_data in stage1['market_cap_categories'].items():
                    config.market_cap_categories[cap_type] = MarketCapConfig(
                        minimum=cap_data.get('minimum'),
                        maximum=cap_data.get('maximum'),
                        label=cap_data.get('label', cap_type)
                    )

        # Liquidity scoring
        if 'liquidity_scoring' in data:
            liq = data['liquidity_scoring']
            weights = liq.get('weights', {})
            db_scoring = liq.get('database_scoring', {})
            stable_range = db_scoring.get('stable_price_range', {})

            config.liquidity_scoring = LiquidityScoringConfig(
                volume_normalization=liq.get('volume_normalization', 1000000),
                volume_weight=weights.get('volume', 0.7),
                spread_weight=weights.get('spread', 0.3),
                spread_multiplier=liq.get('spread_multiplier', 10),
                db_volume_weight=db_scoring.get('volume_weight', 0.8),
                db_price_stability_weight=db_scoring.get('price_stability_weight', 0.2),
                stable_price_min=stable_range.get('minimum', 100),
                stable_price_max=stable_range.get('maximum', 1000),
                stable_price_score=db_scoring.get('stable_price_score', 0.8),
                unstable_price_score=db_scoring.get('unstable_price_score', 0.6)
            )

        # Shares estimation
        if 'shares_estimation' in data:
            shares = data['shares_estimation']
            high = shares.get('high_price_stocks', {})
            mid_high = shares.get('mid_high_price_stocks', {})
            medium = shares.get('medium_price_stocks', {})
            low = shares.get('low_price_stocks', {})
            vol_adj = shares.get('volume_adjustments', {})

            config.shares_estimation = SharesEstimationConfig(
                high_price_threshold=high.get('price_threshold', 1500),
                high_price_shares_min=high.get('shares_range_min', 50000000),
                high_price_shares_max=high.get('shares_range_max', 300000000),
                mid_high_price_threshold=mid_high.get('price_threshold', 500),
                mid_high_shares_min=mid_high.get('shares_range_min', 100000000),
                mid_high_shares_max=mid_high.get('shares_range_max', 800000000),
                medium_price_threshold=medium.get('price_threshold', 100),
                medium_shares_min=medium.get('shares_range_min', 200000000),
                medium_shares_max=medium.get('shares_range_max', 1500000000),
                low_shares_min=low.get('shares_range_min', 500000000),
                low_shares_max=low.get('shares_range_max', 5000000000),
                high_volume_threshold=vol_adj.get('high_volume_threshold', 200000),
                high_volume_mult_min=vol_adj.get('high_volume_multiplier_min', 0.8),
                high_volume_mult_max=vol_adj.get('high_volume_multiplier_max', 1.2),
                medium_volume_threshold=vol_adj.get('medium_volume_threshold', 50000),
                medium_volume_mult_min=vol_adj.get('medium_volume_multiplier_min', 0.9),
                medium_volume_mult_max=vol_adj.get('medium_volume_multiplier_max', 1.3),
                low_volume_mult_min=vol_adj.get('low_volume_multiplier_min', 1.0),
                low_volume_mult_max=vol_adj.get('low_volume_multiplier_max', 2.0)
            )

        # Database and cache settings
        if 'database' in data:
            config.screening_limit = data['database'].get('screening_limit', 1000)

        if 'cache' in data:
            config.cache_duration = data['cache'].get('duration_seconds', 3600)

        # API processing
        if 'api_processing' in data:
            api = data['api_processing']
            config.quote_batch_size = api.get('quote_batch_size', 20)
            config.quote_rate_limit = api.get('quote_rate_limit_delay', 0.5)

        # Sector keywords
        if 'sector_keywords' in data:
            config.sector_keywords = data['sector_keywords']

        # Logging settings
        if 'logging' in data:
            log = data['logging']
            config.enable_discovery_summary = log.get('enable_discovery_summary', True)
            config.top_stocks_per_category = log.get('top_stocks_per_category', 5)

        return config

    def get_config(self) -> StockFilterConfig:
        """Get the loaded configuration."""
        if self._config is None:
            self._load_config()
        return self._config

    def reload_config(self):
        """Reload configuration from file."""
        self._config = None
        self._load_config()
        logger.info("Configuration reloaded")

    def get_config_path(self) -> Optional[Path]:
        """Get the path to the loaded configuration file."""
        return self._config_path


# Convenience function
def get_stock_filter_config() -> StockFilterConfig:
    """Get the stock filter configuration."""
    loader = ConfigLoader()
    return loader.get_config()