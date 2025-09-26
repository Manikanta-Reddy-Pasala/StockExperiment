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
    minimum_avg_volume: int = 50000
    minimum_trades_per_day: int = 100


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
class TradingStatusConfig:
    """Configuration for trading status filters."""
    exclude_suspended: bool = True
    exclude_delisted: bool = True
    exclude_stage_listed: bool = True
    min_listing_days: int = 365


@dataclass
class LiquidityRequirementsConfig:
    """Configuration for basic liquidity requirements."""
    minimum_bid_ask_spread: float = 0.01
    maximum_bid_ask_spread: float = 5.0
    minimum_market_depth: int = 1000


@dataclass
class TechnicalIndicatorConfig:
    """Base configuration for technical indicators."""
    enabled: bool = True


@dataclass
class RSIConfig(TechnicalIndicatorConfig):
    """RSI indicator configuration."""
    oversold_threshold: int = 30
    overbought_threshold: int = 70
    neutral_range_min: int = 40
    neutral_range_max: int = 60
    period: int = 14


@dataclass
class MACDConfig(TechnicalIndicatorConfig):
    """MACD indicator configuration."""
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    histogram_threshold: float = 0.01


@dataclass
class TechnicalIndicatorsConfig:
    """Configuration for all technical indicators."""
    rsi: Optional[Dict] = None
    macd: Optional[Dict] = None
    bollinger_bands: Optional[Dict] = None
    moving_averages: Optional[Dict] = None
    stochastic: Optional[Dict] = None
    williams_r: Optional[Dict] = None
    atr: Optional[Dict] = None


@dataclass
class VolumeAnalysisConfig:
    """Configuration for volume analysis."""
    volume_surge: Optional[Dict] = None
    obv: Optional[Dict] = None
    vpt: Optional[Dict] = None
    mfi: Optional[Dict] = None


@dataclass
class FundamentalRatiosConfig:
    """Configuration for fundamental ratios."""
    pe_ratio: Optional[Dict] = None
    pb_ratio: Optional[Dict] = None
    peg_ratio: Optional[Dict] = None
    roe: Optional[Dict] = None
    roa: Optional[Dict] = None
    profit_margin: Optional[Dict] = None
    debt_to_equity: Optional[Dict] = None
    current_ratio: Optional[Dict] = None
    quick_ratio: Optional[Dict] = None
    revenue_growth: Optional[Dict] = None
    earnings_growth: Optional[Dict] = None


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics."""
    beta: Optional[Dict] = None
    volatility: Optional[Dict] = None
    sharpe_ratio: Optional[Dict] = None
    max_drawdown: Optional[Dict] = None
    var: Optional[Dict] = None


@dataclass
class MomentumConfig:
    """Configuration for momentum indicators."""
    price_momentum: Optional[Dict] = None
    roc: Optional[Dict] = None
    relative_strength: Optional[Dict] = None


@dataclass
class TrendAnalysisConfig:
    """Configuration for trend analysis."""
    trend_direction: Optional[Dict] = None
    support_resistance: Optional[Dict] = None
    chart_patterns: Optional[Dict] = None
    adx: Optional[Dict] = None


@dataclass
class MarketFactorsConfig:
    """Configuration for market factors."""
    sector_performance: Optional[Dict] = None
    market_regime: Optional[Dict] = None
    seasonality: Optional[Dict] = None


@dataclass
class ScoringWeightsConfig:
    """Configuration for scoring weights."""
    technical_score: float = 0.30
    fundamental_score: float = 0.25
    risk_score: float = 0.20
    momentum_score: float = 0.15
    volume_score: float = 0.10


@dataclass
class FilteringThresholdsConfig:
    """Configuration for filtering thresholds."""
    minimum_total_score: int = 60
    minimum_technical_score: int = 50
    minimum_fundamental_score: int = 40
    require_all_categories: bool = False


@dataclass
class Stage1FiltersConfig:
    """Configuration for Stage 1 filters."""
    tradeability: TradeabilityConfig = field(default_factory=TradeabilityConfig)
    market_cap_categories: Dict[str, MarketCapConfig] = field(default_factory=dict)
    price_volume_thresholds: Optional[Dict] = None
    trading_status: TradingStatusConfig = field(default_factory=TradingStatusConfig)
    liquidity: LiquidityRequirementsConfig = field(default_factory=LiquidityRequirementsConfig)


@dataclass
class Stage2FiltersConfig:
    """Configuration for Stage 2 filters."""
    technical_indicators: TechnicalIndicatorsConfig = field(default_factory=TechnicalIndicatorsConfig)
    volume_analysis: VolumeAnalysisConfig = field(default_factory=VolumeAnalysisConfig)
    fundamental_ratios: FundamentalRatiosConfig = field(default_factory=FundamentalRatiosConfig)
    risk_metrics: RiskMetricsConfig = field(default_factory=RiskMetricsConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    trend_analysis: TrendAnalysisConfig = field(default_factory=TrendAnalysisConfig)
    market_factors: MarketFactorsConfig = field(default_factory=MarketFactorsConfig)
    scoring_weights: ScoringWeightsConfig = field(default_factory=ScoringWeightsConfig)
    filtering_thresholds: FilteringThresholdsConfig = field(default_factory=FilteringThresholdsConfig)


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
    # Stage 1 filters (backward compatibility + new structure)
    tradeability: TradeabilityConfig = field(default_factory=TradeabilityConfig)
    market_cap_categories: Dict[str, MarketCapConfig] = field(default_factory=dict)
    trading_status: TradingStatusConfig = field(default_factory=TradingStatusConfig)
    liquidity_requirements: LiquidityRequirementsConfig = field(default_factory=LiquidityRequirementsConfig)

    # Stage 1 and Stage 2 comprehensive filters
    stage_1_filters: Optional[Stage1FiltersConfig] = None
    stage_2_filters: Optional[Stage2FiltersConfig] = None

    # Supporting configurations
    liquidity_scoring: LiquidityScoringConfig = field(default_factory=LiquidityScoringConfig)
    shares_estimation: SharesEstimationConfig = field(default_factory=SharesEstimationConfig)

    # API and cache settings
    cache_duration: int = 3600
    screening_limit: int = 1000
    quote_batch_size: int = 20
    quote_rate_limit: float = 0.5

    # Sector detection
    sector_keywords: Dict[str, list] = field(default_factory=dict)

    # Logging settings
    enable_discovery_summary: bool = True
    top_stocks_per_category: int = 5

    def get_stage1_config(self) -> Stage1FiltersConfig:
        """Get Stage 1 configuration with defaults."""
        if self.stage_1_filters:
            return self.stage_1_filters
        # Return backward compatible configuration
        return Stage1FiltersConfig(
            tradeability=self.tradeability,
            market_cap_categories=self.market_cap_categories,
            trading_status=self.trading_status,
            liquidity=self.liquidity_requirements
        )

    def get_stage2_config(self) -> Optional[Stage2FiltersConfig]:
        """Get Stage 2 configuration if available."""
        return self.stage_2_filters

    def is_stage2_enabled(self) -> bool:
        """Check if Stage 2 filtering is configured and enabled."""
        return self.stage_2_filters is not None

    def get_technical_indicator(self, indicator_name: str) -> Optional[Dict]:
        """Get specific technical indicator configuration."""
        if not self.stage_2_filters or not self.stage_2_filters.technical_indicators:
            return None
        return getattr(self.stage_2_filters.technical_indicators, indicator_name, None)

    def get_scoring_weights(self) -> ScoringWeightsConfig:
        """Get scoring weights with defaults."""
        if self.stage_2_filters and self.stage_2_filters.scoring_weights:
            return self.stage_2_filters.scoring_weights
        return ScoringWeightsConfig()


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

        # Parse Stage 1 filters
        if 'stage_1_filters' in data:
            stage1 = data['stage_1_filters']
            config.stage_1_filters = Stage1FiltersConfig()

            # Tradeability
            if 'tradeability' in stage1:
                trade = stage1['tradeability']
                config.tradeability = TradeabilityConfig(
                    minimum_price=trade.get('minimum_price', 5.0),
                    maximum_price=trade.get('maximum_price', 10000.0),
                    minimum_volume=trade.get('minimum_volume', 10000),
                    minimum_liquidity_score=trade.get('minimum_liquidity_score', 0.3),
                    minimum_avg_volume=trade.get('minimum_avg_volume', 50000),
                    minimum_trades_per_day=trade.get('minimum_trades_per_day', 100)
                )
                config.stage_1_filters.tradeability = config.tradeability

            # Market cap categories
            if 'market_cap_categories' in stage1:
                for cap_type, cap_data in stage1['market_cap_categories'].items():
                    config.market_cap_categories[cap_type] = MarketCapConfig(
                        minimum=cap_data.get('minimum'),
                        maximum=cap_data.get('maximum'),
                        label=cap_data.get('label', cap_type)
                    )
                config.stage_1_filters.market_cap_categories = config.market_cap_categories

            # Trading status
            if 'trading_status' in stage1:
                ts = stage1['trading_status']
                config.trading_status = TradingStatusConfig(
                    exclude_suspended=ts.get('exclude_suspended', True),
                    exclude_delisted=ts.get('exclude_delisted', True),
                    exclude_stage_listed=ts.get('exclude_stage_listed', True),
                    min_listing_days=ts.get('min_listing_days', 365)
                )
                config.stage_1_filters.trading_status = config.trading_status

            # Liquidity requirements
            if 'liquidity' in stage1:
                liq = stage1['liquidity']
                config.liquidity_requirements = LiquidityRequirementsConfig(
                    minimum_bid_ask_spread=liq.get('minimum_bid_ask_spread', 0.01),
                    maximum_bid_ask_spread=liq.get('maximum_bid_ask_spread', 5.0),
                    minimum_market_depth=liq.get('minimum_market_depth', 1000)
                )
                config.stage_1_filters.liquidity = config.liquidity_requirements

            # Price volume thresholds
            if 'price_volume_thresholds' in stage1:
                config.stage_1_filters.price_volume_thresholds = stage1['price_volume_thresholds']

        # Parse Stage 2 filters
        if 'stage_2_filters' in data:
            stage2 = data['stage_2_filters']
            config.stage_2_filters = Stage2FiltersConfig()

            # Technical indicators
            if 'technical_indicators' in stage2:
                config.stage_2_filters.technical_indicators = TechnicalIndicatorsConfig(
                    rsi=stage2['technical_indicators'].get('rsi'),
                    macd=stage2['technical_indicators'].get('macd'),
                    bollinger_bands=stage2['technical_indicators'].get('bollinger_bands'),
                    moving_averages=stage2['technical_indicators'].get('moving_averages'),
                    stochastic=stage2['technical_indicators'].get('stochastic'),
                    williams_r=stage2['technical_indicators'].get('williams_r'),
                    atr=stage2['technical_indicators'].get('atr')
                )

            # Volume analysis
            if 'volume_analysis' in stage2:
                config.stage_2_filters.volume_analysis = VolumeAnalysisConfig(
                    volume_surge=stage2['volume_analysis'].get('volume_surge'),
                    obv=stage2['volume_analysis'].get('obv'),
                    vpt=stage2['volume_analysis'].get('vpt'),
                    mfi=stage2['volume_analysis'].get('mfi')
                )

            # Fundamental ratios
            if 'fundamental_ratios' in stage2:
                fr = stage2['fundamental_ratios']
                config.stage_2_filters.fundamental_ratios = FundamentalRatiosConfig(
                    pe_ratio=fr.get('pe_ratio'),
                    pb_ratio=fr.get('pb_ratio'),
                    peg_ratio=fr.get('peg_ratio'),
                    roe=fr.get('roe'),
                    roa=fr.get('roa'),
                    profit_margin=fr.get('profit_margin'),
                    debt_to_equity=fr.get('debt_to_equity'),
                    current_ratio=fr.get('current_ratio'),
                    quick_ratio=fr.get('quick_ratio'),
                    revenue_growth=fr.get('revenue_growth'),
                    earnings_growth=fr.get('earnings_growth')
                )

            # Risk metrics
            if 'risk_metrics' in stage2:
                rm = stage2['risk_metrics']
                config.stage_2_filters.risk_metrics = RiskMetricsConfig(
                    beta=rm.get('beta'),
                    volatility=rm.get('volatility'),
                    sharpe_ratio=rm.get('sharpe_ratio'),
                    max_drawdown=rm.get('max_drawdown'),
                    var=rm.get('var')
                )

            # Momentum
            if 'momentum' in stage2:
                config.stage_2_filters.momentum = MomentumConfig(
                    price_momentum=stage2['momentum'].get('price_momentum'),
                    roc=stage2['momentum'].get('roc'),
                    relative_strength=stage2['momentum'].get('relative_strength')
                )

            # Trend analysis
            if 'trend_analysis' in stage2:
                ta = stage2['trend_analysis']
                config.stage_2_filters.trend_analysis = TrendAnalysisConfig(
                    trend_direction=ta.get('trend_direction'),
                    support_resistance=ta.get('support_resistance'),
                    chart_patterns=ta.get('chart_patterns'),
                    adx=ta.get('adx')
                )

            # Market factors
            if 'market_factors' in stage2:
                mf = stage2['market_factors']
                config.stage_2_filters.market_factors = MarketFactorsConfig(
                    sector_performance=mf.get('sector_performance'),
                    market_regime=mf.get('market_regime'),
                    seasonality=mf.get('seasonality')
                )

            # Scoring weights
            if 'scoring_weights' in stage2:
                sw = stage2['scoring_weights']
                config.stage_2_filters.scoring_weights = ScoringWeightsConfig(
                    technical_score=sw.get('technical_score', 0.30),
                    fundamental_score=sw.get('fundamental_score', 0.25),
                    risk_score=sw.get('risk_score', 0.20),
                    momentum_score=sw.get('momentum_score', 0.15),
                    volume_score=sw.get('volume_score', 0.10)
                )

            # Filtering thresholds
            if 'filtering_thresholds' in stage2:
                ft = stage2['filtering_thresholds']
                config.stage_2_filters.filtering_thresholds = FilteringThresholdsConfig(
                    minimum_total_score=ft.get('minimum_total_score', 60),
                    minimum_technical_score=ft.get('minimum_technical_score', 50),
                    minimum_fundamental_score=ft.get('minimum_fundamental_score', 40),
                    require_all_categories=ft.get('require_all_categories', False)
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


# Convenience functions
def get_stock_filter_config() -> StockFilterConfig:
    """Get the stock filter configuration."""
    loader = ConfigLoader()
    return loader.get_config()


def get_stage1_config() -> Stage1FiltersConfig:
    """Get Stage 1 filter configuration."""
    config = get_stock_filter_config()
    return config.get_stage1_config()


def get_stage2_config() -> Optional[Stage2FiltersConfig]:
    """Get Stage 2 filter configuration."""
    config = get_stock_filter_config()
    return config.get_stage2_config()


def reload_config():
    """Reload the configuration from file."""
    loader = ConfigLoader()
    loader.reload_config()