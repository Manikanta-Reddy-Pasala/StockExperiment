"""
Enhanced Configuration Loader for Stock Filtering
Supports comprehensive YAML configuration with all filtering parameters
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    """Universe and data quality configuration."""
    exchanges: list = field(default_factory=lambda: ["NSE"])
    instrument_types: list = field(default_factory=lambda: ["EQ"])
    min_history_days: int = 220
    min_non_null_ratio: float = 0.98
    max_price_gap_pct: float = 20.0
    max_spread_pct_hard: float = 5.0
    stale_quote_max_minutes: int = 15


@dataclass
class Stage1Filters:
    """Stage 1 basic filtering criteria."""
    minimum_price: float = 5.0
    maximum_price: float = 10000.0
    minimum_daily_turnover_inr: float = 50000000
    fallback_minimum_volume: int = 50000
    minimum_liquidity_score: float = 0.3
    minimum_trades_per_day: int = 200
    min_listing_days: int = 180
    max_bid_ask_spread_pct: float = 1.0
    min_atr_pct_of_price: float = 1.0
    max_atr_pct_of_price: float = 7.0


@dataclass
class TechnicalIndicators:
    """Technical indicators configuration."""
    rsi_enabled: bool = True
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    macd_enabled: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    bollinger_enabled: bool = True
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    ma_enabled: bool = True
    sma_periods: list = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: list = field(default_factory=lambda: [12, 26, 50])
    price_above_ma200: bool = True
    
    atr_enabled: bool = True
    atr_period: int = 14


@dataclass
class FundamentalRatios:
    """Fundamental ratios configuration."""
    pe_ratio_enabled: bool = True
    pe_min: float = 0
    pe_max: float = 50
    
    pb_ratio_enabled: bool = True
    pb_min: float = 0.5
    pb_max: float = 5.0
    
    roe_enabled: bool = True
    roe_min: float = 12
    
    debt_to_equity_enabled: bool = True
    debt_to_equity_max: float = 1.5
    
    current_ratio_enabled: bool = True
    current_ratio_min: float = 1.2


@dataclass
class RiskMetrics:
    """Risk metrics configuration."""
    beta_enabled: bool = True
    beta_min: float = 0.7
    beta_max: float = 1.6
    
    volatility_enabled: bool = True
    volatility_period: int = 20
    volatility_max_daily: float = 4.0
    volatility_max_annual: float = 45.0
    
    sharpe_ratio_enabled: bool = True
    sharpe_ratio_min: float = 0.5
    risk_free_rate: float = 6.5


@dataclass
class ScoringWeights:
    """Scoring weights for ranking."""
    technical_score: float = 0.30
    fundamental_score: float = 0.20
    risk_score: float = 0.20
    momentum_score: float = 0.25
    volume_score: float = 0.05


@dataclass
class FilteringThresholds:
    """Filtering thresholds."""
    minimum_total_score: float = 60
    minimum_technical_score: float = 50
    minimum_fundamental_score: float = 40
    minimum_risk_score: float = 50
    require_all_categories: bool = False


@dataclass
class SelectionConfig:
    """Final selection configuration."""
    max_suggested_stocks: int = 10
    tie_breaker_priority: list = field(default_factory=lambda: ["momentum_score", "risk_score", "technical_score"])
    sector_concentration_limit_pct: float = 40
    min_large_mid_pct: float = 60
    blacklist_symbols: list = field(default_factory=list)
    whitelist_symbols: list = field(default_factory=list)
    min_distance_from_resistance_pct: float = 2.0
    require_price_above_vwap: bool = True


@dataclass
class EnhancedFilteringConfig:
    """Enhanced filtering configuration with all parameters."""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    stage_1_filters: Stage1Filters = field(default_factory=Stage1Filters)
    technical_indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    fundamental_ratios: FundamentalRatios = field(default_factory=FundamentalRatios)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    filtering_thresholds: FilteringThresholds = field(default_factory=FilteringThresholds)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    
    # Additional configurations
    liquidity_scoring: Dict[str, Any] = field(default_factory=dict)
    shares_estimation: Dict[str, Any] = field(default_factory=dict)
    database: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    api_processing: Dict[str, Any] = field(default_factory=dict)
    sector_keywords: Dict[str, list] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


class EnhancedConfigLoader:
    """Enhanced configuration loader for comprehensive stock filtering."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader."""
        if config_path is None:
            # Default to the config file in the project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "stock_filters.yaml"
        
        self.config_path = config_path
        self._config: Optional[EnhancedFilteringConfig] = None
    
    def load_config(self) -> EnhancedFilteringConfig:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return EnhancedFilteringConfig()
            
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Parse universe configuration
            universe_data = config_data.get('universe', {})
            universe = UniverseConfig(
                exchanges=universe_data.get('exchanges', ["NSE"]),
                instrument_types=universe_data.get('instrument_types', ["EQ"]),
                min_history_days=universe_data.get('history_requirements', {}).get('min_history_days', 220),
                min_non_null_ratio=universe_data.get('history_requirements', {}).get('min_non_null_ratio', 0.98),
                max_price_gap_pct=universe_data.get('data_quality', {}).get('max_price_gap_pct', 20.0),
                max_spread_pct_hard=universe_data.get('data_quality', {}).get('max_spread_pct_hard', 5.0),
                stale_quote_max_minutes=universe_data.get('data_quality', {}).get('stale_quote_max_minutes', 15)
            )
            
            # Parse stage 1 filters
            stage1_data = config_data.get('stage_1_filters', {})
            tradeability = stage1_data.get('tradeability', {})
            liquidity = stage1_data.get('liquidity', {})
            baseline_vol = stage1_data.get('baseline_volatility', {})
            
            stage1_filters = Stage1Filters(
                minimum_price=tradeability.get('minimum_price', 5.0),
                maximum_price=tradeability.get('maximum_price', 10000.0),
                minimum_daily_turnover_inr=tradeability.get('minimum_daily_turnover_inr', 50000000),
                fallback_minimum_volume=tradeability.get('fallback_minimum_volume', 50000),
                minimum_liquidity_score=tradeability.get('minimum_liquidity_score', 0.3),
                minimum_trades_per_day=tradeability.get('minimum_trades_per_day', 200),
                min_listing_days=stage1_data.get('trading_status', {}).get('min_listing_days', 180),
                max_bid_ask_spread_pct=liquidity.get('max_bid_ask_spread_pct', 1.0),
                min_atr_pct_of_price=baseline_vol.get('min_atr_pct_of_price', 1.0),
                max_atr_pct_of_price=baseline_vol.get('max_atr_pct_of_price', 7.0)
            )
            
            # Parse technical indicators
            tech_data = config_data.get('stage_2_filters', {}).get('technical_indicators', {})
            rsi_data = tech_data.get('rsi', {})
            macd_data = tech_data.get('macd', {})
            bb_data = tech_data.get('bollinger_bands', {})
            ma_data = tech_data.get('moving_averages', {})
            atr_data = tech_data.get('atr', {})
            
            technical_indicators = TechnicalIndicators(
                rsi_enabled=rsi_data.get('enabled', True),
                rsi_period=rsi_data.get('period', 14),
                rsi_oversold=rsi_data.get('oversold_threshold', 30),
                rsi_overbought=rsi_data.get('overbought_threshold', 70),
                macd_enabled=macd_data.get('enabled', True),
                macd_fast=macd_data.get('fast_period', 12),
                macd_slow=macd_data.get('slow_period', 26),
                macd_signal=macd_data.get('signal_period', 9),
                bollinger_enabled=bb_data.get('enabled', True),
                bb_period=bb_data.get('period', 20),
                bb_std_dev=bb_data.get('std_dev', 2.0),
                ma_enabled=ma_data.get('enabled', True),
                sma_periods=ma_data.get('sma_periods', [5, 10, 20, 50, 100, 200]),
                ema_periods=ma_data.get('ema_periods', [12, 26, 50]),
                price_above_ma200=ma_data.get('price_above_ma200', True),
                atr_enabled=atr_data.get('enabled', True),
                atr_period=atr_data.get('period', 14)
            )
            
            # Parse fundamental ratios
            fund_data = config_data.get('stage_2_filters', {}).get('fundamental_ratios', {})
            pe_data = fund_data.get('pe_ratio', {})
            pb_data = fund_data.get('pb_ratio', {})
            roe_data = fund_data.get('roe', {})
            de_data = fund_data.get('debt_to_equity', {})
            cr_data = fund_data.get('current_ratio', {})
            
            fundamental_ratios = FundamentalRatios(
                pe_ratio_enabled=pe_data.get('enabled', True),
                pe_min=pe_data.get('minimum', 0),
                pe_max=pe_data.get('maximum', 50),
                pb_ratio_enabled=pb_data.get('enabled', True),
                pb_min=pb_data.get('minimum', 0.5),
                pb_max=pb_data.get('maximum', 5.0),
                roe_enabled=roe_data.get('enabled', True),
                roe_min=roe_data.get('minimum', 12),
                debt_to_equity_enabled=de_data.get('enabled', True),
                debt_to_equity_max=de_data.get('maximum', 1.5),
                current_ratio_enabled=cr_data.get('enabled', True),
                current_ratio_min=cr_data.get('minimum', 1.2)
            )
            
            # Parse risk metrics
            risk_data = config_data.get('stage_2_filters', {}).get('risk_metrics', {})
            beta_data = risk_data.get('beta', {})
            vol_data = risk_data.get('volatility', {})
            sharpe_data = risk_data.get('sharpe_ratio', {})
            
            risk_metrics = RiskMetrics(
                beta_enabled=beta_data.get('enabled', True),
                beta_min=beta_data.get('minimum', 0.7),
                beta_max=beta_data.get('maximum', 1.6),
                volatility_enabled=vol_data.get('enabled', True),
                volatility_period=vol_data.get('period', 20),
                volatility_max_daily=vol_data.get('maximum_daily', 4.0),
                volatility_max_annual=vol_data.get('maximum_annual', 45.0),
                sharpe_ratio_enabled=sharpe_data.get('enabled', True),
                sharpe_ratio_min=sharpe_data.get('minimum', 0.5),
                risk_free_rate=sharpe_data.get('risk_free_rate', 6.5)
            )
            
            # Parse scoring weights
            scoring_data = config_data.get('stage_2_filters', {}).get('scoring_weights', {})
            scoring_weights = ScoringWeights(
                technical_score=scoring_data.get('technical_score', 0.30),
                fundamental_score=scoring_data.get('fundamental_score', 0.20),
                risk_score=scoring_data.get('risk_score', 0.20),
                momentum_score=scoring_data.get('momentum_score', 0.25),
                volume_score=scoring_data.get('volume_score', 0.05)
            )
            
            # Parse filtering thresholds
            threshold_data = config_data.get('stage_2_filters', {}).get('filtering_thresholds', {})
            filtering_thresholds = FilteringThresholds(
                minimum_total_score=threshold_data.get('minimum_total_score', 60),
                minimum_technical_score=threshold_data.get('minimum_technical_score', 50),
                minimum_fundamental_score=threshold_data.get('minimum_fundamental_score', 40),
                minimum_risk_score=threshold_data.get('minimum_risk_score', 50),
                require_all_categories=threshold_data.get('require_all_categories', False)
            )
            
            # Parse selection configuration
            selection_data = config_data.get('selection', {})
            selection = SelectionConfig(
                max_suggested_stocks=selection_data.get('max_suggested_stocks', 10),
                tie_breaker_priority=selection_data.get('tie_breaker_priority', ["momentum_score", "risk_score", "technical_score"]),
                sector_concentration_limit_pct=selection_data.get('sector_concentration_limit_pct', 40),
                min_large_mid_pct=selection_data.get('market_cap_mix', {}).get('min_large_mid_pct', 60),
                blacklist_symbols=selection_data.get('blacklist_symbols', []),
                whitelist_symbols=selection_data.get('whitelist_symbols', []),
                min_distance_from_resistance_pct=selection_data.get('min_distance_from_resistance_pct', 2.0),
                require_price_above_vwap=selection_data.get('require_price_above_vwap', True)
            )
            
            # Create the enhanced configuration
            config = EnhancedFilteringConfig(
                universe=universe,
                stage_1_filters=stage1_filters,
                technical_indicators=technical_indicators,
                fundamental_ratios=fundamental_ratios,
                risk_metrics=risk_metrics,
                scoring_weights=scoring_weights,
                filtering_thresholds=filtering_thresholds,
                selection=selection,
                liquidity_scoring=config_data.get('liquidity_scoring', {}),
                shares_estimation=config_data.get('shares_estimation', {}),
                database=config_data.get('database', {}),
                cache=config_data.get('cache', {}),
                api_processing=config_data.get('api_processing', {}),
                sector_keywords=config_data.get('sector_keywords', {}),
                logging=config_data.get('logging', {})
            )
            
            self._config = config
            logger.info("Enhanced filtering configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Error loading enhanced configuration: {e}")
            return EnhancedFilteringConfig()
    
    def get_config(self) -> EnhancedFilteringConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> EnhancedFilteringConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()


# Global configuration loader instance
_enhanced_config_loader = None


def get_enhanced_config_loader() -> EnhancedConfigLoader:
    """Get the global enhanced configuration loader instance."""
    global _enhanced_config_loader
    if _enhanced_config_loader is None:
        _enhanced_config_loader = EnhancedConfigLoader()
    return _enhanced_config_loader


def get_enhanced_filtering_config() -> EnhancedFilteringConfig:
    """Get the enhanced filtering configuration."""
    loader = get_enhanced_config_loader()
    return loader.get_config()
